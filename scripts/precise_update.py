import dataclasses
import json
import math
import os
import release_defaults
import time
import weakref
from typing import Optional

import torch
from pydantic import BaseModel
from torch.optim.adamw import adamw as _adamw
from transformers import Trainer, TrainerCallback

import precise_carry8 as c8
import phi_rtn_cosine
import precise_probe as probe

ENV = "SN56_PRECISE_UPDATE"
ENV_PHASES = "SN56_PRECISE_PHASES"
ENV_WRITEBACK = "SN56_PRECISE_WRITEBACK"
ENV_KERNEL = "SN56_PRECISE_KERNEL"
ENV_RESERVE_FRAC = "SN56_PRECISE_RESERVE_FRAC"
ENV_RESERVE_MIN_GB = "SN56_PRECISE_RESERVE_MIN_GB"
ENV_CHUNK = "SN56_PRECISE_CHUNK"
ENV_LR_PIN = "SN56_LR_PIN_ABS"
ENV_LR_PIN_MULT = "SN56_LR_PIN"
ENV_SEARCH = "SN56_LR_SEARCH_PRECISE"
ENV_LR_PIN_SKIP = "SN56_LR_PIN_SKIP"
ENV_SOUP_FP32 = "SN56_SOUP_FP32"
ENV_FP32_MASTER = "SN56_FP32_MASTER"
ENV_FP32_OOM_FIX = "SN56_FP32_OOM_FIX"
ENV_FP32_OOM_RELAUNCH = "SN56_FP32_OOM_RELAUNCH"
FP32_SEARCH_SNAPSHOT = "fp32_search_snapshot"

PHASES = ("probe", "train", "member", "devpass")
STOCK = "stock"
WRITEBACKS = ("carry", "rtn")
DEFAULT_RULE = "carry8"
DEFAULT_WRITEBACK = release_defaults.DEFAULTS[ENV_WRITEBACK]
LR_PIN_BY_TASK: dict = {}
DEFAULT_SEARCH_PRECISE = "1"
DEFAULT_LR_PIN = 1.0
_FP32_MASTER_DEFAULT = (release_defaults.DEFAULTS[ENV_FP32_MASTER].strip().lower()
                        not in ("", "0", "false", "no", "off"))
FP32_MASTER_PHASES = ("train", "member", "devpass")
_FP32_REFUSED: Optional[str] = None
FP32_A0_GAP_GB_PER_ROW: dict = {}
CHUNK = 1 << 26
RESERVE_FRAC = 0.25
RESERVE_MIN = 8 << 30
_FALSY = ("0", "false", "no", "off")


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in _FALSY


def enabled() -> bool:
    return _env_on(ENV)


def fp32_master_wanted() -> bool:
    if phi_rtn_cosine.armed():
        return False
    return _env_on(ENV_FP32_MASTER, "1" if _FP32_MASTER_DEFAULT else "0")


def fp32_master_on() -> bool:
    return fp32_master_wanted() and _FP32_REFUSED is None


def fp32_master_refuse(reason: str, **fields) -> None:
    global _FP32_REFUSED
    if _FP32_REFUSED is not None:
        return
    _FP32_REFUSED = reason


def oom_relaunch_plan(saved: Optional[dict], relaunch_oom: bool) -> tuple:
    saved = saved or {}
    if not relaunch_oom or not saved.get("micro"):
        return False, None
    if saved.get("fp32_master"):
        return True, max(1, int(saved["micro"]))
    return False, max(1, int(saved["micro"]) // 2)


def fp32_oom_fix_enabled() -> bool:
    return _env_on(ENV_FP32_OOM_FIX, release_defaults.DEFAULTS[ENV_FP32_OOM_FIX])


def fp32_oom_relaunch_active(saved: Optional[dict]) -> bool:
    saved = saved or {}
    return bool(fp32_oom_fix_enabled() and os.environ.get(ENV_FP32_OOM_RELAUNCH, "") == "1"
                and saved.get("fp32_master") and saved.get("micro"))


def _json_safe(obj):
    try:
        return json.loads(json.dumps(obj))
    except (TypeError, ValueError):
        return None


def search_snapshot(*, t_per_step, micro: int, accum: int, rows: int, v3: Optional[dict],
                    last_selection: dict, last_gns: dict, lr_rescale: float,
                    search_mode: str) -> dict:
    snap = {"t_per_step": (float(t_per_step) if t_per_step else None), "micro": int(micro),
            "accum": int(accum), "rows": int(rows), "v3": _json_safe(v3),
            "last_selection": _json_safe(dict(last_selection or {})) or {},
            "last_gns": _json_safe(dict(last_gns or {})) or {},
            "lr_rescale": float(lr_rescale), "search_mode": str(search_mode)}
    return snap


def relaunch_plan_target(saved: Optional[dict], snapshot: Optional[dict]) -> tuple:
    if snapshot and snapshot.get("micro"):
        return int(snapshot["micro"]), int(snapshot.get("accum") or 1), "search_snapshot"
    saved = saved or {}
    return int(saved["micro"]), int(saved.get("accum") or 1), "batch_policy"


def soup_fp32() -> bool:
    return _env_on(ENV_SOUP_FP32, "0")


def _positive(raw: str) -> Optional[float]:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def lr_pin(task_id: Optional[str] = None) -> Optional[float]:
    raw = os.environ.get(ENV_LR_PIN, "").strip()
    if raw and "=" not in raw:
        return _positive(raw)
    table = dict(LR_PIN_BY_TASK)
    for item in raw.split(","):
        if "=" in item:
            k, v = (s.strip() for s in item.split("=", 1))
            table[k] = v
    tid = str(task_id or "")
    for k, v in table.items():
        if k and (tid == k or tid.startswith(k)):
            return _positive(v)
    return None


def lr_pin_skips_search(task_id: Optional[str] = None) -> bool:
    return lr_pin(task_id) is not None and _env_on(ENV_LR_PIN_SKIP, "0")


def writeback_mode() -> str:
    if phi_rtn_cosine.armed():
        return phi_rtn_cosine.WRITEBACK
    v = os.environ.get(ENV_WRITEBACK, DEFAULT_WRITEBACK).strip().lower()
    v = "carry" if v == "carry8" else v
    return v if v in WRITEBACKS else DEFAULT_WRITEBACK


def lr_pin_mult() -> float:
    raw = os.environ.get(ENV_LR_PIN_MULT, "").strip()
    if not raw:
        return float(DEFAULT_LR_PIN)
    v = _positive(raw)
    return float(v) if v is not None else float(DEFAULT_LR_PIN)


def apply_lr_pin(lr: float) -> tuple:
    pin = lr_pin_mult()
    return float(lr) * pin, pin, float(lr)


def search_precise_enabled() -> bool:
    return _env_on(ENV_SEARCH, DEFAULT_SEARCH_PRECISE)


def kernel_wanted() -> bool:
    return _env_on(ENV_KERNEL)


def _chunk() -> int:
    try:
        return max(1 << 16, int(os.environ.get(ENV_CHUNK, str(CHUNK))))
    except ValueError:
        return CHUNK


@dataclasses.dataclass(frozen=True)
class Hyper:
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float


class UpdateRule:
    name = ""
    state_bytes_per_param = 0.0
    temp_bytes_per_elem = 0.0
    keeps_master = False

    def state_bytes(self, n: int, n_fp32: int = 0) -> int:
        return int(self.state_bytes_per_param * n)

    def temp_bytes(self, chunk: int) -> int:
        return int(self.temp_bytes_per_elem * chunk)

    def prepare(self, device) -> None:
        pass

    def fields(self) -> dict:
        return {"rule": self.name}

    def init_state(self, p: torch.Tensor, state: dict) -> None:
        raise NotImplementedError

    def step(self, params: list, grads: list, states: list, hp: Hyper, prb=None) -> None:
        raise NotImplementedError

    def resync(self, p: torch.Tensor, state: dict) -> None:
        pass

    def master(self, p: torch.Tensor, state: dict) -> Optional[torch.Tensor]:
        return None

    def master_at(self, p: torch.Tensor, state: dict, idx: torch.Tensor) -> Optional[torch.Tensor]:
        m = self.master(p, state)
        return None if m is None else m.reshape(-1)[idx]

    def moments(self, p: torch.Tensor, state: dict, idx: torch.Tensor) -> Optional[tuple]:
        return None


class MasterAdamW(UpdateRule):
    name = "master"
    state_bytes_per_param = 12.0
    temp_bytes_per_elem = 4.0
    keeps_master = True

    def init_state(self, p, state):
        state["master"] = p.detach().to(torch.float32).clone()
        state["exp_avg"] = torch.zeros_like(state["master"])
        state["exp_avg_sq"] = torch.zeros_like(state["master"])
        state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)

    def step(self, params, grads, states, hp, prb=None):
        masters = [s["master"] for s in states]
        g32 = [g if g.dtype == torch.float32 else g.to(torch.float32) for g in grads]
        _adamw(masters, g32, [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states],
               [], [s["step"] for s in states], fused=True, amsgrad=False,
               beta1=hp.beta1, beta2=hp.beta2, lr=hp.lr, weight_decay=hp.weight_decay,
               eps=hp.eps, maximize=False)
        torch._foreach_copy_(params, masters)

    def resync(self, p, state):
        state["master"].copy_(p)

    def master(self, p, state):
        return state.get("master")


_SELFTEST: dict = {}


def selftest_result(device) -> dict:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    key = str(device)
    if key not in _SELFTEST:
        t0 = time.perf_counter()
        res = c8.selftest(device)
        res["compile_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        _SELFTEST[key] = res
    return _SELFTEST[key]


def kernel_refusal(mode: str, device) -> Optional[str]:
    if mode != Carry8.name or device is None or torch.device(device).type != "cuda":
        return None
    if not kernel_wanted():
        return None
    if not c8.HAS_TRITON:
        return "kernel:no_triton"
    res = selftest_result(device)
    if not res.get("ok"):
        return "kernel:selftest_failed"
    return None


class Carry8(UpdateRule):
    name = "carry8"
    keeps_master = True

    def __init__(self, writeback: str = DEFAULT_WRITEBACK, fp32_ids=None, seed: int = 0,
                 kernel: bool = True):
        if writeback not in WRITEBACKS:
            raise ValueError(f"carry8: writeback {writeback!r} (known: {WRITEBACKS})")
        self.writeback = writeback
        self.carry = writeback == "carry"
        self.keeps_master = self.carry
        self.fp32_ids = set(fp32_ids or ())
        self.seed = int(seed)
        self.kernel = bool(kernel) and c8.HAS_TRITON
        self.variant = "eager"
        self.selftest: Optional[dict] = None
        self.compile_ms: Optional[float] = None
        self.n_fp32_params = 0
        self._tid = 0

    def _fp32(self, p) -> bool:
        return id(p) in self.fp32_ids or p.numel() < c8.MIN_8BIT

    def state_bytes(self, n, n_fp32=0):
        n8 = max(0, n - n_fp32)
        return (c8.state_bytes(n8, fp32_moments=False, carry=self.carry)
                + c8.state_bytes(n_fp32, fp32_moments=True, carry=self.carry))

    def temp_bytes(self, chunk):
        return 48 * c8.EAGER_CHUNK

    def prepare(self, device):
        device = torch.device(device)
        if not self.kernel or device.type != "cuda":
            self.variant = "eager"
            return
        self.selftest = selftest_result(device)
        self.compile_ms = self.selftest.get("compile_ms")
        self.variant = "triton" if self.selftest.get("ok") else "eager"

    def fields(self):
        return {"rule": self.name, "rule_variant": self.variant, "writeback": self.writeback,
                "fp32_moment_params": self.n_fp32_params, "triton_compile_ms": self.compile_ms,
                "kernel_selftest_ok": None if self.selftest is None else bool(self.selftest.get("ok"))}

    def init_state(self, p, state):
        fp32 = self._fp32(p)
        if fp32:
            self.n_fp32_params += int(p.numel())
        c8.init_state(p, state, fp32_moments=fp32, carry=self.carry)
        state["fp32"] = fp32
        state["tid"] = self._tid
        self._tid += 1

    def _run(self, p, g, state, hp):
        state["step"] = int(state["step"]) + 1
        kw = dict(lr=hp.lr, b1=hp.beta1, b2=hp.beta2, eps=hp.eps, wd=hp.weight_decay,
                  step=state["step"], seed=c8.seed32(self.seed, state["step"], state["tid"]),
                  mode=self.writeback, fp32_moments=state["fp32"])
        if p.is_contiguous():
            if self.variant == "triton" and p.is_cuda:
                c8.step_triton(p, g, state, **kw)
            else:
                c8.step_reference(p, g, state, **kw)
            return
        pc = p.contiguous()
        c8.step_reference(pc, g, state, **kw)
        p.copy_(pc)

    def step(self, params, grads, states, hp, prb=None):
        for p, g, st in zip(params, grads, states):
            self._run(p, g, st, hp)

    def resync(self, p, state):
        if self.carry:
            state["r8"].zero_()

    def master(self, p, state):
        if not self.carry:
            return None
        return c8.master_at(p, state, None, True).view(p.shape)

    def master_at(self, p, state, idx):
        if not self.carry:
            return None
        return c8.master_at(p, state, idx, True)

    def moments(self, p, state, idx):
        return c8.dequant_at(state, idx, state["fp32"], p.device)


class Master8(Carry8):
    name = "master8"
    keeps_master = True

    def __init__(self, writeback: str = "carry", fp32_ids=None, seed: int = 0, kernel: bool = True):
        super().__init__("rtn", fp32_ids, seed, False)
        self.writeback = "master"
        self.carry = False
        self.keeps_master = True

    def state_bytes(self, n, n_fp32=0):
        n8 = max(0, n - n_fp32)
        return (c8.state_bytes(n8, fp32_moments=False, carry=False, master=True)
                + c8.state_bytes(n_fp32, fp32_moments=True, carry=False, master=True))

    def init_state(self, p, state):
        fp32 = self._fp32(p)
        if fp32:
            self.n_fp32_params += int(p.numel())
        c8.init_state(p, state, fp32_moments=fp32, carry=False, master=True)
        state["fp32"] = fp32
        state["tid"] = self._tid
        self._tid += 1

    def resync(self, p, state):
        state["master"].copy_(p.detach().reshape(-1))

    def master(self, p, state):
        return state["master"].view(p.shape)

    def master_at(self, p, state, idx):
        return state["master"][idx]


RULES = {MasterAdamW.name: MasterAdamW, Carry8.name: Carry8, Master8.name: Master8}


def make_rule(name: str, **opts) -> UpdateRule:
    if name not in RULES:
        raise ValueError(f"precise_update: unknown rule {name!r} (known: {sorted(RULES)})")
    cls = RULES[name]
    return cls(**opts) if cls is not MasterAdamW else cls()


_LIVE: "weakref.WeakSet" = weakref.WeakSet()


class PreciseAdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 rule: str = DEFAULT_RULE, names: Optional[dict] = None, chunk: Optional[int] = None,
                 phase: str = "train", rule_opts: Optional[dict] = None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rule = make_rule(rule, **(rule_opts or {}))
        self.chunk = int(chunk or _chunk())
        self.names = dict(names or {})
        self.phase = phase
        self.dirty = True
        self.n_steps = 0
        self.n_resync = 0
        first = None
        for group in self.param_groups:
            for p in group["params"]:
                first = p if first is None else first
                self.rule.init_state(p, self.state[p])
        self._chunks = self._plan_chunks()
        if first is not None:
            self.rule.prepare(first.device)
        _LIVE.add(self)

    def load_state_dict(self, state_dict) -> None:
        groups, saved = self.param_groups, state_dict["param_groups"]
        if len(groups) != len(saved) or any(len(g["params"]) != len(s["params"])
                                            for g, s in zip(groups, saved)):
            raise ValueError("loaded state dict has a different parameter layout")
        id_map = dict(zip((i for s in saved for i in s["params"]),
                          (p for g in groups for p in g["params"])))
        state: dict = {}
        for k, v in state_dict["state"].items():
            if k in id_map:
                p = id_map[k]
                state[p] = {kk: (vv.to(device=p.device) if torch.is_tensor(vv) else vv)
                            for kk, vv in v.items()}
            else:
                state[k] = v
        new_groups = []
        for g, s in zip(groups, saved):
            ng = dict(s)
            ng["params"] = g["params"]
            new_groups.append(ng)
        self.__setstate__({"state": state, "param_groups": new_groups})

    def _plan_chunks(self) -> list:
        out = []
        for gi, group in enumerate(self.param_groups):
            cur, n = [], 0
            for p in group["params"]:
                if cur and n + p.numel() > self.chunk:
                    out.append((gi, cur))
                    cur, n = [], 0
                cur.append(p)
                n += p.numel()
            if cur:
                out.append((gi, cur))
        return out

    @torch.no_grad()
    def resync(self) -> int:
        n = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st:
                    self.rule.resync(p, st)
                    n += 1
        self.dirty = False
        self.n_resync += 1
        return n

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.dirty:
            n = self.resync()
        for gi, chunk in self._chunks:
            group = self.param_groups[gi]
            live = [p for p in chunk if p.grad is not None]
            if not live:
                continue
            b1, b2 = group["betas"]
            hp = Hyper(lr=float(group["lr"]), beta1=float(b1), beta2=float(b2),
                       eps=float(group["eps"]), weight_decay=float(group["weight_decay"]))
            self.rule.step(live, [p.grad for p in live], [self.state[p] for p in live], hp,
                           None)
        self.n_steps += 1
        return loss

    def release_state(self) -> int:
        n = sum(int(v.numel()) * int(v.element_size()) for st in self.state.values()
                for v in st.values() if torch.is_tensor(v))
        self.state.clear()
        self._chunks = []
        self.__dict__.pop("_probe", None)
        _LIVE.discard(self)
        return n

    def master_of(self, p: torch.Tensor) -> Optional[torch.Tensor]:
        if self.dirty or not self.rule.keeps_master:
            return None
        st = self.state.get(p)
        return self.rule.master(p, st) if st else None

    def master_at(self, p: torch.Tensor, idx: torch.Tensor) -> Optional[torch.Tensor]:
        if self.dirty or not self.rule.keeps_master:
            return None
        st = self.state.get(p)
        return self.rule.master_at(p, st, idx) if st else None

    def moments_at(self, p: torch.Tensor, idx: torch.Tensor) -> Optional[tuple]:
        st = self.state.get(p)
        return self.rule.moments(p, st, idx) if st else None

    @torch.no_grad()
    def cast_error(self) -> dict:
        if self.dirty or not self.rule.keeps_master:
            return {}
        acc: dict = {}
        for group in self.param_groups:
            for p in group["params"]:
                m = self.master_of(p)
                if m is None:
                    continue
                d = m - p.to(torch.float32)
                cls = probe.param_class(self.names.get(id(p), ""))
                for key in (cls, "all"):
                    a = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
                    a[0] += float(d.square().sum())
                    a[1] = max(a[1], float(d.abs().max()))
                    a[2] += float(m.square().sum())
                    a[3] += int(d.numel())
                del d
        out = {}
        for key, (ss, mx, ws, n) in acc.items():
            rms = math.sqrt(ss / max(1, n))
            wrms = math.sqrt(ws / max(1, n))
            out[key] = {"rms": rms, "max": mx, "w_rms": wrms,
                        "rms_rel": rms / wrms if wrms > 0 else None, "n": n}
        return out


def weights_changed() -> int:
    n = 0
    for opt in list(_LIVE):
        opt.dirty = True
        n += 1
    return n


def unwrap(optimizer):
    inner = optimizer
    while inner is not None and not isinstance(inner, PreciseAdamW) and hasattr(inner, "optimizer"):
        inner = inner.optimizer
    return inner if isinstance(inner, PreciseAdamW) else None


def high_precision(model) -> dict:
    live = [o for o in list(_LIVE) if not o.dirty and o.rule.keeps_master and o.state]
    if not live:
        return {}
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for o in live:
            m = o.master_of(p)
            if m is not None:
                out[n] = m
                break
    return out


def residual_stats(model) -> dict:
    for o in list(_LIVE):
        if o.dirty or not o.state:
            continue
        first = next((p for p in model.parameters() if p.requires_grad), None)
        if first is not None and first in o.state:
            return o.cast_error()
    return {}


class PhasePolicy(BaseModel):
    probe: str = DEFAULT_RULE
    train: str = DEFAULT_RULE
    member: str = DEFAULT_RULE
    devpass: str = STOCK
    writeback: str = DEFAULT_WRITEBACK

    def mode_for(self, phase: str) -> str:
        if phase not in PHASES:
            raise ValueError(f"precise_update: unknown phase {phase!r} (known: {PHASES})")
        return getattr(self, phase)

    @classmethod
    def from_env(cls) -> "PhasePolicy":
        kw = {"writeback": writeback_mode()}
        if fp32_master_on():
            kw.update({ph: MasterAdamW.name for ph in FP32_MASTER_PHASES})
        for item in os.environ.get(ENV_PHASES, "").split(","):
            if "=" not in item:
                continue
            k, v = (s.strip() for s in item.split("=", 1))
            if k in PHASES and (v == STOCK or v in RULES):
                kw[k] = v
        return cls(**kw)


class Plan(BaseModel):
    on: bool
    mode: str
    reason: str
    phase: str
    n_params: int = 0
    n_trainable: int = 0
    n_fp32_moments: int = 0
    writeback: str = DEFAULT_WRITEBACK
    trainable_bytes: int = 0
    need_bytes: Optional[int] = None
    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    reserve_bytes: Optional[int] = None
    world_size: int = 1
    optim: str = ""
    chunk: int = 0


class Footprint(BaseModel):
    name: str
    state_bytes: int
    step_temp_bytes: int


def _world_size(training_args) -> int:
    w = int(getattr(training_args, "world_size", 1) or 1)
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            w = max(w, int(torch.distributed.get_world_size()))
    except Exception:
        pass
    return w


def _optim_name(training_args) -> str:
    o = getattr(training_args, "optim", "") or ""
    return str(getattr(o, "value", o))


def device_memory(device) -> tuple:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    free, total = torch.cuda.mem_get_info(device)
    idle = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    return int(free + idle), int(total)


def reserve_bytes(total_bytes: int) -> int:
    try:
        frac = float(os.environ.get(ENV_RESERVE_FRAC, str(RESERVE_FRAC)))
    except ValueError:
        frac = RESERVE_FRAC
    try:
        mn = int(float(os.environ.get(ENV_RESERVE_MIN_GB, str(RESERVE_MIN / 2 ** 30))) * 2 ** 30)
    except ValueError:
        mn = RESERVE_MIN
    return max(mn, int(frac * total_bytes))


def fp32_moment_ids(model) -> set:
    ids = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            ids.update(id(p) for p in module.parameters(recurse=False))
    return ids


def n_fp32_moments(model) -> int:
    ids = fp32_moment_ids(model)
    return sum(int(p.numel()) for p in model.parameters()
               if p.requires_grad and (id(p) in ids or p.numel() < c8.MIN_8BIT))


def footprint(mode: str, n_trainable: int, chunk: Optional[int] = None, n_fp32: int = 0,
              writeback: Optional[str] = None) -> Footprint:
    rule = make_rule(mode, **({"writeback": writeback or writeback_mode()} if mode != "master" else {}))
    c = min(int(chunk or _chunk()), max(1, n_trainable))
    return Footprint(name=mode, state_bytes=int(rule.state_bytes(n_trainable, n_fp32)),
                     step_temp_bytes=int(rule.temp_bytes(c)))


def plan(training_args, model, *, phase: str, policy: Optional[PhasePolicy] = None,
         free_bytes: Optional[int] = None, total_bytes: Optional[int] = None,
         device=None) -> Plan:
    policy = policy or PhasePolicy.from_env()
    mode = policy.mode_for(phase)
    params = [p for p in model.parameters()]
    trainable = [p for p in params if p.requires_grad]
    n_all = sum(int(p.numel()) for p in params)
    n_tr = sum(int(p.numel()) for p in trainable)
    pl = Plan(on=False, mode=mode, reason="", phase=phase, n_params=n_all, n_trainable=n_tr,
              n_fp32_moments=n_fp32_moments(model), writeback=policy.writeback,
              trainable_bytes=sum(int(p.numel()) * int(p.element_size()) for p in trainable),
              world_size=_world_size(training_args), optim=_optim_name(training_args),
              chunk=_chunk())
    if not enabled():
        pl.reason = "env_off"
        return pl
    if mode == STOCK:
        pl.reason = "policy:stock"
        return pl
    if "adamw" not in pl.optim.lower():
        pl.reason = f"optim:{pl.optim}"
        return pl
    if bool(getattr(training_args, "use_lora", False)):
        pl.reason = "lora"
        return pl
    if getattr(training_args, "deepspeed", None) is not None or len(getattr(training_args, "fsdp", "") or "") > 0:
        pl.reason = "sharded"
        return pl
    if pl.world_size > 1:
        pl.reason = f"world_size:{pl.world_size}"
        return pl
    if not trainable:
        pl.reason = "no_trainable_params"
        return pl
    dtypes = {str(p.dtype) for p in trainable}
    if dtypes != {"torch.bfloat16"}:
        pl.reason = "dtype:" + ",".join(sorted(dtypes))
        return pl
    _dev = device if device is not None else trainable[0].device
    _why = kernel_refusal(mode, _dev)
    if _why is not None:
        pl.reason = _why
        return pl
    fp = footprint(mode, n_tr, pl.chunk, pl.n_fp32_moments, policy.writeback)
    pl.need_bytes = int(fp.state_bytes + fp.step_temp_bytes + 2 * n_tr)
    if free_bytes is None and total_bytes is None:
        free_bytes, total_bytes = device_memory(_dev)
    pl.free_bytes, pl.total_bytes = free_bytes, total_bytes
    if free_bytes is None or total_bytes is None:
        pl.on, pl.reason = True, "no_device_memory_info"
        return pl
    pl.reserve_bytes = reserve_bytes(int(total_bytes))
    if int(free_bytes) - pl.need_bytes < pl.reserve_bytes:
        pl.reason = "memory"
        return pl
    pl.on, pl.reason = True, "fits"
    return pl


def _names(model) -> dict:
    return {id(p): n for n, p in model.named_parameters()}


def kwargs_for(training_args, model, mode: str, phase: str, lr: Optional[float] = None,
               writeback: Optional[str] = None) -> dict:
    opts = {}
    if mode != "master":
        opts = {"writeback": writeback or writeback_mode(), "fp32_ids": fp32_moment_ids(model),
                "seed": int(getattr(training_args, "seed", 0) or 0), "kernel": kernel_wanted()}
    return {"lr": float(training_args.learning_rate if lr is None else lr),
            "betas": (float(getattr(training_args, "adam_beta1", 0.9)),
                      float(getattr(training_args, "adam_beta2", 0.999))),
            "eps": float(getattr(training_args, "adam_epsilon", 1e-8)),
            "rule": mode, "names": _names(model), "phase": phase, "rule_opts": opts}


def optimizer_for(training_args, model, *, phase: str, policy: Optional[PhasePolicy] = None,
                  free_bytes: Optional[int] = None, total_bytes: Optional[int] = None) -> tuple:
    pl = plan(training_args, model, phase=phase, policy=policy, free_bytes=free_bytes,
              total_bytes=total_bytes)
    if policy is None and phase in FP32_MASTER_PHASES and fp32_master_on():
        _xtra = fp32_master_extra_bytes(pl.n_trainable, pl.n_fp32_moments)
        _f = dict(phase=phase, extra_gb=round(_xtra / 1e9, 3), need_bytes=pl.need_bytes,
                  free_bytes=pl.free_bytes, reserve_bytes=pl.reserve_bytes)
        if pl.on and pl.mode == MasterAdamW.name:
            pass
        else:
            fp32_master_refuse(pl.reason, **_f)
            pl = plan(training_args, model, phase=phase, policy=None, free_bytes=free_bytes,
                      total_bytes=total_bytes)
    if not pl.on:
        cls, kw = Trainer.get_optimizer_cls_and_kwargs(training_args, model)
        return cls, kw, pl
    return PreciseAdamW, kwargs_for(training_args, model, pl.mode, phase, writeback=pl.writeback), pl


class SearchOptimizer(BaseModel):
    on: bool
    reason: str
    writeback: str = ""
    mode: str = ""
    plan_reason: str = ""


def search_optimizer(training_args, model, *, policy: Optional[PhasePolicy] = None,
                     free_bytes: Optional[int] = None, total_bytes: Optional[int] = None) -> tuple:
    if not search_precise_enabled():
        info = SearchOptimizer(on=False, reason="env_off")
        return None, None, info
    fp32 = policy is None and fp32_master_on()
    policy = policy or PhasePolicy.from_env()
    pl = plan(training_args, model, phase=("train" if fp32 else "probe"), policy=policy,
              free_bytes=free_bytes, total_bytes=total_bytes)
    if fp32:
        _f = dict(phase="search", extra_gb=round(fp32_master_extra_bytes(pl.n_trainable, pl.n_fp32_moments) / 1e9, 3),
                  need_bytes=pl.need_bytes, free_bytes=pl.free_bytes, reserve_bytes=pl.reserve_bytes)
        if pl.on and pl.mode == MasterAdamW.name:
            pass
        else:
            fp32_master_refuse(pl.reason, **_f)
            policy = PhasePolicy.from_env()
            pl = plan(training_args, model, phase="probe", policy=policy, free_bytes=free_bytes,
                      total_bytes=total_bytes)
    if not pl.on:
        info = SearchOptimizer(on=False, reason="plan_refused", plan_reason=pl.reason,
                               writeback=pl.writeback, mode=pl.mode)
        return None, None, info
    kw = kwargs_for(training_args, model, pl.mode, "search", writeback=pl.writeback)
    kw.pop("lr", None)
    kw["weight_decay"] = 0.0
    info = SearchOptimizer(on=True, reason="fits", writeback=pl.writeback, mode=pl.mode,
                           plan_reason=pl.reason)
    return PreciseAdamW, kw, info


def release(optimizer) -> int:
    opt = unwrap(optimizer)
    return opt.release_state() if opt is not None else 0


def build(training_args, model, *, phase: str, lr: float, weight_decay: float = 0.0,
          mode: Optional[str] = None, policy: Optional[PhasePolicy] = None) -> PreciseAdamW:
    policy = policy or PhasePolicy.from_env()
    kw = kwargs_for(training_args, model, mode or policy.mode_for(phase), phase, lr=lr,
                    writeback=policy.writeback)
    return PreciseAdamW([p for p in model.parameters() if p.requires_grad],
                        weight_decay=float(weight_decay), **kw)


def fp32_master_extra_bytes(n_trainable: int, n_fp32: int = 0, base_state_bytes: Optional[float] = None) -> int:
    m = footprint(MasterAdamW.name, n_trainable)
    if base_state_bytes is None:
        base_state_bytes = footprint(DEFAULT_RULE, n_trainable, n_fp32=n_fp32,
                                     writeback=writeback_mode()).state_bytes
    return int(m.state_bytes + m.step_temp_bytes - float(base_state_bytes))


class Fp32Fit(BaseModel):
    enabled: bool
    fits: Optional[bool]
    reason: str
    extra_gb: Optional[float] = None
    base_state_gb: Optional[float] = None
    gap_gb_per_row: Optional[float] = None
    budget_gb: Optional[float] = None
    micro_in: int
    accum_in: int
    micro: int
    accum: int
    rows: int
    peak_gb_base: Optional[float] = None
    projected_peak_gb: Optional[float] = None
    rungs: list = []
    gate: str = "v1"
    worst_row_tokens: Optional[int] = None
    projected_peak_gb_v1: Optional[float] = None


def fp32_a0_gap_per_row(task_id: Optional[str] = None) -> float:
    tid = str(task_id or "")
    for k, v in FP32_A0_GAP_GB_PER_ROW.items():
        if k and (tid == k or tid.startswith(k)):
            return max(0.0, float(v))
    return 0.0


def rung_peak_v2(tries: list, micro: int, worst_row_tokens: Optional[int]) -> dict:
    ok = [t for t in tries if t.get("peak_gb") is not None and not t.get("oom")
          and not t.get("error") and int(t.get("accum", 1) or 1) == 1]
    by = {int(t["micro"]): t for t in ok}
    t = by[int(micro)]
    peak = float(t["peak_gb"]) * 1e9
    out = {"peak": peak, "worst_tokens": None, "measured_tokens": None, "slope_mb_per_token": None,
           "frag": 0.0}
    res = t.get("reserved_gb")
    if res is not None:
        out["frag"] = max(0.0, float(res) * 1e9 - peak)
    tok = max(int(t.get("computed_tokens") or 0), int(t.get("worst_tokens") or 0))
    if not worst_row_tokens or tok <= 0 or t.get("chunks"):
        return out
    worst = int(micro) * int(worst_row_tokens)
    out["measured_tokens"], out["worst_tokens"] = tok, worst
    if worst <= tok:
        return out
    lower = [k for k in by if k < int(micro)
             and max(int(by[k].get("computed_tokens") or 0), int(by[k].get("worst_tokens") or 0)) > 0]
    if lower:
        lo = by[max(lower)]
        tok_lo = max(int(lo.get("computed_tokens") or 0), int(lo.get("worst_tokens") or 0))
        if tok > tok_lo:
            slope = max(0.0, (peak - float(lo["peak_gb"]) * 1e9) / float(tok - tok_lo))
            out["slope_mb_per_token"] = round(slope / 1e6, 5)
            out["peak"] = peak + slope * (worst - tok)
            return out
    out["peak"] = peak * worst / float(tok)
    return out


def fp32_master_fit_batch(probe_res, micro: int, accum: int, world_size: int,
                          n_trainable: int, n_fp32: int = 0, sharded: bool = False,
                          task_id: Optional[str] = None,
                          worst_row_tokens: Optional[int] = None) -> Fp32Fit:
    micro, accum = int(micro), int(accum)
    rows = micro * accum
    out = Fp32Fit(enabled=True, fits=None, reason="", micro_in=micro, accum_in=accum,
                  micro=micro, accum=accum, rows=rows)
    if not fp32_master_on():
        out.enabled, out.fits, out.reason = False, False, f"refused:{_FP32_REFUSED}"
        return out
    if world_size > 1 or sharded:
        out.enabled, out.fits = False, False
        out.reason = "sharded" if sharded else f"world_size:{world_size}"
        fp32_master_refuse(out.reason, micro=micro, accum=accum)
        return out
    tries = [t for t in (getattr(probe_res, "tries", None) or [])
             if t.get("peak_gb") is not None and not t.get("oom") and not t.get("error")
             and int(t.get("accum", 1) or 1) == 1]
    total_gb = getattr(probe_res, "total_gb", None) if probe_res is not None else None
    if not tries or not total_gb:
        out.reason = "no_probe"
        return out
    base_state = float(getattr(probe_res, "state_gb", 0.0) or 0.0) * 1e9
    extra = fp32_master_extra_bytes(n_trainable, n_fp32, base_state_bytes=base_state)
    budget = float(total_gb) * 1e9 * float(probe_res.headroom) - float(probe_res.reserve_gb) * 1e9
    gap_row = fp32_a0_gap_per_row(task_id) * 1e9
    out.gap_gb_per_row = round(gap_row / 1e9, 5)
    out.extra_gb, out.base_state_gb = round(extra / 1e9, 3), round(base_state / 1e9, 3)
    out.budget_gb = round(budget / 1e9, 3)
    peak = {int(t["micro"]): float(t["peak_gb"]) * 1e9 for t in tries}
    v2 = fp32_oom_fix_enabled()
    out.gate = "v2" if v2 else "v1"
    out.worst_row_tokens = int(worst_row_tokens) if (v2 and worst_row_tokens) else None
    v1_pick, m = None, micro
    while m >= 1 and v1_pick is None:
        if rows % m == 0 and m in peak and peak[m] + extra + gap_row * m <= budget:
            v1_pick = m
        m //= 2
    m = micro
    while m >= 1:
        if rows % m == 0 and m in peak:
            proj_v1 = peak[m] + extra + gap_row * m
            rung = {"micro": m, "peak_gb_base": round(peak[m] / 1e9, 3)}
            if v2:
                pv = rung_peak_v2(tries, m, worst_row_tokens)
                proj = pv["peak"] + pv["frag"] + extra + gap_row * m
                rung.update(peak_gb_worst=round(pv["peak"] / 1e9, 3), frag_gb=round(pv["frag"] / 1e9, 3),
                            worst_tokens=pv["worst_tokens"], measured_tokens=pv["measured_tokens"],
                            slope_mb_per_token=pv["slope_mb_per_token"],
                            projected_gb_v1=round(proj_v1 / 1e9, 3))
            else:
                proj = proj_v1
            ok = proj <= budget
            rung.update(projected_gb=round(proj / 1e9, 3), fits=ok)
            out.rungs.append(rung)
            if ok:
                out.fits, out.reason = True, ("fits" if m == micro else "split")
                out.micro, out.accum = m, rows // m
                out.peak_gb_base, out.projected_peak_gb = round(peak[m] / 1e9, 3), round(proj / 1e9, 3)
                out.projected_peak_gb_v1 = round(proj_v1 / 1e9, 3)
                return out
        m //= 2
    out.enabled, out.fits, out.reason = False, False, "memory"
    fp32_master_refuse("memory", extra_gb=out.extra_gb, budget_gb=out.budget_gb)
    return out


class Fp32MasterClockCallback(TrainerCallback):

    def __init__(self, phase: str):
        self.phase = phase
        self.t_last: Optional[float] = None
        self.dts: list = []

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if self.t_last is not None:
            self.dts.append(now - self.t_last)
        self.t_last = now

    def on_train_end(self, args, state, control, **kwargs):
        pass


_CLASSES: dict = {}


class PreciseUpdateMixin:
    precise_phase = "train"
    precise_plan: Optional[Plan] = None

    def create_optimizer(self, model=None):
        if self.optimizer is None and self.optimizer_cls_and_kwargs is None:
            opt_model = self.model if model is None else model
            cls, kw, pl = optimizer_for(self.args, opt_model, phase=self.precise_phase)
            self.precise_plan = pl
            if pl.on:
                self.optimizer_cls_and_kwargs = (cls, kw)
                if pl.mode == MasterAdamW.name and fp32_master_on():
                    self.add_callback(Fp32MasterClockCallback(self.precise_phase))
            opt = super().create_optimizer(model)
            return opt
        return super().create_optimizer(model)


def with_precise_update(cls):
    if not (isinstance(cls, type) and issubclass(cls, Trainer)) or issubclass(cls, PreciseUpdateMixin):
        return cls
    if cls not in _CLASSES:
        _CLASSES[cls] = type(f"PreciseUpdate{cls.__name__}", (PreciseUpdateMixin, cls), {})
    return _CLASSES[cls]
