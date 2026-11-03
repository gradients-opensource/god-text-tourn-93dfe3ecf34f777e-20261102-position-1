import contextlib
import datetime
import gc
import math
import os
import time
from typing import Callable, Optional

import torch
from pydantic import BaseModel
from transformers import TrainerCallback


ENV_SHAPE = "SN56_MEMBER_SHAPE_GUARD"
ENV_TIMEOUT = "SN56_MEMBER_COLLECTIVE_TIMEOUT"
MIN_COLLECTIVE_TIMEOUT_SECS = 60.0
TIMEOUT_STEP_MULT = 20.0
TIMEOUT_POINT_MULT = 4.0
CAPACITY_MARGIN_FLOOR = 1024 ** 3
CAPACITY_MARGIN_FRAC = 0.05
DEFAULT_HEADROOM = 0.88
ENV_GATE_DRIVER = "SN56_MEMBER_GATE_DRIVER"
ENV_SKIP_FAILED = "SN56_MEMBER_SKIP_FAILED"
ENV_MAX_CONSEC_FAIL = "SN56_MEMBER_MAX_CONSEC_FAIL"
DEFAULT_MAX_CONSEC_FAIL = 2


TOKEN_BUDGET: Optional[int] = None


def worst_tokens(micro: int, worst_len: int) -> int:
    if TOKEN_BUDGET is None:
        return int(micro) * int(worst_len)
    return min(int(micro) * int(worst_len), max(int(TOKEN_BUDGET), int(worst_len)))


def _on(env: str) -> bool:
    return os.environ.get(env, "1").strip().lower() not in ("0", "false", "no", "off")


def _dist() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1)


class MemoryProof(BaseModel):
    world_size: int = 1
    a0_micro: int = 1
    a0_accum: int = 1
    a0_eval_micro: int = 1
    a0_gc: bool = False
    max_length: Optional[int] = None
    pad_multiple: int = 1
    row_len_max: Optional[int] = None
    a0_max_rows: int = 0
    a0_max_len: int = 0
    a0_max_tokens: int = 0
    a0_alloc_begin_bytes: Optional[int] = None
    a0_peak_bytes: Optional[int] = None
    device_total_bytes: Optional[int] = None
    a0_state_bytes: Optional[int] = None
    grad_bytes: Optional[int] = None
    a0_increment_bytes: Optional[int] = None
    optimizer_resid_bytes: int = 0

    def increment(self) -> Optional[int]:
        if self.a0_increment_bytes is not None:
            return max(0, int(self.a0_increment_bytes))
        if self.a0_peak_bytes is None or self.a0_alloc_begin_bytes is None:
            return None
        return max(0, int(self.a0_peak_bytes) - int(self.a0_alloc_begin_bytes))

    def worst_len(self) -> Optional[int]:
        m = max(1, int(self.pad_multiple or 1))
        cands = []
        if self.row_len_max:
            cands.append(int(math.ceil(int(self.row_len_max) / m) * m))
        if self.a0_max_len:
            cands.append(int(self.a0_max_len))
        if not cands and self.max_length:
            cands.append(int(math.ceil(int(self.max_length) / m) * m))
        return max(cands) if cands else None

    def act_bytes_per_token(self) -> Optional[float]:
        if not (self.a0_peak_bytes and self.a0_alloc_begin_bytes is not None
                and self.a0_max_tokens > 0):
            return None
        return max(0.0, float(self.a0_peak_bytes - self.a0_alloc_begin_bytes)) / self.a0_max_tokens


class ShapeVerdict(BaseModel):
    ok: bool
    reason: str = ""
    before: dict = {}
    after: dict = {}
    reasons: list = []
    tokens_worst: Optional[int] = None
    a0_max_tokens: int = 0
    worst_len: Optional[int] = None
    projected_peak_bytes: Optional[int] = None
    cap_bytes: Optional[int] = None

    @property
    def clamped(self) -> bool:
        return bool(self.reasons)


def _project(proof: MemoryProof, tokens: int, headroom: float) -> tuple:
    apt = proof.act_bytes_per_token()
    if apt is None or not proof.device_total_bytes or proof.a0_alloc_begin_bytes is None:
        return False, None, None
    proj = int(proof.a0_alloc_begin_bytes + apt * tokens)
    proj = max(proj, int(proof.a0_peak_bytes or 0))
    cap = int(proof.device_total_bytes * headroom)
    return proj <= cap, proj, cap


def member_shape(*, micro: int, accum: int, gc: bool, eval_micro: int,
                 proof: Optional[MemoryProof], headroom: float = DEFAULT_HEADROOM) -> ShapeVerdict:
    micro, accum, eval_micro = max(1, int(micro)), max(1, int(accum)), max(1, int(eval_micro))
    before = {"micro": micro, "accum": accum, "rows": micro * accum, "gc": bool(gc),
              "eval_micro": eval_micro}
    if proof is None or not _on(ENV_SHAPE):
        return ShapeVerdict(ok=True, reason="no_proof" if proof is None else "disabled",
                            before=before, after=dict(before))
    rows = micro * accum
    reasons = []
    if micro > proof.a0_micro:
        micro = max(1, int(proof.a0_micro))
        reasons.append("micro_gt_a0")
    L = proof.worst_len()
    if L and proof.a0_max_tokens > 0 and micro * L > proof.a0_max_tokens:
        m2 = max(1, proof.a0_max_tokens // L)
        if m2 < micro:
            if TOKEN_BUDGET is None:
                micro = m2
                reasons.append("worst_len_tokens_gt_a0")
            else:
                pass
    if proof.a0_gc and not gc:
        gc = True
        reasons.append("gc_off_but_a0_on")
    if eval_micro > max(1, proof.a0_eval_micro):
        eval_micro = max(1, int(proof.a0_eval_micro))
        reasons.append("eval_micro_gt_a0")
    accum = max(1, int(math.ceil(rows / micro)))
    after = {"micro": micro, "accum": accum, "rows": micro * accum, "gc": bool(gc),
             "eval_micro": eval_micro}
    v = ShapeVerdict(ok=True, reason="proven", before=before, after=after, reasons=reasons,
                     a0_max_tokens=int(proof.a0_max_tokens), worst_len=L)
    if L:
        v.tokens_worst = int(worst_tokens(micro, L))
    if L and proof.a0_max_tokens > 0 and worst_tokens(micro, L) > proof.a0_max_tokens:
        fits, proj, cap = _project(proof, worst_tokens(micro, L), headroom)
        v.projected_peak_bytes, v.cap_bytes = proj, cap
        if not fits:
            v.ok, v.reason = False, "memory_unproven"
        else:
            v.reason = "projected"
    elif not proof.a0_max_tokens:
        v.reason = "no_a0_tokens"
    return v


def devfit_shape(*, eval_micro: int, proof: Optional[MemoryProof],
                 headroom: float = DEFAULT_HEADROOM) -> ShapeVerdict:
    v = member_shape(micro=eval_micro, accum=1, gc=True, eval_micro=eval_micro, proof=proof,
                     headroom=headroom)
    if v.ok and any(r in v.reasons for r in ("micro_gt_a0", "worst_len_tokens_gt_a0")):
        fits, proj, cap = _project(proof, worst_tokens(int(eval_micro), int(v.worst_len or 0)), headroom)
        v.projected_peak_bytes, v.cap_bytes = proj, cap
        v.ok, v.reason = (True, "projected") if fits else (False, "memory_unproven")
    return v


def apply_shape(args, v: ShapeVerdict) -> None:
    a = v.after
    args.per_device_train_batch_size = int(a["micro"])
    args.gradient_accumulation_steps = int(a["accum"])
    args.per_device_eval_batch_size = int(a["eval_micro"])
    if a["gc"] and not getattr(args, "gradient_checkpointing", False):
        args.gradient_checkpointing = True


PROVEN_PEAK_TOLERANCE = 512 * 1024 ** 2


def capacity_margin(device_total: Optional[int]) -> int:
    return max(CAPACITY_MARGIN_FLOOR, int(CAPACITY_MARGIN_FRAC * float(device_total or 0)))


def member_need_bytes(proof: Optional[MemoryProof], tokens: Optional[int]) -> tuple:
    inc = proof.increment() if proof is not None else None
    if inc is None:
        return None, "unmeasured"
    a0_tok = int(proof.a0_max_tokens or 0)
    t = int(tokens) if tokens else a0_tok
    if a0_tok <= 0:
        return inc, "a0_increment"
    if proof.a0_state_bytes is not None and proof.grad_bytes is not None:
        fixed = min(inc, max(0, int(proof.a0_state_bytes)) + max(0, int(proof.grad_bytes)))
        act = float(inc - fixed) / a0_tok
        return int(fixed + act * t), "split"
    return int(inc * max(1.0, float(t) / a0_tok)), "lump"


def capacity_gate(proof: Optional[MemoryProof], *, alloc_now: Optional[int],
                  avail_bytes: Optional[int], micro: int,
                  worst_len: Optional[int] = None, driver: Optional[bool] = None) -> tuple:
    drv = _on(ENV_GATE_DRIVER) if driver is None else bool(driver)
    need, model = member_need_bytes(
        proof, worst_tokens(int(micro), int(worst_len)) if (worst_len and micro) else None)
    if need is None or avail_bytes is None:
        return True, {"reason": "unmeasured", "need_model": model,
                      "alloc_now": alloc_now, "avail": avail_bytes}
    total = int(proof.device_total_bytes or 0)
    margin = capacity_margin(total)
    slack = int(avail_bytes) - margin - int(need)
    projected = (int(alloc_now) + int(need)) if alloc_now is not None else None
    within_a0 = (projected is not None and proof.a0_peak_bytes is not None
                 and projected <= int(proof.a0_peak_bytes) + PROVEN_PEAK_TOLERANCE)
    driver_short = bool(drv and within_a0 and int(avail_bytes) < int(need))
    within_a0 = within_a0 and not driver_short
    ok = within_a0 or slack >= 0
    info = {"reason": ("within_a0_peak" if within_a0 else "ok" if ok else
                       "over_capacity_driver" if driver_short else "over_capacity"),
            "driver_check": bool(drv), "driver_short": driver_short,
            "need_model": model,
            "micro": int(micro), "tokens": worst_tokens(int(micro), int(worst_len)) if worst_len else None,
            "worst_len": worst_len, "alloc_now": alloc_now, "avail": int(avail_bytes),
            "margin": int(margin), "need": int(need), "slack": int(slack),
            "projected_peak": projected,
            "proven_tolerance": PROVEN_PEAK_TOLERANCE,
            "device_total": total or None,
            "a0_alloc_begin": proof.a0_alloc_begin_bytes, "a0_peak": proof.a0_peak_bytes,
            "a0_increment": proof.increment(),
            "a0_state_bytes": proof.a0_state_bytes, "grad_bytes": proof.grad_bytes,
            "a0_max_tokens": proof.a0_max_tokens, "fit_micro": None}
    if not ok and worst_len and int(micro) > 1:
        for m in range(int(micro) - 1, 0, -1):
            n_m, _ = member_need_bytes(proof, worst_tokens(m, int(worst_len)))
            if n_m is None:
                continue
            if (int(avail_bytes) - margin - int(n_m) >= 0
                    or (alloc_now is not None and proof.a0_peak_bytes is not None
                        and int(alloc_now) + int(n_m)
                        <= int(proof.a0_peak_bytes) + PROVEN_PEAK_TOLERANCE
                        and (not drv or int(avail_bytes) >= int(n_m)))):
                info["fit_micro"] = m
                break
    return ok, info


def device_capacity() -> tuple:
    vals = [-1.0, 1.0]
    dev = None
    if torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
        alloc = float(torch.cuda.memory_allocated(dev))
        reserved = float(torch.cuda.memory_reserved(dev))
        free, _total = torch.cuda.mem_get_info(dev)
        vals = [alloc, -(float(free) + max(0.0, reserved - alloc))]
    if _dist():
        t = torch.tensor(vals, device=dev if (dev is not None and torch.distributed.get_backend() == "nccl")
                         else "cpu", dtype=torch.float64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        vals = [float(x) for x in t.tolist()]
    if vals[0] < 0:
        return None, None
    return int(vals[0]), int(-vals[1])


def host_ram_gate(terms: Optional[dict]) -> tuple:
    if not terms or terms.get("room") is None or terms.get("need") is None:
        return True, {"host_reason": "unmeasured"}
    ok = int(terms["room"]) >= int(terms["need"])
    info = {f"host_{k}": v for k, v in terms.items() if not k.startswith("cgroup")
            and k not in ("ram_avail", "ram_source")}
    info.update({k: v for k, v in terms.items() if k.startswith("cgroup")
                 or k in ("ram_avail", "ram_source")})
    info["host_reason"] = "ok" if ok else "host_ram"
    return ok, info


def host_terms_agreed(host_fn: Optional[Callable[[], Optional[dict]]]) -> Optional[dict]:
    terms = None
    if host_fn is not None and (not _dist() or torch.distributed.get_rank() == 0):
        try:
            terms = host_fn()
        except Exception:
            terms = None
    if not _dist():
        return terms
    obj = [terms]
    torch.distributed.broadcast_object_list(obj, src=0)
    return obj[0]


class MemberMemoryRefused(RuntimeError):

    def __init__(self, info: dict) -> None:
        super().__init__(f"over_capacity: {info}")
        self.info = dict(info)

    @property
    def fit_micro(self) -> Optional[int]:
        m = self.info.get("fit_micro")
        return int(m) if m else None


class MemberCapacityCallback(TrainerCallback):

    def __init__(self, proof: Optional[MemoryProof], label: str = "",
                 host_fn: Optional[Callable[[], Optional[dict]]] = None) -> None:
        self.proof = proof
        self.label = label
        self.host_fn = host_fn

    def on_train_begin(self, args, state, control, **kwargs):
        host_ok, host_info = True, {}
        if self.host_fn is not None:
            host_ok, host_info = host_ram_gate(host_terms_agreed(self.host_fn))
        if self.proof is None or self.proof.a0_alloc_begin_bytes is None:
            if not host_ok:
                raise MemberMemoryRefused({"reason": "host_ram", "fit_micro": None,
                                               **host_info})
            return
        micro = int(getattr(args, "per_device_train_batch_size", 1) or 1)
        L = self.proof.worst_len()
        alloc, avail = device_capacity()
        ok, info = capacity_gate(self.proof, alloc_now=alloc, avail_bytes=avail,
                                 micro=micro, worst_len=L)
        if not ok:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            alloc, avail = device_capacity()
            ok, info = capacity_gate(self.proof, alloc_now=alloc, avail_bytes=avail,
                                     micro=micro, worst_len=L)
        if ok and not host_ok:
            ok = False
            info = dict(info, reason="host_ram", fit_micro=None)
        info.update(host_info)
        if not ok:
            raise MemberMemoryRefused(info)

def member_failure_kind(exc: Optional[BaseException]) -> Optional[str]:
    if exc is None:
        return None
    msg = str(exc)
    if any(s in msg for s in ("NCCL", "ProcessGroup", "gloo", "Watchdog caught collective")):
        return None
    if isinstance(exc, torch.OutOfMemoryError):
        return "oom"
    low = msg.lower()
    if "out of memory" in low or "cudnn_status_alloc_failed" in low:
        return "oom"
    accel = getattr(torch, "AcceleratorError", None)
    if accel is not None and isinstance(exc, accel):
        return "cuda"
    if ("CUDA error" in msg or "CUBLAS_STATUS" in msg or "CUDNN_STATUS" in msg
            or "cuda runtime error" in low):
        return "cuda"
    return None


def member_skip_vote(*, multi_rank: bool, launched: bool, pre_exc: Optional[BaseException],
                     train_exc: Optional[BaseException], exc: Optional[BaseException]) -> tuple:
    kind = member_failure_kind(train_exc if (multi_rank and not launched) else exc)
    vote = pre_exc is None and (kind is not None
                                or (multi_rank and train_exc is None and not launched))
    return kind, bool(vote)


class MemberFailures(BaseModel):
    enabled: bool = True
    cap: int = DEFAULT_MAX_CONSEC_FAIL
    consecutive: int = 0
    skipped: int = 0

    @classmethod
    def from_env(cls) -> "MemberFailures":
        try:
            cap = int(float(os.environ.get(ENV_MAX_CONSEC_FAIL, "") or DEFAULT_MAX_CONSEC_FAIL))
        except (TypeError, ValueError):
            cap = DEFAULT_MAX_CONSEC_FAIL
        return cls(enabled=_on(ENV_SKIP_FAILED), cap=max(1, cap))

    def on_success(self) -> None:
        self.consecutive = 0

    def on_failure(self, kind: Optional[str]) -> bool:
        self.consecutive += 1
        skip = bool(self.enabled and kind is not None and self.consecutive < self.cap)
        if skip:
            self.skipped += 1
        return skip


def boundary_release(sync: bool = True) -> float:
    t0 = time.perf_counter()
    gc.collect()
    if torch.cuda.is_available():
        try:
            if sync:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
    return round(time.perf_counter() - t0, 4)


class A0MemoryCallback(TrainerCallback):

    def __init__(self) -> None:
        self.max_rows = 0
        self.max_len = 0
        self.max_tokens = 0
        self.alloc_begin: Optional[int] = None
        self.alloc_step1: Optional[int] = None
        self.grad_bytes: Optional[int] = None
        self.peak: Optional[int] = None
        self.resid_bytes = 0
        self._handle = None

    def _hook(self, module, args, kwargs):
        if not module.training or not torch.is_grad_enabled():
            return None
        t = kwargs.get("input_ids")
        if t is None:
            t = kwargs.get("inputs_embeds")
        if t is None and args:
            t = args[0]
        if torch.is_tensor(t) and t.dim() >= 2:
            rows, length = int(t.shape[0]), int(t.shape[1])
            self.max_rows = max(self.max_rows, rows)
            self.max_len = max(self.max_len, length)
            self.max_tokens = max(self.max_tokens, rows * length)
        return None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and self._handle is None:
            try:
                self._handle = model.register_forward_pre_hook(self._hook, with_kwargs=True)
            except Exception:
                pass
        if model is not None:
            try:
                self.grad_bytes = int(sum(p.numel() * p.element_size()
                                          for p in model.parameters() if p.requires_grad))
            except Exception:
                pass
        try:
            self.resid_bytes = optimizer_resid_bytes(kwargs.get("optimizer"))
        except Exception:
            pass
        if torch.cuda.is_available():
            self.alloc_begin = int(torch.cuda.memory_allocated())

    def on_step_end(self, args, state, control, **kwargs):
        if self.alloc_step1 is None and torch.cuda.is_available():
            self.alloc_step1 = int(torch.cuda.memory_allocated())

    @property
    def state_bytes(self) -> Optional[int]:
        if self.alloc_step1 is None or self.alloc_begin is None:
            return None
        return max(0, self.alloc_step1 - self.alloc_begin)

    def on_train_end(self, args, state, control, **kwargs):
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
            self._handle = None
        if torch.cuda.is_available():
            self.peak = int(torch.cuda.max_memory_allocated())


def optimizer_resid_bytes(optimizer) -> int:
    inner = optimizer
    for _ in range(4):
        if inner is None or isinstance(getattr(inner, "state", None), dict):
            break
        inner = getattr(inner, "optimizer", None)
    st = getattr(inner, "state", None)
    if not isinstance(st, dict):
        return 0
    return int(sum(int(v["r8"].numel()) * int(v["r8"].element_size()) for v in st.values()
                   if isinstance(v, dict) and torch.is_tensor(v.get("r8"))))


def row_len_max(datasets, max_length: Optional[int]) -> Optional[int]:
    best = 0
    for ds in datasets:
        if ds is None:
            continue
        lens = getattr(ds, "lengths", None)
        if lens is None:
            try:
                lens = [len(x) for x in ds["input_ids"]]
            except Exception:
                return None
        best = max(best, max((int(x) for x in lens), default=0))
    if not best:
        return None
    return int(min(best, max_length)) if max_length else int(best)


def agree(cb: Optional[A0MemoryCallback], *, training_args, max_length, pad_multiple,
          row_len: Optional[int]) -> MemoryProof:
    vals = [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, float(row_len or 0), -1.0, -1.0, -1.0]
    try:
        if cb is not None:
            vals[0], vals[1], vals[2] = float(cb.max_rows), float(cb.max_len), float(cb.max_tokens)
            vals[3] = float(cb.alloc_begin) if cb.alloc_begin is not None else -1.0
            vals[4] = float(cb.peak) if cb.peak is not None else -1.0
            _sb = cb.state_bytes
            vals[7] = float(_sb) if _sb is not None else -1.0
            vals[8] = float(cb.grad_bytes) if cb.grad_bytes is not None else -1.0
            if cb.peak is not None and cb.alloc_begin is not None:
                vals[9] = float(max(0, cb.peak - cb.alloc_begin))
        if torch.cuda.is_available():
            vals[5] = -float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    except Exception:
        pass
    if _dist():
        dev = ("cuda" if (torch.distributed.get_backend() == "nccl" and torch.cuda.is_available())
               else "cpu")
        t = torch.tensor(vals, dtype=torch.float64, device=dev)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        vals = [float(x) for x in t.tolist()]
    proof = MemoryProof(
        world_size=int(getattr(training_args, "world_size", 1) or 1),
        a0_micro=int(getattr(training_args, "per_device_train_batch_size", 1) or 1),
        a0_accum=int(getattr(training_args, "gradient_accumulation_steps", 1) or 1),
        a0_eval_micro=int(getattr(training_args, "per_device_eval_batch_size", 1) or 1),
        a0_gc=bool(getattr(training_args, "gradient_checkpointing", False)),
        max_length=int(max_length) if max_length else None,
        pad_multiple=int(pad_multiple or 1),
        row_len_max=int(vals[6]) or None,
        a0_max_rows=int(vals[0]), a0_max_len=int(vals[1]), a0_max_tokens=int(vals[2]),
        a0_alloc_begin_bytes=(int(vals[3]) if vals[3] >= 0 else None),
        a0_peak_bytes=(int(vals[4]) if vals[4] >= 0 else None),
        device_total_bytes=(int(-vals[5]) if vals[5] < 0 else None),
        a0_state_bytes=(int(vals[7]) if vals[7] >= 0 else None),
        grad_bytes=(int(vals[8]) if vals[8] >= 0 else None),
        a0_increment_bytes=(int(vals[9]) if vals[9] >= 0 else None),
        optimizer_resid_bytes=int(getattr(cb, "resid_bytes", 0) or 0) if cb is not None else 0,
    )
    return proof


def collective_timeout_secs(t_step: Optional[float], t_point: Optional[float]) -> Optional[float]:
    raw = os.environ.get(ENV_TIMEOUT, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return None
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return max(MIN_COLLECTIVE_TIMEOUT_SECS, TIMEOUT_STEP_MULT * float(t_step or 0.0),
               TIMEOUT_POINT_MULT * float(t_point or 0.0))


def _set_timeout(secs: float) -> None:
    torch.distributed.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=float(secs)))


@contextlib.contextmanager
def short_collective_timeout(secs: Optional[float], restore_secs: float, label: str,
                             setter: Optional[Callable[[float], None]] = None):
    if secs is None or not _dist():
        yield
        return
    set_ = setter or _set_timeout
    try:
        set_(secs)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            set_(restore_secs)
        except Exception:
            pass


ENV_A0_TIMEOUT = "SN56_A0_COLLECTIVE_TIMEOUT"
A0_TIMEOUT_FLOOR_SECS = 300.0
A0_TIMEOUT_POINT_MULT = 20.0


def a0_timeout_secs(max_gap: float) -> Optional[float]:
    raw = os.environ.get(ENV_A0_TIMEOUT, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return None
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return max(A0_TIMEOUT_FLOOR_SECS, A0_TIMEOUT_POINT_MULT * float(max_gap or 0.0))


class A0CollectiveTimeout(TrainerCallback):
    def __init__(self, restore_secs: float, *, setter: Optional[Callable[[float], None]] = None,
                 clock: Optional[Callable[[], float]] = None):
        self.restore_secs = float(restore_secs or 1800.0)
        self._set = setter or _set_timeout
        self._clock = clock or time.time
        self._t_last = None
        self._max_gap = 0.0
        self._seen_eval = False
        self.current: Optional[float] = None

    def _apply(self, secs: float, phase: str) -> None:
        try:
            self._set(secs)
            self.current = secs
        except Exception:
            pass

    def on_train_begin(self, args, state, control, **kwargs):
        self._t_last = self._clock()

    def on_evaluate(self, args, state, control, **kwargs):
        self._seen_eval = True

    def on_step_end(self, args, state, control, **kwargs):
        self._t_last = self._clock()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        now = self._clock()
        if self._seen_eval and self._t_last is not None:
            self._max_gap = max(self._max_gap, now - self._t_last)
            secs = a0_timeout_secs(self._max_gap)
            if secs is not None and (self.current is None or secs > self.current + 1.0):
                self._apply(secs, "set")
        self._t_last = now
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.current is not None:
            self._apply(self.restore_secs, "restore")
            self.current = None


def install_a0_timeout(trainer, restore_secs: float) -> Optional[A0CollectiveTimeout]:
    if not _dist():
        return None
    cb = A0CollectiveTimeout(restore_secs)
    trainer.callback_handler.callbacks.insert(0, cb)
    return cb
