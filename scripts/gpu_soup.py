import os
import time
from typing import Callable, Optional

import torch
from pydantic import BaseModel

import batched_eval
import pinned_stage

ENV = "SN56_GPU_SOUP"
ENV_CANARY = "SN56_GPU_SOUP_CANARY"
ENV_DEVICE = "SN56_GPU_SOUP_DEVICE"
ENV_CHUNK = "SN56_GPU_SOUP_CHUNK"
CHUNK_ELEMS = 1 << 24
MARGIN_BYTES = 1 << 30
WHERE_CPU, WHERE_RESIDENT, WHERE_STREAM = "cpu", "gpu_resident", "gpu_stream"

_CANARY: dict = {}
_DISABLED: Optional[str] = None


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def canary_enabled() -> bool:
    return os.environ.get(ENV_CANARY, "1").strip().lower() not in ("0", "false", "no", "off")


def chunk_elems() -> int:
    try:
        return max(1, int(os.environ.get(ENV_CHUNK, str(CHUNK_ELEMS))))
    except ValueError:
        return CHUNK_ELEMS


def verdict_state() -> tuple:
    return (frozenset(k for k, v in _CANARY.items() if v), _DISABLED)


class DeviceMemory(BaseModel):
    free: int
    alloc: int
    peak: int


class EvalMemory(BaseModel):
    need: int
    floor: int


def memory_probe(device: torch.device) -> Optional[DeviceMemory]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(device)
    alloc = int(torch.cuda.memory_allocated(device))
    cached = max(0, int(torch.cuda.memory_reserved(device)) - alloc)
    return DeviceMemory(free=int(free) + cached, alloc=alloc,
                        peak=int(torch.cuda.max_memory_allocated(device)))


def eval_memory(trainer) -> Optional[EvalMemory]:
    if trainer is None or not getattr(trainer, "_evb_on", False):
        return None
    ds = getattr(trainer, "eval_dataset", None)
    bpt = getattr(trainer, "_evb_bytes_per_token", None)
    plan = getattr(trainer, "_evb_plan", None)
    if plan is None and ds is not None:
        try:
            plan = batched_eval._PLANS.get(ds)
        except TypeError:
            plan = None
    if plan is None or not bpt:
        return None
    need = int(plan.tokens_max * float(bpt))
    return EvalMemory(need=need, floor=int(need / batched_eval.MEM_FRACTION) + 1)


class SoupPlan(BaseModel):
    op: str
    where: str = WHERE_CPU
    reason: str = ""
    device: str = "cpu"
    free: Optional[int] = None
    alloc: Optional[int] = None
    peak: Optional[int] = None
    eval_need: Optional[int] = None
    eval_floor: Optional[int] = None
    room: Optional[int] = None
    transient: int = 0
    acc_bytes: int = 0
    n_acc: int = 0
    cand_bytes: int = 0
    n_cands: int = 0
    pool_vram: int = 0
    acc_host: bool = False


def transient_bytes(max_numel: int, chunk: Optional[int] = None) -> int:
    c = int(chunk or chunk_elems())
    return int(12 * int(max_numel) + 16 * min(c, max(1, int(max_numel))))


def plan_placement(*, op: str, mem: Optional[DeviceMemory], ev: Optional[EvalMemory],
                   acc_bytes: int, n_acc: int, cand_bytes: int, n_cands: int, transient: int,
                   acc_host: bool = False, margin: int = MARGIN_BYTES,
                   device: str = "cuda") -> SoupPlan:
    p = SoupPlan(op=op, device=device, acc_bytes=int(acc_bytes), n_acc=int(n_acc),
                 cand_bytes=int(cand_bytes), n_cands=int(n_cands), transient=int(transient),
                 acc_host=bool(acc_host))
    if mem is None:
        p.reason = "no_device_memory"
        return p
    p.free, p.alloc, p.peak = mem.free, mem.alloc, mem.peak
    if ev is not None:
        p.eval_need, p.eval_floor = ev.need, ev.floor
    under_peak = mem.peak - mem.alloc - margin
    under_free = mem.free - margin
    if transient > under_peak or transient > under_free:
        p.reason = "vram_transient"
        return p
    if ev is None:
        room = 0
    else:
        room = min(mem.peak - mem.alloc - max(ev.need, transient),
                   mem.free - ev.floor - transient) - margin
    room = max(0, int(room))
    p.room = room
    acc_need = int(n_acc) * int(acc_bytes)
    if not acc_host and n_acc > 0 and acc_need <= room:
        p.where = WHERE_RESIDENT
        left = room - acc_need
    else:
        p.where = WHERE_STREAM
        left = room if not acc_host else 0
        p.reason = "acc_host" if acc_host else ("eval_memory_unknown" if ev is None
                                                else "acc_over_room")
    if cand_bytes > 0 and n_cands > 0:
        p.pool_vram = int(max(0, min(int(n_cands), left // int(cand_bytes))))
    return p


def _int_view(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.float32, torch.int32):
        return t.view(torch.int32)
    if t.dtype in (torch.bfloat16, torch.float16, torch.int16):
        return t.view(torch.int16)
    if t.dtype in (torch.float64, torch.int64):
        return t.view(torch.int64)
    return t


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return bool(torch.equal(_int_view(a.contiguous()), _int_view(b.contiguous())))


_DEVICE_FAULT: Optional[Callable] = None


class SoupEngine:

    def __init__(self, plan: SoupPlan, device: torch.device, named: list, trainer=None):
        self.plan = plan
        self.device = device
        self.named = named
        self.trainer = trainer
        self._cache: dict = {}
        self._accs: list = []
        self._adopted: list = []
        self.t_open = time.perf_counter()
        self.n_trials = 0
        self.load_secs = 0.0
        self.last_load_secs = 0.0
        self.trial_secs: list = []
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.canary_runs: list = []
        self.fail: Optional[str] = None
        self.grew_cache = False
        self._closed = False

    @classmethod
    def build(cls, op: str, named: list, *, trainer=None, n_acc: int = 1, cand_bytes: int = 0,
              n_cands: int = 0, numel: int = 0, max_numel: int = 0,
              acc_host: bool = False) -> "SoupEngine":
        named = list(named)
        devs = {p.device for _, p in named}
        forced = os.environ.get(ENV_DEVICE, "").strip().lower()
        sharded = any(hasattr(p, "ds_id") or hasattr(p, "_is_flat_param") for _, p in named)
        reason = ""
        if not enabled():
            reason = "off"
        elif _DISABLED is not None:
            reason = f"disabled:{_DISABLED}"
        elif sharded:
            reason = "sharded"
        elif len(devs) != 1:
            reason = "devices"
        dev = next(iter(devs)) if len(devs) == 1 else torch.device("cpu")
        if not reason and forced == "cpu":
            dev = torch.device("cpu")
        elif not reason and dev.type != "cuda":
            reason = "no_cuda"
        if reason:
            plan = SoupPlan(op=op, reason=reason, device=str(dev), acc_host=bool(acc_host))
            return cls(plan, torch.device("cpu"), named, trainer)
        trans = transient_bytes(max_numel)
        if forced == "cpu":
            mem = _TEST_MEMORY() if _TEST_MEMORY is not None else DeviceMemory(
                free=1 << 46, alloc=0, peak=1 << 46)
            ev = _TEST_EVAL() if _TEST_EVAL is not None else EvalMemory(need=0, floor=0)
        else:
            try:
                mem = memory_probe(dev)
            except Exception:
                mem = None
            try:
                ev = eval_memory(trainer)
            except Exception:
                ev = None
        plan = plan_placement(op=op, mem=mem, ev=ev, acc_bytes=4 * int(numel), n_acc=n_acc,
                              cand_bytes=cand_bytes, n_cands=n_cands, transient=trans,
                              acc_host=acc_host, device=str(dev))
        eng = cls(plan, dev if plan.where != WHERE_CPU else torch.device("cpu"), named, trainer)
        return eng

    @property
    def on_device(self) -> bool:
        return self.plan.where != WHERE_CPU

    @property
    def where(self) -> str:
        return self.plan.where

    def _to_dev(self, t: torch.Tensor) -> torch.Tensor:
        if t.device == self.device and self.device.type == "cuda":
            return t
        if self.device.type == "cpu":
            return t.clone()
        out = torch.empty(t.shape, dtype=t.dtype, device=self.device)
        if t.is_contiguous():
            pinned_stage.host_to_device(t, out)
        else:
            out.copy_(t.to(self.device))
        self.h2d_bytes += t.numel() * t.element_size()
        return out

    def _to_host(self, t: torch.Tensor, dst: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dst is None:
            dst = torch.empty(t.shape, dtype=t.dtype)
        if self.device.type == "cpu":
            dst.copy_(t)
        elif t.is_contiguous() and dst.is_contiguous():
            pinned_stage.device_to_host(t, dst)
        else:
            dst.copy_(t.to("cpu"))
        self.d2h_bytes += t.numel() * t.element_size()
        return dst

    def _is_dev(self, t: torch.Tensor) -> bool:
        if self.device.type == "cuda":
            return t.device.type == "cuda"
        return any(t is a for acc in self._accs + self._adopted for a in acc.values()) or any(
            t is x for _, d in self._cache.values() for x in d.values())

    def _upload(self, t: torch.Tensor) -> torch.Tensor:
        try:
            out = self._to_dev(t)
        except torch.OutOfMemoryError:
            self._fail("oom:upload")
            return t
        key = ("upload", self.where)
        if canary_enabled() and not _CANARY.get(key, False):
            if not same_bits(self._to_host(out), t):
                self._fail("canary_mismatch:upload")
                return t
            _CANARY[key] = True
        return out

    def adopt(self, acc: Optional[dict]) -> None:
        if acc is None or self.where != WHERE_RESIDENT:
            return
        self._adopted.append(acc)
        for k in list(acc.keys()):
            if self.where != WHERE_RESIDENT:
                break
            acc[k] = self._upload(acc[k])

    def release(self, acc: Optional[dict]) -> None:
        if acc is not None:
            self._accs = [a for a in self._accs if a is not acc]

    def home(self, acc: Optional[dict]) -> None:
        if acc is None:
            return
        for k in list(acc.keys()):
            t = acc[k]
            if self._is_dev(t) or (t.device.type == "cuda"):
                acc[k] = self._to_host(t)
        self._accs = [a for a in self._accs if a is not acc]
        self._adopted = [a for a in self._adopted if a is not acc]

    def _dev_state(self, state: Optional[dict]) -> Optional[dict]:
        if state is None or not self.on_device:
            return state
        hit = self._cache.get(id(state))
        if hit is not None and hit[0] is state:
            return hit[1]
        if len(self._cache) >= self.plan.pool_vram:
            return state
        d = {}
        for k, v in state.items():
            d[k] = self._upload(v)
            if not self.on_device:
                return state
        self._cache[id(state)] = (state, d)
        self.grew_cache = True
        return d

    def _operand(self, t: torch.Tensor) -> torch.Tensor:
        return t if self._is_dev(t) else self._to_dev(t)

    def _run(self, kind: str, dst: torch.Tensor, fn: Callable, inputs: list,
             host: Optional[list] = None) -> None:
        try:
            self._run_device(kind, dst, fn, inputs, host)
        except torch.OutOfMemoryError as exc:
            self._oom(kind, dst, fn, inputs, host, exc)

    def _oom(self, kind: str, dst: torch.Tensor, fn: Callable, inputs: list,
             host: Optional[list], exc: BaseException) -> None:
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        host_in = [h if h is not None else (x if not self._is_dev(x) else self._to_host(x))
                   for x, h in zip(inputs, host or [None] * len(inputs))]
        ref = fn(*host_in)
        host_in = None
        self._write(dst, None, ref)
        self._fail(f"oom:{kind}")

    def _run_device(self, kind: str, dst: torch.Tensor, fn: Callable, inputs: list,
                    host: Optional[list] = None) -> None:
        canary = canary_enabled() and not _CANARY.get((kind, self.where), False)
        ref = None
        if canary:
            host_in = [h if h is not None else (x if not self._is_dev(x) else self._to_host(x))
                       for x, h in zip(inputs, host or [None] * len(inputs))]
            ref = fn(*host_in)
            host_in = None
        dst_dev = dst if self._is_dev(dst) or (self.device.type == "cuda"
                                               and dst.device.type == "cuda") else None
        ops = [self._operand(x) for x in inputs]
        out = dst_dev if dst_dev is not None else torch.empty(dst.shape, dtype=dst.dtype,
                                                              device=self.device)
        n = out.numel()
        c = chunk_elems()
        flat_ok = out.is_contiguous() and all(o.is_contiguous() and o.numel() == n for o in ops)
        if flat_ok and n > c:
            of = out.view(-1)
            fl = [o.reshape(-1) for o in ops]
            for lo in range(0, n, c):
                hi = min(n, lo + c)
                r = fn(*[f[lo:hi] for f in fl])
                if _DEVICE_FAULT is not None:
                    r = _DEVICE_FAULT(kind, r)
                of[lo:hi].copy_(r)
                del r
        else:
            r = fn(*ops)
            if _DEVICE_FAULT is not None:
                r = _DEVICE_FAULT(kind, r)
            out.copy_(r)
            del r
        ops = None
        if canary:
            got = self._to_host(out)
            ok = same_bits(got, ref)
            self.canary_runs.append((kind, ok))
            if not ok:
                self._write(dst, out, ref)
                self._fail(f"canary_mismatch:{kind}")
                return
            got = None
        if dst_dev is None:
            self._to_host(out, dst)

    def _write(self, dst: torch.Tensor, out: torch.Tensor, ref: torch.Tensor) -> None:
        if dst.device.type == "cpu" and not self._is_dev(dst):
            dst.copy_(ref)
        else:
            dst.copy_(ref.to(dst.device))

    def _fail(self, reason: str) -> None:
        global _DISABLED
        if _DISABLED is None:
            _DISABLED = reason
        self.fail = reason
        self._migrate_home()
        self.plan.where = WHERE_CPU
        self.device = torch.device("cpu")

    def _migrate_home(self) -> None:
        for acc in self._accs + self._adopted:
            for k in list(acc.keys()):
                t = acc[k]
                if self._is_dev(t) or t.device.type == "cuda":
                    acc[k] = self._to_host(t)
        self._accs = []
        self._adopted = []
        self._cache.clear()

    def _canary_done(self, kind: str) -> None:
        if not self.on_device or not canary_enabled() or _CANARY.get((kind, self.where)):
            return
        runs = [ok for k, ok in self.canary_runs if k == kind]
        if runs and all(runs):
            _CANARY[(kind, self.where)] = True

    def canary_state(self, kind: str = "trial") -> str:
        if not self.on_device:
            return "fail" if self.fail else "n/a"
        if not canary_enabled():
            return "off"
        return "pass" if _CANARY.get((kind, self.where)) else "pending"

    @torch.no_grad()
    def init_acc(self, state: dict) -> dict:
        if self.where != WHERE_RESIDENT:
            return {n: t.float().clone() for n, t in state.items()}
        src = self._dev_state(state)
        acc = {}
        self._accs.append(acc)
        for n, t in src.items():
            if self.where != WHERE_RESIDENT:
                acc[n] = state[n].float().clone()
                continue
            try:
                dst = torch.empty(t.shape, dtype=torch.float32, device=self.device)
            except torch.OutOfMemoryError:
                self._fail("oom:init")
                acc[n] = state[n].float().clone()
                continue
            acc[n] = dst
            self._run("init", dst, lambda a: a.float().clone(), [t], host=[state[n]])
        self._canary_done("init")
        return acc

    @torch.no_grad()
    def init_scaled(self, state: dict, m) -> dict:
        if self.where != WHERE_RESIDENT:
            return {n: t.float().clone() * m for n, t in state.items()}
        src = self._dev_state(state)
        acc = {}
        self._accs.append(acc)
        for n, t in src.items():
            if self.where != WHERE_RESIDENT:
                acc[n] = state[n].float().clone() * m
                continue
            try:
                dst = torch.empty(t.shape, dtype=torch.float32, device=self.device)
            except torch.OutOfMemoryError:
                self._fail("oom:init_scaled")
                acc[n] = state[n].float().clone() * m
                continue
            acc[n] = dst
            self._run("init_scaled", dst, lambda a: a.float().clone() * m, [t], host=[state[n]])
        self._canary_done("init_scaled")
        return acc

    def _timed_load(self, fn) -> None:
        t0 = time.perf_counter()
        fn()
        if self.on_device and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_load_secs = time.perf_counter() - t0
        self.load_secs += self.last_load_secs

    @torch.no_grad()
    def load_trial(self, acc: dict, state: dict, inv: float) -> None:
        def _go():
            st = self._dev_state(state)
            for name, p in self.named:
                if name in acc and name in st:
                    if not self.on_device:
                        p.data.copy_(((acc[name] + state[name].float()) * inv).to(p.dtype))
                        continue
                    dt = p.dtype
                    self._run("trial", p.data, lambda a, b: ((a + b.float()) * inv).to(dt),
                              [acc[name], st[name]], host=[None, state[name]])
            self._canary_done("trial")
        self._timed_load(_go)
        self.n_trials += 1

    @torch.no_grad()
    def load_scaled(self, acc: dict, inv: float) -> None:
        def _go():
            for name, p in self.named:
                if name in acc:
                    if not self.on_device:
                        p.data.copy_((acc[name] * inv).to(p.dtype))
                        continue
                    dt = p.dtype
                    self._run("scaled", p.data, lambda a: (a * inv).to(dt), [acc[name]])
            self._canary_done("scaled")
        self._timed_load(_go)

    @torch.no_grad()
    def load_weighted(self, base: dict, extra: dict, w: float, inv: float) -> None:
        def _go():
            ex = self._dev_state(extra)
            for name, p in self.named:
                if name not in base:
                    continue
                if not self.on_device:
                    t = base[name]
                    if name in extra:
                        t = t + w * extra[name].float()
                    p.data.copy_((t * inv).to(p.dtype))
                    del t
                    continue
                dt = p.dtype
                if name in ex:
                    self._run("weighted", p.data,
                              lambda a, e: ((a + w * e.float()) * inv).to(dt),
                              [base[name], ex[name]], host=[None, extra[name]])
                else:
                    self._run("scaled", p.data, lambda a: (a * inv).to(dt), [base[name]])
            self._canary_done("weighted")
        self._timed_load(_go)

    @torch.no_grad()
    def load_minus(self, acc: dict, sub: dict, w: float, inv: float) -> None:
        def _go():
            for name, p in self.named:
                if name not in acc:
                    continue
                if not self.on_device:
                    t = acc[name]
                    if name in sub:
                        t = t - (1.0 - w) * sub[name].float()
                    p.data.copy_((t * inv).to(p.dtype))
                    del t
                    continue
                dt = p.dtype
                if name in sub:
                    self._run("minus", p.data,
                              lambda a, e: ((a - (1.0 - w) * e.float()) * inv).to(dt),
                              [acc[name], sub[name]])
                else:
                    self._run("scaled", p.data, lambda a: (a * inv).to(dt), [acc[name]])
            self._canary_done("minus")
        self._timed_load(_go)

    @torch.no_grad()
    def add(self, acc: dict, state: dict) -> None:
        st = self._dev_state(state)
        for name in acc:
            if name not in state:
                continue
            if not self.on_device or (self.plan.acc_host and not self._is_dev(acc[name])):
                acc[name] += state[name].float()
                continue
            self._run("add", acc[name], lambda a, b: a + b.float(), [acc[name], st[name]],
                      host=[None, state[name]])
        self._canary_done("add")

    @torch.no_grad()
    def add_acc(self, total: dict, s: dict) -> None:
        for nm in total:
            if nm not in s:
                continue
            if not self.on_device or (self.plan.acc_host and not self._is_dev(total[nm])):
                total[nm] += s[nm]
                continue
            self._run("add_acc", total[nm], lambda a, b: a + b, [total[nm], s[nm]])
        self._canary_done("add_acc")

    def trial_event(self, *, index, eval_secs: float, accepted: Optional[bool] = None,
                    pass_no: int = 1, label: str = "") -> None:
        total = float(self.last_load_secs) + float(eval_secs or 0.0)
        self.trial_secs.append(total)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for acc in list(self._adopted):
            try:
                self.home(acc)
            except Exception:
                raise
        self._cache.clear()
        self._accs = []
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                if self.grew_cache or self.n_trials or self.load_secs or self.h2d_bytes:
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if self.plan.where == WHERE_CPU and not self.fail and self.n_trials == 0 \
                and not self.load_secs:
            return


_TEST_MEMORY: Optional[Callable] = None
_TEST_EVAL: Optional[Callable] = None
