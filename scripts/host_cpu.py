import math
import multiprocessing
import os
import sys
import time
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel

ENV_SWITCH = "SN56_CPU_QUOTA"
ENV_OWNED = "SN56_CPU_ENV_OWNED"
ENV_POOL_MAX = "SN56_TOKENIZE_WORKERS_MAX"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")
POOL_CAP = 8
CGROUP_ROOT = "/sys/fs/cgroup"
PROC_CGROUP = "/proc/self/cgroup"
_V1_MOUNTS = ("cpu,cpuacct", "cpuacct,cpu", "cpu")
_FALSY = ("0", "false", "no", "off")


class CpuQuota(BaseModel):
    host_cpus: int
    affinity_cpus: int
    cgroup_version: Optional[int] = None
    quota_cpus: Optional[float] = None
    quota_source: Optional[str] = None
    local_world_size: int = 1
    effective_cpus: int
    per_process_cpus: int
    constrained: bool


class ThreadPlan(BaseModel):
    quota: CpuQuota
    enabled: bool
    intra_op: Optional[int] = None
    interop: Optional[int] = None
    env: dict[str, str] = {}
    env_owned: list[str] = []
    env_kept: list[str] = []
    tokenize_workers: int = 1


class ApplyResult(BaseModel):
    entry: str
    intra_op_before: Optional[int] = None
    intra_op_after: Optional[int] = None
    interop_before: Optional[int] = None
    interop_after: Optional[int] = None
    interop_error: Optional[str] = None
    torch_loaded: bool = False


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _ancestors(mount: str, rel: str) -> list[str]:
    rel = (rel or "/").strip()
    parts = [p for p in rel.split("/") if p and p != ".."]
    out = []
    for k in range(len(parts), -1, -1):
        d = os.path.join(mount, *parts[:k]) if k else mount
        if d not in out:
            out.append(d)
    return out


def _proc_cgroup_lines(proc_cgroup: str) -> list[tuple[str, list[str], str]]:
    text = _read(proc_cgroup) or ""
    out = []
    for line in text.splitlines():
        bits = line.split(":", 2)
        if len(bits) == 3:
            out.append((bits[0], [c for c in bits[1].split(",") if c], bits[2]))
    return out


def parse_cpu_max(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    parts = text.split()
    try:
        if parts[0] == "max":
            return None
        quota = float(parts[0])
        period = float(parts[1]) if len(parts) > 1 else 100000.0
    except (ValueError, IndexError):
        return None
    return quota / period if quota > 0 and period > 0 else None


def parse_cfs(quota_text: Optional[str], period_text: Optional[str]) -> Optional[float]:
    try:
        quota = float(quota_text) if quota_text else -1.0
        period = float(period_text) if period_text else 0.0
    except ValueError:
        return None
    return quota / period if quota > 0 and period > 0 else None


def read_cgroup_quota(root: str = CGROUP_ROOT,
                      proc_cgroup: str = PROC_CGROUP) -> tuple[Optional[int], Optional[float], Optional[str]]:
    lines = _proc_cgroup_lines(proc_cgroup)
    best: Optional[float] = None
    src: Optional[str] = None
    if os.path.exists(os.path.join(root, "cgroup.controllers")):
        rel = next((p for h, c, p in lines if h == "0" and not c), "/")
        for d in _ancestors(root, rel):
            path = os.path.join(d, "cpu.max")
            q = parse_cpu_max(_read(path))
            if q is not None and (best is None or q < best):
                best, src = q, path
        return 2, best, src
    rel = next((p for h, c, p in lines if "cpu" in c), "/")
    version = None
    for m in _V1_MOUNTS:
        mount = os.path.join(root, m)
        if not os.path.isdir(mount):
            continue
        version = 1
        for d in _ancestors(mount, rel):
            qp = os.path.join(d, "cpu.cfs_quota_us")
            q = parse_cfs(_read(qp), _read(os.path.join(d, "cpu.cfs_period_us")))
            if q is not None and (best is None or q < best):
                best, src = q, qp
        if best is not None:
            break
    return version, best, src


def _affinity() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _local_world_size(env) -> int:
    for k in ("LOCAL_WORLD_SIZE", "SN56_LOCAL_WORLD_SIZE", "LOCAL_SIZE"):
        try:
            v = int(env.get(k, "") or 0)
        except ValueError:
            v = 0
        if v > 0:
            return v
    if str(env.get("LOCAL_RANK", "")).strip() != "":
        try:
            v = int(env.get("WORLD_SIZE", "") or 0)
        except ValueError:
            v = 0
        if v > 1:
            return v
    return 1


def per_process_cpus(effective: int, local_world_size: int) -> int:
    n = max(1, int(local_world_size))
    if n == 1:
        return max(1, int(effective))
    return max(1, (int(effective) - n) // n)


def detect(root: str = CGROUP_ROOT, proc_cgroup: str = PROC_CGROUP,
           host_cpus: Optional[int] = None, affinity_cpus: Optional[int] = None,
           env=None) -> CpuQuota:
    env = os.environ if env is None else env
    host = int(host_cpus if host_cpus is not None else (os.cpu_count() or 1))
    aff = int(affinity_cpus if affinity_cpus is not None else _affinity())
    version, quota, src = read_cgroup_quota(root, proc_cgroup)
    eff = aff
    if quota is not None:
        eff = min(eff, max(1, int(math.floor(quota + 1e-6))))
    eff = max(1, min(eff, host))
    lws = _local_world_size(env)
    return CpuQuota(host_cpus=host, affinity_cpus=aff, cgroup_version=version,
                    quota_cpus=(round(quota, 4) if quota is not None else None),
                    quota_source=src, local_world_size=lws, effective_cpus=eff,
                    per_process_cpus=per_process_cpus(eff, lws), constrained=eff < host)


def switched_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV_SWITCH, "1")).strip().lower() not in _FALSY


def tokenize_workers_for(q: CpuQuota, env=None) -> int:
    env = os.environ if env is None else env
    cap = POOL_CAP
    try:
        if env.get(ENV_POOL_MAX, "").strip():
            cap = max(1, int(env[ENV_POOL_MAX]))
    except ValueError:
        pass
    return max(1, min(q.effective_cpus - 1, cap))


def plan(q: CpuQuota, env=None) -> ThreadPlan:
    env = os.environ if env is None else env
    on = switched_on(env)
    owned_before = {v for v in str(env.get(ENV_OWNED, "")).split(",") if v}
    p = ThreadPlan(quota=q, enabled=on, tokenize_workers=tokenize_workers_for(q, env))
    want = q.per_process_cpus
    target: dict[str, str] = {}
    for var in THREAD_ENV:
        cur = env.get(var)
        mine = var in owned_before
        if cur not in (None, "") and not mine:
            p.env_kept.append(var)
            target[var] = str(cur)
        elif on and q.constrained:
            target[var] = str(want)
            p.env_owned.append(var)
        elif cur not in (None, "") and mine:
            p.env_owned.append(var)
    p.env = target
    if on and q.constrained:
        user_omp = env.get("OMP_NUM_THREADS") if "OMP_NUM_THREADS" in p.env_kept else None
        try:
            intra = max(1, int(user_omp)) if user_omp else want
        except ValueError:
            intra = want
        p.intra_op = intra
        p.interop = want
    return p


_PLAN: Optional[ThreadPlan] = None


def configure(env=None) -> ThreadPlan:
    global _PLAN
    env = os.environ if env is None else env
    q = detect(env=env)
    p = plan(q, env)
    for var in THREAD_ENV:
        if var in p.env_kept:
            continue
        if var in p.env:
            env[var] = p.env[var]
        elif var in p.env_owned:
            env.pop(var, None)
    owned = [v for v in p.env_owned if v in p.env]
    if owned:
        env[ENV_OWNED] = ",".join(owned)
    else:
        env.pop(ENV_OWNED, None)
    _PLAN = p
    return p


def current() -> ThreadPlan:
    if _PLAN is not None:
        return _PLAN
    try:
        return configure()
    except Exception:
        q = CpuQuota(host_cpus=os.cpu_count() or 1, affinity_cpus=os.cpu_count() or 1,
                     effective_cpus=os.cpu_count() or 1, per_process_cpus=os.cpu_count() or 1,
                     constrained=False)
        return ThreadPlan(quota=q, enabled=False)


def apply(entry: str, emit: Optional[Callable[..., None]] = None) -> ApplyResult:
    p = current()
    res = ApplyResult(entry=entry)
    torch = sys.modules.get("torch")
    res.torch_loaded = torch is not None
    try:
        if torch is not None:
            res.intra_op_before = int(torch.get_num_threads())
            res.interop_before = int(torch.get_num_interop_threads())
            if p.intra_op is not None and p.intra_op != res.intra_op_before:
                torch.set_num_threads(int(p.intra_op))
            if p.interop is not None and p.interop != res.interop_before:
                try:
                    torch.set_num_interop_threads(int(p.interop))
                except RuntimeError as exc:
                    res.interop_error = str(exc)[:200]
            res.intra_op_after = int(torch.get_num_threads())
            res.interop_after = int(torch.get_num_interop_threads())
    except Exception as exc:
        res.interop_error = (res.interop_error or "") + f" apply: {type(exc).__name__}: {exc}"[:200]
    if emit is not None:
        try:
            emit("cpu_threads", **res.model_dump(), **p.quota.model_dump(),
                 enabled=p.enabled, env=p.env, env_owned=p.env_owned, env_kept=p.env_kept,
                 tokenize_workers=p.tokenize_workers)
        except Exception:
            pass
    return res


ENV_PROBE_REF = "SN56_CPU_PROBE_REF"


def probe_ref_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV_PROBE_REF, "0")).strip().lower() not in _FALSY


def reference_intra_op(p: Optional[ThreadPlan] = None) -> Optional[int]:
    p = current() if p is None else p
    if p.intra_op is None:
        return None
    if "OMP_NUM_THREADS" in p.env_kept:
        try:
            return max(1, int(p.env["OMP_NUM_THREADS"]))
        except (KeyError, ValueError):
            return None
    return max(1, int(p.quota.affinity_cpus))


class reference_threads:

    def __init__(self, site: str, env=None):
        self.site = site
        self.env = env
        self.info: dict = {"site": site, "on": False, "intra_before": None, "intra_during": None,
                           "intra_after": None, "reference": None, "reason": ""}
        self._restore: Optional[int] = None

    def __enter__(self):
        torch = sys.modules.get("torch")
        try:
            if not probe_ref_on(self.env):
                self.info["reason"] = f"{ENV_PROBE_REF}=0"
                return self
            ref = reference_intra_op()
            self.info["reference"] = ref
            if torch is None or ref is None:
                self.info["reason"] = "no_torch" if torch is None else "unconstrained"
                return self
            cur = int(torch.get_num_threads())
            self.info["intra_before"] = cur
            if cur != ref:
                torch.set_num_threads(int(ref))
                self._restore = cur
            self.info["intra_during"] = int(torch.get_num_threads())
            self.info["on"] = True
            self.info["reason"] = "reference" if self._restore is not None else "already_reference"
        except Exception as exc:
            self.info["reason"] = f"error:{type(exc).__name__}"
        return self

    def __exit__(self, *exc):
        torch = sys.modules.get("torch")
        try:
            if torch is not None and self._restore is not None:
                torch.set_num_threads(int(self._restore))
            if torch is not None:
                self.info["intra_after"] = int(torch.get_num_threads())
        except Exception as e:
            self.info["reason"] += f";restore_error:{type(e).__name__}"
        return False


class PoolRun(BaseModel):
    mode: str
    workers: int
    n_items: int
    n_chunks: int
    secs: float
    error: Optional[str] = None


def chunk(items: Sequence, n_chunks: int) -> list[list]:
    n = len(items)
    n_chunks = max(1, min(int(n_chunks), n)) if n else 1
    size, extra = divmod(n, n_chunks)
    out, lo = [], 0
    for k in range(n_chunks):
        hi = lo + size + (1 if k < extra else 0)
        out.append(list(items[lo:hi]))
        lo = hi
    return out


def _cuda_initialized() -> bool:
    torch = sys.modules.get("torch")
    try:
        return bool(torch is not None and torch.cuda.is_initialized())
    except Exception:
        return True


def _worker_init() -> None:
    for var in THREAD_ENV:
        os.environ[var] = "1"
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass


POOL_TIMEOUT_BASE = 60.0
POOL_TIMEOUT_PER_ITEM = 0.05
POOL_POLL_SECS = 1.0
ENV_POOL_TIMEOUT = "SN56_POOL_TIMEOUT_SCALE"


def pool_timeout(n_items: int, env=None) -> float:
    env = os.environ if env is None else env
    try:
        scale = float(env.get(ENV_POOL_TIMEOUT, "") or 1.0)
    except ValueError:
        scale = 1.0
    return max(1e-3, scale) * (POOL_TIMEOUT_BASE + POOL_TIMEOUT_PER_ITEM * max(0, int(n_items)))


def pool_map(fn: Callable[[Any], Any], chunks: list, workers: Optional[int] = None,
             min_items: int = 0, n_items: Optional[int] = None) -> tuple[list, PoolRun]:
    t0 = time.perf_counter()
    w = int(workers if workers is not None else current().tokenize_workers)
    n = int(n_items if n_items is not None else sum(len(c) for c in chunks))
    w = max(1, min(w, len(chunks)))
    if (w <= 1 or n < min_items or "fork" not in multiprocessing.get_all_start_methods()
            or _cuda_initialized()):
        out = [fn(c) for c in chunks]
        return out, PoolRun(mode="serial", workers=1, n_items=n, n_chunks=len(chunks),
                            secs=round(time.perf_counter() - t0, 3))
    try:
        ctx = multiprocessing.get_context("fork")
        timeout = pool_timeout(n)
        with ctx.Pool(processes=w, initializer=_worker_init) as pool:
            procs = list(getattr(pool, "_pool", []) or [])
            res = pool.map_async(fn, chunks, chunksize=1)
            deadline = time.monotonic() + timeout
            while not res.ready():
                dead = [p.exitcode for p in procs if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f"pool worker died (exitcode {dead[0]})")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"pool_map exceeded {timeout:.0f}s")
                res.wait(min(POOL_POLL_SECS, left))
            out = res.get(timeout=0)
        return out, PoolRun(mode="pool", workers=w, n_items=n, n_chunks=len(chunks),
                            secs=round(time.perf_counter() - t0, 3))
    except Exception as exc:
        out = [fn(c) for c in chunks]
        return out, PoolRun(mode="serial_fallback", workers=1, n_items=n, n_chunks=len(chunks),
                            secs=round(time.perf_counter() - t0, 3),
                            error=f"{type(exc).__name__}: {exc}"[:300])


try:
    configure()
except Exception:
    _PLAN = None
