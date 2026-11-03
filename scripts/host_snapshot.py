import ctypes
import gc
import os
import shutil
import sys
import threading
import time
import weakref
from typing import Optional

import torch
from pydantic import BaseModel

import pinned_stage

try:
    import psutil
except ImportError:
    psutil = None

ENV = "SN56_PINNED_SNAPSHOT"
ENV_STAGE_CANARY_MB = "SN56_STAGE_CANARY_MB"
STAGE_CANARY_BYTES = 512 * 1024 ** 2
ENV_SHARE = "SN56_SNAPSHOT_SHARE"
MODE = "sync_pageable"
CGROUP_MEMORY_FILES = (
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
     "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
)
CGROUP_UNLIMITED = 1 << 62
WRITER_RAM_MARGIN_FLOOR = 2 * 1024 ** 3
WRITER_RAM_MARGIN_SNAPS = 2

_STATS_LOCK = threading.Lock()
STATS = {"snapshots": 0, "bytes": 0, "secs": 0.0, "max_secs": 0.0, "shared": 0,
         "shared_bytes": 0}


HOST_RESERVE_FLOOR = 2 * 1024 ** 3
HOST_RESERVE_FRAC = 0.03
POOL_CAP_FLOOR = 2


class RamBudget(BaseModel):
    limit: Optional[int] = None
    usage: Optional[int] = None
    host_avail: Optional[int] = None
    avail: Optional[int] = None
    source: str = "none"

    def fields(self) -> dict:
        return {"cgroup_limit": self.limit, "cgroup_usage": self.usage,
                "host_avail": self.host_avail, "ram_avail": self.avail, "ram_source": self.source}


def _cgroup_read(files=None, unlimited: int = CGROUP_UNLIMITED) -> tuple:
    for idx, (lim_p, cur_p) in enumerate(CGROUP_MEMORY_FILES if files is None else files):
        src = "cgroup_v2" if str(lim_p).endswith("memory.max") else (
            "cgroup_v1" if "limit_in_bytes" in str(lim_p) else f"cgroup_{idx}")
        try:
            with open(lim_p) as f:
                lim = f.read().strip()
            if lim == "max":
                return None, None, "cgroup_unlimited"
            lim_i = int(lim)
            if lim_i >= unlimited:
                return None, None, "cgroup_unlimited"
            with open(cur_p) as f:
                cur = int(f.read().strip())
            return lim_i, max(0, cur - _cgroup_reclaimable(cur_p)), src
        except Exception:
            continue
    return None, None, "none"


_RECLAIM_KEYS = {
    "cgroup_v2": ("file", ("shmem", "file_dirty", "file_writeback")),
    "cgroup_v1": ("total_cache", ("total_shmem", "total_dirty", "total_writeback")),
}


def _cgroup_reclaimable(cur_p) -> int:
    try:
        stat_p = os.path.join(os.path.dirname(str(cur_p)), "memory.stat")
        with open(stat_p) as f:
            stat = {}
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    stat[parts[0]] = int(parts[1])
        key = "cgroup_v2" if "file" in stat else ("cgroup_v1" if "total_cache" in stat else None)
        if key is None:
            return 0
        total, pinned = _RECLAIM_KEYS[key]
        return max(0, stat.get(total, 0) - sum(stat.get(k, 0) for k in pinned))
    except Exception:
        return 0


def _cgroup_limit(files=None, unlimited: int = CGROUP_UNLIMITED) -> tuple:
    lim, cur, _ = _cgroup_read(files, unlimited)
    return lim, cur


def _host_meminfo_avail() -> Optional[int]:
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return None


def ram_budget(files=None, unlimited: int = CGROUP_UNLIMITED, host_avail_fn=None) -> RamBudget:
    host = (host_avail_fn or _host_meminfo_avail)()
    lim, cur, src = _cgroup_read(files, unlimited)
    avail = host
    if lim is not None:
        room = max(0, lim - cur)
        avail = room if host is None else min(host, room)
    elif src in ("none", "cgroup_unlimited") and host is not None:
        src = "meminfo" if src == "none" else src
    return RamBudget(limit=lim, usage=cur, host_avail=host, avail=avail, source=src)


def host_reserve_bytes(limit: Optional[int]) -> int:
    if not limit:
        return HOST_RESERVE_FLOOR
    return int(max(HOST_RESERVE_FLOOR, HOST_RESERVE_FRAC * float(limit)))


def soup_phase_peak_bytes(snap_bytes: int, numel: int, max_tensor_numel: int = 0) -> int:
    return int(4 * int(numel) + int(snap_bytes) + 2 * 4 * int(max_tensor_numel))


STAGE_TRANSIENT_F32 = 3


def post_pass1_stage_need(stage: str, *, snap_bytes: int, numel: int,
                          max_tensor_numel: int = 0) -> int:
    trans = STAGE_TRANSIENT_F32 * 4 * int(max_tensor_numel)
    f32 = {"pass2": 0, "a0_probe": 1, "blind_soup": 2}[stage]
    return int(f32 * 4 * int(numel) + int(snap_bytes) + trans)


def stage_room(budget: "RamBudget") -> Optional[int]:
    if budget is None or budget.avail is None:
        return None
    return int(budget.avail) - host_reserve_bytes(budget.limit)


ENV_CACHE_DROP = "SN56_WRITE_CACHE_DROP"
COPY_CHUNK = 64 * 1024 ** 2
COPY_FLUSH_EVERY = 1024 ** 3


def cache_drop_enabled() -> bool:
    return os.environ.get(ENV_CACHE_DROP, "1").strip().lower() not in ("0", "false", "no", "off")


def _drop_fd(fd: int, flush: bool = True) -> None:
    if flush:
        os.fsync(fd)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def drop_written_cache(path: str, *, site: str = "") -> dict:
    info = {"files": 0, "bytes": 0, "secs": 0.0, "errors": 0}
    if not cache_drop_enabled() or not path or not os.path.exists(path):
        return info
    t0 = time.time()
    if os.path.isdir(path):
        files = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs]
    else:
        files = [path]
    for f in files:
        try:
            if os.path.islink(f) or not os.path.isfile(f):
                continue
            fd = os.open(f, os.O_RDONLY)
            try:
                _drop_fd(fd)
            finally:
                os.close(fd)
            info["files"] += 1
            info["bytes"] += os.path.getsize(f)
        except Exception:
            info["errors"] += 1
    info["secs"] = round(time.time() - t0, 3)
    return info


def copy_file_nocache(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
    if not cache_drop_enabled():
        return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    if not follow_symlinks and os.path.islink(src):
        return shutil.copy2(src, dst, follow_symlinks=False)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        since = 0
        while True:
            buf = fi.read(COPY_CHUNK)
            if not buf:
                break
            fo.write(buf)
            since += len(buf)
            if since >= COPY_FLUSH_EVERY:
                fo.flush()
                _drop_fd(fo.fileno())
                _drop_fd(fi.fileno(), flush=False)
                since = 0
        fo.flush()
        _drop_fd(fo.fileno())
        _drop_fd(fi.fileno(), flush=False)
    shutil.copystat(src, dst, follow_symlinks=follow_symlinks)
    return dst


HOST_TRIM_ENV = "SN56_HOST_TRIM"
_LIBC = None


def trim_enabled() -> bool:
    return os.environ.get(HOST_TRIM_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _rss_bytes() -> Optional[int]:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def release_host_memory(collect: bool = True) -> dict:
    global _LIBC
    t0 = time.perf_counter()
    if collect:
        gc.collect()
    info = {"trimmed": False, "rss_before": _rss_bytes()}
    if trim_enabled():
        try:
            if _LIBC is None:
                _LIBC = ctypes.CDLL("libc.so.6")
            info["trimmed"] = bool(_LIBC.malloc_trim(0))
        except (OSError, AttributeError):
            pass
    info["rss_after"] = _rss_bytes()
    info["secs"] = round(time.perf_counter() - t0, 4)
    return info


def pool_cap_from_ram(*, limit: Optional[int], usage: Optional[int], pool_bytes: int,
                      snap_bytes: int, soup_peak: int, reserve: int,
                      floor: int = POOL_CAP_FLOOR) -> Optional[int]:
    if not limit or usage is None or snap_bytes <= 0:
        return None
    other = max(0, int(usage) - int(pool_bytes))
    room = int(limit) - other - int(reserve) - int(soup_peak)
    return max(int(floor), room // int(snap_bytes))


def available_ram_bytes(files=None, unlimited: int = CGROUP_UNLIMITED) -> Optional[int]:
    return ram_budget(files, unlimited).avail


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def share_enabled() -> bool:
    return os.environ.get(ENV_SHARE, "1").strip().lower() not in ("0", "false", "no", "off")


def _extent_bytes(t: torch.Tensor) -> int:
    if t.numel() == 0:
        return 0
    span = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    return span * t.element_size()


def plan(tensors: dict) -> tuple:
    by_storage: dict = {}
    order = []
    for k, t in tensors.items():
        st = t.untyped_storage()
        sk = (str(t.device), st.data_ptr(), st.nbytes()) if t.numel() else ("empty", id(t), 0)
        if sk not in by_storage:
            by_storage[sk] = []
            order.append(sk)
        by_storage[sk].append((k, t))
    groups = []
    for sk in order:
        items = by_storage[sk]
        starts = [t.storage_offset() * t.element_size() for _, t in items]
        ends = [s + _extent_bytes(t) for s, (_, t) in zip(starts, items)]
        lo, hi = min(starts), max(ends)
        members = []
        for (k, t), s in zip(items, starts):
            rel = s - lo
            if rel % t.element_size():
                raise ValueError(f"misaligned alias {k}")
            members.append((k, t, rel))
        groups.append((lo, hi - lo, members))
    return tuple(g[1] for g in groups), groups


def layout_of(tensors: dict) -> tuple:
    return plan(tensors)[0]


def _cuda_devices(tensors: dict) -> list:
    devs = []
    for t in tensors.values():
        if t.device.type == "cuda" and t.device not in devs:
            devs.append(t.device)
    return devs


def _synchronize(devs: list) -> None:
    for d in devs:
        torch.cuda.synchronize(d)


class _Copy:
    __slots__ = ("nbytes", "n_storages", "n_devices", "secs", "consumers", "shared",
                 "shared_bytes", "step", "mode")

    def __init__(self, nbytes, n_storages, n_devices, secs, consumer, step, mode=MODE):
        self.nbytes, self.n_storages, self.n_devices = nbytes, n_storages, n_devices
        self.secs, self.step, self.mode = secs, step, mode
        self.consumers = [consumer]
        self.shared = 0
        self.shared_bytes = 0


class _Window:

    def __init__(self, module, step, generation):
        self.module_ref = weakref.ref(module)
        self.step = step
        self.generation = generation
        self.entries: dict = {}
        self.copies: list = []

    def matches(self, module, step, generation) -> bool:
        return (self.module_ref() is module and self.step == step
                and self.generation == generation)


_SHARE_LOCK = threading.RLock()
_WINDOW: Optional[_Window] = None
_GENERATION = 0


def end_share(reason: str = "") -> None:
    global _WINDOW
    with _SHARE_LOCK:
        _, _WINDOW = _WINDOW, None


def weights_changed(reason: str = "") -> None:
    global _GENERATION
    with _SHARE_LOCK:
        _GENERATION += 1
    end_share(reason or "weights_changed")
    pu = sys.modules.get("precise_update")
    if pu is not None:
        pu.weights_changed()


@torch.no_grad()
def restore_params(named, state: dict) -> str:
    used = "pageable"
    check = []
    for name, p in named:
        if name not in state:
            continue
        src, dst = state[name], p.data
        if (src.device.type == "cpu" and dst.device.type == "cuda" and src.dtype == dst.dtype
                and src.shape == dst.shape and pinned_stage.STAGE.h2d is not False
                and pinned_stage.host_to_device(src, dst) == "staged"):
            used = "staged"
            if pinned_stage.STAGE.h2d is None:
                check.append((dst, src))
            continue
        dst.copy_(src.to(dst.device))
    if check:
        bad = nbytes = 0
        sample = _canary_sample([(pinned_stage.PinnedStager._flat_bytes(d),
                                  pinned_stage.PinnedStager._flat_bytes(s)) for d, s in check],
                                stage_canary_bytes())
        for dst_b, src_b in sample:
            back = dst_b.to("cpu")
            if back.numel() and not torch.equal(back, src_b):
                bad += 1
        if bad:
            for dst, src in check:
                dst.copy_(src.to(dst.device))
        pinned_stage.STAGE.h2d = bad == 0
        if bad:
            pinned_stage.stage_broken(f"h2d_mismatch:{bad}")
    return used


def _storage_key(t: torch.Tensor):
    st = t.untyped_storage()
    return (str(t.device), st.data_ptr(), st.nbytes())


def _snapshot_mode(devs: list) -> str:
    st = pinned_stage.STAGE
    if not devs or st.d2h is False or st.broken or pinned_stage.stager() is None:
        return MODE
    return pinned_stage.MODE_STAGED if st.d2h else "canary"


def stage_canary_bytes() -> int:
    try:
        return max(1, int(float(os.environ.get(ENV_STAGE_CANARY_MB, "")) * 1024 ** 2))
    except ValueError:
        return STAGE_CANARY_BYTES


def _canary_sample(pairs: list, limit: int) -> list:
    out, left = [], int(limit)
    for a, b in pairs:
        if left <= 0:
            break
        n = int(a.numel())
        if n <= left:
            out.append((a, b))
            left -= n
        else:
            out.append((a[:left], b[:left]))
            left = 0
    return out


def _stage_canary(pairs: list, nbytes: int, pageable_secs: float, devs: list) -> None:
    sg_ = pinned_stage.stager()
    if sg_ is None or not pairs:
        return
    pairs = _canary_sample(pairs, stage_canary_bytes())
    sample_bytes = sum(int(src.numel()) * int(src.element_size()) for src, _h in pairs)
    bad = 0
    t0 = time.time()
    try:
        _synchronize(devs)
        for src, host in pairs:
            tmp = torch.empty_like(host)
            sg_.d2h(src, tmp)
            if not torch.equal(tmp, host):
                bad += 1
            del tmp
        _synchronize(devs)
    except Exception as exc:
        pinned_stage.stage_broken(f"d2h_canary:{type(exc).__name__}: {exc}")
        return
    staged_secs = time.time() - t0
    st = pinned_stage.STAGE
    st.pageable_gbps = round(nbytes / pageable_secs / 1e9, 2) if pageable_secs > 0 else None
    st.staged_gbps = round(sample_bytes / staged_secs / 1e9, 2) if staged_secs > 0 else None
    fast = (staged_secs * pinned_stage.staged_min_speedup() * nbytes
            <= pageable_secs * sample_bytes)
    if bad:
        pinned_stage.stage_broken(f"d2h_mismatch:{bad}")
    else:
        pinned_stage.STAGE.d2h = bool(fast)
        if not fast:
            pinned_stage.STAGE.reason = "d2h_not_faster"


@torch.no_grad()
def _copy(tensors: dict, consumer: str = "", share_with=None) -> dict:
    global _WINDOW
    _, groups = plan(tensors)
    window = None
    if share_with is not None and not share_enabled():
        share_with = None
    if share_with is not None:
        module, step = share_with
        with _SHARE_LOCK:
            if _WINDOW is not None and not _WINDOW.matches(module, step, _GENERATION):
                _, _WINDOW = _WINDOW, None
            else:
                pass
            if _WINDOW is None:
                _WINDOW = _Window(module, step, _GENERATION)
            window = _WINDOW
    hosts: list = [None] * len(groups)
    reused_from: dict = {}
    if window is not None:
        for gi, (lo, n, members) in enumerate(groups):
            t_first = members[0][1]
            if not t_first.numel():
                continue
            ent = window.entries.get(_storage_key(t_first))
            if ent is None:
                continue
            ref, e_lo, e_n, e_copy = ent
            ust = ref()
            if ust is None or e_lo != lo or e_n != n:
                continue
            hosts[gi] = ust
            reused_from.setdefault(id(e_copy), [e_copy, 0])[1] += n
    if window is None:
        todo = list(range(len(groups)))
    else:
        todo = []
        for gi, (lo, n, members) in enumerate(groups):
            if hosts[gi] is not None:
                continue
            if n:
                todo.append(gi)
            else:
                hosts[gi] = torch.empty(0, dtype=torch.uint8).untyped_storage()
    secs = 0.0
    copied_bytes = 0
    if todo:
        devs = _cuda_devices({i: groups[gi][2][0][1] for i, gi in enumerate(todo)})
        mode = _snapshot_mode(devs)
        canary = []
        t0 = time.time()
        _synchronize(devs)
        for gi in todo:
            lo, n, members = groups[gi]
            t_first = members[0][1]
            host = torch.empty(n, dtype=torch.uint8)
            if n:
                src = torch.empty(0, dtype=torch.uint8, device=t_first.device).set_(
                    t_first.untyped_storage(), lo, (n,))
                if mode == pinned_stage.MODE_STAGED and src.device.type == "cuda":
                    if pinned_stage.device_to_host(src, host) != "staged":
                        mode = MODE
                else:
                    host.copy_(src, non_blocking=False)
                if mode == "canary" and src.device.type == "cuda":
                    canary.append((src, host))
                del src
            hosts[gi] = host.untyped_storage()
            copied_bytes += n
            del host
        _synchronize(devs)
        secs = time.time() - t0
        if mode == "canary":
            mode = MODE
            _stage_canary(canary, copied_bytes, secs, devs)
        canary = None
        with _STATS_LOCK:
            STATS["snapshots"] += 1
            STATS["bytes"] += copied_bytes
            STATS["secs"] += secs
            STATS["max_secs"] = max(STATS["max_secs"], secs)
        c = _Copy(copied_bytes, len(todo), len(devs), secs, consumer,
                  share_with[1] if share_with is not None else None, mode)
        if window is None:
            pass
        else:
            with _SHARE_LOCK:
                window.copies.append(c)
                for gi in todo:
                    lo, n, members = groups[gi]
                    if members[0][1].numel():
                        window.entries[_storage_key(members[0][1])] = (
                            weakref.ref(hosts[gi]), lo, n, c)
    if reused_from:
        shared_total = 0
        with _SHARE_LOCK:
            for e_copy, nb in reused_from.values():
                if consumer not in e_copy.consumers:
                    e_copy.consumers.append(consumer)
                e_copy.shared += 1
                e_copy.shared_bytes += nb
                shared_total += nb
        with _STATS_LOCK:
            STATS["shared"] += 1
            STATS["shared_bytes"] += shared_total
    out: dict = {}
    for gi, (lo, n, members) in enumerate(groups):
        ust = hosts[gi]
        for k, t, rel in members:
            v = torch.empty(0, dtype=t.dtype)
            v.set_(ust, rel // t.element_size(), t.shape, t.stride())
            out[k] = v
    hosts = None
    return {k: out[k] for k in tensors}


def stats() -> dict:
    with _STATS_LOCK:
        d = dict(STATS)
    d["secs"] = round(d["secs"], 3)
    d["max_secs"] = round(d["max_secs"], 3)
    d["mode"] = pinned_stage.MODE_STAGED if pinned_stage.STAGE.d2h else MODE
    d["stage"] = pinned_stage.STAGE.model_dump()
    d["share"] = share_enabled()
    return d


def shared_bytes_available(tensors: dict, share_with) -> int:
    if share_with is None or not share_enabled():
        return 0
    module, step = share_with
    try:
        _, groups = plan(tensors)
    except Exception:
        return 0
    with _SHARE_LOCK:
        w = _WINDOW
        if w is None or not w.matches(module, step, _GENERATION):
            return 0
        hit = 0
        for lo, n, members in groups:
            t_first = members[0][1]
            if not n or not t_first.numel():
                continue
            ent = w.entries.get(_storage_key(t_first))
            if ent is None:
                continue
            ref, e_lo, e_n, _ = ent
            if ref() is not None and e_lo == lo and e_n == n:
                hit += n
        return int(hit)


def writer_admit(layout: tuple, avail_fn=None, *, shared: int = 0) -> tuple:
    total = int(sum(layout))
    need = max(0, total - int(shared or 0))
    avail = (avail_fn or available_ram_bytes)()
    info = {"need": need, "layout_bytes": total, "shared_bytes": int(shared or 0),
            "avail": (int(avail) if avail is not None else None)}
    try:
        lim, cur = _cgroup_limit()
        info.update(cgroup_limit=lim, cgroup_usage=cur)
    except Exception:
        pass
    if avail is None or need == 0:
        return True, info
    margin = max(WRITER_RAM_MARGIN_FLOOR, WRITER_RAM_MARGIN_SNAPS * need)
    info["margin"] = int(margin)
    return (avail - margin) >= need, info


def snapshot(tensors: dict, *, consumer: str = "", share_with=None) -> dict:
    if enabled():
        try:
            if all(isinstance(t, torch.Tensor) and t.layout == torch.strided
                   and t.device.type != "meta" for t in tensors.values()):
                return _copy(tensors, consumer=consumer, share_with=share_with)
        except ValueError:
            pass
    return {k: v.cpu().clone() for k, v in tensors.items()}
