import os
import threading
import time
from typing import Optional

import torch
from pydantic import BaseModel


ENV_STAGED = "SN56_STAGED_SNAPSHOT"
ENV_STAGED_SPEEDUP = "SN56_STAGED_MIN_SPEEDUP"
MODE_STAGED = "staged_pinned"
STAGE_CHUNK_BYTES = 64 * 1024 ** 2
STAGE_MIN_SPEEDUP = 1.25


def staged_enabled() -> bool:
    return os.environ.get(ENV_STAGED, "1").strip().lower() not in ("0", "false", "no", "off")


def staged_min_speedup() -> float:
    try:
        return float(os.environ.get(ENV_STAGED_SPEEDUP, str(STAGE_MIN_SPEEDUP)))
    except ValueError:
        return STAGE_MIN_SPEEDUP


class StageVerdict(BaseModel):
    d2h: Optional[bool] = None
    h2d: Optional[bool] = None
    broken: bool = False
    reason: str = ""
    pageable_gbps: Optional[float] = None
    staged_gbps: Optional[float] = None


STAGE = StageVerdict()


class PinnedStager:

    def __init__(self, chunk: int = STAGE_CHUNK_BYTES, nbuf: int = 2, pinned: bool = True):
        self.chunk = max(1, int(chunk))
        self.nbuf = max(2, int(nbuf))
        self.pinned = bool(pinned)
        self._buf: Optional[list] = None
        self._lock = threading.Lock()
        self.bytes_d2h = 0
        self.bytes_h2d = 0
        self.secs_d2h = 0.0
        self.secs_h2d = 0.0

    def _ensure(self) -> list:
        if self._buf is None:
            self._buf = [torch.empty(self.chunk, dtype=torch.uint8, pin_memory=self.pinned)
                         for _ in range(self.nbuf)]
        return self._buf

    @staticmethod
    def _flat_bytes(t: torch.Tensor) -> torch.Tensor:
        if not t.is_contiguous():
            raise ValueError("stager needs contiguous tensors")
        return t.reshape(-1).view(torch.uint8) if t.dtype != torch.uint8 else t.reshape(-1)

    def _event(self, device):
        if not self.pinned or device.type != "cuda":
            return None
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(device))
        return ev

    @staticmethod
    def _wait(ev) -> None:
        if ev is not None:
            ev.synchronize()

    @torch.no_grad()
    def d2h(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        s, d = self._flat_bytes(src), self._flat_bytes(dst)
        n = int(s.numel())
        if n != int(d.numel()):
            raise ValueError(f"stager size mismatch {n} != {d.numel()}")
        if n == 0:
            return
        t0 = time.perf_counter()
        with self._lock:
            bufs = self._ensure()
            evs = [None] * self.nbuf
            pending = []
            off = i = 0
            while off < n:
                m = min(self.chunk, n - off)
                k = i % self.nbuf
                if len(pending) >= self.nbuf:
                    kk, o, mm = pending.pop(0)
                    self._wait(evs[kk])
                    d[o:o + mm].copy_(bufs[kk][:mm])
                bufs[k][:m].copy_(s[off:off + m], non_blocking=self.pinned)
                evs[k] = self._event(src.device)
                pending.append((k, off, m))
                if len(pending) >= 2:
                    kk, o, mm = pending.pop(0)
                    self._wait(evs[kk])
                    d[o:o + mm].copy_(bufs[kk][:mm])
                off += m
                i += 1
            for kk, o, mm in pending:
                self._wait(evs[kk])
                d[o:o + mm].copy_(bufs[kk][:mm])
            self.bytes_d2h += n
            self.secs_d2h += time.perf_counter() - t0

    @torch.no_grad()
    def h2d(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        s, d = self._flat_bytes(src), self._flat_bytes(dst)
        n = int(s.numel())
        if n != int(d.numel()):
            raise ValueError(f"stager size mismatch {n} != {d.numel()}")
        if n == 0:
            return
        t0 = time.perf_counter()
        with self._lock:
            bufs = self._ensure()
            evs = [None] * self.nbuf
            off = i = 0
            while off < n:
                m = min(self.chunk, n - off)
                k = i % self.nbuf
                self._wait(evs[k])
                bufs[k][:m].copy_(s[off:off + m])
                d[off:off + m].copy_(bufs[k][:m], non_blocking=self.pinned)
                evs[k] = self._event(dst.device)
                off += m
                i += 1
            for ev in evs:
                self._wait(ev)
            self.bytes_h2d += n
            self.secs_h2d += time.perf_counter() - t0


_STAGER: Optional[PinnedStager] = None
_STAGER_LOCK = threading.Lock()


def stager() -> Optional[PinnedStager]:
    global _STAGER
    if not staged_enabled() or STAGE.broken or not torch.cuda.is_available():
        return None
    with _STAGER_LOCK:
        if _STAGER is None:
            try:
                st = PinnedStager()
                st._ensure()
                _STAGER = st
            except Exception as exc:
                STAGE.broken, STAGE.reason = True, f"pinned_alloc:{type(exc).__name__}"[:200]
                return None
        return _STAGER


def stage_broken(reason: str) -> None:
    if not STAGE.broken:
        STAGE.broken, STAGE.reason = True, str(reason)[:300]
        STAGE.d2h = False
        STAGE.h2d = False


def stager_for(device: torch.device, verdict: Optional[bool]) -> Optional[PinnedStager]:
    if device.type != "cuda" or verdict is False:
        return None
    return stager()


def device_to_host(src: torch.Tensor, dst: torch.Tensor) -> str:
    sg_ = stager_for(src.device, STAGE.d2h)
    if sg_ is not None and src.is_contiguous() and dst.is_contiguous() \
            and src.dtype == dst.dtype and src.numel() == dst.numel():
        try:
            sg_.d2h(src, dst)
            return "staged"
        except Exception as exc:
            stage_broken(f"d2h:{type(exc).__name__}: {exc}")
    dst.copy_(src, non_blocking=False)
    return "pageable"


def host_to_device(src: torch.Tensor, dst: torch.Tensor) -> str:
    sg_ = stager_for(dst.device, STAGE.h2d)
    if sg_ is not None and src.is_contiguous() and dst.is_contiguous() \
            and src.dtype == dst.dtype and src.numel() == dst.numel():
        try:
            sg_.h2d(src, dst)
            return "staged"
        except Exception as exc:
            stage_broken(f"h2d:{type(exc).__name__}: {exc}")
    dst.copy_(src.to(dst.device))
    return "pageable"
