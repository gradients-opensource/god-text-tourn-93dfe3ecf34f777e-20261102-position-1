import argparse
import hashlib
import json
import math
import os
import release_defaults
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

ENV_SWITCH = "SN56_RESTORE_SCALING"
_FALSY = ("0", "false", "no", "off")

MANIFEST = "restore_scaling.json"
SAMPLE_ELEMS = 2_000_000
CHUNK_ELEMS = 4_000_000
JUDGE_MIN = 30.0
SMALL_MIN_EXPECTED = 10.0
SPREAD_FRACTION = 0.25
GOD_S_RANGE = (0.5, 1.5)
GUARD_MAX_ROWS = 256
ALIAS_LLR_MIN = 10.0


def enabled() -> bool:
    return release_defaults.raw(ENV_SWITCH).strip().lower() not in _FALSY


def is_candidate(name: str) -> bool:
    low = name.lower()
    return "weight" in name and "norm" not in low and "embed" not in low


@dataclass(frozen=True)
class Grid:
    name: str
    st_dtype: str
    p: int
    exp_bits: int

    @property
    def M(self) -> int:
        return 1 << self.p

    @property
    def exp_max(self) -> int:
        return (1 << self.exp_bits) - 1


FP16 = Grid("fp16", "F16", 10, 5)
BF16 = Grid("bf16", "BF16", 7, 8)
GRIDS: Dict[str, Grid] = {g.st_dtype: g for g in (FP16, BF16)}


def mantissa_hist(bits: np.ndarray, grid: Grid) -> np.ndarray:
    b = bits.astype(np.int32, copy=False)
    e = (b >> grid.p) & grid.exp_max
    ok = (e > 1) & (e < grid.exp_max)
    return np.bincount(b[ok] & (grid.M - 1), minlength=grid.M).astype(np.int64)


def bits_to_f32(bits: np.ndarray, grid: Grid) -> np.ndarray:
    if grid is FP16:
        return bits.view(np.float16).astype(np.float32)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bits(x: np.ndarray, grid: Grid) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if grid is FP16:
        return x.astype(np.float16).view(np.uint16)
    u = x.view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    if nan.any():
        rounded[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return rounded


def scale_bits(bits: np.ndarray, grid: Grid, factor: float, divide: bool) -> np.ndarray:
    x = bits_to_f32(bits, grid)
    f = np.float32(factor)
    y = x / f if divide else x * f
    return f32_to_bits(y, grid)


def _ordered(bits: np.ndarray) -> np.ndarray:
    b = bits.astype(np.int32)
    return np.where(b & 0x8000, -(b & 0x7FFF), b & 0x7FFF)


def _from_ordered(o: np.ndarray) -> np.ndarray:
    return np.where(o < 0, 0x8000 | (-o), o).astype(np.uint16)


@lru_cache(maxsize=8)
def preimage_table(grid_name: str, s: float) -> Tuple[np.ndarray, np.ndarray]:
    grid = {"fp16": FP16, "bf16": BF16}[grid_name]
    y = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    o = _ordered(scale_bits(y, grid, s, divide=True))
    cands = np.stack([o - 1, o, o + 1])
    valid = np.stack([scale_bits(_from_ordered(c), grid, s, divide=False) == y for c in cands])
    n_valid = valid.sum(0)
    rank = np.cumsum(valid, axis=0) - 1
    table = np.empty((3, y.size), dtype=np.uint16)
    cols = np.arange(y.size)
    for k in range(3):
        chosen = np.argmax(valid & (rank == k), axis=0)
        out = np.where(n_valid > k, cands[chosen, cols], o)
        table[k] = _from_ordered(out)
    finite = np.isfinite(bits_to_f32(y, grid))
    table[:, ~finite] = y[~finite]
    n_valid = n_valid.astype(np.uint8)
    n_valid.setflags(write=False)
    table.setflags(write=False)
    return n_valid, table


def restore_bits(bits: np.ndarray, grid: Grid, s: float, start_index: int = 0) -> Tuple[np.ndarray, int]:
    n_valid_t, table = preimage_table(grid.name, float(s))
    bits = np.asarray(bits, dtype=np.uint16)
    nv = np.take(n_valid_t, bits)
    out = np.take(table[0], bits)
    amb = np.flatnonzero(nv > 1)
    if amb.size:
        idx = amb.astype(np.uint64) + np.uint64(start_index)
        h = ((idx * np.uint64(0x9E3779B1)) >> np.uint64(16)) & np.uint64(0xFFFF)
        pick = (h % np.take(nv, amb).astype(np.uint64)).astype(np.intp)
        out[amb] = table[pick, np.take(bits, amb)]
    return out, int(amb.size)


_ITEMSIZE = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
             "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    shard: str
    offset: int
    nbytes: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


def read_header(path: str) -> Tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def shard_files(model_dir: str) -> List[str]:
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(model_dir, s) for s in files]
    return sorted(os.path.join(model_dir, n) for n in os.listdir(model_dir)
                  if n.endswith(".safetensors"))


def list_tensors(model_dir: str) -> List[TensorEntry]:
    out: List[TensorEntry] = []
    for path in shard_files(model_dir):
        hlen, header = read_header(path)
        base = 8 + hlen
        for name, t in header.items():
            if name == "__metadata__":
                continue
            a, b = t["data_offsets"]
            out.append(TensorEntry(name, t["dtype"], tuple(int(d) for d in t["shape"]),
                                   os.path.realpath(path), base + int(a), int(b) - int(a)))
    return out


def read_bits(entry: TensorEntry, start: int = 0, count: Optional[int] = None) -> np.ndarray:
    n = entry.numel
    count = n - start if count is None else max(0, min(count, n - start))
    with open(entry.shard, "rb") as f:
        f.seek(entry.offset + 2 * start)
        return np.fromfile(f, dtype="<u2", count=count)


@dataclass
class Sweep:
    M: int
    lo: np.ndarray
    hi: np.ndarray
    run_bin: np.ndarray
    run_i0: np.ndarray
    run_i1: np.ndarray

    @property
    def n_pieces(self) -> int:
        return int(self.lo.size)


def _f32_ulp(M: int) -> float:
    return M / float(1 << 23)


def images(s_fold: float, M: int) -> np.ndarray:
    a = np.arange(M, 2 * M, dtype=np.float64)
    v = a * s_fold
    q = np.where(v >= 2 * M, v / 2.0, v)
    u = _f32_ulp(M)
    q32 = np.rint(q / u) * u
    r = np.rint(q32)
    return np.where(r >= 2 * M, 0, r - M).astype(np.int64)


def unreachable(s_fold: float, M: int) -> np.ndarray:
    reach = np.zeros(M, dtype=bool)
    reach[images(s_fold, M)] = True
    return ~reach


def multiplicity(s_fold: float, M: int) -> np.ndarray:
    return np.bincount(images(s_fold, M), minlength=M)


def fold(s: float) -> Tuple[float, int]:
    k = math.floor(math.log2(s))
    return s / (2.0 ** k), k


@lru_cache(maxsize=4)
def build_sweep(M: int) -> Sweep:
    a = np.arange(M, 2 * M, dtype=np.int64)
    hu = 0.5 * _f32_ulp(M)
    s_parts, from_parts = [], []
    for ai in a:
        n1 = np.arange(ai, 2 * M, dtype=np.int64)
        n2 = np.arange(M, ai, dtype=np.int64)
        s_parts.append((n1 + 0.5 + np.where(n1 % 2 == 0, hu, -hu)) / ai)
        s_parts.append(2.0 * (n2 + 0.5 + np.where(n2 % 2 == 0, hu, -hu)) / ai)
        from_parts.append(n1 - M)
        from_parts.append(n2 - M)
    s = np.concatenate(s_parts)
    fb = np.concatenate(from_parts)
    tb = (fb + 1) % M
    order = np.argsort(s, kind="stable")
    s, fb, tb = s[order], fb[order], tb[order]
    N = s.size
    lo = np.concatenate(([1.0], s))
    hi = np.concatenate((s, [2.0]))
    ev = np.arange(N, dtype=np.int64)
    bins = np.concatenate((fb, tb))
    evs = np.concatenate((ev, ev))
    delta = np.concatenate((-np.ones(N, np.int64), np.ones(N, np.int64)))
    o = np.lexsort((evs, bins))
    bins, evs, delta = bins[o], evs[o], delta[o]
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    csum = np.cumsum(delta)
    group_base = np.repeat(csum[starts] - delta[starts], np.diff(np.r_[starts, bins.size]))
    cnt_after = 1 + csum - group_base
    zero_rows = np.flatnonzero(cnt_after == 0)
    nxt = zero_rows + 1
    has_next = (nxt < bins.size)
    has_next[has_next] = bins[nxt[has_next]] == bins[zero_rows[has_next]]
    run_bin = bins[zero_rows]
    run_i0 = evs[zero_rows] + 1
    run_i1 = np.where(has_next, evs[np.minimum(nxt, bins.size - 1)] + 1, N + 1)
    return Sweep(M, lo, hi, run_bin.astype(np.int64), run_i0.astype(np.int64),
                 run_i1.astype(np.int64))


def smooth(h: np.ndarray, half_window: int) -> np.ndarray:
    c = np.concatenate(([0.0], np.cumsum(h.astype(np.float64))))
    i = np.arange(h.size)
    a = np.maximum(0, i - half_window)
    b = np.minimum(h.size, i + half_window + 1)
    return (c[b] - c[a]) / (b - a)


def judged_and_empty(h: np.ndarray, grid: Grid) -> Tuple[np.ndarray, np.ndarray]:
    exp_ = smooth(h, max(2, grid.M // 64))
    judged = exp_ >= JUDGE_MIN
    return judged, judged & (h == 0)


def match_pieces(sw: Sweep, judged: np.ndarray, empty: np.ndarray) -> np.ndarray:
    w = np.zeros(sw.M, dtype=np.float64)
    w[judged & empty] = -1.0
    w[judged & ~empty] = 1.0
    wr = w[sw.run_bin]
    keep = wr != 0
    P = sw.n_pieces
    d = (np.bincount(sw.run_i0[keep], weights=wr[keep], minlength=P + 1)
         - np.bincount(sw.run_i1[keep], weights=wr[keep], minlength=P + 1))
    return np.rint(int((judged & empty).sum()) + np.cumsum(d)[:P]).astype(np.int64)


@dataclass
class TensorStat:
    name: str
    dtype: str
    shape: List[int]
    kind: str
    n_sampled: int
    rms: float
    occupied: int
    spread: bool
    n_judged: int = 0
    n_empty: int = 0
    mismatch_at_s: Optional[int] = None
    small_expected: Optional[float] = None
    small_observed: Optional[int] = None
    flagged: bool = False
    rule: str = ""


@dataclass
class Detection:
    detected: bool
    reason: str
    grid: Optional[str] = None
    s: Optional[float] = None
    s_fold_lo: Optional[float] = None
    s_fold_hi: Optional[float] = None
    piece_lo: Optional[float] = None
    piece_hi: Optional[float] = None
    octave_k: Optional[int] = None
    octave_resolution: Optional[str] = None
    n_alias_regions: int = 0
    alias_llr: Optional[float] = None
    n_candidates: int = 0
    n_half: int = 0
    n_spread: int = 0
    n_gapped: int = 0
    n_matched: int = 0
    n_flagged: int = 0
    flagged: List[str] = field(default_factory=list)
    unmatched_gapped: List[str] = field(default_factory=list)
    tensors: List[TensorStat] = field(default_factory=list)
    secs: float = 0.0

    def names_hash(self) -> str:
        return hashlib.sha256("\n".join(sorted(self.flagged)).encode()).hexdigest()[:16]


def role_of(name: str) -> str:
    return re.sub(r"\.\d+\.", ".*.", name)


def _rms_bits(bits: np.ndarray, grid: Grid) -> float:
    x = bits_to_f32(bits, grid).astype(np.float64)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def detect(model_dir: str, sample_elems: int = SAMPLE_ELEMS) -> Detection:
    t0 = time.time()
    entries = [e for e in list_tensors(model_dir) if is_candidate(e.name)]
    det = Detection(detected=False, reason="", n_candidates=len(entries))
    half = [e for e in entries if e.dtype in GRIDS]
    det.n_half = len(half)
    if not half:
        det.reason = "no_half_grid" if entries else "no_candidates"
        det.secs = round(time.time() - t0, 3)
        return det
    by_grid: Dict[str, int] = {}
    for e in half:
        by_grid[e.dtype] = by_grid.get(e.dtype, 0) + 1
    gtag = max(by_grid, key=lambda k: by_grid[k])
    grid = GRIDS[gtag]
    det.grid = grid.name
    M = grid.M

    stats: Dict[str, TensorStat] = {}
    hists: Dict[str, np.ndarray] = {}
    gapped: List[str] = []
    vectors: List[str] = []
    for e in half:
        if e.dtype != gtag:
            continue
        is_vec = len(e.shape) <= 1
        bits = read_bits(e, 0, None if is_vec else sample_elems)
        h = mantissa_hist(bits, grid)
        occ = int((h > 0).sum())
        st = TensorStat(e.name, e.dtype, list(e.shape), "vector" if is_vec else "matrix",
                        int(bits.size), _rms_bits(bits, grid), occ,
                        spread=(not is_vec) and occ >= SPREAD_FRACTION * M)
        stats[e.name] = st
        hists[e.name] = h
        if is_vec:
            vectors.append(e.name)
            continue
        if not st.spread:
            continue
        det.n_spread += 1
        judged, empty = judged_and_empty(h, grid)
        st.n_judged, st.n_empty = int(judged.sum()), int(empty.sum())
        if st.n_empty > 0:
            gapped.append(e.name)
    det.n_gapped = len(gapped)
    det.tensors = [stats[n] for n in gapped]
    if not gapped:
        det.reason = "no_gapped_matrices"
        det.secs = round(time.time() - t0, 3)
        return det

    sw = build_sweep(M)
    P = sw.n_pieces
    width_ok = (sw.hi - sw.lo) > 0
    votes = np.zeros(P, dtype=np.int64)
    matches: Dict[str, np.ndarray] = {}
    for n in gapped:
        judged, empty = judged_and_empty(hists[n], grid)
        ok = (match_pieces(sw, judged, empty) == 0) & width_ok
        matches[n] = ok
        votes += ok
    best = int(votes.max())
    if best == 0:
        det.reason = "no_consistent_s"
        det.unmatched_gapped = list(gapped)
        det.secs = round(time.time() - t0, 3)
        return det
    top = np.flatnonzero((votes == best) & width_ok)
    agree = [n for n in gapped if matches[n][top].all()] or list(gapped)

    H = np.stack([hists[n].astype(np.float64) for n in agree])
    D = np.stack([smooth(h, max(2, M // 64)) for h in H])
    Hs = H.sum(1)
    cnts = np.stack([multiplicity(0.5 * (sw.lo[i] + sw.hi[i]), M) for i in top])
    uniq, inv = np.unique(cnts, axis=0, return_inverse=True)
    ll_u = np.empty(len(uniq))
    for k, cnt in enumerate(uniq.astype(np.float64)):
        lam = cnt[None, :] * D
        c = Hs / np.maximum(lam.sum(1), 1e-300)
        pos = lam > 0
        ll_u[k] = float(np.sum(H[pos] * np.log(lam[pos])) + np.sum(Hs * np.log(c)) - Hs.sum())
    ll = ll_u[np.asarray(inv).reshape(-1)]
    best_i = int(top[int(np.argmax(ll))])
    region = np.zeros(top.size, dtype=np.int64)
    for j in range(1, top.size):
        gap = np.arange(top[j - 1] + 1, top[j])
        same = gap.size == 0 or not np.any(width_ok[gap])
        region[j] = region[j - 1] + (0 if same else 1)
    r_best = region[int(np.argmax(ll))]
    others = ll[region != r_best]
    det.n_alias_regions = int(region.max() + 1)
    det.alias_llr = None if others.size == 0 else round(float(ll.max() - others.max()), 2)
    if others.size and ll.max() - others.max() < ALIAS_LLR_MIN:
        det.reason = "ambiguous_s"
        det.secs = round(time.time() - t0, 3)
        return det
    in_r = top[region == r_best]
    det.s_fold_lo, det.s_fold_hi = float(sw.lo[in_r[0]]), float(sw.hi[in_r[-1]])
    det.piece_lo, det.piece_hi = float(sw.lo[best_i]), float(sw.hi[best_i])
    s_fold = 0.5 * (det.piece_lo + det.piece_hi)
    U = unreachable(s_fold, M)

    flagged_mats = []
    for n in gapped:
        judged, empty = judged_and_empty(hists[n], grid)
        mm = int(((U & judged) != empty).sum())
        stats[n].mismatch_at_s = mm
        if mm == 0:
            stats[n].flagged, stats[n].rule = True, "exact"
            flagged_mats.append(n)
        else:
            det.unmatched_gapped.append(n)
    det.n_matched = len(flagged_mats)

    ks = [k for k in (-1, 0) if GOD_S_RANGE[0] < s_fold * 2.0 ** k < GOD_S_RANGE[1]] or [-1, 0]
    flagged_set = set(flagged_mats)
    sib: Dict[str, List[float]] = {}
    for n, st in stats.items():
        if st.kind == "matrix" and n not in flagged_set:
            sib.setdefault(role_of(n), []).append(st.rms)
    ratios = [stats[n].rms / float(np.median(sib[role_of(n)]))
              for n in flagged_mats if sib.get(role_of(n)) and np.median(sib[role_of(n)]) > 0]
    if ratios:
        lr = np.log(np.asarray(ratios))
        k = min(ks, key=lambda kk: float(np.median(np.abs(lr - math.log(s_fold * 2.0 ** kk)))))
        det.octave_resolution = "siblings"
    else:
        k = min(ks, key=lambda kk: abs(math.log(s_fold * 2.0 ** kk)))
        det.octave_resolution = "prior"
    det.octave_k = int(k)
    det.s = float(s_fold * 2.0 ** k)

    flagged_vecs = []
    for n in vectors:
        h = hists[n]
        st = stats[n]
        if int((h > 0).sum()) < M // 8:
            continue
        exp_u = float(smooth(h, max(1, M // 16))[U].sum())
        obs_u = int(h[U].sum())
        st.small_expected, st.small_observed = round(exp_u, 2), obs_u
        if exp_u >= SMALL_MIN_EXPECTED and obs_u == 0:
            st.flagged, st.rule = True, "small"
            flagged_vecs.append(n)
            det.tensors.append(st)

    det.flagged = flagged_mats + flagged_vecs
    det.n_flagged = len(det.flagged)
    det.detected = bool(flagged_mats)
    det.reason = "constant_multiple" if det.detected else "no_consistent_s"
    det.secs = round(time.time() - t0, 3)
    return det


_COPY_BLOCK = 64 << 20


def _copy_bytes(fi, fo, n: int) -> None:
    while n > 0:
        buf = fi.read(min(n, _COPY_BLOCK))
        if not buf:
            raise IOError(f"short read: {n} bytes missing")
        fo.write(buf)
        n -= len(buf)


def iter_restored(entry: TensorEntry, grid: Grid, s: float, fi=None):
    for start in range(0, entry.numel, CHUNK_ELEMS):
        if fi is None:
            bits = read_bits(entry, start, CHUNK_ELEMS)
        else:
            n = min(CHUNK_ELEMS, entry.numel - start)
            bits = np.frombuffer(fi.read(2 * n), dtype="<u2")
            if bits.size != n:
                raise IOError(f"short read in {entry.name}")
        new, amb = restore_bits(bits, grid, s, start)
        yield start, bits, new, amb


def write_restored(model_dir: str, out_dir: str, det: Detection) -> dict:
    assert det.detected and det.s
    t0 = time.time()
    grid = {"fp16": FP16, "bf16": BF16}[det.grid]
    model_dir = os.path.realpath(model_dir)
    if os.path.lexists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    flagged = set(det.flagged)
    entries = {e.name: e for e in list_tensors(model_dir)}
    shards = {}
    for n in flagged:
        shards.setdefault(entries[n].shard, []).append(entries[n])
    per = []
    for name in sorted(os.listdir(model_dir)):
        src = os.path.join(model_dir, name)
        dst = os.path.join(out_dir, name)
        real = os.path.realpath(src)
        if real in shards:
            with open(real, "rb") as fi, open(dst, "wb") as fo:
                pos = 0
                for e in sorted(shards[real], key=lambda x: x.offset):
                    _copy_bytes(fi, fo, e.offset - pos)
                    n_changed = n_ambiguous = 0
                    for _, bits, new, amb in iter_restored(e, grid, det.s, fi):
                        n_changed += int(np.count_nonzero(new != bits))
                        n_ambiguous += amb
                        fo.write(new.astype("<u2", copy=False).tobytes())
                    pos = e.offset + 2 * e.numel
                    per.append({"name": e.name, "numel": e.numel, "changed": n_changed,
                                "two_preimages": n_ambiguous, "shard": name})
                shutil.copyfileobj(fi, fo, _COPY_BLOCK)
        else:
            os.symlink(real, dst)
    manifest = {
        "s": det.s, "octave_k": det.octave_k, "octave_resolution": det.octave_resolution,
        "s_fold_interval": [det.s_fold_lo, det.s_fold_hi], "piece": [det.piece_lo, det.piece_hi],
        "grid": det.grid, "method": "analytic_sweep_exact_match+consistent_preimage",
        "source_dir": model_dir, "tensors": sorted(flagged), "names_hash": det.names_hash(),
        "n_gapped": det.n_gapped, "n_matched": det.n_matched,
        "unmatched_gapped": det.unmatched_gapped,
        "per_tensor": [asdict(t) for t in det.tensors], "written": per,
        "secs_detect": det.secs, "secs_write": round(time.time() - t0, 3),
    }
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


@dataclass
class GuardResult:
    kept: bool
    reason: str
    n_rows: int = 0
    dev_before: Optional[float] = None
    dev_after: Optional[float] = None
    diff: Optional[float] = None
    se: Optional[float] = None
    secs: float = 0.0
    secs_apply: Optional[float] = None


def keep_decision(after_rows: Sequence[float], before_rows: Sequence[float]) -> GuardResult:
    a = np.asarray(after_rows, dtype=np.float64)
    b = np.asarray(before_rows, dtype=np.float64)
    n = int(a.size)
    if n < 2:
        return GuardResult(False, "too_few_rows", n_rows=n)
    d = a - b
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    kept = mean < -se
    return GuardResult(kept, "improved" if kept else "not_improved", n_rows=n,
                       dev_before=float(b.mean()), dev_after=float(a.mean()), diff=mean, se=se)


def load_dev_rows(dev_path: str, max_rows: int, max_length: int) -> List[dict]:
    with open(dev_path) as f:
        rows = json.load(f)
    out = []
    for r in rows:
        ids = list(r["input_ids"])[:max_length]
        labels = list(r["labels"])[:max_length]
        if sum(1 for x in labels[1:] if x != -100) == 0:
            continue
        out.append({"input_ids": ids, "labels": labels})
        if len(out) >= max_rows:
            break
    return out


@torch.inference_mode()
def score_rows(model, rows: List[dict], device) -> List[float]:
    losses = []
    for r in rows:
        ids = torch.tensor([r["input_ids"]], device=device)
        lab = torch.tensor([r["labels"]], device=device)
        logits = model(input_ids=ids).logits[0, :-1]
        tgt = lab[0, 1:]
        sel = tgt != -100
        lp = torch.nn.functional.cross_entropy(logits[sel].float(), tgt[sel], reduction="mean")
        losses.append(float(lp))
    return losses


def resolve_param(params: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name in params:
        return params[name]
    hits = [k for k in params if k.endswith("." + name) or name.endswith("." + k)]
    if len(hits) != 1:
        raise KeyError(f"cannot map checkpoint tensor {name!r} onto the model ({len(hits)} hits)")
    return params[hits[0]]


def guard_spec(model_dir: str, det: Detection) -> dict:
    return {"source_dir": os.path.realpath(model_dir), "tensors": list(det.flagged), "s": det.s,
            "grid": det.grid}


def _as_torch(bits: np.ndarray, grid: Grid) -> torch.Tensor:
    return torch.from_numpy(bits.view(np.int16)).view(torch.float16 if grid is FP16 else torch.bfloat16)


def restore_in_place(model, spec: dict) -> int:
    entries = {e.name: e for e in list_tensors(spec["source_dir"])}
    grid = {"fp16": FP16, "bf16": BF16}[spec["grid"]]
    params = dict(model.named_parameters())
    n = 0
    for name in spec["tensors"]:
        e = entries[name]
        p = resolve_param(params, name)
        if tuple(p.shape) != tuple(e.shape):
            raise ValueError(f"shape mismatch for {name}: {tuple(p.shape)} vs {tuple(e.shape)}")
        if p.data.is_contiguous():
            flat = p.data.view(-1)
            for start, _, new, _ in iter_restored(e, grid, float(spec["s"])):
                x = _as_torch(new, grid)
                flat[start:start + x.numel()].copy_(x.to(dtype=p.dtype, device=p.device))
        else:
            full = np.empty(e.numel, dtype=np.uint16)
            for start, _, new, _ in iter_restored(e, grid, float(spec["s"])):
                full[start:start + new.size] = new
            p.data.copy_(_as_torch(full, grid).reshape(e.shape).to(dtype=p.dtype, device=p.device))
        n += 1
    return n


def load_for_guard(model_dir: str, dtype) -> torch.nn.Module:
    last = None
    for attn in ("sdpa", "eager"):
        try:
            return AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn)
        except (ValueError, ImportError) as exc:
            last = exc
    raise last


def run_guard(spec: dict, dev_path: str, max_length: int,
              max_rows: int = GUARD_MAX_ROWS, model=None) -> GuardResult:
    t0 = time.time()
    rows = load_dev_rows(dev_path, max_rows, max_length)
    if len(rows) < 2:
        return GuardResult(False, "too_few_rows", n_rows=len(rows), secs=round(time.time() - t0, 2))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = load_for_guard(spec["source_dir"], dtype).to(device)
    model.eval()
    before = score_rows(model, rows, device)
    t_apply = time.time()
    restore_in_place(model, spec)
    secs_apply = round(time.time() - t_apply, 2)
    after = score_rows(model, rows, device)
    res = keep_decision(after, before)
    res.secs_apply = secs_apply
    res.secs = round(time.time() - t0, 2)
    return res


def restored_dir_for(output_dir: str) -> str:
    return output_dir.rstrip("/") + "_restored_base"


def apply_for_training(train_info: dict, *, dev_path: str, emit: Optional[Callable] = None,
                       log: Callable = print,
                       guard_timeout: float = 1200.0) -> dict:
    req = train_info["train_request"]
    model_path = req["model_path"]
    try:
        det = detect(model_path)
    except Exception as exc:
        if emit is not None:
            emit("restore_scaling", detected=False, reason=f"detect_error: {type(exc).__name__}: {exc}"[:500])
        return train_info
    if not det.detected:
        if emit is not None:
            emit("restore_scaling", detected=False, reason=det.reason, grid=det.grid,
                 n_candidates=det.n_candidates, n_half=det.n_half, n_spread=det.n_spread,
                 n_gapped=det.n_gapped, secs=det.secs)
        return train_info
    t0 = time.time()
    out_dir = restored_dir_for(req["output_dir"])
    guard = GuardResult(False, "not_run")
    secs_write = 0.0
    work = tempfile.mkdtemp(prefix="restore_guard_")
    try:
        spec_path = os.path.join(work, "spec.json")
        decision = os.path.join(work, "restore_guard.json")
        with open(spec_path, "w") as f:
            json.dump(guard_spec(model_path, det), f)
        cmd = [sys.executable, os.path.abspath(__file__), "guard", "--spec", spec_path,
               "--dev", dev_path, "--max-length", str(int(req.get("max_length") or 2048)),
               "--out", decision]
        proc = subprocess.run(cmd, timeout=guard_timeout, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.isfile(decision):
            with open(decision) as f:
                guard = GuardResult(**json.load(f))
        else:
            guard = GuardResult(False, f"guard_failed rc={proc.returncode}: {proc.stderr[-400:]}")
    except Exception as exc:
        guard = GuardResult(False, f"guard_error: {type(exc).__name__}: {exc}"[:500])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if guard.kept:
        try:
            secs_write = write_restored(model_path, out_dir, det)["secs_write"]
        except Exception as exc:
            guard.kept = False
            guard.reason = f"write_error: {type(exc).__name__}: {exc}"[:500]
    if guard.kept:
        req["model_path"] = out_dir
        if "model_path" in train_info:
            train_info["model_path"] = out_dir
    else:
        shutil.rmtree(out_dir, ignore_errors=True)
    if emit is not None:
        emit("restore_scaling", detected=True, kept=guard.kept, guard_reason=guard.reason,
             s=det.s, s_fold_interval=[det.s_fold_lo, det.s_fold_hi],
             piece=[det.piece_lo, det.piece_hi], octave_k=det.octave_k,
             octave_resolution=det.octave_resolution, grid=det.grid,
             n_candidates=det.n_candidates, n_gapped=det.n_gapped, n_matched=det.n_matched,
             n_tensors=det.n_flagged, names_hash=det.names_hash(), names=det.flagged,
             unmatched_gapped=det.unmatched_gapped,
             dev_before=guard.dev_before, dev_after=guard.dev_after, diff=guard.diff, se=guard.se,
             n_rows=guard.n_rows, model_path=req["model_path"], secs=round(det.secs + time.time() - t0, 2),
             secs_detect=det.secs, secs_write=secs_write if guard.kept else 0.0, secs_guard=guard.secs,
             secs_guard_apply=guard.secs_apply)
    fmt = (lambda v: "?" if v is None else f"{v:.4f}")
    log(f"[sn56][restauro] escala s={det.s:.6f} em {det.n_flagged} tensores; dev "
        f"{fmt(guard.dev_before)} -> {fmt(guard.dev_after)} (ep {fmt(guard.se)}) - "
        f"{'mantido' if guard.kept else 'revertido'} ({guard.reason})")
    return train_info


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("guard")
    g.add_argument("--spec", required=True)
    g.add_argument("--dev", required=True)
    g.add_argument("--max-length", type=int, default=2048)
    g.add_argument("--out", required=True)
    d = sub.add_parser("detect")
    d.add_argument("model_dir")
    d.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "guard":
        with open(a.spec) as f:
            spec = json.load(f)
        res = run_guard(spec, a.dev, a.max_length)
        with open(a.out, "w") as f:
            json.dump(asdict(res), f)
        return 0
    det = detect(a.model_dir)
    out = asdict(det)
    out["names_hash"] = det.names_hash() if det.detected else None
    if a.write and det.detected:
        out["manifest"] = write_restored(a.model_dir, a.write, det)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
