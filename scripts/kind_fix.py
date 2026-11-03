from __future__ import annotations

import base64
import math
import os
import time
import zlib
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, Field

_FALSY = ("0", "false", "no", "off")


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in _FALSY


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


DIV_LR_BAND = _env_flag("SN56_KF_DIV_LR_BAND", "1")
DIV_LR_GATE = _env_flag("SN56_KF_DIV_LR_GATE", "1")
ROWS_GATE = _env_flag("SN56_KF_ROWS_GATE", "1")
DIVTEL = _env_flag("SN56_KF_DIVTEL", "0")
BAND_DEX = _env_float("SN56_KF_BAND_DEX", 0.291)
MIN_RATIO = _env_float("SN56_KF_MIN_RATIO", 1.2)
ROWS_UPE_MAX = _env_float("SN56_KF_ROWS_UPE_MAX", 221.0)
P_DLR_BAND = _env_float("SN56_KF_P_DLR_BAND", 0.87)
STAT_BUDGET = int(_env_float("SN56_KF_STAT_BUDGET", float(1 << 23)))
_BLOCK = 256
SKETCH_K = int(_env_float("SN56_KF_SKETCH_K", 2048))
ROWS_CAP = int(_env_float("SN56_KF_ROWS_CAP", 2048))

POOL = _env_flag("SN56_KF_POOL", "0")
POOL_PAIRS = int(_env_float("SN56_KF_POOL_PAIRS", 3))
POOL_SHIP = (os.environ.get("SN56_KF_POOL_SHIP", "seed").strip().lower() or "seed")
DEV2 = _env_flag("SN56_KF_DEV2", "1" if POOL else "0")
POOL_SEEDS = (44, 45, 46, 47, 48, 49)
DEV2_DS: dict = {}
POOL_DIV_SEEDS = (10001, 10201, 10202, 10203, 10204, 10205)


def switches() -> dict:
    return {"div_lr_band": DIV_LR_BAND, "div_lr_gate": DIV_LR_GATE, "rows_gate": ROWS_GATE,
            "divtel": DIVTEL, "band_dex": BAND_DEX, "min_ratio": MIN_RATIO,
            "rows_upe_max": ROWS_UPE_MAX, "p_dlr_band": P_DLR_BAND, "pool": POOL,
            "pool_pairs": POOL_PAIRS, "pool_ship": POOL_SHIP, "dev2": DEV2}


class LrBand(BaseModel):
    x_vertex: Optional[float] = None
    vertex_source: str = ""
    x_hot_probe: Optional[float] = None
    scale: float = 1.0
    lr_selected: Optional[float] = None
    lr_final: Optional[float] = None


def band_from_selection(sel: Optional[dict], lr_final: Optional[float]) -> LrBand:
    sel = sel or {}
    out = LrBand(lr_selected=sel.get("best_lr"), lr_final=lr_final)
    if out.lr_selected and lr_final and out.lr_selected > 0 and lr_final > 0:
        out.scale = float(lr_final) / float(out.lr_selected)
    fit = sel.get("fit") or {}
    xv = fit.get("x_v_combined") if isinstance(fit, dict) else None
    pool = {float(k): float(v) for k, v in (sel.get("pool") or {}).items()
            if v is not None and math.isfinite(float(v))}
    if xv is not None and math.isfinite(float(xv)) and not sel.get("degenerate"):
        out.x_vertex, out.vertex_source = float(xv), "fit_combined"
    elif pool:
        out.x_vertex, out.vertex_source = min(pool, key=pool.get), "pool_argmin"
    probed = [float(k) for k in (sel.get("probed") or [])] + list(pool)
    out.x_hot_probe = max(probed) if probed else None
    return out


class DivLrChoice(BaseModel):
    lr: Optional[float] = None
    offered: bool = False
    reason: str = ""
    rule: str = ""
    family_lr: Optional[float] = None
    ratio_family: Optional[float] = None
    dx_vertex: Optional[float] = None
    capped_at_probe: bool = False


def div_lr_choice(band: Optional[LrBand], family_lr: Optional[float], mult: float = 2.0,
                  band_on: Optional[bool] = None, gate_on: Optional[bool] = None,
                  band_dex: Optional[float] = None,
                  min_ratio: Optional[float] = None) -> DivLrChoice:
    band_on = DIV_LR_BAND if band_on is None else bool(band_on)
    gate_on = DIV_LR_GATE if gate_on is None else bool(gate_on)
    band_dex = BAND_DEX if band_dex is None else float(band_dex)
    min_ratio = MIN_RATIO if min_ratio is None else float(min_ratio)
    out = DivLrChoice(family_lr=family_lr, rule="band" if band_on else "x2")
    if not family_lr or family_lr <= 0:
        out.reason = "no_family_lr"
        return out
    if band_on:
        if band is None or band.x_vertex is None:
            out.reason = "no_band"
            return out
        x = band.x_vertex + band_dex
        if band.x_hot_probe is not None and x > band.x_hot_probe:
            x, out.capped_at_probe = band.x_hot_probe, True
        out.lr = float(10 ** x * band.scale)
    else:
        out.lr = float(family_lr) * float(mult)
    out.ratio_family = out.lr / float(family_lr)
    if band is not None and band.x_vertex is not None and band.scale > 0:
        out.dx_vertex = math.log10(out.lr / band.scale) - band.x_vertex
    if gate_on and out.ratio_family <= min_ratio + 1e-9:
        out.reason = "within_min_ratio_of_family"
        return out
    out.offered, out.reason = True, "ok"
    return out


class DeltaStats(BaseModel):
    n_tensors: int = 0
    numel: int = 0
    sampled: int = 0
    norm_m_a0: Optional[float] = None
    norm_soup_a0: Optional[float] = None
    norm_m_soup: Optional[float] = None
    cos_m_soup: Optional[float] = None
    sketch: Optional[str] = None
    sketch_scale: Optional[float] = None
    sketch_k: int = 0
    sketch_crc: Optional[int] = None
    secs: float = 0.0
    skipped: str = ""


def _coords(name: str, numel: int, k: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(zlib.crc32(name.encode()) & 0x7FFFFFFF)
    return torch.randint(0, max(1, numel), (k,), generator=g)


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1)


def _sample(t: torch.Tensor, stride: int) -> torch.Tensor:
    f = _flat(t)
    if stride <= 1:
        return f
    nb = int(f.numel()) // _BLOCK
    if nb < stride:
        return f[: _BLOCK] if nb else f
    return f[: nb * _BLOCK].view(nb, _BLOCK)[::stride].reshape(-1)


def delta_stats(member: Optional[dict], a0: Optional[dict], soup_sum: Optional[dict],
                soup_n: int, budget: Optional[int] = None,
                sketch_k: Optional[int] = None) -> DeltaStats:
    _nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _delta_stats(member, a0, soup_sum, soup_n, budget, sketch_k)
    finally:
        torch.set_num_threads(_nt)


def _delta_stats(member, a0, soup_sum, soup_n, budget, sketch_k) -> DeltaStats:
    t0 = time.perf_counter()
    out = DeltaStats()
    budget = STAT_BUDGET if budget is None else int(budget)
    sketch_k = SKETCH_K if sketch_k is None else int(sketch_k)
    if not member or not a0:
        out.skipped = "no_state"
        return out
    names = sorted(n for n in member if n in a0 and torch.is_tensor(member[n])
                   and torch.is_tensor(a0[n]) and member[n].numel() == a0[n].numel())
    if not names:
        out.skipped = "no_common_tensors"
        return out
    total = sum(int(member[n].numel()) for n in names)
    stride = max(1, int(math.ceil(total / max(1, budget))))
    have_soup = bool(soup_sum) and int(soup_n) >= 1 and all(n in soup_sum for n in names)
    inv = 1.0 / float(max(1, int(soup_n)))
    mm = ss = ms = 0.0
    sampled = 0
    sk_vals, sk_crc = [], 0
    with torch.no_grad():
        for n in names:
            numel = int(member[n].numel())
            m = _sample(member[n], stride)
            a = _sample(a0[n], stride)
            dm = m.float().cpu() - a.float().cpu()
            mm += float(torch.dot(dm, dm))
            sampled += int(dm.numel())
            if have_soup:
                s = _sample(soup_sum[n], stride)
                ds = s.float().cpu() * inv - a.float().cpu()
                ss += float(torch.dot(ds, ds))
                ms += float(torch.dot(dm, ds))
                del ds
            del dm
            k = max(1, int(round(sketch_k * numel / total)))
            idx = _coords(n, numel, k)
            full_m, full_a = _flat(member[n]), _flat(a0[n])
            sk_vals.append(full_m.index_select(0, idx.to(full_m.device)).float().cpu()
                           - full_a.index_select(0, idx.to(full_a.device)).float().cpu())
            sk_crc = zlib.crc32(f"{n}:{numel}:{k};".encode(), sk_crc)
    f = float(total) / float(max(1, sampled))
    out.n_tensors, out.numel, out.sampled = len(names), total, sampled
    out.norm_m_a0 = math.sqrt(mm * f)
    if have_soup:
        out.norm_soup_a0 = math.sqrt(ss * f)
        out.norm_m_soup = math.sqrt(max(0.0, mm + ss - 2.0 * ms) * f)
        den = math.sqrt(mm * ss)
        out.cos_m_soup = (ms / den) if den > 0 else None
    sk = torch.cat(sk_vals) if sk_vals else torch.zeros(0)
    scale = float(sk.abs().max()) if sk.numel() else 0.0
    if scale > 0:
        out.sketch = base64.b64encode(
            (sk / scale).to(torch.float16).numpy().tobytes()).decode("ascii")
        out.sketch_scale, out.sketch_k, out.sketch_crc = scale, int(sk.numel()), int(sk_crc)
    out.secs = round(time.perf_counter() - t0, 4)
    return out


def decode_fp16(b64: Optional[str], scale: float = 1.0) -> Optional[np.ndarray]:
    if not b64:
        return None
    return np.frombuffer(base64.b64decode(b64), dtype=np.float16).astype(np.float32) * scale


class RowStats(BaseModel):
    n: int = 0
    n_kept: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    crc: Optional[int] = None
    rows: Optional[str] = None
    corr_soup: Optional[float] = None
    corr_soup_resid: Optional[float] = None
    skipped: str = ""


def row_stats(rows: Optional[torch.Tensor], soup_rows: Optional[torch.Tensor] = None,
              cap: Optional[int] = None, with_vector: bool = True) -> RowStats:
    cap = ROWS_CAP if cap is None else int(cap)
    out = RowStats()
    if rows is None or not torch.is_tensor(rows) or rows.numel() == 0:
        out.skipped = "no_rows"
        return out
    v = rows.detach().float().cpu().reshape(-1)
    ok = ~torch.isnan(v)
    out.n = int(v.numel())
    if bool(ok.any()):
        vv = v[ok]
        q = torch.quantile(vv, torch.tensor([0.1, 0.5, 0.9]))
        out.mean, out.std = float(vv.mean()), float(vv.std()) if vv.numel() > 1 else 0.0
        out.p10, out.p50, out.p90 = (float(x) for x in q)
    keep = v[:cap]
    out.n_kept = int(keep.numel())
    if with_vector:
        b = keep.to(torch.float16).numpy().tobytes()
        out.rows = base64.b64encode(b).decode("ascii")
        out.crc = int(zlib.crc32(b))
    if soup_rows is not None and torch.is_tensor(soup_rows) and soup_rows.numel() == v.numel():
        s = soup_rows.detach().float().cpu().reshape(-1)
        m = ok & ~torch.isnan(s)
        if int(m.sum()) > 2:
            a, c = v[m], s[m]
            out.corr_soup = _corr(a, c)
            out.corr_soup_resid = _corr(a - c, c)
    return out


def _corr(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    a = a - a.mean()
    b = b - b.mean()
    den = float(torch.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / den if den > 0 else None


class PoolSlot(BaseModel):
    kind: str
    seed: int
    pair: int


class PoolPlan(BaseModel):
    slots: list = Field(default_factory=list)
    pairs: int = 0
    ship: str = "seed"


def pool_plan(pairs: Optional[int] = None, ship: Optional[str] = None) -> PoolPlan:
    pairs = POOL_PAIRS if pairs is None else int(pairs)
    pairs = max(1, min(pairs, len(POOL_SEEDS), len(POOL_DIV_SEEDS)))
    ship = (POOL_SHIP if ship is None else ship).strip().lower()
    if ship not in ("seed", "div", "mix"):
        ship = "seed"
    slots = []
    for i in range(pairs):
        slots.append(PoolSlot(kind="t_seed", seed=POOL_SEEDS[i], pair=i))
        slots.append(PoolSlot(kind="t_div_lr", seed=POOL_DIV_SEEDS[i], pair=i))
    return PoolPlan(slots=slots, pairs=pairs, ship=ship)


def soup_add(acc: dict, state: dict) -> None:
    with torch.no_grad():
        for n, t in acc.items():
            if n in state:
                t.add_(state[n].to(device=t.device, dtype=torch.float32))


def soup_clone(state: dict) -> dict:
    with torch.no_grad():
        return {n: t.detach().to("cpu", torch.float32).clone() for n, t in state.items()}
