import math
import os
import release_defaults
from typing import Optional

import torch
from pydantic import BaseModel

import train_step_clock as tsc
import wsd_a0

FAMILY = "stop_to_decay"
FLAT_TOL = 1e-9

_FALSY = ("0", "false", "no", "off")


def enabled() -> bool:
    return release_defaults.raw("SN56_STOP_TO_DECAY").strip().lower() not in _FALSY


def ee_priced_enabled() -> bool:
    return enabled() and release_defaults.raw("SN56_EE_PRICED").strip().lower() not in _FALSY


def min_decay_steps() -> int:
    try:
        return max(1, int(float(os.environ.get("SN56_ST_MIN_DECAY_STEPS", "") or 20)))
    except (TypeError, ValueError):
        return 20


def member_eval_frac() -> float:
    try:
        v = float(os.environ.get("SN56_ST_EVAL_FRAC", "") or 0.25)
    except (TypeError, ValueError):
        return 0.25
    return v if v > 0 else 0.25


def single_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size() <= 1
    return int(os.environ.get("WORLD_SIZE", "1") or 1) <= 1


def wsd_keywords(sched) -> Optional[dict]:
    lam = tsc.lambda_scheduler(sched)
    if lam is None or not getattr(lam, "lr_lambdas", None):
        return None
    kw = getattr(lam.lr_lambdas[0], "keywords", None) or {}
    if "num_stable_steps" not in kw or "num_decay_steps" not in kw:
        return None
    return dict(kw)


def _cads(eval_steps) -> tuple:
    if isinstance(eval_steps, (list, tuple, set)):
        c = tuple(sorted({int(x) for x in eval_steps if x and int(x) > 0}))
    else:
        c = (int(eval_steps),) if eval_steps and int(eval_steps) > 0 else ()
    return c or (1,)


def n_points(start: int, horizon: int, eval_steps) -> int:
    start, horizon = int(start), int(horizon)
    if horizon <= start:
        return 0
    cads = _cads(eval_steps)
    pts = set()
    for c in cads:
        pts.update(range((start // c + 1) * c, horizon + 1, c))
    pts.add(horizon)
    return len(pts)


def decay_cost(start: int, horizon: int, t_step: float, t_point: float, eval_steps) -> float:
    return ((int(horizon) - int(start)) * float(t_step)
            + n_points(start, horizon, eval_steps) * float(t_point))


def steps_fitting(*, avail_secs: float, t_step: float, t_point: float, eval_steps,
                  start: int, max_h: int) -> int:
    if max_h <= start or decay_cost(start, start + 1, t_step, t_point, eval_steps) > avail_secs:
        return int(start)
    lo, hi = int(start) + 1, int(max_h)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if decay_cost(start, mid, t_step, t_point, eval_steps) <= avail_secs:
            lo = mid
        else:
            hi = mid - 1
    return lo


class Handoff(BaseModel):
    ok: bool
    reason: str = ""
    step: int
    lr_frac: Optional[float] = None
    warmup: int = 0
    decay_start_before: Optional[int] = None
    horizon_before: Optional[int] = None
    decay_frac: Optional[float] = None
    decay_by_frac: int = 0
    decay_fit: Optional[int] = None
    decay: int = 0
    horizon: int = 0
    avail_secs: Optional[float] = None
    t_step: Optional[float] = None
    t_point: Optional[float] = None
    eval_steps: int = 0
    cadences: list = []
    gate_step: int = 0
    cost_secs: Optional[float] = None


def plan_handoff(*, step: int, kw: Optional[dict], avail_secs: Optional[float],
                 t_step: Optional[float], t_point: Optional[float], eval_steps,
                 lr_frac: Optional[float] = None, gate_step: int = 0) -> Handoff:
    step = int(step)
    cads = list(_cads(eval_steps))
    h = Handoff(ok=False, step=step, eval_steps=int(min(cads)), cadences=cads,
                gate_step=int(gate_step or 0),
                avail_secs=(None if avail_secs is None else round(float(avail_secs), 2)),
                t_step=t_step, t_point=t_point)
    if not kw:
        h.reason = "not_wsd"
        return h
    w0 = int(kw.get("num_warmup_steps", 0) or 0)
    s0, d0 = int(kw["num_stable_steps"]), int(kw["num_decay_steps"])
    total0 = w0 + s0 + d0
    h.warmup, h.decay_start_before, h.horizon_before = w0, w0 + s0, total0
    if lr_frac is None:
        lr_frac = wsd_a0.wsd_lr_lambda(
            step, num_warmup_steps=w0, num_stable_steps=s0, num_decay_steps=d0,
            min_lr_rate=float(kw.get("min_lr_rate", kw.get("min_lr_ratio", 0.0)) or 0.0))
    lr_frac = float(lr_frac)
    h.lr_frac = round(lr_frac, 8)
    if step < w0 or step >= w0 + s0 or lr_frac < 1.0 - FLAT_TOL:
        h.reason = "not_flat"
        return h
    frac = (d0 / total0) if total0 > 0 else wsd_a0.decay_frac()
    h.decay_frac = round(frac, 6)
    if frac >= 1.0:
        by_frac = d0
    else:
        by_frac = int(round(step * frac / (1.0 - frac)))
    by_frac = max(1, min(by_frac, total0 - step))
    h.decay_by_frac = by_frac
    if not t_step or t_step <= 0 or avail_secs is None or t_point is None:
        h.reason = "no_clock"
        return h
    t_point = max(0.0, float(t_point))
    hi = steps_fitting(avail_secs=float(avail_secs), t_step=float(t_step), t_point=t_point,
                       eval_steps=cads, start=step, max_h=step + by_frac)
    fit = hi - step
    h.decay_fit = int(fit)
    decay = min(by_frac, fit)
    floor = min(min_decay_steps(), by_frac)
    if decay < max(1, floor):
        h.reason = "decay_does_not_fit"
        return h
    if int(step + decay) < int(gate_step or 0):
        h.reason = "before_gate"
        return h
    h.decay, h.horizon = int(decay), int(step + decay)
    h.cost_secs = round(decay_cost(step, h.horizon, t_step, t_point, cads), 2)
    h.ok, h.reason = True, ("shortened_to_fit" if decay < by_frac else "")
    return h


def apply_handoff(sched, state, h: Handoff) -> dict:
    out = tsc.reshape_schedule(sched, horizon=int(h.horizon), warmup=int(h.warmup),
                               decay_start=int(h.step))
    if out.get("applied"):
        state.max_steps = int(h.horizon)
    return out


def member_floor(warmup: int) -> int:
    return max(1, int(warmup or 0)) + min_decay_steps() + 1


def member_cap(best_step: int, steps_run: int) -> int:
    return max(int(best_step or 0), int(steps_run or 0))


def longest_fitting(lo: int, hi: int, fits) -> Optional[int]:
    lo, hi = int(lo), int(hi)
    if hi < lo or not fits(lo):
        return None
    if fits(hi):
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def early_eval_cap(*, gate_step: int, t_step: Optional[float], frac: float) -> Optional[float]:
    if not t_step or t_step <= 0 or not math.isfinite(float(t_step)) or gate_step <= 0:
        return None
    return float(frac) * float(gate_step) * float(t_step)
