import array
import collections
import functools
import itertools
import math
import os
import time
import types
from typing import Callable, Optional

import numpy as np
import torch
from pydantic import BaseModel
from transformers import TrainerCallback

import length_grouping
import member_pricing as mp
import packed_fill
import release_defaults
import second_trajectory as traj2
import spare_budget as sb
import train_step_clock as tsc
import a0_cosine
import wsd_a0
from customized_trainer import eval_gate_step

_FALSY = ("0", "false", "no", "off")

DEFAULT_MAX_EPOCHS = 8.0
DRIFT_SAFETY = 0.95
STALL_MIN_EVALS = 4
STALL_MIN_EVALS_PHASED = 5
STALL_REL_FLOOR = 0.001
RESOLVE_REL = 0.01
MEMBER_MIN_FRAC = packed_fill.MEMBER_MIN_FRAC
REASON = "measured_remainder"


def enabled() -> bool:
    return release_defaults.raw("SN56_DEPTH_FIRST").strip().lower() not in _FALSY


def should_arm(dyn_pad: bool, packed: bool) -> bool:
    return bool(enabled() and dyn_pad and not packed)


def max_epochs() -> float:
    try:
        v = float(os.environ.get("SN56_DEPTH_MAX_EPOCHS", DEFAULT_MAX_EPOCHS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_EPOCHS
    return v if (math.isfinite(v) and v > 0) else DEFAULT_MAX_EPOCHS


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def _union_multiples(lo: int, hi: int, cads) -> int:
    if hi <= lo:
        return 0
    n = 0
    for k in range(1, len(cads) + 1):
        for combo in itertools.combinations(cads, k):
            m = functools.reduce(_lcm, combo)
            n += (-1) ** (k + 1) * (hi // m - lo // m)
    return n


def eval_points(start: int, h: int, cadences, gate: int = 0, pre_cadences=()) -> int:
    cads = sorted({int(c) for c in (cadences or ()) if c and int(c) > 0})
    pre = sorted({int(c) for c in (pre_cadences or ()) if c and int(c) > 0})
    start, h, gate = int(start), int(h), max(0, int(gate or 0))
    if h <= start:
        return 0
    cut = min(h, max(start, gate - 1))
    n = _union_multiples(start, cut, pre) + _union_multiples(cut, h, cads)
    last = cads if h >= gate else pre
    if not any(h % c == 0 for c in last):
        n += 1
    return n


def fit_horizon(*, avail_secs: float, t_step: float, t_point: float, cadences, start: int,
                max_h: int, gate_fn: Optional[Callable[[int], int]] = None, pre_cadences=()) -> int:
    lo, hi = int(start) + 1, max(int(start) + 1, int(max_h))

    def cost(h: int) -> float:
        gate = gate_fn(h) if gate_fn is not None else 0
        return ((h - start) * float(t_step)
                + eval_points(start, h, cadences, gate, pre_cadences) * float(t_point))

    if cost(lo) > avail_secs:
        return lo
    if cost(hi) <= avail_secs:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= avail_secs:
            lo = mid
        else:
            hi = mid - 1
    return lo


class DepthTail(BaseModel):
    total: float
    guard: float
    reserve: float
    devpass: float
    a0_soup: float
    pool: int
    t_eval: float
    source: str
    binding: str


def post_a0_tail(ckpt_avg, *, guard_secs: float, t_eval: float) -> DepthTail:
    facts_d = getattr(ckpt_avg, "stop_rule_facts", None) if ckpt_avg is not None else None
    pool = 0
    if ckpt_avg is not None:
        pool = max(len(getattr(ckpt_avg, "pool", []) or []), int(getattr(ckpt_avg, "soup_max", 0) or 0))
    _l2 = traj2.measured_tail(ckpt_avg) if packed_fill.active() else None
    if _l2 is not None:
        reserve, devpass, source = float(_l2.reserve), float(_l2.devpass_reserve), "packed_fill"
    elif facts_d is None:
        reserve, devpass, source = float(traj2._RESERVE_SECS), 0.0, "no_planner_facts"
    else:
        facts = traj2.Attempt0Facts(**facts_d)
        facts.eval_runtime = float(t_eval) if t_eval else facts.eval_runtime
        ckpt_avg._attach_clock(facts)
        dp_legacy = float(getattr(ckpt_avg, "stop_rule_devpass_reserve", 0.0) or 0.0)
        t = traj2.tail_reserves(facts, devfit_enabled=dp_legacy > 0, devpass_legacy=dp_legacy,
                                measured_devpass=ckpt_avg._measured_devpass(facts))
        reserve, devpass, source = float(t.reserve), float(t.devpass_reserve), str(t.source)
    soup = float(mp.a0_soup_secs(pool, float(t_eval or 0.0)))
    core = reserve + devpass + soup
    guard = float(guard_secs or 0.0)
    return DepthTail(total=max(guard, core), guard=guard, reserve=reserve, devpass=devpass,
                     a0_soup=soup, pool=pool, t_eval=float(t_eval or 0.0), source=source,
                     binding=("guard" if guard >= core else "tail"))


def megabatch_phase(step: int, updates_per_epoch: int, period: Optional[int]) -> Optional[int]:
    if not period or period <= 0 or step <= 0:
        return None
    upe = max(1, int(updates_per_epoch or 0)) if updates_per_epoch else None
    s = int(step) - 1
    if upe:
        s = s % upe
    return int(s % int(period))


class StallVerdict(BaseModel):
    fire: bool
    reason: str
    step: int
    n_window: int = 0
    window_first: Optional[int] = None
    span: int = 0
    phase_adjusted: bool = False
    slope_per_epoch: Optional[float] = None
    slope_se_per_epoch: Optional[float] = None
    phase_coef: Optional[float] = None
    dev_fit_now: Optional[float] = None
    drop: Optional[float] = None
    noise: Optional[float] = None
    paired_se: Optional[float] = None
    within_noise: Optional[bool] = None
    member_steps: Optional[int] = None
    depth_gain_member_steps: Optional[float] = None
    member_value: Optional[float] = None
    worth_member: Optional[bool] = None


def fit_slope(steps, devs, phases=None, period: Optional[int] = None) -> Optional[tuple]:
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(devs, dtype=np.float64)
    cols = [np.ones_like(x), x - x.mean()]
    phased = phases is not None and period
    if phased:
        cols.append(np.asarray(phases, dtype=np.float64) / float(period))
    X = np.stack(cols, axis=1)
    n, p = X.shape
    if n <= p or float(np.ptp(x)) <= 0:
        return None
    if phased and float(np.ptp(X[:, 2])) <= 0:
        X = X[:, :2]
        n, p = X.shape
        phased = False
    xtx = X.T @ X
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None
    beta = inv @ X.T @ y
    resid = y - X @ beta
    s2 = float(resid @ resid) / max(1, n - p)
    se_b = math.sqrt(max(0.0, s2 * float(inv[1, 1])))
    c = float(beta[2]) if phased else None
    last = float(beta[0] + beta[1] * (x[-1] - x.mean()))
    if phased:
        last += float(beta[2]) * float(np.mean(X[:, 2]))
    return float(beta[1]), se_b, c, last


class StallTracker:

    def __init__(self, min_evals: int = STALL_MIN_EVALS, min_evals_phased: int = STALL_MIN_EVALS_PHASED,
                 rel_floor: float = STALL_REL_FLOOR):
        self.min_evals = int(min_evals)
        self.min_evals_phased = int(min_evals_phased)
        self.rel_floor = float(rel_floor)
        self.steps: list = []
        self.devs: list = []
        self.phases: list = []
        self.rows: "collections.OrderedDict[int, array.array]" = collections.OrderedDict()

    def add(self, step: int, dev: float, phase: Optional[int], rows: Optional[list],
            keep_after: Optional[int] = None) -> bool:
        step, dev = int(step), float(dev)
        if not math.isfinite(dev) or (self.steps and step <= self.steps[-1]):
            return False
        self.steps.append(step)
        self.devs.append(dev)
        self.phases.append(phase)
        if rows:
            self.rows[step] = array.array("d", (float(x) for x in rows))
        if keep_after is not None:
            for s in [s for s in self.rows if s < keep_after]:
                self.rows.pop(s, None)
        return True

    def check(self, *, steps_per_epoch: int, period: Optional[int], member_steps: Optional[int],
              member_value: float, min_step: int = 0) -> StallVerdict:
        n = len(self.steps)
        g = self.steps[-1] if n else 0
        spe = int(steps_per_epoch or 0)
        if n == 0 or spe <= 0 or g < spe:
            return StallVerdict(fire=False, reason="under_one_epoch", step=int(g))
        phased = bool(period) and all(p is not None for p in self.phases)
        need = self.min_evals_phased if phased else self.min_evals
        idx = [i for i in range(n) if self.steps[i] > max(g - spe, int(min_step))]
        if len(idx) < need:
            idx = [i for i in range(n) if self.steps[i] > int(min_step)][-need:]
        if len(idx) < need:
            return StallVerdict(fire=False, reason="too_few_evals", step=int(g), n_window=len(idx))
        st = [self.steps[i] for i in idx]
        dv = [self.devs[i] for i in idx]
        ph = [self.phases[i] for i in idx] if phased else None
        fit = fit_slope(st, dv, ph, period if phased else None)
        if fit is None:
            return StallVerdict(fire=False, reason="too_few_evals", step=int(g), n_window=len(idx))
        b, se_b, c, level = fit
        span = st[-1] - st[0]
        drop = -b * span
        ra, rb = self.rows.get(st[0]), self.rows.get(st[-1])
        pse = sb.paired_se(list(ra), list(rb)) if (ra is not None and rb is not None) else None
        noise = max(pse or 0.0, se_b * span, self.rel_floor * abs(level))
        within = bool(drop <= noise)
        m_steps = int(member_steps) if member_steps else 0
        gain = max(0.0, -b) * m_steps
        worth = bool(gain <= float(member_value))
        fire = within and worth
        reason = "stall" if fire else ("descending" if not within else "worth_depth")
        return StallVerdict(
            fire=fire, reason=reason, step=int(g), n_window=len(idx), window_first=int(st[0]),
            span=int(span), phase_adjusted=bool(phased), slope_per_epoch=b * spe,
            slope_se_per_epoch=se_b * spe, phase_coef=c, dev_fit_now=level, drop=drop,
            noise=noise, paired_se=pse, within_noise=within, member_steps=m_steps,
            depth_gain_member_steps=gain, member_value=float(member_value), worth_member=worth)


def member_value() -> float:
    return max(float(sb.G_MIN), float(mp.member_ev(2)))


def decay_horizon(g: int, frac: float) -> int:
    frac = float(frac)
    return int(g) + max(1, int(round(frac / (1.0 - frac) * int(g))))


def wsd_phase(lr_scheduler) -> Optional[dict]:
    lam = tsc.lambda_scheduler(lr_scheduler)
    if lam is None or not lam.lr_lambdas:
        return None
    return wsd_a0.describe(lam.lr_lambdas[0])


class A0Phase(BaseModel):
    family: str
    horizon: int
    decay_start: Optional[int] = None
    decay_frac: float

    @property
    def cosine(self) -> bool:
        return self.family == a0_cosine.FAMILY

    def resolvable_at(self, g: int) -> bool:
        return True if self.cosine else bool(int(g) < int(self.decay_start))


def a0_phase(lr_scheduler) -> Optional[A0Phase]:
    lam = tsc.lambda_scheduler(lr_scheduler)
    if lam is None or not lam.lr_lambdas:
        return None
    f = lam.lr_lambdas[0]
    c = a0_cosine.describe(f)
    if c is not None:
        return A0Phase(family=a0_cosine.FAMILY, horizon=int(c["horizon"]),
                       decay_frac=wsd_a0.decay_frac())
    ph = wsd_a0.describe(f)
    if ph is None:
        return None
    return A0Phase(family=str(ph["family"]), horizon=int(ph["horizon"]),
                   decay_start=int(ph["decay_start"]),
                   decay_frac=((ph["decay"] / ph["horizon"]) if ph["horizon"] > 0
                               else wsd_a0.decay_frac()))


class TurnVerdict(BaseModel):
    boundary: int
    epoch: int
    step: int
    dev: float
    best: float
    rise: float
    noise: Optional[float] = None
    noise_step: Optional[int] = None
    g1_disarm: bool = False
    trial: Optional[dict] = None
    trial_gain: Optional[float] = None
    trial_improves: Optional[bool] = None
    disarm: bool = False
    reason: str = ""


class TurnGate:

    def __init__(self) -> None:
        self.evals: list = []
        self.checks: list = []
        self.judged: set = set()
        self.verdicts: list = []
        self.disarmed: Optional[TurnVerdict] = None

    def add_check(self, step: int, noise: Optional[float]) -> None:
        if noise is not None and math.isfinite(float(noise)):
            self.checks.append((int(step), float(noise)))

    def seen(self, step: int) -> bool:
        return bool(self.evals) and int(step) <= self.evals[-1][0]

    def observe(self, step: int, dev: float, *, steps_per_epoch: int, stride: int,
                trial: Optional[Callable[[], Optional[dict]]] = None) -> list:
        step, dev = int(step), float(dev)
        if self.seen(step) or not math.isfinite(dev):
            return []
        out = []
        spe = int(steps_per_epoch or 0)
        if spe > 0 and self.evals and self.disarmed is None:
            half = max(1, int(stride or 0) // 2)
            trial_out = None
            tried = False
            k = 1
            while k * spe + half < step:
                b = k * spe
                if b in self.judged:
                    k += 1
                    continue
                pre = [d for s, d in self.evals if s <= b]
                self.judged.add(b)
                if not pre:
                    k += 1
                    continue
                best = min(pre)
                ref = [(s, n) for s, n in self.checks if s <= b]
                v = TurnVerdict(boundary=b, epoch=k, step=step, dev=dev, best=best, rise=dev - best)
                if not ref:
                    v.reason = "no_noise"
                else:
                    v.noise_step, v.noise = ref[-1]
                    v.g1_disarm = bool(v.rise > v.noise)
                    if not v.g1_disarm:
                        v.reason = "within_noise"
                    else:
                        if trial is not None and not tried:
                            tried = True
                            try:
                                trial_out = trial()
                            except Exception:
                                trial_out = {"ran": False, "why": "error"}
                        v.trial = dict(trial_out) if trial_out is not None else None
                        gain = (trial_out or {}).get("gain") if (trial_out or {}).get("ran") else None
                        if gain is not None and math.isfinite(float(gain)):
                            v.trial_gain = float(gain)
                            v.trial_improves = bool(v.trial_gain > v.noise)
                        if v.trial_improves:
                            v.reason = "soup_improves"
                        else:
                            v.disarm = True
                            v.reason = "turn" if v.trial_gain is not None else "turn_no_trial"
                out.append(v)
                self.verdicts.append(v)
                if v.disarm:
                    self.disarmed = v
                    break
                k += 1
        self.evals.append((step, dev))
        return out


def install_loop(trainer, cap_epochs: float) -> dict:
    orig = trainer.set_initial_training_values
    loop = int(math.ceil(float(cap_epochs)))
    info = {"loop_epochs": loop}

    def set_initial_training_values(self, args, dataloader):
        vals = list(orig(args, dataloader))
        if int(getattr(args, "max_steps", -1) or -1) <= 0 and vals[0] < loop:
            info["loop_epochs_hf"] = int(vals[0])
            vals[0] = loop
        return tuple(vals)

    trainer.set_initial_training_values = types.MethodType(set_initial_training_values, trainer)
    return info


def _single_process() -> bool:
    return not (torch.distributed.is_available() and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1)


class DepthFirst(TrainerCallback):

    def __init__(self, *, trainer, ckpt_avg, handler, eval_save_cb=None,
                 cap_epochs: Optional[float] = None, restate_gate=None,
                 log: Callable[[str], None] = print) -> None:
        self.trainer = trainer
        self.ckpt_avg = ckpt_avg
        self.handler = handler
        self.eval_save_cb = eval_save_cb
        self.cap = float(max_epochs() if cap_epochs is None else cap_epochs)
        self.restate_gate = restate_gate
        self.log = log
        self.clock_cb = None
        self.tracker = StallTracker()
        self.active = False
        self.decaying = False
        self.done = False
        self.last_t_eval: Optional[float] = None
        self.last_plan: Optional[dict] = None
        self.period: Optional[int] = None
        self.upe: Optional[int] = None
        self.warmup: int = 0
        self._clk_mark: Optional[tuple] = None
        self._wall: dict = {}
        self.turn = TurnGate()
        self.disarmed = False
        self.legacy_gate = None
        self.legacy_loop: Optional[int] = None
        self.legacy_h: Optional[int] = None
        self.legacy_applied = False

    def guard_secs(self) -> float:
        h = self.handler
        return float(getattr(h, "save_before_remaining_time", 0) or 0) * 60.0 if h is not None else 0.0

    def cadences(self, args, state) -> tuple:
        cads = []
        if str(getattr(args, "eval_strategy", "no")).lower().endswith("steps"):
            cads.append(int(getattr(state, "eval_steps", 0) or getattr(args, "eval_steps", 0) or 0))
        h = self.handler
        if h is not None:
            cads.append(int(getattr(h, "periodic_save_steps", -1) or -1))
            cads.append(int(getattr(h, "steps_per_epoch", -1) or -1))
        return tuple(c for c in cads if c > 0)

    def gate_at(self, spe: int) -> Callable[[int], int]:
        total = int(getattr(self.eval_save_cb, "total_steps_all_epochs", 0) or 0) \
            if self.eval_save_cb is not None else 0
        spe = max(1, int(spe))

        def gate(h: int) -> int:
            t = min(total, int(h)) if total > 0 else int(h)
            return int(eval_gate_step(t, round(int(h) / spe, 4)))

        return gate

    def pre_cadences(self) -> tuple:
        ee = getattr(self.eval_save_cb, "early_evals", None) if self.eval_save_cb is not None else None
        if ee is None or not getattr(ee, "active", False):
            return ()
        return (int(getattr(ee, "cad", 0) or 0),)

    def t_point_prior(self) -> tuple:
        ee = getattr(self.eval_save_cb, "early_evals", None) if self.eval_save_cb is not None else None
        try:
            u = ee.gate.unit() if ee is not None else None
        except Exception:
            u = None
        if u is not None and math.isfinite(u) and u > 0:
            return float(u), ("early_evals_measured" if ee.gate.n_done > 0 else "early_evals_prior")
        if self.last_t_eval:
            return float(self.last_t_eval), "a0_eval_runtime"
        return float(traj2._EVAL_SECS_FALLBACK), "fallback"

    def _a0_clock(self):
        pc = getattr(self.ckpt_avg, "planner_clock", None) if self.ckpt_avg is not None else None
        return getattr(pc, "clock", None)

    def mark_clock(self) -> None:
        clk = self._a0_clock()
        if clk is not None:
            self._clk_mark = (len(clk.free), len(clk.points))

    def measured_clock(self) -> Optional[tuple]:
        clk = self._a0_clock()
        if clk is None or self._clk_mark is None:
            return None
        free = list(clk.free[self._clk_mark[0]:])
        pts = list(clk.points[self._clk_mark[1]:])
        if len(pts) < 1 or len(free) < 10:
            return None
        t_step = sum(free) / len(free)
        t_point = max(0.0, sum(pts) / len(pts) - t_step)
        if not (math.isfinite(t_step) and t_step > 0 and math.isfinite(t_point)):
            return None
        return t_step, t_point, len(free), len(pts)

    def t_eval_now(self, t_point: float) -> float:
        return float(self.last_t_eval) if self.last_t_eval else float(t_point)

    def horizon(self, *, g: int, remaining: float, t_step: float, t_point: float, args, state,
                spe: int, t_point_source: str) -> tuple:
        tail = post_a0_tail(self.ckpt_avg, guard_secs=self.guard_secs(),
                            t_eval=self.t_eval_now(t_point))
        cads = self.cadences(args, state)
        pre = self.pre_cadences()
        gate_fn = self.gate_at(spe)
        avail = DRIFT_SAFETY * (float(remaining) - tail.total)
        max_h = max(int(g) + 1, int(self.cap * max(1, int(spe))))
        h = fit_horizon(avail_secs=avail, t_step=float(t_step), t_point=float(t_point),
                        cadences=cads, start=int(g), max_h=max_h, gate_fn=gate_fn,
                        pre_cadences=pre)
        info = {"depth_first": True, "cap": self.cap, "max_h": max_h,
                "capped": bool(h >= max_h and max_h > int(g) + 1), "drift_safety": DRIFT_SAFETY,
                "tail": tail.model_dump(), "tail_secs": round(tail.total, 1),
                "avail_secs": round(avail, 1), "remaining": round(float(remaining), 1),
                "t_step": round(float(t_step), 5), "t_point": round(float(t_point), 3),
                "t_point_source": t_point_source, "cadences": list(cads),
                "pre_gate_cadences": list(pre), "gate": int(gate_fn(h)),
                "n_points": eval_points(int(g), h, cads, gate_fn(h), pre), "horizon": int(h),
                "start": int(g),
                "epochs": round(h / max(1, int(spe)), 4), "steps_per_epoch": int(spe)}
        return int(h), info

    def steps_per_epoch(self) -> int:
        spe = int(getattr(self.handler, "steps_per_epoch", 0) or 0) if self.handler is not None else 0
        if spe <= 0 and self.clock_cb is not None and self.clock_cb.plan is not None:
            spe = int(self.clock_cb.plan.steps_per_epoch)
        return spe

    def legacy_horizon(self, *, t_step: float, g: int, state, legacy_plan) -> Optional[int]:
        cur = int(self.legacy_h if self.legacy_h is not None
                  else (getattr(state, "max_steps", 0) or 0))
        if cur <= 0:
            return None
        h = cur
        if legacy_plan is not None and self.legacy_gate is not None:
            try:
                ok, _ = self.legacy_gate(float(t_step), legacy_plan, int(g), state)
            except Exception:
                ok = False
            if ok:
                hl = int(legacy_plan.new_total_steps)
                if self.legacy_loop:
                    hl = min(hl, int(self.legacy_loop))
                hl = max(int(g) + 1, hl)
                if hl > cur:
                    h = hl
                    self.legacy_applied = True
        self.legacy_h = int(h)
        return self.legacy_h

    def clock_fit(self, *, g: int, remaining: float, t_step: float, t_point: float,
                  state) -> Optional[int]:
        if not tsc.shrink_enabled():
            return None
        try:
            guard = tsc.guard_priced_secs(self.guard_secs(), float(t_step), float(t_point))
            return int(tsc.steps_fitting(avail_secs=float(remaining) - guard, t_step=float(t_step),
                                         t_point=float(t_point),
                                         eval_steps=int(getattr(state, "eval_steps", 0) or 0),
                                         start=int(g)))
        except Exception:
            return None

    def _floor(self, h: int, info: dict, fit: Optional[int] = None) -> int:
        lh = self.legacy_h
        info["legacy_horizon"] = lh
        if lh is not None and fit is not None and int(fit) < int(lh):
            info["legacy_clock_fit"] = int(fit)
            info["reason"] = "legacy_overrun"
            lh = max(1, int(fit))
        info["legacy_floor"] = bool(lh is not None and int(h) < int(lh))
        if info["legacy_floor"]:
            info["horizon_measured"] = int(h)
            h = int(lh)
            info["horizon"] = h
            try:
                spe = int(info.get("steps_per_epoch") or 0)
                gate = int(self.gate_at(spe)(h)) if spe > 0 else 0
                info["gate"] = gate
                if "start" in info:
                    info["n_points"] = eval_points(int(info["start"]), h, info.get("cadences") or (),
                                                   gate, info.get("pre_gate_cadences") or ())
                info["capped"] = bool(h >= int(info.get("max_h") or 0) > int(info.get("start", 0)) + 1)
            except Exception:
                pass
        return int(h)

    def plan(self, *, t_step: float, g: int, args, state, remaining: float, steps_per_epoch: int,
             warmup_steps_arg: int, eval_steps_arg: float, legacy_plan=None) -> tsc.EpochPlan:
        tp, src = self.t_point_prior()
        h, info = self.horizon(g=g, remaining=remaining, t_step=t_step, t_point=tp, args=args,
                               state=state, spe=steps_per_epoch, t_point_source=src)
        self.legacy_horizon(t_step=t_step, g=g, state=state, legacy_plan=legacy_plan)
        h = self._floor(h, info)
        info["epochs"] = round(h / max(1, int(steps_per_epoch)), 4)
        self.last_plan = info
        self.mark_clock()
        spe = max(1, int(steps_per_epoch))
        return tsc.EpochPlan(
            source=tsc.SOURCE_TRAIN, optimal_epochs=round(h / spe, 4),
            warmup=min(warmup_steps_arg, max(10, int(0.03 * h))), eval_steps=float(eval_steps_arg),
            new_total_steps=int(h), steps_per_epoch=int(spe), achievable_steps=int(h),
            remaining_secs=float(remaining), t_per_step=float(t_step), max_epochs=self.cap,
            steps_done=int(g), guard_secs=float(info["tail_secs"]), below_one_epoch=bool(h < spe))

    def gate(self, legacy):

        self.legacy_gate = legacy

        def depth_gate(t_step, plan, g, state):
            try:
                ok_legacy, info = legacy(t_step, plan, g, state)
            except Exception:
                ok_legacy, info = None, {}
            return True, {**dict(info or {}), "depth_first": True, "legacy_ok": ok_legacy,
                          "reason": REASON,
                          **({"depth_plan": self.last_plan} if self.last_plan else {})}

        return depth_gate

    def locked(self) -> bool:
        ds = getattr(self.ckpt_avg, "decay_started", None) if self.ckpt_avg is not None else None
        return bool(isinstance(ds, dict) and ds.get("by") != "depth_first")

    def holds_stop(self, g: int) -> bool:
        try:
            if not self.active or self.disarmed or not _single_process():
                return False
            if self.decaying:
                return True
            sched = getattr(self.trainer, "lr_scheduler", None) if self.trainer is not None else None
            ph = a0_phase(sched)
            if ph is None or not (ph.cosine or ph.resolvable_at(int(g))):
                return False
            spe = self.steps_per_epoch()
            return bool(int(g) < spe or self.turn.checks)
        except Exception:
            return False

    def resolvable_at(self, g: int, lr_scheduler) -> bool:
        ph = a0_phase(lr_scheduler)
        return bool(ph is not None and ph.resolvable_at(int(g)))

    def flat_at(self, g: int, lr_scheduler) -> bool:
        return self.resolvable_at(g, lr_scheduler)

    def reprice_fit(self, *, g: int, remaining: float, t_step: float, t_point: float, args,
                    state, legacy_avail: Optional[float] = None) -> tuple:
        h, info = self.horizon(g=g, remaining=remaining, t_step=t_step, t_point=t_point, args=args,
                               state=state, spe=self.steps_per_epoch(),
                               t_point_source="measured_point")
        if (legacy_avail is not None and self.legacy_applied and self.legacy_h is not None):
            try:
                lf = tsc.steps_fitting(avail_secs=float(legacy_avail), t_step=float(t_step),
                                       t_point=float(t_point),
                                       eval_steps=int(getattr(state, "eval_steps", 0) or 0),
                                       start=int(g))
                if lf < int(self.legacy_h):
                    self.legacy_h = int(lf)
                info["legacy_reprice_fit"] = int(lf)
            except Exception:
                pass
        h = self._floor(h, info, self.clock_fit(g=g, remaining=remaining, t_step=t_step,
                                                t_point=t_point, state=state))
        return h, info

    def eval_fields(self, state) -> dict:
        if not self.active:
            return {}
        return {"megabatch_phase": self.phase_of(int(state.global_step)),
                "megabatch_period": self.period}

    def phase_of(self, g: int) -> Optional[int]:
        if length_grouping.dealcal_enabled():
            return length_grouping.sampler_phase(g) if self.period else None
        return megabatch_phase(g, self.upe, self.period)

    def on_train_begin(self, args, state, control, train_dataloader=None, **kwargs):
        self.active = True
        try:
            self.warmup = int(args.get_warmup_steps(int(state.max_steps or 0)))
        except Exception:
            self.warmup = int(getattr(args, "warmup_steps", 0) or 0)
        try:
            n = len(train_dataloader)
            accum = max(1, int(args.gradient_accumulation_steps))
            self.upe = max(1, n // accum + int(n % accum > 0))
        except Exception:
            self.upe = None
        cur = int(getattr(state, "max_steps", 0) or 0)
        if int(getattr(args, "max_steps", -1) or -1) > 0 or self.upe is None:
            self.legacy_loop = cur or None
        else:
            self.legacy_loop = max(cur, int(math.ceil(float(args.num_train_epochs))) * self.upe)
        self.period = (int(length_grouping.MEGABATCH_MULT)
                       if (length_grouping.ACTIVE and length_grouping.LENGTHS is not None) else None)
        if self.period and length_grouping.dealcal_enabled():
            self.period = length_grouping.sampler_period() or self.period
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.active = False
        return control

    def on_evaluate(self, args, state, control, metrics=None, lr_scheduler=None, **kwargs):
        if not self.active or not metrics:
            return control
        if self.ckpt_avg is not None and getattr(self.ckpt_avg, "_evaluating", False):
            return control
        rt = metrics.get("eval_runtime")
        if rt:
            self.last_t_eval = float(rt)
        dev = metrics.get("eval_loss")
        if dev is not None:
            self.turn_gate(args, state, kwargs.get("model"), float(dev))
        if dev is None or self.done or not _single_process():
            return control
        try:
            g = int(state.global_step)
            spe = self.steps_per_epoch()
            rows = getattr(self.trainer, "last_eval_rows", None)
            if not self.tracker.add(g, float(dev), self.phase_of(g), rows,
                                    keep_after=(g - 2 * spe if spe > 0 else None)):
                return control
            self._wall[g] = time.time()
            if self.decaying:
                return control
            if self.locked():
                self.done = True
                return control
            sched = lr_scheduler or getattr(self.trainer, "lr_scheduler", None)
            ph = a0_phase(sched)
            if ph is None:
                self.done = True
                return control
            if not ph.resolvable_at(g):
                self.done = True
                return control
            if self._stall(args, state, sched, spe):
                return control
            self._resolve(args, state, sched, spe)
        except Exception:
            pass
        return control

    def eval_stride(self, args, state) -> int:
        if not str(getattr(args, "eval_strategy", "no")).lower().endswith("steps"):
            return 0
        return int(getattr(state, "eval_steps", 0) or getattr(args, "eval_steps", 0) or 0)

    def trial_budget(self, args, state) -> tuple:
        g = int(state.global_step)
        clk = self.clock_cb
        t_eval = float(self.last_t_eval or 0.0)
        write = 0.0
        if self.ckpt_avg is not None and hasattr(self.ckpt_avg, "soup_trial_write_secs"):
            write = float(self.ckpt_avg.soup_trial_write_secs())
        est = float(mp.probe_cost_est(t_eval, write))
        if clk is None or t_eval <= 0:
            return est, None
        m = self.measured_clock()
        if m is not None:
            t_step, t_point = m[0], m[1]
        elif clk.t_per_step:
            t_step, t_point = float(clk.t_per_step), self.t_point_prior()[0]
        else:
            return est, None
        tail = post_a0_tail(self.ckpt_avg, guard_secs=self.guard_secs(), t_eval=t_eval)
        h = int(getattr(state, "max_steps", 0) or 0)
        spe = self.steps_per_epoch()
        need = (max(0, h - g) * float(t_step)
                + eval_points(g, h, self.cadences(args, state),
                              self.gate_at(spe)(h) if spe > 0 else 0, self.pre_cadences())
                * float(t_point))
        return est, DRIFT_SAFETY * (float(clk.remaining_fn()) - tail.total) - need

    def turn_gate(self, args, state, model, dev: float) -> None:
        if not self.active or self.disarmed or not _single_process():
            return
        if self.ckpt_avg is not None and getattr(self.ckpt_avg, "_evaluating", False):
            return
        g = int(state.global_step)
        if self.turn.seen(g):
            return
        try:
            def trial():
                if self.ckpt_avg is None or not hasattr(self.ckpt_avg, "soup_pair_trial"):
                    return {"ran": False, "why": "no_trial"}
                clk = self.clock_cb
                if clk is not None and clk.plan is not None and not clk.repriced:
                    return {"ran": False, "why": "reprice_pending"}
                est, budget = self.trial_budget(args, state)
                if budget is None:
                    return {"ran": False, "why": "no_clock", "est_secs": round(est, 2)}
                return self.ckpt_avg.soup_pair_trial(state, model, est_secs=est,
                                                     budget_secs=budget)

            verdicts = self.turn.observe(g, float(dev), steps_per_epoch=self.steps_per_epoch(),
                                         stride=self.eval_stride(args, state), trial=trial)
            for v in verdicts:
                fields = {k: (round(x, 6) if isinstance(x, float) else x)
                          for k, x in v.model_dump().items()}
                if v.disarm:
                    self._disarm(state, fields)
                else:
                    pass
        except Exception:
            pass

    def _disarm(self, state, fields: dict) -> None:
        self.disarmed = True
        self.done = True
        self.decaying = False
        try:
            self.log(f"[sn56][profundidade] virada de memorização no passo {fields.get('step')} "
                     f"(subida {fields.get('rise')} > ruído {fields.get('noise')}, sopa "
                     f"{fields.get('trial_gain')}): profundidade desarmada no resto do a0")
        except Exception:
            pass

    def _remaining_after_tail(self) -> Optional[float]:
        clk = self.clock_cb
        if clk is None:
            return None
        try:
            tail = post_a0_tail(self.ckpt_avg, guard_secs=self.guard_secs(),
                                t_eval=self.t_eval_now(traj2._EVAL_SECS_FALLBACK))
            return float(clk.remaining_fn()) - tail.total
        except Exception:
            return None

    def _member_steps(self, g: int) -> Optional[int]:
        best = int(getattr(self.ckpt_avg, "best_step", 0) or 0) if self.ckpt_avg is not None else 0
        if best <= 0:
            best = int(g)
        st = self.tracker.steps
        w0 = next((s for s in st if s in self._wall and s >= max(1, g - self.steps_per_epoch())), None)
        rem = self._remaining_after_tail()
        if w0 is None or w0 >= g or rem is None:
            return best
        t_eff = (self._wall[g] - self._wall[w0]) / float(g - w0)
        if not (t_eff > 0 and math.isfinite(t_eff)):
            return best
        return int(max(0, min(best, rem / t_eff)))

    def _stall(self, args, state, sched, spe: int) -> bool:
        g = int(state.global_step)
        v = self.tracker.check(steps_per_epoch=spe, period=self.period,
                               member_steps=self._member_steps(g), member_value=member_value(),
                               min_step=self.warmup)
        if v.reason == "under_one_epoch":
            return False
        self.turn.add_check(g, v.noise)
        if not v.fire:
            return False
        ph = a0_phase(sched)
        frac = ph.decay_frac if ph is not None else wsd_a0.decay_frac()
        current = int(getattr(state, "max_steps", 0) or 0)
        want = decay_horizon(g, frac)
        fit = None
        m = self.measured_clock()
        clk = self.clock_cb
        if clk is not None and (m is not None or clk.t_per_step):
            t_step, t_point, src = ((m[0], m[1], "a0_clock") if m is not None else
                                    (float(clk.t_per_step), *self.t_point_prior()))
            try:
                fit, _ = self.horizon(g=g, remaining=float(clk.remaining_fn()), t_step=t_step,
                                      t_point=t_point, args=args, state=state, spe=spe,
                                      t_point_source=src)
            except Exception:
                pass
        new_h = min(want, current) if current > 0 else want
        if fit is not None:
            new_h = min(new_h, int(fit))
        new_h = max(g + 1, int(new_h))
        self.done = True
        if new_h >= current > 0:
            return True
        if self.ckpt_avg is not None and hasattr(self.ckpt_avg, "claim_decay") \
                and not self.ckpt_avg.claim_decay("depth_first", g):
            return True
        lr = self._apply_horizon(args, state, sched, new_h, spe, stage="stall")
        self.decaying = True
        try:
            self.log(f"[sn56][profundidade] estagnação do a0 no passo {g} (inclinação "
                     f"{v.slope_per_epoch:+.4f}/época, queda {v.drop:.4f} <= ruído {v.noise:.4f}): "
                     f"decaimento agora, horizonte {current} -> {new_h}")
        except Exception:
            pass
        return True

    def _resolve(self, args, state, sched, spe: int) -> None:
        clk = self.clock_cb
        m = self.measured_clock()
        if clk is None or clk.plan is None or m is None:
            return
        g = int(state.global_step)
        h, info = self.horizon(g=g, remaining=float(clk.remaining_fn()), t_step=m[0],
                               t_point=m[1], args=args, state=state, spe=spe,
                               t_point_source="a0_clock")
        h = self._floor(h, info, self.clock_fit(g=g, remaining=float(clk.remaining_fn()),
                                                t_step=m[0], t_point=m[1], state=state))
        info["epochs"] = round(h / max(1, int(spe)), 4)
        current = int(getattr(state, "max_steps", 0) or 0)
        apply = abs(h - current) >= max(1, int(RESOLVE_REL * current))
        if apply:
            self._apply_horizon(args, state, sched, h, spe, stage="resolve")
            try:
                self.log(f"[sn56][profundidade] horizonte re-resolvido no passo {g}: {current} -> "
                         f"{h} (t_passo={m[0]:.4f}s, t_ponto={m[1]:.2f}s)")
            except Exception:
                pass

    def _apply_horizon(self, args, state, sched, new_h: int, spe: int, stage: str) -> dict:
        lr = tsc.reshape_schedule(sched, horizon=new_h, warmup=int(args.get_warmup_steps(new_h)))
        args.num_train_epochs = round(new_h / max(1, spe), 4)
        state.max_steps = int(new_h)
        state.num_train_epochs = int(math.ceil(args.num_train_epochs))
        clk = self.clock_cb
        if clk is not None:
            clk.horizon = int(new_h)
            if stage == "stall":
                clk.repriced = True
                clk._point_pending = False
        if self.eval_save_cb is not None and self.restate_gate is not None:
            try:
                gate = self.restate_gate(self.eval_save_cb, int(new_h), args.num_train_epochs)
            except Exception:
                pass
        return lr
