import datetime
import functools
import math
import os
import time
from typing import Callable, Optional, Union

import torch
from pydantic import BaseModel
from transformers import TrainerCallback

import a0_cosine
import rank_agreement
import release_defaults
import wsd_a0

SKIP_STEPS = 2
MEASURE_STEPS = 8
EXHAUST_FRAC = 0.85
SEARCH_OVERSHOOT = 1.25
SEARCH_MIN_EPOCHS = 1.3

SOURCE_SEARCH = "lr_search"
SOURCE_DS = "train_steps_ds"
SOURCE_TRAIN = "train_steps"

_FALSY = ("0", "false", "no", "off")


def enabled() -> bool:
    return os.environ.get("SN56_TRAIN_STEP_CLOCK", "1").strip().lower() not in _FALSY


def _env_int(name: str, default: int, lo: int) -> int:
    try:
        return max(lo, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def extend_enabled() -> bool:
    return os.environ.get("SN56_A0_EXTEND", "1").strip().lower() not in _FALSY


REMEASURE_AT = 0


def extend_remeasure_at() -> int:
    try:
        return max(0, int(os.environ.get("SN56_A0_EXTEND_REMEASURE", REMEASURE_AT)))
    except (TypeError, ValueError):
        return REMEASURE_AT


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def shrink_enabled() -> bool:
    return release_defaults.raw("SN56_A0_SHRINK").strip().lower() not in _FALSY


def guard_priced_secs(guard_secs: float, t_step: float, t_point: float) -> float:
    g = float(guard_secs or 0.0)
    if g <= 0:
        return g
    return max(g, float(t_step) + float(t_point))


def extend_reserve_secs() -> float:
    return _env_float("SN56_A0_EXTEND_RESERVE", 180.0)


def no_member_fits_gate(remaining_fn: Callable[[], float], planned_steps_fn: Callable[[], int],
                        member_ratio: float = 1.1, reserve_secs: Optional[float] = None,
                        reserve_fn: Optional[Callable[[], Optional[float]]] = None):
    reserve_fixed = extend_reserve_secs() if reserve_secs is None else float(reserve_secs)

    def gate(t_step: float, plan: "EpochPlan", g: int, state) -> tuple:
        reserve, src = reserve_fixed, "fixed"
        if reserve_fn is not None:
            try:
                r = reserve_fn()
            except Exception:
                r = None
            if r is not None and r == r and r >= 0:
                reserve, src = float(r), "tail_reserves"
        remaining = float(remaining_fn())
        planned = int(planned_steps_fn() or 0)
        t = float(t_step)
        after = remaining - max(0, planned - int(g)) * t
        member = float(member_ratio) * planned * t + reserve
        ok = bool(planned > 0 and after < member)
        return ok, {"remaining": round(remaining, 1), "planned": planned, "t_step": round(t, 5),
                    "after": round(after, 1), "member": round(member, 1),
                    "member_ratio": float(member_ratio), "reserve": float(reserve),
                    "step": int(g),
                    **({"reserve_source": src} if reserve_fn is not None else {}),
                    "planned_measured": int(plan.new_total_steps) if plan is not None else None}

    return gate


EXTEND_MEASURE_SECS = 60.0
EXTEND_MEASURE_MAX = 512


def extend_measure_steps(t_search: Optional[float]) -> int:
    base = measure_steps()
    secs = _env_float("SN56_A0_EXTEND_MEASURE_SECS", EXTEND_MEASURE_SECS)
    if not t_search or t_search <= 0 or secs <= 0:
        return base
    return int(max(base, min(EXTEND_MEASURE_MAX, math.ceil(secs / float(t_search)))))


EXTEND_WARMUP_FLOOR = 20
EXTEND_SKIP_PAD = 2
EXTEND_MEASURE_MIN = 100
EXTEND_CYCLE = 50
EXTEND_DECIDE_FRAC = 0.10
WINDOW_SETTLED = "settled"
WINDOW_FALLBACK_SHORT = "fallback_short_run"
WINDOW_LEGACY = "legacy"


def extend_settle_enabled() -> bool:
    return os.environ.get("SN56_A0_EXTEND_SETTLE", "1").strip().lower() not in _FALSY


class ExtendWindow(BaseModel):
    skip: int
    measure: int
    rule: str
    warmup: int
    horizon: int
    installed_skip: int
    installed_measure: int
    decide_by: int
    t_first: Optional[float] = None
    clipped: bool = False


def _round_up(n: int, m: int) -> int:
    return int(math.ceil(max(0, int(n)) / float(m)) * m)


def extend_window(*, installed_skip: int, installed_measure: int, warmup: int,
                  horizon: int) -> ExtendWindow:
    skip = max(int(warmup), EXTEND_WARMUP_FLOOR) + EXTEND_SKIP_PAD
    decide_by = int(EXTEND_DECIDE_FRAC * max(0, int(horizon)))
    base = dict(warmup=int(warmup), horizon=int(horizon), installed_skip=int(installed_skip),
                installed_measure=int(installed_measure), decide_by=decide_by)
    legacy = dict(skip=int(installed_skip), measure=int(installed_measure))
    if not extend_settle_enabled():
        return ExtendWindow(rule=WINDOW_LEGACY, **legacy, **base)
    if skip + max(EXTEND_MEASURE_MIN, int(installed_measure)) > decide_by:
        return ExtendWindow(rule=WINDOW_FALLBACK_SHORT, **legacy, **base)
    return ExtendWindow(skip=skip, measure=EXTEND_MEASURE_MIN, rule=WINDOW_SETTLED, **base)


def settle_measure(t_first: float, *, skip: int, decide_by: int) -> tuple:
    secs = _env_float("SN56_A0_EXTEND_MEASURE_SECS", EXTEND_MEASURE_SECS)
    n = EXTEND_MEASURE_MIN
    if t_first and t_first > 0 and math.isfinite(t_first) and secs > 0:
        n = max(n, int(math.ceil(secs / float(t_first))))
    n = _round_up(n, EXTEND_CYCLE)
    room = max(EXTEND_MEASURE_MIN, int(decide_by) - int(skip))
    return (min(n, room), n > room)


def _warmup_steps_of(args, horizon: int) -> int:
    try:
        return int(args.get_warmup_steps(int(horizon)))
    except Exception:
        w = getattr(args, "warmup_steps", 0) or 0
        return int(w) if float(w) >= 1 else int(math.ceil(float(w) * max(0, int(horizon))))


def skip_steps() -> int:
    return _env_int("SN56_TRAIN_STEP_CLOCK_SKIP", SKIP_STEPS, 1)


def measure_steps() -> int:
    return _env_int("SN56_TRAIN_STEP_CLOCK_STEPS", MEASURE_STEPS, 1)


def max_epochs_for(n_train: int) -> float:
    return 4.0 if n_train < 10000 else 3.0


class EpochPlan(BaseModel):
    source: str
    optimal_epochs: float
    warmup: Union[int, float]
    eval_steps: float
    new_total_steps: int
    steps_per_epoch: int
    achievable_steps: int
    remaining_secs: float
    t_per_step: float
    max_epochs: float
    steps_done: int = 0
    guard_secs: float = 0.0
    below_one_epoch: bool = False


def plan_epochs(*, remaining_secs: float, t_per_step: float, steps_per_epoch: int, n_train: int,
                warmup_steps_arg: int, eval_steps_arg: float, is_final_run: bool,
                source: str = SOURCE_SEARCH, steps_done: int = 0,
                guard_secs: float = 0.0) -> Optional[EpochPlan]:
    if steps_per_epoch <= 0 or not t_per_step or t_per_step <= 0:
        return None
    max_epochs = max_epochs_for(n_train)
    if source == SOURCE_SEARCH:
        achievable_steps = int(remaining_secs * EXHAUST_FRAC / t_per_step)
        optimal_epochs = round(max(SEARCH_MIN_EPOCHS,
                                   min(max_epochs, SEARCH_OVERSHOOT * achievable_steps / steps_per_epoch)), 2)
        new_total_steps = int(steps_per_epoch * optimal_epochs)
        eval_basis = steps_per_epoch
    else:
        avail = max(0.0, float(remaining_secs) - float(guard_secs))
        achievable_steps = int(steps_done) + int(avail * EXHAUST_FRAC / t_per_step)
        new_total_steps = max(int(steps_done) + 1,
                              min(int(steps_per_epoch * max_epochs), achievable_steps))
        optimal_epochs = round(new_total_steps / steps_per_epoch, 4)
        eval_basis = min(steps_per_epoch, new_total_steps)
    warmup = min(warmup_steps_arg, max(10, int(0.03 * new_total_steps)))
    eval_steps = float(eval_steps_arg)
    if is_final_run:
        eval_steps = float(max(20, eval_basis // 8))
    return EpochPlan(source=source, optimal_epochs=optimal_epochs, warmup=warmup,
                     eval_steps=eval_steps, new_total_steps=int(new_total_steps),
                     steps_per_epoch=int(steps_per_epoch), achievable_steps=int(achievable_steps),
                     remaining_secs=float(remaining_secs), t_per_step=float(t_per_step),
                     max_epochs=float(max_epochs), steps_done=int(steps_done),
                     guard_secs=float(guard_secs),
                     below_one_epoch=bool(new_total_steps < steps_per_epoch))


def steps_fitting(*, avail_secs: float, t_step: float, t_point: float, eval_steps: int,
                  start: int) -> int:
    eval_steps = max(1, int(eval_steps or 1))

    def cost(h: int) -> float:
        n_points = (h // eval_steps) - (start // eval_steps)
        if h % eval_steps:
            n_points += 1
        return (h - start) * t_step + n_points * t_point

    lo = start + 1
    if cost(lo) > avail_secs:
        return lo
    hi = lo
    while cost(hi * 2) <= avail_secs and hi < 10 ** 9:
        hi *= 2
    hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= avail_secs:
            lo = mid
        else:
            hi = mid - 1
    return lo


def lambda_scheduler(sched):
    for _ in range(6):
        if sched is None or hasattr(sched, "lr_lambdas"):
            break
        sched = getattr(sched, "scheduler", None) or getattr(sched, "lr_scheduler", None)
    return sched if (sched is not None and hasattr(sched, "lr_lambdas")) else None


def reshape_schedule(sched, *, horizon: int, warmup: int, decay_start: Optional[int] = None) -> dict:
    lam = lambda_scheduler(sched)
    if lam is None:
        return {"applied": False, "reason": "no_lambda_scheduler"}
    new = []
    before = None
    step = int(lam.last_epoch)
    wsd = None
    cos = None
    for f in lam.lr_lambdas:
        kw = getattr(f, "keywords", None)
        is_wsd = bool(isinstance(f, functools.partial) and kw
                      and "num_stable_steps" in kw and "num_decay_steps" in kw)
        is_cos = bool(not is_wsd and a0_cosine.is_cosine_a0(f))
        if not isinstance(f, functools.partial) or not kw or (
                "num_training_steps" not in kw and not is_wsd):
            return {"applied": False,
                    "reason": f"unsupported_lambda:{getattr(getattr(f, 'func', f), '__name__', type(f).__name__)}"}
        if decay_start is not None and not is_wsd:
            return {"applied": False, "reason": "wsd_only"}
        before = before or dict(kw)
        kw = dict(kw)
        if is_wsd:
            w, s, d, cont = wsd_a0.repoint(kw, step=step, horizon=int(horizon), warmup=int(warmup),
                                           frac_now=float(f(step)), decay_start=decay_start)
            wsd = {"stable_before": int(kw["num_stable_steps"]),
                   "decay_before": int(kw["num_decay_steps"]),
                   "stable_after": s, "decay_after": d, "decay_start_after": w + s,
                   "continuous": cont}
            kw["num_warmup_steps"], kw["num_stable_steps"], kw["num_decay_steps"] = w, s, d
        elif is_cos:
            kw, rec = a0_cosine.repoint(kw, step=step, horizon=int(horizon), warmup=int(warmup))
            cos = rec.model_dump()
        else:
            kw["num_training_steps"] = int(horizon)
            if "num_warmup_steps" in kw:
                kw["num_warmup_steps"] = int(warmup)
        new.append(functools.partial(f.func, *f.args, **kw))
    frac_before = float(lam.lr_lambdas[0](step))
    lam.lr_lambdas = new
    lrs = [float(base) * float(f(step)) for base, f in zip(lam.base_lrs, new)]
    for group, lr in zip(lam.optimizer.param_groups, lrs):
        group["lr"] = lr
    lam._last_lr = list(lrs)
    h_before = (int(before.get("num_training_steps", 0)) if wsd is None else
                int(before.get("num_warmup_steps", 0) or 0) + int(before["num_stable_steps"])
                + int(before["num_decay_steps"]))
    out = {"applied": True, "reason": "", "family": getattr(new[0].func, "__name__", ""),
            "step": step, "horizon_before": h_before,
            "warmup_before": int(before.get("num_warmup_steps", 0) or 0),
            "horizon_after": int(horizon), "warmup_after": int(warmup),
            "lr_frac_before": round(frac_before, 8), "lr_frac_now": round(float(new[0](step)), 8),
            "lr_frac_at_horizon": round(float(new[0](int(horizon))), 8),
            "lr_now": lrs[0] if lrs else None}
    if wsd is not None:
        out["wsd"] = wsd
    if cos is not None:
        out["cosine_a0"] = cos
    return out


def _multi() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1)


def _rank() -> int:
    return int(torch.distributed.get_rank()) if _multi() else 0


def _gather(v):
    if not _multi():
        return [v]
    out = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(out, v)
    return out


def _from_rank0(v):
    if not _multi():
        return v
    box = [v if _rank() == 0 else None]
    torch.distributed.broadcast_object_list(box, src=0)
    return box[0]


def _log(log, msg: str) -> None:
    if _rank() == 0:
        log(msg)


def _device_sync() -> None:
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception:
        pass


def remaining_secs_until(end_time: str) -> float:
    end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=datetime.timezone.utc)
    return (end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()


class TrainStepClockCallback(TrainerCallback):

    _clock = staticmethod(time.time)

    def __init__(self, *, source: str, end_time: str, n_train_fn: Callable[[], int],
                 is_final_run: bool, eval_save_cb=None, when_to_eval_handler=None,
                 cadence_horizon_fraction: float = 0.5, restate_gate=None, restate_cadence=None,
                 skip: Optional[int] = None, measure: Optional[int] = None,
                 log: Callable[[str], None] = print,
                 remaining_fn: Optional[Callable[[], float]] = None,
                 extend_only: bool = False, apply_gate=None,
                 remeasure_at: Optional[int] = None, depth=None) -> None:
        self.source = source
        self.end_time = end_time
        self.n_train_fn = n_train_fn
        self.is_final_run = bool(is_final_run)
        self.eval_save_cb = eval_save_cb
        self.handler = when_to_eval_handler
        self.cadence_frac = float(cadence_horizon_fraction)
        self.restate_gate = restate_gate
        self.restate_cadence = restate_cadence
        self.skip = skip_steps() if skip is None else int(skip)
        self.measure = measure_steps() if measure is None else int(measure)
        self.log = log
        self.remaining_fn = remaining_fn or (lambda: remaining_secs_until(self.end_time))
        self.loop_max_steps: Optional[int] = None
        self.extend_only = bool(extend_only)
        self.apply_gate = apply_gate
        self.window: Optional[ExtendWindow] = None
        self.loop_ceiling: Optional[int] = None
        self.t_per_step: Optional[float] = None
        self.plan: Optional[EpochPlan] = None
        self.horizon: Optional[int] = None
        self.repriced = False
        self.watch: Optional[EpochPlan] = None
        self._w_sum = 0.0
        self._w_n = 0
        self._t_prev: Optional[float] = None
        self._busy = False
        self._free_sum = 0.0
        self._n_free = 0
        self._n_excl = 0
        self._all_sum = 0.0
        self._point_pending = False
        self.remeasure_at = (int(remeasure_at) if (remeasure_at and extend_only
                                                   and int(remeasure_at) > self.skip + self.measure)
                             else None)
        self._remeasured = False
        self._r_prev: Optional[float] = None
        self._r_busy = False
        self._r_free_sum = 0.0
        self._r_n_free = 0
        self._r_n_excl = 0
        self.depth = depth

    def _depth_locked(self) -> bool:
        f = getattr(self.depth, "locked", None)
        return bool(f()) if callable(f) else False

    def _depth_live(self):
        d = self.depth
        return None if (d is None or getattr(d, "disarmed", False)) else d

    def on_train_begin(self, args, state, control, **kwargs):
        self.loop_max_steps = int(getattr(state, "max_steps", 0) or 0)
        if self.extend_only:
            self.loop_ceiling = self._epoch_loop_ceiling(args, state, kwargs.get("train_dataloader"))
            self.window = extend_window(
                installed_skip=self.skip, installed_measure=self.measure,
                warmup=_warmup_steps_of(args, self.loop_max_steps), horizon=self.loop_max_steps)
            self.skip, self.measure = self.window.skip, self.window.measure
            r = getattr(self, "remeasure_at", None)
            if r is not None and r <= self.skip + self.measure:
                self.remeasure_at = None

    def on_evaluate(self, args, state, control, **kwargs):
        self._busy = True
        self._r_busy = True
        if (self.plan is not None or self.watch is not None) and not self.repriced:
            self._point_pending = True

    def on_save(self, args, state, control, **kwargs):
        self._busy = True
        self._r_busy = True

    def on_step_end(self, args, state, control, **kwargs):
        g = int(state.global_step)
        start, stop = self.skip, self.skip + self.measure
        if self.plan is None and self.t_per_step is None:
            if g == start:
                _device_sync()
                self._t_prev = self._clock()
                self._busy = False
                return control
            if start < g <= stop and self._t_prev is not None:
                if g == stop:
                    _device_sync()
                now = self._clock()
                dt = now - self._t_prev
                self._all_sum += dt
                if self._busy:
                    self._n_excl += 1
                else:
                    self._free_sum += dt
                    self._n_free += 1
                self._t_prev = now
                self._busy = False
                if g == stop and self._settle_window(g):
                    return control
                if g == stop:
                    self._plan_and_apply(args, state, control, kwargs.get("lr_scheduler"))
                    self._t_prev = self._clock()
                    if self.remeasure_at is not None:
                        self._r_prev = self._t_prev
                        self._r_busy = False
            return control
        if self.remeasure_at is not None and not self._remeasured and self._r_prev is not None:
            self._remeasure_step(args, state, control, g, kwargs.get("lr_scheduler"))
        armed = self.plan is not None or self.watch is not None
        if (self.watch is not None and not self.repriced and not self._point_pending
                and not self._busy and self._t_prev is not None):
            self._w_sum += self._clock() - self._t_prev
            self._w_n += 1
        if armed and not self.repriced and self._point_pending:
            self._point_pending = False
            self._reprice(args, state, kwargs.get("lr_scheduler"))
        if armed:
            self._t_prev = self._clock()
            self._busy = False
        return control

    def _settle_window(self, g: int) -> bool:
        w = self.window
        if w is None or w.rule != WINDOW_SETTLED or w.t_first is not None:
            return False
        per_rank = _gather(self._local_t())
        ts = [v for v in per_rank if v is not None and math.isfinite(v)]
        t = max(ts) if (ts and len(ts) == len(per_rank)) else None
        n, clipped = settle_measure(t, skip=self.skip, decide_by=w.decide_by)
        w.t_first = (round(t, 5) if t else 0.0)
        w.clipped = bool(clipped)
        w.measure = int(n)
        if n <= self.measure:
            return False
        self.measure = int(n)
        r = getattr(self, "remeasure_at", None)
        if r is not None and r <= self.skip + self.measure:
            self.remeasure_at = None
        return True

    def _local_t(self) -> Optional[float]:
        if self._n_free > 0:
            return self._free_sum / self._n_free
        n = self._n_free + self._n_excl
        return (self._all_sum / n) if n > 0 else None

    def _guard_secs(self) -> float:
        h = self.handler
        return float(getattr(h, "save_before_remaining_time", 0) or 0) * 60.0 if h is not None else 0.0

    def _steps_per_epoch(self, args) -> int:
        eff = max(1, int(args.per_device_train_batch_size) * int(args.gradient_accumulation_steps)
                  * int(getattr(args, "world_size", 1) or 1))
        return int(self.n_train_fn()) // eff

    def _epoch_loop_ceiling(self, args, state, dataloader) -> int:
        cur = int(getattr(state, "max_steps", 0) or 0)
        if int(getattr(args, "max_steps", -1) or -1) > 0:
            return cur
        try:
            n = len(dataloader)
        except Exception:
            return cur
        accum = max(1, int(args.gradient_accumulation_steps))
        upe = max(1, n // accum + int(n % accum > 0))
        if self.depth is not None:
            return max(cur, int(math.ceil(float(self.depth.cap))) * upe)
        return max(cur, int(math.ceil(float(args.num_train_epochs))) * upe)

    def _remeasure_step(self, args, state, control, g: int, lr_scheduler) -> None:
        if g == self.remeasure_at:
            _device_sync()
        now = self._clock()
        dt = now - self._r_prev
        if self._r_busy:
            self._r_n_excl += 1
        else:
            self._r_free_sum += dt
            self._r_n_free += 1
        self._r_prev = now
        self._r_busy = False
        if g < self.remeasure_at:
            return
        self._remeasured = True
        local = (self._r_free_sum / self._r_n_free) if self._r_n_free > 0 else None
        horizon = int(getattr(state, "max_steps", 0) or 0)
        if horizon > 0 and g * 10 > horizon:
            return
        prev = self.plan
        self._plan_and_apply(args, state, control, lr_scheduler, local_t=local, stage="remeasure")
        if self.plan is None:
            self.plan = prev
        self._t_prev = self._clock()
        self._busy = False

    def _plan_and_apply(self, args, state, control, lr_scheduler,
                        local_t: Optional[float] = None, stage: str = "plan") -> None:
        g = int(state.global_step)
        local = self._local_t() if local_t is None else local_t
        extra = {} if stage == "plan" else {"stage": stage}
        per_rank = _gather({"t": local, "n_free": self._n_free, "n_excl": self._n_excl}
                           if stage == "plan" else
                           {"t": local, "n_free": self._r_n_free, "n_excl": self._r_n_excl})
        ts = [p["t"] for p in per_rank if p and p.get("t") is not None and math.isfinite(p["t"])]
        t_step = max(ts) if (ts and len(ts) == len(per_rank)) else None
        plan = None
        if _rank() == 0 and t_step is not None and t_step > 0:
            try:
                n_train = int(self.n_train_fn())
                remaining = float(self.remaining_fn())
                legacy = plan_epochs(
                    remaining_secs=remaining, t_per_step=float(t_step),
                    steps_per_epoch=self._steps_per_epoch(args), n_train=n_train,
                    warmup_steps_arg=int(getattr(args, "warmup_steps", 0) or 0),
                    eval_steps_arg=float(getattr(args, "eval_steps", 0) or 0),
                    is_final_run=self.is_final_run, source=self.source, steps_done=g,
                    guard_secs=self._guard_secs())
                if self._depth_live() is not None:
                    plan = self.depth.plan(
                        t_step=float(t_step), g=g, args=args, state=state,
                        remaining=remaining,
                        steps_per_epoch=self._steps_per_epoch(args),
                        warmup_steps_arg=int(getattr(args, "warmup_steps", 0) or 0),
                        eval_steps_arg=float(getattr(args, "eval_steps", 0) or 0),
                        legacy_plan=legacy)
                else:
                    plan = legacy
            except Exception:
                plan = None
        gate = None
        if self.apply_gate is not None and _rank() == 0 and plan is not None:
            try:
                ok, info = self.apply_gate(float(t_step), plan, g, state)
                gate = {"ok": bool(ok), "info": dict(info or {})}
            except Exception as exc:
                gate = {"ok": False, "info": {"error": type(exc).__name__}}
        box = _from_rank0({"plan": plan.model_dump(), "gate": gate} if plan is not None else None)
        self.t_per_step = t_step
        if box is None:
            return
        plan = EpochPlan(**box["plan"])
        gate = box.get("gate")
        if self.apply_gate is not None:
            info = dict((gate or {}).get("info") or {})
            ok = bool((gate or {}).get("ok"))
            if not ok:
                _log(self.log, f"[sn56][relogio] extensão do a0 recusada: {info}")
                return
        self.plan = plan
        if not self._apply(args, state, control, lr_scheduler, plan, **extra):
            self.plan = None

    def _apply(self, args, state, control, lr_scheduler, plan: EpochPlan, **extra) -> bool:
        g = int(state.global_step)
        prior_epochs = float(args.num_train_epochs)
        if self.extend_only:
            loop_max = int(self.loop_ceiling or self.loop_max_steps or getattr(state, "max_steps", 0) or 0)
        else:
            loop_max = int(self.loop_max_steps or getattr(state, "max_steps", 0) or 0)
        horizon = int(plan.new_total_steps)
        clamped_to_loop = bool(loop_max > 0 and horizon > loop_max)
        if clamped_to_loop:
            horizon = loop_max
        horizon = max(g + 1, horizon)
        current = int(getattr(state, "max_steps", 0) or 0)
        _d = self._depth_live()
        _skip = (horizon <= current if _d is None else
                 (horizon == current or self._depth_locked()
                  or (horizon < current and not _d.resolvable_at(g, lr_scheduler))))
        if self.extend_only and _skip:
            _sf = bool(_d is None and shrink_enabled())
            _, _ = self._steps_overrun(plan, g, current) if _sf else (False, None)
            _log(self.log, f"[sn56][relogio] plano medido {plan.new_total_steps} <= horizonte "
                           f"{current}: sem extensão")
            if _sf or (_d is not None and shrink_enabled()):
                self.watch = plan
                self.horizon = current
            return False
        epochs = round(horizon / max(1, plan.steps_per_epoch), 4) if clamped_to_loop else plan.optimal_epochs
        args.num_train_epochs = epochs
        args.warmup_steps = int(plan.warmup)
        if self.is_final_run:
            args.eval_steps = int(plan.eval_steps)
            args.save_steps = int(plan.eval_steps)
        horizon_before = int(getattr(state, "max_steps", 0) or 0)
        state.max_steps = horizon
        state.num_train_epochs = int(math.ceil(epochs))
        state.compute_steps(args, horizon)
        self.horizon = horizon
        lr = reshape_schedule(lr_scheduler, horizon=horizon,
                              warmup=int(args.get_warmup_steps(horizon)))
        _log(self.log,
            f"[sn56][relogio] Epochs {prior_epochs} -> {epochs} "
            f"(t_per_step={plan.t_per_step:.3f}s, remaining={plan.remaining_secs:.0f}s, "
            f"achievable={plan.achievable_steps}, steps/epoch={plan.steps_per_epoch}, "
            f"warmup={plan.warmup}, eval_steps={int(args.eval_steps or 0)}) "
            f"fonte={plan.source} passo={g} horizonte {horizon_before} -> {horizon}"
            f"{' (teto do loop)' if clamped_to_loop else ''}, lr {lr.get('reason') or 'reformatado'}")
        if self.eval_save_cb is not None and self.restate_gate is not None:
            try:
                gate = self.restate_gate(self.eval_save_cb, horizon, args.num_train_epochs)
                if self.handler is not None and self.restate_cadence is not None:
                    hf_eval = (int(args.eval_steps)
                               if str(getattr(args, "eval_strategy", "no")).lower().endswith("steps")
                               else 0)
                    cad = self.restate_cadence(
                        self.handler, plan.steps_per_epoch, gate_step=gate["gate_step_after"],
                        horizon=int(self.cadence_frac * min(plan.achievable_steps, horizon)),
                        total_steps=horizon, hf_eval_steps=hf_eval)
            except Exception:
                pass
        self._agree(args, state, extra.get("stage", "plan"))
        return True

    def _steps_overrun(self, plan: EpochPlan, g: int, current: int) -> tuple:
        if not shrink_enabled() or not plan.t_per_step or plan.t_per_step <= 0:
            return False, None
        avail = float(plan.remaining_secs) - float(plan.guard_secs)
        fit = steps_fitting(avail_secs=avail, t_step=float(plan.t_per_step), t_point=0.0,
                            eval_steps=1, start=int(g))
        return bool(current > fit), int(fit)

    def _agree(self, args, state, stage: str) -> None:
        diff = rank_agreement.agree(f"a0_train_step_clock_{stage}",
                                    rank_agreement.plan_tuple(args, state))
        if diff:
            raise rank_agreement.RankDisagreement(
                f"a0 train-step clock ({stage}): ranks disagree on {sorted(diff)}: {diff}")

    def _reprice(self, args, state, lr_scheduler) -> None:
        self.repriced = True
        watch = self.plan is None
        g = int(state.global_step)
        now = self._clock()
        local = (now - self._t_prev) if self._t_prev is not None else None
        t_step = self.t_per_step
        if watch:
            w_local = (self._w_sum / self._w_n) if self._w_n > 0 else None
            per_rank = _gather((local, w_local))
            ws = [p[1] for p in per_rank if p and p[1] is not None and math.isfinite(p[1])]
            if ws and len(ws) == len(per_rank) and max(ws) > 0:
                t_step = max(ws)
            per_rank = [p[0] if p else None for p in per_rank]
        else:
            per_rank = _gather(local)
        vals = [v for v in per_rank if v is not None and math.isfinite(v)]
        if len(vals) != len(per_rank) or not t_step:
            return
        t_point = max(0.0, max(vals) - float(t_step))
        decision = None
        if _rank() == 0:
            try:
                dinfo = None
                if self._depth_live() is not None:
                    guard_before = self._guard_secs()
                    guard = guard_before
                    if self.handler is not None and shrink_enabled():
                        guard = guard_priced_secs(guard_before, float(t_step), t_point)
                    raised = guard > guard_before
                    fit, dinfo = self.depth.reprice_fit(
                        g=g, remaining=float(self.remaining_fn()), t_step=float(t_step),
                        t_point=t_point, args=args, state=state,
                        legacy_avail=float(self.remaining_fn()) - guard)
                    avail = float(dinfo["avail_secs"])
                    h_now = int(self.horizon or 0)
                    new_h = fit if (fit < h_now or (fit > h_now
                                                    and self.depth.resolvable_at(g, lr_scheduler))) else None
                    if watch:
                        new_h = (fit if (dinfo.get("reason") == "legacy_overrun" and fit < h_now)
                                 else None)
                    if self._depth_locked():
                        new_h = None
                    decision = {"new_h": new_h, "guard": (guard if raised else None)}
                else:
                    guard_before = self._guard_secs()
                    guard = guard_before
                    if self.handler is not None and shrink_enabled():
                        guard = guard_priced_secs(guard_before, float(t_step), t_point)
                    avail = float(self.remaining_fn()) - guard
                    fit = steps_fitting(avail_secs=avail, t_step=float(t_step),
                                        t_point=t_point, eval_steps=int(state.eval_steps or 0), start=g)
                    new_h = fit if fit < int(self.horizon or 0) else None
                    raised = guard > guard_before
                    decision = {"new_h": new_h, "guard": (guard if raised else None)}
            except Exception:
                decision = None
        decision = _from_rank0(decision) or {}
        if decision.get("guard") is not None and self.handler is not None:
            self.handler.save_before_remaining_time = float(decision["guard"]) / 60.0
            _log(self.log, f"[sn56][relogio] guarda de fim {self._guard_secs():.0f}s "
                           f"(passo + ponto de avaliação medidos)")
        new_h = decision.get("new_h")
        if new_h is None:
            return
        horizon_before = int(self.horizon or 0)
        new_h = max(g + 1, int(new_h))
        spe = max(1, int((self.plan or self.watch).steps_per_epoch))
        args.num_train_epochs = round(new_h / spe, 4)
        state.max_steps = new_h
        state.num_train_epochs = int(math.ceil(args.num_train_epochs))
        self.horizon = new_h
        lr = reshape_schedule(lr_scheduler, horizon=new_h,
                              warmup=int(args.get_warmup_steps(new_h)))
        if self.eval_save_cb is not None and self.restate_gate is not None:
            try:
                gate = self.restate_gate(self.eval_save_cb, new_h, args.num_train_epochs)
            except Exception:
                pass
        _log(self.log, f"[sn56][relogio] repreço no 1º ponto de avaliação: t_point={t_point:.2f}s, "
                 f"horizonte {horizon_before} -> {new_h}")
        self._agree(args, state, "reprice")
