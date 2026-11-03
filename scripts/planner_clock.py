import math
import os
import statistics
from typing import Optional

from pydantic import BaseModel

EVAL_B1_FACTOR: Optional[float] = None

MIN_POINTS = 1
MIN_FREE_STEPS = 10


class ClockStats(BaseModel):
    t_step: float
    n_free: int
    t_point: float
    t_eval: float
    t_save: float
    t_point_member: float
    t_startup: float
    n_points: int
    n_saves: int
    t_write: Optional[float] = None
    n_writes: int = 0
    save_mode: str = "sync"
    t_eval_b1: Optional[float] = None


def eval_b1_secs(stats: Optional["ClockStats"], fallback: Optional[float] = None) -> Optional[float]:
    if stats is not None and stats.t_eval_b1 is not None and stats.t_eval_b1 == stats.t_eval_b1 \
            and stats.t_eval_b1 > 0:
        if fallback:
            return float(fallback) * float(stats.t_eval_b1) / float(stats.t_eval or stats.t_eval_b1)
        return float(stats.t_eval_b1)
    if fallback is not None:
        return fallback
    return float(stats.t_eval) if stats is not None else None


def write_secs(stats: Optional["ClockStats"]) -> float:
    if stats is None:
        return 0.0
    if stats.t_write is not None and stats.t_write == stats.t_write and stats.t_write >= 0:
        return float(stats.t_write)
    return float(stats.t_save)


class A0Clock:

    def __init__(self) -> None:
        self.t_call: Optional[float] = None
        self.t_begin: Optional[float] = None
        self.t_first_step: Optional[float] = None
        self._prev: Optional[tuple] = None
        self._point_pending = False
        self._t_evaluated: Optional[float] = None
        self.free: list = []
        self.points: list = []
        self.eval_runtimes: list = []
        self.saves: list = []
        self.writes: list = []
        self.eval_b1_factor: Optional[float] = None

    def write(self, secs: float) -> None:
        if secs is not None and secs == secs and secs >= 0:
            self.writes.append(float(secs))

    def train_call(self, t: float) -> None:
        self.t_call = float(t)

    def train_begin(self, t: float) -> None:
        self.t_begin = float(t)
        if self.t_call is None:
            self.t_call = float(t)

    def step_end(self, step: int, t: float) -> None:
        if self.t_first_step is None:
            self.t_first_step = float(t)
        if self._prev is not None and int(step) == self._prev[0] + 1:
            dt = float(t) - self._prev[1]
            if dt > 0:
                (self.points if self._point_pending else self.free).append(dt)
        self._prev = (int(step), float(t))
        self._point_pending = False
        self._t_evaluated = None

    def evaluate(self, t: float, eval_runtime: Optional[float]) -> None:
        self._point_pending = True
        self._t_evaluated = float(t)
        if eval_runtime is not None and eval_runtime == eval_runtime and eval_runtime > 0:
            self.eval_runtimes.append(float(eval_runtime))

    def save(self, t: float) -> None:
        self._point_pending = True
        if self._t_evaluated is not None:
            dt = float(t) - self._t_evaluated
            if dt >= 0:
                self.saves.append(dt)

    def stats(self) -> Optional[ClockStats]:
        if len(self.points) < MIN_POINTS or len(self.free) < MIN_FREE_STEPS:
            return None
        t_step = statistics.fmean(self.free)
        t_point = max(0.0, statistics.median(self.points) - t_step)
        t_eval = statistics.median(self.eval_runtimes) if self.eval_runtimes else t_point
        t_save = statistics.median(self.saves) if self.saves else 0.0
        t_save = min(t_save, t_point)
        t_write = statistics.median(self.writes) if self.writes else t_save
        startup = 0.0
        if self.t_first_step is not None and self.t_call is not None:
            startup = max(0.0, self.t_first_step - self.t_call - t_step)
        vals = (t_step, t_point, t_eval, t_save, startup)
        if not all(math.isfinite(v) for v in vals) or t_step <= 0:
            return None
        return ClockStats(t_step=t_step, n_free=len(self.free), t_point=t_point,
                          t_eval=t_eval, t_save=t_save,
                          t_point_member=max(t_eval, t_point - t_save),
                          t_startup=startup, n_points=len(self.points), n_saves=len(self.saves),
                          t_write=t_write, n_writes=len(self.writes),
                          save_mode="async" if self.writes else "sync",
                          t_eval_b1=(t_eval * float(self.eval_b1_factor)
                                     if self.eval_b1_factor else None))


RESERVE_K = 1.5
RESERVE_FLOOR_SECS = 30.0
TAIL_WRITES = 3
EXIT_FRAC = 0.05
KILL_MARGIN_SECS = 170.0
LEGACY_RESERVE_SECS = 180.0
DEV_C = 6.0
DEVPASS_FLOOR_SECS = 20.0
DEVPASS_F = 5.142
DEVPASS_B = 2.0029
DEVPASS_MARGIN = 1.903


DEVPASS2_F = 2.2858
DEVPASS2_A = 1.2779
DEVPASS2_MARGIN = 1.535


def devpass_floor_on() -> bool:
    return os.environ.get("SN56_ST_DEVPASS_FLOOR", "1").strip().lower() not in ("0", "false", "no", "off")


def devpass_floor_secs(n_dev: Optional[int], fb1: Optional[float], w: float) -> Optional[float]:
    try:
        n, f = int(n_dev or 0), float(fb1) if fb1 is not None else None
    except (TypeError, ValueError):
        return None
    if n <= 0 or f is None or f != f or f <= 0:
        return None
    return max(DEVPASS_FLOOR_SECS, DEVPASS2_MARGIN * (DEVPASS2_F + DEVPASS2_A * n * f) + float(w or 0.0))


def devpass_measured_on() -> bool:
    return os.environ.get("SN56_ST_DEVPASS_MEASURED", "1").strip().lower() not in ("0", "false", "no", "off")


class TailReserve(BaseModel):
    reserve: float
    devpass_reserve: float
    source: str
    devpass_source: str = "k_dev_c"
    devpass_k_dev_c: Optional[float] = None
    devpass_fit: Optional[float] = None
    devpass_floor: Optional[float] = None
    t_save: Optional[float] = None
    t_load: Optional[float] = None
    t_eval: Optional[float] = None
    exit_excess: float = 0.0
    reserve_legacy: float = LEGACY_RESERVE_SECS
    devpass_legacy: float = 0.0


def devpass_hard_on() -> bool:
    return os.environ.get("SN56_MP_DEVPASS_HARD", "1").strip().lower() not in ("0", "false", "no", "off")


def devpass_gate_save(save_secs: float) -> float:
    return max(2.5 * max(0.0, float(save_secs or 0.0)), 20.0)


def tail_reserve(stats: Optional[ClockStats], budget_secs: Optional[float],
                 t_load: Optional[float], devfit_enabled: bool,
                 devpass_legacy: float, reserve_legacy: float = LEGACY_RESERVE_SECS,
                 measured_devpass: bool = False, n_dev: Optional[int] = None,
                 fb1: Optional[float] = None) -> TailReserve:
    if stats is None:
        return TailReserve(reserve=reserve_legacy,
                           devpass_reserve=devpass_legacy if devfit_enabled else 0.0,
                           source="legacy", reserve_legacy=reserve_legacy,
                           devpass_legacy=devpass_legacy)
    load = float(t_load) if (t_load is not None and t_load == t_load and t_load >= 0) else 0.0
    exit_excess = 0.0
    if budget_secs and budget_secs > 0:
        exit_excess = min(reserve_legacy,
                          max(0.0, EXIT_FRAC * float(budget_secs) - KILL_MARGIN_SECS))
    w = write_secs(stats)
    reserve = max(RESERVE_FLOOR_SECS,
                  RESERVE_K * (TAIL_WRITES * w + load) + exit_excess)
    devpass = 0.0
    dp_src, dp_old = "k_dev_c", None
    dp_fit = dp_floor = None
    if devfit_enabled:
        e1 = eval_b1_secs(stats)
        dp_old = max(DEVPASS_FLOOR_SECS, RESERVE_K * (DEV_C * e1 + w))
        devpass = dp_old
        if measured_devpass:
            devpass = max(DEVPASS_FLOOR_SECS, DEVPASS_MARGIN * (DEVPASS_F + DEVPASS_B * e1) + w)
            dp_src = "measured_fit"
            dp_fit = devpass
            if devpass_floor_on():
                dp_floor = devpass_floor_secs(n_dev, fb1, w)
                if dp_floor is not None and dp_floor > devpass:
                    devpass = dp_floor
                    dp_src = "measured_floor"
    if devfit_enabled and devpass > 0 and devpass_hard_on():
        devpass = devpass - w + devpass_gate_save(w)
        dp_src = dp_src + "+gate_save"
    return TailReserve(reserve=reserve, devpass_reserve=devpass, source="measured",
                       devpass_source=dp_src,
                       devpass_k_dev_c=(round(dp_old, 2) if dp_old is not None else None),
                       devpass_fit=(round(dp_fit, 2) if dp_fit is not None else None),
                       devpass_floor=(round(dp_floor, 2) if dp_floor is not None else None),
                       t_save=w, t_load=load, t_eval=stats.t_eval,
                       exit_excess=exit_excess, reserve_legacy=reserve_legacy,
                       devpass_legacy=devpass_legacy)
