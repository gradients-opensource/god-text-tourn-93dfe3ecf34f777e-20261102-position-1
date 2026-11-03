import functools
import os
import types
from typing import Optional

from torch.optim.lr_scheduler import LambdaLR


DECAY_FRAC = 0.20
FAMILY = "wsd_a0"

_FALSY = ("0", "false", "no", "off")


def enabled() -> bool:
    return os.environ.get("SN56_WSD", "1").strip().lower() not in _FALSY


def decay_frac() -> float:
    try:
        v = float(os.environ.get("SN56_WSD_DECAY_FRAC", DECAY_FRAC))
    except (TypeError, ValueError):
        return DECAY_FRAC
    return DECAY_FRAC if not (0.0 < v <= 1.0) else v


def wsd_lr_lambda(current_step: int, *, num_warmup_steps: int, num_stable_steps: int,
                  num_decay_steps: int, min_lr_rate: float = 0.0) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        return max(0.0, (1.0 - progress) * (1.0 - min_lr_rate) + min_lr_rate)
    return float(min_lr_rate)


def split(horizon: int, warmup: int, frac: Optional[float] = None) -> tuple:
    frac = decay_frac() if frac is None else float(frac)
    horizon, warmup = max(0, int(horizon)), max(0, int(warmup))
    decay = max(1, int(round(frac * horizon)))
    decay = min(decay, max(1, horizon - warmup))
    stable = max(0, horizon - warmup - decay)
    return stable, decay


def make_lambda(horizon: int, warmup: int, min_lr_rate: float,
                frac: Optional[float] = None) -> functools.partial:
    stable, decay = split(horizon, warmup, frac)
    return functools.partial(wsd_lr_lambda, num_warmup_steps=int(warmup),
                             num_stable_steps=int(stable), num_decay_steps=int(decay),
                             min_lr_rate=float(min_lr_rate))


def describe(lam) -> Optional[dict]:
    kw = getattr(lam, "keywords", None) or {}
    if "num_stable_steps" not in kw or "num_decay_steps" not in kw:
        return None
    w, s, d = (int(kw.get("num_warmup_steps", 0) or 0), int(kw["num_stable_steps"]),
               int(kw["num_decay_steps"]))
    return {"family": getattr(getattr(lam, "func", None), "__name__", ""),
            "warmup": w, "stable": s, "decay": d, "horizon": w + s + d,
            "decay_start": w + s,
            "min_lr_rate": float(kw.get("min_lr_rate", kw.get("min_lr_ratio", 0.0)) or 0.0)}


def install(trainer, *, min_lr_rate: float, frac: Optional[float] = None) -> dict:
    frac = decay_frac() if frac is None else float(frac)
    info = {"decay_frac": frac, "min_lr_rate": float(min_lr_rate)}

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            opt = self.optimizer if optimizer is None else optimizer
            warm = int(self.args.get_warmup_steps(num_training_steps))
            lam = make_lambda(int(num_training_steps), warm, float(min_lr_rate), frac)
            self.lr_scheduler = LambdaLR(opt, lam)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    trainer.create_scheduler = types.MethodType(create_scheduler, trainer)
    trainer._wsd_a0 = info
    return info


def repoint(kw: dict, *, step: int, horizon: int, warmup: int, frac_now: float,
            decay_start: Optional[int] = None) -> tuple:
    if decay_start is not None:
        w0 = int(kw.get("num_warmup_steps", 0) or 0)
        s0 = int(kw["num_stable_steps"])
        flat = bool(max(int(warmup), w0) <= int(step) < w0 + s0 and float(frac_now) >= 1.0 - 1e-9)
        if flat and int(horizon) > int(step):
            ds = max(int(warmup), int(step), min(int(decay_start), int(horizon) - 1))
            return int(warmup), int(ds - int(warmup)), max(1, int(horizon) - ds), True
    w0 = int(kw.get("num_warmup_steps", 0) or 0)
    s0, d0 = int(kw["num_stable_steps"]), int(kw["num_decay_steps"])
    total0 = w0 + s0 + d0
    floor = float(kw.get("min_lr_rate", kw.get("min_lr_ratio", 0.0)) or 0.0)
    frac = (d0 / total0) if total0 > 0 else decay_frac()
    stable, decay = split(horizon, warmup, frac)
    if step < warmup or (step < w0 + s0 and step < warmup + stable):
        return int(warmup), int(stable), int(decay), True
    span = 1.0 - floor
    p = 1.0 if span <= 0 else min(1.0, max(0.0, (1.0 - float(frac_now)) / span))
    if p >= 1.0:
        return int(warmup), max(0, int(step) - int(warmup) - 1), 1, True
    dlen = int(round((int(horizon) - int(step)) / (1.0 - p)))
    start = int(horizon) - dlen
    if start < int(warmup):
        return int(warmup), 0, max(1, int(horizon) - int(warmup)), False
    return int(warmup), int(start - warmup), max(1, dlen), True
