import functools
import json
import math
import types
from typing import Optional

from pydantic import BaseModel
from torch.optim.lr_scheduler import LambdaLR

import phi_rtn_cosine
import release_defaults

FAMILY = "cosine_a0"
NUM_CYCLES = 0.5

_FALSY = ("0", "false", "no", "off")


def enabled() -> bool:
    if phi_rtn_cosine.armed():
        return True
    return release_defaults.raw("SN56_A0_COSINE").strip().lower() not in _FALSY


def progress(current_step: int, *, num_warmup_steps: int, num_training_steps: int,
             anchor_step: int = -1, anchor_progress: float = 0.0) -> float:
    if int(anchor_step) < 0:
        p = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    else:
        pa = float(anchor_progress)
        p = pa + (1.0 - pa) * (float(current_step - anchor_step)
                               / float(max(1, num_training_steps - anchor_step)))
    return min(1.0, max(0.0, p))


def cosine_a0(current_step: int, *, num_warmup_steps: int, num_training_steps: int,
              min_lr_rate: float = 0.0, anchor_step: int = -1, anchor_progress: float = 0.0):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    p = progress(current_step, num_warmup_steps=num_warmup_steps,
                 num_training_steps=num_training_steps, anchor_step=anchor_step,
                 anchor_progress=anchor_progress)
    factor = 0.5 * (1.0 + math.cos(math.pi * float(NUM_CYCLES) * 2.0 * p))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def make_lambda(horizon: int, warmup: int, min_lr_rate: float) -> functools.partial:
    return functools.partial(cosine_a0, num_warmup_steps=int(warmup),
                             num_training_steps=int(horizon), min_lr_rate=float(min_lr_rate),
                             anchor_step=-1, anchor_progress=0.0)


def is_cosine_a0(lam) -> bool:
    kw = getattr(lam, "keywords", None) or {}
    return bool(isinstance(lam, functools.partial) and "anchor_step" in kw
                and "num_training_steps" in kw)


def describe(lam) -> Optional[dict]:
    if not is_cosine_a0(lam):
        return None
    kw = lam.keywords
    a = int(kw.get("anchor_step", -1))
    return {"family": FAMILY, "warmup": int(kw.get("num_warmup_steps", 0) or 0),
            "horizon": int(kw["num_training_steps"]),
            "min_lr_rate": float(kw.get("min_lr_rate", 0.0) or 0.0),
            "anchor_step": a, "anchor_progress": (float(kw.get("anchor_progress", 0.0)) if a >= 0 else None)}


def min_lr_rate_of(lr_scheduler_kwargs) -> Optional[float]:
    kw = lr_scheduler_kwargs or {}
    if isinstance(kw, str):
        try:
            kw = json.loads(kw)
        except (json.JSONDecodeError, ValueError):
            return None
    try:
        v = kw.get("min_lr_rate")
        return None if v is None else float(v)
    except (AttributeError, TypeError, ValueError):
        return None


class Repoint(BaseModel):
    anchor_before: Optional[dict] = None
    anchor_after: Optional[dict] = None
    progress_at_step: Optional[float] = None
    horizon_after: int
    in_warmup: bool = False
    continuous: bool = True


def _anchor(kw: dict) -> Optional[dict]:
    a = int(kw.get("anchor_step", -1))
    return {"step": a, "progress": float(kw.get("anchor_progress", 0.0))} if a >= 0 else None


def repoint(kw: dict, *, step: int, horizon: int, warmup: int) -> tuple:
    old = dict(kw)
    step, horizon = int(step), int(horizon)
    w0 = int(old.get("num_warmup_steps", 0) or 0)
    lam0 = functools.partial(cosine_a0, **old)
    new = dict(old)
    if step < w0:
        new.update(num_training_steps=horizon, num_warmup_steps=int(warmup), anchor_step=-1,
                   anchor_progress=0.0)
        rec = Repoint(anchor_before=_anchor(old), anchor_after=None, horizon_after=horizon,
                      in_warmup=True,
                      continuous=bool(cosine_a0(step, **new) == lam0(step)))
        return new, rec
    pa = progress(step, num_warmup_steps=w0, num_training_steps=int(old["num_training_steps"]),
                  anchor_step=int(old.get("anchor_step", -1)),
                  anchor_progress=float(old.get("anchor_progress", 0.0)))
    h1 = max(step + 1, horizon)
    if h1 != int(old["num_training_steps"]):
        new.update(num_training_steps=h1, num_warmup_steps=w0, anchor_step=step,
                   anchor_progress=min(1.0, pa))
    rec = Repoint(anchor_before=_anchor(old), anchor_after=_anchor(new), progress_at_step=pa,
                  horizon_after=int(new["num_training_steps"]),
                  continuous=bool(cosine_a0(step, **new) == lam0(step)))
    return new, rec


def install(trainer, *, min_lr_rate: float) -> dict:
    info = {"family": FAMILY, "min_lr_rate": float(min_lr_rate), "num_cycles": NUM_CYCLES}

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            opt = self.optimizer if optimizer is None else optimizer
            warm = int(self.args.get_warmup_steps(num_training_steps))
            lam = make_lambda(int(num_training_steps), warm, float(min_lr_rate))
            self.lr_scheduler = LambdaLR(opt, lam)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    trainer.create_scheduler = types.MethodType(create_scheduler, trainer)
    trainer._a0_cosine = info
    return info
