import os
import time
from typing import Optional

import torch
from pydantic import BaseModel
from transformers import TrainerCallback

import stop_to_decay

CADENCE_DIV = 8
COST_FRAC = 0.02
EVAL_FWD_SHARE = 1.0 / 3.0
T_SNAP_BYTES_PER_SEC = 5e9

_FALSY = ("0", "false", "no", "off")


def enabled() -> bool:
    return os.environ.get("SN56_EARLY_EVALS", "1").strip().lower() not in _FALSY


def cost_frac() -> float:
    try:
        return max(0.0, float(os.environ.get("SN56_EARLY_EVALS_COST_FRAC", COST_FRAC)))
    except (TypeError, ValueError):
        return COST_FRAC


def single_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size() <= 1
    return int(os.environ.get("WORLD_SIZE", "1") or 1) <= 1


def cadence(steps_per_epoch: int) -> int:
    return max(1, int(steps_per_epoch or 0) // CADENCE_DIV)


def points(after: int, gate_step: int, cad: int) -> list:
    if cad <= 0 or gate_step <= 0:
        return []
    s = max(1, (max(0, int(after) - 1) // cad + 1)) * cad
    return list(range(s, int(gate_step), cad))


def unit_prior(*, n_dev: int, rows_per_step: int, t_per_step: Optional[float],
               trainable_bytes: float) -> Optional[float]:
    if not t_per_step or t_per_step <= 0 or n_dev <= 0:
        return None
    return (float(n_dev) / max(1, int(rows_per_step)) * float(t_per_step) * EVAL_FWD_SHARE
            + float(trainable_bytes) / T_SNAP_BYTES_PER_SEC)


def _rng_state() -> dict:
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["cpu"])
    if st.get("cuda") is not None:
        torch.cuda.set_rng_state_all(st["cuda"])


class CostGate(BaseModel):
    budget_secs: float
    frac: float
    spent: float = 0.0
    n_done: int = 0
    unit_prior: Optional[float] = None
    open: bool = True
    reason: str = ""
    cap_secs: Optional[float] = None

    @property
    def cap(self) -> float:
        if self.cap_secs is not None:
            return float(self.cap_secs)
        return float(self.frac) * float(self.budget_secs)

    def unit(self) -> Optional[float]:
        if self.n_done > 0:
            return self.spent / self.n_done
        return self.unit_prior

    def projected(self, n_left: int) -> Optional[float]:
        u = self.unit()
        return None if u is None else self.spent + max(0, int(n_left)) * u

    def check(self, n_left: int) -> bool:
        if not self.open:
            return False
        p = self.projected(n_left)
        if p is None:
            self.open, self.reason = False, "no_unit_estimate"
        elif p > self.cap:
            self.open, self.reason = False, "over_cost_frac"
        return self.open

    def charge(self, secs: float) -> None:
        self.spent += max(0.0, float(secs))
        self.n_done += 1


class EarlyEvals(TrainerCallback):

    _clock = staticmethod(time.perf_counter)

    def __init__(self, *, steps_per_epoch: int, budget_secs: float, unit_prior: Optional[float],
                 frac: Optional[float] = None, t_step_prior: Optional[float] = None,
                 priced: Optional[bool] = None) -> None:
        self.spe = max(1, int(steps_per_epoch or 1))
        self.cad = cadence(self.spe)
        self.gate = CostGate(budget_secs=float(budget_secs or 0.0),
                             frac=cost_frac() if frac is None else float(frac),
                             unit_prior=unit_prior)
        self.active = True
        self.steps: list = []
        self._t0: Optional[float] = None
        self._t0_step: Optional[int] = None
        self._rng: Optional[dict] = None
        self.priced = bool(stop_to_decay.ee_priced_enabled() if priced is None else priced)
        self.member_frac = stop_to_decay.member_eval_frac()
        self.t_step_prior = t_step_prior
        self._ts: Optional[float] = None
        self._step_secs: list = []

    def t_step(self) -> Optional[float]:
        if self._step_secs:
            tail = self._step_secs[-max(1, self.cad):]
            return sum(tail) / len(tail)
        return self.t_step_prior

    def _price(self, gate_step: int) -> None:
        if not self.priced:
            return
        cap = stop_to_decay.early_eval_cap(gate_step=int(gate_step), t_step=self.t_step(),
                                           frac=self.member_frac)
        if cap is not None:
            self.gate.cap_secs = cap

    def decide(self, gate_step: int) -> dict:
        pts = points(self.cad, gate_step, self.cad)
        self._price(gate_step)
        full = self.gate.projected(len(pts))
        ok = bool(pts) and self.gate.check(1)
        if not pts:
            reason = "no_points_before_gate"
        else:
            reason = "" if ok else self.gate.reason
        self.active = bool(ok)
        info = {"ok": bool(ok), "reason": reason, "stage": "plan", "cadence": self.cad,
                "projected_full_prior": (round(full, 2) if full is not None else None),
                "full_fits_prior": (bool(full <= self.gate.cap) if full is not None else None),
                "steps_per_epoch": self.spe, "gate_step": int(gate_step), "n_extra": len(pts),
                "first_step": (pts[0] if pts else None), "unit_prior": self.gate.unit_prior,
                "projected_first": self.gate.projected(1), "cap": round(self.gate.cap, 2),
                "budget_secs": self.gate.budget_secs, "frac": self.gate.frac,
                "priced": self.priced, "member_frac": self.member_frac,
                "t_step": self.t_step(), "priced_cap": self.gate.cap_secs}
        return info

    def allow(self, step: int, gate_step: int) -> bool:
        if not self.active or step <= 0 or step % self.cad:
            return False
        n_left = len(points(step, gate_step, self.cad))
        self._price(gate_step)
        if not self.gate.check(n_left if self.gate.n_done > 0 else 1):
            self.active = False
            return False
        self._rng = _rng_state()
        self._t0 = self._clock()
        self._t0_step = int(step)
        self.steps.append(int(step))
        return True

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._rng is not None:
            _set_rng_state(self._rng)
            self._rng = None
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._ts is not None and self.active:
            self._step_secs.append(max(0.0, self._clock() - self._ts))
            del self._step_secs[:-max(1, self.cad)]
        self._ts = None
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if self._rng is not None:
            _set_rng_state(self._rng)
            self._rng = None
        if self._t0 is not None:
            secs = self._clock() - self._t0
            self.gate.charge(secs)
            self._t0 = None
            self._t0_step = None
        self._ts = self._clock()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.active = False
        return control
