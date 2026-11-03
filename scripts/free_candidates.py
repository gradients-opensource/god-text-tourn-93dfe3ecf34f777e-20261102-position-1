import os
from typing import Optional

from pydantic import BaseModel


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


ON = _flag("SN56_FREE_CANDIDATES", "1")
A0_HELD = _flag("SN56_FREE_A0_HELD", "0")
MEMBER_REJECTS = _flag("SN56_FREE_MEMBER_REJECTS", "1")
NFIT_MAX = _int("SN56_FREE_NFIT_MAX", 8)
MARGIN_SNAPS = _int("SN56_FREE_MARGIN_SNAPS", 1)

SRC_A0 = "a0_held"
SRC_REJECT = "member_reject"


class FreeGate(BaseModel):
    active: bool
    reason: str
    n_fit: Optional[int] = None
    nfit_max: int
    a0_held: bool
    member_rejects: bool

    def sources(self) -> list:
        if not self.active:
            return []
        return ([SRC_A0] if self.a0_held else []) + ([SRC_REJECT] if self.member_rejects else [])


def gate(n_fit: Optional[int], *, path_ok: bool, on: Optional[bool] = None,
         nfit_max: Optional[int] = None, a0_held: Optional[bool] = None,
         member_rejects: Optional[bool] = None) -> FreeGate:
    on = ON if on is None else bool(on)
    nfit_max = NFIT_MAX if nfit_max is None else int(nfit_max)
    a0_held = A0_HELD if a0_held is None else bool(a0_held)
    member_rejects = MEMBER_REJECTS if member_rejects is None else bool(member_rejects)
    kw = dict(n_fit=(int(n_fit) if n_fit is not None else None), nfit_max=nfit_max,
              a0_held=a0_held, member_rejects=member_rejects)
    if not on:
        return FreeGate(active=False, reason="flag_off", **kw)
    if not (a0_held or member_rejects):
        return FreeGate(active=False, reason="no_source", **kw)
    if not path_ok:
        return FreeGate(active=False, reason="not_online_path", **kw)
    if n_fit is None:
        return FreeGate(active=False, reason="n_fit_unknown", **kw)
    if int(n_fit) > nfit_max:
        return FreeGate(active=False, reason="n_fit_high", **kw)
    return FreeGate(active=True, reason="low_affordance", **kw)


def held_keep_count(room: Optional[int], *, n_held: int, snap_bytes: int, numel: int,
                    online_snaps: int, margin_snaps: Optional[int] = None) -> int:
    if room is None or n_held <= 0 or snap_bytes <= 0:
        return 0
    margin = MARGIN_SNAPS if margin_snaps is None else int(margin_snaps)
    need = 4 * int(numel) + (int(online_snaps) + 1 + margin) * int(snap_bytes)
    k = (int(room) + int(n_held) * int(snap_bytes) - need) // int(snap_bytes)
    return int(max(0, min(int(n_held), k)))


def can_hold_reject(room: Optional[int], *, snap_bytes: int, online_snaps: int,
                    margin_snaps: Optional[int] = None) -> bool:
    if room is None or snap_bytes <= 0:
        return False
    margin = MARGIN_SNAPS if margin_snaps is None else int(margin_snaps)
    return int(room) >= (int(online_snaps) + margin) * int(snap_bytes)


def trial_fits(left: Optional[float], per_trial: float, *, tail: float) -> bool:
    if left is None:
        return False
    return float(left) - float(per_trial) >= float(tail)


def entry(state: dict, *, loss: float, step: int, src: str, origin: str, **meta) -> dict:
    return {"loss": float(loss), "step": int(step), "src": src, "state": state,
            "free_src": origin, **meta}


def describe(e: dict) -> dict:
    return {"src": e.get("src"), "step": e.get("step"), "free_src": e.get("free_src"),
            "loss": (round(float(e["loss"]), 6) if isinstance(e.get("loss"), float) else
                     e.get("loss")),
            **{k: e[k] for k in ("soup_n_at_reject", "dev_at_reject") if k in e}}
