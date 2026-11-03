import math
import os
from typing import Optional

from pydantic import BaseModel, Field


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


ENABLED_ENV = "SN56_COMPUTE_AWARE"


def enabled() -> bool:
    return _env_flag(ENABLED_ENV, "0")


S2_ENV = "SN56_COMPUTE_AWARE_S2"


def s2_enabled() -> bool:
    return enabled() and _env_flag(S2_ENV, "0")


MEMBER_FLOOR = 2
DIV_CUT_BELOW = MEMBER_FLOOR + 1
S2_EPOCH_FRAC = 0.40
S2_RUNG_CUT_EPOCHS = 2.0
FIT_MIN_RUNGS = 3
ARGMIN_MIN_RUNGS = 2
S2_RUNG_OFFSET = 0.15


class CutItem(BaseModel):
    item: str
    order: int
    active: bool
    reason: str
    value: Optional[float] = None
    threshold: Optional[float] = None


class Affordance(BaseModel):
    stage: str
    epochs_affordable: Optional[float] = None
    members_affordable: Optional[int] = None
    t_per_step: Optional[float] = None
    budget_secs: Optional[float] = None
    cuts: list[CutItem] = Field(default_factory=list)
    protected: list[str] = Field(default_factory=lambda: ["a0_epochs", "a0_soup_candidates"])
    enabled: bool = True

    def cut_list(self) -> list[str]:
        return [c.item for c in self.cuts if c.active]


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def member_cuts(n_fit: Optional[int], on: Optional[bool] = None) -> list[CutItem]:
    on = enabled() if on is None else bool(on)
    m = int(n_fit) if n_fit is not None else None
    div_on = bool(on and m is not None and m < DIV_CUT_BELOW)
    seed_on = bool(on and m is not None and m < MEMBER_FLOOR)
    return [
        CutItem(item="div_batch_members", order=1, active=div_on,
                reason=(f"members_affordable {m} < {DIV_CUT_BELOW}: the slot past the head goes to "
                        "seed/LR members" if div_on else
                        "disabled" if not on else
                        "unmeasured" if m is None else f"members_affordable {m} >= {DIV_CUT_BELOW}"),
                value=m, threshold=DIV_CUT_BELOW),
        CutItem(item="extra_seeds", order=2, active=seed_on,
                reason=(f"members_affordable {m} < {MEMBER_FLOOR}: head = t_seed only "
                        "(cold-LR member needs n_fit >= 2)" if seed_on else
                        "disabled" if not on else
                        "unmeasured" if m is None else f"members_affordable {m} >= {MEMBER_FLOOR}"),
                value=m, threshold=MEMBER_FLOOR),
    ]


def post_a0_affordance(n_fit: Optional[int], t_step: Optional[float] = None,
                       remaining: Optional[float] = None,
                       epochs_affordable: Optional[float] = None,
                       on: Optional[bool] = None) -> Affordance:
    on = enabled() if on is None else bool(on)
    return Affordance(stage="post_a0", members_affordable=(int(n_fit) if n_fit is not None else None),
                      epochs_affordable=epochs_affordable, t_per_step=t_step,
                      budget_secs=remaining, cuts=member_cuts(n_fit, on=on), enabled=on)


def div_kinds_cut(kinds: list, eligible: list, cut_active: bool,
                  is_div) -> tuple[list, list, bool]:
    if not cut_active:
        return list(eligible), [], False
    keep = [k for k in eligible if not is_div(k)]
    cut = [k for k in eligible if is_div(k)]
    if not cut:
        return list(eligible), [], False
    if not keep:
        return list(eligible), [], True
    return keep, cut, False


class S2Plan(BaseModel):
    mode: str
    n_rungs: int = 0
    steps: int = 0
    offsets: list[float] = Field(default_factory=list)
    reason: str = ""
    steps_clock: int = 0
    horizon_examples: Optional[int] = None
    floor: int = 0
    max_rungs: int = 3
    min_rungs: int = FIT_MIN_RUNGS
    time_left: Optional[float] = None
    t_per_step: Optional[float] = None
    share_secs: Optional[float] = None
    epochs_affordable: Optional[float] = None
    side: int = 0
    changed: bool = False


def examples_horizon(n_rows: Optional[int], rows_per_step: Optional[int], floor: int,
                     max_steps: int, frac: float = S2_EPOCH_FRAC) -> Optional[int]:
    if not n_rows or not rows_per_step or not frac or frac <= 0:
        return None
    ex = int(math.ceil(float(frac) * float(n_rows) / max(1, int(rows_per_step))))
    return max(int(floor), min(int(max_steps), ex))


def nearer_edge_side(center: float, lo: float, hi: float) -> int:
    return -1 if (center - lo) <= (hi - center) else 1


def s2_offsets(n_rungs: int, side: int, base: tuple = (-S2_RUNG_OFFSET, 0.0, S2_RUNG_OFFSET)) -> list:
    if n_rungs >= 3:
        return list(base)
    if n_rungs == 2:
        return [0.0, (S2_RUNG_OFFSET if side > 0 else -S2_RUNG_OFFSET)]
    return [0.0][:max(0, n_rungs)]


def plan_s2(time_left: float, t_per_step: float, floor: int, max_steps: int,
            clock_share: float, horizon: Optional[int], epochs_affordable: Optional[float],
            min_rungs: int = FIT_MIN_RUNGS, side: int = -1, on: Optional[bool] = None,
            base_rungs: int = 3) -> S2Plan:
    on = s2_enabled() if on is None else bool(on)
    t = max(float(t_per_step), 1e-9)
    share = float(time_left) * float(clock_share)
    steps_clock = min(int(max_steps), int(share / ((base_rungs + 1) * t)))
    common = dict(steps_clock=steps_clock, horizon_examples=horizon, floor=int(floor),
                  min_rungs=int(min_rungs), time_left=round(float(time_left), 1),
                  t_per_step=round(float(t_per_step), 5), share_secs=round(share, 1),
                  epochs_affordable=(round(float(epochs_affordable), 4)
                                     if _finite(epochs_affordable) else None))
    today = (S2Plan(mode="clock", n_rungs=base_rungs, steps=steps_clock,
                    offsets=s2_offsets(base_rungs, side), reason="clock_depth", **common)
             if steps_clock >= floor else
             S2Plan(mode="skip", reason="shallower_than_s1", **common))
    if not on:
        today.reason = today.reason + "|disabled"
        return today
    e_cut = bool(_finite(epochs_affordable) and float(epochs_affordable) < S2_RUNG_CUT_EPOCHS)
    max_rungs = min(base_rungs, 2) if e_cut else base_rungs
    reaches = horizon is None or steps_clock >= int(horizon)
    if not e_cut and reaches:
        today.max_rungs = max_rungs
        return today
    why = []
    if e_cut:
        why.append(f"epochs_affordable {float(epochs_affordable):.2f} < {S2_RUNG_CUT_EPOCHS:g}")
    if not reaches:
        why.append(f"clock depth {steps_clock} < examples horizon {horizon}")
    for n in range(max_rungs, 0, -1):
        if n < int(min_rungs):
            break
        if horizon is not None:
            depth = int(horizon)
            fits = (n + 1) * depth * t <= share
        else:
            depth = min(int(max_steps), int(share / ((n + 1) * t)))
            fits = depth >= floor
        if fits and depth >= floor:
            p = S2Plan(mode="held", n_rungs=n, steps=depth, offsets=s2_offsets(n, side),
                       reason="; ".join(why) + f" -> {n} rungs @ {depth}", max_rungs=max_rungs,
                       side=(side if n == 2 else 0), **common)
            p.changed = (p.mode != "clock" and
                         (today.mode, today.n_rungs, today.steps) != ("clock", p.n_rungs, p.steps))
            return p
    p = S2Plan(mode="skip", reason="; ".join(why) + (
        f" -> no readable tier (selector needs >= {min_rungs} rungs)"), max_rungs=max_rungs, **common)
    p.changed = today.mode != "skip"
    return p


def search_affordance(epochs_affordable: Optional[float], t_per_step: Optional[float],
                      budget_secs: Optional[float], s2: Optional[S2Plan] = None,
                      on: Optional[bool] = None) -> Affordance:
    on = s2_enabled() if on is None else bool(on)
    cuts = []
    if s2 is not None:
        cut = bool(on and (s2.mode == "skip" and s2.changed or
                           (s2.mode == "held" and s2.n_rungs < 3)))
        cuts.append(CutItem(item="s2_rungs", order=3, active=cut,
                            reason=(s2.reason if cut else
                                    "disabled" if not on else f"{s2.mode}: {s2.reason}"),
                            value=(float(epochs_affordable) if _finite(epochs_affordable) else None),
                            threshold=S2_RUNG_CUT_EPOCHS))
    return Affordance(stage="search",
                      epochs_affordable=(float(epochs_affordable) if _finite(epochs_affordable)
                                         else None),
                      t_per_step=t_per_step, budget_secs=budget_secs, cuts=cuts, enabled=on)
