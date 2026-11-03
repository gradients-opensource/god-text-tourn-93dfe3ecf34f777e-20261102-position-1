import os
import release_defaults
from typing import Optional

_FALSY = ("0", "false", "no", "off")

MEMBER_MIN_FRAC = 4
LR_BATCH_EXP = 0.25

ACTIVE = False
LR_RESCALE = 1.0


def enabled() -> bool:
    return release_defaults.raw("SN56_PACKED_FILL").strip().lower() not in _FALSY


def arm(packed: bool) -> bool:
    global ACTIVE
    ACTIVE = bool(enabled() and packed)
    return ACTIVE


def active() -> bool:
    return bool(ACTIVE)


EXT_SEARCH_PADDED = "search_padded_to_max_length"
EXT_PACKED_FILL = "packed_fill"


def extension_reason(dyn_pad: bool, packed: bool, deepspeed: bool) -> Optional[str]:
    if dyn_pad and not packed:
        return EXT_SEARCH_PADDED
    if active() and packed and not deepspeed:
        return EXT_PACKED_FILL
    return None


def reset() -> None:
    global ACTIVE, LR_RESCALE
    ACTIVE = False
    LR_RESCALE = 1.0


def member_floor(horizon: int) -> int:
    return max(1, int(horizon or 0) // MEMBER_MIN_FRAC)


def noise_cap_lr_scale(rows_before: Optional[int], rows_after: Optional[int]) -> float:
    try:
        b, a = float(rows_before or 0), float(rows_after or 0)
    except (TypeError, ValueError):
        return 1.0
    if b <= 0 or a <= 0 or a >= b:
        return 1.0
    return (a / b) ** LR_BATCH_EXP


def scaled_argmin(argmin_lr: Optional[float]) -> Optional[float]:
    if argmin_lr is None or not active():
        return argmin_lr
    return float(argmin_lr) * float(LR_RESCALE)
