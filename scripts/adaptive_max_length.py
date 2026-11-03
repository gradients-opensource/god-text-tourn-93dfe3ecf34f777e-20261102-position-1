
from typing import Optional

_MIN_MAX_LENGTH = 128

_ALIGNMENT = 64


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def compute_max_length(
    seq_length_distribution: Optional[dict],
    default: int = 2048,
    packing: bool = True,
    model_max_length: Optional[int] = None,
) -> int:
    if seq_length_distribution is None:
        return default

    p99 = seq_length_distribution.get("p99")
    p50 = seq_length_distribution.get("p50")

    if p99 is None or p99 <= 0:
        return default

    if packing:
        target = max(p99, 2 * (p50 or p99))
    else:
        target = p99

    target = int(target * 1.1)
    target = _align_up(target)

    ceiling = default
    if model_max_length is not None and model_max_length > default:
        ceiling = model_max_length

    target = max(_MIN_MAX_LENGTH, min(target, ceiling))

    if target > default:
        increase_pct = (target / default - 1) * 100
        print(
            f"[adaptive_max_length] {target} (p99={p99}, p50={p50}, "
            f"packing={packing}, model_max={model_max_length}) — "
            f"{increase_pct:.0f}% LARGER than default {default} to avoid truncation",
            flush=True,
        )
    elif target < default:
        savings_pct = (1 - target / default) * 100
        print(
            f"[adaptive_max_length] {target} (p99={p99}, p50={p50}, "
            f"packing={packing}) — {savings_pct:.0f}% smaller than default {default}",
            flush=True,
        )
    else:
        print(
            f"[adaptive_max_length] {target} (no change from default {default})",
            flush=True,
        )

    return target
