import math
from typing import Optional

import torch
from transformers import TrainerCallback


class RankDisagreement(RuntimeError):
    pass


def _multi() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1)


FLOAT_REL_TOL = 1e-6


def _scalar(v):
    if hasattr(v, "value"):
        return str(v.value)
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def plan_tuple(args, state=None, train_dataloader=None) -> dict:
    max_steps = int(getattr(state, "max_steps", 0) or 0) if state is not None else None
    try:
        n_batches = int(len(train_dataloader)) if train_dataloader is not None else None
    except TypeError:
        n_batches = None
    try:
        warmup = int(args.get_warmup_steps(max_steps)) if max_steps else None
    except Exception:
        warmup = None
    out = {
        "micro": getattr(args, "per_device_train_batch_size", None),
        "accum": getattr(args, "gradient_accumulation_steps", None),
        "max_steps": max_steps,
        "batches_per_epoch": n_batches,
        "warmup_steps": warmup,
        "learning_rate": getattr(args, "learning_rate", None),
        "lr_scheduler_type": getattr(args, "lr_scheduler_type", None),
        "seed": getattr(args, "seed", None),
        "data_seed": getattr(args, "data_seed", None),
    }
    return {k: _scalar(v) for k, v in out.items()}


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=FLOAT_REL_TOL, abs_tol=0.0)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=FLOAT_REL_TOL, abs_tol=0.0)
        except (TypeError, ValueError):
            return False
    return a == b


def differing(plans: list) -> dict:
    keys = list(plans[0]) if plans else []
    return {k: [p.get(k) for p in plans] for k in keys
            if any(not _same(p.get(k), plans[0].get(k)) for p in plans)}


def agree(label: str, plan: dict) -> Optional[dict]:
    if not _multi():
        return None
    got = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(got, plan)
    diff = differing(got)
    if not diff:
        return None
    return diff


class RankAgreementCallback(TrainerCallback):

    def __init__(self, label: str) -> None:
        self.label = label

    def on_train_begin(self, args, state, control, train_dataloader=None, **kwargs):
        diff = agree(self.label, plan_tuple(args, state, train_dataloader))
        if diff:
            raise RankDisagreement(f"{self.label}: ranks disagree on {sorted(diff)}: {diff}")
