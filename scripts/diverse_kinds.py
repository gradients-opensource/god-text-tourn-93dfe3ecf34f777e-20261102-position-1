import dataclasses
import os
from typing import Any, Optional

import torch
from pydantic import BaseModel

import spare_budget as sb

T_BAG = sb.T_BAG
T_NEFT = sb.T_NEFT
T_PERTURB = sb.T_PERTURB
KINDS = (T_BAG, T_NEFT, T_PERTURB)
SEED_BASE = {T_BAG: 10_600, T_NEFT: 10_700, T_PERTURB: 10_500}

_FALSY = ("0", "false", "no", "off")


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in _FALSY


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


ENV_ON, ENV_FORCE = "SN56_DK", "SN56_DK_FORCE"
BAG_FRAC = _float("SN56_DK_BAG_FRAC", 0.5)
BAG_MIN_ROWS = int(_float("SN56_DK_BAG_MIN_ROWS", 8))
NEFT_ALPHA = _float("SN56_DK_NEFT_ALPHA", 5.0)
NEFT_MULT = _float("SN56_DK_NEFT_MULT", 3.0)
PERTURB_LAMBDA = _float("SN56_DK_PERTURB_LAMBDA", 0.5)
PERTURB_KAPPA = _float("SN56_DK_PERTURB_KAPPA", 0.5)
_KIND_ENV = {T_BAG: "SN56_DK_BAG", T_NEFT: "SN56_DK_NEFT", T_PERTURB: "SN56_DK_PERTURB"}


def enabled() -> bool:
    return _flag(ENV_ON, "1")


def forced() -> bool:
    return enabled() and _flag(ENV_FORCE, "0")


def kinds() -> list:
    if not enabled():
        return []
    return [k for k in KINDS if _flag(_KIND_ENV[k], "1")]


def continuation_kinds() -> list:
    return kinds()


def owns(kind: str) -> bool:
    return kind in kinds()


def describe() -> dict:
    return {"enabled": enabled(), "forced": forced(), "kinds": kinds(),
            "bag_frac": BAG_FRAC, "neft_alpha": NEFT_ALPHA, "neft_mult": NEFT_MULT,
            "perturb_lambda": PERTURB_LAMBDA, "perturb_kappa": PERTURB_KAPPA}


def seed_for(kind: str, k: int) -> int:
    return int(SEED_BASE[kind]) + max(0, int(k))


def candidate(kind: str, head: Any, lr: float, k: int) -> Any:
    return dataclasses.replace(head, kind=kind, seed=seed_for(kind, k), learning_rate=float(lr))


class Prepared(BaseModel):
    kind: str
    seed: int
    applied: bool = False
    reason: str = ""
    n_train: Optional[int] = None
    n_subset: Optional[int] = None
    frac: Optional[float] = None
    alpha_a0: Optional[float] = None
    alpha: Optional[float] = None
    lam: Optional[float] = None
    kappa: Optional[float] = None


def neft_alpha(a0_alpha: Optional[float]) -> float:
    try:
        a0 = float(a0_alpha) if a0_alpha is not None else None
    except (TypeError, ValueError):
        a0 = None
    if a0 is not None and a0 >= NEFT_ALPHA:
        return float(NEFT_MULT * a0)
    return float(NEFT_ALPHA)


def apply_args(kind: str, args2: Any, a0_args: Any) -> Prepared:
    seed = int(getattr(args2, "seed", 0) or 0)
    if kind != T_NEFT:
        return Prepared(kind=kind, seed=seed, applied=False, reason="not_args")
    a0_alpha = getattr(a0_args, "neftune_noise_alpha", None)
    alpha = neft_alpha(a0_alpha)
    args2.neftune_noise_alpha = alpha
    return Prepared(kind=kind, seed=seed, applied=True, reason="neftune_alpha",
                    alpha_a0=(float(a0_alpha) if a0_alpha is not None else None), alpha=alpha)


def subset_indices(n: int, frac: float, seed: int) -> list:
    n = max(0, int(n))
    if n <= 0:
        return []
    m = int(round(float(frac) * n))
    m = max(min(n, BAG_MIN_ROWS), min(n, m))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)[:m]
    return sorted(int(i) for i in perm.tolist())


def apply_trainer(kind: str, cand: Any, trainer2: Any) -> Prepared:
    seed = int(getattr(cand, "seed", 0) or 0)
    if kind != T_BAG:
        return Prepared(kind=kind, seed=seed, applied=False, reason="not_trainer")
    ds = getattr(trainer2, "train_dataset", None)
    if ds is None:
        return Prepared(kind=kind, seed=seed, applied=False, reason="no_train_dataset")
    try:
        n = len(ds)
    except TypeError:
        return Prepared(kind=kind, seed=seed, applied=False, reason="no_length")
    idx = subset_indices(n, BAG_FRAC, seed)
    if not idx or len(idx) >= n:
        return Prepared(kind=kind, seed=seed, applied=False, reason="subset_not_smaller",
                        n_train=n, n_subset=len(idx), frac=BAG_FRAC)
    trainer2.train_dataset = torch.utils.data.Subset(ds, idx)
    return Prepared(kind=kind, seed=seed, applied=True, reason="subset", n_train=n,
                    n_subset=len(idx), frac=BAG_FRAC)


def perturb_params(legacy_lam: float, legacy_kappa: float) -> tuple:
    if owns(T_PERTURB):
        return float(PERTURB_LAMBDA), float(PERTURB_KAPPA)
    return float(legacy_lam), float(legacy_kappa)
