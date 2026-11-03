

import copy
import dataclasses
import datetime
import functools
import gc
import math
import os
import statistics
import sys
import time
import traceback
import types
from dataclasses import dataclass, field, asdict
from typing import Callable, Literal, Optional, TypedDict

import torch
from pydantic import BaseModel
from transformers import TrainerCallback
from transformers.optimization import get_scheduler

import async_save
import dev_pass
import diverse_kinds as dk
import eval_squeeze
import free_candidates as fc
import gpu_soup
import host_snapshot
import kind_fix
import logit_calib
import lr_search
import member_guard
import member_pricing as mp
import packed_fill
import planner_clock
import rank_agreement
import release_defaults
import save_guard as sg
import spare_budget as sb
import stop_to_decay as std
import compute_affordance
import run_state

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

_FALSY = ("0", "false", "no", "off")


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in _FALSY


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


_RESERVE_SECS = _env_float("SN56_ST_RESERVE_SECS", 180.0)
_LOAD_EST_SECS = _env_float("SN56_ST_LOAD_SECS", 180.0)
_EVAL_SECS_FALLBACK = _env_float("SN56_ST_EVAL_SECS", 25.0)
UNION_SOUP_MAX = int(_env_float("SN56_ST_SOUP_MAX", 12))
_SAFETY = 1.1
_LR_RATIO_PREFER_LR = 1.6
_EVAL_SKIP_EPOCH_FRAC = 0.75

_EVAL_TIME_FRAC = _env_float("SN56_ST_EVAL_FRAC", 0.25)
_MIN_EVAL_STRIDE = int(_env_float("SN56_ST_MIN_EVAL_STRIDE", 10))
_MIN_EVALS = int(_env_float("SN56_ST_MIN_EVALS", 3))
_WARMUP_MAX_FRAC = _env_float("SN56_ST_WARMUP_FRAC", 0.25)

_WSD_DECAY_FRAC = _env_float("SN56_ST_DECAY_FRAC", 0.20)
_WSD_MIN_DECAY_STEPS = int(_env_float("SN56_ST_MIN_DECAY_STEPS", 20))
_WSD_DECAY_TYPE = os.environ.get("SN56_ST_DECAY_TYPE", "linear").strip() or "linear"
_WSD_MIN_LR_RATIO = _env_float("SN56_ST_MIN_LR_RATIO", 0.0)
LR_SHAPE = "wsd_linear_to_zero"
_ANNEAL_TOL = _env_float("SN56_ST_ANNEAL_TOL", 0.05)

T_SEED = "t_seed"
T_LR = "t_lr"

T_DIV_LR = sb.T_DIV_LR
T_DIV_BATCH = "t_div_batch"
DIVERSE_KINDS = (T_DIV_LR, T_DIV_BATCH)
_DIVERSE = _env_flag("SN56_DIVERSE_MEMBERS", "1")
_DIV_LR_MULT = 2.0
_DIV_ROWS_DIV = 4
_DIV_LR_BATCH_EXP = 0.25
_TLR_FRESH_SEED = _env_flag("SN56_TLR_FRESH_SEED", "1")
FRESH_SEED_BASE = 10_000
SEED_T_LR = FRESH_SEED_BASE
SEED_DIV_LR = FRESH_SEED_BASE + 1
SEED_DIV_BATCH = FRESH_SEED_BASE + 2
SEED_DIV_BATCH_BASE = FRESH_SEED_BASE + 100
SEED_DIV_LR_BASE = FRESH_SEED_BASE + 200
_DIV_LR_ON = _env_flag("SN56_ST_DIV_LR", "0")
_DIV_REPEAT = _env_flag("SN56_ST_DIV_REPEAT", "1")
_DIV_MAX = int(_env_float("SN56_ST_DIV_MAX", 64))
_DIV_FLOOR_FRAC = _env_float("SN56_ST_DIV_FLOOR_FRAC", 0.5)
_DIV_PRICE_MEASURED = _env_flag("SN56_ST_DIV_PRICE_MEASURED", "1")

_SB = _env_flag("SN56_ST_SPARE_BUDGET", "1")
_SB_ONLINE_SNAPS = int(_env_float("SN56_SB_ONLINE_SNAPS", 3))
_SB_POOL_FALLBACK_1GPU = _env_flag("SN56_SB_POOL_FALLBACK_1GPU", "0")
T_PERTURB = sb.T_PERTURB
SEED_PERTURB_BASE = FRESH_SEED_BASE + 500
_PERTURB = _env_flag("SN56_ST_PERTURB", "0")
_PERTURB_KAPPA = _env_float("SN56_ST_PERTURB_KAPPA", 0.5)
_PERTURB_LAMBDA = _env_float("SN56_ST_PERTURB_LAMBDA", 1.0)
_DK_KINDS = (dk.T_BAG, dk.T_NEFT)

_EXTRA = _env_flag("SN56_SPARE_EXTRA", "1")
_SPARE_MIN_FRAC = _env_float("SN56_SPARE_MIN_FRAC", 0.20)
_LOAD_EST_SECS_HONEST = _env_float("SN56_ST_LOAD_SECS_HONEST", 20.0)
_PREFER_SEED_ON_OVERFIT = _env_flag("SN56_ST_PREFER_SEED_ON_OVERFIT", "1")
_MAX_EXTRA_SEEDS = int(_env_float("SN56_ST_MAX_EXTRA_SEEDS", 3))
_SOUP_PLATEAU_PATIENCE = int(_env_float("SN56_ST_SOUP_PATIENCE", 2))
_SOUP_ACCEPT_MARGIN = _env_float("SN56_ST_SOUP_MARGIN", 0.004)
_SOUP_PLATEAU_MARGIN = _env_float("SN56_ST_SOUP_PLATEAU_MARGIN", 0.008)
_PLATEAU_BLIND_AVG = _env_flag("SN56_ST_PLATEAU_BLIND_AVG", "1")
_BLIND_SOUP_EVALS = int(_env_float("SN56_ST_BLIND_SOUP_EVALS", 6))
_DEVPASS_RESERVE_MIN_SECS = _env_float("SN56_ST_DEVPASS_RESERVE_SECS", 60.0)
_SB_TAIL_MEASURED = _env_flag("SN56_SB_TAIL_MEASURED", "1")
_SB_TAIL_FIRST = _env_flag("SN56_SB_TAIL_FIRST", "1")
_FIRST_UNIT_V2 = _env_flag("SN56_SB_FIRST_UNIT_V2", "1")
FIRST_UNIT_RATIO = _env_float("SN56_SB_FIRST_UNIT_RATIO", 1.821 if _FIRST_UNIT_V2 else 1.853)
_SB_TAIL_BASIN = _env_flag("SN56_SB_TAIL_BASIN", "1")
_SB_TAIL_GAP = _env_flag("SN56_SB_TAIL_GAP", "1")
TAIL_GAP_SECS = _env_float("SN56_SB_TAIL_GAP_SECS", 3.7)
_SB_TAIL_ONLY = _env_flag("SN56_SB_TAIL_ONLY", "1")
_SB_TAIL_IDLE_FLOOR = _env_flag("SN56_SB_TAIL_IDLE_FLOOR", "1")
CONTINUATION_BUILDERS: dict = {}
_DEVFIT_UNTRAINED = _env_flag("SN56_ST_DEVFIT_UNTRAINED", "1")


class MemberPhaseSwitches(BaseModel):
    guard: bool
    guard_reason: str
    world: int
    spare_budget: bool
    div_lr: bool
    div_repeat: bool
    price_measured: bool
    devfit_untrained: bool


def member_phase_switches(world: Optional[int] = None) -> MemberPhaseSwitches:
    g = sb.multi_gpu_guard(_dist_world() if world is None else world)
    if g.on:
        return MemberPhaseSwitches(guard=True, guard_reason=g.reason, world=g.world,
                                   spare_budget=False, div_lr=True, div_repeat=False,
                                   price_measured=False, devfit_untrained=False)
    return MemberPhaseSwitches(guard=False, guard_reason=g.reason, world=g.world,
                               spare_budget=bool(_SB), div_lr=bool(_DIV_LR_ON),
                               div_repeat=bool(_DIV_REPEAT),
                               price_measured=bool(_DIV_PRICE_MEASURED),
                               devfit_untrained=bool(_DEVFIT_UNTRAINED))
_MEASURED = _env_flag("SN56_PLANNER_MEASURED", "1")


_SEED_LOOP = _env_flag("SN56_SEED_LOOP", "1")
_LOOP_GATE_ON_SPARE = _env_flag("SN56_ST_LOOP_SPARE_GATE", "1")
_LOOP_GATE_EXCLUDE_STOPS = ("deadline", "save_failures")
_LOOP_EVAL_LIGHT = _env_flag("SN56_ST_LOOP_EVAL_LIGHT", "1")
_LOOP_POOL_PRUNE = _env_flag("SN56_ST_LOOP_POOL_PRUNE", "1")
_LOOP_DROP_SOS = _env_flag("SN56_ST_LOOP_DROP_SOS", "1")
_LOOP_NO_CAP = _env_flag("SN56_ST_LOOP_NO_CAP", "1")
_LOOP_MAX_SEEDS = int(_env_float("SN56_ST_LOOP_MAX_SEEDS", 999))
_LOOP_POOL_BASE = int(_env_float("SN56_ST_LOOP_POOL_BASE", 4))


def loop_pool_cap(n_seeds: int) -> int:
    return _LOOP_POOL_BASE + max(0, int(n_seeds))


_LOOP_SECOND_PASS = _env_flag("SN56_ST_LOOP_SECOND_PASS", "1")
_LOOP_A0_WEIGHT_PROBE = _env_flag("SN56_ST_LOOP_A0_WEIGHT_PROBE", "1")


def _loop_on() -> bool:
    return bool(_EXTRA and _SEED_LOOP)


_COLD_LR = _env_flag("SN56_ST_COLD_LR", "1")
_COLD_LR_MARGIN = _env_float("SN56_ST_COLD_LR_MARGIN", 0.008)
_COLD_LR_MAX_DEX = _env_float("SN56_ST_COLD_LR_MAX_DEX", 0.30)
_COLD_LR_TIERS = tuple(
    t.strip() for t in os.environ.get("SN56_ST_COLD_LR_TIERS", "full").split(",")
    if t.strip())


def cold_lr_dex(lr0: Optional[float], argmin: Optional[float]) -> Optional[float]:
    
    if not lr0 or not argmin or lr0 <= 0 or argmin <= 0:
        return None
    return math.log10(lr0) - math.log10(argmin)


def _cold_lr_eligible(facts: "Attempt0Facts", n_fit: int) -> bool:
    
    if not _COLD_LR or n_fit < 2:
        return False
    if (facts.pool_tier or "") not in _COLD_LR_TIERS:
        return False
    dex = cold_lr_dex(facts.learning_rate, facts.argmin_lr)
    if dex is None or dex <= 1e-9:
        return False
    return dex <= _COLD_LR_MAX_DEX + 1e-9


def cold_lr_decide(t_lr_dev: Optional[float], t_seed_dev: Optional[float],
                   a0_single_dev: Optional[float], margin: float = _COLD_LR_MARGIN,
                   t_lr_diverged: bool = False) -> dict:
    
    out = {"t_lr_dev": t_lr_dev, "t_seed_dev": t_seed_dev,
           "a0_single_dev": a0_single_dev, "margin": float(margin),
           "d_seed": None, "d_a0": None, "switched": False, "reason": ""}
    if t_lr_diverged:
        out["reason"] = "t_lr_diverged"
        return out
    vals = {"t_lr": t_lr_dev, "t_seed": t_seed_dev, "a0_single": a0_single_dev}
    missing = [k for k, v in vals.items()
               if v is None or not isinstance(v, (int, float)) or v != v]
    if missing:
        out["reason"] = "missing:" + ",".join(missing)
        return out
    assert t_lr_dev is not None and t_seed_dev is not None and a0_single_dev is not None
    d_seed = float(t_lr_dev) - float(t_seed_dev)
    d_a0 = float(t_lr_dev) - float(a0_single_dev)
    out["d_seed"], out["d_a0"] = round(d_seed, 6), round(d_a0, 6)
    seed_ok = d_seed < -margin
    a0_ok = d_a0 < -margin
    out["switched"] = bool(seed_ok and a0_ok)
    out["reason"] = ("switch" if out["switched"] else
                     "both_margins" if not (seed_ok or a0_ok) else
                     "seed_margin" if not seed_ok else "a0_margin")
    return out


def cold_lr_remint(candidates: list, after_idx: int, lr_cold: float) -> list:
    
    out = []
    for c in candidates[after_idx + 1:]:
        if c.kind == T_SEED:
            out.append(Candidate(c.kind, c.seed, float(lr_cold), c.max_steps,
                                 horizon_steps=c.horizon_steps,
                                 warmup_steps=c.warmup_steps,
                                 decay_steps=c.decay_steps,
                                 eval_steps=c.eval_steps, n_evals=c.n_evals,
                                 eval_at=tuple(c.eval_at)))
        else:
            out.append(c)
    return out


def _extra_active(facts: "Attempt0Facts") -> bool:
    
    if not _EXTRA:
        return False
    frac = facts.spare_frac
    if _SEED_LOOP and _LOOP_GATE_ON_SPARE and frac is not None:
        if (facts.stop_reason or "") in _LOOP_GATE_EXCLUDE_STOPS:
            return False
        return frac >= _SPARE_MIN_FRAC
    if (facts.stop_reason or "") != "overfit":
        return False
    if frac is None:
        return True
    return frac >= _SPARE_MIN_FRAC


@dataclass
class Attempt0Facts:
    
    best_step: Optional[int] = None
    best_loss: Optional[float] = None
    steps_run: int = 0
    steps_per_epoch: int = 0
    num_train_epochs: float = 0.0
    eval_steps: int = 0
    n_evals: int = 0
    train_runtime: Optional[float] = None
    eval_runtime: Optional[float] = None
    t_per_step_search: Optional[float] = None
    learning_rate: Optional[float] = None
    seed: int = 42
    argmin_lr: Optional[float] = None
    pool_tier: Optional[str] = None
    world_size: int = 1
    avg_mode: str = "ram"
    is_final_run: bool = True
    pool_size: int = 0
    horizon_steps: int = 0
    warmup_steps: int = 0
    min_lr_rate: float = 0.0
    lr_scheduler_type: str = ""
    stop_reason: Optional[str] = None
    total_secs: Optional[float] = None
    remaining_at_stop: Optional[float] = None
    pool_size_loop: Optional[int] = None
    clock: Optional[dict] = None
    t_load_a0: Optional[float] = None
    memory: Optional[dict] = None
    owns_submission: bool = True
    soup_secs_per_cand: Optional[float] = None
    step_ladder: Optional[list] = None
    n_dev: int = 0
    a0_trial_secs: Optional[list] = None
    a0_soup_steps: Optional[list] = None
    a0_stop_to_decay: bool = False
    member_steps: Optional[int] = None

    @property
    def memory_proof(self) -> Optional[member_guard.MemoryProof]:
        if not self.memory:
            return None
        try:
            return member_guard.MemoryProof(**self.memory)
        except Exception:
            return None

    @property
    def clock_stats(self) -> Optional[planner_clock.ClockStats]:
        if not _MEASURED or not self.clock:
            return None
        try:
            return planner_clock.ClockStats(**self.clock)
        except Exception:
            return None

    @property
    def a0_keep_steps(self) -> set:
        
        if not self.best_step or self.best_step <= 0:
            return set()
        keep = {int(self.best_step)}
        spe = int(self.steps_per_epoch or 0)
        if spe > 0:
            keep.add(int(self.best_step) + spe)
            if self.best_step - spe > 0:
                keep.add(int(self.best_step) - spe)
        return keep

    @property
    def spare_frac(self) -> Optional[float]:
        
        if not self.total_secs or self.total_secs <= 0:
            return None
        if self.remaining_at_stop is None:
            return None
        return max(0.0, min(1.0, self.remaining_at_stop / self.total_secs))

    @property
    def a0_horizon(self) -> int:
        
        return int(self.horizon_steps or self.steps_run or 0)

    @property
    def t_step_realised(self) -> Optional[float]:
        
        if not self.train_runtime or self.steps_run <= 0:
            return None
        return self.train_runtime / self.steps_run

    def n_evals_to(self, step: int) -> int:
        
        if self.eval_steps <= 0:
            return 0
        first = self.eval_steps
        if self.num_train_epochs > 1 and self.steps_per_epoch > 0:
            floor = int(_EVAL_SKIP_EPOCH_FRAC * self.steps_per_epoch)
            first = max(first, -(-floor // self.eval_steps) * self.eval_steps)
        if step < first:
            return 0
        return (step - first) // self.eval_steps + 1


def cosine_lr_frac(step: int, horizon: int, warmup: int,
                   min_lr_rate: float = 0.0, num_cycles: float = 0.5) -> float:
    
    if horizon <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, horizon - warmup))
    progress = min(1.0, max(0.0, progress))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    return max(0.0, factor * (1.0 - min_lr_rate) + min_lr_rate)


def wsd_lr_frac(step: int, horizon: int, warmup: int, decay_steps: int,
                min_lr_ratio: float = 0.0, decay_type: str = "linear") -> float:
    
    if horizon <= 0:
        return 1.0
    warmup = max(0, int(warmup))
    decay_steps = max(0, int(decay_steps))
    stable = max(0, horizon - warmup - decay_steps)
    if warmup > 0 and step < warmup:
        return max(0.0, (float(step) / float(max(1, warmup))) * (1.0 - min_lr_ratio) + min_lr_ratio)
    if step < warmup + stable:
        return 1.0
    if step < warmup + stable + decay_steps:
        progress = float(step - warmup - stable) / float(max(1, decay_steps))
        if decay_type == "1-sqrt":
            factor = 1.0 - math.sqrt(progress)
        elif decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            factor = 1.0 - progress
        return max(0.0, factor * (1.0 - min_lr_ratio) + min_lr_ratio)
    return min_lr_ratio


def preflight_schedule(args2, horizon: int) -> dict:
    
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=float(getattr(args2, "learning_rate", 1.0) or 1.0))
    name = str(getattr(args2, "lr_scheduler_type", "") or "")
    name = name.split(".")[-1].lower()
    sched = get_scheduler(
        name, opt,
        num_warmup_steps=int(getattr(args2, "warmup_steps", 0) or 0),
        num_training_steps=int(horizon),
        scheduler_specific_kwargs=dict(getattr(args2, "lr_scheduler_kwargs", None) or {}),
    )
    lam = sched.lr_lambdas[0]
    w = int(getattr(args2, "warmup_steps", 0) or 0)
    d = int((getattr(args2, "lr_scheduler_kwargs", None) or {}).get("num_decay_steps", 0) or 0)
    stable = max(0, int(horizon) - w - d)
    pts = {"0": 0, "warmup_end": w, "stable_end": w + stable,
           "last_step": w + stable + d - 1, "horizon": w + stable + d}
    return {f"lr_frac_at_{k}": round(float(lam(v)), 12) for k, v in pts.items()}


def decay_steps_for(max_steps: int, warmup: int) -> int:
    
    if max_steps <= 0:
        return 0
    d = max(_WSD_MIN_DECAY_STEPS, int(round(_WSD_DECAY_FRAC * max_steps)))
    return max(1, min(d, max(1, max_steps - max(0, warmup) - 1)))


def eval_cadence(facts: Attempt0Facts, max_steps: int,
                 t_step: Optional[float], t_eval: Optional[float],
                 decay_steps: int = 0) -> tuple:
    
    if max_steps <= 0:
        return (facts.eval_steps or 0), 0, ()
    target = facts.n_evals or facts.n_evals_to(max_steps) or _MIN_EVALS
    caps = [target, max(1, max_steps // max(1, _MIN_EVAL_STRIDE))]
    if t_step and t_eval and t_eval > 0:
        caps.append(int((_EVAL_TIME_FRAC * max_steps * t_step) / t_eval))
    n = max(_MIN_EVALS, min(caps))
    n = min(n, max_steps)
    eval_steps = max(1, int(max_steps // n))
    pts = eval_points(max_steps, decay_steps, n)
    return eval_steps, len(pts), pts


def _placement_stride(max_steps: int) -> int:
    
    if max_steps <= 0:
        return 1
    return max(1, min(_MIN_EVAL_STRIDE, max_steps // max(1, _MIN_EVALS)))


def _spread(pts, anchors, stride: int) -> list:
    
    out: list = []
    for p in sorted(set(pts)):
        if p in anchors:
            while out and p - out[-1] < stride and out[-1] not in anchors:
                out.pop()
            out.append(p)
        elif not out or p - out[-1] >= stride:
            out.append(p)
    return out


def eval_points(max_steps: int, decay_steps: int, n: int) -> tuple:
    
    if max_steps <= 0 or n <= 0:
        return ()
    d = max(0, min(int(decay_steps or 0), max_steps))
    decay_start = max(0, max_steps - d)
    stride = _placement_stride(max_steps)
    if d <= 1 or decay_start <= 0:
        step = max(1, max_steps // n)
        pts = {min(max_steps, j * step) for j in range(1, n + 1)}
        pts.add(max_steps)
        return tuple(_spread(pts, {max_steps}, stride))
    n_decay = min(max(1, -(-n // 2)), d)
    anchors = {decay_start, decay_start + d // 2, max_steps}
    dec = {decay_start + max(1, round(i * d / n_decay))
           for i in range(1, n_decay + 1)}
    dec.add(decay_start + d // 2)
    dec.add(max_steps)
    dec = _spread([x for x in dec if decay_start < x <= max_steps], anchors, stride)
    n_stable = max(1, min(max(0, n - n_decay), len(dec)))
    stable = {max(1, (j * decay_start) // n_stable)
              for j in range(1, n_stable + 1)}
    stable = _spread([x for x in stable if 0 < x <= decay_start],
                     {decay_start}, stride)
    return tuple(sorted(set(stable) | set(dec)))


def loop_eval_points(max_steps: int, decay_steps: int) -> tuple:
    
    if max_steps <= 0:
        return ()
    d = max(0, min(int(decay_steps or 0), max_steps))
    decay_start = max(0, max_steps - d)
    pts = {max_steps}
    mid = decay_start + d // 2
    if d > 1 and 0 < mid < max_steps:
        pts.add(mid)
    return tuple(sorted(pts))


def warmup_for(facts: Attempt0Facts, max_steps: int) -> int:
    
    if facts.warmup_steps <= 0 or max_steps <= 0:
        return 0
    return max(1, min(int(facts.warmup_steps), int(_WARMUP_MAX_FRAC * max_steps)))


@dataclass
class Candidate:
    kind: str
    seed: int
    learning_rate: float
    max_steps: int
    horizon_steps: int = 0
    warmup_steps: int = 0
    decay_steps: int = 0
    eval_steps: int = 0
    n_evals: int = 0
    eval_at: tuple = ()
    per_device: int = 0
    accum: int = 0


class _Sched(TypedDict):
    
    horizon_steps: int
    warmup_steps: int
    decay_steps: int
    eval_steps: int
    n_evals: int
    eval_at: tuple


@dataclass
class Decision:
    
    remaining: float = 0.0
    reserve: float = 0.0
    t_soup: float = 0.0
    t_traj: float = 0.0
    t_traj_decomposed: float = 0.0
    t_traj_realised: float = 0.0
    t_load_est: float = 0.0
    spendable: float = 0.0
    shortfall_next: float = 0.0
    n_fit: int = 0
    candidates: list = field(default_factory=list)
    reason: str = ""
    extra: bool = False
    devpass_reserve: float = 0.0
    loop: bool = False
    eval_steps: int = 0
    n_evals: int = 0
    n_evals_a0_cadence: int = 0
    t_step_eval_free: float = 0.0
    t_eval: float = 0.0
    t_steps: float = 0.0
    t_evals: float = 0.0
    clock_source: str = "legacy"
    t_traj_legacy: float = 0.0
    t_member_extra: float = 0.0
    t_core: float = 0.0
    t_core_priced: float = 0.0
    t_prelaunch: float = 0.0
    t_launch_tail: float = 0.0
    member_len: Optional[int] = None

    @property
    def launch(self) -> bool:
        return bool(self.candidates)


@dataclass
class TrajectoryResult:
    kind: str
    seed: int
    learning_rate: float
    max_steps: int
    launched: bool = False
    stop_step: int = 0
    best_step: Optional[int] = None
    best_loss: Optional[float] = None
    still_descending_at_stop: bool = False
    n_evals: int = 0
    secs: float = 0.0
    load_secs: float = 0.0
    error: Optional[str] = None
    horizon_steps: int = 0
    a0_horizon_steps: int = 0
    lr_shape: str = ""
    warmup_steps: int = 0
    stable_steps: int = 0
    decay_steps: int = 0
    eval_steps: int = 0
    min_lr_ratio: float = 0.0
    lr_peak: Optional[float] = None
    lr_at_stop: Optional[float] = None
    lr_at_stop_over_peak: Optional[float] = None
    lr_at_stop_over_peak_a0_horizon: Optional[float] = None
    dev_at_decay_mid: Optional[float] = None
    annealed_at_stop: bool = False
    best_is_last_eval: bool = False
    dev_delta_last: Optional[float] = None
    stop_reason: Optional[str] = None
    diverged: bool = False
    tail: bool = False


@dataclass
class DevfitPlan:
    
    enabled: bool = False
    n_dev: int = 0
    min_lr: float = 0.0
    min_lr_rate: float = 0.0
    max_grad_norm: float = 0.0
    train_per_device: int = 1
    train_grad_accum: int = 1
    local_rank: int = 0
    budget_secs_a0: Optional[float] = None
    eval_runtime: Optional[float] = None
    skip_reason: str = ""
    memory: Optional[dict] = None
    collective_timeout: Optional[float] = None
    ddp_timeout: float = 1800.0
    dev_micro_batch: Optional[int] = None


@dataclass
class Outcome:
    
    launched: bool = False
    mutated: bool = False
    reason: str = ""
    decision: Optional[Decision] = None
    results: list = field(default_factory=list)
    soup_ran: bool = False
    spare: dict = field(default_factory=dict)
    spare_final: dict = field(default_factory=dict)
    artefacts_meta: dict = field(default_factory=dict)
    devfit: Optional[dict] = None


DEPTH_MEMBERS = False
DEPTH_MEMBER_MIN_FRAC = packed_fill.MEMBER_MIN_FRAC


def member_ref(facts: Attempt0Facts) -> int:
    return int(getattr(facts, "member_steps", None) or getattr(facts, "best_step", None) or 0)


def plan(facts: Attempt0Facts, remaining: float,
         reserve: float = _RESERVE_SECS,
         load_est: float = _LOAD_EST_SECS,
         devpass_reserve: float = 0.0,
         remainder: Optional[bool] = None, depth: Optional[bool] = None) -> Decision:
    kw = dict(reserve=reserve, load_est=load_est, devpass_reserve=devpass_reserve)
    facts.member_steps = None
    d = _plan(facts, remaining, **kw)
    if remainder is False:
        return d
    layer = remainder_layer(facts, depth) or ("packed_fill" if remainder else None)
    if layer is None:
        return d
    if layer != "stop_to_decay" and (d.n_fit > 0 or d.reason != "no_fit"):
        return d
    return _plan_remainder(facts, remaining, d, layer=layer, **kw)


def _handoff(facts: Attempt0Facts) -> bool:
    return bool(getattr(facts, "a0_stop_to_decay", False) and std.enabled())


def remainder_layer(facts: Attempt0Facts, depth: Optional[bool] = None) -> Optional[str]:
    if (DEPTH_MEMBERS if depth is None else depth):
        return "depth"
    if _handoff(facts):
        return "stop_to_decay"
    if packed_fill.active():
        return "packed_fill"
    return None


def member_bounds(facts: Attempt0Facts, handoff: bool = False) -> tuple:
    best = int(facts.best_step or 0)
    lo = max(best // packed_fill.MEMBER_MIN_FRAC, std.member_floor(warmup_for(facts, best)))
    hi = std.member_cap(best, facts.steps_run) if handoff else best
    return int(lo), int(hi)


def _plan_remainder(facts: Attempt0Facts, remaining: float, d_full: Decision, *,
                    layer: str, **kw) -> Decision:
    best = int(facts.best_step or 0)
    if best <= 0:
        return d_full
    handoff = layer == "stop_to_decay"
    lo, hi = member_bounds(facts, handoff)
    memo: dict = {}

    def at(n: int) -> Decision:
        if n not in memo:
            memo[n] = _plan(dataclasses.replace(facts, best_step=int(n), member_steps=None),
                            remaining, **kw)
        return memo[n]

    L = std.longest_fitting(lo, hi, lambda n: at(n).n_fit > 0)
    facts.remaining_at_stop = remaining
    tag = {"depth": "depth_remainder", "stop_to_decay": "stop_decay_members",
           "packed_fill": "packed_fill_remainder"}[layer]
    if L is None:
        if handoff:
            d_full.reason = f"{d_full.reason}+{tag}(none_fits,floor={lo})"
        return d_full
    d = at(L)
    d.reason = f"{d.reason}+{tag}(len={L}/{best}" + (f",cap={hi})" if handoff else ")")
    member = int(L) if int(L) != best else None
    if layer == "packed_fill":
        d.member_len = member
    elif handoff:
        facts.member_steps = member
    return d


def _plan(facts: Attempt0Facts, remaining: float,
          reserve: float = _RESERVE_SECS,
          load_est: float = _LOAD_EST_SECS,
          devpass_reserve: float = 0.0) -> Decision:

    facts.remaining_at_stop = remaining
    extra = _extra_active(facts)
    if extra:
        load_est = min(load_est, _LOAD_EST_SECS_HONEST)
    clk = facts.clock_stats
    if clk is not None:
        load_est = member_load_est(facts, load_est)
    d = Decision(remaining=remaining, reserve=reserve, t_load_est=load_est)
    d.extra = extra
    d.loop = bool(extra and _SEED_LOOP)
    d.devpass_reserve = float(devpass_reserve) if extra else 0.0

    if not facts.best_step or facts.best_step <= 0:
        d.reason = "no_best_step"
        return d

    t_eval = facts.eval_runtime or _EVAL_SECS_FALLBACK
    t_step = facts.t_per_step_search or facts.t_step_realised
    t_eval_l, t_step_l = t_eval, t_step
    if clk is not None:
        t_eval, t_step = clk.t_eval, clk.t_step
    if not t_step:
        d.reason = "no_step_time"
        return d

    if not facts.learning_rate or facts.learning_rate <= 0:
        d.reason = "no_lr"
        return d

    _warm = warmup_for(facts, facts.best_step)
    _decay = decay_steps_for(facts.best_step, _warm)
    d.eval_steps, n_evals, eval_at = eval_cadence(facts, facts.best_step,
                                                  t_step, t_eval, _decay)
    if d.loop and _LOOP_EVAL_LIGHT:
        eval_at = loop_eval_points(facts.best_step, _decay)
        n_evals = len(eval_at)
        d.eval_steps = max(1, int(facts.best_step // max(1, n_evals)))
    d.n_evals = n_evals
    d.n_evals_a0_cadence = facts.n_evals_to(facts.best_step)

    realised_free = None
    if facts.train_runtime and facts.steps_run > 0 and t_step_l:
        _net = facts.train_runtime - (facts.n_evals or 0) * t_eval_l
        if _net > 0:
            realised_free = _net / facts.steps_run
    d.t_step_eval_free = max(t_step_l or 0.0, realised_free or 0.0)
    d.t_eval = t_eval_l
    d.t_steps = facts.best_step * d.t_step_eval_free
    d.t_evals = n_evals * t_eval_l
    d.t_traj_decomposed = _SAFETY * (facts.best_step * (t_step_l or 0.0) + n_evals * t_eval_l)
    d.t_traj_realised = (_SAFETY * (facts.best_step * realised_free + n_evals * t_eval_l)
                         if realised_free else 0.0)
    d.t_traj = max(d.t_traj_decomposed, d.t_traj_realised) + load_est
    d.t_traj_legacy = d.t_traj
    if clk is not None:
        d.clock_source = "measured"
        d.t_step_eval_free = clk.t_step
        d.t_eval = clk.t_eval
        d.t_steps = facts.best_step * clk.t_step
        d.t_evals = n_evals * clk.t_point_member
        d.t_member_extra = planner_clock.write_secs(clk)
        d.t_core = clk.t_startup + d.t_steps + d.t_evals
        d.t_core_priced = mp.priced_core(d.t_core, _SAFETY)
        d.t_traj_decomposed = d.t_core_priced
        d.t_traj_realised = 0.0
        d.t_traj = d.t_traj_decomposed + load_est + d.t_member_extra
        d.t_prelaunch = mp.prelaunch_reserve(d.t_member_extra, pool_size=facts.pool_size)

    n_union = min(UNION_SOUP_MAX, facts.pool_size + 2 * max(1, n_evals))
    _blind = _BLIND_SOUP_EVALS if (extra and not (d.loop and _LOOP_DROP_SOS)) else 0
    n_soup_evals = n_union + _blind
    _soup_over = soup_overhead_secs(facts, t_eval)
    d.t_soup = (t_eval + _soup_over) * n_soup_evals

    if d.loop and _LOOP_NO_CAP:
        pool0 = (facts.pool_size_loop if facts.pool_size_loop is not None
                 else facts.pool_size + 1)
        d.t_traj = d.t_traj + t_eval
        base = remaining - reserve - d.devpass_reserve - d.t_prelaunch
        if clk is not None:
            d.t_launch_tail = mp.launch_tail(max(0, pool0 - 1), t_eval, _soup_over,
                                             d.t_member_extra)

        def _t_soup(n: int) -> float:
            return max(d.t_launch_tail,
                       (2.0 * t_eval + _soup_over) * (min(loop_pool_cap(n), pool0 + max(1, n_evals) * n)
                                                      + _blind))

        n_affordable = 0
        while (n_affordable < _LOOP_MAX_SEEDS
               and (n_affordable + 1) * d.t_traj + _t_soup(n_affordable + 1) <= base):
            n_affordable += 1
        d.t_soup = _t_soup(max(1, n_affordable))
        d.spendable = base - d.t_soup
        d.shortfall_next = max(0.0, (n_affordable + 1) * d.t_traj
                               + _t_soup(n_affordable + 1) - base)
        d.n_fit = n_affordable
    else:
        budget = remaining - reserve - d.t_soup - d.devpass_reserve - d.t_prelaunch
        d.spendable = budget
        n_affordable = max(0, int(budget // d.t_traj)) if d.t_traj > 0 else 0
        d.shortfall_next = max(0.0, (n_affordable + 1) * d.t_traj - budget)
        if extra and _PREFER_SEED_ON_OVERFIT:
            d.n_fit = min(n_affordable, _MAX_EXTRA_SEEDS + 1)
        else:
            d.n_fit = min(2, n_affordable)

    lr0 = facts.learning_rate
    argmin = facts.argmin_lr
    lr_ok = bool(lr0 and argmin and argmin > 0
                 and abs(math.log10(argmin) - math.log10(lr0)) > 1e-9)
    ratio = (lr0 / argmin) if (lr_ok and argmin is not None) else 1.0

    _sched: _Sched = _Sched(horizon_steps=facts.best_step, warmup_steps=_warm,
                            decay_steps=_decay, eval_steps=d.eval_steps,
                            n_evals=n_evals, eval_at=tuple(eval_at))
    t_seed = Candidate(T_SEED, 43, lr0, facts.best_step, **_sched)
    t_lr = (Candidate(T_LR, (SEED_T_LR if _TLR_FRESH_SEED else facts.seed), argmin,
                      facts.best_step, **_sched)
            if (lr_ok and argmin is not None) else None)

    if extra and _PREFER_SEED_ON_OVERFIT and d.n_fit >= 1:
        _loop_cap = bool(d.loop and _LOOP_NO_CAP)
        _base_reason = "loop_seeds" if _loop_cap else "overfit_prefer_seed"
        if t_lr is not None and _cold_lr_eligible(facts, d.n_fit):
            seeds = _iter_seeds(d.n_fit - 1, loop=_loop_cap)
            d.candidates = ([Candidate(T_SEED, seeds[0], lr0, facts.best_step, **_sched),
                             t_lr]
                            + [Candidate(T_SEED, s, lr0, facts.best_step, **_sched)
                               for s in seeds[1:]])
            d.reason = f"{_base_reason}+cold_lr(n={len(d.candidates)})"
            return d
        seeds = _iter_seeds(d.n_fit, loop=_loop_cap)
        d.candidates = [Candidate(T_SEED, s, lr0, facts.best_step, **_sched)
                        for s in seeds]
        d.reason = f"{_base_reason}(n={len(d.candidates)})"
        return d

    if d.n_fit <= 0:
        d.reason = "no_fit"
    elif d.n_fit >= 2 and t_lr is not None:
        d.candidates = [t_seed, t_lr]
        d.reason = "both_fit"
    elif t_lr is None:
        d.candidates = [t_seed]
        d.reason = "argmin_equals_edge" if d.n_fit >= 2 else "one_fits"
    elif ratio >= _LR_RATIO_PREFER_LR:
        d.candidates = [t_lr]
        d.reason = f"one_fits_prefer_lr(ratio={ratio:.2f})"
    else:
        d.candidates = [t_seed]
        d.reason = f"one_fits_prefer_seed(ratio={ratio:.2f})"
    return d


def member_load_est(facts: Attempt0Facts, load_est: float) -> float:
    t = facts.t_load_a0
    if t is None or t != t or t < 0:
        return min(load_est, _LOAD_EST_SECS_HONEST)
    return min(load_est, _LOAD_EST_SECS_HONEST, float(t))


def soup_overhead_secs(facts: "Attempt0Facts", t_eval: float) -> float:
    m = getattr(facts, "soup_secs_per_cand", None)
    if m is None or m != m:
        return 0.0
    return max(0.0, float(m) - float(t_eval or 0.0))


def sft_objective(trainer) -> bool:
    try:
        names = [c.__name__.lower() for c in type(trainer).__mro__]
    except Exception:
        return True
    return not any(("dpo" in n) or ("grpo" in n) for n in names)


def tail_reserves(facts: Attempt0Facts, devfit_enabled: bool,
                  devpass_legacy: float, measured_devpass: bool = False) -> planner_clock.TailReserve:
    return logit_calib.price_tail(_tail_reserve_base(facts, devfit_enabled, devpass_legacy,
                                                     measured_devpass),
                                  facts.eval_runtime, planner_clock.write_secs(facts.clock_stats))


def _calib_ship(model2, ckpt_avg, submission_dir: str, end_time, hold_secs: float,
                facts: Attempt0Facts, log: Callable[[str], None], out) -> None:
    if not logit_calib.enabled():
        return
    try:
        _left = _remaining_secs(end_time)
        if _multi():
            _left = _r0(_left)
        res = logit_calib.calibrate(
            model2, logit_calib.trainer_evaluator(None, ckpt_avg), site="cross_soup",
            budget_secs=(None if _left is None else _left - float(hold_secs)),
            t_eval=facts.eval_runtime, t_write=planner_clock.write_secs(facts.clock_stats),
            multi=_multi(), log=log)
        if res is not None:
            out.artefacts_meta["logit_calib"] = res.model_dump()
            if res.kept and _is_main():
                sg.ship_marker_update(activity="logit_calib_write")
                if dev_pass._save_weights_only(logit_calib._unwrap(model2), submission_dir, log):
                    sg.ship_marker_update(activity="logit_calib_written")
    except Exception:
        pass


def _tail_reserve_base(facts: Attempt0Facts, devfit_enabled: bool,
                       devpass_legacy: float, measured_devpass: bool) -> planner_clock.TailReserve:
    return planner_clock.tail_reserve(
        facts.clock_stats, facts.total_secs,
        member_load_est(facts, _LOAD_EST_SECS_HONEST) if facts.clock_stats is not None else None,
        devfit_enabled=bool(devfit_enabled), devpass_legacy=float(devpass_legacy or 0.0),
        reserve_legacy=_RESERVE_SECS, measured_devpass=bool(measured_devpass),
        n_dev=int(getattr(facts, "n_dev", 0) or 0), fb1=micro1_fwd_bwd(facts.step_ladder))


def measured_tail(ckpt_avg) -> Optional[planner_clock.TailReserve]:
    facts_d = getattr(ckpt_avg, "stop_rule_facts", None) if ckpt_avg is not None else None
    if not isinstance(facts_d, dict):
        return None
    f = Attempt0Facts(**facts_d)
    pc = getattr(ckpt_avg, "planner_clock", None)
    f.clock = pc.stats_dict() if pc is not None else None
    dpr = float(getattr(ckpt_avg, "stop_rule_devpass_reserve", 0.0) or 0.0)
    return tail_reserves(f, devfit_enabled=dpr > 0, devpass_legacy=dpr)


def measured_tail_secs(ckpt_avg) -> Optional[float]:
    t = measured_tail(ckpt_avg)
    return None if t is None else float(t.reserve) + float(t.devpass_reserve)


def micro1_fwd_bwd(ladder: Optional[list]) -> Optional[float]:
    for e in ladder or []:
        try:
            if int(e.get("micro", 0)) == 1 and int(e.get("accum", 1) or 1) == 1 \
                    and e.get("fwd_bwd_secs") and float(e["fwd_bwd_secs"]) > 0:
                return float(e["fwd_bwd_secs"])
        except (TypeError, ValueError, AttributeError):
            continue
    return None


def set_pool_facts(facts: Attempt0Facts, ckpt_avg) -> None:
    facts.pool_size = len(getattr(ckpt_avg, "pool", []) or [])
    if _loop_on() and _LOOP_POOL_PRUNE:
        facts.pool_size_loop = (
            loop_pool_survivors(getattr(ckpt_avg, "pool", []) or [], facts.a0_keep_steps)
            + (1 if getattr(ckpt_avg, "best_state", None) is not None else 0))


def _iter_seeds(n: int, loop: bool = False) -> list:
    
    cap = _LOOP_MAX_SEEDS if loop else _MAX_EXTRA_SEEDS + 1
    n = max(1, min(int(n), cap))
    return [43 + i for i in range(n)]


def rung_down(micro: int, accum: int, div: int = _DIV_ROWS_DIV) -> tuple:
    return sb.rung_down(micro, accum, div)


def _sched_for(facts: Attempt0Facts, horizon: int, t_step: Optional[float],
               t_eval: Optional[float], light: bool) -> _Sched:
    warm = warmup_for(facts, horizon)
    decay = decay_steps_for(horizon, warm)
    eval_steps, n_evals, eval_at = eval_cadence(facts, horizon, t_step, t_eval, decay)
    if light:
        eval_at = loop_eval_points(horizon, decay)
        n_evals = len(eval_at)
        eval_steps = max(1, int(horizon // max(1, n_evals)))
    return _Sched(horizon_steps=horizon, warmup_steps=warm, decay_steps=decay,
                  eval_steps=eval_steps, n_evals=n_evals, eval_at=tuple(eval_at))


def div_batch_member(facts: Attempt0Facts, micro: int, accum: int, family_lr: float,
                     t_step: Optional[float], t_eval: Optional[float], light: bool,
                     seed: int = SEED_DIV_BATCH, div: int = _DIV_ROWS_DIV) -> Optional[Candidate]:
    if member_ref(facts) <= 0 or not family_lr or family_lr <= 0:
        return None
    rows = max(1, int(micro)) * max(1, int(accum))
    m2, a2 = rung_down(micro, accum, div)
    rows2 = m2 * a2
    if rows2 >= rows:
        return None
    horizon = int(math.ceil(member_ref(facts) * rows / rows2))
    lr = float(family_lr) * (rows2 / rows) ** _DIV_LR_BATCH_EXP
    return Candidate(T_DIV_BATCH, int(seed), lr, horizon, per_device=m2, accum=a2,
                     **_sched_for(facts, horizon, t_step, t_eval, light))


def div_batch_seed(k: int) -> int:
    return SEED_DIV_BATCH if int(k) <= 0 else SEED_DIV_BATCH_BASE + int(k)


def div_lr_seed(k: int) -> int:
    return SEED_DIV_LR if int(k) <= 0 else SEED_DIV_LR_BASE + int(k)


def div_lr_member(facts: Attempt0Facts, family_lr: float, t_step: Optional[float],
                  t_eval: Optional[float], light: bool,
                  seed: int = SEED_DIV_LR, lr: Optional[float] = None) -> Optional[Candidate]:
    if member_ref(facts) <= 0 or not family_lr or family_lr <= 0:
        return None
    _lr = float(lr) if (lr is not None and lr > 0) else float(family_lr) * _DIV_LR_MULT
    return Candidate(T_DIV_LR, int(seed), _lr, member_ref(facts),
                     **_sched_for(facts, member_ref(facts), t_step, t_eval, light))


def diverse_members(facts: Attempt0Facts, micro: int, accum: int, family_lr: float,
                    t_step: Optional[float], t_eval: Optional[float],
                    light: bool, div_lr: Optional[bool] = None) -> list:
    if member_ref(facts) <= 0 or not family_lr or family_lr <= 0:
        return []
    out = []
    if (_DIV_LR_ON if div_lr is None else div_lr):
        b = div_lr_member(facts, family_lr, t_step, t_eval, light)
        if b is not None:
            out.append(b)
    a = div_batch_member(facts, micro, accum, family_lr, t_step, t_eval, light)
    if a is not None:
        out.append(a)
    return out


def step_ladder_from_tries(tries) -> Optional[list]:
    out = []
    for t in tries or []:
        try:
            if not t.get("fits") or not t.get("secs") or t.get("error") or t.get("oom"):
                continue
            out.append({"micro": int(t.get("micro", 1)), "accum": int(t.get("accum", 1) or 1),
                        "secs": float(t["secs"]),
                        "fwd_bwd_secs": (float(t["fwd_bwd_secs"])
                                         if t.get("fwd_bwd_secs") is not None else None),
                        "opt_secs": (float(t["opt_secs"])
                                     if t.get("opt_secs") is not None else None)})
        except (TypeError, ValueError, AttributeError):
            continue
    out.sort(key=lambda e: (e["micro"], e["accum"]))
    return out or None


def _ladder_fb_opt(ladder: list) -> Optional[tuple]:
    pts = sorted({int(e["micro"]): float(e["fwd_bwd_secs"]) for e in ladder or []
                  if int(e.get("accum", 1) or 1) == 1 and e.get("fwd_bwd_secs")
                  and float(e["fwd_bwd_secs"]) > 0}.items())
    opts = [float(e["opt_secs"]) for e in ladder or []
            if e.get("opt_secs") is not None and float(e["opt_secs"]) >= 0]
    if not pts or not opts:
        return None
    return pts, statistics.median(opts)


def _fb_at(pts: list, m: int) -> float:
    m = max(1, int(m))
    if m <= pts[0][0]:
        return pts[0][1]
    if m >= pts[-1][0]:
        return pts[-1][1] * m / pts[-1][0]
    for (m0, t0), (m1, t1) in zip(pts, pts[1:]):
        if m0 <= m <= m1:
            if m == m0:
                return t0
            f = (math.log(m) - math.log(m0)) / (math.log(m1) - math.log(m0))
            return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
    return pts[-1][1]


def update_secs_ratio(ladder: Optional[list], micro: int, accum: int, micro2: int,
                      accum2: int, floor_frac: float = _DIV_FLOOR_FRAC) -> tuple:
    rows = max(1, int(micro)) * max(1, int(accum))
    rows2 = max(1, int(micro2)) * max(1, int(accum2))
    raw, src = None, "floor"
    fo = _ladder_fb_opt(ladder) if ladder else None
    if fo is not None:
        pts, opt = fo
        t0 = max(1, int(accum)) * _fb_at(pts, micro) + opt
        t1 = max(1, int(accum2)) * _fb_at(pts, micro2) + opt
        if t0 > 0 and t1 > 0:
            raw, src = t1 / t0, "ladder"
    if raw is None:
        raw = max(float(floor_frac), rows2 / rows)
    return min(1.0, _SAFETY * raw), src


def member_fixed_secs(d: "Decision") -> float:
    steps = _SAFETY * float(d.t_steps)
    if mp.price_on() and d.t_core > 0 and d.t_core_priced > 0:
        steps = float(d.t_core_priced) * float(d.t_steps) / float(d.t_core)
    return max(0.0, float(d.t_traj) - float(d.t_load_est) - steps)


def basin_frac(facts: Attempt0Facts) -> Optional[float]:
    if not _SB_TAIL_BASIN:
        return None
    best = int(facts.best_step or 0)
    steps = [int(x) for x in (getattr(facts, "a0_soup_steps", None) or []) if x is not None and int(x) > 0]
    if best <= 0 or not steps:
        return None
    return max(1e-6, min(1.0, min(steps) / float(best)))


def tail_min_steps_info(facts: Attempt0Facts, horizon: int) -> tuple:
    horizon = max(0, int(horizon or 0))
    f = basin_frac(facts)
    if f is not None and horizon > 0:
        out = (max(1, min(horizon, int(math.ceil(horizon * f)))), "a0_soup_basin", round(f, 4))
    else:
        out = (tail_min_steps(facts, horizon), "one_epoch", None)
    if packed_fill.active() and horizon > 0:
        pf = min(packed_fill.member_floor(horizon), max(1, tail_min_steps(facts, horizon)))
        if pf < out[0]:
            return pf, "packed_fill_remainder", out[2]
    return out


def tail_only_floor_info(facts: Attempt0Facts, horizon: int) -> tuple:
    floor, src, frac = tail_min_steps_info(facts, horizon)
    if _SB_TAIL_IDLE_FLOOR and src == "a0_soup_basin":
        floor2 = max(1, tail_min_steps(facts, horizon))
        if floor2 < floor:
            return floor2, "one_epoch_idle", frac, floor
    return floor, src, frac, None


def tail_launch_gap(gaps: Optional[list] = None) -> float:
    if not _SB_TAIL_GAP:
        return 0.0
    run = [float(g) for g in (gaps or []) if g is not None and g == g and g >= 0]
    return max(TAIL_GAP_SECS, sb.TAIL_SAFETY * max(run)) if run else TAIL_GAP_SECS


def first_trial_unit(facts: Attempt0Facts) -> Optional[float]:
    if not _SB_TAIL_FIRST:
        return None
    tr = [float(t) for t in (getattr(facts, "a0_trial_secs", None) or []) if t is not None and t == t and t > 0]
    if len(tr) < 2:
        return None
    return FIRST_UNIT_RATIO * max(tr[1:])


def tail_only_head(decision: "Decision", facts: Attempt0Facts, light: bool) -> tuple:
    if not (_SB_TAIL_ONLY and sb.TAIL_FILL and decision.loop and decision.extra and _LOOP_NO_CAP):
        return None, None
    best = int(facts.best_step or 0)
    if decision.candidates or best <= 0 or not facts.learning_rate or decision.t_traj <= 0:
        return None, None
    t_eval = float(decision.t_eval or 0.0)
    over = soup_overhead_secs(facts, t_eval)
    pool0 = (facts.pool_size_loop if facts.pool_size_loop is not None else facts.pool_size + 1)
    blind = _BLIND_SOUP_EVALS if (decision.extra and not (decision.loop and _LOOP_DROP_SOS)) else 0
    t_soup1 = (2.0 * t_eval + over) * (min(loop_pool_cap(1), pool0 + max(1, decision.n_evals)) + blind)
    spendable = (decision.remaining - decision.reserve - decision.devpass_reserve - t_soup1
                 - tail_launch_gap() - decision.t_prelaunch)
    fixed = float(decision.t_load_est) + member_fixed_secs(decision)
    opt = sb.TailOption(horizon=best, price=float(decision.t_traj), fixed=min(float(decision.t_traj), fixed))
    steps = min(sb.tail_steps(opt, spendable), best - 1)
    floor, src, frac = tail_min_steps_info(facts, best)
    info = {"spendable": round(spendable, 1), "steps": int(max(0, steps)), "horizon": best,
            "min_steps": int(floor), "floor_source": src, "basin_frac": frac,
            "t_soup1": round(t_soup1, 1)}
    if steps < max(1, floor) and _SB_TAIL_IDLE_FLOOR and src == "a0_soup_basin":
        floor2 = tail_min_steps(facts, best)
        if steps >= max(1, floor2):
            info.update(floor_source="one_epoch_idle", basin_floor=int(floor), min_steps=int(floor2))
    if steps < max(1, info["min_steps"]):
        info["skip"] = "min_steps"
        return None, info
    sched = _sched_for(facts, best, decision.t_step_eval_free or None, decision.t_eval or None, light)
    return Candidate(T_SEED, 43, float(facts.learning_rate), best, **sched), info


def tail_min_steps(facts: Attempt0Facts, horizon: int) -> int:
    horizon = max(0, int(horizon or 0))
    best = member_ref(facts)
    spe = int(facts.steps_per_epoch or 0)
    if horizon <= 0 or best <= 0 or spe <= 0 or spe >= best:
        return horizon
    return min(horizon, int(math.ceil(horizon * spe / best)))


def tail_fill_reserve(decision: "Decision", *, n_keep: int, trial_secs: list, interim: bool,
                      write: float, n_free: int = 0) -> tuple:
    plan_soup = float(decision.t_soup)
    parts = {"t_soup_plan": round(plan_soup, 1), "reserve": round(float(decision.reserve), 1),
             "devpass_reserve": round(float(decision.devpass_reserve), 1),
             "t_unit_measured": None, "t_soup_tail": round(plan_soup, 1), "source": "plan"}
    trials = [float(t) for t in (trial_secs or []) if t is not None and t == t and t >= 0]
    soup = plan_soup
    if trials:
        u = sb.TAIL_SAFETY * max(trials)
        soup = ((max(0, int(n_keep)) + 4 + max(0, int(n_free or 0))) * u
                + float(write or 0.0) * (1.0 if interim else 2.0))
        parts.update(t_unit_measured=round(u, 2), t_soup_tail=round(soup, 1), source="measured")
        if n_free:
            parts["n_free"] = int(n_free)
    return soup + float(decision.reserve) + float(decision.devpass_reserve), parts


def tail_fill_trials(trial_secs: list, canary: Optional[list] = None) -> list:
    trials = list(trial_secs or [])
    if not canary or len(canary) != len(trials) or not any(canary):
        return trials
    steady = [t for t, c in zip(trials, canary) if not c]
    return steady or trials


def tail_fill_inputs(decision: "Decision", facts: Attempt0Facts, horizons: dict, prices: dict,
                     *, extra: float, es_write: float, left: Optional[float], n_keep: int,
                     trial_secs: list, interim: bool, write: float, n_free: int = 0,
                     gap: float = 0.0, tail_only: bool = False) -> tuple:
    tail_true, parts = tail_fill_reserve(decision, n_keep=n_keep, trial_secs=trial_secs,
                                         interim=interim, write=write, n_free=n_free)
    spendable = (float(left) - tail_true - float(gap or 0.0)) if left is not None else None
    parts["launch_gap"] = round(float(gap or 0.0), 2)
    u = parts.get("t_unit_measured")
    extra_tail = (u + float(es_write)) if u is not None else float(extra)
    fixed = float(decision.t_load_est) + member_fixed_secs(decision)
    _floor = tail_only_floor_info if tail_only else tail_min_steps_info
    options = {}
    for k, h in horizons.items():
        if prices.get(k) is None or not h:
            continue
        price = float(prices[k])
        options[k] = sb.TailOption(horizon=int(h), price=price,
                                   fixed=min(price, fixed + float(extra)),
                                   tail_fixed=fixed + extra_tail,
                                   min_steps=_floor(facts, int(h))[0])
    _ms = _floor(facts, int(max(horizons.values() or [0])))
    parts["min_steps_source"], parts["basin_frac"] = _ms[1], _ms[2]
    if tail_only and len(_ms) > 3 and _ms[3] is not None:
        parts["basin_floor"] = _ms[3]
    return spendable, tail_true, parts, options


def _member_secs_old(cand: Candidate, facts: Attempt0Facts, t_traj: float,
                     load_est: float) -> float:
    best = member_ref(facts)
    if best <= 0 or cand.max_steps <= best:
        return t_traj
    return load_est + max(0.0, t_traj - load_est) * (cand.max_steps / best)


def member_secs(cand: Candidate, facts: Attempt0Facts, t_traj: float, load_est: float,
                a0_shape: Optional[tuple] = None, fixed: float = 0.0,
                measured: Optional[bool] = None) -> float:
    if not (_DIV_PRICE_MEASURED if measured is None else measured):
        return _member_secs_old(cand, facts, t_traj, load_est)
    best = member_ref(facts)
    if best <= 0 or (cand.max_steps <= best and not cand.per_device):
        return t_traj
    ratio = 1.0
    if cand.per_device and a0_shape is not None:
        ratio, _ = update_secs_ratio(facts.step_ladder, a0_shape[0], a0_shape[1],
                                     cand.per_device, cand.accum or 1)
    fixed = max(0.0, min(float(fixed), max(0.0, t_traj - load_est)))
    per_step = max(0.0, t_traj - load_est - fixed)
    return load_est + fixed + per_step * (cand.max_steps / best) * ratio


class A0ClockCallback(TrainerCallback):

    _clock = staticmethod(time.time)

    def __init__(self) -> None:
        self.clock = planner_clock.A0Clock()
        self._writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.clock.train_begin(self._clock())
        self._writer = sg.ASYNC_WRITER

    def on_step_end(self, args, state, control, **kwargs):
        self.clock.step_end(int(state.global_step), self._clock())

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.clock.evaluate(self._clock(), (metrics or {}).get("eval_runtime"))

    def on_save(self, args, state, control, **kwargs):
        self.clock.save(self._clock())

    def stats_dict(self) -> Optional[dict]:
        try:
            w = self._writer
            if w is not None:
                self.clock.writes = [float(x) for x in list(getattr(w, "write_secs", []) or [])]
            self.clock.eval_b1_factor = planner_clock.EVAL_B1_FACTOR
            s = self.clock.stats()
        except Exception:
            return None
        return s.model_dump() if s is not None else None


def _last_eval_runtime(trainer) -> Optional[float]:
    try:
        for e in reversed(trainer.state.log_history or []):
            if isinstance(e, dict) and e.get("eval_runtime") is not None:
                return float(e["eval_runtime"])
    except Exception:
        pass
    return None


def _online_base_loss(ckpt_avg, model2, trainer2, acc: Optional[dict], soup: dict,
                      rec, member_state: Optional[dict],
                      measured: Optional[list] = None) -> float:
    mode = eval_squeeze.reuse_base_mode()
    why = "disabled" if mode == "off" else eval_squeeze.base_reuse_check(
        rec, int(soup["n"]), trainer2, _dist_world())
    if mode == "on" and not why:
        skip_load = bool(member_state is not None and acc is not None
                         and set(acc) == set(member_state))
        if not skip_load:
            ckpt_avg._load_soup_into_model(model2, acc, int(soup["n"]))
        rt = _last_eval_runtime(trainer2)
        est = (rec.secs_per_cand if (skip_load and rec.secs_per_cand) else rt)
        eval_squeeze.LEDGER.add("base_eval", est)
        return float(rec.loss)
    l0 = ckpt_avg._online_eval_soup(model2, acc, soup["n"])
    if measured is not None:
        measured.append(True)
    return l0


class TrajectoryCallback(TrainerCallback):
    

    def __init__(self, ckpt_avg, kind: str, max_steps: int,
                 facts: Attempt0Facts, hard_deadline: Optional[float] = None,
                 eval_steps: int = 0, warmup_steps: int = 0, decay_steps: int = 0,
                 min_lr_ratio: float = 0.0, peak_lr: Optional[float] = None,
                 eval_at: tuple = (), log: Callable[[str], None] = print,
                 pool_admit: bool = True, skip_mid: bool = False):
        self.ckpt_avg = ckpt_avg
        self.skip_mid = bool(skip_mid)
        self.mid_skipped: list = []
        self.kind = kind
        self.pool_admit = bool(pool_admit)
        self.max_steps = max_steps
        self.facts = facts
        self.hard_deadline = hard_deadline
        self.eval_steps = int(eval_steps or facts.eval_steps or 0)
        self.eval_at = frozenset(int(x) for x in (eval_at or ()))
        self.warmup_steps = int(warmup_steps or 0)
        self.decay_steps = int(decay_steps or 0)
        self.min_lr_ratio = float(min_lr_ratio or 0.0)
        self.peak_lr = float(peak_lr) if peak_lr else None
        self.log = log
        self.best_loss: Optional[float] = None
        self.best_step: Optional[int] = None
        self.best_state: Optional[dict] = None
        self.last_eval_step: Optional[int] = None
        self.last_eval_loss: Optional[float] = None
        self.prev_eval_loss: Optional[float] = None
        self.dev_at_decay_mid: Optional[float] = None
        self.n_evals = 0
        self.stop_reason: Optional[str] = None
        self.lr_seen_max: Optional[float] = None
        self.lr_at_stop: Optional[float] = None
        self.frozen = False

    @property
    def decay_start(self) -> int:
        
        return max(0, self.max_steps - self.decay_steps) if self.decay_steps else self.max_steps

    def _eval_is_due(self, step: int) -> bool:
        if step <= 0:
            return False
        if self.eval_at:
            return step in self.eval_at
        if self.eval_steps > 0 and step % self.eval_steps == 0:
            return True
        if self.decay_steps > 1:
            return step == self.decay_start + self.decay_steps // 2
        return False

    def _note_lr(self, lr) -> None:
        try:
            lr = float(lr)
        except (TypeError, ValueError):
            return
        if lr != lr:
            return
        self.lr_at_stop = lr
        if self.lr_seen_max is None or lr > self.lr_seen_max:
            self.lr_seen_max = lr

    def on_log(self, args, state, control, logs=None, **kwargs):
        
        if logs:
            self._note_lr(logs.get("learning_rate"))
        return control

    @property
    def lr_over_peak(self) -> Optional[float]:
        
        peak = self.peak_lr or self.lr_seen_max
        if not peak or self.lr_at_stop is None:
            return None
        return float(self.lr_at_stop) / float(peak)

    def on_train_begin(self, args, state, control, **kwargs):
        host_snapshot.end_share("traj2.train_begin")

    def on_step_begin(self, args, state, control, **kwargs):
        host_snapshot.end_share("traj2.step_begin")

    def on_train_end(self, args, state, control, **kwargs):
        host_snapshot.end_share("traj2.train_end")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        sched = kwargs.get("lr_scheduler")
        if sched is not None:
            try:
                self._note_lr(sched.get_last_lr()[0])
            except Exception:
                pass
        due = self._eval_is_due(step)
        if due and self.skip_mid and step < self.max_steps:
            self.mid_skipped.append(int(step))
            due = False
        control.should_evaluate = due
        control.should_save = False
        if step >= self.max_steps:
            control.should_evaluate = True
            control.should_training_stop = True
            self.stop_reason = self.stop_reason or "max_steps"
        elif self.hard_deadline is not None and _r0_flag(time.time() > self.hard_deadline):
            control.should_evaluate = True
            control.should_training_stop = True
            self.stop_reason = "deadline"
            self.log(f"[sn56][traj2] {self.kind}: prazo, parando em {step}")
        if control.should_evaluate and self.mid_skipped and self.mid_skipped[-1] == step:
            self.mid_skipped.pop()
        return control

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if self.frozen:
            return
        loss = (metrics or {}).get("eval_loss")
        if model is None or loss is None or loss != loss:
            return
        self.n_evals += 1
        self.prev_eval_loss = self.last_eval_loss
        self.last_eval_loss = float(loss)
        self.last_eval_step = state.global_step
        if (self.decay_steps > 1
                and state.global_step == self.decay_start + self.decay_steps // 2):
            self.dev_at_decay_mid = float(loss)
        _pool_rank = _is_main()
        _snap_secs = None
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss, self.best_step = loss, state.global_step
            try:
                if _pool_rank:
                    _t_snap = time.perf_counter()
                    self.best_state = self.ckpt_avg._snapshot(
                        model, step=state.global_step, consumer="traj2.best_state")
                    _snap_secs = time.perf_counter() - _t_snap
            except Exception:
                pass
        try:
            if _pool_rank and self.pool_admit:
                self.ckpt_avg._consider_for_pool(model, loss, state.global_step)
        except Exception:
            pass
        if self.mid_skipped and self.n_evals == 1:
            _rt = float((metrics or {}).get("eval_runtime") or 0.0)
            _est = _rt * len(self.mid_skipped) + float(_snap_secs or 0.0)
            eval_squeeze.LEDGER.add("member_mideval", _est, n=len(self.mid_skipped))


def _record_schedule(res: "TrajectoryResult", cand: Candidate,
                     facts: Attempt0Facts, cb: "TrajectoryCallback") -> None:
    
    res.horizon_steps = cand.horizon_steps or cand.max_steps
    res.a0_horizon_steps = facts.a0_horizon
    res.lr_shape = LR_SHAPE
    res.warmup_steps = cb.warmup_steps
    res.decay_steps = cb.decay_steps
    res.stable_steps = max(0, res.horizon_steps - cb.warmup_steps - cb.decay_steps)
    res.eval_steps = cb.eval_steps
    res.min_lr_ratio = cb.min_lr_ratio
    res.stop_reason = cb.stop_reason
    res.lr_peak = cb.peak_lr or cb.lr_seen_max
    res.lr_at_stop = cb.lr_at_stop
    res.dev_at_decay_mid = cb.dev_at_decay_mid
    stop = res.stop_step or cand.max_steps
    ratio = cb.lr_over_peak
    if ratio is None:
        ratio = wsd_lr_frac(stop, res.horizon_steps, cb.warmup_steps,
                            cb.decay_steps, cb.min_lr_ratio, _WSD_DECAY_TYPE)
    res.lr_at_stop_over_peak = round(float(ratio), 6)
    res.lr_at_stop_over_peak_a0_horizon = round(
        cosine_lr_frac(stop, res.a0_horizon_steps or res.horizon_steps,
                       facts.warmup_steps, facts.min_lr_rate), 6)
    res.annealed_at_stop = bool(res.lr_at_stop_over_peak <= cb.min_lr_ratio + _ANNEAL_TOL)
    res.best_is_last_eval = bool(
        cb.best_step is not None and cb.last_eval_step is not None
        and cb.best_step == cb.last_eval_step)
    res.still_descending_at_stop = bool(res.best_is_last_eval
                                        and not res.annealed_at_stop)
    if cb.prev_eval_loss is not None and cb.last_eval_loss is not None:
        res.dev_delta_last = round(cb.prev_eval_loss - cb.last_eval_loss, 6)


def _devpass_tol_on() -> bool:
    return _env_flag("SN56_MP_DEVPASS_TOL", "1")


def post_soup_dev_pass(*, plan: "DevfitPlan", trainer2, model2, submission_dir,
                       end_time: str, reserve: float, save_secs: float,
                       eval_secs: Optional[float],
                       log: Callable[[str], None] = print, label: str = "cross_soup",
                       prepare: Optional[Callable[[], dict]] = None) -> dict:
    info = {
        "ran": False, "reason": "", "applied_to": label,
        "shipped": label, "post_soup": True,
        "variant": "annealed-devpass",
        "n_dev": plan.n_dev, "min_lr": plan.min_lr,
        "budget_secs_a0": plan.budget_secs_a0,
        "save_secs_measured": round(save_secs, 2),
        "eval_secs": round(eval_secs, 2) if eval_secs else None,
    }
    if not plan.enabled:
        info["reason"] = plan.skip_reason or "devfit_plan_disabled"
        return info

    remaining = _remaining_secs(end_time)
    if _multi():
        remaining = _r0(remaining)
    if remaining is None:
        info["reason"] = "no_end_time"
        return info

    ev = float(eval_secs or _EVAL_SECS_FALLBACK)
    floor = 3.0 * ev + 2.0 * save_secs
    spendable = remaining - reserve
    save_est = planner_clock.devpass_gate_save(save_secs)
    overshoot_ok = (max(0.0, save_est - 2.5 * float(save_secs or 0.0))
                    if planner_clock.devpass_hard_on() and _devpass_tol_on() else 0.0)
    if planner_clock.devpass_hard_on() and _devpass_tol_on():
        info["overshoot_ok"] = round(overshoot_ok, 2)
    deadline = time.time() + spendable
    if _multi():
        deadline = _r0(deadline)
    info.update(remaining=round(remaining, 1), reserve=round(reserve, 1),
                spendable=round(spendable, 1), floor=round(floor, 1),
                save_secs_est=round(save_est, 1))
    if spendable < floor:
        info["reason"] = "no_budget"
        log(f"[sn56][devfit] sem orçamento pós-soup: {spendable:.0f}s gastáveis "
            f"< piso {floor:.0f}s — mantendo o soup")
        return info

    if prepare is not None:
        _t_prep = time.time()
        _prep_exc = None
        try:
            info["prepare"] = prepare()
        except Exception as exc:
            _prep_exc = exc
        if _multi() and not _all_ok(_prep_exc is None) and _prep_exc is None:
            _prep_exc = RuntimeError("devfit prepare failed on a peer rank")
        info["prepare_secs"] = round(time.time() - _t_prep, 2)
        if _prep_exc is not None:
            info["reason"] = f"prepare_failed:{type(_prep_exc).__name__}: {_prep_exc}"[:300]
            log(f"[sn56][devfit] preparo do treinador falhou ({_prep_exc}) — mantendo o soup")
            return info

    log(f"[sn56][devfit] pós-soup: {plan.n_dev} amostras de dev, 1 época "
        f"@ lr={plan.min_lr:.2e}, prazo {spendable:.0f}s")

    t0 = time.time()
    try:
        res = dev_pass.run_dev_pass(
            trainer2,
            submission_dir=submission_dir,
            min_lr=plan.min_lr,
            max_grad_norm=plan.max_grad_norm,
            train_per_device=plan.train_per_device,
            train_grad_accum=plan.train_grad_accum,
            local_rank=plan.local_rank, log=log,
            deadline=deadline, save_secs_est=save_est,
            **({"overshoot_ok": overshoot_ok} if "overshoot_ok" in info else {}),
            memory_proof=plan.memory, collective_timeout=plan.collective_timeout,
            restore_timeout=plan.ddp_timeout,
            micro_batch=plan.dev_micro_batch,
        )
        info.update({k: v for k, v in (res or {}).items()})
    except Exception as exc:
        info["reason"] = f"failed:{type(exc).__name__}: {exc}"[:300]
        info["ran"] = False
        log(f"[sn56][devfit] pós-soup falhou ({exc}), o soup continua sendo a submissão")
    info["secs"] = round(time.time() - t0, 2)

    if _multi():
        info["ran"] = _all_ok(bool(info.get("ran")))
    if info.get("ran"):
        info["shipped"] = f"{label}_devfit"
    return info


def a0_soup_dev_pass(*, plan: "DevfitPlan", ckpt_avg, a0_state: Optional[dict],
                     model_builder, trainer_builder, training_args, submission_dir: str,
                     end_time: str, reserve: float, load_secs: float, save_secs: float,
                     eval_secs: Optional[float],
                     log: Callable[[str], None] = print) -> dict:
    info = {"ran": False, "reason": "", "applied_to": "a0_soup", "shipped": "a0_soup",
            "post_soup": True, "variant": "annealed-devpass", "rebuilt": True}
    def _skip(reason: str) -> dict:
        info["reason"] = reason
        return info

    if not plan.enabled:
        return _skip(plan.skip_reason or "devfit_plan_disabled")
    owns = bool(sg.submission_owner(getattr(training_args, "output_dir", "") or "")["owns"])
    if _r0_flag(not (owns and a0_state is not None)):
        return _skip("shipped_model_not_in_memory")
    left = _remaining_secs(end_time)
    left = _r0(left) if _multi() else left
    ev = float(eval_secs or _EVAL_SECS_FALLBACK)
    floor = 3.0 * ev + 2.0 * float(save_secs or 0.0)
    need = float(load_secs or 0.0) + reserve + floor
    info.update(remaining=(round(left, 1) if left is not None else None),
                load_secs_est=round(float(load_secs or 0.0), 1), floor=round(floor, 1))
    if left is None or left < need:
        return _skip("no_budget")
    t0 = time.time()
    model = trainer = None
    exc_ = None
    try:
        model = model_builder()
        args = copy.copy(training_args)
        trainer = trainer_builder(model, args)
        if getattr(args, "gradient_checkpointing", False) and \
                hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=getattr(args, "gradient_checkpointing_kwargs", None))
    except Exception as exc:
        exc_ = exc
    if _multi() and not _all_ok(exc_ is None) and exc_ is None:
        exc_ = RuntimeError("rebuild failed on a peer rank")
    if exc_ is not None:
        _release_trainer(trainer)
        trainer = model = None
        return _skip(f"rebuild_failed:{type(exc_).__name__}: {exc_}"[:300])
    if _is_main():
        ckpt_avg._restore(model, a0_state)
    ckpt_avg._bcast_params(model)
    if _multi():
        trainer.model_wrapped = trainer.accelerator.prepare_model(trainer.model)
    info["rebuild_secs"] = round(time.time() - t0, 2)
    res = post_soup_dev_pass(plan=plan, trainer2=trainer, model2=model,
                             submission_dir=submission_dir, end_time=end_time,
                             reserve=reserve, save_secs=save_secs, eval_secs=eval_secs,
                             log=log, label="a0_soup")
    res["rebuilt"] = True
    res["rebuild_secs"] = info["rebuild_secs"]
    _release_trainer(trainer)
    model = trainer = None
    _collect()
    return res


T2_TRAINED = "trained"
T2_REFUSED = "refused"
T2_RECOVERED = "recovered"
T2_FAILED = "failed"


def devfit_route(origin: Optional[str], enabled: bool = True) -> tuple:
    if origin == T2_TRAINED:
        return "trained", ""
    if not enabled:
        return "skip", "recovered_trainer"
    if origin in (T2_REFUSED, T2_RECOVERED):
        return "untrained", ""
    if origin == T2_FAILED:
        return "skip", "member_train_failed"
    return "skip", "recovered_trainer"


def prepare_untrained_for_devfit(trainer, model) -> dict:
    info = {"sharded": dev_pass._sharded(trainer) or None, "gc_enabled": False,
            "gc_already": False, "ddp_wrapped": False, "ddp_already": False}
    if info["sharded"]:
        return info
    args = trainer.args
    if getattr(args, "gradient_checkpointing", False) and \
            hasattr(model, "gradient_checkpointing_enable"):
        if getattr(model, "is_gradient_checkpointing", False):
            info["gc_already"] = True
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=getattr(args, "gradient_checkpointing_kwargs", None))
            info["gc_enabled"] = True
    if _multi():
        bare = getattr(trainer, "model_wrapped", None) in (None, trainer.model)
        all_bare, all_wrapped = _all_ok(bare), _all_ok(not bare)
        if not (all_bare or all_wrapped):
            raise RuntimeError("ranks disagree on whether the trainer is DDP-wrapped")
        if all_bare:
            trainer.model_wrapped = trainer.accelerator.prepare_model(trainer.model)
            info["ddp_wrapped"] = True
        else:
            info["ddp_already"] = True
    return info


def loop_pool_survivors(pool: list, keep_steps: set) -> int:
    
    n = 0
    for e in pool or []:
        if e.get("src", "a0") != "a0" or int(e.get("step", -1)) in keep_steps:
            n += 1
    return n


def prune_union_pool(ckpt_avg, keep_steps: set, hold: Optional[list] = None) -> dict:
    pool = list(getattr(ckpt_avg, "pool", []) or [])
    kept, removed = [], []
    n_held = 0
    for e in pool:
        src = e.get("src", "a0")
        if src == "a0" and int(e.get("step", -1)) not in keep_steps:
            removed.append(int(e.get("step", -1)))
            if hold is not None and e.get("state") is not None:
                hold.append(fc.entry(e["state"], loss=e["loss"], step=int(e.get("step", -1)),
                                     src="a0", origin=fc.SRC_A0))
                n_held += 1
            e["state"] = None
        else:
            kept.append(e)
    ckpt_avg.pool = kept
    if removed and hold is None:
        gc.collect()
    out = {
        "keep_steps": sorted(keep_steps),
        "kept_a0_steps": sorted(int(e.get("step", -1)) for e in kept
                                if e.get("src", "a0") == "a0"),
        "removed_steps": sorted(removed),
        "removed": len(removed),
        "pool_size": len(kept),
    }
    if hold is not None:
        out["held"] = n_held
    return out


def restore_best_local(ckpt_avg, model, exc: Optional[BaseException] = None) -> bool:
    try:
        if _is_main() and getattr(ckpt_avg, "best_state", None) is not None:
            ckpt_avg._restore(model, ckpt_avg.best_state)
            return True
    except Exception:
        pass
    return False


def free_held_drop(held: list, why: str, keep: int = 0) -> list:
    order = sorted([e for e in held if e.get("state") is not None], key=lambda e: e["loss"])
    dropped = order[max(0, int(keep)):]
    for e in dropped:
        e["state"] = None
    held[:] = order[:max(0, int(keep))]
    if dropped:
        gc.collect()
    return [dict(fc.describe(e), why=why) for e in dropped]


def loop_pool_pin(keep_steps: set, endpoint_step: int,
                  extra_ends: Optional[dict] = None) -> Callable[[dict], bool]:
    
    keep = {int(s) for s in keep_steps}
    end = int(endpoint_step)
    extra = extra_ends if extra_ends is not None else {}

    def _pin(e: dict) -> bool:
        src = str(e.get("src", "a0"))
        step = int(e.get("step", -1))
        if src == "a0_soup":
            return True
        if src == "a0":
            return step in keep
        if src.startswith(T_SEED):
            return step == end
        if src in extra:
            return step == int(extra[src])
        return False
    return _pin


def _collect() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _cuda_alloc() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.memory_allocated())
    except Exception:
        return None


_INTERIM: Optional["async_save.InterimWriter"] = None


def _member_host_terms(ckpt_avg, model) -> Optional[dict]:
    b = ckpt_avg._ram_budget()
    if b.avail is None:
        return None
    snap = int(ckpt_avg._trainable_snapshot_bytes(model))
    peak = int(ckpt_avg._soup_peak_bytes(model))
    reserve = int(host_snapshot.host_reserve_bytes(b.limit))
    live = len(ckpt_avg._live_pool())
    give = max(0, live - host_snapshot.POOL_CAP_FLOOR) * snap
    return {"need": snap, "room": int(b.avail) - reserve - peak + give, "reserve": reserve,
            "soup_peak": peak, "pool_live": live, "pool_give": give, **b.fields()}


def _finite_or_none(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if (x == x and x not in (float("inf"), float("-inf"))) else None


def _same_loss(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-9


def _early_ship_soup(model, loss, n_members, *, submission_dir, record: dict, log,
                     save_fn=None) -> bool:
    loss = _finite_or_none(loss)
    st = sg.ship_marker_state()
    shipped = _finite_or_none(st.get("shipped_loss")) if st else None
    if loss is None or shipped is None or not (loss < shipped - 1e-6):
        return False
    sg.ship_marker_update(activity="cross_soup_early_write")
    ok = bool((save_fn or dev_pass._save_weights_only)(model, submission_dir, lambda m: log(m)))
    if ok:
        sg.ship_marker_update(shipped="cross_soup_pass1", shipped_loss=loss,
                              activity="cross_soup_pass2")
        _close_interim("cross_soup_early")
        record.update(saved=True, loss=loss, n_members=int(n_members))
    return ok


def _close_interim(why: str) -> None:
    global _INTERIM
    w, _INTERIM = _INTERIM, None
    if w is not None:
        try:
            w.close(discard=True)
        except Exception:
            pass


def cap_member_micro(cand: "Candidate", cap: Optional[int], a0_micro: int,
                     a0_accum: int) -> "Candidate":
    micro, accum = ((int(cand.per_device), max(1, int(cand.accum))) if cand.per_device > 0
                    else (int(a0_micro), int(a0_accum)))
    if not cap or micro <= int(cap):
        return cand
    rows = micro * accum
    m = max(1, int(cap))
    return dataclasses.replace(cand, per_device=m, accum=max(1, int(math.ceil(rows / m))))


def member_args(training_args, cand, facts: "Attempt0Facts"):
    args2 = copy.copy(training_args)
    args2.seed = cand.seed
    args2.data_seed = cand.seed
    args2.learning_rate = cand.learning_rate
    args2.max_steps = cand.max_steps
    a0_micro = int(getattr(training_args, "per_device_train_batch_size", 1) or 1)
    a0_accum = int(getattr(training_args, "gradient_accumulation_steps", 1) or 1)
    micro, accum = ((int(cand.per_device), max(1, int(cand.accum))) if cand.per_device > 0
                    else (a0_micro, a0_accum))
    proof = facts.memory_proof
    if proof is None:
        proof = member_guard.MemoryProof(
            a0_micro=a0_micro, a0_accum=a0_accum,
            a0_eval_micro=int(getattr(training_args, "per_device_eval_batch_size", 1) or 1),
            a0_gc=bool(getattr(training_args, "gradient_checkpointing", False)))
    v = member_guard.member_shape(
        micro=micro, accum=accum, gc=bool(getattr(args2, "gradient_checkpointing", False)),
        eval_micro=int(getattr(args2, "per_device_eval_batch_size", 1) or 1), proof=proof)
    member_guard.apply_shape(args2, v)
    return args2, v


def _unwrap_module(model):
    while model is not None and hasattr(model, "module"):
        model = model.module
    return model


def _free_module_storage(module) -> int:
    if module is None:
        return 0
    n = 0
    seen: set = set()
    with torch.no_grad():
        for p in module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            p.grad = None
            if p.device.type == "meta" or p.numel() == 0:
                continue
            if p.device.type == "cuda":
                n += int(p.numel()) * int(p.element_size())
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        for m in module.modules():
            for k, b in list(m._buffers.items()):
                if b is None or id(b) in seen or b.device.type == "meta" or b.numel() == 0:
                    continue
                seen.add(id(b))
                if b.device.type == "cuda":
                    n += int(b.numel()) * int(b.element_size())
                m._buffers[k] = torch.empty(0, dtype=b.dtype, device=b.device)
    return int(n)


def _referrer_report(obj, limit: int = 12) -> list:
    out: list = []
    if obj is None:
        return out
    refs = None
    try:
        refs = gc.get_referrers(obj)
        for r in refs:
            if len(out) >= limit:
                break
            if isinstance(r, types.FrameType):
                out.append(f"frame:{r.f_code.co_name}")
            elif isinstance(r, dict):
                key = next((k for k, v in r.items() if v is obj), None)
                owner = next((o for o in gc.get_referrers(r)
                              if getattr(o, "__dict__", None) is r), None)
                out.append(f"{type(owner).__module__}.{type(owner).__qualname__}.{key}"
                           if owner is not None else f"dict[{str(key)[:60]}]")
            elif isinstance(r, (list, tuple, set)):
                out.append(f"{type(r).__name__}(len={len(r)})")
            else:
                out.append(f"{type(r).__module__}.{type(r).__qualname__}")
    except Exception as exc:
        out.append(f"report_failed:{type(exc).__name__}")
    finally:
        refs = None
    return out


def _cuda_residue(limit: int = 8) -> list:
    rows: dict = {}
    try:
        for o in gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda and o.numel():
                    st = o.untyped_storage()
                    k = st.data_ptr()
                    if k not in rows:
                        rows[k] = (int(st.nbytes()), list(o.shape)[:4], str(o.dtype),
                                   type(o).__name__)
            except Exception:
                continue
    except Exception as exc:
        return [f"residue_failed:{type(exc).__name__}"]
    top = sorted(rows.values(), key=lambda r: -r[0])[:limit]
    return [{"bytes": b, "shape": sh, "dtype": dt, "type": ty} for b, sh, dt, ty in top]


_RESIDUE_REPORT_BYTES = 1 << 30


def _release_trainer(trainer) -> dict:
    info: dict = {"storage_bytes": 0}
    if trainer is None:
        return info
    handler = getattr(trainer, "callback_handler", None)
    _models = []
    for m in (getattr(trainer, "model_wrapped", None), getattr(trainer, "model", None)):
        m = _unwrap_module(m)
        if m is not None and all(m is not x for x in _models):
            _models.append(m)

    def _opt():
        info.update(dev_pass.release_optimizer_state(getattr(trainer, "optimizer", None)))

    for step in (_opt,
                 lambda: trainer.accelerator.free_memory(),
                 lambda: setattr(trainer, "optimizer", None),
                 lambda: setattr(trainer, "lr_scheduler", None),
                 lambda: setattr(trainer, "model_wrapped", None),
                 lambda: setattr(trainer, "model", None),
                 lambda: setattr(handler, "model", None),
                 lambda: setattr(handler, "optimizer", None),
                 lambda: setattr(handler, "lr_scheduler", None),
                 lambda: setattr(handler, "train_dataloader", None),
                 lambda: setattr(handler, "eval_dataloader", None)):
        try:
            step()
        except Exception:
            pass
    for m in _models:
        try:
            info["storage_bytes"] += _free_module_storage(m)
        except Exception as exc:
            info["storage_failed"] = f"{type(exc).__name__}: {exc}"[:200]
    _models = None
    return info


def _remaining_secs(end_time: str) -> Optional[float]:
    try:
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=datetime.timezone.utc)
        return (end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    except Exception:
        return None


def _restore_rng(rng_state) -> bool:
    
    if not rng_state:
        return False
    try:
        torch.set_rng_state(rng_state["cpu"])
        if rng_state.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        return True
    except Exception:
        return False


_SPIKE_WINDOW = 8
_SPIKE_MIN_EXCESS = 0.5


_SPIKE_LEVEL_WINDOW = 32


def spike_reference(losses: list, level: bool) -> float:
    if not level:
        return min(losses)
    w = _SPIKE_LEVEL_WINDOW
    if len(losses) < w:
        return statistics.median(losses)
    return min(statistics.median(losses[i - w + 1:i + 1]) for i in range(w - 1, len(losses)))


def _train_diverged(trainer2, spike_mult: float = 3.0, level: bool = False) -> tuple:
    
    try:
        hist = getattr(getattr(trainer2, "state", None), "log_history", None) or []
    except Exception:
        return False, ""
    losses = [e["loss"] for e in hist
              if isinstance(e, dict) and "loss" in e and e.get("loss") is not None]
    if not losses:
        return False, ""
    for v in losses:
        if v != v or v in (float("inf"), float("-inf")):
            return True, "nan_inf_train_loss"
    running_min = spike_reference(losses, level)
    tail = statistics.median(losses[-_SPIKE_WINDOW:])
    if (running_min > 0 and tail > spike_mult * running_min
            and tail - running_min > _SPIKE_MIN_EXCESS):
        return True, f"train_spike({tail:.3f}>{spike_mult:g}x{running_min:.3f})"
    return False, ""


def emit_spare(stage: str, *, end_time: str, decision: Optional[Decision],
               results: Optional[list] = None) -> dict:
    
    left = _remaining_secs(end_time)
    d = decision or Decision()
    results = [r for r in (results or []) if r.launched]
    realised = [r.secs + r.load_secs for r in results]
    cost = (sum(realised) / len(realised)) if realised else d.t_traj
    spendable = (left - d.reserve - d.t_soup) if left is not None else None
    n_further = (max(0, int(spendable // cost))
                 if (spendable is not None and cost > 0) else None)
    shortfall = (max(0.0, ((n_further or 0) + 1) * cost - spendable)
                 if spendable is not None and cost > 0 else None)
    payload = dict(
        stage=stage,
        remaining=round(left, 1) if left is not None else None,
        reserve=d.reserve,
        t_soup_est=round(d.t_soup, 1),
        spendable=round(spendable, 1) if spendable is not None else None,
        t_traj_est=round(d.t_traj, 1),
        t_traj_realised=round(cost, 1) if realised else None,
        n_launched=len(results),
        secs_spent=round(sum(realised), 1) if realised else 0.0,
        n_further_fit=n_further,
        shortfall_to_next=round(shortfall, 1) if shortfall is not None else None,
        kinds=[r.kind for r in results],
    )
    return payload


_MULTI_GPU = _env_flag("SN56_ST_MULTI_GPU", "1")


def _dist_world() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except Exception:
        pass
    return 1


def _multi() -> bool:
    return _dist_world() > 1


def _is_main() -> bool:
    if not _multi():
        return True
    return int(torch.distributed.get_rank()) == 0


def members_on_world(world_size: int) -> bool:
    return int(world_size or 1) == 1 or bool(_MULTI_GPU)


def sharded_kind(training_args) -> str:
    if training_args is not None:
        if getattr(training_args, "deepspeed", None):
            return "deepspeed"
        if len(getattr(training_args, "fsdp", None) or "") > 0:
            return "fsdp"
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "").strip().lower() in ("1", "true"):
        return "deepspeed"
    if os.environ.get("ACCELERATE_USE_FSDP", "").strip().lower() in ("1", "true"):
        return "fsdp"
    return ""


def _r0(value):
    if not _multi():
        return value
    box = [value if _is_main() else None]
    torch.distributed.broadcast_object_list(box, src=0)
    return box[0]


def _all_ok(ok: bool) -> bool:
    if not _multi():
        return bool(ok)
    got = [None] * _dist_world()
    torch.distributed.all_gather_object(got, bool(ok))
    return all(bool(g) for g in got)


MEMBER_DESYNC_EXIT = 86
_TRAIN_AGREE_SEQ = [0]


class TrainVerdict(BaseModel):
    verdict: Literal["ok", "fail", "desync"]
    statuses: list[Optional[str]] = []
    waited_secs: float = 0.0
    store: bool = True


def _is_comm_error(exc: Optional[BaseException]) -> bool:
    if exc is None:
        return False
    if isinstance(exc, torch.distributed.DistError) and not isinstance(
            exc, torch.distributed.DistStoreError):
        return True
    msg = str(exc)
    return any(s in msg for s in ("ProcessGroupNCCL", "ProcessGroupGloo", "NCCL communicator",
                                  "gloo/transport", "Connection closed by peer",
                                  "Watchdog caught collective"))


def store_verdict(store, tag: str, rank: int, world: int, status: str,
                  wait_secs: float) -> TrainVerdict:
    store.set(f"{tag}/r{rank}", status)
    keys = [f"{tag}/r{r}" for r in range(world)]
    t0 = time.perf_counter()
    statuses: list[Optional[str]] = [None] * world
    try:
        store.wait(keys, datetime.timedelta(seconds=max(0.1, float(wait_secs))))
        statuses = [store.get(k).decode() for k in keys]
        if any(s_ == "comm" for s_ in statuses):
            cand = "desync"
        else:
            cand = "ok" if all(s_ == "ok" for s_ in statuses) else "fail"
    except torch.distributed.DistStoreError:
        for i, k in enumerate(keys):
            try:
                if store.check([k]):
                    statuses[i] = store.get(k).decode()
            except Exception:
                pass
        cand = "desync"
    got = store.compare_set(f"{tag}/verdict", "", cand)
    got = got.decode() if isinstance(got, (bytes, bytearray)) else str(got)
    return TrainVerdict(verdict=got if got in ("ok", "fail", "desync") else "desync",
                        statuses=statuses, waited_secs=round(time.perf_counter() - t0, 3))


def _train_verdict(exc: Optional[BaseException], wait_secs: float) -> TrainVerdict:
    status = "ok" if exc is None else ("comm" if _is_comm_error(exc) else "fail")
    if not _multi():
        return TrainVerdict(verdict="ok" if exc is None else "fail", statuses=[status])
    try:
        store = torch.distributed.distributed_c10d._get_default_store()
    except Exception:
        store = None
    if store is None:
        ok = _all_ok(exc is None)
        return TrainVerdict(verdict="ok" if ok else "fail", store=False)
    _TRAIN_AGREE_SEQ[0] += 1
    return store_verdict(store, f"sn56/member_train/{_TRAIN_AGREE_SEQ[0]}",
                         int(torch.distributed.get_rank()), _dist_world(), status, wait_secs)


def _exit_desync(label: str, verdict: TrainVerdict, exc: Optional[BaseException]) -> None:
    sys.stderr.write(f"[sn56][traj2] {label}: ranks desynchronised in train() "
                     f"(statuses {verdict.statuses}); exiting {MEMBER_DESYNC_EXIT}\n")
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(MEMBER_DESYNC_EXIT)


def _flag_device():
    try:
        if torch.distributed.get_backend() == "nccl" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
    except Exception:
        pass
    return torch.device("cpu")


def _r0_flag(value: bool) -> bool:
    if not _multi():
        return bool(value)
    t = torch.tensor([1.0 if value else 0.0], device=_flag_device())
    torch.distributed.broadcast(t, src=0)
    return bool(t.item() > 0.5)


def _drop_peer_pool(ckpt_avg) -> int:
    n = len(getattr(ckpt_avg, "pool", []) or [])
    for e in getattr(ckpt_avg, "pool", []) or []:
        e["state"] = None
    ckpt_avg.pool = []
    ckpt_avg.best_state = None
    gc.collect()
    return n


def _state_size(state: Optional[dict]) -> tuple:
    if not state:
        return 0, 0, 0
    b = n = big = 0
    for t in state.values():
        k = int(t.numel())
        b += k * int(t.element_size())
        n += k
        big = max(big, k)
    return b, n, big


def online_soup_fits(snap_bytes: int, numel: int, room: Optional[int],
                     snaps: int = _SB_ONLINE_SNAPS) -> tuple:
    need = int(4 * int(numel) + int(snaps) * int(snap_bytes))
    return (room is None or int(room) >= need), need


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt()) if t.numel() else 0.0


@torch.no_grad()
def perturb_init(model, a0_state: dict, *, lam: float = _PERTURB_LAMBDA,
                 kappa: float = _PERTURB_KAPPA, seed: int = 0) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    named = {n: p for n, p in _unwrap_module(model).named_parameters() if p.requires_grad}
    pairs = {}
    for n in named:
        if ".lora_B." in n:
            a = n.replace(".lora_B.", ".lora_A.")
            if a in named and a in a0_state and n in a0_state:
                pairs[n] = a
    a_side = set(pairs.values())
    st = {"tensors": 0, "lora_pairs": len(pairs), "floor_bound": 0, "missing": 0,
          "changed_frac_min": None, "changed_frac_mean": None, "sigma_over_rms_delta_med": None,
          "lam": float(lam), "kappa": float(kappa), "seed": int(seed)}
    fracs, ratios = [], []
    for n, p in named.items():
        if n not in a0_state:
            st["missing"] += 1
            continue
        w0 = p.data.detach().float().cpu()
        wa = a0_state[n].float()
        start = w0 + float(lam) * (wa - w0)
        eps = (float(torch.finfo(p.dtype).eps) if p.dtype.is_floating_point else 0.0)
        noise = None
        if n in pairs:
            A = a0_state[pairs[n]].float()
            E = torch.randn(wa.shape, generator=gen, dtype=torch.float32)
            rms_d = _rms(wa @ A)
            rms_p = _rms(E @ A)
            s_ = (float(kappa) * rms_d / rms_p) if rms_p > 0 else 0.0
            floor = eps * _rms(start)
            if s_ < floor:
                s_ = floor
                st["floor_bound"] += 1
            noise = s_ * E
            if rms_d > 0:
                ratios.append(s_ * rms_p / rms_d)
        elif n not in a_side:
            delta = wa - w0
            sigma = float(kappa) * _rms(delta)
            floor = eps * _rms(start)
            if sigma < floor:
                sigma = floor
                st["floor_bound"] += 1
            rd = _rms(delta)
            if rd > 0:
                ratios.append(sigma / rd)
            noise = sigma * torch.randn(wa.shape, generator=gen, dtype=torch.float32)
        t = start if noise is None else start + noise
        cast = t.to(p.dtype)
        if noise is not None:
            fracs.append(float((cast != start.to(p.dtype)).float().mean()))
        p.data.copy_(cast.to(p.device))
        st["tensors"] += 1
        del w0, wa, start, t, cast, noise
    if fracs:
        st["changed_frac_min"] = round(min(fracs), 6)
        st["changed_frac_mean"] = round(sum(fracs) / len(fracs), 6)
    if ratios:
        st["sigma_over_rms_delta_med"] = round(statistics.median(ratios), 6)
    host_snapshot.weights_changed("traj2.perturb_init")
    return st


def run(*, ckpt_avg, trainer, training_args, train_request, facts: Attempt0Facts,
        model_builder: Callable[[], object],
        trainer_builder: Callable[[object, object], object],
        rng_state=None, devfit: Optional[DevfitPlan] = None,
        log: Callable[[str], None] = print) -> Outcome:
    
    out = Outcome()
    try:
        return _run(out, ckpt_avg=ckpt_avg, trainer=trainer,
                    training_args=training_args, train_request=train_request,
                    facts=facts, model_builder=model_builder,
                    trainer_builder=trainer_builder, rng_state=rng_state,
                    devfit=devfit or DevfitPlan(), log=log)
    except Exception as exc:
        out.reason = f"error:{type(exc).__name__}: {exc}"[:500]
        log(f"[sn56][traj2] falhou ({exc})")
        return out
    finally:
        _close_interim("second_trajectory_end")


def _gate(facts: Attempt0Facts, training_args=None) -> Optional[str]:
    if not _env_flag("SN56_SECOND_TRAJECTORY"):
        return "disabled"
    if not facts.is_final_run:
        return "not_final_run"
    if not facts.owns_submission:
        return "shipped_not_current_attempt"
    if facts.world_size != 1:
        _sh = sharded_kind(training_args)
        if _sh:
            return f"sharded:{_sh}"
    if facts.avg_mode != "ram":
        return f"avg_mode={facts.avg_mode}"
    if facts.world_size != 1:
        if not _MULTI_GPU:
            return f"world_size={facts.world_size}"
        if _dist_world() != int(facts.world_size):
            return f"world_size={facts.world_size}:process_group={_dist_world()}"
    if not facts.best_step:
        return "no_best_step"
    return None


def _run(out: Outcome, *, ckpt_avg, trainer, training_args, train_request, facts,
         model_builder, trainer_builder, rng_state, devfit, log) -> Outcome:
    gate = _gate(facts, training_args)
    _mr = _multi()
    if _mr:
        gate = _r0(gate)
    if gate:
        out.reason = gate
        log(f"[sn56][traj2] inativo ({gate})")
        return out

    end_time = train_request.get("end_time", "")
    remaining = _remaining_secs(end_time)
    if _mr:
        remaining = _r0(remaining)
    if remaining is None:
        out.reason = "no_end_time"
        return out

    set_pool_facts(facts, ckpt_avg)
    _devfit_on = bool(getattr(devfit, "enabled", False))
    _devpass_legacy = (max(_DEVPASS_RESERVE_MIN_SECS,
                           float(getattr(devfit, "budget_secs_a0", 0.0) or 0.0))
                       if _devfit_on else 0.0)
    if _mr:
        _f0, _dl0, _don0 = _r0((asdict(facts), _devpass_legacy, _devfit_on))
        for _k, _v in _f0.items():
            setattr(facts, _k, _v)
        _devpass_legacy, _devfit_on = _dl0, _don0
    _tail = tail_reserves(facts, _devfit_on, _devpass_legacy,
                          measured_devpass=(planner_clock.devpass_measured_on()
                                            and not sb.multi_gpu_guard(_dist_world()).on
                                            and sft_objective(trainer)))
    decision = plan(facts, remaining, reserve=_tail.reserve,
                    devpass_reserve=_tail.devpass_reserve)
    if _mr:
        decision = _r0(decision)
        _clk0 = facts.clock or {}
    out.decision = decision
    _aff = compute_affordance.post_a0_affordance(
        decision.n_fit, t_step=(facts.clock or {}).get("t_step"), remaining=remaining)


    for kind in (T_SEED, T_LR):
        c = next((c for c in decision.candidates if c.kind == kind), None)

    if (not decision.launch and decision.reason == "no_fit" and _dist_world() <= 1
            and member_phase_switches(1).spare_budget and sft_objective(trainer)):
        _to_head, _to_info = tail_only_head(decision, facts,
                                            light=bool(decision.loop and _LOOP_EVAL_LIGHT))
        if _to_head is not None:
            decision.candidates = [_to_head]
            decision.reason = "no_fit+tail_only"
            log(f"[sn56][traj2] nenhum membro completo cabe; cauda: t_seed "
                f"{_to_info['steps']}/{_to_info['horizon']} passos cabem")

    if not decision.launch:
        out.reason = decision.reason or "no_fit"
        out.spare = emit_spare("no_launch", end_time=end_time, decision=decision)
        log(f"[sn56][traj2] sem folga ({out.reason}): "
            f"restam {remaining:.0f}s, precisa {decision.t_traj:.0f}s + "
            f"soup {decision.t_soup:.0f}s + reserva {decision.reserve:.0f}s "
            f"(faltam {decision.shortfall_next:.0f}s)")
        return out

    out.mutated = True
    global _INTERIM
    _close_interim("restart")
    _interim_best = (float(ckpt_avg.best_loss)
                     if getattr(ckpt_avg, "best_loss", None) is not None else None)
    if (_is_main() and _interim_best is not None
            and async_save.interim_enabled(int(facts.world_size or 1))):
        try:
            _INTERIM = async_save.InterimWriter(train_request["submission_dir"])
        except Exception:
            _INTERIM = None
    _clk0 = facts.clock_stats
    _ct_secs = member_guard.collective_timeout_secs(
        (_clk0.t_step if _clk0 is not None else decision.t_step_eval_free),
        (_clk0.t_point if _clk0 is not None else decision.t_eval)) if _mr else None
    _ddp_timeout = float(getattr(training_args, "ddp_timeout", 1800) or 1800)

    _dual_on = bool(decision.extra and _COLD_LR
                    and any(c.kind == T_LR for c in decision.candidates))
    _a0_single_dev: Optional[float] = None
    _a0_single_src = "none"
    if _dual_on:
        _a0_losses = [float(e["loss"]) for e in (getattr(ckpt_avg, "pool", []) or [])
                      if e.get("src", "a0") == "a0" and e.get("loss") is not None
                      and e["loss"] == e["loss"]]
        if _a0_losses:
            _a0_single_dev, _a0_single_src = min(_a0_losses), "pool_min"
        elif facts.best_loss is not None:
            _a0_single_dev, _a0_single_src = float(facts.best_loss), "facts.best_loss"
    if _mr:
        _a0_single_dev, _a0_single_src = _r0((_a0_single_dev, _a0_single_src))

    _a0_soup_state = getattr(ckpt_avg, "best_state", None)
    if getattr(ckpt_avg, "best_state", None) is not None:
        ckpt_avg.pool.append({"loss": ckpt_avg.best_loss, "step": -1,
                              "src": "a0_soup", "state": ckpt_avg.best_state,
                              "members": int(getattr(ckpt_avg, "soup_n_members", 1) or 1)})
        ckpt_avg.pool.sort(key=lambda e: e["loss"])
    ckpt_avg.soup_max = UNION_SOUP_MAX
    _sw_early = member_phase_switches()
    _free_gate = fc.gate(decision.n_fit, path_ok=bool(
        decision.loop and _LOOP_POOL_PRUNE and _sw_early.spare_budget and decision.extra
        and _LOOP_NO_CAP))
    _free_held: list = []
    _free_hold_a0 = bool(_free_gate.active and _free_gate.a0_held and _is_main())
    if decision.loop and _LOOP_POOL_PRUNE:
        _pruned = prune_union_pool(ckpt_avg, facts.a0_keep_steps,
                                   hold=(_free_held if _free_hold_a0 else None))
        log(f"[sn56][traj2] pool a0 podado: mantidos {_pruned['kept_a0_steps']}, "
            f"removidos {_pruned['removed']} -> pool {_pruned['pool_size']}")
        ckpt_avg.pool_pin = loop_pool_pin(facts.a0_keep_steps, int(facts.best_step))
    if decision.loop and _LOOP_NO_CAP:
        _cap = loop_pool_cap(len(decision.candidates))
        ckpt_avg.soup_max = _cap if _LOOP_POOL_PRUNE else max(UNION_SOUP_MAX, _cap)
    _diverged_srcs: set = set()
    _running_soup_losses: list = []

    if _mr and not _is_main():
        _drop_peer_pool(ckpt_avg)

    _a0_module = getattr(trainer, "model_wrapped", None)
    _a0_module = _unwrap_module(_a0_module if _a0_module is not None
                                else getattr(trainer, "model", None))
    _rel = _release_trainer(trainer)
    ckpt_avg.trainer = None
    _collect()
    _holders = [h for h in _referrer_report(_a0_module) if h != "frame:_run"]
    _a0_module = None
    _collect()
    _alloc_post_release = _cuda_alloc()
    _residue = (_cuda_residue() if (_alloc_post_release or 0) > _RESIDUE_REPORT_BYTES else None)
    _pf = dev_pass.current_paged_free()

    model2 = trainer2 = last_args = last_cb = None
    trainer2_trained = False
    trainer2_origin: Optional[str] = None
    t_traj_est_live = decision.t_traj
    _dual_decided = False
    _lr_hot = float(facts.learning_rate or 0.0)
    _family_lr_now = _lr_hot
    _div_phase = False
    _div_ends: dict = {}
    _div_live: dict = {}
    _div_batch_k = 0
    _a0_micro = int(getattr(training_args, "per_device_train_batch_size", 1) or 1)
    _a0_accum = int(getattr(training_args, "gradient_accumulation_steps", 1) or 1)
    _cand_idx = -1
    _mfail = member_guard.MemberFailures.from_env()
    _mem_micro_cap: Optional[int] = None
    _early: dict = {}
    _sw = member_phase_switches()
    if _sw.guard:
        log(f"[sn56][traj2] guarda multi-GPU ativa (world={_sw.world}): fase de membros")
    _sb_on = bool(_sw.spare_budget and decision.extra and decision.loop and _LOOP_NO_CAP)
    if _free_held and not _sb_on:
        free_held_drop(_free_held, "no_member_generator", keep=0)
    _sb_plan: Optional[sb.MemberPlanner] = None
    _sb_prices: dict = {}
    _sb_acc: Optional[dict] = None
    _sb_soup = {"n": 1, "loss": None, "srcs": ["a0_soup"], "measured": False, "online": False}
    _sb_base_rec = None
    _sb_head: Optional[Candidate] = None
    _sb_seed_next = 44
    _sb_kind_of: dict = {}
    _sb_secs: dict = {}
    _perturb_live: Optional[float] = None
    _sb_t_unit = 0.0
    _sb_trial_secs: list = []
    _sb_tail_now: dict = {}
    _sb_trial_canary: list = []
    _sb_tail_of: dict = {}
    _sb_tail_launched = 0
    _sb_launch_refused = {"n": 0, "pending": False}
    _sb_hold_of: dict = {}
    _sb_n_keep = 0
    _kf_band: Optional[kind_fix.LrBand] = None
    _kf_dlr_last: dict = {}
    _sb_gaps: list = []
    _sb_t_issue: dict = {}
    _sb_bar_on = False
    _sb_n_trials = 0
    _sb_n_cont = 0
    _sb_cont_idx: set = set()
    _sb_inc_rows: Optional[list] = None
    if _sb_on:
        _sb_head = decision.candidates[0]
        _head = [decision.candidates[0]] + [c for c in decision.candidates[1:2] if c.kind == T_LR]
        decision.candidates = list(_head)
        _sb_seed_next = max([c.seed for c in _head if c.kind == T_SEED] + [43]) + 1
        _base_state = getattr(ckpt_avg, "best_state", None) if _is_main() else None
        _info = {"online": False, "reason": "", "need": None, "room": None, "snap_bytes": None,
                 "numel": None, "member_cap": None, "n_keep": 0}
        if _is_main():
            _sb_b, _sb_n, _sb_big = _state_size(_base_state)
            _info.update(snap_bytes=_sb_b, numel=_sb_n)
            if getattr(ckpt_avg, "_snap_bytes", None) is None and _sb_b:
                ckpt_avg._snap_bytes = int(_sb_b)
            if getattr(ckpt_avg, "_numel", None) is None and _sb_n:
                ckpt_avg._numel = (int(_sb_n), int(_sb_big))
            _info["n_keep"] = len([e for e in ckpt_avg.pool if e.get("state") is not None
                                   and e.get("src") != "a0_soup"])
            if _base_state is None or ckpt_avg.best_loss is None \
                    or ckpt_avg.best_loss != ckpt_avg.best_loss:
                _info["reason"] = "no_a0_soup"
            else:
                try:
                    _rb = ckpt_avg._ram_budget()
                    _room = host_snapshot.stage_room(_rb)
                except Exception:
                    _rb, _room = None, None
                _held_b = len([e for e in _free_held if e.get("state") is not None]) * int(_sb_b)
                _fits, _need = online_soup_fits(
                    _sb_b, _sb_n, (_room + _held_b) if (_room is not None) else None)
                _info.update(need=_need, room=((_room + _held_b) if _room is not None
                                               else None), online=bool(_fits),
                             reason=("" if _fits else "host_ram"))
                _info["_room_held"] = _room
                if not _fits:
                    try:
                        _cap, _ = ckpt_avg._ram_pool_cap(None, _rb)
                        if _cap is not None:
                            _info["member_cap"] = max(1, int(_cap) - len(ckpt_avg._live_pool()))
                    except Exception:
                        pass
        _room_held = _info.pop("_room_held", None)
        _info = _r0(_info) if _mr else _info
        _sb_soup["online"] = bool(_info["online"])
        if _free_gate.active and _is_main():
            _n_held0 = len(_free_held)
            if _sb_soup["online"]:
                _keep = fc.held_keep_count(_room_held, n_held=_n_held0,
                                           snap_bytes=int(_info.get("snap_bytes") or 0),
                                           numel=int(_info.get("numel") or 0),
                                           online_snaps=_SB_ONLINE_SNAPS)
                _dropped = free_held_drop(_free_held, "host_ram", keep=_keep)
            else:
                _dropped = free_held_drop(_free_held, "not_online", keep=0)
        if _sb_soup["online"]:
            if _is_main():
                _sb_acc = {n: t.float().clone() for n, t in _base_state.items()}
                ckpt_avg.online_resident = True
                _rec = getattr(ckpt_avg, "soup_eval_record", None)
                _sb_base_rec = (_rec if (_rec is not None and _rec.state_id == id(_base_state)
                                         and ckpt_avg.best_loss is not None
                                         and float(ckpt_avg.best_loss) == float(_rec.loss))
                                else None)
            _sb_soup["loss"] = float(_r0(float(ckpt_avg.best_loss)) if _mr
                                     else float(ckpt_avg.best_loss))
        _base_state = None
        _sb_plan = sb.MemberPlanner.build(int(facts.steps_per_epoch or 0), _a0_micro * _a0_accum,
                                          perturb=bool(_PERTURB),
                                          online=bool(_sb_soup["online"]),
                                          member_cap=_info["member_cap"],
                                          micro=_a0_micro, accum=_a0_accum,
                                          generate=bool(_sb_soup["online"] or _dist_world() > 1
                                                        or _SB_POOL_FALLBACK_1GPU),
                                          n_fit=int(decision.n_fit),
                                          extra_kinds=(dk.kinds() if dk.forced() else None),
                                          rows_gate_upe=(kind_fix.ROWS_UPE_MAX
                                                         if kind_fix.ROWS_GATE else None),
                                          p_dlr=(kind_fix.P_DLR_BAND
                                                 if kind_fix.DIV_LR_BAND else None))
        _kf_band = kind_fix.band_from_selection(lr_search.LAST_SELECTION,
                                                float(facts.learning_rate or 0.0) or None)
        if _mr:
            _kf_band = kind_fix.LrBand(**_r0(_kf_band.model_dump()))
        _kf_first = kind_fix.div_lr_choice(_kf_band, _family_lr_now, mult=_DIV_LR_MULT)
        _sb_sft = sft_objective(trainer)
        _sb_plan.rate_choice = bool(sb.RATE_CHOICE and _sb_sft)
        _sb_plan.value_model = bool(sb.PRIOR_REFIT and _sb_sft)
        _te = float(decision.t_eval or 0.0)
        _sb_t_unit = _te + soup_overhead_secs(facts, _te)
        _u_first = first_trial_unit(facts) if _sb_sft else None
        if _u_first is not None:
            _sb_t_unit, _ = float(_u_first), "a0_trials"
        _sb_plan.continuation_on = bool(sb.CONTINUATION and _sb_sft)
        _sb_plan.rescue_on = bool(sb.REJECT_RESCUE and _sb_plan.continuation_on and not _mr
                                  and _sb_soup["online"])
        _ck_pref = {dk.T_BAG: 0, T_PERTURB: 1, dk.T_NEFT: 2}
        _sb_plan.cont_kinds = (sorted(dk.continuation_kinds(), key=lambda k: _ck_pref.get(k, 9))
                               if (_sb_plan.continuation_on and not dk.forced()) else [])
        _sb_bar_on = bool(sb.ACCEPT_BAR and _sb_sft)
        _sb_n_keep = int(_info["n_keep"] or 0)
        if _sb_soup["online"]:
            _w = planner_clock.write_secs(facts.clock_stats)
            decision.t_soup = ((int(_info["n_keep"]) + 4) * _sb_t_unit
                               + _w * (1.0 if _INTERIM is not None else 2.0))
            if _mr:
                decision.t_soup = float(_r0(decision.t_soup))
        log(f"[sn56][traj2] fase de membros: online={_sb_soup['online']} d={_sb_plan.d} "
            f"degraus={_sb_plan.rungs} "
            f"rotação={_sb_plan.rotation} (a0 {_sb_plan.a0_updates_per_epoch} upd/época)")

    def _sb_tail_decision(ch, cands: dict, prices: dict, extra: float, es_write: float,
                          left: Optional[float], light: bool) -> Optional[dict]:
        tail_plan = decision.t_soup + decision.reserve + decision.devpass_reserve
        spendable, tail_true, parts, options = tail_fill_inputs(
            decision, facts, {k: int(c.max_steps) for k, c in cands.items()
                              if c is not None and c.max_steps},
            prices, extra=extra, es_write=es_write, left=left, n_keep=_sb_n_keep,
            trial_secs=tail_fill_trials(_sb_trial_secs, _sb_trial_canary),
            interim=_INTERIM is not None,
            write=planner_clock.write_secs(facts.clock_stats),
            n_free=len([e for e in _free_held if e.get("state") is not None]),
            gap=tail_launch_gap(_sb_gaps),
            tail_only=bool(decision.reason == "no_fit+tail_only"))
        blocked = _sb_plan.n_tail > _sb_tail_launched
        tc = (_sb_plan.tail_fill(ch.stop, options, spendable) if not blocked
              else sb.TailChoice(stop="previous_tail_not_launched",
                                 spendable=(round(spendable, 1) if spendable is not None
                                            else None)))
        event = dict(tc.model_dump(), left=(round(float(left), 1) if left is not None else None),
                     tail_true=round(tail_true, 1), tail_plan=round(tail_plan, 1),
                     n_trials=len(_sb_trial_secs),
                     n_canary_trials=sum(1 for c in _sb_trial_canary if c), **parts)
        payload = {"event": event, "cand": None, "kind": tc.kind, "price": tc.price,
                   "horizon": tc.horizon, "tail_true": tail_true}
        if tc.kind:
            payload["cand"] = dict(max_steps=int(tc.steps), **_sched_for(
                facts, int(tc.steps), decision.t_step_eval_free or None,
                decision.t_eval or None, light))
            event["schedule"] = {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in payload["cand"].items()}
        return payload

    def _sb_next() -> Optional[Candidate]:
        nonlocal _sb_plan, _sb_seed_next
        tail_payload = None
        light = bool(decision.loop and _LOOP_EVAL_LIGHT)
        es_write = (planner_clock.write_secs(facts.clock_stats)
                    if (_sb_soup["online"] and _INTERIM is None) else 0.0)
        extra = (soup_overhead_secs(facts, float(decision.t_eval or 0.0))
                 + (_sb_t_unit if (_sb_soup["online"] and not _sb_soup["measured"]) else 0.0)
                 + es_write)
        _mf = (dataclasses.replace(facts, best_step=int(decision.member_len))
               if decision.member_len else facts)
        k_div = sum(int(_sb_plan.issued.get(sb.div_kind(r), 0)) for r in _sb_plan.rungs)
        k_pert = int(_sb_plan.issued.get(T_PERTURB, 0))
        cands = {T_SEED: dataclasses.replace(_sb_head, kind=T_SEED, seed=_sb_seed_next,
                                             learning_rate=float(_family_lr_now))}
        prices = {T_SEED: t_traj_est_live + extra}
        for _r in _sb_plan.rungs:
            _pk = sb.div_kind(_r)
            c = div_batch_member(_mf, _a0_micro, _a0_accum, _family_lr_now,
                                 decision.t_step_eval_free or None, decision.t_eval or None,
                                 light, seed=div_batch_seed(k_div), div=int(_r))
            if c is not None:
                c = cap_member_micro(c, _mem_micro_cap, _a0_micro, _a0_accum)
                cands[_pk] = c
                prices[_pk] = (_div_live.get(_pk) or member_secs(
                    c, _mf, t_traj_est_live, decision.t_load_est,
                    a0_shape=(_a0_micro, _a0_accum),
                    fixed=member_fixed_secs(decision))) + extra
        if sb.T_DIV_LR in _sb_plan.rotation:
            _dlr = kind_fix.div_lr_choice(_kf_band, _family_lr_now, mult=_DIV_LR_MULT)
            _dkey = (_dlr.offered, _dlr.reason, round(float(_dlr.lr or 0.0), 12),
                     round(float(_family_lr_now or 0.0), 12))
            if _dkey != _kf_dlr_last.get("key"):
                _kf_dlr_last["key"] = _dkey
            c = (div_lr_member(_mf, _family_lr_now, decision.t_step_eval_free or None,
                               decision.t_eval or None, light,
                               seed=div_lr_seed(int(_sb_plan.issued.get(sb.T_DIV_LR, 0))),
                               lr=_dlr.lr)
                 if _dlr.offered else None)
            if c is not None:
                cands[sb.T_DIV_LR] = c
                prices[sb.T_DIV_LR] = (_div_live.get(sb.T_DIV_LR) or t_traj_est_live) + extra
        _price_rot = _sb_plan.active_rotation()
        if _sb_plan.rescue_on and not _sb_plan.rescue_done and _sb_plan.cont:
            _price_rot = _price_rot + [k for k in _sb_plan.cont_excluded if k not in _price_rot]
        for _dk in _price_rot:
            if _dk not in dk.KINDS:
                continue
            cands[_dk] = dk.candidate(_dk, _sb_head, _family_lr_now,
                                      int(_sb_plan.issued.get(_dk, 0)))
            if _dk == T_PERTURB:
                prices[_dk] = (_perturb_live or (t_traj_est_live
                               + planner_clock.write_secs(facts.clock_stats))) + extra
            else:
                prices[_dk] = (_div_live.get(_dk) or t_traj_est_live) + extra
        for _ck in (_price_rot if _sb_plan.cont else []):
            _bld = CONTINUATION_BUILDERS.get(_ck)
            if _ck in cands or _bld is None:
                continue
            try:
                _built = _bld(_mf, family_lr=float(_family_lr_now),
                              seed=FRESH_SEED_BASE + 1000 + int(_sb_plan.issued.get(_ck, 0)),
                              light=light, t_step=decision.t_step_eval_free or None,
                              t_eval=decision.t_eval or None, a0_micro=_a0_micro,
                              a0_accum=_a0_accum)
            except Exception:
                _built = None
            if _built is not None:
                cands[_ck], prices[_ck] = _built[0], float(_built[1]) + extra
        if _sb_plan.perturb and T_PERTURB not in cands:
            cands[T_PERTURB] = dataclasses.replace(_sb_head, kind=T_PERTURB,
                                                   seed=SEED_PERTURB_BASE + k_pert,
                                                   learning_rate=float(_family_lr_now))
            prices[T_PERTURB] = (_perturb_live or (t_traj_est_live
                                 + planner_clock.write_secs(facts.clock_stats))) + extra
        if _is_main():
            left = _remaining_secs(end_time)
            tail = decision.t_soup + decision.reserve + decision.devpass_reserve
            _tail_src = "plan"
            if _SB_TAIL_MEASURED and _sb_trial_secs:
                tail, _ = tail_fill_reserve(
                    decision, n_keep=_sb_n_keep,
                    trial_secs=tail_fill_trials(_sb_trial_secs, _sb_trial_canary),
                    interim=_INTERIM is not None,
                    write=planner_clock.write_secs(facts.clock_stats),
                    n_free=len([e for e in _free_held if e.get("state") is not None]))
                _tail_src = "measured_trials"
            _sb_tail_now["tail"], _sb_tail_now["source"] = float(tail), _tail_src
            if _sb_launch_refused["pending"]:
                _sb_launch_refused["pending"] = False
                ch = sb.Choice(left=left, stop="time", prices={k: round(float(v), 1)
                                                               for k, v in prices.items()},
                               rule="packed_fill_launch_no_fit")
            else:
                _tf = tail_fill_trials(_sb_trial_secs, _sb_trial_canary)
                _sb_plan.trial_secs = float(max(_tf)) if _tf else 0.0
                ch = _sb_plan.choose(prices, left, tail)
            if ch.kind is None and sb.TAIL_FILL:
                tail_payload = _sb_tail_decision(ch, cands, prices, extra, es_write, left, light)
            payload = (ch.model_dump(), _sb_plan.model_dump())
        else:
            payload = None
        if _mr:
            payload = _r0(payload)
        ch = sb.Choice(**payload[0])
        _sb_plan = sb.MemberPlanner(**payload[1])
        if _mr and ch.kind is None and sb.TAIL_FILL:
            tail_payload = _r0(tail_payload)
        nxt = cands.get(ch.kind) if ch.kind else None
        if tail_payload is not None:
            if tail_payload.get("cand") is not None:
                nxt = dataclasses.replace(cands[tail_payload["kind"]], **tail_payload["cand"])
                if tail_payload["kind"] == T_SEED:
                    _sb_seed_next += 1
                _sb_prices[len(decision.candidates)] = float(tail_payload["price"])
                _sb_kind_of[len(decision.candidates)] = tail_payload["kind"]
                _sb_tail_of[len(decision.candidates)] = float(tail_payload["tail_true"])
                _sb_t_issue[len(decision.candidates)] = time.time()
                if _sb_plan.cont:
                    _sb_cont_idx.add(len(decision.candidates))
                log(f"[sn56][traj2] tail fill: {tail_payload['kind']} {nxt.max_steps}/"
                    f"{tail_payload['horizon']} passos (~{tail_payload['price']:.0f}s, "
                    f"{tail_payload['event'].get('spendable')}s gastáveis)")
                return nxt
        if nxt is None:
            log(f"[sn56][traj2] fim dos membros ({ch.stop}): {_sb_plan.n_launched} treinados, "
                f"soup online n={_sb_soup['n']}")
            return None
        if ch.kind == T_SEED:
            _sb_seed_next += 1
        _sb_prices[len(decision.candidates)] = float(ch.prices.get(ch.kind, prices[ch.kind]))
        _sb_kind_of[len(decision.candidates)] = ch.kind
        _sb_t_issue[len(decision.candidates)] = time.time()
        if (packed_fill.active() and _dist_world() <= 1 and _is_main()
                and _sb_tail_now.get("source") == "measured_trials"
                and ch.rule != "packed_fill_launch_no_fit"):
            _sb_hold_of[len(decision.candidates)] = float(_sb_tail_now["tail"])
        if _sb_plan.cont:
            _sb_cont_idx.add(len(decision.candidates))
        return nxt

    while True:
        if _cand_idx + 1 >= len(decision.candidates):
            if not _sb_on:
                break
            _nxt = _sb_next()
            if _nxt is None:
                break
            decision.candidates.append(_nxt)
        _cand_idx += 1
        cand = decision.candidates[_cand_idx]
        if (decision.extra and not _sb_on and not _div_phase
                and len(_running_soup_losses) > _SOUP_PLATEAU_PATIENCE):
            recent = _running_soup_losses[-(_SOUP_PLATEAU_PATIENCE + 1):]
            if all((recent[j] is None) or (recent[j + 1] is None)
                   or (recent[j] - recent[j + 1] < _SOUP_PLATEAU_MARGIN)
                   for j in range(len(recent) - 1)):
                _divs = (diverse_members(facts, _a0_micro, _a0_accum, _family_lr_now,
                                         decision.t_step_eval_free or None, decision.t_eval or None,
                                         light=bool(decision.loop and _LOOP_EVAL_LIGHT),
                                         div_lr=_sw.div_lr)
                         if _DIVERSE else [])
                if not _divs:
                    break
                _div_phase = True
                decision.candidates[_cand_idx:] = _divs
                cand = decision.candidates[_cand_idx]
                _launched_n = len([r for r in out.results if r.launched])
                ckpt_avg.soup_max = max(int(ckpt_avg.soup_max or 0),
                                        loop_pool_cap(_launched_n + len(_divs)))
                _te = float(decision.t_eval or 0.0)
                decision.t_soup += (2.0 * _te + soup_overhead_secs(facts, _te)) * len(_divs)
                if getattr(ckpt_avg, "pool_pin", None) is not None:
                    ckpt_avg.pool_pin = loop_pool_pin(facts.a0_keep_steps, int(facts.best_step),
                                                      extra_ends=_div_ends)
        _is_div = cand.kind in DIVERSE_KINDS
        _cand_secs = _sb_prices[_cand_idx] if _cand_idx in _sb_prices else ((_div_live.get(cand.kind) if _sw.price_measured else None) or member_secs(
                          cand, facts, t_traj_est_live, decision.t_load_est,
                          a0_shape=(_a0_micro, _a0_accum), fixed=member_fixed_secs(decision),
                          measured=_sw.price_measured)
                      if _is_div else t_traj_est_live)
        res = TrajectoryResult(kind=cand.kind, seed=cand.seed,
                               learning_rate=cand.learning_rate,
                               max_steps=cand.max_steps)
        out.results.append(res)
        left = _remaining_secs(end_time) or 0.0
        _tail_hold = _sb_tail_of.get(_cand_idx, _sb_hold_of.get(
            _cand_idx, decision.t_soup + decision.reserve + decision.devpass_reserve))
        res.tail = _cand_idx in _sb_tail_of
        need = _cand_secs + _tail_hold
        if _r0_flag(left < need):
            res.error = f"no_fit_before_launch(left={left:.0f}s,need={need:.0f}s)"
            if (_sb_on and _cand_idx in _sb_kind_of
                    and _cand_idx not in _sb_tail_of):
                _sb_launch_refused["n"] += 1
                _sb_launch_refused["pending"] = bool(_is_main())
            continue

        cand = cap_member_micro(cand, _mem_micro_cap, _a0_micro, _a0_accum)
        args2, _shape = member_args(training_args, cand, facts)
        _dk_args = dk.apply_args(cand.kind, args2, training_args)
        if _r0_flag(not _shape.ok):
            res.error = f"memory_unproven(tokens={_shape.tokens_worst},a0={_shape.a0_max_tokens})"
            log(f"[sn56][traj2] {cand.kind}: forma de memória não provada pelo a0, pulando")
            continue
        ckpt_avg.trainer = None
        _rel2 = _release_trainer(trainer2) if trainer2 is not None else None
        model2 = trainer2 = None
        _bnd_secs = member_guard.boundary_release()
        _bnd_secs += host_snapshot.release_host_memory(collect=False)["secs"]
        trainer2_trained = False
        trainer2_origin = None
        t0 = time.time()
        _load_exc = None
        try:
            model2 = model_builder()
        except Exception as exc:
            _load_exc = exc
        if _mr and not _all_ok(_load_exc is None) and _load_exc is None:
            _load_exc = RuntimeError("model_load failed on a peer rank")
        if _load_exc is not None:
            exc = _load_exc
            res.error = f"model_load:{type(exc).__name__}: {exc}"[:300]
            break
        if cand.kind == T_PERTURB:
            _pst = {"ok": False}
            if _is_main():
                try:
                    _lam, _kap = dk.perturb_params(_PERTURB_LAMBDA, _PERTURB_KAPPA)
                    _pst = perturb_init(model2, _a0_soup_state, seed=cand.seed,
                                        lam=_lam, kappa=_kap)
                    _pst["ok"] = True
                except Exception as exc:
                    _pst = {"ok": False, "exc": f"{type(exc).__name__}: {exc}"[:500]}
            if _mr:
                _pst = _r0(_pst)
            ckpt_avg._bcast_params(model2)
        res.load_secs = time.time() - t0

        left = _remaining_secs(end_time) or 0.0
        traj_budget = max(_cand_secs - decision.t_load_est, 0.0)
        if _cand_idx in _sb_t_issue:
            _sb_gaps.append(max(0.0, time.time() - _sb_t_issue.pop(_cand_idx)
                                - float(decision.t_load_est or 0.0)))
        if _mr:
            left = _r0(left)
        if _r0_flag(left < (traj_budget + _tail_hold)):
            res.error = f"no_fit_after_load(left={left:.0f}s)"
            if (packed_fill.active() and _sb_on and _cand_idx in _sb_kind_of
                    and _cand_idx not in _sb_tail_of):
                _sb_launch_refused["n"] += 1
                _sb_launch_refused["pending"] = bool(_is_main())
                continue
            break

        args2.lr_scheduler_type = "warmup_stable_decay"
        args2.lr_scheduler_kwargs = {
            "num_decay_steps": int(cand.decay_steps),
            "decay_type": _WSD_DECAY_TYPE,
            "warmup_type": "linear",
            "min_lr_ratio": _WSD_MIN_LR_RATIO,
        }
        if cand.warmup_steps > 0:
            args2.warmup_steps = cand.warmup_steps
        args2.save_strategy = "no"
        args2.eval_strategy = "steps"
        args2.eval_steps = max(1, int(cand.eval_steps or facts.eval_steps or 1))
        args2.output_dir = os.path.join(
            getattr(training_args, "output_dir", "") or ".", f"traj_{cand.kind}")

        try:
            _sched_proof = preflight_schedule(args2, cand.max_steps)
        except Exception as exc:
            res.error = f"schedule:{type(exc).__name__}: {exc}"[:300]
            log(f"[sn56][traj2] {cand.kind}: agendador impossível: {exc}")
            break

        last_args = args2
        hard_deadline = (time.time() + max(0.0, left - _tail_hold))
        cb = TrajectoryCallback(ckpt_avg, cand.kind, cand.max_steps, facts,
                                hard_deadline=hard_deadline,
                                eval_steps=cand.eval_steps,
                                eval_at=cand.eval_at,
                                warmup_steps=cand.warmup_steps or facts.warmup_steps,
                                decay_steps=cand.decay_steps,
                                min_lr_ratio=_WSD_MIN_LR_RATIO,
                                peak_lr=cand.learning_rate, log=log,
                                pool_admit=not (_sb_on and _sb_soup["online"]),
                                skip_mid=bool(_sb_on and _sb_soup["online"]
                                              and _dist_world() == 1
                                              and eval_squeeze.skip_member_mideval_enabled()))
        last_cb = cb
        _pool_src = (f"{cand.kind}#{cand.seed}"
                     if ((decision.extra and cand.kind == T_SEED) or _is_div
                         or cand.kind == T_PERTURB or cand.kind in _DK_KINDS) else cand.kind)
        if _is_div:
            _div_ends[_pool_src] = int(cand.max_steps)
        if (_dual_on and cand.kind == T_SEED and _lr_hot > 0
                and abs(float(cand.learning_rate) - _lr_hot) > 1e-12 * _lr_hot):
            _pool_src = f"{_pool_src}@cold"
        ckpt_avg.pool_tag = _pool_src
        run_state.STOP_REASON["reason"] = None
        _pre_exc = None
        _train_exc = None
        try:
            trainer2 = trainer_builder(model2, args2)
            if cand.kind in dk.KINDS:
                _dk_prep = dk.apply_trainer(cand.kind, cand, trainer2)
                _, _ = _dk_args.model_dump(), _dk_prep.model_dump()
            trainer2.add_callback(member_guard.MemberCapacityCallback(
                facts.memory_proof, label=f"{cand.kind}#{cand.seed}",
                host_fn=functools.partial(_member_host_terms, ckpt_avg, model2)))
            trainer2.add_callback(rank_agreement.RankAgreementCallback(
                f"member:{cand.kind}#{cand.seed}"))
            trainer2.add_callback(cb)
            ckpt_avg.trainer = trainer2
        except Exception as exc:
            _pre_exc = exc
        if _mr and not _all_ok(_pre_exc is None) and _pre_exc is None:
            _pre_exc = RuntimeError("trainer_build failed on a peer rank")
        try:
            if _pre_exc is not None:
                raise _pre_exc
            rng_ok = (_restore_rng(rng_state)
                      if (cand.kind == T_LR and not _TLR_FRESH_SEED) else False)
            t1 = time.time()
            _train_exc = None
            sg.ship_marker_update(activity=f"member:{cand.kind}#{cand.seed}")
            with member_guard.short_collective_timeout(_ct_secs, _ddp_timeout,
                                                       f"member:{cand.kind}"):
                try:
                    trainer2.train()
                except Exception as exc:
                    if not _mr:
                        raise
                    _train_exc = exc
                _agreed_ok = True
                if _mr:
                    _tv = _train_verdict(_train_exc, float(_ct_secs or _ddp_timeout))
                    if _tv.verdict == "desync":
                        _exit_desync(f"member:{cand.kind}#{cand.seed}", _tv, _train_exc)
                    _agreed_ok = _tv.verdict == "ok"
            if _mr and not _agreed_ok:
                raise (_train_exc or RuntimeError("train failed on a peer rank"))
            res.secs = time.time() - t1
            res.launched = True
            _mfail.on_success()
            trainer2_trained = True
            trainer2_origin = T2_TRAINED
            res.stop_step = int(getattr(trainer2.state, "global_step", 0) or 0)
            res.best_step, res.best_loss, res.n_evals = cb.best_step, cb.best_loss, cb.n_evals
            _record_schedule(res, cand, facts, cb)
            if decision.extra:
                _div, _why = _train_diverged(trainer2, level=(cand.kind == T_DIV_BATCH))
                _bad = _div or (res.best_loss is None) or (res.best_loss != res.best_loss)
                if _r0_flag(_bad):
                    _diverged_srcs.add(_pool_src)
                    res.diverged = True
                    if _sb_on:
                        _sb_plan.record(_sb_kind_of.get(_cand_idx, cand.kind), None,
                                        diverged=True)
                elif _sb_on:
                    _verdict, _tl, _gain = None, None, None
                    _ok, _before, _ev_se = False, None, None
                    _bar_blocked = False
                    _have = bool(cb.best_state is not None) if _is_main() else False
                    if _mr:
                        _have = bool(_r0(_have))
                    if _sb_soup["online"] and _have:
                        _bar_z, _tr_rows = 0.0, None
                        _cb_frozen_prev = cb.frozen
                        cb.frozen = True
                        try:
                            if not _sb_soup["measured"]:
                                _t_tr, _gv0 = time.time(), gpu_soup.verdict_state()
                                _base_measured: list = []
                                _l0 = _online_base_loss(
                                    ckpt_avg, model2, trainer2, _sb_acc, _sb_soup, _sb_base_rec,
                                    member_state=(cb.best_state if _is_main() else None),
                                    measured=_base_measured)
                                if _base_measured:
                                    _sb_trial_secs.append(time.time() - _t_tr)
                                    _sb_trial_canary.append(gpu_soup.verdict_state() != _gv0)
                                    _sb_inc_rows = getattr(trainer2, "last_eval_rows", None)
                                if _l0 == _l0 and _l0 != float("inf"):
                                    _sb_soup["loss"] = float(_l0)
                                _sb_soup["measured"] = True
                            _sb_n_trials += 1
                            if _cand_idx in _sb_cont_idx:
                                _sb_n_cont += 1
                            _bar_z = (sb.accept_z(_sb_n_cont)
                                      if (_sb_bar_on and _cand_idx in _sb_cont_idx) else 0.0)
                            if (_bar_z == 0.0 and _sb_bar_on and sb.ACCEPT_BAR_SHADOW
                                    and sb.SHADOW_ALL_TRIALS):
                                _bar_z = sb.accept_z(max(1, _sb_n_trials))
                            if (_bar_z > 0 and _sb_inc_rows is None and not _mr
                                    and _sb_acc is not None and _cand_idx in _sb_cont_idx):
                                ckpt_avg._online_eval_soup(model2, _sb_acc, _sb_soup["n"])
                                _sb_inc_rows = getattr(trainer2, "last_eval_rows", None)
                            _t_tr, _gv0 = time.time(), gpu_soup.verdict_state()
                            _tl = ckpt_avg._online_trial(
                                model2, _sb_acc, _sb_soup["n"],
                                cb.best_state if _is_main() else None)
                            _sb_trial_secs.append(time.time() - _t_tr)
                            _sb_trial_canary.append(gpu_soup.verdict_state() != _gv0)
                            _tr_rows = getattr(trainer2, "last_eval_rows", None)
                        except Exception:
                            _tl = None
                        finally:
                            cb.frozen = _cb_frozen_prev
                        _ok = (_tl is not None and _tl == _tl and _tl != float("inf"))
                        _bar_se = (sb.paired_se(_sb_inc_rows, _tr_rows)
                                   if (_bar_z > 0 and _tl is not None) else None)
                        _bar = (_bar_z * _bar_se) if (_bar_se is not None) else 0.0
                        _ev_se = (_bar_se if _bar_se is not None else
                                  (sb.paired_se(_sb_inc_rows, _tr_rows) if _tl is not None
                                   else None))
                        _bar_use = 0.0 if sb.ACCEPT_BAR_SHADOW else _bar
                        _verdict = bool(_ok and _sb_soup["loss"] is not None
                                        and _tl < float(_sb_soup["loss"]) - max(1e-6, _bar_use))
                        _bar_would = bool(_ok and _sb_soup["loss"] is not None
                                          and _tl < float(_sb_soup["loss"]) - max(1e-6, _bar))
                        _bar_blocked = bool(_ok and _sb_soup["loss"] is not None and not _bar_would
                                            and _tl < float(_sb_soup["loss"]) - 1e-6)
                        if _mr:
                            _verdict = bool(_r0(_verdict))
                        _before = _sb_soup["loss"]
                        if _verdict:
                            _gain = float(_before) - float(_tl)
                            if _is_main():
                                ckpt_avg._online_add(model2, _sb_acc, cb.best_state)
                            _sb_soup["n"] += 1
                            _sb_soup["loss"] = float(_tl)
                            _sb_soup["srcs"].append(_pool_src)
                            _sb_inc_rows = _tr_rows
                            if _INTERIM is not None:
                                if (_interim_best is not None and _tl < _interim_best
                                        and _INTERIM.offer(model2, loss=_tl,
                                                           label=f"online_soup:{_sb_soup['n']}",
                                                           n_members=_sb_soup["n"]) == "queued"):
                                    _interim_best = _tl
                            elif _is_main():
                                _early_ship_soup(model2, _tl, _sb_soup["n"],
                                                 submission_dir=train_request["submission_dir"],
                                                 record=_early, log=log)
                        if (not _verdict and not (_bar_blocked and not sb.ACCEPT_BAR_SHADOW)
                                and _free_gate.active
                                and _free_gate.member_rejects
                                and _is_main() and cb.best_state is not None
                                and res.best_loss is not None):
                            _hold_ok, _hold_room = False, None
                            try:
                                _hold_room = host_snapshot.stage_room(ckpt_avg._ram_budget())
                                _hold_ok = fc.can_hold_reject(
                                    _hold_room,
                                    snap_bytes=int(ckpt_avg._trainable_snapshot_bytes(model2)),
                                    online_snaps=_SB_ONLINE_SNAPS)
                            except Exception:
                                pass
                            _fe = fc.entry(cb.best_state, loss=float(res.best_loss),
                                           step=int(res.best_step or res.stop_step or 0),
                                           src=_pool_src, origin=fc.SRC_REJECT,
                                           soup_n_at_reject=int(_sb_soup["n"]),
                                           dev_at_reject=_tl)
                            if _hold_ok:
                                _free_held.append(_fe)
                            _fe = None
                    _sk_rec = _sb_kind_of.get(_cand_idx, cand.kind)
                    _raw_rec = (float(_before) - float(_tl)
                                if (_sb_soup["online"] and _have and _ok and _before is not None)
                                else None)
                    _se_rec = _ev_se if (_sb_soup["online"] and _have) else None
                    _sb_plan.record(_sk_rec, _verdict, _gain,
                                    soup_n=(int(_sb_soup["n"]) if _sb_soup["online"] else None),
                                    bar_blocked=bool(_bar_blocked and not sb.ACCEPT_BAR_SHADOW),
                                    raw_gain=_raw_rec, se=_se_rec)
                else:
                    _left = _remaining_secs(end_time) or 0.0
                    if _r0_flag(_left > (decision.t_soup + decision.reserve
                                         + decision.devpass_reserve)):
                        _live = [e["state"] for e in ckpt_avg.pool
                                 if e.get("state") is not None
                                 and e.get("src") != "a0_soup"]
                        _live_n = len(_live)
                        _cb_frozen_prev = cb.frozen
                        cb.frozen = True
                        try:
                            _rs = ckpt_avg._eval_uniform_soup(model2, _live)
                        except Exception:
                            _rs = None
                        finally:
                            cb.frozen = _cb_frozen_prev
                            _live = None
                        _ok = (_rs is not None and _rs == _rs
                               and _rs != float("inf"))
                        _running_soup_losses.append(_rs if _ok else None)
                        if (_ok and _INTERIM is not None and _interim_best is not None
                                and _rs < _interim_best):
                            if _INTERIM.offer(model2, loss=_rs,
                                              label=f"running_soup:{_live_n}",
                                              n_members=_live_n) == "queued":
                                _interim_best = _rs
        except member_guard.MemberMemoryRefused as exc:
            res.error = f"memory_refused:{exc.info.get('reason')}"
            trainer2_origin = T2_REFUSED if trainer2 is not None else None
            _m_now = int(getattr(args2, "per_device_train_batch_size", 1) or 1)
            _fit = exc.fit_micro
            if _fit is not None and _fit < _m_now:
                _mem_micro_cap = int(_fit)
                _cand_idx -= 1
                log(f"[sn56][traj2] {cand.kind}: memória insuficiente no micro {_m_now}, "
                    f"tentando micro {_fit}")
                continue
            log(f"[sn56][traj2] {cand.kind}: memória insuficiente ({exc.info.get('reason')}), "
                f"encerrando membros")
            break
        except Exception as exc:
            res.error = f"train:{type(exc).__name__}: {exc}"[:300]
            trainer2_origin = (T2_FAILED if _pre_exc is None
                               else (T2_REFUSED if trainer2 is not None else None))
            log(f"[sn56][traj2] {cand.kind} falhou: {exc}")
            if decision.extra:
                _diverged_srcs.add(_pool_src)
                res.diverged = True
            _fkind, _vote = member_guard.member_skip_vote(
                multi_rank=_mr, launched=res.launched, pre_exc=_pre_exc, train_exc=_train_exc,
                exc=exc)
            if _mr:
                _vote = _all_ok(_vote)
            if _mfail.on_failure((_fkind or "peer") if _vote else None):
                log(f"[sn56][traj2] {cand.kind}#{cand.seed}: membro pulado ({_fkind or 'peer'}), "
                    f"seguindo para o próximo")
                _train_exc = _pre_exc = None
                ckpt_avg.trainer = None
                cb.best_state = None
                continue
            break
        finally:
            ckpt_avg.pool_tag = None

        if cb.best_state is not None:
            try:
                ckpt_avg._restore(model2, cb.best_state)
            except Exception as exc:
                log(f"[sn56][traj2] restauração {cand.kind} falhou: {exc}")
            cb.best_state = None
            gc.collect()
        out.launched = out.launched or res.launched
        if decision.extra:
            _done_secs = [r.secs + r.load_secs for r in out.results
                          if r.launched and r.kind not in DIVERSE_KINDS
                          and r.kind != T_PERTURB and r.kind not in _DK_KINDS and not r.tail]
            if _done_secs:
                t_traj_est_live = sum(_done_secs) / len(_done_secs)
                t_traj_est_live += decision.t_member_extra
                if decision.loop and _LOOP_NO_CAP:
                    t_traj_est_live += decision.t_eval
            if res.tail and res.launched:
                _sb_tail_launched += 1
                _remaining_secs(end_time)
            if cand.kind == T_PERTURB and res.launched and not res.tail:
                _kp = [r.secs + r.load_secs for r in out.results
                       if r.launched and r.kind == T_PERTURB and not r.tail]
                _perturb_live = (sum(_kp) / len(_kp) + decision.t_member_extra
                                 + (decision.t_eval if (decision.loop and _LOOP_NO_CAP) else 0.0))
            if (_is_div or (_sb_on and cand.kind != T_LR)) and res.launched and not res.tail:
                _pk = _sb_kind_of.get(_cand_idx, cand.kind) if _sb_on else cand.kind
                if _sb_on:
                    _sb_secs.setdefault(_pk, []).append(res.secs + res.load_secs)
                    _kd = list(_sb_secs[_pk])
                else:
                    _kd = [r.secs + r.load_secs for r in out.results
                           if r.launched and r.kind == cand.kind]
                _div_live[_pk] = (sum(_kd) / len(_kd) + decision.t_member_extra
                                  + (decision.t_eval if (decision.loop and _LOOP_NO_CAP)
                                     else 0.0))
            if (_sb_on and not _sb_soup["online"] and res.launched
                    and len([r for r in out.results if r.launched]) > int(decision.n_fit or 0)):
                _te = float(decision.t_eval or 0.0)
                decision.t_soup += 2.0 * _te + soup_overhead_secs(facts, _te)
                ckpt_avg.soup_max = max(int(ckpt_avg.soup_max or 0), loop_pool_cap(
                    len([r for r in out.results if r.launched]) + 1))
            if (_div_phase and _sw.div_repeat and cand.kind == T_DIV_BATCH and res.launched
                    and not res.diverged and _cand_idx + 1 >= len(decision.candidates)
                    and _div_batch_k + 1 < _DIV_MAX):
                _div_batch_k += 1
                _nxt = dataclasses.replace(cand, seed=div_batch_seed(_div_batch_k))
                decision.candidates.append(_nxt)
                _te = float(decision.t_eval or 0.0)
                decision.t_soup += 2.0 * _te + soup_overhead_secs(facts, _te)
                ckpt_avg.soup_max = int(ckpt_avg.soup_max or 0) + 1

        if _dual_on and cand.kind == T_LR and not _dual_decided:
            _dual_decided = True
            _t_seed43 = next((r for r in out.results
                              if r.kind == T_SEED and r.seed == 43 and r.launched
                              and not r.diverged), None)
            _dl = cold_lr_decide(
                res.best_loss if res.launched else None,
                _t_seed43.best_loss if _t_seed43 is not None else None,
                _a0_single_dev, margin=_COLD_LR_MARGIN,
                t_lr_diverged=bool(res.diverged or not res.launched))
            if not res.launched and _dl["reason"] == "t_lr_diverged":
                _dl["reason"] = "t_lr_not_launched"
            if _mr:
                _dl = _r0(_dl)
            _tail = decision.candidates[_cand_idx + 1:]
            if _dl["switched"]:
                decision.candidates[_cand_idx + 1:] = cold_lr_remint(
                    decision.candidates, _cand_idx, float(cand.learning_rate))
            _family_lr = (float(cand.learning_rate) if _dl["switched"] else _lr_hot)
            _family_lr_now = _family_lr
            log(f"[sn56][traj2] cold-LR: t_lr={_dl['t_lr_dev']} t_seed#43="
                f"{_dl['t_seed_dev']} a0_single={_dl['a0_single_dev']} "
                f"d_seed={_dl['d_seed']} d_a0={_dl['d_a0']} -> "
                f"{'SWITCH' if _dl['switched'] else 'keep'} ({_dl['reason']}); "
                f"{len(_tail)} seeds at lr={_family_lr:.3e}")

    if _dual_on and not _dual_decided:
        _t_lr_c = next((c for c in decision.candidates if c.kind == T_LR), None)
        _dl = cold_lr_decide(None, None, _a0_single_dev, margin=_COLD_LR_MARGIN)
        _dl["reason"] = "t_lr_not_run"

    if _mfail.consecutive:
        member_guard.boundary_release()
    if decision.extra and _diverged_srcs:
        _before = len(ckpt_avg.pool)
        for e in ckpt_avg.pool:
            if e.get("src") in _diverged_srcs and e.get("state") is not None:
                e["state"] = None
        ckpt_avg.pool = [e for e in ckpt_avg.pool if e.get("state") is not None]
        gc.collect()

    out.spare = emit_spare("after_trajectories", end_time=end_time,
                           decision=decision, results=out.results)

    if _mr:
        out.launched = _all_ok(out.launched)
    if not out.launched and _sb_acc is not None:
        _sb_acc = None
        ckpt_avg.online_resident = False
        gc.collect()
    if not out.launched and devfit is not None and devfit.enabled:
        ckpt_avg.trainer = None
        _release_trainer(trainer2)
        model2 = trainer2 = None
        _collect()
        _clk_fb = facts.clock_stats
        try:
            out.devfit = a0_soup_dev_pass(
                plan=devfit, ckpt_avg=ckpt_avg, a0_state=_a0_soup_state,
                model_builder=model_builder, trainer_builder=trainer_builder,
                training_args=training_args, submission_dir=train_request["submission_dir"],
                end_time=end_time, reserve=decision.reserve, load_secs=decision.t_load_est,
                save_secs=planner_clock.write_secs(_clk_fb),
                eval_secs=planner_clock.eval_b1_secs(_clk_fb, facts.eval_runtime),
                log=log)
        except Exception as exc:
            out.devfit = {"ran": False, "reason": f"failed:{type(exc).__name__}: {exc}"[:300],
                          "applied_to": "a0_soup", "shipped": "a0_soup", "post_soup": True,
                          "rebuilt": True}
        if _mr:
            out.devfit["ran"] = _all_ok(bool(out.devfit.get("ran")))
        log(f"[sn56][devfit] sem membros: dev-pass sobre o soup a0 ran={out.devfit.get('ran')} "
            f"({out.devfit.get('reason')})")
    if out.launched and _r0_flag(trainer2 is None or model2 is None):
        left = _remaining_secs(end_time) or 0.0
        if _r0_flag(left > (decision.t_load_est + decision.t_soup + decision.reserve
                            + decision.devpass_reserve)):
            _rec_exc = None
            try:
                model2 = model_builder()
                trainer2 = trainer_builder(model2, last_args or copy.copy(training_args))
                trainer2_trained = False
                trainer2_origin = T2_RECOVERED
            except Exception as exc:
                _rec_exc = exc
            if _mr and not _all_ok(_rec_exc is None) and _rec_exc is None:
                _rec_exc = RuntimeError("recover failed on a peer rank")
            if _rec_exc is None:
                pass
            else:
                exc = _rec_exc
                model2 = trainer2 = None
        else:
            pass

    if out.launched and trainer2 is not None and model2 is not None:
        if last_cb is not None:
            last_cb.frozen = True
            try:
                trainer2.remove_callback(last_cb)
            except Exception:
                pass
        try:
            ckpt_avg.pool_tag = None
            ckpt_avg.trainer = trainer2
            ckpt_avg.end_time = end_time
            if _is_main():
                ckpt_avg.on_soup_pass1 = functools.partial(
                    _early_ship_soup, submission_dir=train_request["submission_dir"],
                    record=_early, log=log)
            _sb_final = bool(_sb_on and _sb_soup["online"])
            try:
                if _sb_final:
                    if not _sb_soup["measured"]:
                        _l0 = _online_base_loss(ckpt_avg, model2, trainer2, _sb_acc, _sb_soup,
                                                _sb_base_rec, member_state=None)
                        if _l0 == _l0 and _l0 != float("inf"):
                            _sb_soup["loss"] = float(_l0)
                        _sb_soup["measured"] = True
                    _free_kw = {}
                    if _free_gate.active:
                        _t_probe = float(facts.eval_runtime or _EVAL_SECS_FALLBACK)
                        _t_w = planner_clock.write_secs(facts.clock_stats)
                        _free_tail = (decision.reserve + decision.devpass_reserve
                                      + 3.0 * _t_probe + 2.0 * _t_w)
                        _steady = tail_fill_trials(_sb_trial_secs, _sb_trial_canary)
                        _free_unit = (sb.TAIL_SAFETY * max(_steady)) if _steady else 0.0
                        _free_kw = dict(
                            extra=(_free_held if _is_main() else []),
                            extra_fits=lambda per: fc.trial_fits(
                                _remaining_secs(end_time), max(float(per), _free_unit),
                                tail=_free_tail))
                    out.soup_ran = bool(ckpt_avg._greedy_extend(
                        model2, _sb_acc, _sb_soup["n"], _sb_soup["loss"],
                        srcs=list(_sb_soup["srcs"]), **_free_kw))
                    free_held_drop(_free_held, "final_pass_done", keep=0)
                    out.artefacts_meta["online_soup"] = {
                        "n": int(_sb_soup["n"]), "loss": _sb_soup["loss"],
                        "srcs": list(_sb_soup["srcs"]),
                        "final_n": int(getattr(ckpt_avg, "soup_n_members", 0) or 0),
                        "final_loss": _finite_or_none(ckpt_avg.best_loss)}
                else:
                    out.soup_ran = bool(ckpt_avg._greedy_soup(
                        model2, second_pass=bool(decision.loop and _LOOP_SECOND_PASS),
                        second_pass_max_secs=(decision.t_soup / 2.0 if decision.loop else None)))
            except BaseException as _soup_exc:
                restore_best_local(ckpt_avg, model2, _soup_exc)
                raise
            finally:
                ckpt_avg.on_soup_pass1 = None
            ckpt_avg._restore_best(model2)
            _t_eval_probe = float(facts.eval_runtime or _EVAL_SECS_FALLBACK)
            if (decision.loop and _LOOP_A0_WEIGHT_PROBE and out.soup_ran
                    and _r0_flag((_remaining_secs(end_time) or 0.0)
                                 > (decision.reserve + decision.devpass_reserve
                                    + 3.0 * _t_eval_probe))):
                try:
                    out.artefacts_meta["a0_weight_probe"] = (
                        ckpt_avg._a0_weight_probe_sum(
                            model2, _sb_acc, int(getattr(ckpt_avg, "soup_n_members", 1) or 1),
                            _a0_soup_state if _is_main() else None,
                            margin=_SOUP_ACCEPT_MARGIN)
                        if _sb_final else
                        ckpt_avg._a0_weight_probe(model2, margin=_SOUP_ACCEPT_MARGIN))
                except Exception as exc:
                    log(f"[sn56][traj2] sonda de peso a0 falhou: {exc}")
                finally:
                    ckpt_avg._restore_best(model2)
            if _sb_final:
                _sb_acc = None
                ckpt_avg.online_resident = False
                gc.collect()
            _left_blind = _remaining_secs(end_time) or 0.0
            if (decision.extra and _PLATEAU_BLIND_AVG
                    and not (decision.loop and _LOOP_DROP_SOS)
                    and _r0_flag(_left_blind > (decision.reserve + decision.devpass_reserve
                                                + _BLIND_SOUP_EVALS * (facts.eval_runtime
                                                                       or _EVAL_SECS_FALLBACK)))):
                try:
                    _blind = ckpt_avg._soup_of_soups(
                        model2, margin=_SOUP_ACCEPT_MARGIN)
                    out.artefacts_meta["blind_soup"] = _blind
                except Exception as exc:
                    log(f"[sn56][traj2] soup-of-soups falhou: {exc}")
                finally:
                    ckpt_avg._restore_best(model2)
            _save_weights_only = dev_pass._save_weights_only
            _t_save = time.time()
            if _is_main():
                _final_loss = _finite_or_none(ckpt_avg.best_loss)
                if _early.get("saved") and _same_loss(_final_loss, _early.get("loss")):
                    sg.ship_marker_update(shipped="cross_soup", shipped_loss=_final_loss,
                                          activity="cross_soup_written")
                    _close_interim("cross_soup_written")
                else:
                    sg.ship_marker_update(activity="cross_soup_write")
                    if _save_weights_only(model2, train_request["submission_dir"],
                                          lambda m: log(m)):
                        sg.ship_marker_update(shipped="cross_soup", shipped_loss=_final_loss,
                                              activity="cross_soup_written")
                        _close_interim("cross_soup_written")
            _save_secs = max(0.0, time.time() - _t_save)
            if _mr:
                _save_secs = _r0(_save_secs)
            _calib_ship(model2, ckpt_avg, train_request["submission_dir"], end_time,
                        decision.reserve + decision.devpass_reserve, facts, log, out)

            if trainer2_trained:
                out.devfit = post_soup_dev_pass(
                    plan=devfit, trainer2=trainer2, model2=model2,
                    submission_dir=train_request["submission_dir"],
                    end_time=end_time, reserve=decision.reserve,
                    save_secs=_save_secs,
                    eval_secs=planner_clock.eval_b1_secs(facts.clock_stats, facts.eval_runtime),
                    log=log)
            elif not _sw.devfit_untrained:
                out.devfit = {"ran": False, "reason": "recovered_trainer",
                              "applied_to": "cross_soup", "shipped": "cross_soup",
                              "post_soup": True,
                              "variant": "annealed-devpass"}
            else:
                _route, _why = devfit_route(trainer2_origin, enabled=devfit is not None)
                if _mr:
                    _route, _why = _r0((_route, _why))
                if _route == "untrained":
                    out.devfit = post_soup_dev_pass(
                        plan=devfit, trainer2=trainer2, model2=model2,
                        submission_dir=train_request["submission_dir"],
                        end_time=end_time, reserve=decision.reserve,
                        save_secs=_save_secs,
                        eval_secs=planner_clock.eval_b1_secs(facts.clock_stats,
                                                             facts.eval_runtime),
                        log=log,
                        prepare=functools.partial(prepare_untrained_for_devfit,
                                                  trainer2, model2))
                    out.devfit["trainer_origin"] = trainer2_origin
                else:
                    out.devfit = {"ran": False, "reason": _why or "recovered_trainer",
                                  "applied_to": "cross_soup", "shipped": "cross_soup",
                                  "post_soup": True,
                                  "variant": "annealed-devpass",
                                  "trainer_origin": trainer2_origin}
        except Exception as exc:
            log(f"[sn56][traj2] soup cruzado falhou: {exc}")

    if _free_held:
        free_held_drop(_free_held, "no_final_pass", keep=0)
    out.spare_final = emit_spare("after_soup", end_time=end_time,
                                 decision=Decision(reserve=decision.reserve,
                                                   t_traj=decision.t_traj),
                                 results=out.results)

    if _is_main():
        eval_squeeze.LEDGER.emit()
    return out
