import compute_affordance
import release_defaults
"""
In-process adaptive learning rate search.

Runs inside the training script after model and data are loaded. Tests
learning rates via short training trials (restore weights between each),
narrowing toward the optimum in three time-gated passes:

  S1  coarse  : 4 LRs around the estimate; ascends the high side only while it
                doesn't diverge, else pivots to lower LRs. S1 gets first claim on
                the budget and goes as deep as it can (the estimate is strong).
  S2  refine  : the S1 winner + 3 neighbours at full depth (skipped if it won't
                fit a real floor — a shallow refine is noisier than the S1 win).
  S3  polish  : opportunistic — extend past an edge winner AND fill the
                widest interior gap, while clock remains.

Budget is epoch-keyed and capped at _SEARCH_FRACTION (20%) of the total:
  <1 epoch  → validate (3-LR scan) if it fits, else skip to the estimate.
  1 <= e < 2 → two_stage (S1 + S2).   e >= 2 → full (S1 + S2 + S3).
S2/S3 run only if S1 cleared the refine gate AND time remains.

Probe data is cached once and recorded identically across trials (fair
comparison); windows are per-tier and consecutive (refine continues from S1).

Two pruning mechanisms abort hopeless trials early:
  - divergence : rolling loss non-finite or > 1.75x the trial's own best-seen.
  - relative   : a trial is much worse than the best trial seen at the
                 *same* step index (apples-to-apples).

Key properties:
- Single model load — no subprocess overhead per trial.
- Probe weights are DISCARDED — training restarts from the original weights on
  full data, so it never re-sees (over-weights) any probe's batches.
- Loss comparisons only happen within a tier (same step count), so the
  edge-of-stability selection threshold is meaningful.

References:
- Smith 2015: "Cyclical Learning Rates" (LR range test)
- Izmailov et al. 2018: "Averaging Weights Leads to Wider Optima"
- "Stepping on the Edge" (NeurIPS 2024): optimal LR sits at the stability edge
"""

import gc
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from collections.abc import Mapping


def _is_oom_error(exception: Exception) -> bool:
    
    _statements = [
        " out of memory.",
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
        "DefaultCPUAllocator: can't allocate memory",
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def _clear_memory():
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _materialize_batches(loader, n_batches: int, max_rows: Optional[int] = None,
                         min_batches: Optional[int] = None) -> list:
    batches = []
    rows = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        if max_rows is not None and min_batches is not None and i >= int(min_batches) \
                and (rows <= 0 or rows >= int(max_rows)):
            break
        if isinstance(batch, Mapping) and isinstance(batch.get("input_ids"), torch.Tensor):
            rows += int(batch["input_ids"].shape[0])
        if isinstance(batch, dict):
            batches.append({
                k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            })
        else:
            batches.append(batch)
    return batches


def _pad_batches_to(batches: list, target_len: int, pad_token_id: int,
                    padding_side: str) -> list:
    
    if not target_len or target_len <= 0:
        return batches
    fill = {"input_ids": int(pad_token_id), "attention_mask": 0, "labels": -100,
            "position_ids": 0}
    out = []
    for batch in batches:
        if not isinstance(batch, Mapping) or "input_ids" not in batch:
            raise TypeError(
                "_pad_batches_to cannot hold a batch of type "
                f"{type(batch).__name__} at max_length; the LR-search hold would be "
                "silently disabled. Teach it this batch shape or pass warmup_pad_to=None."
            )
        cur = batch["input_ids"].shape[-1]
        if cur >= target_len:
            out.append(batch)
            continue
        gap = target_len - cur
        padded = {}
        for key, val in batch.items():
            if not isinstance(val, torch.Tensor) or val.dim() != 2 or val.shape[-1] != cur:
                padded[key] = val
                continue
            block = torch.full((val.shape[0], gap), fill.get(key, 0), dtype=val.dtype)
            padded[key] = (torch.cat([block, val], dim=1) if padding_side == "left"
                           else torch.cat([val, block], dim=1))
        out.append(padded)
    bad = [b["input_ids"].shape[-1] for b in out
           if isinstance(b, Mapping) and "input_ids" in b
           and b["input_ids"].shape[-1] < target_len]
    if bad:
        raise AssertionError(
            f"LR-search hold failed: {len(bad)} of {len(out)} batches shorter than "
            f"{target_len} (min {min(bad)})"
        )
    return out


_S1_HALF_RANGE = 0.30
_S1_N = 4
_S1_MIN_STEPS = 25
_S1_MAX_STEPS = 100
_S1_REFINE_GATE = 50
_S2_MIN_STEPS = 50
_S2_MAX_STEPS = 150
_MIN_GAP = 0.04
_DEDUP_TOL = 0.02
_MAX_CACHED_BATCHES = 400
_MAX_CACHED_ROWS = _MAX_CACHED_BATCHES * 16


def s1_steps_for_horizon(time_steps: int, s1_min: int, s1_max: int, n_rows: Optional[int] = None,
                         rows_per_step: Optional[int] = None, frac: float = 0.0) -> tuple[int, Optional[int]]:
    steps = int(time_steps)
    ex_steps = None
    if frac and n_rows and rows_per_step:
        ex_steps = max(1, int(math.ceil(float(frac) * float(n_rows) / max(1, int(rows_per_step)))))
        steps = min(steps, ex_steps)
    return max(int(s1_min), min(int(s1_max), steps)), ex_steps


def _split_batch(batch: dict) -> tuple[dict, dict]:
    rows = int(batch["input_ids"].shape[0])
    h = rows // 2
    a, b = {}, {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and int(v.shape[0]) == rows:
            a[k], b[k] = v[:h], v[h:]
        else:
            a[k], b[k] = v, v
    return a, b


def _grad_sq_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return total


def _batch_rows_tokens(batch) -> tuple[int, int]:
    if not isinstance(batch, Mapping) or "input_ids" not in batch:
        return 0, 0
    rows = int(batch["input_ids"].shape[0])
    mask = batch.get("attention_mask")
    tokens = int(mask.sum().item()) if isinstance(mask, torch.Tensor) else int(batch["input_ids"].numel())
    return rows, tokens


class GradNoiseAccumulator:

    def __init__(self, world_size: int = 1, first_n: int = 3):
        self.world = max(1, int(world_size))
        self.active = True
        self.trS = 0.0
        self.G2 = 0.0
        self.n = 0
        self.n_neg = 0
        self.tokens = 0.0
        self.rows_small = 0
        self.rows_big = 0
        self.first_n = int(first_n)
        self.first_trS = 0.0
        self.first_G2 = 0.0

    def add(self, g_small_sq: float, g_big_sq: float, rows_small: int, rows_big: int, tokens: int) -> None:
        bs = int(rows_small) * self.world
        bb = int(rows_big) * self.world
        if bs <= 0 or bb <= bs or not (math.isfinite(g_small_sq) and math.isfinite(g_big_sq)):
            return
        trS = (g_small_sq - g_big_sq) / (1.0 / bs - 1.0 / bb)
        G2 = g_big_sq - trS / bb
        self.trS += trS
        self.G2 += G2
        self.n += 1
        if trS < 0:
            self.n_neg += 1
        self.tokens += float(tokens) * self.world
        self.rows_small, self.rows_big = bs, bb
        if self.n <= self.first_n:
            self.first_trS += trS
            self.first_G2 += G2

    @staticmethod
    def _ratio(trS: float, G2: float) -> tuple[Optional[float], str]:
        if trS <= 0:
            return None, "no_noise"
        if G2 <= 0:
            return float("inf"), "no_signal"
        return trS / G2, "ok"

    def result(self) -> dict:
        b_rows, status = self._ratio(self.trS, self.G2)
        b_first, _ = self._ratio(self.first_trS, self.first_G2)
        tok_per_row = (self.tokens / (self.n * self.rows_big)) if (self.n and self.rows_big) else None
        return {
            "n_steps": self.n, "n_neg_steps": self.n_neg, "rows_small": self.rows_small,
            "rows_big": self.rows_big, "trS": self.trS, "G2": self.G2,
            "b_noise_rows": b_rows, "status": status,
            "b_noise_rows_first": b_first,
            "tokens_per_row": tok_per_row,
            "b_noise_tokens": (b_rows * tok_per_row if (b_rows is not None and tok_per_row
                                                        and math.isfinite(b_rows)) else b_rows),
        }


GNS_MIN_ROWS_MULTI_RANK = 1.0


def pool_gns(per_rank: list, world: int) -> Optional[dict]:
    rs = [r for r in per_rank if r]
    if not rs or len(rs) < len(per_rank):
        return None
    w = max(1, int(world))
    trs = sum(float(r.get("trS") or 0.0) for r in rs) / w
    g2 = sum(float(r.get("G2") or 0.0) for r in rs)
    b, status = GradNoiseAccumulator._ratio(trs, g2)
    if status == "ok" and b is not None and b < GNS_MIN_ROWS_MULTI_RANK:
        status = "sub_row"
    n = min(int(r.get("n_steps") or 0) for r in rs)
    tpr = [r.get("tokens_per_row") for r in rs if r.get("tokens_per_row")]
    tok_per_row = (sum(tpr) / len(tpr)) if tpr else None
    return {
        "n_steps": n, "n_neg_steps": sum(int(r.get("n_neg_steps") or 0) for r in rs),
        "rows_small": int(rs[0].get("rows_small") or 0) // w,
        "rows_big": int(rs[0].get("rows_big") or 0) // w,
        "trS": trs, "G2": g2, "b_noise_rows": b,
        "status": status, "b_noise_rows_first": None, "tokens_per_row": tok_per_row,
        "b_noise_tokens": (b * tok_per_row if (b is not None and tok_per_row and math.isfinite(b))
                           else b),
        "pruned": any(bool(r.get("pruned")) for r in rs),
        "pooled_ranks": len(rs),
        "per_rank_b_noise_rows": [r.get("b_noise_rows") for r in rs],
        "per_rank_status": [r.get("status") for r in rs],
    }


def gns_reliable(res: Optional[dict], min_steps: Optional[int] = None) -> tuple[bool, str]:
    lo = int(_S1_MIN_STEPS if min_steps is None else min_steps)
    if not res:
        return False, "no_estimate"
    status = res.get("status")
    if status != "ok":
        return False, f"status_{status}"
    if res.get("pruned"):
        return False, "pruned"
    n = int(res.get("n_steps") or 0)
    if n < lo:
        return False, f"n_steps_{n}_lt_{lo}"
    b = res.get("b_noise_rows")
    if b is None or not math.isfinite(float(b)) or float(b) <= 0:
        return False, "no_estimate"
    return True, "ok"


def _probe_steps_within_cache(opt_steps: int, grad_accum_steps: int, n_cached: int) -> int:
    
    ga = max(1, int(grad_accum_steps or 1))
    if n_cached <= 0 or opt_steps * ga <= n_cached:
        return opt_steps
    return max(1, n_cached // ga)


def _probe_cache_policy(opt_steps: int, grad_accum_steps: int, n_cached: int,
                        s1_steps: int) -> tuple:
    
    ga = max(1, int(grad_accum_steps or 1))
    if n_cached <= 0:
        return opt_steps, "none"
    if n_cached < s1_steps * ga:
        capped = _probe_steps_within_cache(opt_steps, ga, n_cached)
        return capped, ("capped" if capped < opt_steps else "none")
    if opt_steps * ga > n_cached:
        return opt_steps, "tolerated"
    return opt_steps, "none"
_PROBE_WARMUP = 5

_EDGE_TOLERANCE = 0.025

LAST_SELECTION: dict = {}
LAST_GNS: dict = {}

_DPO_HOLDOUT_BATCHES = 24
_MAX_EDGE_EXTENSIONS = 4

_PRUNE_WINDOW = 5
_PRUNE_WINDOW_ROWS = 32
_PRUNE_MIN_STEP = 3
_T_HINT_SLACK = 1.25
_PRUNE_DIVERGE_FACTOR = 1.75
_PRUNE_REL_MARGIN = 0.50
_PRUNE_REL_STEP = 5

_TRAIN_UTIL = 0.85
_SEARCH_FRACTION = 0.20
_LEGACY_CAP_SECS = 20 * 60
_VALIDATE_N = 3
_VALIDATE_MIN_STEPS = 15

_MAX_OOM_RETRIES = 8

_WARMUP_STEPS = 8
_WC_MIN_POINTS = 3
_WC_NEUTRAL_DROP = 0.04
_WC_GAIN = 4.0
_WC_MIN, _WC_MAX = 0.6, 1.6


def _warmup_curvature_factor(step_losses: list) -> tuple[float, dict]:
    
    pts = [float(l) for l in step_losses if l is not None and math.isfinite(l)]
    if len(pts) < _WC_MIN_POINTS:
        return 1.0, {"reason": "too_few_points", "n": len(pts)}
    k = max(1, len(pts) // 3)
    l_start = sum(pts[:k]) / k
    l_end = sum(pts[-k:]) / k
    rel_drop = (l_start - l_end) / max(abs(l_start), 1e-6)
    factor = 1.0 + _WC_GAIN * (rel_drop - _WC_NEUTRAL_DROP)
    factor = max(_WC_MIN, min(_WC_MAX, factor))
    return factor, {"l_start": l_start, "l_end": l_end, "rel_drop": rel_drop, "n": len(pts)}


_DEGENERATE_REL_SPREAD = 1e-6
_PROBE_NEFT_SEED = 1234


def _input_embeddings(model):
    
    try:
        base = model
        if hasattr(base, "base_model") and hasattr(base.base_model, "model") and \
                type(base).__name__.startswith("Peft"):
            base = base.base_model.model
        emb = base.get_input_embeddings()
        return emb if isinstance(emb, torch.nn.Module) else None
    except Exception:
        return None


class _ProbeNeftune:
    

    def __init__(self, model, alpha: Optional[float], seed: int = _PROBE_NEFT_SEED):
        self.model = model
        self.alpha = float(alpha) if alpha else 0.0
        self.seed = seed
        self.active = False
        self._handle = None
        self._gen = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.active = False
        return False

    def engage(self) -> bool:
        
        if self.active or self.alpha <= 0:
            return False
        emb = _input_embeddings(self.model)
        if emb is None:
            return False
        self._handle = emb.register_forward_hook(self._hook)
        self.active = True
        self.reset()
        return True

    def reset(self) -> None:
        self._gen = None

    def _hook(self, module, inputs, output):
        if not module.training or not isinstance(output, torch.Tensor) or output.dim() < 3:
            return output
        if self._gen is None or self._gen.device != output.device:
            self._gen = torch.Generator(device=output.device)
            self._gen.manual_seed(self.seed)
        mag = self.alpha / math.sqrt(output.size(1) * output.size(2))
        return output + torch.empty_like(output).uniform_(-mag, mag, generator=self._gen)


def _degenerate_pool(scores: dict) -> Optional[str]:
    
    vals = [v for v in scores.values() if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return None
    lo = min(vals)
    if max(vals) - lo <= _DEGENERATE_REL_SPREAD * max(1.0, abs(lo)):
        return "flat_scores"
    return None


class _BatchChanged(Exception):
    pass


class _SearchExhausted(Exception):
    pass


@dataclass
class BudgetPlan:
    
    mode: str
    budget_secs: float
    epochs_affordable: float
    reason: str


def plan_budget(
    total_secs: float,
    steps_per_epoch: Optional[int],
    t_per_step: float,
) -> BudgetPlan:
    
    search_budget = _SEARCH_FRACTION * total_secs
    _pct = f"{_SEARCH_FRACTION:.0%}"

    if not steps_per_epoch or steps_per_epoch <= 0 or t_per_step <= 0:
        return BudgetPlan(
            mode="full",
            budget_secs=min(search_budget, _LEGACY_CAP_SECS),
            epochs_affordable=float("nan"),
            reason=f"legacy: full search within {_pct} (no steps_per_epoch)",
        )

    epoch_secs = steps_per_epoch * t_per_step
    epochs_affordable = total_secs * _TRAIN_UTIL / epoch_secs
    validate_cost = _VALIDATE_N * _VALIDATE_MIN_STEPS * t_per_step
    s1_cost = _S1_N * _S1_MIN_STEPS * t_per_step

    def _scan_or_skip(tag: str) -> BudgetPlan:
        if search_budget >= validate_cost:
            return BudgetPlan(
                mode="validate", budget_secs=search_budget,
                epochs_affordable=epochs_affordable,
                reason=f"{tag} ({epochs_affordable:.2f}), 3-LR scan within {_pct}",
            )
        return BudgetPlan(
            mode="skip", budget_secs=0.0, epochs_affordable=epochs_affordable,
            reason=f"{tag} ({epochs_affordable:.2f}), scan won't fit {_pct}",
        )

    if epochs_affordable < 1.0:
        return _scan_or_skip("<1 epoch")

    if epochs_affordable < 2.0:
        if search_budget >= s1_cost:
            return BudgetPlan(
                mode="two_stage", budget_secs=search_budget,
                epochs_affordable=epochs_affordable,
                reason=f"1<=epochs<2 ({epochs_affordable:.2f}), S1+S2 within {_pct}",
            )
        return _scan_or_skip("1<=epochs<2 but tight")

    if search_budget >= s1_cost:
        return BudgetPlan(
            mode="full", budget_secs=search_budget,
            epochs_affordable=epochs_affordable,
            reason=f">=2 epochs ({epochs_affordable:.2f}), S1+S2+S3 within {_pct}",
        )
    return _scan_or_skip(">=2 epochs but tight")


def restate_steps_per_epoch_for_plan(stale, steps_per_epoch_fn, adopt, oom_retries=0):
    
    if steps_per_epoch_fn is None:
        return stale
    try:
        live = steps_per_epoch_fn()
    except Exception:
        return stale
    if not live or live == stale:
        return stale
    if not adopt:
        return stale
    print(
        f"[sn56][farejando] passos/época corrigidos {stale} -> {live} "
        f"({oom_retries} corte(s) de lote no aquecimento)",
        flush=True,
    )
    return live


def select_edge_lr(
    scores: dict[float, float], tolerance: float = _EDGE_TOLERANCE
) -> tuple[float, float]:
    
    edge_log, edge_loss = _select_edge_log(scores, tolerance)
    return 10 ** edge_log, edge_loss


def _select_edge_log(scores: dict, tolerance: float = _EDGE_TOLERANCE) -> tuple[float, float]:
    finite = {k: v for k, v in scores.items() if v is not None and math.isfinite(v)}
    if not finite:
        k0 = min(scores)
        return k0, float("inf")
    scores = finite
    best_loss = min(scores.values())
    threshold = best_loss * (1 + tolerance)
    stable = {lg: ls for lg, ls in scores.items() if ls <= threshold}
    edge_log = max(stable.keys())
    return edge_log, stable[edge_log]


_CENTRE_CONST = 0.75
_SCORE_TRIM = 0.10
_SCORE_TAIL_FRAC = 0.20
_SCORE_TAIL_FRAC_SMALL = 0.40
_SCORE_SMALL_ROWS = 8
_FIT_OFFSET = float(os.environ.get("SN56_LR_FIT_OFFSET", "0"))
_FIT_FLOOR = float(os.environ.get("SN56_LR_FIT_FLOOR", "0"))
_B_FIT_OFFSET = 0.20
_B_FIT_FLOOR = 0.10
_FIT_EDGE_PAD = 0.15
_FIT_HOT_CAP = 0.15
_FIT_DIVERGE_STEP = 0.15
_EXT_COLD_TRIGGER = 0.05
_EXT_COLD_STEP = 0.30
_EXT_HOT_STEP = 0.15
_EXT_HOT_TOL = _EDGE_TOLERANCE
_CARRY_LR_CAL_ENV = "SN56_CARRY_LR_CAL"
_EDGE_COLD_CONCAVE = "edge_cold_concave"


CARRY_LR_CAL_SUPPRESSED: Optional[str] = None


def carry_lr_cal_enabled() -> bool:
    return release_defaults.raw(_CARRY_LR_CAL_ENV).strip().lower() not in ("", "0", "false", "no", "off")


_COLDEDGE_FUND_ENV = "SN56_LR_COLDEDGE_FUND"


def coldedge_fund_enabled() -> bool:
    return release_defaults.raw(_COLDEDGE_FUND_ENV).strip().lower() not in ("", "0", "false", "no", "off")


def measured_floor(x: float, scored: list, funded_rung: Optional[float],
                   caps: Optional[dict] = None) -> tuple[float, Optional[dict]]:
    if not scored:
        return x, None
    floor, why = min(scored), "coldest_measured"
    if funded_rung is not None and funded_rung > floor:
        floor, why = funded_rung, "funded_rung"
    if x >= floor - 1e-12:
        return x, None
    x2 = floor
    live = [v for k, v in (caps or {}).items() if not k.startswith("_")]
    if live:
        x2 = min(x2, min(live))
    if x2 <= x:
        return x, None
    return x2, {"why": why, "floor": floor, "x_before": x, "x_after": x2,
                "lr_before": 10 ** x, "lr_after": 10 ** x2}


def coldedge_fund_decision(pool: dict, x_probe: float, cost_secs: float,
                           remaining_secs: Optional[float]) -> dict:
    d = {"fired": False, "probe_log_lr": x_probe, "cost_secs": round(cost_secs, 1),
         "remaining_secs": round(remaining_secs, 1) if remaining_secs is not None else None,
         "gain_pred": None, "cost_frac": None}
    if remaining_secs is None or not math.isfinite(remaining_secs) or remaining_secs <= cost_secs:
        d["reason"] = "no_run_time"
        return d
    fin = {k: v for k, v in pool.items() if v is not None and math.isfinite(v)}
    xv, a, flag = fit_local(fin)
    d.update(x_v=xv, a=a, flag=flag)
    lo = min(fin) if fin else None
    if xv is None or a is None or not a > 0 or lo is None or not fin[lo] > 0:
        d["reason"] = "no_curvature"
        return d
    xt = min(max(xv, x_probe), lo)
    g = a * ((lo - xv) ** 2 - (xt - xv) ** 2) / fin[lo]
    c = cost_secs / remaining_secs
    d.update(gain_pred=g, cost_frac=c, fired=bool(g >= c),
             reason=("ev_covers_cost" if g >= c else "ev_below_cost"))
    return d
_EXT_EPOCH_STRETCH = 1.25
_S2_FIT_OFFSETS = (-0.15, 0.0, 0.15)
_S2_CLOCK_SHARE = 0.6
_S2_SLOTS = len(_S2_FIT_OFFSETS) + 1


def _trimmed_tail_score(step_losses: list, rows_per_step: Optional[int]) -> float:
    frac = _SCORE_TAIL_FRAC_SMALL if (rows_per_step and int(rows_per_step) <= _SCORE_SMALL_ROWS) \
        else _SCORE_TAIL_FRAC
    tail_start = max(1, int(len(step_losses) * (1.0 - frac)))
    tail = sorted(step_losses[tail_start:]) or sorted(step_losses)
    k = max(1, int(_SCORE_TRIM * len(tail)))
    kept = tail[k:-k] if len(tail) > 2 * k else tail
    return sum(kept) / len(kept)


def fit_local(pool: dict) -> tuple[Optional[float], Optional[float], str]:
    pool = {k: v for k, v in pool.items() if v is not None and math.isfinite(v)}
    xs = sorted(pool)
    if len(xs) < 3:
        return None, None, "n<3"
    ys = [pool[x] for x in xs]
    i = int(np.argmin(ys))
    if i == 0:
        idx = [0, 1, 2]; flag = "edge_cold"
    elif i == len(xs) - 1:
        idx = [len(xs) - 3, len(xs) - 2, len(xs) - 1]; flag = "edge_hot"
    else:
        idx = [i - 1, i, i + 1]; flag = "interior"
    x = np.array([xs[j] for j in idx]); y = np.array([ys[j] for j in idx])
    a, b, _ = np.polyfit(x, y, 2)
    if not a > 0:
        return None, float(a), flag + "_concave"
    xv = float(-b / (2 * a))
    if not math.isfinite(xv):
        return None, float(a), flag + "_nonfinite"
    return xv, float(a), flag


def _clamp_vertex(xv: float, lo: float, hi: float) -> float:
    return min(max(xv, lo - _FIT_EDGE_PAD), hi + _FIT_EDGE_PAD)


def _argmin_floor(x: float, pool: dict, floor: Optional[float], hi: float,
                  diverged: Optional[list] = None) -> float:
    if floor is None:
        return x
    fin = {k: v for k, v in pool.items() if v is not None and math.isfinite(v)}
    if not fin:
        return x
    am = min(fin, key=fin.get)
    x2 = min(max(x, am + floor), hi + _FIT_HOT_CAP)
    if diverged:
        x2 = min(x2, min(diverged) - _FIT_DIVERGE_STEP)
    return x2


def fit_pick(pool: dict, offset: float = _FIT_OFFSET,
             diverged: Optional[list] = None,
             floor: Optional[float] = _FIT_FLOOR) -> tuple[float, str]:
    xs = sorted(k for k, v in pool.items() if v is not None and math.isfinite(v))
    xv, _, flag = fit_local(pool)
    if xv is None:
        x = _select_edge_log(pool)[0]
        return (_argmin_floor(x, pool, floor, xs[-1], diverged) if xs else x), "fallback_" + flag
    lo, hi = xs[0], xs[-1]
    x = _clamp_vertex(xv, lo, hi) + offset
    x = min(x, hi + _FIT_HOT_CAP)
    if diverged:
        x = min(x, min(diverged) - _FIT_DIVERGE_STEP)
    return _argmin_floor(x, pool, floor, hi, diverged), flag


@dataclass
class TierFit:
    tier: str
    pool: dict
    n_steps: list


def combine_tiers(tiers: list, offset: float = _FIT_OFFSET,
                  floor: Optional[float] = _FIT_FLOOR) -> tuple[float, str, dict]:
    used = []
    diag: dict = {}
    for t in tiers:
        if len(t.pool) < 3:
            continue
        xv, a, flag = fit_local(t.pool)
        diag[t.tier] = {"x_v": xv, "a": a, "flag": flag}
        if xv is None:
            continue
        n = float(np.median(t.n_steps or [1]))
        _fin = [k for k, v in t.pool.items() if v is not None and math.isfinite(v)]
        lo, hi = min(_fin), max(_fin)
        used.append((_clamp_vertex(xv, lo, hi), n, hi))
        diag[t.tier].update(n=n, x_v_clamped=used[-1][0], clamped=used[-1][0] != xv)
    if not used:
        deepest = next((t.pool for t in reversed(tiers) if t.pool), None) or {}
        if not deepest:
            return float("nan"), "none", diag
        x = _select_edge_log(deepest)[0]
        _fin = [k for k, v in deepest.items() if v is not None and math.isfinite(v)]
        return (_floor_diag(x, deepest, floor, max(_fin), diag) if _fin else x), "fit_fallback_edge", diag
    xv = sum(x * n for x, n, _ in used) / sum(n for _, n, _ in used)
    hi = max(h for _, _, h in used)
    diag["x_v_combined"] = xv
    x = min(xv + offset, hi + _FIT_HOT_CAP)
    deepest = next((t.pool for t in reversed(tiers) if t.pool), None) or {}
    x = _floor_diag(x, deepest, floor, hi, diag)
    return x, ("fit_comb" if len(used) > 1 else "fit"), diag


def _floor_diag(x: float, pool: dict, floor: Optional[float], hi: float, diag: dict) -> float:
    x2 = _argmin_floor(x, pool, floor, hi)
    fin = {k: v for k, v in pool.items() if v is not None and math.isfinite(v)}
    am = min(fin, key=fin.get) if fin else None
    diag.update(x_fit=x, x_argmin_floor=(am + floor if (am is not None and floor is not None) else None),
                floor=floor, floor_bound=bool(x2 > x), floor_dex=x2 - x)
    return x2


def final_caps(x_ship: float, rule: str, x_ref: Optional[float], pools: list,
               pruned: list, hot_rejected, hot_accepted: bool) -> tuple[float, dict]:
    caps: dict = {}
    if rule == "fit_fallback_edge":
        return x_ship, caps
    ref = x_ref if (x_ref is not None and math.isfinite(x_ref)) else -math.inf
    rej = [r for r in hot_rejected if r > ref]

    def _is_rej(k):
        return any(abs(k - r) < _DEDUP_TOL for r in hot_rejected)

    div = [x for x, why in pruned if is_diverged_reason(why) and x > ref]
    ignored = [x for x, why in pruned if is_diverged_reason(why) and x <= ref]
    if div:
        caps["diverged"] = min(div) - _FIT_DIVERGE_STEP
    if rej:
        r0 = min(rej)
        below = [k for pool_ in pools for k, v in pool_.items()
                 if v is not None and math.isfinite(v) and k < r0 - _DEDUP_TOL and not _is_rej(k)]
        caps["hot_rejected"] = max([r0 - _FIT_DIVERGE_STEP] + below)
    scored = [k for pool_ in pools for k, v in pool_.items()
              if v is not None and math.isfinite(v) and not _is_rej(k)]
    if scored:
        top = max(scored)
        caps["hottest_probed" if not hot_accepted else "hottest_probed_plus_rung"] = (
            top + (_FIT_HOT_CAP if hot_accepted else 0.0))
    x = min([x_ship] + list(caps.values()))
    if ignored:
        caps["_ignored_cold_pruned"] = ignored
        x = min([x_ship] + [v for k, v in caps.items() if not k.startswith("_")])
    return x, caps


def extension_request(x_v: Optional[float], lo: float, hi: float,
                      hi_all: Optional[float] = None,
                      offset: float = _FIT_OFFSET) -> Optional[str]:
    if x_v is None or not math.isfinite(x_v):
        return None
    if x_v < lo - _EXT_COLD_TRIGGER:
        return "cold"
    top = hi if hi_all is None else hi_all
    if _clamp_vertex(x_v, lo, hi) + offset > top:
        return "hot"
    return None


def cold_cal_request(x_v: Optional[float], flag: Optional[str], lo: float, hi: float,
                     hi_all: Optional[float] = None) -> Optional[str]:
    d = extension_request(x_v, lo, hi, hi_all)
    if d is None and (x_v is None or not math.isfinite(x_v)) and flag == _EDGE_COLD_CONCAVE:
        return "cold"
    return d


def extension_is_free(
    remaining_secs: Optional[float],
    probe_secs: float,
    t_per_step: float,
    steps_per_epoch: Optional[int],
    max_epochs_cap: Optional[float],
) -> tuple[bool, dict]:
    d = {"remaining_secs": round(remaining_secs, 1) if remaining_secs is not None else None,
         "probe_secs": round(probe_secs, 1),
         "steps_per_epoch": steps_per_epoch,
         "max_epochs_cap": max_epochs_cap,
         "achievable_steps_before": None, "achievable_steps_after": None,
         "epochs_capped_before": None, "epochs_capped_after": None,
         "headroom_steps": None}
    if (remaining_secs is None or max_epochs_cap is None
            or not steps_per_epoch or steps_per_epoch <= 0 or t_per_step <= 0):
        d["slack_reason"] = "unknown (no deadline, epoch cap or steps/epoch) - unproven"
        return False, d
    before = remaining_secs * _TRAIN_UTIL / t_per_step
    after = max(0.0, remaining_secs - probe_secs) * _TRAIN_UTIL / t_per_step
    need = max_epochs_cap * steps_per_epoch / _EXT_EPOCH_STRETCH
    d["achievable_steps_before"] = int(before)
    d["achievable_steps_after"] = int(after)
    d["epochs_capped_before"] = before >= need
    d["epochs_capped_after"] = after >= need
    d["headroom_steps"] = int(after - need)
    if not d["epochs_capped_before"]:
        d["slack_reason"] = "clock-bound before the probe - a probe is steps not taken"
        return False, d
    if not d["epochs_capped_after"]:
        d["slack_reason"] = "the probe would make the run clock-bound"
        return False, d
    d["slack_reason"] = "epoch-capped after the probe - the seconds are headroom"
    return True, d


def is_diverged_reason(reason: Optional[str]) -> bool:
    return bool(reason) and str(reason).startswith(("diverge", "rel(step"))


_ALLDIVERGE_FIX_ENV = "SN56_LR_ALLDIVERGE_FIX"
_ABOVE_INIT_REASON = "diverge_above_init"
_ABOVE_INIT_Z = 2.0


def alldiverge_fix_enabled() -> bool:
    return release_defaults.raw(_ALLDIVERGE_FIX_ENV).strip().lower() not in (
        "", "0", "false", "no", "off")


def _tail_indices(n_scored: int, rows_per_step: Optional[int]) -> list:
    frac = _SCORE_TAIL_FRAC_SMALL if (rows_per_step and int(rows_per_step) <= _SCORE_SMALL_ROWS) \
        else _SCORE_TAIL_FRAC
    tail_start = max(1, int(n_scored * (1.0 - frac)))
    if tail_start >= n_scored:
        return list(range(n_scored))
    return list(range(tail_start, n_scored))


@torch.no_grad()
def _init_tail_score(model, batches, opt_steps: int, n_scored: int, grad_accum_steps: int = 1,
                     data_offset: int = 0, warmup_steps: int = _PROBE_WARMUP,
                     loss_fn: Optional[Callable] = None, chunker=None,
                     score_rows: Optional[int] = None,
                     out_tail: Optional[dict] = None) -> float:
    ga = max(1, int(grad_accum_steps or 1))
    n_cached = len(batches)
    if n_cached == 0 or n_scored <= 0:
        return float("inf")
    warmup = min(warmup_steps, max(0, opt_steps // 3))
    idx = _tail_indices(n_scored, score_rows)
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    was_training = model.training
    model.train()
    _devs = [device.index if device.index is not None else torch.cuda.current_device()] \
        if use_amp else []
    losses = {}
    try:
        with torch.random.fork_rng(devices=_devs):
            for j in idx:
                opt_idx = warmup + j + 1
                acc = 0.0
                for m in range((opt_idx - 1) * ga, opt_idx * ga):
                    b = batches[(data_offset + m) % n_cached]
                    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                    if chunker is not None:
                        out = chunker.fwd_bwd(model, b, loss_fn, 1.0 / ga,
                                              autocast_dtype=(torch.bfloat16 if use_amp else None),
                                              backward=False)
                    elif use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            out = (loss_fn(model, b) if loss_fn is not None else model(**b).loss) / ga
                    else:
                        out = (loss_fn(model, b) if loss_fn is not None else model(**b).loss) / ga
                    acc += float(out.item()) * ga
                losses[j] = acc / ga
    finally:
        if not was_training:
            model.eval()
    if out_tail is not None:
        out_tail.update(losses)
    full = [losses.get(j, 0.0) for j in range(n_scored)]
    return _trimmed_tail_score(full, score_rows)


def _paired_tail_se(step_losses: list, init_tail: dict) -> Optional[float]:
    d = [step_losses[j] - v for j, v in init_tail.items()
         if 0 <= j < len(step_losses) and math.isfinite(step_losses[j]) and math.isfinite(v)]
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return math.sqrt(var / n)


@torch.no_grad()
def _save_trainable_state(model) -> dict[str, torch.Tensor]:
    
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.cpu().clone()
    return state


def _restore_trainable_state(model, state: dict[str, torch.Tensor]) -> None:
    
    for name, param in model.named_parameters():
        if name in state:
            param.data.copy_(state[name].to(param.device))
            if not param.requires_grad:
                param.requires_grad_(True)


def _run_trial(*args, **kwargs) -> tuple[float, Optional[str]]:
    box: list = []
    try:
        return _run_trial_body(*args, _opt_box=box, **kwargs)
    finally:
        for opt in box:
            rel = getattr(opt, "release_state", None)
            if callable(rel):
                rel()


def _run_trial_body(
    model,
    batches,
    lr: float,
    opt_steps: int,
    optimizer_cls,
    optimizer_kwargs: dict,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    trainer=None,
    use_reward: bool = False,
    prune_fn: Optional[Callable[[int, float, float], Optional[str]]] = None,
    data_offset: int = 0,
    warmup_steps: int = 0,
    eval_fn: Optional[Callable[[], float]] = None,
    out_step_losses: Optional[list] = None,
    loss_fn: Optional[Callable] = None,
    gns: Optional[GradNoiseAccumulator] = None,
    prune_window: int = _PRUNE_WINDOW,
    score_rows: Optional[int] = None,
    _opt_box: Optional[list] = None,
    chunker=None,
) -> tuple[float, Optional[str]]:

    if hasattr(model, "gradient_checkpointing_enable"):
        gc_kwargs = {"use_reentrant": False}
        if trainer is not None and hasattr(trainer, 'args'):
            gc_kwargs = getattr(trainer.args, 'gradient_checkpointing_kwargs', None) or gc_kwargs
            trainer.args.gradient_checkpointing_kwargs = gc_kwargs
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_cls(trainable_params, lr=lr, **optimizer_kwargs)
    if _opt_box is not None:
        _opt_box.append(optimizer)

    device = next(model.parameters()).device
    use_amp = device.type == "cuda"

    if use_reward and trainer is not None and hasattr(trainer, '_metrics'):
        trainer._metrics["train"].clear()

    model.train()
    step_losses = []
    micro_loss_accum = 0.0
    micro_count = 0
    total_batches = opt_steps * grad_accum_steps
    prune_reason: Optional[str] = None
    n_cached = len(batches)
    if n_cached == 0:
        return float("inf"), prune_reason
    warmup = min(warmup_steps, max(0, opt_steps // 3))
    opt_idx = 0
    nonfinite_steps: list = []
    n_nonfinite_scored = 0
    gns_on = gns is not None and gns.active
    gns_pending: Optional[tuple] = None

    def _micro_fwd_bwd(b, scale: float):
        if trainer is not None:
            out = trainer.training_step(model, b)
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out, device=device)
            return out
        if chunker is not None:
            return chunker.fwd_bwd(model, b, loss_fn, scale,
                                   autocast_dtype=(torch.bfloat16 if use_amp else None))
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = (loss_fn(model, b) if loss_fn is not None else model(**b).loss) * scale
        else:
            out = (loss_fn(model, b) if loss_fn is not None else model(**b).loss) * scale
        out.backward()
        return out

    for i in range(total_batches):
        batch = batches[(data_offset + i) % n_cached]
        if trainer is None:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        rows, tokens = _batch_rows_tokens(batch) if gns_on else (0, 0)
        split = gns_on and grad_accum_steps == 1 and rows >= 2
        try:
            if split:
                b1, b2 = _split_batch(batch)
                loss1 = _micro_fwd_bwd(b1, 0.5)
                g_small_sq = _grad_sq_norm(trainable_params) * (1.0 if trainer is not None else 4.0)
                loss2 = _micro_fwd_bwd(b2, 0.5)
                if trainer is not None:
                    for p in trainable_params:
                        if p.grad is not None:
                            p.grad.mul_(0.5)
                    loss = (loss1 + loss2) * 0.5
                else:
                    loss = loss1 + loss2
                gns_pending = (g_small_sq, rows // 2, rows, tokens)
            else:
                loss = _micro_fwd_bwd(batch, 1.0 / grad_accum_steps)
                if gns_on and grad_accum_steps > 1 and micro_count == 0:
                    g_small_sq = _grad_sq_norm(trainable_params) * float(grad_accum_steps) ** 2
                    gns_pending = (g_small_sq, rows, rows * grad_accum_steps, tokens * grad_accum_steps)
        except RuntimeError as e:
            if _is_oom_error(e):
                optimizer.zero_grad()
                _clear_memory()
                raise
            raise

        micro_loss_accum += loss.item() * grad_accum_steps
        micro_count += 1

        if (i + 1) % grad_accum_steps == 0:
            opt_idx += 1
            for g in optimizer.param_groups:
                g["lr"] = lr * min(1.0, opt_idx / (warmup + 1))
            grad_finite = True
            if max_grad_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                grad_finite = bool(torch.isfinite(total_norm))
                if gns_pending is not None:
                    gns.add(gns_pending[0], float(total_norm) ** 2, gns_pending[1], gns_pending[2],
                            gns_pending[3])
            elif gns_pending is not None:
                gns.add(gns_pending[0], _grad_sq_norm(trainable_params), gns_pending[1],
                        gns_pending[2], gns_pending[3])
            gns_pending = None
            if grad_finite:
                optimizer.step()
            else:
                nonfinite_steps.append(opt_idx)
                if opt_idx > warmup:
                    n_nonfinite_scored += 1
            optimizer.zero_grad()
            step_loss = micro_loss_accum / micro_count
            micro_loss_accum = 0.0
            micro_count = 0
            if opt_idx <= warmup:
                continue
            step_losses.append(step_loss)

            if prune_fn is not None and not use_reward:
                opt_step = len(step_losses)
                window = step_losses[-max(1, int(prune_window)):]
                rolling = sum(window) / len(window)
                prune_reason = prune_fn(opt_step, rolling, step_losses[0])
                if prune_reason is not None:
                    break

    if out_step_losses is not None:
        out_step_losses.extend(step_losses)

    if nonfinite_steps:
        print(f"[sn56][farejando] lr={lr:.2e}: {len(nonfinite_steps)}/{opt_idx} passos "
              f"com grad não-finito (pulados)", flush=True)
        if len(nonfinite_steps) >= opt_idx or (
                step_losses and n_nonfinite_scored >= len(step_losses)):
            return float("inf"), "nonfinite_grad"

    if not step_losses:
        return float("inf"), prune_reason

    if eval_fn is not None:
        return eval_fn(), prune_reason

    if use_reward and trainer is not None and hasattr(trainer, '_metrics'):
        rewards = trainer._metrics["train"].get("reward", [])
        if rewards:
            tail_start = max(1, int(len(rewards) * 0.7))
            mean_reward = sum(rewards[tail_start:]) / len(rewards[tail_start:])
            return -mean_reward, prune_reason

    return _trimmed_tail_score(step_losses, score_rows), prune_reason


def run_lr_search(*args, neftune_noise_alpha: Optional[float] = None, **kwargs
                  ) -> tuple[float, Optional[float]]:
    
    model = kwargs["model"] if "model" in kwargs else args[0]
    with _ProbeNeftune(model, neftune_noise_alpha) as probe_noise:
        return _run_lr_search_body(*args, _probe_noise=probe_noise, **kwargs)


def _run_lr_search_body(
    model,
    train_dataloader,
    initial_lr: float,
    hours_to_complete: float,
    optimizer_cls=None,
    optimizer_kwargs: Optional[dict] = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    dataloader_factory=None,
    trainer=None,
    sync_loss_fn=None,
    use_reward: bool = False,
    steps_per_epoch: Optional[int] = None,
    steps_per_epoch_fn=None,
    restate_steps_per_epoch: bool = False,
    eval_dataloader=None,
    warmup_pad_to: Optional[int] = None,
    warmup_pad_token_id: int = 0,
    warmup_padding_side: str = "right",
    _probe_noise: Optional[_ProbeNeftune] = None,
    probe_loss_fn: Optional[Callable] = None,
    probe_horizon: Optional[dict] = None,
    gns_cfg: Optional[dict] = None,
    recentre_fn: Optional[Callable[[float, dict], Optional[float]]] = None,
    recentre_min_dex: float = 0.15,
    t_per_step_hint: Optional[float] = None,
    prune_rows_per_step: Optional[int] = None,
    centre_inputs: Optional[dict] = None,
    remaining_secs_fn: Optional[Callable[[], float]] = None,
    max_epochs_cap: Optional[float] = None,
    chunker=None,
) -> tuple[float, Optional[float]]:

    global LAST_GNS
    LAST_GNS = {}
    _prune_window = _PRUNE_WINDOW
    if prune_rows_per_step and int(prune_rows_per_step) > 0:
        _prune_window = max(_PRUNE_WINDOW, int(math.ceil(_PRUNE_WINDOW_ROWS / int(prune_rows_per_step))))
    _prune_rel_step = max(_PRUNE_REL_STEP, _prune_window)
    _prune_min_step = max(_PRUNE_MIN_STEP, _prune_window // 2)
    _gns_world = int((gns_cfg or {}).get("world_size", 1) or 1)
    _gns_rungs = int((gns_cfg or {}).get("rungs", 0) or 0) if gns_cfg else 0
    _gns_min_steps = int((gns_cfg or {}).get("min_steps", _S1_MIN_STEPS) or _S1_MIN_STEPS)
    _gns_state = {"s1_runs": 0, "last": None}

    def _gns_emit(phase: str, acc: Optional[GradNoiseAccumulator], lr_used: float,
                  pruned: bool = False) -> Optional[dict]:
        if acc is None or acc.n == 0:
            return None
        res = acc.result()
        res["pruned"] = bool(pruned)
        print(f"[sn56][farejando] escala de ruído ({phase}): B_noise={res['b_noise_rows']} linhas "
              f"(~{res['b_noise_tokens']} tokens, {res['n_steps']} passos, {res['status']})", flush=True)
        return res

    if optimizer_cls is None:
        from torch.optim import AdamW
        optimizer_cls = AdamW
    if optimizer_kwargs is None:
        optimizer_kwargs = {"weight_decay": 0.0}

    _orig_disable_tqdm = None
    if trainer is not None and hasattr(trainer, 'args'):
        _orig_disable_tqdm = getattr(trainer.args, 'disable_tqdm', None)
        trainer.args.disable_tqdm = True

    def _restore_tqdm():
        if trainer is not None and _orig_disable_tqdm is not None:
            trainer.args.disable_tqdm = _orig_disable_tqdm

    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_secs = hours_to_complete * 3600
    print(
        f"[sn56][farejando] Starting: initial_lr={initial_lr:.2e}, "
        f"device={device}, params={n_params/1e6:.1f}M, "
        f"grad_accum={grad_accum_steps}, max_grad_norm={max_grad_norm}, "
        f"hours={hours_to_complete:.2f}h, steps/epoch={steps_per_epoch}",
        flush=True,
    )

    print(f"[sn56][farejando] {_WARMUP_STEPS} warmup steps (timing + curvatura)...", flush=True)
    warmup_state = _save_trainable_state(model)
    warmup_oom_retries = 0
    def _warmup_materialize(loader):
        got = _materialize_batches(loader, _WARMUP_STEPS * grad_accum_steps)
        if warmup_pad_to:
            got = _pad_batches_to(got, warmup_pad_to, warmup_pad_token_id,
                                  warmup_padding_side)
        return got

    warmup_batches = _warmup_materialize(train_dataloader)
    warmup_losses: list = []
    while True:
        try:
            warmup_losses.clear()
            if _probe_noise is not None:
                _probe_noise.reset()
            t0 = time.perf_counter()
            _warm_gns = GradNoiseAccumulator(_gns_world) if gns_cfg else None
            _, warmup_prune = _run_trial(model, warmup_batches, initial_lr, _WARMUP_STEPS, optimizer_cls,
                       optimizer_kwargs, grad_accum_steps, max_grad_norm,
                       trainer=trainer, use_reward=use_reward,
                       out_step_losses=warmup_losses, gns=_warm_gns,
                       **({"chunker": chunker} if chunker is not None else {}))
            t_warmup = time.perf_counter() - t0
            _warm_res = _gns_emit("warmup", _warm_gns, initial_lr)
            if _warm_res is not None:
                LAST_GNS["warmup"] = _warm_res
            if (warmup_prune == "nonfinite_grad" and _probe_noise is not None
                    and _probe_noise.engage()):
                print(f"[sn56][farejando] aquecimento sem update (grad não-finito): "
                      f"provas com NEFTune alpha={_probe_noise.alpha:g}, refazendo",
                      flush=True)
                _restore_trainable_state(model, warmup_state)
                continue
            break
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            _clear_memory()
            warmup_oom_retries += 1
            _restore_trainable_state(model, warmup_state)
            if warmup_oom_retries > _MAX_OOM_RETRIES or dataloader_factory is None:
                print(f"[sn56][farejando] OOM during warmup after {warmup_oom_retries} retries, skipping search", flush=True)
                del warmup_state
                _restore_tqdm()
                return initial_lr, None
            print(f"[sn56][farejando] OOM during warmup, halving batch (retry {warmup_oom_retries}/{_MAX_OOM_RETRIES})", flush=True)
            train_dataloader = dataloader_factory()
            warmup_batches = _warmup_materialize(train_dataloader)

    _t_warmup_step = t_warmup / _WARMUP_STEPS
    t_per_step = _t_warmup_step
    if t_per_step_hint and t_per_step_hint > 0:
        t_per_step = min(_t_warmup_step, float(t_per_step_hint) * _T_HINT_SLACK)
    _restore_trainable_state(model, warmup_state)
    del warmup_state

    if warmup_prune == "nonfinite_grad":
        wc_factor, _ = 1.0, {"reason": "nonfinite_grad", "n": len(warmup_losses)}
    else:
        wc_factor, _ = _warmup_curvature_factor(warmup_losses)
    _lr_finder_scaled = initial_lr
    _sft_edge_path = not use_reward and not (eval_dataloader is not None and trainer is not None)
    _centre_factor = _CENTRE_CONST if _sft_edge_path else wc_factor
    initial_lr = initial_lr * _centre_factor
    print(
        f"[sn56][farejando] centro: lr {_lr_finder_scaled:.2e} x{_centre_factor:.3f} "
        f"-> {initial_lr:.2e} (curvatura medida x{wc_factor:.3f}, só registro)",
        flush=True,
    )

    steps_per_epoch = restate_steps_per_epoch_for_plan(
        steps_per_epoch, steps_per_epoch_fn,
        adopt=restate_steps_per_epoch, oom_retries=warmup_oom_retries)

    plan = plan_budget(total_secs, steps_per_epoch, t_per_step)
    print(
        f"[sn56][farejando] Warmup: {t_per_step:.3f}s/opt_step "
        f"({t_per_step/max(grad_accum_steps,1):.3f}s/batch, accum={grad_accum_steps}); "
        f"budget mode={plan.mode} ({plan.budget_secs/60:.1f}min) — {plan.reason}",
        flush=True,
    )

    if plan.mode == "skip":
        print(f"[sn56][farejando] Skipping search, using estimate lr={initial_lr:.2e}", flush=True)
        _clear_memory()
        _restore_tqdm()
        return initial_lr, t_per_step

    budget_secs = plan.budget_secs
    is_validate = plan.mode == "validate"
    s1_n = _VALIDATE_N if is_validate else _S1_N
    s1_min = _VALIDATE_MIN_STEPS if is_validate else _S1_MIN_STEPS
    s1_max = _S2_MAX_STEPS if is_validate else _S1_MAX_STEPS

    _s1_time_steps = int(budget_secs / (s1_n * t_per_step))
    _ph = probe_horizon or {}
    _ph_rows_per_step = int(_ph.get("rows_per_step") or 0)
    s1_steps, _ = s1_steps_for_horizon(
        _s1_time_steps, s1_min, s1_max, n_rows=_ph.get("n_rows"),
        rows_per_step=_ph_rows_per_step, frac=float(_ph.get("frac") or 0.0))
    if s1_n * s1_min * t_per_step > budget_secs:
        print(
            f"[sn56][farejando] Budget too small for S1 "
            f"(need {s1_n*s1_min*t_per_step:.0f}s, have {budget_secs:.0f}s), "
            f"using estimate lr={initial_lr:.2e}",
            flush=True,
        )
        _clear_memory()
        _restore_tqdm()
        return initial_lr, t_per_step

    initial_state = _save_trainable_state(model)
    log_center = math.log10(initial_lr)
    search_start = time.perf_counter()

    full_data_offset = s1_steps * grad_accum_steps
    _refine_span = 0 if is_validate else _S2_MAX_STEPS
    _cap_batches = (s1_steps + _refine_span) * grad_accum_steps
    _cache_kw = dict(max_rows=_MAX_CACHED_ROWS, min_batches=_MAX_CACHED_BATCHES)
    probe_batches = _pad_batches_to(
        _materialize_batches(train_dataloader, _cap_batches, **_cache_kw),
        warmup_pad_to, warmup_pad_token_id, warmup_padding_side,
    ) if warmup_pad_to else _materialize_batches(train_dataloader, _cap_batches, **_cache_kw)
    print(f"[sn56][farejando] cacheei {len(probe_batches)} lotes p/ provas "
          f"(S1@0, refino@{full_data_offset}); mesma data por etapa", flush=True)

    _eval_scored = (
        eval_dataloader is not None and trainer is not None and not use_reward
    )
    eval_fn: Optional[Callable[[], float]] = None
    if _eval_scored:
        holdout_batches = _materialize_batches(eval_dataloader, _DPO_HOLDOUT_BATCHES)
        if warmup_pad_to:
            holdout_batches = _pad_batches_to(holdout_batches, warmup_pad_to,
                                              warmup_pad_token_id, warmup_padding_side)
        if not holdout_batches:
            print("[sn56][farejando] sem lotes de validação — caio no loss de treino", flush=True)
            _eval_scored = False
        else:
            def eval_fn() -> float:
                
                was_training = model.training
                model.eval()
                total, n = 0.0, 0
                try:
                    with torch.no_grad():
                        for b in holdout_batches:
                            prepared = trainer._prepare_inputs(b)
                            loss = trainer.compute_loss(model, prepared)
                            if isinstance(loss, tuple):
                                loss = loss[0]
                            total += float(loss.item())
                            n += 1
                finally:
                    if was_training:
                        model.train()
                return total / n if n else float("inf")
            print(f"[sn56][farejando] score = DPO loss held-out em "
                  f"{len(holdout_batches)} lotes (igual ao validador); seleção=argmin",
                  flush=True)
            _init_score = eval_fn()
            print(f"[sn56][farejando] score inicial (sem treino) = {_init_score:.4f}", flush=True)

    s1_scores: dict[float, float] = {}
    full_scores: dict[float, float] = {}
    tested_logs: set[float] = set()
    best_full_log: Optional[float] = None
    best_full_loss = float("inf")
    best_s1_loss = float("inf")
    best_at_checkpoint: dict[int, float] = {}
    _bench_x: dict = {}

    oom_state = {"retries": 0, "loader": train_dataloader, "batches": probe_batches}
    step_timer = {"secs": 0.0, "steps": 0}

    def _make_prune_fn(log_lr: float, rel_check: bool = True) -> Callable[[int, float, float], Optional[str]]:
        trial_min = {"v": float("inf")}
        def prune_fn(opt_step: int, rolling: float, first: float) -> Optional[str]:
            if math.isfinite(rolling):
                trial_min["v"] = min(trial_min["v"], rolling)
            if opt_step < _prune_min_step:
                return None
            base = trial_min["v"] if math.isfinite(trial_min["v"]) else first
            if not math.isfinite(rolling) or rolling > base * _PRUNE_DIVERGE_FACTOR:
                return f"diverge(roll={rolling:.3f} vs min={base:.3f})"
            if opt_step == _prune_rel_step:
                prev = best_at_checkpoint.get(_prune_rel_step)
                bench_x = _bench_x.get("x")
                if (rel_check and prev is not None and rolling > prev * (1 + _PRUNE_REL_MARGIN)
                        and (bench_x is None or log_lr > bench_x + 1e-9)):
                    return f"rel(step{_prune_rel_step} roll={rolling:.3f} vs best={prev:.3f})"
                if prev is None or rolling < prev:
                    best_at_checkpoint[_prune_rel_step] = rolling
                    _bench_x["x"] = log_lr
            return None
        return prune_fn


    def _already_tested(log_lr: float) -> bool:
        return any(abs(log_lr - t) < _DEDUP_TOL for t in tested_logs)

    refine_center = log_center
    s2_steps = _S2_MIN_STEPS
    probe_log: list = []
    fit_state = {"gen": 0, "ext_done": set(), "hot_rejected": set(), "hot_accepted": False,
                 "fits": {}}
    _cal_on = carry_lr_cal_enabled() and not CARRY_LR_CAL_SUPPRESSED
    _fix_on = alldiverge_fix_enabled()
    _c_on = _fix_on and not use_reward and not _eval_scored and trainer is None
    _init_tail: dict = {}
    _walk_state: dict = {"fired": False, "record": None}
    _coldedge = {"on": coldedge_fund_enabled(), "used": False}

    def _rows_per_step() -> Optional[int]:
        b = oom_state["batches"][0] if oom_state["batches"] else None
        r, _ = _batch_rows_tokens(b)
        return int(r * max(1, int(grad_accum_steps or 1)) * _gns_world) if r else prune_rows_per_step

    def _on_batch_halved() -> None:
        
        nonlocal log_center, refine_center, t_per_step, best_full_loss
        fit_state["gen"] += 1
        _coldedge["rung"] = None
        fit_state["ext_done"] = set()
        fit_state["hot_rejected"] = set()
        fit_state["hot_accepted"] = False
        fit_state["fits"] = {}
        if best_full_log is not None and full_scores:
            best_known = best_full_log
        elif s1_scores:
            best_known = min(s1_scores, key=s1_scores.get)
        else:
            best_known = log_center
        shift = math.log10(math.sqrt(0.5))
        log_center = best_known + shift
        refine_center = log_center
        s1_scores.clear()
        full_scores.clear()
        best_at_checkpoint.clear()
        _bench_x.clear()
        tested_logs.clear()
        best_full_loss = float("inf")
        t_per_step *= 0.5
        step_timer["secs"] = 0.0
        step_timer["steps"] = 0
        print(
            f"[sn56][farejando] OOM→lote/2: recentro lr={10**log_center:.2e} "
            f"(shift {shift:+.3f}), placar limpo, retomando refino",
            flush=True,
        )

    def _s1_near(log_lr: float) -> Optional[float]:
        for k, v in s1_scores.items():
            if abs(log_lr - k) < _DEDUP_TOL:
                return v
        return None

    def _init_reference(offset: int, opt_steps: int, n_scored: int) -> tuple:
        nonlocal search_start
        key = (fit_state["gen"], offset, opt_steps, n_scored)
        if key in _init_tail:
            return _init_tail[key]
        _tail_l: dict = {}
        _restore_trainable_state(model, initial_state)
        if _probe_noise is not None:
            _probe_noise.reset()
        t0 = time.perf_counter()
        try:
            v = _init_tail_score(model, oom_state["batches"], opt_steps, n_scored, grad_accum_steps,
                                 data_offset=offset, warmup_steps=_PROBE_WARMUP,
                                 loss_fn=probe_loss_fn, chunker=chunker,
                                 score_rows=_rows_per_step(), out_tail=_tail_l)
            if sync_loss_fn is not None:
                v = sync_loss_fn(v)
        except Exception as e:
            if isinstance(e, RuntimeError) and _is_oom_error(e):
                _clear_memory()
            else:
                pass
            v = None
        dt = time.perf_counter() - t0
        search_start += dt
        _init_tail[key] = (v, _tail_l)
        if v is not None:
            print(f"[sn56][farejando] score sem treino na cauda ({opt_steps}st, offset {offset}) "
                  f"= {v:.4f}", flush=True)
        return v, _tail_l

    def test_lr(log_lr: float, opt_steps: int, tier: str,
                extension: Optional[str] = None) -> Optional[float]:
        nonlocal search_start

        if tier != "s1" and _already_tested(log_lr):
            return None
        if tier == "s1" and _s1_near(log_lr) is not None:
            return _s1_near(log_lr)
        lr = 10 ** log_lr
        _prune_safe = not (use_reward or trainer is not None or sync_loss_fn is not None)
        prune_fn = _make_prune_fn(log_lr, rel_check=(extension != "cold")) if _prune_safe else None

        _restore_trainable_state(model, initial_state)
        _offset = 0 if tier == "s1" else full_data_offset
        _n_cached = len(oom_state["batches"])
        opt_steps, _ = _probe_cache_policy(
            opt_steps, grad_accum_steps, _n_cached, s1_steps)
        if _probe_noise is not None:
            _probe_noise.reset()
        _call_t0 = time.perf_counter()
        _step_losses = []
        _gns_acc = None
        if tier == "s1" and _gns_rungs > 0 and _gns_state["s1_runs"] < _gns_rungs:
            _gns_acc = GradNoiseAccumulator(_gns_world)
        try:
            loss, pruned = _run_trial(
                model, oom_state["batches"], lr, opt_steps,
                optimizer_cls, optimizer_kwargs, grad_accum_steps,
                max_grad_norm, trainer=trainer, use_reward=use_reward,
                prune_fn=prune_fn, data_offset=_offset,
                warmup_steps=_PROBE_WARMUP, eval_fn=eval_fn,
                out_step_losses=_step_losses,
                loss_fn=(probe_loss_fn if (trainer is None and not use_reward) else None),
                gns=_gns_acc, prune_window=_prune_window,
                score_rows=_rows_per_step(),
                **({"chunker": chunker} if chunker is not None else {}),
            )
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            _clear_memory()
            oom_state["retries"] += 1
            if oom_state["retries"] > _MAX_OOM_RETRIES or dataloader_factory is None:
                print(f"[sn56][farejando] OOM após {oom_state['retries']} tentativas, encerrando busca", flush=True)
                raise _SearchExhausted from e
            print(f"[sn56][farejando] OOM em lr={lr:.2e}, lote/2 "
                  f"(tentativa {oom_state['retries']}/{_MAX_OOM_RETRIES})", flush=True)
            _restore_trainable_state(model, initial_state)
            oom_state["loader"] = dataloader_factory()
            oom_state["batches"] = _materialize_batches(
                oom_state["loader"], len(oom_state["batches"]))
            if warmup_pad_to:
                oom_state["batches"] = _pad_batches_to(
                    oom_state["batches"], warmup_pad_to, warmup_pad_token_id,
                    warmup_padding_side)
            _on_batch_halved()
            raise _BatchChanged from e

        _call_dt = time.perf_counter() - _call_t0
        _rows_now = _rows_per_step()
        if _gns_acc is not None:
            _gns_state["s1_runs"] += 1
            _gns_res = _gns_emit("s1", _gns_acc, lr, pruned=pruned is not None)
            if _gns_res is not None:
                _gns_state["last"] = _gns_res
                LAST_GNS["s1"] = _gns_res
        if sync_loss_fn is not None:
            loss = sync_loss_fn(loss)

        _init_ref: Optional[float] = None
        _init_se: Optional[float] = None
        _above_init: Optional[float] = None
        if _c_on and pruned is None and _step_losses and math.isfinite(loss):
            _init_ref, _init_tl = _init_reference(_offset, opt_steps, len(_step_losses))
            if _init_ref is not None and math.isfinite(_init_ref):
                _init_se = _paired_tail_se(_step_losses, _init_tl)
                if _init_se is not None and sync_loss_fn is not None:
                    _init_se = sync_loss_fn(_init_se)
            if (_init_se is not None and math.isfinite(_init_se)
                    and loss - _init_ref > _ABOVE_INIT_Z * _init_se):
                step_timer["secs"] += _call_dt
                step_timer["steps"] += opt_steps
                _above_init = loss
                pruned = (f"{_ABOVE_INIT_REASON}(score={loss:.3f} vs init={_init_ref:.3f}, "
                          f"se={_init_se:.4f}, z={(loss - _init_ref) / _init_se if _init_se > 0 else float('inf'):.1f})")
        _c_fields = ({"init_score": _init_ref, "init_se": _init_se, "init_z": _ABOVE_INIT_Z}
                     if _init_ref is not None else {})
        if _above_init is not None:
            _c_fields["score_above_init"] = _above_init

        if tier != "s1":
            tested_logs.add(log_lr)
        elapsed = time.perf_counter() - search_start
        metric = f"reward={-loss:.4e}" if use_reward else f"loss={loss:.4e}"

        probe_log.append({"tier": tier, "log_lr": log_lr, "lr": lr, "rows": _rows_now,
                          "score": (None if pruned is not None else loss),
                          "n": len(_step_losses), "pruned": pruned is not None,
                          "reason": pruned, "extension": extension, "gen": fit_state["gen"]})
        if pruned is not None:
            print(f"[sn56][farejando] {tier.upper()} [{elapsed:5.0f}s] "
                  f"lr={lr:.2e} {metric} PRUNED {pruned}", flush=True)
            return None

        step_timer["secs"] += _call_dt
        step_timer["steps"] += opt_steps

        nonlocal best_full_log, best_full_loss, best_s1_loss
        (s1_scores if tier == "s1" else full_scores)[log_lr] = loss
        marker = ""
        if tier == "s1":
            if loss < best_s1_loss:
                best_s1_loss = loss
                marker = " *** BEST"
        elif loss < best_full_loss:
            best_full_loss = loss
            best_full_log = log_lr
            marker = " *** BEST"
        print(f"[sn56][farejando] {tier.upper()} [{elapsed:5.0f}s] "
              f"lr={lr:.2e} {metric} ({opt_steps}st){marker}", flush=True)
        return loss

    def _time_left() -> float:
        return budget_secs - (time.perf_counter() - search_start)

    def _walk_eval_edges() -> None:
        _, stop = 0, "max_extensions"
        for _ in range(_MAX_EDGE_EXTENSIONS):
            fin = {k: v for k, v in s1_scores.items() if v is not None and math.isfinite(v)}
            if not fin:
                stop = "no_scored_rung"
                break
            xs = sorted(fin)
            best_log = min(fin, key=fin.get)
            no_gain = (_init_score is not None and math.isfinite(_init_score)
                       and fin[best_log] >= _init_score)
            if no_gain or best_log <= xs[0] + 1e-9:
                direction, frm = "cold", xs[0]
                nxt = frm - _S1_HALF_RANGE
                why = "no_gain_over_init" if no_gain else "cold_edge"
                between = lambda x: nxt - _DEDUP_TOL <= x < frm
            elif best_log >= xs[-1] - 1e-9:
                direction, frm = "hot", xs[-1]
                nxt = frm + _S1_HALF_RANGE
                why = "hot_edge"
                between = lambda x: frm < x <= nxt + _DEDUP_TOL
            else:
                stop = "interior"
                break
            if _time_left() <= (s1_steps + _DPO_HOLDOUT_BATCHES) * t_per_step:
                stop = "no_time"
                break
            _pruned_s1 = [p["log_lr"] for p in probe_log if p["tier"] == "s1" and p["pruned"]
                          and p["gen"] == fit_state["gen"]]
            if any(between(x) for x in _pruned_s1):
                stop = f"pruned_rung_{direction}"
                break
            if any(abs(nxt - k) < _DEDUP_TOL for k in s1_scores):
                stop = "already_probed"
                break
            print(f"[sn56][farejando] caminhada {direction} ({why}): provo {10**nxt:.2e} "
                  f"@ {s1_steps}st (melhor {10**best_log:.2e} = {fin[best_log]:.4f})", flush=True)
            r = test_lr(nxt, s1_steps, "s1", extension=f"walk_{direction}")
            wall = bool(direction == "hot" and r is not None and math.isfinite(r)
                        and r > fin[best_log] * (1 + _EXT_HOT_TOL))
            if wall:
                fit_state["hot_rejected"].add(nxt)
            if r is None or not math.isfinite(r):
                stop = "pruned"
                break
        _fin = {k: v for k, v in s1_scores.items() if v is not None and math.isfinite(v)}

    def _maybe_recentre() -> bool:
        nonlocal log_center, refine_center
        if recentre_fn is None or _gns_state["last"] is None:
            return False
        last = _gns_state["last"]
        ok, why = gns_reliable(last, _gns_min_steps)
        if not ok:
            print(f"[sn56][farejando] recentro pela escala de ruído ignorado: estimativa não confiável "
                  f"({why})", flush=True)
            return False
        try:
            new = recentre_fn(log_center, last)
        except Exception:
            return False
        if new is None or not math.isfinite(new):
            return False
        shift = float(new) - log_center
        applied = abs(shift) >= recentre_min_dex
        if not applied:
            return False
        print(f"[sn56][farejando] recentro pela escala de ruído: lr {10**log_center:.2e} -> "
              f"{10**float(new):.2e} ({shift:+.3f} dex)", flush=True)
        log_center = float(new)
        refine_center = log_center
        return True

    def _walk_all_diverged() -> None:
        if s1_scores:
            return
        probed = [p["log_lr"] for p in probe_log
                  if p["tier"] == "s1" and p["gen"] == fit_state["gen"]]
        if not probed:
            return
        anchor = min(probed)
        cost = (s1_steps + (_DPO_HOLDOUT_BATCHES if _eval_scored else 0)) * t_per_step
        rec = {"anchor_log_lr": anchor, "anchor_lr": 10 ** anchor, "gen": fit_state["gen"],
               "planned": [{"log_lr": p["log_lr"], "reason": p["reason"]} for p in probe_log
                           if p["tier"] == "s1" and p["gen"] == fit_state["gen"]],
               "rungs": [], "completed_log_lr": None, "bracket_log_lr": None,
               "step_dex": _S1_HALF_RANGE, "max_walk": _MAX_EDGE_EXTENSIONS, "s1_steps": s1_steps}
        _walk_state["fired"] = True
        _walk_state["record"] = rec
        print(f"[sn56][farejando] todas as LRs do S1 podadas: desço de {10**anchor:.2e} "
              f"em passos de {_S1_HALF_RANGE} dec", flush=True)
        nxt = anchor - _S1_HALF_RANGE
        n_walk, stop = 0, "max_extensions"
        while True:
            bracket = rec["completed_log_lr"] is not None
            if not bracket and n_walk >= _MAX_EDGE_EXTENSIONS:
                stop = "max_extensions"
                break
            if _time_left() <= cost:
                stop = "no_time_bracket" if bracket else "no_time"
                break
            n_before = len(probe_log)
            r = test_lr(nxt, s1_steps, "s1",
                        extension=("all_diverged_bracket" if bracket else "all_diverged_walk"))
            p = probe_log[-1] if len(probe_log) > n_before else None
            done = r is not None and math.isfinite(r)
            rec["rungs"].append({"log_lr": nxt, "lr": 10 ** nxt, "bracket": bracket,
                                 "completed": done, "score": (r if done else None),
                                 "reason": (p["reason"] if p is not None else None),
                                 "n_steps": (p["n"] if p is not None else None)})
            if bracket:
                if done:
                    rec["bracket_log_lr"] = nxt
                stop = "bracketed" if done else "bracket_pruned"
                break
            n_walk += 1
            if done:
                rec["completed_log_lr"] = nxt
            nxt -= _S1_HALF_RANGE
        rec["stop"] = stop
        rec["n_walk"] = n_walk
        rec["time_left"] = round(_time_left(), 1)

    def _do_s1() -> None:
        if plan.mode == "validate":
            print(f"[sn56][farejando] validate-S1: 3 LRs @ {s1_steps}st, "
                  f"±{_S1_HALF_RANGE} dec (centro lr={10**log_center:.2e})", flush=True)
            test_lr(log_center, s1_steps, "s1")
            if _maybe_recentre():
                test_lr(log_center, s1_steps, "s1")
            for off in (-_S1_HALF_RANGE, _S1_HALF_RANGE):
                test_lr(log_center + off, s1_steps, "s1")
            if _fix_on and not s1_scores:
                _walk_all_diverged()
            if _eval_scored:
                _walk_eval_edges()
            return
        print(f"[sn56][farejando] S1: {_S1_N} LRs @ {s1_steps}st, "
              f"±{_S1_HALF_RANGE} dec (centro lr={10**log_center:.2e})", flush=True)
        ran = 0
        _s1_sc = {}
        _s1_sc[0.0] = test_lr(log_center, s1_steps, "s1"); ran += 1
        _s1_n = _S1_N
        if _maybe_recentre():
            _s1_n += 1
            _s1_sc[0.0] = test_lr(log_center, s1_steps, "s1"); ran += 1
        _s1_sc[-_S1_HALF_RANGE] = test_lr(log_center - _S1_HALF_RANGE, s1_steps, "s1"); ran += 1
        _c, _lo = _s1_sc.get(0.0), _s1_sc.get(-_S1_HALF_RANGE)
        _halfrung = (_c is not None and _lo is not None
                     and _lo < _c * (1 - _EDGE_TOLERANCE)
                     and _lo > _c * (1 - 2 * _EDGE_TOLERANCE))
        if _halfrung:
            print("[sn56][farejando] borda baixa vence por 2.5-5% — meia-rung -0.15/+0.15", flush=True)
            highs = (-_S1_HALF_RANGE / 2, _S1_HALF_RANGE / 2)
            lows = ()
        else:
            highs = (_S1_HALF_RANGE / 2, _S1_HALF_RANGE)
            lows = (-_S1_HALF_RANGE / 2, -_S1_HALF_RANGE * 1.5)
        diverged = False
        for off in highs:
            if ran >= _s1_n:
                break
            r = test_lr(log_center + off, s1_steps, "s1"); ran += 1
            if r is None:
                diverged = True
                break
        if diverged:
            print("[sn56][farejando] LR alta divergiu — pivô pra baixo", flush=True)
            for off in lows:
                if ran >= _s1_n:
                    break
                test_lr(log_center + off, s1_steps, "s1"); ran += 1
        if _fix_on and not s1_scores:
            _walk_all_diverged()
        if _eval_scored:
            _walk_eval_edges()

    def _tier_pool(tier: str) -> dict:
        return s1_scores if tier == "s1" else full_scores

    def _tier_steps(tier: str) -> list:
        return [p["n"] for p in probe_log
                if p["tier"] == tier and not p["pruned"] and p["gen"] == fit_state["gen"]]

    def _hi_all() -> Optional[float]:
        xs = [k for pool_ in (s1_scores, full_scores) for k, v in pool_.items()
              if v is not None and math.isfinite(v)
              and not any(abs(k - r) < _DEDUP_TOL for r in fit_state["hot_rejected"])]
        return max(xs) if xs else None

    def _is_rejected(k: float) -> bool:
        return any(abs(k - r) < _DEDUP_TOL for r in fit_state["hot_rejected"])

    def _fit_pool(tier: str) -> dict:
        return {k: v for k, v in _tier_pool(tier).items() if not _is_rejected(k)}

    def _fit_and_extend(tier: str, steps: int) -> None:
        pool = _fit_pool(tier)
        if len(pool) < 3 or _degenerate_pool(pool) is not None:
            return
        _fin = [k for k, v in pool.items() if v is not None and math.isfinite(v)]
        lo, hi = min(_fin), max(_fin)
        xv, a, flag = fit_local(pool)
        hi_all = _hi_all()
        _cal = _cal_on and tier == "s1" and not use_reward and not _eval_scored
        direction = (cold_cal_request(xv, flag, lo, hi, hi_all) if _cal
                     else extension_request(xv, lo, hi, hi_all))
        ext = None
        cal_walk: list = []
        if direction is not None and tier not in fit_state["ext_done"]:
            fit_state["ext_done"].add(tier)
            while True:
                ext = _extend_once(tier, steps, direction, xv, lo, hi_all,
                                   funded=(_cal and direction == "cold"))
                pool = _fit_pool(tier)
                xv, a, flag = fit_local(pool)
                _fin = [k for k, v in pool.items() if v is not None and math.isfinite(v)]
                lo, hi = min(_fin), max(_fin)
                if not (_cal and direction == "cold"):
                    break
                ext.update(x_v_after=xv, flag_after=flag)
                cal_walk.append(ext)
                if ext["status"] != "refit":
                    break
                if len(cal_walk) >= _MAX_EDGE_EXTENSIONS:
                    ext["walk_stop"] = "max_extensions"
                    break
                hi_all = _hi_all()
                direction = cold_cal_request(xv, flag, lo, hi, hi_all)
                if direction != "cold":
                    break
        _fin_t = {k: v for k, v in pool.items() if v is not None and math.isfinite(v)}
        _ref_t = xv if xv is not None else (min(_fin_t, key=_fin_t.get) if _fin_t else -math.inf)
        _div_tier = [p["log_lr"] for p in probe_log if p["tier"] == tier and p["pruned"]
                     and p["gen"] == fit_state["gen"] and is_diverged_reason(p["reason"])
                     and p["log_lr"] > _ref_t]
        x_tier, tflag = fit_pick(pool, diverged=_div_tier)
        _, _ = fit_pick(pool, diverged=_div_tier, floor=None)
        fit_state["fits"][tier] = {"x_v": xv, "a": a, "flag": flag, "x_ship_uncapped": x_tier}
        print(f"[sn56][farejando] ajuste {tier}: vértice "
              f"{('%.2e' % 10**_clamp_vertex(xv, lo, hi)) if xv is not None else 'n/a'} ({flag}) "
              f"-> lr {10**x_tier:.2e} (antes do teto)", flush=True)

    def _coldedge_fund(ext: dict, tier: str, direction: str, need: float,
                       rem: Optional[float]) -> bool:
        if not _coldedge["on"] or direction != "cold" or _eval_scored or use_reward:
            return False
        if _coldedge["used"]:
            d = {"fired": False, "reason": "once_per_search", "probe_log_lr": ext["log_lr"],
                 "cost_secs": round(need, 1)}
        else:
            d = coldedge_fund_decision(_fit_pool(tier), ext["log_lr"], need, rem)
        d.update(tier=tier, time_left=round(_time_left(), 1))
        if d["fired"]:
            _coldedge["used"] = True
            d["source"] = "search_budget" if _time_left() >= need else "a0_budget"
            ext["funded"] = "coldedge_" + d["source"]
            ext["coldedge_fund"] = d
        else:
            ext["coldedge_refused"] = d
        print(f"[sn56][farejando] borda fria ({tier}): ganho previsto {d.get('gain_pred')} vs "
              f"custo {d.get('cost_frac')} -> {'financiada' if d['fired'] else 'recusada'} "
              f"({d['reason']})", flush=True)
        return d["fired"]

    def _measured_floor(x: float, caps: dict, fit_diag: dict) -> float:
        if not _coldedge["on"]:
            return x
        scored = [k for pool_ in (s1_scores, full_scores) for k, v in pool_.items()
                  if v is not None and math.isfinite(v)]
        x2, d = measured_floor(x, scored, _coldedge.get("rung"), caps)
        if d is None:
            return x
        fit_diag.update(measured_floor=d["floor"], measured_floor_why=d["why"],
                        x_below_floor=x, x_ship=x2)
        print(f"[sn56][farejando] piso medido ({d['why']}): lr {10**x:.2e} -> {10**x2:.2e}",
              flush=True)
        return x2

    def _extend_once(tier: str, steps: int, direction: str, xv: Optional[float], lo: float,
                     hi_all: Optional[float], funded: bool = False) -> dict:
        x_new = lo - _EXT_COLD_STEP if direction == "cold" else hi_all + _EXT_HOT_STEP
        need = steps * t_per_step
        try:
            _rem = remaining_secs_fn() if remaining_secs_fn is not None else None
        except Exception:
            _rem = None
        free, free_diag = extension_is_free(_rem, need, t_per_step, steps_per_epoch,
                                            max_epochs_cap)
        ext = {"direction": direction, "log_lr": x_new, "lr": 10 ** x_new, "steps": steps,
               "x_v_before": xv, "time_left": round(_time_left(), 1), "cost_est": round(need, 1),
               "free": free, "free_diag": free_diag}
        if funded:
            ext["funded"] = "free" if free else "search_budget"
        _pruned_here = [p["log_lr"] for p in probe_log
                        if p["pruned"] and p["gen"] == fit_state["gen"]]
        _tested = list(_tier_pool(tier)) + _pruned_here + list(fit_state["hot_rejected"])
        if any(abs(x_new - k) < _DEDUP_TOL for k in _tested):
            ext["status"] = "already_probed"
            if direction == "hot":
                fit_state["hot_rejected"].add(x_new)
        elif direction == "hot" and any(
                p["log_lr"] > hi_all and p["log_lr"] - _FIT_DIVERGE_STEP <= hi_all + 1e-9
                for p in probe_log if p["pruned"] and p["gen"] == fit_state["gen"]
                and is_diverged_reason(p["reason"])):
            ext["status"] = "diverged_above"
        elif not free and not funded and not _coldedge_fund(ext, tier, direction, need, _rem):
            ext["status"] = "not_free"
        elif _time_left() < need and "coldedge_fund" not in ext:
            ext["status"] = "no_time"
        else:
            _why = ("grátis: run limitado por épocas" if free
                    else "pago do orçamento da busca")
            print(f"[sn56][farejando] extensão {direction} ({tier}): provo {10**x_new:.2e} "
                  f"@ {steps}st ({_why})", flush=True)
            r = test_lr(x_new, steps, tier, extension=direction)
            ext["score"] = r
            if "coldedge_fund" in ext and r is not None and math.isfinite(r):
                _coldedge["rung"] = x_new
            if direction == "hot":
                others = [v for k, v in _tier_pool(tier).items()
                          if abs(k - x_new) >= _DEDUP_TOL and v is not None and math.isfinite(v)]
                best = min(others) if others else None
                if r is None or not math.isfinite(r) or (best is not None and r > best * (1 + _EXT_HOT_TOL)):
                    fit_state["hot_rejected"].add(x_new)
                    ext["status"] = "hot_rejected_cap_hi"
                else:
                    fit_state["hot_accepted"] = True
                    ext["status"] = "refit"
            else:
                ext["status"] = "refit" if r is not None else "pruned"
        if ext["status"] != "refit" and direction == "hot":
            print(f"[sn56][farejando] extensão quente ({tier}) {ext['status']}: lr limitado à "
                  f"rung mais quente provada {10**hi_all:.2e}", flush=True)
        return ext

    def _do_s2() -> bool:
        nonlocal s2_steps
        _top_before = _hi_all()
        ran = _do_s2_rungs()
        if ran and _top_before is not None:
            _fin = {k: v for k, v in full_scores.items() if v is not None and math.isfinite(v)}
            if _fin:
                _m = min(_fin.values())
                for k, v in _fin.items():
                    if k > _top_before + 1e-9 and v > _m * (1 + _EXT_HOT_TOL):
                        fit_state["hot_rejected"].add(k)
        return ran

    def _do_s2_rungs() -> bool:
        nonlocal s2_steps
        floor = max(s1_steps, _S2_MIN_STEPS)
        if _eval_scored:
            floor = min(s1_steps, _S2_MIN_STEPS)
        _tl = _time_left()
        _fin_s1 = [k for k, v in s1_scores.items() if v is not None and math.isfinite(v)]
        _side = (compute_affordance.nearer_edge_side(refine_center, min(_fin_s1), max(_fin_s1))
                 if _fin_s1 else -1)
        _hz = compute_affordance.examples_horizon(_ph.get("n_rows"), _ph_rows_per_step, floor,
                                                  _S2_MAX_STEPS)
        _s2p = compute_affordance.plan_s2(
            _tl, t_per_step, floor, _S2_MAX_STEPS, _S2_CLOCK_SHARE, _hz, plan.epochs_affordable,
            min_rungs=(compute_affordance.ARGMIN_MIN_RUNGS if (_eval_scored or use_reward)
                       else compute_affordance.FIT_MIN_RUNGS),
            side=_side, base_rungs=len(_S2_FIT_OFFSETS))
        _aff = compute_affordance.search_affordance(plan.epochs_affordable, t_per_step,
                                                    plan.budget_secs, _s2p)
        s2_steps = _s2p.steps_clock if _s2p.mode == "skip" else _s2p.steps
        if _s2p.mode == "skip":
            if not _s2p.changed:
                print(f"[sn56][farejando] S2 caberia só @ {s2_steps}st < S1 {floor}st — fica S1",
                      flush=True)
            else:
                print(f"[sn56][farejando] S2 fora ({_s2p.reason}) — fica S1", flush=True)
            return False
        print(f"[sn56][farejando] S2: vértice S1 + {tuple(_s2p.offsets)} @ {s2_steps}st "
              f"(centro lr={10**refine_center:.2e})", flush=True)
        for off in _s2p.offsets:
            test_lr(refine_center + off, s2_steps, "full")
        return True

    stage = 1
    while True:
        try:
            if stage <= 1:
                _do_s1()
                if not s1_scores:
                    break
                if step_timer["steps"] >= 10:
                    t_per_step = step_timer["secs"] / step_timer["steps"]
                if not use_reward and not _eval_scored:
                    _fit_and_extend("s1", s1_steps)
                if _eval_scored:
                    _fit_and_extend("s1", s1_steps)
                _s1_fit = fit_state["fits"].get("s1") or {}
                if _s1_fit.get("x_v") is not None:
                    refine_center = _clamp_vertex(_s1_fit["x_v"], min(s1_scores), max(s1_scores))
                    _top = _hi_all()
                    if _top is not None:
                        refine_center = min(refine_center, _top)
                else:
                    refine_center = min(s1_scores, key=s1_scores.get)
                stage = 2
                if step_timer["steps"] >= 10:
                    t_per_step = step_timer["secs"] / step_timer["steps"]
            if plan.mode in ("two_stage", "full") and s1_steps >= _S1_REFINE_GATE:
                if stage <= 2:
                    ran_s2 = _do_s2()
                    stage = 3
                    if ran_s2 and full_scores and not use_reward and not _eval_scored:
                        _fit_and_extend("full", s2_steps)
                    if ran_s2 and full_scores and _eval_scored:
                        _fit_and_extend("full", s2_steps)
            elif plan.mode in ("two_stage", "full"):
                print(f"[sn56][farejando] S1 raso ({s1_steps}st < porta "
                      f"{_S1_REFINE_GATE}st): sem refino, fica o melhor do S1", flush=True)
            break
        except _BatchChanged:
            stage = 1 if stage == 1 else 2
            continue
        except _SearchExhausted:
            break

    pool = full_scores if full_scores else s1_scores
    degenerate: Optional[str] = None
    edge_pick: Optional[float] = None
    fit_diag: dict = {}
    rule = "edge"
    _probed_gen = [p["log_lr"] for p in probe_log if p["gen"] == fit_state["gen"]]
    _fallback_cold = bool(_fix_on and not pool and _probed_gen)
    if not pool and _fallback_cold:
        best_lr, best_loss = 10 ** (min(_probed_gen) - _FIT_DIVERGE_STEP), float("inf")
        print(f"[sn56][farejando] Sem resultados: rung mais fria provada "
              f"{10**min(_probed_gen):.2e} - {_FIT_DIVERGE_STEP} dec -> lr={best_lr:.2e}", flush=True)
    elif not pool:
        best_lr, best_loss = 10 ** log_center, float("inf")
        print(f"[sn56][farejando] Sem resultados, estimativa lr={best_lr:.2e}", flush=True)
    elif use_reward:
        best_log = min(pool, key=pool.get)
        best_lr, best_loss = 10 ** best_log, pool[best_log]
    elif _eval_scored:
        best_log = min(pool, key=pool.get)
        best_lr, best_loss = 10 ** best_log, pool[best_log]
    else:
        degenerate = _degenerate_pool(pool)
        edge_pick = select_edge_lr(pool)[0]
        if degenerate is not None:
            best_lr, best_loss = 10 ** log_center, None
            print(f"[sn56][farejando] pool degenerado ({degenerate}): sem sinal de LR, "
                  f"fico com o prior lr={best_lr:.2e}", flush=True)
        else:
            tiers = [TierFit("s1", _fit_pool("s1"), _tier_steps("s1"))]
            if full_scores:
                tiers.append(TierFit("full", _fit_pool("full"), _tier_steps("full")))
            x_ship, rule, fit_diag = combine_tiers(tiers)
            _pruned = [(p["log_lr"], p["reason"]) for p in probe_log
                       if p["pruned"] and p["gen"] == fit_state["gen"]]
            _dp = tiers[-1].pool if tiers[-1].pool else tiers[0].pool
            _fin = {k: v for k, v in _dp.items() if v is not None and math.isfinite(v)}
            _refs = [r for r in (fit_diag.get("x_v_combined"),
                                 (min(_fin, key=_fin.get) if _fin else None)) if r is not None]
            _x_ref = max(_refs) if _refs else None
            x_capped, caps = final_caps(x_ship, rule, _x_ref, [s1_scores, full_scores], _pruned,
                                        fit_state["hot_rejected"], fit_state["hot_accepted"])
            fit_diag.update(x_uncapped=x_ship, caps=caps, x_ship=x_capped)
            x_capped = _measured_floor(x_capped, caps, fit_diag)
            try:
                _xb, _rb, _db = combine_tiers(tiers, offset=_B_FIT_OFFSET, floor=_B_FIT_FLOOR)
                _xbc, _ = final_caps(_xb, _rb, _x_ref, [s1_scores, full_scores], _pruned,
                                     fit_state["hot_rejected"], fit_state["hot_accepted"])
                fit_diag.update(lr_b_rule=10 ** _xbc, x_b_rule=_xbc,
                                b_rule_minus_pick_dex=_xbc - x_capped)
            except Exception:
                pass
            best_lr = 10 ** x_capped
            best_loss = min(v for v in pool.values() if v is not None and math.isfinite(v))
            _s1_base = {p["log_lr"]: p["score"] for p in probe_log
                        if p["tier"] == "s1" and not p["pruned"] and not p["extension"]
                        and p["gen"] == fit_state["gen"] and p["score"] is not None
                        and math.isfinite(p["score"])}
            if _s1_base:
                _s1e = _select_edge_log(_s1_base)[0]
                fit_diag.update(s1_edge_pick=10 ** _s1e, pick_minus_s1_edge_dex=x_capped - _s1e,
                                colder_than_s1_edge_gt_010=bool(full_scores) and (x_capped - _s1e) < -0.10)

    _argmin_log = min(pool, key=pool.get) if pool else None
    _walk_rec = _walk_state["record"]
    _walk_now = bool(_fix_on and _walk_rec is not None and _walk_rec["gen"] == fit_state["gen"])
    if _walk_now and pool and _argmin_log is not None:
        _hot_pruned = [p["log_lr"] for p in probe_log if p["pruned"] and p["gen"] == fit_state["gen"]
                       and p["log_lr"] > _argmin_log + 1e-9]
        if _hot_pruned:
            _cap = min(_hot_pruned) - _FIT_DIVERGE_STEP
            if math.log10(best_lr) > _cap:
                best_lr = 10 ** _cap
    global LAST_SELECTION
    LAST_SELECTION = {
        "best_lr": best_lr,
        "argmin_lr": (best_lr if degenerate is not None else
                      (10 ** _argmin_log) if _argmin_log is not None else None),
        "argmin_loss": pool.get(_argmin_log) if _argmin_log is not None else None,
        "degenerate": degenerate,
        "pool_tier": "full" if full_scores else ("s1" if s1_scores else "none"),
        "n_pool": len(pool),
        "edge_pick": edge_pick,
        "fit": dict(fit_diag),
        "pool": {float(k): v for k, v in pool.items()},
        "probed": sorted({float(k) for k in list(s1_scores) + list(full_scores)}),
    }
    _rule_name = ("fallback_cold_rung" if _fallback_cold else
                  "fallback_center" if not pool else
                  "degenerate_pool" if degenerate is not None else
                  "argmin_reward" if use_reward else
                  "argmin_heldout" if _eval_scored else rule)
    print(f"[sn56][farejando] seleção {_rule_name}: lr={best_lr:.2e}"
          f"{(' (regra de borda daria %.2e)' % edge_pick) if edge_pick else ''}", flush=True)

    _restore_trainable_state(model, initial_state)
    del initial_state

    try:
        model.zero_grad(set_to_none=True)
    except Exception:
        pass
    _clear_memory()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(
            f"[sn56][farejando] limpeza pós-busca: "
            f"{torch.cuda.mem_get_info()[0]/1e9:.1f}GB livres / "
            f"{torch.cuda.memory_reserved()/1e9:.1f}GB reservados por torch",
            flush=True,
        )

    if step_timer["steps"] >= 10:
        measured_t = step_timer["secs"] / step_timer["steps"]
        print(f"[sn56][farejando] t/passo refinado {measured_t:.3f}s "
              f"(warmup era {t_per_step:.3f}s; {step_timer['steps']} passos cronometrados)",
              flush=True)
        t_per_step = measured_t
    elif t_per_step_hint and t_per_step_hint > 0:
        t_per_step = min(t_per_step, float(t_per_step_hint) * _T_HINT_SLACK)

    _restore_tqdm()
    elapsed = time.perf_counter() - search_start
    tier_name = "full" if full_scores else ("S1" if s1_scores else "estimate")
    print(
        f"[sn56][farejando] Done in {elapsed:.0f}s ({elapsed/60:.1f}min): "
        f"best_lr={best_lr:.2e} (loss={'n/a' if best_loss is None else f'{best_loss:.4e}'}, tier={tier_name}, "
        f"{len(s1_scores)} S1 + {len(full_scores)} full kept)",
        flush=True,
    )
    return best_lr, t_per_step
