import gc
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import torch
from transformers import Trainer

import precise_update
import release_defaults

try:
    from transformers.trainer_optimizer import is_optimizer_factory
except ImportError:
    def is_optimizer_factory(_cls) -> bool:
        return False

_FALSY = ("0", "false", "no", "off")

DEFAULT_MODE = "v3"
MODES = ("tokens", "throughput", "v3")
DEFAULT_CLOCK_MAX = 0.5
DEFAULT_CLOCK_EPOCHS = 2.5
DEFAULT_TRAIN_UTIL = 0.85
DEFAULT_ACCUM_RUNGS = (2, 4, 8)
DEFAULT_PROBE_EPOCH_FRAC = 0.25
DEFAULT_GNS_RUNGS = 1
DEFAULT_RECENTRE_MIN_DEX = 0.15
DEFAULT_GNS_MIN_STEPS = 25
DEFAULT_TOKENS = 65536
DEFAULT_MIN_MICRO_TOKENS = 8192
DEFAULT_MAX_ACCUM = 32
DEFAULT_HEADROOM = 0.88
DEFAULT_THROUGHPUT_FRAC = 0.90
DEFAULT_LR_SCALE_MIN = 0.25
OPTIMIZER_BYTES_PER_PARAM = 2.0
PROBE_MAX_TRIES = 6
LADDER_MAX_MICRO = 256
LADDER_MAX_TOKENS = 524288
LADDER_PLATEAU_GAIN = 0.05
LADDER_REPEATS = 2
LADDER_WARMUP = 1
LADDER_ROW_SEED = 7
DEFAULT_LR_SCALE_SKIP_DIV = 4
RUNG_NOISE_SIG0 = 0.0051
RUNG_NOISE_TAU = 0.00135
DEFAULT_RUNG_BAND_Z = 1.0
DEFAULT_FLOOR_SECS = 0.150
FLOOR_ROWS = 4
DEFAULT_RETIME_MARGIN_SD = 2.0
DEFAULT_RETIME_REPEATS = 6


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "") or default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def mode() -> str:
    raw = (os.environ.get("SN56_BATCH_POLICY", DEFAULT_MODE) or DEFAULT_MODE).strip().lower()
    if raw in _FALSY:
        return "off"
    return raw if raw in MODES else "off"


def enabled() -> bool:
    return mode() in MODES


def throughput_frac() -> float:
    return min(1.0, max(0.5, _env_float("SN56_BATCH_THROUGHPUT_FRAC", DEFAULT_THROUGHPUT_FRAC)))


def rung_band_z() -> float:
    return min(3.0, max(0.0, _env_float("SN56_BATCH_RUNG_BAND_Z", DEFAULT_RUNG_BAND_Z)))


def floor_enabled() -> bool:
    return (os.environ.get("SN56_BATCH_FLOOR", "1") or "1").strip().lower() not in _FALSY


def floor_secs() -> float:
    return max(0.0, _env_float("SN56_BATCH_FLOOR_SECS", DEFAULT_FLOOR_SECS))


def fixed_cost_bound(t: dict, limit: Optional[float] = None) -> bool:
    secs = float(t.get("secs") or 0.0)
    return 0.0 < secs < (floor_secs() if limit is None else float(limit))


def rung_ref_mode() -> str:
    raw = (os.environ.get("SN56_BATCH_RUNG_REF", "ladder") or "ladder").strip().lower()
    return "cut" if raw == "cut" else "ladder"


def retime_enabled() -> bool:
    return (os.environ.get("SN56_BATCH_RETIME", "1") or "1").strip().lower() not in _FALSY


def retime_margin_sd() -> float:
    return max(0.0, _env_float("SN56_BATCH_RETIME_MARGIN_SD", DEFAULT_RETIME_MARGIN_SD))


def retime_repeats() -> int:
    return max(1, _env_int("SN56_BATCH_RETIME_REPEATS", DEFAULT_RETIME_REPEATS))


def lr_scale_min() -> float:
    return min(1.0, max(0.01, _env_float("SN56_BATCH_LR_SCALE_MIN", DEFAULT_LR_SCALE_MIN)))


def target_tokens() -> int:
    return max(1024, _env_int("SN56_BATCH_TOKENS", DEFAULT_TOKENS))


def min_micro_tokens() -> int:
    return max(0, _env_int("SN56_BATCH_MIN_MICRO_TOKENS", DEFAULT_MIN_MICRO_TOKENS))


def max_accum() -> int:
    return max(1, _env_int("SN56_BATCH_MAX_ACCUM", DEFAULT_MAX_ACCUM))


def probe_enabled() -> bool:
    return (os.environ.get("SN56_BATCH_PROBE", "1") or "1").strip().lower() not in _FALSY


def headroom() -> float:
    return min(0.98, max(0.5, _env_float("SN56_BATCH_HEADROOM", DEFAULT_HEADROOM)))


def lr_scale_rule() -> str:
    raw = (os.environ.get("SN56_BATCH_LR_SCALE", "sqrt") or "sqrt").strip().lower()
    return "none" if raw in _FALSY or raw == "none" else "sqrt"


def lr_scale_skip_div() -> float:
    return max(0.0, _env_float("SN56_BATCH_LR_SCALE_SKIP_DIV", DEFAULT_LR_SCALE_SKIP_DIV))


def clock_epochs(configured: float) -> float:
    v = _env_float("SN56_BATCH_CLOCK_EPOCHS", DEFAULT_CLOCK_EPOCHS)
    return min(float(configured), v) if v > 0 else float(configured)


def clock_max() -> float:
    return max(0.05, _env_float("SN56_BATCH_CLOCK_MAX", DEFAULT_CLOCK_MAX))


def accum_rungs() -> tuple:
    raw = (os.environ.get("SN56_BATCH_ACCUM_RUNGS", "") or "").strip()
    if raw.lower() in _FALSY:
        return ()
    if not raw:
        return DEFAULT_ACCUM_RUNGS
    out = []
    for part in raw.split(","):
        try:
            k = int(part)
        except ValueError:
            continue
        if k > 1:
            out.append(k)
    return tuple(sorted(set(out)))


def gns_enabled() -> bool:
    return (os.environ.get("SN56_GNS", "1") or "1").strip().lower() not in _FALSY


def gns_rungs() -> int:
    return max(0, _env_int("SN56_GNS_RUNGS", DEFAULT_GNS_RUNGS))


def probe_epoch_frac() -> float:
    return max(0.0, min(2.0, _env_float("SN56_PROBE_EPOCH_FRAC", DEFAULT_PROBE_EPOCH_FRAC)))


def recentre_min_dex() -> float:
    return max(0.0, _env_float("SN56_BATCH_RECENTRE_MIN_DEX", DEFAULT_RECENTRE_MIN_DEX))


def noise_cap_enabled() -> bool:
    return (os.environ.get("SN56_BATCH_NOISE_CAP", "1") or "1").strip().lower() not in _FALSY


def gns_min_steps() -> int:
    return max(1, _env_int("SN56_GNS_MIN_STEPS", DEFAULT_GNS_MIN_STEPS))


def coverage_clock_enabled() -> bool:
    return release_defaults.raw("SN56_COVERAGE_CLOCK").strip().lower() not in _FALSY


def coverage_clock(t_per_step, hint: Optional[float], padded: bool, slack: float,
                   kl_overhead: float = 1.0) -> tuple:
    if not padded or hint is None or not hint or float(hint) <= 0:
        return t_per_step, "search"
    bound = float(hint) * float(kl_overhead) * float(slack)
    if bound < float(t_per_step):
        return bound, "ladder_hint"
    return t_per_step, "search"


@dataclass
class BatchPlan:
    rows: int
    world_size: int
    tokens_target: int
    row_tokens: float
    tokens_est: float
    n_rows: int
    steps_per_epoch: int
    floor_steps: int
    floor_bound: bool
    micro: int = 1
    accum: int = 1
    micro_tokens_floor: int = 0
    micro_floor_bound: bool = False


def plan_rows(
    n_rows: int,
    total_tokens: float,
    tokens_target: int,
    min_steps: int,
    world_size: int = 1,
    row_capacity: Optional[float] = None,
    micro_tokens_floor: int = 0,
) -> BatchPlan:
    n_rows = max(1, int(n_rows))
    world_size = max(1, int(world_size))
    row_tokens = max(1.0, float(total_tokens) / n_rows)
    rows = max(1, int(round(tokens_target / row_tokens)))
    floor_steps = max(1, int(min_steps or 0))
    floor_bound = False
    if n_rows // rows < floor_steps and rows > 1:
        rows = max(1, n_rows // floor_steps)
        floor_bound = True
    micro_floor_bound = False
    if micro_tokens_floor and row_capacity:
        min_rows = max(1, int(math.ceil(micro_tokens_floor / row_capacity))) * world_size
        if rows < min_rows and n_rows // min_rows >= 1:
            rows = min_rows
            micro_floor_bound = True
    rows = max(world_size, (rows // world_size) * world_size)
    steps_per_epoch = max(1, n_rows // rows)
    return BatchPlan(
        rows=rows, world_size=world_size, tokens_target=int(tokens_target),
        row_tokens=row_tokens, tokens_est=rows * row_tokens, n_rows=n_rows,
        steps_per_epoch=steps_per_epoch, floor_steps=floor_steps, floor_bound=floor_bound,
        micro=max(1, rows // world_size), accum=1,
        micro_tokens_floor=int(micro_tokens_floor), micro_floor_bound=micro_floor_bound,
    )


def split_rows(plan: BatchPlan, micro_max: int, accum_cap: Optional[int] = None) -> BatchPlan:
    cap = max(1, int(accum_cap if accum_cap is not None else max_accum()))
    per_rank = max(1, plan.rows // plan.world_size)
    micro = max(1, min(int(micro_max), per_rank))
    accum = max(1, int(math.ceil(per_rank / micro)))
    if accum > cap:
        accum = cap
    micro = max(1, int(math.ceil(per_rank / accum)))
    micro = min(micro, max(1, int(micro_max)))
    plan.micro = micro
    plan.accum = accum
    plan.rows = micro * accum * plan.world_size
    plan.tokens_est = plan.rows * plan.row_tokens
    plan.steps_per_epoch = max(1, plan.n_rows // plan.rows)
    return plan


def lr_scale(tokens_before: float, tokens_after: float, rule: Optional[str] = None,
             floor: Optional[float] = None) -> float:
    rule = rule or lr_scale_rule()
    if rule == "none" or not tokens_before or not tokens_after or tokens_before <= 0 or tokens_after <= 0:
        return 1.0
    lo = lr_scale_min() if floor is None else max(0.01, min(1.0, float(floor)))
    raw = math.sqrt(float(tokens_after) / float(tokens_before))
    return max(lo, min(1.0 / lo, raw))


def lr_scale_for_plan(table_rows: int, rows: int, tokens_table: float, tokens_est: float,
                      rule: Optional[str] = None, skip_div: Optional[float] = None) -> tuple[float, str]:
    rule = rule or lr_scale_rule()
    if rule == "none":
        return 1.0, "none"
    div = lr_scale_skip_div() if skip_div is None else max(0.0, float(skip_div))
    if div > 0 and table_rows > 0 and rows * div < table_rows:
        return 1.0, "below_table_div"
    scale = lr_scale(tokens_table, tokens_est, rule=rule)
    return scale, ("sqrt_floor" if scale <= lr_scale_min() + 1e-12 else "sqrt")


def plan_from_micro(micro: int, world_size: int, n_rows: int, row_tokens: float) -> BatchPlan:
    world_size = max(1, int(world_size))
    micro = max(1, int(micro))
    n_rows = max(1, int(n_rows))
    rows = micro * world_size
    return BatchPlan(
        rows=rows, world_size=world_size, tokens_target=0, row_tokens=float(row_tokens),
        tokens_est=rows * float(row_tokens), n_rows=n_rows,
        steps_per_epoch=max(1, n_rows // rows), floor_steps=0, floor_bound=False,
        micro=micro, accum=1,
    )


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cudnn_status_alloc_failed" in text


@dataclass
class ProbeResult:
    micro: int
    requested: int
    tries: list = field(default_factory=list)
    total_gb: Optional[float] = None
    reserve_gb: float = 0.0
    headroom: float = DEFAULT_HEADROOM
    secs: float = 0.0
    reason: str = ""


def _synthetic_batch(micro: int, seq_len: int, vocab: int, device, pad_id: int,
                     with_position_ids: bool, real_len: Optional[int] = None) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(LADDER_ROW_SEED)
    hi = max(3, int(vocab) - 1)
    ids = torch.randint(3, hi, (micro, seq_len), generator=gen)
    ids[:, 0] = max(0, min(pad_id, hi)) if pad_id is not None else 1
    mask = torch.ones(micro, seq_len, dtype=torch.long)
    labels = ids.clone()
    real = seq_len if real_len is None else max(1, min(int(seq_len), int(round(real_len))))
    if real < seq_len:
        pad = max(0, min(pad_id, hi)) if pad_id is not None else 0
        ids[:, real:] = pad
        mask[:, real:] = 0
        labels[:, real:] = -100
    batch = {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}
    if with_position_ids:
        batch["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0).expand(micro, -1).contiguous()
    return batch


def real_tokens(batch: dict) -> int:
    mask = batch.get("attention_mask") if isinstance(batch, dict) else None
    if mask is not None:
        return int(mask.sum().item())
    return int(batch["input_ids"].numel())


def dataset_rows(dataset, collator: Callable, device, max_rows: int = LADDER_MAX_MICRO,
                 seed: int = LADDER_ROW_SEED, lengths: Optional[list] = None,
                 grouped: Optional[str] = None, target_len: Optional[float] = None
                 ) -> Optional[Callable[[int], Optional[dict]]]:
    try:
        n = len(dataset)
    except TypeError:
        return None
    if n < 1 or collator is None:
        return None
    idx = random.Random(seed).sample(range(n), min(n, max(1, int(max_rows))))
    if grouped and lengths is not None and len(lengths) == n:
        idx = sorted(idx, key=lambda i: int(lengths[i]))
        sorted_len = [int(lengths[i]) for i in idx]
        cache: dict = {}

        def _rows_grouped(micro: int) -> Optional[dict]:
            if micro > len(idx):
                return None
            if grouped == "worst":
                lo, hi = len(idx) - micro, len(idx)
            else:
                import length_grouping as lg
                lo, hi = lg.grouped_window(sorted_len, micro, float(target_len or sorted_len[len(sorted_len) // 2]))
            rows = []
            for j in range(lo, hi):
                if j not in cache:
                    cache[j] = dataset[idx[j]]
                rows.append(cache[j])
            batch = collator(rows)
            return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        return _rows_grouped
    cache: list = []

    def _rows(micro: int) -> Optional[dict]:
        if micro > len(idx):
            return None
        while len(cache) < micro:
            cache.append(dataset[idx[len(cache)]])
        batch = collator(list(cache[:micro]))
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    return _rows


def probe_micro_batch(
    model,
    requested: int,
    seq_len: int,
    vocab: int,
    pad_id: int = 0,
    loss_fn: Optional[Callable] = None,
    with_position_ids: bool = False,
    max_tries: int = PROBE_MAX_TRIES,
    reserve_bytes: float = 0.0,
    headroom_frac: Optional[float] = None,
) -> ProbeResult:
    t0 = time.perf_counter()
    hr = headroom() if headroom_frac is None else headroom_frac
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    res = ProbeResult(micro=max(1, int(requested)), requested=max(1, int(requested)), headroom=hr,
                      reserve_gb=reserve_bytes / 1e9)
    if cuda:
        res.total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    budget = (res.total_gb * 1e9 * hr - reserve_bytes) if cuda else None
    was_training = model.training
    model.train()
    micro = res.micro
    lo_ok = 0
    for _ in range(max_tries):
        if micro < 1:
            break
        entry = {"micro": micro}
        try:
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            batch = _synthetic_batch(micro, seq_len, vocab, device, pad_id, with_position_ids)
            if loss_fn is not None:
                loss = loss_fn(model, batch)
            else:
                loss = model(**batch).loss
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device) if cuda else 0.0
            entry["peak_gb"] = round(peak / 1e9, 3)
            fits = (budget is None) or (peak <= budget)
            entry["fits"] = bool(fits)
            res.tries.append(entry)
            if fits:
                lo_ok = micro
                break
            scaled = int(micro * budget / max(peak, 1.0))
            micro = max(1, min(micro - 1, scaled))
        except Exception as exc:
            if not _is_oom(exc):
                res.reason = f"probe_error:{type(exc).__name__}"
                res.tries.append({**entry, "error": type(exc).__name__})
                break
            entry["oom"] = True
            res.tries.append(entry)
            micro = micro // 2
        finally:
            model.zero_grad(set_to_none=True)
            try:
                del batch, loss
            except NameError:
                pass
            if cuda:
                torch.cuda.empty_cache()
    if not was_training:
        model.eval()
    res.micro = max(1, lo_ok or micro or 1)
    if not res.reason:
        res.reason = "fits" if lo_ok else "fallback_min"
    res.secs = round(time.perf_counter() - t0, 2)
    return res


@dataclass
class ThroughputResult:
    micro: int
    frac: float
    tries: list = field(default_factory=list)
    best_tok_s: float = 0.0
    best_micro: int = 0
    total_gb: Optional[float] = None
    reserve_gb: float = 0.0
    headroom: float = DEFAULT_HEADROOM
    secs: float = 0.0
    reason: str = ""
    optimizer: str = "none"
    rows: str = "synthetic"
    state_gb: float = 0.0
    rung_read: dict = field(default_factory=dict)


def _rung_rows(t: dict) -> int:
    return int(t.get("micro", 1)) * int(t.get("accum", 1) or 1)


def _rung_ok(t: dict, allow_accum: bool) -> bool:
    return bool(t.get("fits")) and (t.get("tok_s") or 0) > 0 and (allow_accum or int(t.get("accum", 1) or 1) == 1)


def _rung_key(t: dict) -> tuple:
    return int(t["micro"]), int(t.get("accum", 1) or 1)


def choose_rung_raw(tries: list, frac: float, allow_accum: bool = False) -> tuple[int, int, float, int, int]:
    ok = [t for t in tries if _rung_ok(t, allow_accum)]
    if not ok:
        return 1, 1, 0.0, 0, 1
    best = max(ok, key=lambda t: t["tok_s"])
    for t in sorted(ok, key=_rung_rows):
        if t["tok_s"] >= frac * best["tok_s"]:
            return (int(t["micro"]), int(t.get("accum", 1) or 1), float(best["tok_s"]),
                    int(best["micro"]), int(best.get("accum", 1) or 1))
    return (int(best["micro"]), int(best.get("accum", 1) or 1), float(best["tok_s"]),
            int(best["micro"]), int(best.get("accum", 1) or 1))


def rung_sd(secs: float) -> float:
    return math.sqrt(RUNG_NOISE_SIG0 ** 2 + (RUNG_NOISE_TAU / max(float(secs), 1e-6)) ** 2)


def choose_rung_read(tries: list, frac: float, allow_accum: bool = False, ladder: Optional[list] = None,
                     z: Optional[float] = None) -> tuple[tuple[int, int, float, int, int], dict]:
    zz = rung_band_z() if z is None else max(0.0, float(z))
    raw = choose_rung_raw(tries, frac, allow_accum)
    cand = sorted([t for t in tries if _rung_ok(t, allow_accum)], key=_rung_rows)
    diag: dict = {"frac": frac, "z": zz, "allow_accum": bool(allow_accum), "ref_mode": rung_ref_mode(),
                  "raw_micro": raw[0], "raw_accum": raw[1], "raw_rows": raw[0] * raw[1]}
    if not cand:
        diag["reason"] = "no_rungs"
        return raw, diag
    diag["floor_secs"] = floor_secs() if floor_enabled() else None
    if diag["floor_secs"] is not None and any(fixed_cost_bound(t) for t in cand
                                              if _rung_rows(t) < FLOOR_ROWS):
        _lim = FLOOR_ROWS if any(_rung_rows(t) >= FLOOR_ROWS for t in cand) else 2
        _keep = [t for t in cand if not (_rung_rows(t) < _lim and fixed_cost_bound(t))]
        if len(_keep) < len(cand):
            diag["floor"] = "lifted" if _keep else "only_rows_1"
            diag["floor_rows"] = _lim
        cand = _keep or cand
    pool = [t for t in (ladder if (ladder is not None and diag["ref_mode"] == "ladder") else tries) if _rung_ok(t, allow_accum)]
    if not pool:
        pool = cand
    ref = max(pool, key=lambda t: float(t["tok_s"]))
    ref_tok = float(ref["tok_s"])
    ref_sd = rung_sd(ref["secs"]) if (ref.get("secs") or 0) > 0 else 0.0
    pick, ratio, sd, below = None, 1.0, 0.0, None
    for t in cand:
        r = float(t["tok_s"]) / ref_tok
        s = r * math.sqrt((rung_sd(t["secs"]) ** 2 if (t.get("secs") or 0) > 0 else 0.0) + (0.0 if t is ref else ref_sd ** 2))
        if r + zz * s >= frac:
            pick, ratio, sd = t, r, s
            break
        below = round(r, 4)
    if pick is None:
        pick = cand[-1]
        ratio = float(pick["tok_s"]) / ref_tok
        sd = ratio * math.sqrt((rung_sd(pick["secs"]) ** 2 if (pick.get("secs") or 0) > 0 else 0.0) + ref_sd ** 2)
        below = round(float(cand[-2]["tok_s"]) / ref_tok, 4) if len(cand) > 1 else None
        diag["reason"] = "cut_below_peak"
    else:
        diag["reason"] = "read"
    margin = (ratio - frac) / sd if sd > 0 else None
    diag.update({
        "ref_micro": int(ref["micro"]), "ref_accum": int(ref.get("accum", 1) or 1), "ref_rows": _rung_rows(ref),
        "ref_tok_s": round(ref_tok, 1), "ratio": round(ratio, 4), "ratio_sd": round(sd, 4),
        "margin_sd": (round(margin, 2) if margin is not None else None),
        "near_line": (abs(margin) < 2.0) if margin is not None else False, "ratio_below": below,
        "micro": int(pick["micro"]), "accum": int(pick.get("accum", 1) or 1), "rows": _rung_rows(pick),
    })
    diag["agrees_raw"] = (diag["micro"], diag["accum"]) == (raw[0], raw[1])
    return ((diag["micro"], diag["accum"], ref_tok, int(ref["micro"]), int(ref.get("accum", 1) or 1)), diag)


def deciding_rungs(tries: list, diag: dict, frac: float, margin_sd: Optional[float] = None,
                   allow_accum: bool = False) -> list:
    m = retime_margin_sd() if margin_sd is None else float(margin_sd)
    zz = float(diag.get("z", rung_band_z()))
    cand = sorted([t for t in tries if _rung_ok(t, allow_accum)], key=_rung_rows)
    if not cand or "ref_tok_s" not in diag:
        return []
    ref_tok = float(diag["ref_tok_s"])
    ref = next((t for t in cand if _rung_key(t) == (diag.get("ref_micro"), diag.get("ref_accum", 1))), None)
    ref_sd = rung_sd(ref["secs"]) if ref is not None and (ref.get("secs") or 0) > 0 else 0.0
    pick_key = (diag.get("micro"), diag.get("accum", 1))
    keys = [_rung_key(t) for t in cand]
    if pick_key not in keys:
        return []
    i = keys.index(pick_key)
    near = []
    for t in cand[max(0, i - 1):i + 1]:
        r = float(t["tok_s"]) / ref_tok
        sd = r * math.sqrt((rung_sd(t["secs"]) ** 2 if (t.get("secs") or 0) > 0 else 0.0)
                           + (0.0 if t is ref else ref_sd ** 2))
        if sd > 0 and abs((r + zz * sd - frac) / sd) < m and t is not ref:
            near.append(_rung_key(t))
    if not near:
        return []
    out = sorted(set(near) | ({_rung_key(ref)} if ref is not None else set()), key=lambda k: k[0] * k[1])
    return out


def _fwd_bwd(model, batch: dict, loss_fn: Optional[Callable], chunker=None) -> None:
    if chunker is not None:
        chunker.fwd_bwd(model, batch, loss_fn)
        return
    loss = loss_fn(model, batch) if loss_fn is not None else model(**batch).loss
    loss.backward()
    del loss


def _opt_step(model, optimizer, scheduler, max_grad_norm: float) -> None:
    if optimizer is not None:
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    model.zero_grad(set_to_none=True)


def _state_bytes(optimizer) -> float:
    if optimizer is None:
        return 0.0
    total = 0
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return float(total)


def build_probe_optimizer(model, training_args):
    if training_args is None:
        return None, None, "no_args"
    try:
        cls, kwargs, _plan = precise_update.optimizer_for(training_args, model, phase="probe")
    except Exception as exc:
        return None, None, f"cls:{type(exc).__name__}"
    kwargs = dict(kwargs)
    if is_optimizer_factory(cls) or any(k in kwargs for k in ("params", "model", "optimizer_dict")):
        return None, None, f"unsupported:{getattr(cls, '__name__', cls)}"
    kwargs["lr"] = 0.0
    if "is_paged" in kwargs:
        kwargs["is_paged"] = False
    try:
        decay = set(Trainer.get_decay_parameter_names(None, model))
    except Exception:
        decay = set()
    groups = [
        {"params": [p for n, p in model.named_parameters() if n in decay and p.requires_grad],
         "weight_decay": float(getattr(training_args, "weight_decay", 0.0) or 0.0)},
        {"params": [p for n, p in model.named_parameters() if n not in decay and p.requires_grad],
         "weight_decay": 0.0},
    ]
    if "bitsandbytes" in str(cls) and kwargs.get("optim_bits", None) == 8:
        try:
            import bitsandbytes
            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
            for module in model.modules():
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
        except Exception as exc:
            return None, None, f"bnb:{type(exc).__name__}"
    try:
        optimizer = cls(groups, **kwargs)
    except Exception as exc:
        return None, None, f"init:{type(exc).__name__}"
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler, f"{cls.__module__}.{cls.__name__}"


def release_probe_optimizer(model, optimizer, scheduler, cuda: bool) -> None:
    model.zero_grad(set_to_none=True)
    if optimizer is not None:
        try:
            optimizer.state.clear()
            for g in optimizer.param_groups:
                g["params"] = []
        except Exception:
            pass
    del optimizer, scheduler
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()


def probe_throughput(
    model,
    seq_len: int,
    vocab: int,
    pad_id: int = 0,
    loss_fn: Optional[Callable] = None,
    with_position_ids: bool = False,
    max_micro: int = LADDER_MAX_MICRO,
    reserve_bytes: float = 0.0,
    headroom_frac: Optional[float] = None,
    frac: Optional[float] = None,
    repeats: int = LADDER_REPEATS,
    training_args=None,
    row_source: Optional[Callable[[int], Optional[dict]]] = None,
    row_tokens: Optional[float] = None,
    warmup: int = LADDER_WARMUP,
    accum_rungs: tuple = (),
    fit_source: Optional[Callable[[int], Optional[dict]]] = None,
    chunker=None,
) -> ThroughputResult:
    t0 = time.perf_counter()
    hr = headroom() if headroom_frac is None else headroom_frac
    fr = throughput_frac() if frac is None else frac
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    res = ThroughputResult(micro=1, frac=fr, headroom=hr, reserve_gb=reserve_bytes / 1e9)
    if cuda:
        res.total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    budget = (res.total_gb * 1e9 * hr - reserve_bytes) if cuda else None
    was_training = model.training
    model.train()
    max_micro = max(1, int(max_micro))
    max_grad_norm = float(getattr(training_args, "max_grad_norm", 0.0) or 0.0) if training_args is not None else 0.0
    optimizer, scheduler, res.optimizer = build_probe_optimizer(model, training_args)
    res.rows = "dataset" if row_source is not None else "synthetic"
    if row_source is not None and fit_source is not None:
        res.rows = "dataset_grouped"
    if chunker is not None:
        res.rows = "dataset_chunked"
    micro = 1

    def _fb(b: dict) -> None:
        if chunker is None:
            _fwd_bwd(model, b, loss_fn)
        else:
            _fwd_bwd(model, b, loss_fn, chunker)

    def _make_batch(m: int) -> Optional[dict]:
        if row_source is not None:
            return row_source(m)
        return _synthetic_batch(m, seq_len, vocab, device, pad_id, with_position_ids, real_len=row_tokens)

    retime: dict = {"on": retime_enabled(), "rungs": [], "repeats": 0}
    floor_on, floor_held = floor_enabled(), 0
    try:
        while micro <= max_micro and (micro * seq_len if chunker is None
                                      else micro * float(row_tokens or seq_len)) <= LADDER_MAX_TOKENS:
            entry = {"micro": micro, "accum": 1, "tokens": micro * seq_len}
            if chunker is not None:
                entry["tokens"] = int(round(micro * float(row_tokens or seq_len)))
            batch = None
            try:
                if cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                batch = _make_batch(micro)
                if batch is None:
                    res.reason = "dataset_end"
                    break
                n_real = real_tokens(batch)
                entry["real_tokens"] = n_real
                if isinstance(batch, dict) and torch.is_tensor(batch.get("input_ids")):
                    entry["computed_tokens"] = int(batch["input_ids"].numel())
                    if chunker is not None:
                        _cs = chunker.split(batch)
                        entry["computed_tokens"] = int(sum(int(c["input_ids"].numel()) for c in _cs))
                        entry["chunks"] = len(_cs)
                        entry["max_chunk_tokens"] = int(max(int(c["input_ids"].numel()) for c in _cs))
                        del _cs
                for _ in range(max(0, int(warmup))):
                    _fb(batch)
                    _opt_step(model, optimizer, scheduler, max_grad_norm)
                if cuda:
                    torch.cuda.synchronize(device)
                best = None
                for _ in range(max(1, int(repeats))):
                    if cuda:
                        torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    _fb(batch)
                    if cuda:
                        torch.cuda.synchronize(device)
                    t2 = time.perf_counter()
                    _opt_step(model, optimizer, scheduler, max_grad_norm)
                    if cuda:
                        torch.cuda.synchronize(device)
                    t3 = time.perf_counter()
                    if best is None or (t3 - t1) < best[0]:
                        best = (t3 - t1, t2 - t1, t3 - t2)
                peak = torch.cuda.max_memory_allocated(device) if cuda else 0.0
                if fit_source is not None:
                    worst = fit_source(micro)
                    if worst is not None:
                        entry["worst_tokens"] = int(worst["input_ids"].numel())
                        _fb(worst)
                        model.zero_grad(set_to_none=True)
                        if cuda:
                            torch.cuda.synchronize(device)
                        peak = max(peak, torch.cuda.max_memory_allocated(device) if cuda else 0.0)
                        del worst
                res.state_gb = round(_state_bytes(optimizer) / 1e9, 3)
                entry["secs"] = round(best[0], 5)
                entry["fwd_bwd_secs"] = round(best[1], 5)
                entry["opt_secs"] = round(best[2], 5)
                entry["tok_s"] = round(n_real / max(best[0], 1e-9), 1)
                entry["real_tok_s"] = entry["tok_s"]
                entry["peak_gb"] = round(peak / 1e9, 3)
                if cuda:
                    entry["reserved_gb"] = round(torch.cuda.memory_reserved(device) / 1e9, 3)
                entry["fits"] = bool(budget is None or peak <= budget)
                res.tries.append(entry)
                if not entry["fits"]:
                    res.reason = "memory_budget"
                    break
                ok = [t for t in res.tries if t.get("fits")]
                if len(ok) >= 3:
                    g1 = ok[-1]["tok_s"] / max(ok[-2]["tok_s"], 1e-9) - 1.0
                    g2 = ok[-2]["tok_s"] / max(ok[-3]["tok_s"], 1e-9) - 1.0
                    if g1 < LADDER_PLATEAU_GAIN and g2 < LADDER_PLATEAU_GAIN:
                        if floor_on and fixed_cost_bound(ok[-1]):
                            floor_held += 1
                        else:
                            res.reason = "plateau"
                            break
            except Exception as exc:
                if not _is_oom(exc):
                    res.reason = f"probe_error:{type(exc).__name__}"
                    res.tries.append({**entry, "error": type(exc).__name__})
                    break
                res.tries.append({**entry, "oom": True})
                res.reason = "oom"
                break
            finally:
                model.zero_grad(set_to_none=True)
                if batch is not None:
                    del batch
                if cuda:
                    torch.cuda.empty_cache()
            micro *= 2
        ok = [t for t in res.tries if t.get("fits")]
        rising = (len(ok) >= 2 and ok[-1]["tok_s"] / max(ok[-2]["tok_s"], 1e-9) - 1.0 >= LADDER_PLATEAU_GAIN) \
            or len(ok) == 1
        if accum_rungs and ok and res.reason in ("oom", "memory_budget", "dataset_end") and rising:
            top = int(ok[-1]["micro"])
            batch = _make_batch(top)
            if batch is not None:
                try:
                    n_real = real_tokens(batch)
                    for k in accum_rungs:
                        k = int(k)
                        if k <= 1:
                            continue
                        entry = {"micro": top, "accum": k, "tokens": k * top * seq_len,
                                 "real_tokens": k * n_real, "accum_rung": True}
                        best = None
                        for _ in range(max(1, int(repeats))):
                            if cuda:
                                torch.cuda.synchronize(device)
                            t1 = time.perf_counter()
                            for _ in range(k):
                                _fb(batch)
                            if cuda:
                                torch.cuda.synchronize(device)
                            t2 = time.perf_counter()
                            _opt_step(model, optimizer, scheduler, max_grad_norm)
                            if cuda:
                                torch.cuda.synchronize(device)
                            t3 = time.perf_counter()
                            if best is None or (t3 - t1) < best[0]:
                                best = (t3 - t1, t2 - t1, t3 - t2)
                        entry["secs"] = round(best[0], 5)
                        entry["fwd_bwd_secs"] = round(best[1], 5)
                        entry["opt_secs"] = round(best[2], 5)
                        entry["tok_s"] = round(k * n_real / max(best[0], 1e-9), 1)
                        entry["real_tok_s"] = entry["tok_s"]
                        entry["peak_gb"] = ok[-1].get("peak_gb")
                        entry["fits"] = True
                        res.tries.append(entry)
                except Exception as exc:
                    res.tries.append({"micro": top, "accum_rung": True, "error": type(exc).__name__})
                finally:
                    model.zero_grad(set_to_none=True)
                    del batch
                    if cuda:
                        torch.cuda.empty_cache()
        if retime["on"]:
            _, _rd0 = choose_rung_read(res.tries, fr, allow_accum=False, ladder=res.tries)
            keys = deciding_rungs(res.tries, _rd0, fr)
            retime["first_read"] = {k: _rd0.get(k) for k in ("micro", "ratio", "margin_sd", "ratio_below")}
            for key in keys:
                if key[1] != 1:
                    continue
                t = next((x for x in res.tries if _rung_key(x) == key and x.get("tok_s")), None)
                if t is None:
                    continue
                batch = _make_batch(int(key[0]))
                if batch is None:
                    continue
                try:
                    n_real = real_tokens(batch)
                    samples = []
                    for _ in range(retime_repeats()):
                        if cuda:
                            torch.cuda.synchronize(device)
                        t1 = time.perf_counter()
                        _fb(batch)
                        if cuda:
                            torch.cuda.synchronize(device)
                        t2 = time.perf_counter()
                        _opt_step(model, optimizer, scheduler, max_grad_norm)
                        if cuda:
                            torch.cuda.synchronize(device)
                        t3 = time.perf_counter()
                        samples.append((t3 - t1, t2 - t1, t3 - t2))
                    best = min(samples)
                    before = {"secs": t["secs"], "tok_s": t["tok_s"]}
                    if best[0] < float(t["secs"]):
                        t["secs"] = round(best[0], 5)
                        t["fwd_bwd_secs"] = round(best[1], 5)
                        t["opt_secs"] = round(best[2], 5)
                        t["tok_s"] = round(n_real / max(best[0], 1e-9), 1)
                        t["real_tok_s"] = t["tok_s"]
                    t["retimed"] = int(len(samples))
                    retime["rungs"].append({"micro": int(key[0]), "before": before,
                                            "after": {"secs": t["secs"], "tok_s": t["tok_s"]}})
                    retime["repeats"] = int(len(samples))
                except Exception as exc:
                    if not _is_oom(exc):
                        retime["error"] = type(exc).__name__
                finally:
                    model.zero_grad(set_to_none=True)
                    del batch
                    if cuda:
                        torch.cuda.empty_cache()
    finally:
        release_probe_optimizer(model, optimizer, scheduler, cuda)
        optimizer = scheduler = None
    if not was_training:
        model.eval()
    if not res.reason:
        res.reason = "ladder_end"
    (res.micro, _, res.best_tok_s, res.best_micro, _), res.rung_read = choose_rung_read(res.tries, fr, allow_accum=False, ladder=res.tries)
    res.rung_read["floor_plateau_held"] = floor_held
    if res.micro == 1:
        res.rung_read["micro_one"] = micro_one_reason(res)
    res.rung_read["retime"] = retime
    res.secs = round(time.perf_counter() - t0, 2)
    return res


def micro_one_reason(res: ThroughputResult) -> str:
    rd = res.rung_read or {}
    fits = [t for t in res.tries if t.get("fits") and int(t.get("accum", 1) or 1) == 1]
    if len(fits) <= 1:
        return f"only_rung_1_fits:{res.reason}"
    if rd.get("floor") == "only_rows_1":
        return "only_rows_1"
    return "read_above_floor" if rd.get("floor_secs") is not None else "read_floor_off"


def emit_throughput(res: ThroughputResult, **extra) -> None:
    pass


def trainable_bytes(model) -> float:
    return float(sum(p.numel() for p in model.parameters() if p.requires_grad)) * OPTIMIZER_BYTES_PER_PARAM


def emit(site: str, plan: Optional[BatchPlan], **extra) -> None:
    fields = dict(site=site, mode=mode())
    if plan is not None:
        fields.update(asdict(plan))
    fields.update(extra)


def emit_probe(res: ProbeResult, **extra) -> None:
    pass


@dataclass
class V3Plan:
    micro: int
    accum: int
    world_size: int
    rows: int
    n_rows: int
    row_tokens: float
    tokens_est: float
    steps_per_epoch: int
    b_thr_micro: int
    b_thr_tok_s: float
    best_tok_s: float
    best_micro: int
    best_accum: int
    branch: str
    c_thr: float
    c_final: float
    epochs_plan: float
    t_step_thr: float
    t_step_final: float
    total_secs: float
    frac: float
    clock_max: float
    b_noise_rows: Optional[float] = None
    b_noise_tokens: Optional[float] = None
    cap_applied: bool = False
    rows_before_cap: int = 0
    floor_rows: int = 0
    cap_reason: str = ""
    rung_read_thr: dict = field(default_factory=dict)
    rung_read_clock: dict = field(default_factory=dict)


def _clock_share(epochs_plan: float, n_rows: int, rows: int, secs: float, total_secs: float,
                 train_util: float) -> float:
    usable = max(1.0, float(total_secs) * float(train_util))
    steps = max(1.0, float(n_rows) / max(1, rows))
    return float(epochs_plan) * steps * float(secs) / usable


def plan_v3(tries: list, world_size: int, n_rows: int, row_tokens: float, total_secs: float,
            epochs_plan: float, frac: Optional[float] = None, clock_cap: Optional[float] = None,
            train_util: float = DEFAULT_TRAIN_UTIL, ladder: Optional[list] = None) -> V3Plan:
    fr = throughput_frac() if frac is None else frac
    cmax = clock_max() if clock_cap is None else clock_cap
    world_size = max(1, int(world_size))
    n_rows = max(1, int(n_rows))
    secs_of = {(int(t["micro"]), int(t.get("accum", 1) or 1)): float(t["secs"])
               for t in tries if t.get("fits") and t.get("secs")}
    (micro_thr, _, thr_best, _, _), read_thr = choose_rung_read(tries, fr, allow_accum=False, ladder=ladder)
    read_clock: dict = {}
    t_thr = secs_of.get((micro_thr, 1), 0.0)
    rows_thr = micro_thr * world_size
    c_thr = _clock_share(epochs_plan, n_rows, rows_thr, t_thr, total_secs, train_util)
    micro, accum, best_tok_s, best_micro, best_accum = micro_thr, 1, thr_best, micro_thr, 1
    branch = "data"
    if c_thr > cmax:
        (micro, accum, best_tok_s, best_micro, best_accum), read_clock = choose_rung_read(
            tries, fr, allow_accum=True, ladder=ladder)
        branch = "clock_accum" if accum > 1 else "clock"
    rows = micro * accum * world_size
    t_final = secs_of.get((micro, accum), t_thr)
    tok_s_thr = next((float(t["tok_s"]) for t in tries if t.get("fits")
                      and int(t["micro"]) == micro_thr and int(t.get("accum", 1) or 1) == 1), 0.0)
    return V3Plan(
        micro=micro, accum=accum, world_size=world_size, rows=rows, n_rows=n_rows,
        row_tokens=float(row_tokens), tokens_est=rows * float(row_tokens),
        steps_per_epoch=max(1, n_rows // rows),
        b_thr_micro=micro_thr, b_thr_tok_s=tok_s_thr, best_tok_s=best_tok_s,
        best_micro=best_micro, best_accum=best_accum, branch=branch,
        c_thr=round(c_thr, 4),
        c_final=round(_clock_share(epochs_plan, n_rows, rows, t_final, total_secs, train_util), 4),
        epochs_plan=float(epochs_plan), t_step_thr=t_thr, t_step_final=t_final,
        total_secs=float(total_secs), frac=fr, clock_max=cmax,
        rung_read_thr=read_thr, rung_read_clock=read_clock,
    )


def cap_to_noise(plan: V3Plan, b_noise_rows: Optional[float], floor_rows: Optional[int] = None,
                 reliable: bool = True) -> V3Plan:
    plan.b_noise_rows = None if b_noise_rows is None else float(b_noise_rows)
    plan.b_noise_tokens = None if b_noise_rows is None else float(b_noise_rows) * plan.row_tokens
    plan.rows_before_cap = plan.rows
    plan.floor_rows = int(plan.b_thr_micro * plan.world_size if floor_rows is None else floor_rows)
    plan.cap_applied = False
    if not reliable or b_noise_rows is None or not math.isfinite(b_noise_rows) or b_noise_rows <= 0:
        plan.cap_reason = "unreliable_estimate"
        return plan
    noise_rows = int(math.ceil(b_noise_rows))
    if noise_rows >= plan.rows:
        plan.cap_reason = "above_noise_ok"
        return plan
    target_rows = max(plan.floor_rows, noise_rows)
    plan.cap_reason = "floor_bound" if plan.floor_rows > noise_rows else "applied"
    if target_rows >= plan.rows:
        return plan
    target = max(1, target_rows // plan.world_size)
    accum = max(1, min(plan.accum, int(math.ceil(target / max(1, plan.micro)))))
    micro = max(1, min(plan.micro, int(math.ceil(target / accum))))
    plan.micro, plan.accum = micro, accum
    plan.rows = micro * accum * plan.world_size
    plan.tokens_est = plan.rows * plan.row_tokens
    plan.steps_per_epoch = max(1, plan.n_rows // plan.rows)
    plan.cap_applied = plan.rows < plan.rows_before_cap
    return plan


def lr_prior_scale_v3(tokens_table: float, tokens_search: float,
                      b_noise_tokens: Optional[float] = None, table_rows: int = 0, rows: int = 0,
                      skip_div: Optional[float] = None) -> tuple[float, str]:
    if not tokens_table or not tokens_search or tokens_table <= 0 or tokens_search <= 0:
        return 1.0, "none"
    div = lr_scale_skip_div() if skip_div is None else max(0.0, float(skip_div))
    if div > 0 and table_rows > 0 and rows > 0 and rows * div < table_rows:
        return 1.0, "below_table_div"
    if b_noise_tokens is None or not math.isfinite(b_noise_tokens) or b_noise_tokens <= 0:
        return math.sqrt(float(tokens_search) / float(tokens_table)), "sqrt"
    lo = min(float(tokens_search), float(b_noise_tokens))
    hi = min(float(tokens_table), float(b_noise_tokens))
    return math.sqrt(lo / hi), "sqrt_capped"


LR_CENTRE_EXPONENT = 0.25


def lr_prior_scale_fit(tokens_table: float, tokens_search: float) -> tuple[float, str]:
    if not tokens_table or not tokens_search or tokens_table <= 0 or tokens_search <= 0:
        return 1.0, "none"
    return (float(tokens_search) / float(tokens_table)) ** LR_CENTRE_EXPONENT, "pow025"


def recentre_log_lr(log_center: float, scale_applied: float, tokens_table: float,
                    tokens_search: float, b_noise_tokens: Optional[float]) -> tuple[Optional[float], dict]:
    scale, reason = lr_prior_scale_v3(tokens_table, tokens_search, b_noise_tokens)
    if reason != "sqrt_capped" or not scale_applied or scale_applied <= 0:
        return None, {"reason": reason, "scale_capped": scale, "scale_applied": scale_applied}
    new = log_center + math.log10(scale / scale_applied)
    return new, {"reason": reason, "scale_capped": scale, "scale_applied": scale_applied,
                 "shift_dex": round(new - log_center, 4)}


def emit_v3(site: str, plan: V3Plan, **extra) -> None:
    fields = dict(site=site, mode=mode())
    fields.update(asdict(plan))
    fields.update(extra)
