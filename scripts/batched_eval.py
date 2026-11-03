import array
import copy
import math
import os
import time
import weakref
import zlib
from typing import Any, Optional

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

import planner_clock
from row_weighted_loss import (IGNORE_INDEX, DevSummary, nll_from_logits, reduce_rows,
                               segment_ids, shifted_targets, short_rows_correct, NO_CAP)
from utility import DynamicPaddingCollator

ENV = "SN56_EVAL_BATCH"
ENV_MAX = "SN56_EVAL_BATCH_MAX"
DEFAULT_MAX = 16
MEM_FRACTION = 0.6
CANARY_TOKEN_MEAN_ABS = 0.02
CANARY_MEAN_SHIFT = 0.01

_PLANS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_CANARY: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
B1_TIME_FACTOR: Optional[float] = None
_BYTES_PER_TOKEN: dict = {}

ENV_COLLATE_CACHE = "SN56_EVAL_COLLATE_CACHE"
ENV_COLLATE_CACHE_MB = "SN56_EVAL_COLLATE_CACHE_MB"
DEFAULT_COLLATE_CACHE_MB = 1024
_COLLATED: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_COLLATE_OVER: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def batch_max() -> int:
    try:
        return max(1, int(os.environ.get(ENV_MAX, str(DEFAULT_MAX))))
    except ValueError:
        return DEFAULT_MAX


class EvalBatchPlan(BaseModel):
    batches: list[list[int]]
    tokens_max: int
    rows_max: int
    n_rows: int
    max_len: int
    padded_tokens: int
    real_tokens: int


class RowTotals(BaseModel):
    row_mean_sum: float = 0.0
    n_rows: int = 0
    n_empty: int = 0
    tok_sum: float = 0.0
    n_tok: int = 0
    ex_mean_sum: float = 0.0
    n_ex: int = 0
    n_short: int = 0
    short_mean_sum: float = 0.0
    short_ok: int = 0
    n_batches: int = 0
    max_batch: int = 0
    n_oom_splits: int = 0


class CollatedDev(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    key: tuple
    batches: dict
    nbytes: int
    collate_secs: float


class CollateInfo(BaseModel):
    cache: str
    reason: Optional[str] = None
    secs: float = 0.0
    nbytes: int = 0


def collate_cache_enabled() -> bool:
    return os.environ.get(ENV_COLLATE_CACHE, "1").strip().lower() not in ("0", "false", "no", "off")


def collate_cache_max_bytes() -> int:
    try:
        mb = float(os.environ.get(ENV_COLLATE_CACHE_MB, str(DEFAULT_COLLATE_CACHE_MB)))
    except ValueError:
        mb = DEFAULT_COLLATE_CACHE_MB
    return int(max(0.0, mb) * (1 << 20))


_COLLATOR_FIELDS = ("pad_token_id", "padding_side", "label_pad_token_id", "pad_to_multiple_of",
                    "max_length", "with_seq_idx", "vocab_limit", "signature_columns")


def collator_signature(collate) -> tuple:
    sig = []
    c, depth = collate, 0
    while c is not None and depth < 4:
        fields = []
        for f in _COLLATOR_FIELDS:
            if hasattr(c, f):
                v = getattr(c, f)
                fields.append((f, tuple(v) if isinstance(v, (list, set)) else v))
        sig.append((type(c).__module__, type(c).__qualname__, tuple(fields)))
        c = getattr(c, "data_collator", None) or getattr(c, "inner", None)
        depth += 1
    return tuple(sig)


def _value_bytes(v: Any) -> bytes:
    if torch.is_tensor(v):
        return v.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(v, (list, tuple)):
        try:
            return array.array("q", v).tobytes()
        except (TypeError, OverflowError):
            return repr(v).encode()
    return repr(v).encode()


def _row_crc(row: Any, crc: int = 0) -> int:
    if isinstance(row, dict):
        for k in sorted(row, key=str):
            crc = zlib.crc32(str(k).encode(), crc)
            crc = zlib.crc32(_value_bytes(row[k]), crc)
        return crc
    return zlib.crc32(_value_bytes(row), crc)


def dataset_content_crc(ds) -> int:
    crc = 0
    raw = getattr(ds, "data_points", None)
    if not isinstance(raw, list):
        raw = getattr(ds, "eval_dataset", None)
    if isinstance(raw, list):
        for row in raw:
            crc = _row_crc(row, crc)
        groups = getattr(ds, "groups", None)
        if groups is not None:
            crc = zlib.crc32(repr(groups).encode(), crc)
        return crc
    for i in range(len(ds)):
        crc = _row_crc(ds[i], crc)
    return crc


def dataset_signature(ds, lengths: list[int]) -> tuple:
    return (type(ds).__qualname__, len(ds), tuple(int(x) for x in lengths),
            getattr(ds, "max_length", None), getattr(ds, "dynamic_padding", None),
            getattr(ds, "max_input_length", None), dataset_content_crc(ds))


def _batch_nbytes(batch: dict) -> int:
    return int(sum(v.numel() * v.element_size() for v in batch.values() if torch.is_tensor(v)))


class _Precollated:
    __slots__ = ("batch",)

    def __init__(self, batch: dict):
        self.batch = batch


class PrecollatedDev(Dataset):

    def __init__(self, ds, collated: CollatedDev):
        self.ds = ds
        self.collated = collated

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]

    def __getitems__(self, idx):
        hit = self.collated.batches.get(tuple(int(i) for i in idx))
        if hit is None:
            return [self.ds[i] for i in idx]
        return _Precollated(hit)


class CachedCollate:

    def __init__(self, collate, clone: bool):
        self.collate = collate
        self.clone = bool(clone)

    def __call__(self, data):
        if isinstance(data, _Precollated):
            b = data.batch
            if self.clone:
                return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b.items()}
            return dict(b)
        return self.collate(data)


def _rows_list(rows: Optional[list]) -> Optional[list]:
    if not rows:
        return None
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized() \
                and torch.distributed.get_world_size() > 1:
            return None
        return [float(x) for x in torch.cat([r.reshape(-1).cpu() for r in rows]).tolist()]
    except Exception:
        return None


def plan_batches(lengths: list[int], tokens_max: int, rows_max: int) -> EvalBatchPlan:
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and ((len(cur) + 1) * max(1, lengths[cur[0]]) > tokens_max or len(cur) >= rows_max):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    padded = sum(len(b) * lengths[b[0]] for b in batches)
    real = sum(lengths)
    return EvalBatchPlan(batches=batches, tokens_max=int(tokens_max), rows_max=int(rows_max),
                         n_rows=len(lengths), max_len=max(lengths) if lengths else 0,
                         padded_tokens=int(padded), real_tokens=int(real))


def dataset_lengths(ds) -> list[int]:
    lens = getattr(ds, "lengths", None)
    if lens is not None and len(lens) == len(ds) and not hasattr(ds, "groups"):
        return [int(x) for x in lens]
    if hasattr(ds, "groups") and hasattr(ds, "max_input_length"):
        return [int(ds.max_input_length)] * len(ds)
    return [int(len(ds[i]["input_ids"])) for i in range(len(ds))]


def right_pad_collator(base):
    inner = getattr(base, "inner", base)
    if not isinstance(inner, DynamicPaddingCollator):
        return None
    c = copy.copy(inner)
    c.padding_side = "right"
    c.stats = {k: 0 for k in inner.stats}
    return c


def _row_terms(logits: torch.Tensor, labels: torch.Tensor, position_ids, attention_mask):
    B = labels.shape[0]
    tgt = shifted_targets(labels)
    seg_all = segment_ids(labels, position_ids, attention_mask)
    row_mean = torch.full((B,), float("nan"), dtype=torch.float64, device=logits.device)
    n_tok = torch.zeros(B, dtype=torch.long, device=logits.device)
    ex = torch.zeros(7, dtype=torch.float64, device=logits.device)
    for b in range(B):
        idx = (tgt[b] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        lg = logits[b].index_select(0, idx)
        t = tgt[b].index_select(0, idx)
        nll = nll_from_logits(lg, t).to(torch.float64)
        row_mean[b] = nll.sum() / idx.numel()
        n_tok[b] = idx.numel()
        seg = seg_all[b].index_select(0, idx)
        terms = reduce_rows(nll, seg, NO_CAP)
        ok = short_rows_correct(lg, t, seg)
        ex += torch.stack([terms.tok_sum.to(torch.float64), terms.n_tok.to(torch.float64),
                           terms.row_mean_sum.to(torch.float64), terms.n_rows.to(torch.float64),
                           terms.n_short.to(torch.float64), terms.short_mean_sum.to(torch.float64),
                           ok.to(torch.float64)])
        del lg, nll
    return row_mean, n_tok, ex


def _split(inputs: dict, lo: int, hi: int) -> dict:
    return {k: (v[lo:hi] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == _bs(inputs) else v)
            for k, v in inputs.items()}


def _bs(inputs: dict) -> int:
    return int(inputs["input_ids"].shape[0])


def _trim(inputs: dict) -> dict:
    am = inputs.get("attention_mask")
    if am is None or am.dim() != 2 or "position_ids" in inputs:
        return inputs
    keep = int(am.sum(dim=0).nonzero().max().item()) + 1 if bool(am.any()) else am.shape[1]
    if keep == am.shape[1]:
        return inputs
    return {k: (v[:, :keep] if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == am.shape[1]
                else v) for k, v in inputs.items()}


class BatchedEvalMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evb_reason = self._evb_eligibility()
        self._evb_collator = right_pad_collator(self.data_collator) if not self._evb_reason else None
        if not self._evb_reason and self._evb_collator is None:
            self._evb_reason = "collator_not_dynamic_padding"
        self._evb_on = not self._evb_reason
        self._evb_now = False
        self._evb_tot: Optional[RowTotals] = None
        self._evb_bytes_per_token: Optional[float] = None
        self._evb_tokens_cap: Optional[int] = None
        self._evb_canary_done = False
        self._evb_plan: Optional[EvalBatchPlan] = None
        self._evb_collate = None
        self._evb_collate_info: Optional[CollateInfo] = None
        self._evb_acc = None
        self._evb_ex = None
        self._evb_rows: Optional[list] = None
        self.last_eval_rows: Optional[list] = None

    def _evb_eligibility(self) -> str:
        if not enabled():
            return "disabled_env"
        if int(getattr(self.args, "world_size", 1) or 1) != 1:
            return "world_size"
        if getattr(self, "kl_coef", None) is not None:
            return "kl_trainer"
        if getattr(self, "label_smoother", None) is not None:
            return "label_smoothing"
        if getattr(self, "compute_loss_func", None) is not None:
            return "compute_loss_func"
        m = self.model
        while hasattr(m, "module"):
            m = m.module
        cfg = getattr(m, "config", None)
        if bool(getattr(cfg, "output_router_logits", False)):
            return "router_aux_loss"
        if getattr(m, "loss_type", None) not in (None, "ForCausalLM"):
            return f"loss_type_{getattr(m, 'loss_type', None)}"
        if getattr(self, "is_deepspeed_enabled", False) or getattr(self, "is_fsdp_enabled", False):
            return "sharded"
        return ""

    def _evb_vocab(self) -> int:
        m = self.model
        while hasattr(m, "module"):
            m = m.module
        try:
            return int(m.get_output_embeddings().weight.shape[0])
        except Exception:
            return int(getattr(getattr(m, "config", None), "vocab_size", 32000) or 32000)

    def _evb_free_bytes(self) -> Optional[int]:
        dev = self.args.device
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info(dev)
        cached = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        return int(free + max(0, cached))

    def _evb_tokens_budget(self, max_len: int) -> int:
        free = self._evb_free_bytes()
        if free is None:
            bpt = self._evb_bytes_per_token or (self._evb_vocab() * 2 * 2.0)
            tokens = int((1 << 30) / bpt)
        else:
            bpt = self._evb_bytes_per_token or (self._evb_vocab() * 2 * 3.0)
            tokens = int(MEM_FRACTION * free / bpt)
        if self._evb_tokens_cap is not None:
            tokens = min(tokens, self._evb_tokens_cap)
        return max(int(max_len), tokens)

    def _evb_measure(self, model, ds, lengths) -> None:
        key = (type(self.model).__name__, self._evb_vocab(),
               sum(p.numel() for p in self.model.parameters()))
        if self._evb_bytes_per_token is None and key in _BYTES_PER_TOKEN:
            self._evb_bytes_per_token = _BYTES_PER_TOKEN[key]
        if self._evb_bytes_per_token is not None or self._evb_free_bytes() is None:
            return
        i = max(range(len(lengths)), key=lambda j: (lengths[j], -j))
        batch = self._evb_batch(ds, [i])
        dev = self.args.device
        torch.cuda.synchronize(dev)
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        was_training = model.training
        model.eval()
        try:
            self._evb_forward_terms(model, batch)
        finally:
            if was_training:
                model.train()
        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev) - base
        self._evb_bytes_per_token = max(1.0, peak / max(1, lengths[i]))
        _BYTES_PER_TOKEN[key] = self._evb_bytes_per_token

    def _evb_batch(self, ds, idx: list) -> dict:
        if self._evb_collate is None:
            self._evb_collate = self._get_collator_with_removed_columns(
                self._evb_collator, description="Evaluation")
        return self._prepare_inputs(self._evb_collate([ds[i] for i in idx]))

    def _evb_forward_terms(self, model, inputs: dict):
        fwd = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad(), self.compute_loss_context_manager():
            out = model(**fwd)
        logits = out.logits if hasattr(out, "logits") else out[0]
        res = _row_terms(logits, inputs["labels"], inputs.get("position_ids"),
                         inputs.get("attention_mask"))
        del out, logits
        return res

    def _evb_terms_oom_safe(self, model, inputs: dict):
        try:
            return self._evb_forward_terms(model, inputs)
        except torch.cuda.OutOfMemoryError:
            n = _bs(inputs)
            if n <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            L = int(inputs["input_ids"].shape[1])
            self._evb_tokens_cap = max(L, (n // 2) * L) if self._evb_tokens_cap is None \
                else max(L, min(self._evb_tokens_cap, (n // 2) * L))
            if self._evb_tot is not None:
                self._evb_tot.n_oom_splits += 1
            h = n // 2
            a = self._evb_terms_oom_safe(model, _trim(_split(inputs, 0, h)))
            b = self._evb_terms_oom_safe(model, _trim(_split(inputs, h, n)))
            return torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]), a[2] + b[2]

    def _evb_canary(self, model, ds, lengths, plan: EvalBatchPlan) -> None:
        self._evb_canary_done = True
        try:
            verdict = _CANARY.get(ds)
        except TypeError:
            verdict = None
        if verdict is not None:
            if not verdict:
                self._evb_on, self._evb_reason = False, "canary_failed"
            return
        rows = min(plan.rows_max, len(lengths))
        if rows < 2:
            return
        order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
        pick = sorted({order[round(k * (len(order) - 1) / (rows - 1))] for k in range(rows)},
                      key=lambda i: (-lengths[i], i))
        while pick and len(pick) * lengths[pick[0]] > plan.tokens_max:
            pick = pick[1:]
        if len(pick) < 2:
            return
        was_training = model.training
        model.eval()
        _cuda = torch.cuda.is_available() and next(model.parameters()).device.type == "cuda"
        try:
            if _cuda:
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            bt = self._evb_forward_terms(model, self._evb_batch(ds, pick))
            if _cuda:
                torch.cuda.synchronize()
            _t1 = time.perf_counter()
            sg = [self._evb_forward_terms(model, self._evb_batch(ds, [i])) for i in pick]
            if _cuda:
                torch.cuda.synchronize()
            _t2 = time.perf_counter()
        finally:
            if was_training:
                model.train()
        _b1_factor = ((_t2 - _t1) / (_t1 - _t0)) if (_t1 - _t0) > 0 else None
        batched, n_tok = bt[0].cpu(), bt[1].cpu().to(torch.float64)
        single = torch.cat([x[0] for x in sg]).cpu()
        ok = ~(torch.isnan(batched) | torch.isnan(single))
        d, w = (batched - single)[ok], n_tok[ok]
        tw = float(w.sum()) or 1.0
        tok_mean_abs = float((d.abs() * w).sum()) / tw if d.numel() else 0.0
        shift = float((d * w).sum()) / tw if d.numel() else 0.0
        passed = tok_mean_abs <= CANARY_TOKEN_MEAN_ABS and abs(shift) <= CANARY_MEAN_SHIFT
        global B1_TIME_FACTOR
        if passed and _b1_factor is not None and B1_TIME_FACTOR is None:
            B1_TIME_FACTOR = max(1.0, float(_b1_factor))
            planner_clock.EVAL_B1_FACTOR = B1_TIME_FACTOR
        try:
            _CANARY[ds] = passed
        except TypeError:
            pass
        if not passed:
            self._evb_on = False
            self._evb_reason = "canary_failed"

    def _evb_shared_plan(self, ds, lengths) -> EvalBatchPlan:
        budget = self._evb_tokens_budget(max(lengths) if lengths else 1)
        try:
            plan = _PLANS.get(ds)
        except TypeError:
            plan = None
        if (plan is None or plan.tokens_max > budget or plan.rows_max != batch_max()
                or plan.n_rows != len(lengths)):
            plan = plan_batches(lengths, budget, batch_max())
            try:
                _PLANS[ds] = plan
            except TypeError:
                pass
        return plan

    def _evb_collated(self, ds, lengths, plan: EvalBatchPlan) -> tuple[Optional[CollatedDev], CollateInfo]:
        if not collate_cache_enabled():
            return None, CollateInfo(cache="off", reason="disabled_env")
        if int(self.args.dataloader_num_workers or 0) > 0:
            return None, CollateInfo(cache="off", reason="dataloader_workers")
        try:
            key = (tuple(tuple(b) for b in plan.batches), collator_signature(self._evb_collate),
                   dataset_signature(ds, lengths))
        except Exception as exc:
            return None, CollateInfo(cache="off", reason=f"signature:{type(exc).__name__}")
        try:
            hit = _COLLATED.get(ds)
        except TypeError:
            return None, CollateInfo(cache="off", reason="unhashable_dataset")
        if hit is not None and hit.key == key:
            return hit, CollateInfo(cache="hit", nbytes=hit.nbytes)
        cap = collate_cache_max_bytes()
        over = _COLLATE_OVER.get(ds)
        if over is not None and over[0] == key and over[1] == cap:
            return None, CollateInfo(cache="off", reason="size_memo", nbytes=over[2])
        t0 = time.perf_counter()
        batches, nbytes = {}, 0
        for b in plan.batches:
            out = self._evb_collate([ds[i] for i in b])
            nbytes += _batch_nbytes(out)
            if nbytes > cap:
                _COLLATE_OVER[ds] = (key, cap, nbytes)
                return None, CollateInfo(cache="off", reason="size",
                                         secs=round(time.perf_counter() - t0, 3), nbytes=nbytes)
            batches[tuple(int(i) for i in b)] = out
        entry = CollatedDev(key=key, batches=batches, nbytes=nbytes,
                            collate_secs=round(time.perf_counter() - t0, 3))
        try:
            _COLLATED[ds] = entry
        except TypeError:
            pass
        return entry, CollateInfo(cache="miss", secs=entry.collate_secs, nbytes=nbytes)

    def stock_eval_dataloader(self, batch_size: int):
        saved = self.args.per_device_eval_batch_size
        self.args.per_device_eval_batch_size = int(batch_size)
        try:
            return super().get_eval_dataloader()
        finally:
            self.args.per_device_eval_batch_size = saved

    def get_eval_dataloader(self, eval_dataset=None):
        if not self._evb_now:
            return super().get_eval_dataloader(eval_dataset)
        ds = (self.eval_dataset[eval_dataset] if isinstance(eval_dataset, str)
              else eval_dataset if eval_dataset is not None else self.eval_dataset)
        lengths = dataset_lengths(ds)
        model = self.model_wrapped if self.model_wrapped is not None else self.model
        self._evb_measure(model, ds, lengths)
        plan = self._evb_shared_plan(ds, lengths)
        if not self._evb_canary_done:
            self._evb_canary(model, ds, lengths, plan)
            if not self._evb_on:
                plan = plan_batches(lengths, max(lengths) if lengths else 1, 1)
        self._evb_plan = plan
        if self._evb_collate is None:
            self._evb_collate = self._get_collator_with_removed_columns(
                self._evb_collator, description="Evaluation")
        collated, self._evb_collate_info = self._evb_collated(ds, lengths, plan)
        if collated is not None:
            return DataLoader(PrecollatedDev(ds, collated), batch_sampler=plan.batches,
                              collate_fn=CachedCollate(self._evb_collate, clone=True),
                              num_workers=0, pin_memory=False)
        return DataLoader(ds, batch_sampler=plan.batches, collate_fn=self._evb_collate,
                          num_workers=self.args.dataloader_num_workers,
                          pin_memory=False)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not (self._evb_now and prediction_loss_only and "labels" in inputs):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        inputs = self._prepare_inputs(inputs)
        row_mean, _, ex = self._evb_terms_oom_safe(model, inputs)
        tot = self._evb_tot
        ok = ~torch.isnan(row_mean)
        vals = torch.stack([row_mean[ok].sum(), ok.sum().to(torch.float64),
                            (~ok).sum().to(torch.float64)])
        self._evb_acc = vals if self._evb_acc is None else self._evb_acc + vals
        self._evb_ex = ex if self._evb_ex is None else self._evb_ex + ex
        if self._evb_rows is not None:
            self._evb_rows.append(row_mean.detach().to(torch.float64))
        tot.n_batches += 1
        tot.max_batch = max(tot.max_batch, _bs(inputs))
        loss = row_mean[ok].mean() if bool(ok.any()) else row_mean.new_zeros(())
        return loss.to(torch.float32).detach(), None, None

    def evaluate(self, *args, **kwargs):
        self._evb_now = bool(self._evb_on)
        self._evb_collate_info = None
        self._evb_tot = RowTotals()
        self._evb_acc = None
        self._evb_ex = None
        self._evb_rows = [] if self._evb_now else None
        self.last_eval_rows = None
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self._evb_now = False

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        if not self._evb_now:
            return output
        prefix = kwargs.get("metric_key_prefix", args[4] if len(args) > 4 else "eval")
        acc = self._evb_acc.tolist() if self._evb_acc is not None else [0.0, 0.0, 0.0]
        ex = self._evb_ex.tolist() if self._evb_ex is not None else [0.0] * 7
        tot = self._evb_tot
        tot.row_mean_sum, tot.n_rows, tot.n_empty = acc[0], int(acc[1]), int(acc[2])
        (tot.tok_sum, n_tok, tot.ex_mean_sum, n_ex, n_short, tot.short_mean_sum, short_ok) = ex
        tot.n_tok, tot.n_ex, tot.n_short, tot.short_ok = int(n_tok), int(n_ex), int(n_short), int(short_ok)
        metrics = output.metrics
        self.last_eval_rows = _rows_list(self._evb_rows)
        self._evb_rows = None
        if tot.n_rows > 0 and isinstance(metrics, dict):
            hf = metrics.get(f"{prefix}_loss")
            metrics[f"{prefix}_loss"] = tot.row_mean_sum / tot.n_rows
            if hf is not None:
                metrics[f"{prefix}_loss_hf_batched"] = hf
            summary = DevSummary(
                dev_macro=(tot.ex_mean_sum / tot.n_ex) if tot.n_ex else float("nan"),
                dev_micro=tot.tok_sum / max(tot.n_tok, 1),
                dev_n_rows=tot.n_ex,
                dev_mc=(tot.short_mean_sum / tot.n_short) if tot.n_short else None,
                dev_mc_acc=(tot.short_ok / tot.n_short) if tot.n_short else None,
                dev_mc_n=tot.n_short)
            for k, v in summary.model_dump().items():
                if v is not None:
                    metrics[f"{prefix}_{k}"] = v
        return output


_CLASSES: dict = {}


def with_batched_eval(cls):
    if not issubclass(cls, Trainer):
        return cls
    if cls not in _CLASSES:
        _CLASSES[cls] = type(f"BatchedEval{cls.__name__}", (BatchedEvalMixin, cls), {})
    return _CLASSES[cls]
