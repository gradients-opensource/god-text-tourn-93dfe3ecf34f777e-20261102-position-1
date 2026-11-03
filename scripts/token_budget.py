from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from pydantic import BaseModel
from transformers import TrainerCallback
from transformers.training_args import OptimizerNames

import batch_policy
import host_cpu
import release_defaults
from length_grouping import pad_up
from row_weighted_loss import IGNORE_INDEX, _Window, window_row_weight

try:
    from accelerate.utils import DistributedType
    from accelerate.utils.memory import clear_device_cache
except ImportError:
    DistributedType = None

    def clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

_FALSY = ("0", "false", "no", "off")


def enabled() -> bool:
    return release_defaults.raw("SN56_TOKEN_BUDGET").strip().lower() not in _FALSY


class Gate(BaseModel):
    engaged: bool
    reason: str
    switch: bool
    mode: str
    probe_fits: bool
    dyn_pad: bool
    packed: bool
    use_kl: bool
    world_size: int
    deepspeed: bool
    rows_plan_rowsized: Optional[int] = None
    rows_target_uncut: Optional[int] = None


def gate(*, switch: bool, mode: str, probe_fits: bool, dyn_pad: bool, packed: bool, use_kl: bool,
         world_size: int, deepspeed: bool, r0: Optional[int], r1: Optional[int]) -> Gate:
    if not switch:
        reason = "switch_off"
    elif mode != "v3":
        reason = "not_v3"
    elif not probe_fits:
        reason = "no_probe"
    elif packed:
        reason = "packed"
    elif not dyn_pad:
        reason = "no_dynamic_padding"
    elif use_kl:
        reason = "kl"
    elif int(world_size) != 1:
        reason = "multi_gpu"
    elif deepspeed:
        reason = "deepspeed"
    elif r0 is None or r1 is None:
        reason = "no_plan"
    elif int(r1) <= int(r0):
        reason = "r1_le_r0"
    else:
        reason = "engaged"
    return Gate(engaged=(reason == "engaged"), reason=reason, switch=bool(switch), mode=str(mode),
                probe_fits=bool(probe_fits), dyn_pad=bool(dyn_pad), packed=bool(packed),
                use_kl=bool(use_kl), world_size=int(world_size), deepspeed=bool(deepspeed),
                rows_plan_rowsized=(int(r0) if r0 is not None else None),
                rows_target_uncut=(int(r1) if r1 is not None else None))


def fitting_tries(tries: Optional[list]) -> list:
    return [t for t in (tries or []) if t.get("fits") and t.get("secs")]


def rows_uncut(tries: Optional[list], micro_cap: Optional[int] = None, **plan_kw) -> Optional[int]:
    ladder = fitting_tries(tries)
    ok = [t for t in ladder if micro_cap is None or int(t["micro"]) <= int(micro_cap)]
    if not ok:
        return None
    return int(batch_policy.plan_v3(ok, ladder=ladder, **plan_kw).rows)


class Budget(BaseModel):
    tokens: int
    m_hold: int
    source: str
    iid_fit_tokens: int
    max_length: int
    pad_multiple: int


def budget(m_hold: int, max_length: int, tries: Optional[list], pad_multiple: int = 1) -> Budget:
    L = int(max_length)
    hold = max(0, int(m_hold)) * L
    iid = 0
    for t in tries or []:
        if t.get("fits") and int(t.get("accum", 1) or 1) == 1:
            iid = max(iid, int(t.get("computed_tokens") or 0), int(t.get("worst_tokens") or 0))
    floor = pad_up(L, pad_multiple)
    tokens = max(hold, iid, floor)
    source = ("hold" if hold >= max(iid, floor) else "iid_fit" if iid >= floor else "max_length")
    return Budget(tokens=int(tokens), m_hold=int(m_hold), source=source, iid_fit_tokens=int(iid),
                  max_length=L, pad_multiple=max(1, int(pad_multiple or 1)))


def row_lengths(batch: dict) -> list:
    ids = batch["input_ids"]
    mask = batch.get("attention_mask")
    if torch.is_tensor(mask) and mask.dim() == 2 and mask.shape == ids.shape:
        return [int(x) for x in mask.sum(dim=1).tolist()]
    return [int(ids.shape[1])] * int(ids.shape[0])


def group_rows(lengths: list, budget_tokens: int, pad_multiple: int = 1) -> list:
    order = sorted(range(len(lengths)), key=lambda i: (int(lengths[i]), i))
    groups: list = []
    cur: list = []
    for i in order:
        w = pad_up(max(1, int(lengths[i])), pad_multiple)
        if cur and (len(cur) + 1) * w > int(budget_tokens):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def chunks(batch: dict, budget_tokens: int, pad_multiple: int = 1, padding_side: str = "right") -> list:
    ids = batch["input_ids"]
    rows, width = int(ids.shape[0]), int(ids.shape[1])
    lens = row_lengths(batch)
    groups = group_rows(lens, budget_tokens, pad_multiple)
    if len(groups) <= 1:
        return [batch]
    out = []
    dev = ids.device
    for g in groups:
        w = min(width, pad_up(max(1, max(lens[i] for i in g)), pad_multiple))
        idx = torch.tensor(g, dtype=torch.long, device=dev)
        c = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.dim() == 2 and int(v.shape[0]) == rows and int(v.shape[1]) == width:
                v = v.index_select(0, idx.to(v.device))
                c[k] = v[:, width - w:] if padding_side == "left" else v[:, :w]
            else:
                c[k] = v
        out.append(c)
    return out


def supervised_tokens(batch: dict) -> int:
    labels = batch.get("labels")
    if not torch.is_tensor(labels):
        return 0
    return int(labels[..., 1:].ne(IGNORE_INDEX).sum().item())


class Chunker:

    def __init__(self, budget_tokens: int, pad_multiple: int = 1, padding_side: str = "right"):
        self.tokens = int(budget_tokens)
        self.pad_multiple = max(1, int(pad_multiple or 1))
        self.padding_side = padding_side

    def split(self, batch: dict) -> list:
        return chunks(batch, self.tokens, self.pad_multiple, self.padding_side)

    def fwd_bwd(self, model, batch: dict, loss_fn: Optional[Callable] = None, scale: float = 1.0,
                autocast_dtype=None, backward: bool = True) -> torch.Tensor:
        cs = self.split(batch)
        n_all = supervised_tokens(batch) if (len(cs) > 1 and loss_fn is None) else 0
        if len(cs) == 1 or (loss_fn is None and n_all == 0):
            cs, weights, kw = [batch], [1.0], {}
        else:
            weights = ([1.0] * len(cs) if loss_fn is not None
                       else [supervised_tokens(c) / float(n_all) for c in cs])
            kw = {"norm_batch": batch} if loss_fn is not None else {}
        total = None
        for c, w in zip(cs, weights):
            if w == 0.0:
                continue
            if autocast_dtype is not None:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    loss = (loss_fn(model, c, **kw) if loss_fn is not None else model(**c).loss) * (w * scale)
            else:
                loss = (loss_fn(model, c, **kw) if loss_fn is not None else model(**c).loss) * (w * scale)
            if backward:
                loss.backward()
            d = loss.detach()
            total = d if total is None else total + d
        if total is None:
            total = torch.zeros((), device=next(model.parameters()).device)
        return total


class _State:
    budget: Optional[Budget] = None
    padding_side: str = "right"


STATE = _State()


def install(b: Budget, padding_side: str) -> None:
    STATE.budget = b
    STATE.padding_side = padding_side


def reset() -> None:
    STATE.budget = None
    STATE.padding_side = "right"


def _new_stats() -> dict:
    return {"steps": 0, "chunks": 0, "rows": 0, "padded": 0, "real": 0, "max_tokens": 0,
            "stock_steps": 0, "stock_reason": None}


class TokenBudgetMixin:

    def training_step(self, model, inputs, num_items_in_batch=None):
        b = STATE.budget
        if not hasattr(self, "_tb_stats"):
            self._tb_stats = _new_stats()
        st = self._tb_stats
        why = (None if b is not None else "no_budget")
        if why is None and (not getattr(self, "model_accepts_loss_kwargs", False)
                            or getattr(self, "compute_loss_func", None) is not None
                            or getattr(self, "label_smoother", None) is not None):
            why = "loss_without_num_items"
        if why is None and (not isinstance(inputs, dict) or "labels" not in inputs
                            or "input_ids" not in inputs):
            why = "no_labels"
        if why is not None:
            st["stock_steps"] += 1
            if st["stock_reason"] is None:
                st["stock_reason"] = why
            return super().training_step(model, inputs, num_items_in_batch)
        cs0 = chunks(inputs, b.tokens, b.pad_multiple, STATE.padding_side)
        self._tb_count(inputs, cs0)
        if len(cs0) == 1:
            return super().training_step(model, inputs, num_items_in_batch)

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            cs = chunks(inputs, b.tokens, b.pad_multiple, STATE.padding_side)
            was_none = num_items_in_batch is None
            n_items = num_items_in_batch
            if was_none:
                n_items = inputs["labels"][..., 1:].ne(IGNORE_INDEX).sum()
            has_rw = hasattr(self, "_rw_window")
            win_saved = getattr(self, "_rw_window", None) if has_rw else None
            live = bool(win_saved is not None and win_saved.remaining > 0)
            rem = win_saved.remaining if live else 0
            if has_rw and not live:
                rows_t = torch.tensor(window_row_weight(inputs, self.row_loss_cap),
                                      dtype=torch.float64, device=inputs["input_ids"].device)
                tmp = _Window(rows=rows_t, remaining=1, world=1)
                self._rw_window = tmp
            kwargs = {}
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()
            if DistributedType is not None and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            total = None
            try:
                for c in cs:
                    if has_rw:
                        if live:
                            win_saved.remaining = rem
                        else:
                            tmp.remaining = 1
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(model, c, num_items_in_batch=n_items)
                    if self.args.n_gpu > 1:
                        loss = loss.mean()
                    if was_none:
                        loss = loss / self.current_gradient_accumulation_steps
                    self.accelerator.backward(loss, **kwargs)
                    d = loss.detach()
                    total = d if total is None else total + d
            finally:
                if has_rw:
                    if live:
                        win_saved.remaining = rem - 1
                    else:
                        self._rw_window = win_saved
            del inputs
            if (self.args.torch_empty_cache_steps is not None
                    and self.state.global_step % self.args.torch_empty_cache_steps == 0):
                clear_device_cache()
            return total

    def _tb_count(self, inputs: dict, cs: list) -> None:
        st = self._tb_stats
        st["steps"] += 1
        st["chunks"] += len(cs)
        for c in cs:
            ids = c["input_ids"]
            st["rows"] += int(ids.shape[0])
            st["padded"] += int(ids.numel())
            st["max_tokens"] = max(st["max_tokens"], int(ids.numel()))
        st["real"] += sum(row_lengths(inputs))


def with_token_budget(cls):
    if cls in _CLASSES:
        return _CLASSES[cls]
    _CLASSES[cls] = type(f"TokenBudget{cls.__name__}", (TokenBudgetMixin, cls), {})
    return _CLASSES[cls]


_CLASSES: dict = {}


class StepsCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        pass


def ladder_ooms(res) -> list:
    return [int(t.get("micro", 0)) for t in (getattr(res, "tries", None) or []) if t.get("oom")]


def budget_from_hold(hold, max_length: int, tries: Optional[list], pad_multiple: int) -> Budget:
    m_hold = int(hold.micro) if (hold is not None and hold.reason == "fits") else 0
    return budget(m_hold, max_length, tries, pad_multiple)


@dataclass
class Engaged:
    budget: Budget
    plan: Any
    probe: Any


def engage(model, *, gate: Gate, probe, micro_cap: int, probe_kw: dict, plan_kw: dict, train_ds,
           collator, lengths: list, row_tokens: float, max_length: int, padding_side: str,
           training_args=None, relaunch_oom: bool = False,
           log: Callable[[str], None] = print) -> Optional[Engaged]:
    try:
        pad_mult = int(getattr(collator, "pad_to_multiple_of", 1) or 1)
        hold = batch_policy.probe_micro_batch(model, int(micro_cap), **probe_kw)
        batch_policy.emit_probe(hold, relaunch_oom=relaunch_oom, token_budget_hold=True)
        b = budget_from_hold(hold, int(max_length), probe.tries, pad_mult)
        chunker = Chunker(b.tokens, pad_mult, padding_side)
        rows = batch_policy.dataset_rows(train_ds, collator, next(model.parameters()).device,
                                         lengths=lengths, grouped="typical", target_len=row_tokens)
        if rows is None:
            raise RuntimeError("no dataset rows for the chunked ladder")
        with host_cpu.reference_threads("batch_probe_chunked") as ref:
            res = batch_policy.probe_throughput(
                model, row_source=rows, chunker=chunker,
                max_micro=(int(micro_cap) if relaunch_oom else batch_policy.LADDER_MAX_MICRO),
                training_args=training_args, row_tokens=row_tokens, accum_rungs=(), **probe_kw)
        batch_policy.emit_throughput(res, relaunch_oom=relaunch_oom, grouped=False,
                                     token_budget=b.tokens)
        ooms = ladder_ooms(res)
        if ooms:
            log(f"[sn56][lote] ERRO: OOM no degrau em pedaços {ooms} (pedaços <= {b.tokens} tokens)")
        tries = fitting_tries(res.tries)
        if not tries:
            raise RuntimeError(f"chunked ladder: no fitting rung ({res.reason})")
        plan = batch_policy.plan_v3(tries, ladder=tries, **plan_kw)
        install(b, padding_side)
        log(f"[sn56][lote] orçamento de tokens T={b.tokens} ({b.source}, m_hold={b.m_hold}, "
            f"iid {b.iid_fit_tokens}): linhas {gate.rows_plan_rowsized} -> {plan.rows} por passo, "
            f"em pedaços")
        return Engaged(budget=b, plan=plan, probe=res)
    except Exception as exc:
        log(f"[sn56][lote] orçamento de tokens falhou ({exc}), plano por linhas mantido")
        reset()
        return None
