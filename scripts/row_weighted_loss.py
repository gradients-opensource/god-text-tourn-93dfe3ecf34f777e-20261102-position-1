
import os
from typing import Any, Callable, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from pydantic import BaseModel
from transformers import Trainer

from utility import log_info

IGNORE_INDEX = -100
ENV_ALPHA = "SN56_ROW_WEIGHT_ALPHA"
ENV_CAP = "SN56_ROW_LOSS_CAP"
ENV_CHUNK = "SN56_ROW_LOSS_CHUNK"
DEFAULT_ALPHA = 0.25
DEFAULT_CAP = 30.0
NO_CAP = float("inf")
DEFAULT_CHUNK = 2048
SHORT_ROW_TOKENS = 2

MODE_HIDDEN = "hidden"
MODE_LOGITS = "logits"


def row_weight_alpha(default: float = DEFAULT_ALPHA) -> float:
    
    raw = os.environ.get(ENV_ALPHA)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        val = float(raw)
    except ValueError:
        log_info(f"[sn56][rowloss] {ENV_ALPHA}={raw!r} inválido, usando {default}")
        return float(default)
    if val != val:
        return float(default)
    return max(0.0, min(1.0, val))


def row_loss_cap(default: float = DEFAULT_CAP) -> float:
    
    raw = os.environ.get(ENV_CAP)
    if raw is None or raw.strip() == "":
        return float(default)
    txt = raw.strip().lower()
    if txt in ("inf", "none", "off"):
        return NO_CAP
    try:
        val = float(txt)
    except ValueError:
        log_info(f"[sn56][rowloss] {ENV_CAP}={raw!r} inválido, usando {default}")
        return float(default)
    if val != val or val <= 0.0:
        log_info(f"[sn56][rowloss] {ENV_CAP}={raw!r} inválido, usando {default}")
        return float(default)
    return val


def cap_for_record(cap: float) -> Optional[float]:
    
    return None if cap == NO_CAP else float(cap)


def _chunk_size() -> int:
    try:
        return max(1, int(os.environ.get(ENV_CHUNK, str(DEFAULT_CHUNK))))
    except ValueError:
        return DEFAULT_CHUNK


def shifted_targets(labels: torch.Tensor) -> torch.Tensor:
    
    return F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def segment_ids(
    labels: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    
    bsz, seqlen = labels.shape
    if position_ids is not None and position_ids.shape == labels.shape:
        seg = torch.cumsum((position_ids == 0).to(torch.long), dim=1)
    elif (
        attention_mask is not None
        and attention_mask.dim() == 2
        and attention_mask.shape == labels.shape
        and bool((attention_mask > 1).any())
    ):
        seg = attention_mask.to(torch.long)
    else:
        seg = torch.zeros_like(labels, dtype=torch.long)
    seg = torch.cat([seg[:, 1:], seg[:, -1:]], dim=1)
    offset = torch.arange(bsz, device=labels.device, dtype=torch.long).unsqueeze(1) * (seqlen + 2)
    return seg + offset


def _row_lengths(batch: dict) -> Optional[torch.Tensor]:
    
    labels = batch.get("labels")
    if labels is None or not torch.is_tensor(labels):
        return None
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    pos = batch.get("position_ids")
    mask = batch.get("attention_mask")
    seg = segment_ids(labels, pos if torch.is_tensor(pos) else None,
                      mask if torch.is_tensor(mask) else None)
    tgt = shifted_targets(labels)
    sel = tgt != IGNORE_INDEX
    if not bool(sel.any()):
        return None
    _, cnt = torch.unique(seg[sel], return_counts=True)
    return cnt


def row_weights(row_cnt: torch.Tensor, cap: float) -> torch.Tensor:
    
    w = row_cnt.to(torch.float64)
    if cap == NO_CAP:
        return torch.ones_like(w)
    return (w / float(cap)).clamp_min(1.0)


def window_row_weight(batch: dict, cap: float) -> float:
    
    cnt = _row_lengths(batch)
    return 0.0 if cnt is None else float(row_weights(cnt, cap).sum())


def _ce_from_hidden(h, weight, bias, t, softcap):
    logits = F.linear(h.to(weight.dtype), weight, bias).float()
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, t, reduction="none")


def nll_from_hidden(hidden, weight, bias, targets, softcap=None, chunk_size=None):
    
    chunk_size = chunk_size or _chunk_size()
    use_ckpt = torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    parts = []
    for start in range(0, hidden.shape[0], chunk_size):
        h = hidden[start:start + chunk_size]
        t = targets[start:start + chunk_size]
        if use_ckpt:
            parts.append(torch.utils.checkpoint.checkpoint(
                _ce_from_hidden, h, weight, bias, t, softcap, use_reentrant=False))
        else:
            parts.append(_ce_from_hidden(h, weight, bias, t, softcap))
    if not parts:
        return hidden.new_zeros((0,), dtype=torch.float32)
    return torch.cat(parts)


def nll_from_logits(logits, targets, chunk_size=None):
    
    chunk_size = chunk_size or _chunk_size()
    parts = [
        F.cross_entropy(logits[s:s + chunk_size].float(), targets[s:s + chunk_size], reduction="none")
        for s in range(0, logits.shape[0], chunk_size)
    ]
    if not parts:
        return logits.new_zeros((0,), dtype=torch.float32)
    return torch.cat(parts)


class RowTerms(NamedTuple):
    
    tok_sum: torch.Tensor
    row_mean_sum: torch.Tensor
    n_tok: torch.Tensor
    n_rows: torch.Tensor
    n_short: torch.Tensor
    short_tok: torch.Tensor
    short_mean_sum: torch.Tensor
    row_w_sum: torch.Tensor
    grader_mean_sum: torch.Tensor


def reduce_rows(nll: torch.Tensor, seg: torch.Tensor, cap: float = NO_CAP) -> RowTerms:
    
    device = nll.device
    if nll.numel() == 0:
        zero_l = torch.zeros((), dtype=torch.long, device=device)
        z = nll.sum()
        return RowTerms(z, z, zero_l, zero_l, zero_l, zero_l, z.detach(),
                        torch.zeros((), dtype=torch.float64, device=device), z.detach())
    uniq, inv = torch.unique(seg, return_inverse=True)
    n_rows = int(uniq.numel())
    row_sum = torch.zeros(n_rows, dtype=nll.dtype, device=device).index_add(0, inv, nll)
    row_cnt = torch.zeros(n_rows, dtype=torch.long, device=device).index_add(
        0, inv, torch.ones_like(inv))
    row_mean = row_sum / row_cnt.to(nll.dtype)
    if cap == NO_CAP:
        n_eff = row_cnt.to(nll.dtype)
    else:
        n_eff = row_cnt.to(nll.dtype).clamp_max(float(cap))
    short = row_cnt <= SHORT_ROW_TOKENS
    return RowTerms(
        tok_sum=nll.sum(),
        row_mean_sum=(row_sum / n_eff).sum(),
        n_tok=torch.tensor(nll.numel(), dtype=torch.long, device=device),
        n_rows=torch.tensor(n_rows, dtype=torch.long, device=device),
        n_short=short.sum(),
        short_tok=row_cnt[short].sum(),
        short_mean_sum=row_mean.detach()[short].sum(),
        row_w_sum=row_weights(row_cnt, cap).sum(),
        grader_mean_sum=row_mean.detach().sum(),
    )


def short_rows_correct(logits: torch.Tensor, targets: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    
    if targets.numel() == 0:
        return torch.zeros((), dtype=torch.long, device=targets.device)
    uniq, inv = torch.unique(seg, return_inverse=True)
    n = int(uniq.numel())
    ok = (logits.argmax(dim=-1) == targets).to(torch.long)
    cnt = torch.zeros(n, dtype=torch.long, device=targets.device).index_add(0, inv, torch.ones_like(inv))
    good = torch.zeros(n, dtype=torch.long, device=targets.device).index_add(0, inv, ok)
    return ((cnt <= SHORT_ROW_TOKENS) & (good == cnt)).sum()


def _unwrap(model):
    m = model
    while hasattr(m, "module") and isinstance(getattr(m, "module"), nn.Module):
        m = m.module
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    return m


def _softcap_of(model) -> Optional[float]:
    cfg = getattr(model, "config", None)
    val = getattr(cfg, "final_logit_softcapping", None)
    if val is None and cfg is not None and hasattr(cfg, "get_text_config"):
        try:
            val = getattr(cfg.get_text_config(), "final_logit_softcapping", None)
        except Exception:
            val = None
    return float(val) if val else None


class _HeadCapture:
    

    def __init__(self, head: nn.Module, pass_through: bool):
        self.head = head
        self.pass_through = pass_through
        self.hidden: Optional[torch.Tensor] = None
        self.calls = 0
        self._had_instance_forward = "forward" in head.__dict__
        self._prev = head.__dict__.get("forward")

    def __enter__(self):
        orig = self.head.forward

        def capture(x, *args, **kwargs):
            self.hidden = x
            self.calls += 1
            if self.pass_through:
                return orig(x, *args, **kwargs)
            return F.linear(x, self.head.weight[:0])

        self.head.forward = capture
        return self

    def __exit__(self, *exc):
        if self._had_instance_forward:
            self.head.forward = self._prev
        else:
            del self.head.forward
        return False


def _sharded_params(model) -> bool:
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
        if is_deepspeed_zero3_enabled():
            return True
    except Exception:
        pass
    m = model
    while m is not None:
        if type(m).__name__.startswith("FullyShardedDataParallel") or type(m).__name__.startswith("FSDP"):
            return True
        m = getattr(m, "module", None) if isinstance(getattr(m, "module", None), nn.Module) else None
    return False


def _forward_inputs(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k not in ("labels", "shift_labels", "num_items_in_batch")}


def resolve_mode(model, inputs: dict) -> tuple[str, dict]:
    
    base = _unwrap(model)
    cached = getattr(base, "_sn56_rowloss_mode", None)
    if cached is not None:
        return cached, {"cached": True}
    diag: dict = {"cached": False}
    head = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
    if head is None or type(head) is not nn.Linear:
        mode, diag["reason"] = MODE_LOGITS, f"head_type={type(head).__name__}"
    elif _sharded_params(model):
        mode, diag["reason"] = MODE_LOGITS, "sharded_params"
    else:
        one = {k: (v[:1] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == inputs["input_ids"].shape[0] else v)
               for k, v in _forward_inputs(inputs).items()}
        try:
            with torch.no_grad(), _HeadCapture(head, pass_through=True) as cap:
                out = base(**one)
            logits = getattr(out, "logits", None)
            if cap.calls != 1 or cap.hidden is None or logits is None:
                mode, diag["reason"] = MODE_LOGITS, f"head_calls={cap.calls}"
            else:
                ref = F.linear(cap.hidden.to(head.weight.dtype), head.weight, head.bias).float()
                sc = _softcap_of(base)
                if sc is not None:
                    ref = sc * torch.tanh(ref / sc)
                got = logits.float()
                if got.shape != ref.shape:
                    mode, diag["reason"] = MODE_LOGITS, "shape_mismatch"
                else:
                    err = float((got - ref).abs().max().item())
                    scale = float(ref.abs().max().item()) + 1.0
                    diag["max_abs_err"] = err
                    ok = err <= 1e-2 * scale
                    mode = MODE_HIDDEN if ok else MODE_LOGITS
                    diag["reason"] = "probe_match" if ok else "probe_mismatch"
        except Exception as exc:
            mode, diag["reason"] = MODE_LOGITS, f"probe_error:{type(exc).__name__}:{str(exc)[:160]}"
    try:
        base._sn56_rowloss_mode = mode
    except Exception:
        pass
    return mode, diag


def compute_row_terms(model, inputs: dict, mode: str, cap: float = NO_CAP):
    
    labels = inputs["labels"]
    position_ids = inputs.get("position_ids")
    attention_mask = inputs.get("attention_mask")
    fwd = _forward_inputs(inputs)
    tgt = shifted_targets(labels)
    seg = segment_ids(labels, position_ids, attention_mask)
    sel = tgt != IGNORE_INDEX
    flat_idx = sel.reshape(-1).nonzero(as_tuple=True)[0]
    tgt_sel = tgt.reshape(-1)[flat_idx]
    seg_sel = seg.reshape(-1)[flat_idx]

    if mode == MODE_HIDDEN:
        base = _unwrap(model)
        head = base.get_output_embeddings()
        with _HeadCapture(head, pass_through=False) as hc:
            outputs = model(**fwd)
        hidden = hc.hidden
        if hidden is None or hidden.shape[:2] != labels.shape:
            raise RuntimeError(
                f"[sn56][rowloss] lm_head capture failed (calls={hc.calls}, "
                f"shape={None if hidden is None else tuple(hidden.shape)})")
        h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, flat_idx)
        nll = nll_from_hidden(h, head.weight, head.bias, tgt_sel, _softcap_of(base))
        if nll.numel() == 0:
            terms = reduce_rows(nll, seg_sel, cap)
            z = hidden.float().sum() * 0.0
            terms = terms._replace(tok_sum=terms.tok_sum + z, row_mean_sum=terms.row_mean_sum + z)
            return terms, outputs
    else:
        outputs = model(**fwd)
        logits = outputs.logits
        lg = logits.reshape(-1, logits.shape[-1]).index_select(0, flat_idx)
        nll = nll_from_logits(lg, tgt_sel)
        if nll.numel() == 0:
            terms = reduce_rows(nll, seg_sel, cap)
            z = logits.float().sum() * 0.0
            terms = terms._replace(tok_sum=terms.tok_sum + z, row_mean_sum=terms.row_mean_sum + z)
            return terms, outputs
    return reduce_rows(nll, seg_sel, cap), outputs


def _aux_term(model, outputs) -> Optional[torch.Tensor]:
    aux = getattr(outputs, "aux_loss", None)
    if aux is None:
        return None
    coef = getattr(getattr(_unwrap(model), "config", None), "router_aux_loss_coef", 0.0) or 0.0
    return coef * aux


def blend(alpha: float, micro: torch.Tensor, macro: torch.Tensor) -> torch.Tensor:
    if alpha <= 0.0:
        return micro
    if alpha >= 1.0:
        return macro
    return (1.0 - alpha) * micro + alpha * macro


def make_probe_loss_fn(alpha: float, cap: Optional[float] = None) -> Callable[[Any, dict], torch.Tensor]:
    

    cap_v = row_loss_cap() if cap is None else float(cap)
    state = {"logged": False}

    def probe_loss(model, batch: dict, norm_batch: Optional[dict] = None) -> torch.Tensor:
        mode, _ = resolve_mode(model, batch)
        terms, outputs = compute_row_terms(model, batch, mode, cap_v)
        if norm_batch is None:
            n_tok = terms.n_tok.clamp_min(1).to(terms.tok_sum.dtype)
            row_w = terms.row_w_sum.clamp_min(1.0).to(terms.row_mean_sum.dtype)
        else:
            n_tok = shifted_targets(norm_batch["labels"]).ne(IGNORE_INDEX).sum().clamp_min(1).to(
                terms.tok_sum.dtype)
            row_w = torch.tensor(window_row_weight(norm_batch, cap_v), dtype=torch.float64,
                                 device=terms.row_mean_sum.device).clamp_min(1.0).to(
                terms.row_mean_sum.dtype)
        micro = terms.tok_sum / n_tok
        aux = _aux_term(model, outputs)
        if aux is not None:
            micro = micro + aux
        macro = terms.row_mean_sum / row_w
        if not state["logged"]:
            state["logged"] = True
            log_info(f"[sn56][rowloss] busca LR com objetivo por linha alpha={alpha} cap={cap_v} modo={mode} "
                     f"micro={float(micro.detach()):.4f} macro={float(macro.detach()):.4f}")
        return blend(alpha, micro, macro)

    return probe_loss


class DevSummary(BaseModel):
    
    dev_macro: float
    dev_micro: float
    dev_n_rows: int
    dev_mc: Optional[float]
    dev_mc_acc: Optional[float]
    dev_mc_n: int


class TrainLossSnapshot(BaseModel):
    
    step: int
    alpha: float
    cap: Optional[float]
    mode: str
    micro: Optional[float]
    macro: Optional[float]
    macro_grader: Optional[float]
    row_w_sum: float
    n_tok: int
    n_rows: int
    short_rows: int
    short_grad_share: Optional[float]
    micro_batches: int


class _Window:
    
    __slots__ = ("rows", "remaining", "world")

    def __init__(self, rows: torch.Tensor, remaining: int, world: int):
        self.rows = rows
        self.remaining = remaining
        self.world = world


class RowWeightedLossMixin:
    

    def __init__(self, *args, row_weight_alpha_value: Optional[float] = None,
                 row_loss_cap_value: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_weight_alpha = (row_weight_alpha() if row_weight_alpha_value is None
                                 else max(0.0, min(1.0, float(row_weight_alpha_value))))
        self.row_loss_cap = (row_loss_cap() if row_loss_cap_value is None
                             else (float(row_loss_cap_value) if float(row_loss_cap_value) > 0.0 else NO_CAP))
        self._rw_window: Optional[_Window] = None
        self._rw_mode: Optional[str] = None
        self._rw_acc: Optional[torch.Tensor] = None
        self._rw_dev: Optional[torch.Tensor] = None
        self._rw_warned_stock = False
        log_info(f"[sn56][rowloss] objetivo por linha ativo alpha={self.row_weight_alpha} cap={self.row_loss_cap}")


    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples, num_items_in_batch = super().get_batch_samples(epoch_iterator, num_batches, device)
        rows = sum(window_row_weight(b, self.row_loss_cap) for b in batch_samples if isinstance(b, dict))
        rows_t = torch.tensor(rows, dtype=torch.float64, device=device)
        world = int(getattr(self.accelerator, "num_processes", 1) or 1)
        if world > 1:
            rows_t = self.accelerator.gather(rows_t).sum()
        self._rw_window = _Window(rows=rows_t, remaining=len(batch_samples), world=world)
        return batch_samples, num_items_in_batch


    def _use_stock_train_loss(self) -> Optional[str]:
        if getattr(self, "label_smoother", None) is not None:
            return "label_smoothing"
        if getattr(self, "compute_loss_func", None) is not None:
            return "compute_loss_func"
        return None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not model.training or "labels" not in inputs:
            return self._eval_compute_loss(model, inputs, return_outputs, num_items_in_batch)
        why = self._use_stock_train_loss()
        if why is not None:
            if not self._rw_warned_stock:
                self._rw_warned_stock = True
                log_info(f"[sn56][rowloss] caindo pro loss padrão ({why})")
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

        alpha = self.row_weight_alpha
        cap = self.row_loss_cap
        if self._rw_mode is None:
            self._rw_mode, diag = resolve_mode(model, inputs)
            log_info(f"[sn56][rowloss] modo={self._rw_mode} ({diag.get('reason')})")
        terms, outputs = compute_row_terms(model, inputs, self._rw_mode, cap)
        dtype = terms.tok_sum.dtype

        accepts = bool(getattr(self, "model_accepts_loss_kwargs", False))
        divides = not accepts or num_items_in_batch is None
        stock_mult = 1
        if getattr(self.args, "average_tokens_across_devices", False) and accepts and num_items_in_batch is not None:
            stock_mult = int(getattr(self.accelerator, "num_processes", 1) or 1)
            pc = getattr(self.accelerator, "parallelism_config", None)
            if pc is not None:
                stock_mult //= pc.tp_size

        if accepts and num_items_in_batch is not None:
            denom = num_items_in_batch
            if torch.is_tensor(denom):
                denom = denom.to(terms.tok_sum.device).to(dtype)
            micro = terms.tok_sum / denom
        else:
            micro = terms.tok_sum / terms.n_tok.clamp_min(1).to(dtype)
        aux = _aux_term(model, outputs)
        if aux is not None:
            micro = micro + aux
        loss_micro = micro * stock_mult if stock_mult != 1 else micro

        if alpha <= 0.0:
            loss = loss_micro
        else:
            win = self._rw_window if (self._rw_window is not None and self._rw_window.remaining > 0) else None
            if win is not None:
                macro = terms.row_mean_sum / win.rows.clamp_min(1.0).to(dtype)
                scale = float(win.world)
                if divides:
                    scale *= float(getattr(self, "current_gradient_accumulation_steps", 1) or 1)
                macro = macro * scale
            else:
                macro = terms.row_mean_sum / terms.row_w_sum.clamp_min(1.0).to(dtype)
            loss = (1.0 - alpha) * loss_micro + alpha * macro
        if self._rw_window is not None and self._rw_window.remaining > 0:
            self._rw_window.remaining -= 1

        self._rw_accumulate(terms, alpha)
        return (loss, outputs) if return_outputs else loss


    def _rw_accumulate(self, terms: RowTerms, alpha: float) -> None:
        with torch.no_grad():
            n_tok = terms.n_tok.to(torch.float64)
            n_rows = terms.n_rows.to(torch.float64)
            row_w = terms.row_w_sum.to(torch.float64)
            share = ((1.0 - alpha) * terms.short_tok.to(torch.float64) / n_tok.clamp_min(1)
                     + alpha * terms.n_short.to(torch.float64) / row_w.clamp_min(1.0))
            vec = torch.stack([
                terms.tok_sum.detach().to(torch.float64), n_tok,
                terms.row_mean_sum.detach().to(torch.float64), n_rows,
                terms.n_short.to(torch.float64), share,
                torch.ones((), dtype=torch.float64, device=n_tok.device),
                row_w, terms.grader_mean_sum.to(torch.float64),
            ])
            self._rw_acc = vec if self._rw_acc is None else self._rw_acc + vec

    def log(self, logs, *args, **kwargs):
        if self._rw_acc is not None and isinstance(logs, dict) and "loss" in logs:
            (tok_sum, n_tok, row_sum, n_rows, n_short, share_sum, n_mb,
             row_w, grader_sum) = self._rw_acc.tolist()
            self._rw_acc = None
            snap = TrainLossSnapshot(
                step=int(getattr(self.state, "global_step", 0) or 0),
                alpha=self.row_weight_alpha,
                cap=cap_for_record(self.row_loss_cap),
                mode=self._rw_mode or "?",
                micro=(tok_sum / n_tok) if n_tok > 0 else None,
                macro=(row_sum / row_w) if row_w > 0 else None,
                macro_grader=(grader_sum / n_rows) if n_rows > 0 else None,
                row_w_sum=row_w,
                n_tok=int(n_tok), n_rows=int(n_rows), short_rows=int(n_short),
                short_grad_share=(share_sum / n_mb) if n_mb > 0 else None,
                micro_batches=int(n_mb),
            )
            if snap.micro is not None:
                logs["loss_micro"] = round(snap.micro, 6)
            if snap.macro is not None:
                logs["loss_macro"] = round(snap.macro, 6)
            if snap.macro_grader is not None:
                logs["loss_macro_grader"] = round(snap.macro_grader, 6)
        return super().log(logs, *args, **kwargs)


    def _eval_compute_loss(self, model, inputs, return_outputs, num_items_in_batch):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        try:
            logits = getattr(outputs, "logits", None)
            labels = inputs.get("labels")
            if logits is not None and labels is not None and logits.shape[:2] == labels.shape:
                with torch.no_grad():
                    tgt = shifted_targets(labels)
                    seg = segment_ids(labels, inputs.get("position_ids"), inputs.get("attention_mask"))
                    sel = tgt != IGNORE_INDEX
                    idx = sel.reshape(-1).nonzero(as_tuple=True)[0]
                    lg = logits.reshape(-1, logits.shape[-1]).index_select(0, idx)
                    tgt_sel, seg_sel = tgt.reshape(-1)[idx], seg.reshape(-1)[idx]
                    terms = reduce_rows(nll_from_logits(lg, tgt_sel), seg_sel, NO_CAP)
                    n_short_ok = short_rows_correct(lg, tgt_sel, seg_sel)
                    vec = torch.stack([
                        terms.tok_sum.to(torch.float64), terms.n_tok.to(torch.float64),
                        terms.row_mean_sum.to(torch.float64), terms.n_rows.to(torch.float64),
                        terms.n_short.to(torch.float64), terms.short_mean_sum.to(torch.float64),
                        n_short_ok.to(torch.float64),
                    ])
                    self._rw_dev = vec if self._rw_dev is None else self._rw_dev + vec
        except Exception:
            pass
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        self._rw_dev = None
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self._rw_dev = None

    def evaluation_loop(self, *args, **kwargs):
        
        output = super().evaluation_loop(*args, **kwargs)
        prefix = kwargs.get("metric_key_prefix", args[4] if len(args) > 4 else "eval")
        try:
            dev = self._rw_dev
            device = self.args.device
            if dev is None:
                dev = torch.zeros(7, dtype=torch.float64, device=device)
            dev = dev.to(device)
            if torch.distributed.is_available() and torch.distributed.is_initialized() \
                    and torch.distributed.get_world_size() > 1:
                torch.distributed.all_reduce(dev)
            tok_sum, n_tok, row_sum, n_rows, n_short, short_sum, short_ok = dev.tolist()
            metrics = output.metrics
            if n_rows > 0 and isinstance(metrics, dict):
                summary = DevSummary(
                    dev_macro=row_sum / n_rows,
                    dev_micro=tok_sum / max(n_tok, 1.0),
                    dev_n_rows=int(n_rows),
                    dev_mc=(short_sum / n_short) if n_short > 0 else None,
                    dev_mc_acc=(short_ok / n_short) if n_short > 0 else None,
                    dev_mc_n=int(n_short),
                )
                for k, v in summary.model_dump().items():
                    if v is not None:
                        metrics[f"{prefix}_{k}"] = v
        except Exception:
            pass
        self._rw_dev = None
        return output


class RowWeightedTrainer(RowWeightedLossMixin, Trainer):
    pass
