import os
from typing import Callable, Optional

import torch
from pydantic import BaseModel

import batch_policy

ENV_ON = "SN56_CLOCK_HYGIENE"
A0_LOGGING_STEPS = 25


def enabled() -> bool:
    return os.environ.get(ENV_ON, "1").strip().lower() not in ("0", "false", "no", "off")


class GcProbe(BaseModel):
    gc_off: bool
    reason: str
    micro: int
    seq_len: int
    peak_gb: Optional[float] = None
    budget_gb: Optional[float] = None
    total_gb: Optional[float] = None
    optimizer: Optional[str] = None


def gc_off_fits(model, micro: int, seq_len: int, vocab: int, pad_id: int,
                loss_fn: Optional[Callable], with_position_ids: bool, training_args,
                headroom_frac: Optional[float] = None) -> GcProbe:
    micro, seq_len = max(1, int(micro)), max(1, int(seq_len))
    device = next(model.parameters()).device
    if device.type != "cuda":
        return GcProbe(gc_off=False, reason="not_cuda", micro=micro, seq_len=seq_len)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    hr = batch_policy.headroom() if headroom_frac is None else headroom_frac
    total = torch.cuda.get_device_properties(device).total_memory
    budget = total * hr - batch_policy.trainable_bytes(model)
    optimizer, scheduler, opt_name = batch_policy.build_probe_optimizer(model, training_args)
    if optimizer is None:
        return GcProbe(gc_off=False, reason=f"no_probe_optimizer:{opt_name}", micro=micro,
                       seq_len=seq_len, budget_gb=round(budget / 1e9, 3), total_gb=round(total / 1e9, 3))
    max_grad_norm = float(getattr(training_args, "max_grad_norm", 0.0) or 0.0)
    was_training = model.training
    model.train()
    batch = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        batch = batch_policy._synthetic_batch(micro, seq_len, vocab, device, pad_id, with_position_ids)
        batch_policy._fwd_bwd(model, batch, loss_fn)
        batch_policy._opt_step(model, optimizer, scheduler, max_grad_norm)
        torch.cuda.synchronize(device)
        peak = float(torch.cuda.max_memory_allocated(device))
        fits = peak <= budget
        return GcProbe(gc_off=fits, reason="fits" if fits else "over_budget", micro=micro,
                       seq_len=seq_len, peak_gb=round(peak / 1e9, 3), budget_gb=round(budget / 1e9, 3),
                       total_gb=round(total / 1e9, 3), optimizer=str(opt_name))
    except Exception as exc:
        reason = "oom" if batch_policy._is_oom(exc) else f"error:{type(exc).__name__}"
        return GcProbe(gc_off=False, reason=reason, micro=micro, seq_len=seq_len,
                       budget_gb=round(budget / 1e9, 3), total_gb=round(total / 1e9, 3),
                       optimizer=str(opt_name))
    finally:
        model.zero_grad(set_to_none=True)
        del batch
        batch_policy.release_probe_optimizer(model, optimizer, scheduler, True)
        if not was_training:
            model.eval()


def apply_gc_decision(model, training_args, probe: GcProbe) -> None:
    if probe.gc_off:
        training_args.gradient_checkpointing = False
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()


def a0_logging(training_args, keep_steps: bool = False) -> dict:
    saved = {"logging_steps": training_args.logging_steps,
             "logging_nan_inf_filter": training_args.logging_nan_inf_filter}
    if not keep_steps:
        training_args.logging_steps = A0_LOGGING_STEPS
    training_args.logging_nan_inf_filter = False
    return saved


def restore_trajectory_logging(training_args, saved: Optional[dict]) -> None:
    if not saved:
        return
    training_args.logging_steps = saved["logging_steps"]
    training_args.logging_nan_inf_filter = saved["logging_nan_inf_filter"]
