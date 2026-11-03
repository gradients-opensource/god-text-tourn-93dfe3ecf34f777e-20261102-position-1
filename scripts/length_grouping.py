from __future__ import annotations

import math
import os
from typing import Callable, Optional

import torch
from transformers.trainer_pt_utils import LengthGroupedSampler, get_length_grouped_indices

import release_defaults

_FALSY = ("0", "false", "no", "off")
MEGABATCH_MULT = 50
LENGTHS: Optional[list] = None
ACTIVE = False

PHASE: dict = {}


def dealcal_enabled() -> bool:
    return (release_defaults.raw("SN56_DEALCAL") or release_defaults.DEFAULTS["SN56_DEALCAL"]
            ).strip().lower() not in _FALSY


def dealias_enabled() -> bool:
    if not dealcal_enabled():
        return False
    return (release_defaults.raw("SN56_EVAL_PHASE_DEALIAS") or "0").strip().lower() not in _FALSY


def hf_mega_batch_mult(n: int, batch_size: int) -> int:
    m = min(int(n) // (max(1, int(batch_size)) * 4), MEGABATCH_MULT)
    return m if m > 0 else 1


def shuffle_batches_in_megabatches(indices: list, batch_size: int, megabatch_size: int,
                                   generator: torch.Generator) -> list:
    bs = max(1, int(batch_size))
    out: list = []
    for m0 in range(0, len(indices), megabatch_size):
        mega = indices[m0:m0 + megabatch_size]
        n_full = len(mega) // bs
        batches = [mega[k * bs:(k + 1) * bs] for k in range(n_full)]
        tail = mega[n_full * bs:]
        head = [batches.pop(0)] if (m0 == 0 and batches) else []
        if len(batches) > 1:
            perm = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[p] for p in perm]
        for b in head + batches:
            out.extend(b)
        out.extend(tail)
    return out


def _order_seed(indices: list) -> int:
    h = 1469598103934665603
    for i in indices[:256]:
        h = ((h ^ int(i)) * 1099511628211) & 0x7FFFFFFFFFFFFFFF
    return h


def update_positions(n_rows: int, rows_per_step: int, micro_rows: int, grad_accum: int,
                     mult: int) -> list:
    n_micro = int(math.ceil(int(n_rows) / max(1, int(micro_rows))))
    ga = max(1, int(grad_accum))
    upe = max(1, n_micro // ga + int(n_micro % ga > 0))
    m = max(1, int(mult))
    return [k % m for k in range(upe)]


def sampler_phase(step: Optional[int]) -> Optional[int]:
    if not PHASE or PHASE.get("dealias") or step is None or int(step) < 1:
        return None
    s = int(step)
    for start, pos in reversed(PHASE.get("epochs") or []):
        if start < s:
            k = s - 1 - start
            return int(pos[k]) if k < len(pos) else None
    return None


def sampler_period() -> Optional[int]:
    return int(PHASE["period"]) if PHASE.get("period") else None


class PhaseLengthGroupedSampler(LengthGroupedSampler):

    def __init__(self, batch_size: int, lengths: list, generator=None, *, micro_rows: int,
                 grad_accum: int, dealias: bool, step_now: Optional[Callable[[], int]] = None,
                 record: bool = True):
        super().__init__(int(batch_size), lengths=lengths, generator=generator)
        self.micro_rows = int(micro_rows)
        self.grad_accum = int(grad_accum)
        self.dealias = bool(dealias)
        self.step_now = step_now
        self.record = bool(record)
        self.mult = hf_mega_batch_mult(len(lengths), self.batch_size)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        if self.record:
            try:
                start = int(self.step_now()) if self.step_now is not None else None
            except Exception:
                start = None
            if start is not None:
                eps = [e for e in PHASE.setdefault("epochs", []) if e[0] != start]
                eps.append((start, update_positions(len(indices), self.batch_size,
                                                    self.micro_rows, self.grad_accum, self.mult)))
                PHASE["epochs"] = eps
        if self.dealias:
            g = torch.Generator()
            g.manual_seed(_order_seed(indices))
            indices = shuffle_batches_in_megabatches(indices, self.batch_size,
                                                     self.mult * self.batch_size, g)
        return iter(indices)


def enabled() -> bool:
    return (os.environ.get("SN56_BATCH_GROUP", "1") or "1").strip().lower() not in _FALSY


def when_mode() -> str:
    raw = (os.environ.get("SN56_BATCH_GROUP_WHEN", "fixed") or "fixed").strip().lower()
    return raw if raw in ("fixed", "always") else "fixed"


def repick_mode() -> str:
    raw = (os.environ.get("SN56_BATCH_GROUP_REPICK", "grouped") or "grouped").strip().lower()
    return raw if raw in ("grouped", "none") else "grouped"


def set_lengths(lengths) -> None:
    global LENGTHS
    LENGTHS = [int(x) for x in lengths] if lengths is not None else None


def activate(on: bool = True) -> None:
    global ACTIVE
    ACTIVE = bool(on)


def pad_up(n: int, multiple: int) -> int:
    m = max(1, int(multiple or 1))
    return int(math.ceil(max(1, int(n)) / m) * m)


def grouped_window(sorted_lengths: list, m: int, target: float) -> tuple[int, int]:
    n = len(sorted_lengths)
    m = max(1, int(m))
    if m >= n:
        return 0, n
    best, best_d = 0, None
    for s in range(0, n - m + 1):
        d = abs(float(sorted_lengths[s + m - 1]) - float(target))
        if best_d is None or d < best_d:
            best, best_d = s, d
    return best, best + m


def sampler_generator_enabled() -> bool:
    return os.environ.get("SN56_GROUP_SAMPLER_GEN", "0").strip().lower() not in ("0", "false", "no", "off")


def sampler_generator(seed: int):
    if not sampler_generator_enabled():
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def with_length_grouping(cls):
    try:
        from transformers import Trainer
        from transformers.trainer_pt_utils import LengthGroupedSampler
    except Exception:
        return cls
    if not issubclass(cls, Trainer):
        return cls
    if cls in _CLASSES:
        return _CLASSES[cls]

    class LengthGroupedMixin:
        def _get_train_sampler(self, train_dataset=None):
            ds = train_dataset if train_dataset is not None else getattr(self, "train_dataset", None)
            if ACTIVE and ds is not None:
                try:
                    n = len(ds)
                except TypeError:
                    n = None
                lengths = _lengths_for(ds, n)
                if lengths is not None:
                    bs = int(self.args.train_batch_size) * int(self.args.gradient_accumulation_steps)
                    gen = sampler_generator(int(getattr(self.args, "seed", 0) or 0))
                    if dealcal_enabled():
                        return _dealcal_sampler(self, bs, list(lengths), gen)
                    if gen is not None:
                        return LengthGroupedSampler(bs, lengths=list(lengths), generator=gen)
                    return LengthGroupedSampler(bs, lengths=list(lengths))
            return super()._get_train_sampler(train_dataset)

    _CLASSES[cls] = type(f"LengthGrouped{cls.__name__}", (LengthGroupedMixin, cls), {})
    return _CLASSES[cls]


_CLASSES: dict = {}


def _dealcal_sampler(trainer, bs: int, lengths: list, gen) -> "PhaseLengthGroupedSampler":
    world = int(getattr(trainer.args, "world_size", 1) or 1)
    dealias = dealias_enabled()
    smp = PhaseLengthGroupedSampler(
        bs, lengths, generator=gen, micro_rows=int(trainer.args.train_batch_size),
        grad_accum=int(trainer.args.gradient_accumulation_steps), dealias=dealias,
        step_now=lambda: int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0),
        record=(world == 1))
    PHASE.clear()
    PHASE.update(period=smp.mult, dealias=dealias, epochs=[])
    return smp


def _lengths_for(ds, n: Optional[int]) -> Optional[list]:
    if n is None:
        return None
    try:
        own = getattr(ds, "lengths", None)
        if own is not None and not callable(own) and len(own) == n:
            return [int(x) for x in own]
    except Exception:
        pass
    if LENGTHS is not None and len(LENGTHS) == n:
        return LENGTHS
    return None


def grouped_row_sources(dataset, collator: Callable, device, lengths: list, target_len: float,
                        max_rows: int, seed: int) -> tuple[Optional[Callable], Optional[Callable]]:
    import batch_policy as bp
    typ = bp.dataset_rows(dataset, collator, device, max_rows=max_rows, seed=seed,
                          lengths=lengths, grouped="typical", target_len=target_len)
    worst = bp.dataset_rows(dataset, collator, device, max_rows=max_rows, seed=seed,
                            lengths=lengths, grouped="worst", target_len=target_len)
    return typ, worst
