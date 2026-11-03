import copy
import functools
from typing import Iterable, Optional

import torch
from pydantic import BaseModel
from transformers import AutoConfig


PAD_FALLBACK_ROLES = ("eos", "unk")
SPECIAL_ROLES = ("pad", "bos", "eos", "unk")
_ORIGINAL_PAD_ATTR = "_sn56_original_pad_token"


class SpecialToken(BaseModel):
    role: str
    token: Optional[str]
    id: Optional[int]
    in_range: bool


class TokenBoundsFix(BaseModel):
    site: str
    n_emb: int
    n_tokenizer: int
    specials: list[SpecialToken]
    out_of_range: list[str]
    pad_remapped: bool = False
    pad_from: Optional[str] = None
    pad_from_id: Optional[int] = None
    pad_to: Optional[str] = None
    pad_to_id: Optional[int] = None
    pad_to_role: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.pad_remapped


class TokenIdOutOfRange(ValueError):
    pass


def embedding_rows(model) -> Optional[int]:
    try:
        emb = model.get_input_embeddings()
    except Exception:
        return None
    n = getattr(emb, "num_embeddings", None)
    if not isinstance(n, int) or n <= 0:
        w = getattr(emb, "weight", None)
        n = int(w.shape[0]) if w is not None and w.dim() == 2 and w.shape[0] > 0 else None
    return n


def config_embedding_rows(model_path: str, resize_to: Optional[int] = None) -> Optional[int]:
    if resize_to is not None:
        return int(resize_to)
    try:
        cfg = AutoConfig.from_pretrained(model_path)
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        n = getattr(text_cfg, "vocab_size", None)
        return int(n) if isinstance(n, int) and n > 0 else None
    except Exception:
        return None


def _role_id(tokenizer, role: str) -> Optional[int]:
    tid = getattr(tokenizer, f"{role}_token_id", None)
    if isinstance(tid, (list, tuple)):
        tid = tid[0] if tid else None
    return int(tid) if isinstance(tid, int) and tid >= 0 else None


def _save_with_original_pad(tokenizer, *args, **kwargs):
    twin = copy.deepcopy(tokenizer)
    twin.__dict__.pop("save_pretrained", None)
    twin.pad_token = getattr(tokenizer, _ORIGINAL_PAD_ATTR)
    return twin.save_pretrained(*args, **kwargs)


def _install_save_original(tokenizer, original_pad: str) -> None:
    if getattr(tokenizer, _ORIGINAL_PAD_ATTR, None) is None:
        setattr(tokenizer, _ORIGINAL_PAD_ATTR, original_pad)
        tokenizer.save_pretrained = functools.partial(_save_with_original_pad, tokenizer)


def fit_special_tokens(tokenizer, n_emb: Optional[int], site: str) -> Optional[TokenBoundsFix]:
    if tokenizer is None or not isinstance(n_emb, int) or n_emb <= 0:
        return None
    specials = []
    for role in SPECIAL_ROLES:
        tid = _role_id(tokenizer, role)
        specials.append(SpecialToken(role=role, token=getattr(tokenizer, f"{role}_token", None),
                                     id=tid, in_range=tid is None or tid < n_emb))
    rep = TokenBoundsFix(site=site, n_emb=int(n_emb), n_tokenizer=len(tokenizer),
                         specials=specials,
                         out_of_range=[s.role for s in specials if not s.in_range])
    if not rep.out_of_range:
        return rep
    by_role = {s.role: s for s in specials}
    pad = by_role["pad"]
    if not pad.in_range:
        target = next((by_role[r] for r in PAD_FALLBACK_ROLES
                       if by_role[r].id is not None and by_role[r].in_range), None)
        if target is None:
            raise TokenIdOutOfRange(
                f"[sn56][token_bounds] pad token {pad.token!r} id {pad.id} >= embedding rows "
                f"{n_emb} ({site}) and no in-range fallback ({', '.join(PAD_FALLBACK_ROLES)}); "
                f"the first forward would index past the embedding")
        _install_save_original(tokenizer, pad.token)
        tokenizer.pad_token = target.token
        if _role_id(tokenizer, "pad") != target.id:
            raise TokenIdOutOfRange(
                f"[sn56][token_bounds] remapping pad to {target.token!r} gave id "
                f"{_role_id(tokenizer, 'pad')}, expected {target.id} ({site})")
        rep.pad_remapped = True
        rep.pad_from, rep.pad_from_id = pad.token, pad.id
        rep.pad_to, rep.pad_to_id, rep.pad_to_role = target.token, target.id, target.role
    print(f"[sn56][token_bounds] {site}: embedding rows {n_emb}, tokenizer {len(tokenizer)}; "
          f"out of range: {rep.out_of_range}"
          + (f"; pad {rep.pad_from!r}({rep.pad_from_id}) -> {rep.pad_to!r}({rep.pad_to_id}) "
             f"for training (saved tokenizer keeps {rep.pad_from!r})" if rep.pad_remapped else ""),
          flush=True)
    return rep


def check_ids_in_range(ids, n_emb: Optional[int], where: str) -> None:
    if not isinstance(n_emb, int) or n_emb <= 0 or ids is None:
        return
    if torch.is_tensor(ids):
        if ids.numel() == 0:
            return
        hi, lo = int(ids.max()), int(ids.min())
    else:
        hi, lo = _scan_max_min(ids)
        if hi is None:
            return
    if hi >= n_emb or lo < 0:
        bad = hi if hi >= n_emb else lo
        raise TokenIdOutOfRange(
            f"[sn56][token_bounds] token id {bad} outside the input embedding [0, {n_emb}) "
            f"in {where}; it would crash the embedding lookup (CUDA vectorized_gather_kernel "
            f"assert). Check the tokenizer's added/special tokens against the checkpoint's vocab.")


def _scan_max_min(rows: Iterable):
    hi = lo = None
    for r in rows:
        if isinstance(r, int):
            r = (r,)
        if not r:
            continue
        a, b = max(r), min(r)
        hi = a if hi is None or a > hi else hi
        lo = b if lo is None or b < lo else lo
    return hi, lo


def check_rows_in_range(rows: Iterable[dict], n_emb: Optional[int], where: str,
                        key: str = "input_ids") -> None:
    if rows is None:
        return
    check_ids_in_range((dp[key] for dp in rows if key in dp), n_emb, where)


def check_batch_in_range(batch, n_emb: Optional[int], where: str) -> None:
    if not isinstance(n_emb, int) or n_emb <= 0 or not isinstance(batch, dict):
        return
    for k, v in batch.items():
        if (k == "input_ids" or k.endswith("_ids")) and k != "position_ids" and torch.is_tensor(v) \
                and not v.is_floating_point():
            check_ids_in_range(v, n_emb, f"{where}[{k}]")


class GuardedCollator:

    def __init__(self, inner, n_emb: int, where: str):
        self.inner = inner
        self.n_emb = int(n_emb)
        self.where = where

    def __call__(self, features):
        batch = self.inner(features)
        check_batch_in_range(batch, self.n_emb, self.where)
        return batch

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def reconcile(model, tokenizer, collator, expected: Optional[int], site: str) -> Optional[int]:
    n = embedding_rows(model)
    if n is None:
        return expected
    if n != expected:
        fit_special_tokens(tokenizer, n, site)
        if collator is not None:
            if hasattr(collator, "pad_token_id") and tokenizer.pad_token_id is not None:
                collator.pad_token_id = int(tokenizer.pad_token_id)
            if hasattr(collator, "vocab_limit"):
                collator.vocab_limit = n
    return n


def guard_trainer_collator(trainer, where: str, n_emb: Optional[int] = None) -> Optional[int]:
    if n_emb is None:
        n_emb = embedding_rows(getattr(trainer, "model", None))
    inner = getattr(trainer, "data_collator", None)
    if n_emb is None or inner is None or isinstance(inner, GuardedCollator):
        return n_emb
    trainer.data_collator = GuardedCollator(inner, n_emb, where)
    return n_emb
