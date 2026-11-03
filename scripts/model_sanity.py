
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch


_LIVE_ROLES = {"bos", "eos", "unk"}
_CHUNK_ROWS = 8192


@dataclass
class MatrixSanity:
    name: str
    shape: list
    n_zero_rows: int = 0
    zero_row_ids: list = field(default_factory=list)
    zero_special: dict = field(default_factory=dict)
    n_nonfinite: int = 0
    skipped: Optional[str] = None


@dataclass
class ModelSanity:
    matrices: list = field(default_factory=list)
    n_zero_rows: int = 0
    n_nonfinite: int = 0
    zero_special: dict = field(default_factory=dict)
    tied: bool = False

    @property
    def ok(self) -> bool:
        
        hit = {k.rsplit(".", 1)[-1] for k in self.zero_special}
        return self.n_nonfinite == 0 and not (hit & _LIVE_ROLES)


def _special_ids(model, tokenizer) -> dict:
    
    out = {}
    cfg = getattr(model, "config", None)
    for role in ("bos", "eos", "pad", "unk"):
        tid = getattr(tokenizer, f"{role}_token_id", None) if tokenizer is not None else None
        if tid is None and cfg is not None:
            tid = getattr(cfg, f"{role}_token_id", None)
        if isinstance(tid, (list, tuple)):
            tid = tid[0] if tid else None
        if isinstance(tid, int) and tid >= 0:
            out[role] = tid
    return out


@torch.no_grad()
def _scan(name: str, weight, specials: dict) -> MatrixSanity:
    shape = list(getattr(weight, "shape", []))
    rep = MatrixSanity(name=name, shape=shape)
    if weight is None or weight.dim() != 2 or weight.numel() == 0:
        rep.skipped = "not_2d_or_empty"
        return rep
    if weight.device.type == "meta":
        rep.skipped = "meta"
        return rep
    zero_ids: list = []
    for start in range(0, weight.shape[0], _CHUNK_ROWS):
        blk = weight[start:start + _CHUNK_ROWS]
        rep.n_nonfinite += int((~torch.isfinite(blk)).sum())
        z = torch.nonzero((blk == 0).all(dim=1)).flatten()
        zero_ids.extend(int(i) + start for i in z.tolist())
    rep.n_zero_rows = len(zero_ids)
    rep.zero_row_ids = zero_ids[:32]
    zs = set(zero_ids)
    rep.zero_special = {role: tid for role, tid in specials.items() if tid in zs}
    return rep


def check_model(model, tokenizer=None) -> Optional[ModelSanity]:
    
    try:
        specials = _special_ids(model, tokenizer)
        emb = model.get_input_embeddings()
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        rep = ModelSanity()
        mats = [("embed_tokens", getattr(emb, "weight", None))]
        w_head = getattr(head, "weight", None) if head is not None else None
        rep.tied = w_head is not None and mats[0][1] is not None and \
            w_head.data_ptr() == mats[0][1].data_ptr()
        if w_head is not None and not rep.tied:
            mats.append(("lm_head", w_head))
        for name, w in mats:
            if w is None:
                continue
            m = _scan(name, w, specials)
            rep.matrices.append(m)
            rep.n_zero_rows += m.n_zero_rows
            rep.n_nonfinite += m.n_nonfinite
            rep.zero_special.update({f"{name}.{r}": t for r, t in m.zero_special.items()})
        if not rep.ok:
            sys.__stderr__.write(
                f"[sn56][model_sanity] WARNING: {rep.n_zero_rows} all-zero embedding/head "
                f"row(s), special tokens hit: {rep.zero_special or 'none'}; "
                f"{rep.n_nonfinite} non-finite value(s). A zero row for a token in every "
                f"sequence (BOS) gives an inf gradient through RMSNorm.\n")
            sys.__stderr__.flush()
        return rep
    except Exception:
        return None


@torch.no_grad()
def zero_row_ids(weight) -> list:
    if weight is None or weight.dim() != 2 or weight.numel() == 0 or weight.device.type == "meta":
        return []
    out: list = []
    for start in range(0, weight.shape[0], _CHUNK_ROWS):
        blk = weight[start:start + _CHUNK_ROWS]
        out.extend(int(i) + start for i in torch.nonzero((blk == 0).all(dim=1)).flatten().tolist())
    return out


def input_embedding_zero_rows(model) -> list:
    try:
        emb = model.get_input_embeddings()
    except Exception:
        return []
    return zero_row_ids(getattr(emb, "weight", None))
