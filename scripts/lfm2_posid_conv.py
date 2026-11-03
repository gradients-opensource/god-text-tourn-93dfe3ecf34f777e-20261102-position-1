
import importlib
import json
import os
from dataclasses import dataclass

import torch
import transformers

_log = transformers.utils.logging.get_logger(__name__)

SEGCONV_TARGETS = {
    "lfm2": ("transformers.models.lfm2.modeling_lfm2", "Lfm2ShortConv"),
    "lfm2_moe": ("transformers.models.lfm2_moe.modeling_lfm2_moe", "Lfm2MoeShortConv"),
}
_MARK = "_sn56_segconv_orig"

NO_PACK_MODEL_TYPES = frozenset({"falcon_h1"})


def _norm(model_type) -> str:
    return str(model_type or "").strip().lower()


def needs_segconv(model_type) -> bool:
    return _norm(model_type) in SEGCONV_TARGETS


def refuses_packing(model_type) -> bool:
    return _norm(model_type) in NO_PACK_MODEL_TYPES


def packing_guard(model_type) -> str:
    
    if refuses_packing(model_type):
        return "no_pack"
    if needs_segconv(model_type):
        return "segconv"
    return "none"


_HYBRID_NAME_HINTS = (("falcon-h1", "falcon_h1"), ("falcon_h1", "falcon_h1"), ("lfm2", "lfm2"))


def _type_from_config_obj(cfg) -> str:
    for c in (cfg, getattr(cfg, "text_config", None)):
        mt = getattr(c, "model_type", None)
        if isinstance(mt, str) and mt:
            return mt
    return ""


def _type_from_config_json(model_path) -> str:
    try:
        with open(os.path.join(str(model_path), "config.json")) as f:
            raw = json.load(f)
    except Exception:
        return ""
    for c in (raw, raw.get("text_config") if isinstance(raw.get("text_config"), dict) else None):
        mt = c.get("model_type") if c else None
        if isinstance(mt, str) and mt:
            return mt
    return ""


@dataclass(frozen=True)
class GuardDecision:
    
    model_type: str
    source: str
    guard: str
    reason: str

    def fields(self) -> dict:
        return {"model_type": self.model_type, "source": self.source,
                "guard": self.guard, "reason": self.reason}


_REASONS = {"none": "attention_only", "segconv": "conv_state_leak", "no_pack": "ssm_state_leak"}


def resolve_guard(model_path, model_name: str = "") -> GuardDecision:
    
    mt, source = "", "unresolved"
    try:
        mt = _type_from_config_obj(transformers.AutoConfig.from_pretrained(model_path))
        source = "auto_config" if mt else source
    except Exception:
        mt = ""
    if not mt:
        mt = _type_from_config_json(model_path)
        source = "config_json" if mt else source
    if mt:
        guard = packing_guard(mt)
        return GuardDecision(mt, source, guard, _REASONS[guard])
    hay = f"{model_name} {model_path}".lower()
    for frag, family in _HYBRID_NAME_HINTS:
        if frag in hay:
            return GuardDecision(family, "name_hint", "no_pack", "model_type_unresolved_hybrid_name")
    return GuardDecision("", "unresolved", "none", "model_type_unresolved")


def model_type_of(model_path) -> str:
    try:
        cfg = transformers.AutoConfig.from_pretrained(model_path)
    except Exception:
        return _type_from_config_json(model_path)
    return _type_from_config_obj(cfg) or _type_from_config_json(model_path)


def segment_ids(position_ids: torch.Tensor) -> torch.Tensor:
    return position_ids.eq(0).cumsum(-1).sub(1).to(torch.int32)


def segment_offsets(seq_idx: torch.Tensor) -> torch.Tensor:
    L = seq_idx.shape[-1]
    t = torch.arange(L, device=seq_idx.device)
    starts = torch.ones_like(seq_idx, dtype=torch.bool)
    starts[..., 1:] = seq_idx[..., 1:] != seq_idx[..., :-1]
    start_pos = torch.where(starts, t, torch.zeros_like(t)).cummax(-1).values
    return t - start_pos


def boundary_safe_conv(conv: torch.nn.Conv1d, Bx: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
    L = Bx.shape[-1]
    full = conv(Bx)[..., :L]
    w = conv.weight.reshape(conv.weight.shape[0], -1)
    K = w.shape[-1]
    if K <= 1:
        return full
    offs = segment_offsets(seq_idx).unsqueeze(1)
    x = Bx.float()
    leak = torch.zeros_like(x)
    for k in range(1, K):
        prev = torch.nn.functional.pad(x, (k, 0))[..., :L]
        leak = leak + w[:, K - 1 - k].float().unsqueeze(-1) * prev * (offs < k).to(x.dtype)
    return (full.float() - leak).to(Bx.dtype)


def _takes_slow_path(mod, hidden_states) -> bool:
    fast = bool(getattr(mod, "is_fast_path_available", False))
    on_cuda = "cuda" in hidden_states.device.type
    return not (fast and on_cuda and not torch.compiler.is_compiling())


def _segconv_forward(mod, orig):
    def forward(self, hidden_states, past_key_values=None, attention_mask=None, seq_idx=None):
        slow = _takes_slow_path(mod, hidden_states)
        if seq_idx is not None and past_key_values is not None and slow:
            _log.warning_once("lfm2_posid_conv: seq_idx with a cache on the torch path is NOT "
                              "segment-isolated (falls through to the original forward)")
        if seq_idx is None or past_key_values is not None or not slow:
            extra = {"seq_idx": seq_idx} if seq_idx is not None else {}
            return orig(self, hidden_states, past_key_values=past_key_values,
                        attention_mask=attention_mask, **extra)
        x = mod.apply_mask_to_padding_states(hidden_states, attention_mask)
        B, C, x = self.in_proj(x).transpose(-1, -2).chunk(3, dim=-2)
        y = C * boundary_safe_conv(self.conv, B * x, seq_idx)
        return self.out_proj(y.transpose(-1, -2).contiguous())
    return forward


def _target(model_type):
    spec = SEGCONV_TARGETS.get(_norm(model_type))
    if spec is None:
        return None, None
    try:
        mod = importlib.import_module(spec[0])
        return mod, getattr(mod, spec[1])
    except Exception:
        return None, None


def patch_short_conv(model_type) -> bool:
    mod, cls = _target(model_type)
    if cls is None:
        return False
    if _MARK in cls.__dict__:
        return True
    setattr(cls, _MARK, cls.forward)
    cls.forward = _segconv_forward(mod, cls.forward)
    return True
