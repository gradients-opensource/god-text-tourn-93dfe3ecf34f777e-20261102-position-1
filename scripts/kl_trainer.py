
import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer

from utility import log_info


def load_base_model(
    model_path: str,
    attn_implementation: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> transformers.PreTrainedModel:
    
    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    base.to(device)
    base.eval()
    try:
        base.config.use_cache = False
    except Exception:
        pass
    for p in base.parameters():
        p.requires_grad_(False)
    return base


class KLRegularizedTrainer(Trainer):
    

    def __init__(self, *args, kl_coef: float = 0.0, base_model=None, use_lora_base: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coef = kl_coef
        self.base_model = base_model
        self.use_lora_base = use_lora_base
        self._kl_logged = False
        self.model_accepts_loss_kwargs = False

    def _kl_active(self) -> bool:
        return bool(self.kl_coef) and (self.use_lora_base or self.base_model is not None)

    def _base_logits(self, input_ids, attention_mask, position_ids=None, seq_idx=None):
        
        extra = {"seq_idx": seq_idx} if seq_idx is not None else {}
        with torch.no_grad():
            if self.use_lora_base:
                with self.model.disable_adapter():
                    return self.model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, **extra,
                    ).logits
            if self.base_model.device != input_ids.device:
                self.base_model.to(input_ids.device)
            return self.base_model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, **extra,
            ).logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]
        position_ids = inputs.get("position_ids")
        seq_idx = inputs.get("seq_idx")
        extra = {"seq_idx": seq_idx} if seq_idx is not None else {}

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            **extra,
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=getattr(self.args, "label_smoothing_factor", 0.0) or 0.0,
        )

        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            ce_loss = ce_loss + getattr(self.model.config, "router_aux_loss_coef", 0.0) * aux_loss

        if self._kl_active():
            mask = labels != -100
            if mask.any():
                sel_ft = logits[mask]
                base_logits = self._base_logits(input_ids, attention_mask, position_ids)
                sel_base = base_logits[mask]

                ft_log_probs = F.log_softmax(sel_ft, dim=-1)
                base_log_probs = F.log_softmax(sel_base, dim=-1)
                kl_per_token = (ft_log_probs.exp() * (ft_log_probs - base_log_probs)).sum(dim=-1)
                kl = kl_per_token.mean()
            else:
                kl = logits.new_zeros(())

            loss = ce_loss + self.kl_coef * kl
            if not self._kl_logged:
                log_info(
                    f"[sn56][kl] ativo kl_coef={self.kl_coef}, "
                    f"fonte={'lora' if self.use_lora_base else 'copia'}, "
                    f"ce={ce_loss.item():.4f}, kl={float(kl):.4f}, loss={loss.item():.4f}"
                )
                self._kl_logged = True
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss
