
import types

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from transformers.modeling_outputs import CausalLMOutputWithPast

GRANITE_MODEL_TYPE = "granite"


def granite_flce_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                         past_key_values=None, inputs_embeds=None, labels=None,
                         use_cache=None, logits_to_keep=0, **kwargs):
    
    shift_labels = kwargs.pop("shift_labels", None)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept = hidden_states[:, slice_indices, :]

    logits = None
    loss = None
    if self.training and (labels is not None or shift_labels is not None):
        result = LigerForCausalLMLoss(
            hidden_states=kept / self.config.logits_scaling,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss = unpack_cross_entropy_result(result)[0]
    else:
        logits = self.lm_head(kept) / self.config.logits_scaling
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, shift_labels=shift_labels,
                                      vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def wants_granite_flce(model, use_liger: bool, zero3: bool) -> bool:
    
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return bool(use_liger) and model_type == GRANITE_MODEL_TYPE and not zero3


def enable_granite_flce(model):
    
    model.forward = types.MethodType(granite_flce_forward, model)
    return model
