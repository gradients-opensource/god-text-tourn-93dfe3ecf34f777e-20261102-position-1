import inspect
import torch
import transformers
import transformers.masking_utils as _hf_masking
import transformers.modeling_flash_attention_utils as _hf_fa
from transformers.utils.generic import is_flash_attention_requested
from typing import Optional
import sys
import random 
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple


class PackingIsolationError(Exception):
    pass


_ISOLATION_ACTIVE = False

_PROBE_NAME = "_is_packed_sequence"
_PROBE_PARAMS = ["position_ids", "batch_size"]


def _is_packed_sequence(position_ids, batch_size):
    
    if position_ids is None:
        if _ISOLATION_ACTIVE:
            raise PackingIsolationError(
                "packing-isolation-fix: flash attention was reached with position_ids=None "
                "while packing isolation is active, so the examples packed into this row "
                "would attend to each other. Either this architecture does not forward "
                "position_ids to its attention layers, or a caller dropped them from the "
                "batch (see kl_trainer.compute_loss)."
            )
        return False
    increasing_position_sequences = (
        torch.arange(position_ids.shape[1], device=position_ids.device) + position_ids.min()
    )
    packed = (increasing_position_sequences - position_ids).abs().sum().bool()
    return packed


def _fail(reason):
    raise PackingIsolationError(f"packing-isolation-fix: {reason}")


def _install_packing_isolation():
    
    stock = getattr(_hf_fa, _PROBE_NAME, None)
    if stock is None:
        _fail(f"transformers.modeling_flash_attention_utils has no {_PROBE_NAME}; the "
              "packed-sequence path moved — re-derive against this version")
    params = list(inspect.signature(stock).parameters)
    if params != _PROBE_PARAMS:
        _fail(f"{_PROBE_NAME} signature changed ({params} != {_PROBE_PARAMS}) — "
              "re-derive against this version")

    code = getattr(_hf_fa._flash_attention_forward, "__code__", None)
    if code is None or _PROBE_NAME not in code.co_names:
        _fail(f"_flash_attention_forward no longer resolves {_PROBE_NAME} at call time — "
              "the rebind would be a silent no-op")
    if "_prepare_from_posids" not in code.co_names:
        _fail("_flash_attention_forward no longer builds cu_seqlens from position_ids — "
              "the mechanism has moved")

    probe_positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0, 1, 2]])
    (cu_q, cu_k), _ = _hf_fa.prepare_fa_kwargs_from_position_ids(probe_positions)
    if cu_q.tolist() != [0, 4, 7, 10] or cu_k.tolist() != [0, 4, 7, 10]:
        _fail(f"prepare_fa_kwargs_from_position_ids gave {cu_q.tolist()} for a 4/3/3 pack, "
              "expected [0, 4, 7, 10] — the position-reset convention changed")
    if not bool(stock(probe_positions, 1)):
        _fail("upstream no longer recognises position resets as packing — re-derive")
    if bool(stock(torch.arange(10).unsqueeze(0), 1)):
        _fail("upstream now calls an unpacked row packed — re-derive")

    setattr(_hf_fa, _PROBE_NAME, _is_packed_sequence)
    if _hf_fa._flash_attention_forward.__globals__.get(_PROBE_NAME) is not _is_packed_sequence:
        _fail("_flash_attention_forward is built from a different module namespace — "
              "the rebind would be a silent no-op")
    if not bool(_hf_fa._is_packed_sequence(probe_positions.repeat(2, 1), 2)):
        _fail("the installed predicate still refuses a multi-row packed batch")

    ones = torch.ones(2, 10, dtype=torch.bool)
    if _hf_masking.flash_attention_mask(
        batch_size=2, q_length=10, kv_length=10, attention_mask=ones
    ) is not None:
        _fail("masking_utils.flash_attention_mask no longer drops an all-ones mask; flash "
              "attention would take the padding branch and cu_seqlens would be one segment "
              "per row")
    return stock


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        new_attention_mask = (attention_mask != 0).int().to(attention_mask.device)
        batch_size, sequence_length = new_attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        expert_attention_mask = (
            new_attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        router_per_expert_attention_mask = (
            new_attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def monkey_patch_packing_for_model(pretrained_model, attn_implementation):
    
    global _ISOLATION_ACTIVE
    model_config = transformers.AutoConfig.from_pretrained(pretrained_model)
    config_type = type(model_config).__name__.lower()
    if is_flash_attention_requested(requested_attention_implementation=attn_implementation):
        _install_packing_isolation()
        _ISOLATION_ACTIVE = True
    else:
        _ISOLATION_ACTIVE = False
    if config_type == "mixtralconfig":
        transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func = (
            load_balancing_loss_func
        )


def pack_data_points_naive(
    data_points: List[Dict], tokenizer: Any, model_max_length: int
) -> Dict:
    
    input_ids = []
    label_ids = []
    position_ids = []

    for item in data_points:
        seq_len = len(item["input_ids"])
        input_ids += item["input_ids"]
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        position_ids += list(range(seq_len))

    pad_leng = model_max_length - len(input_ids)

    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_leng
        label_ids = label_ids + [-100] * pad_leng
        attention_mask = [1] * (model_max_length - pad_leng) + [0] * pad_leng
        position_ids = position_ids + [0] * pad_leng
    else:
        input_ids = [tokenizer.pad_token_id] * pad_leng + input_ids
        label_ids = [-100] * pad_leng + label_ids
        attention_mask = [0] * pad_leng + [1] * (model_max_length - pad_leng)
        position_ids = [0] * pad_leng + position_ids

    assert len(input_ids) == len(label_ids) == len(attention_mask) == model_max_length
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.tensor(attention_mask),
        "position_ids": torch.tensor(position_ids),
    }


def pack_data_points_FA(
    data_points: List[Dict], tokenizer: Any, model_max_length: int, *, isolated: bool
) -> Dict:
    
    input_ids = []
    lengths = []
    label_ids = []
    attention_mask = []
    position_ids = []

    for index, item in enumerate(data_points):
        input_ids += item["input_ids"]
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))
        attention_mask += [1 if isolated else index + 1 for _ in range(len(item["input_ids"]))]
        position_ids += list(range(len(item["input_ids"])))

    pad_leng = model_max_length - len(input_ids)
    pad_positions = list(range(pad_leng))
    pad_mask = [1 if isolated else 0 for _ in range(pad_leng)]

    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
        attention_mask = attention_mask + pad_mask
        position_ids = position_ids + pad_positions
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids
        attention_mask = pad_mask + attention_mask
        position_ids = pad_positions + position_ids

    assert len(input_ids) == len(label_ids) == len(attention_mask) == model_max_length
    packed = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.tensor(
            attention_mask
        ),
    }
    if isolated:
        packed["position_ids"] = torch.tensor(position_ids)
    return packed


def pack_data_points_by_length(
    lengths: List[int], max_length: int, max_size: int = -1
) -> List[List[int]]:
    result = []
    current_concatenated_length = 0
    current_list = []
    for i in range(len(lengths)):
        cur_length = lengths[i]
        if cur_length + current_concatenated_length <= max_length and (
            max_size == -1 or len(current_list) < max_size
        ):
            current_concatenated_length += cur_length
            current_list.append(i)
        else:
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    assert sum([len(indices) for indices in result]) == len(lengths)
    return result


def merge_intervals(
    lengths: List[int], max_length: int, max_size: int = -1, min_item_num: int = -1
) -> List[List[int]]:
    result = []
    current_concatenated_length = 0
    current_list = []
    total_item_num = len(lengths)
    for i in range(len(lengths)):
        cur_length = lengths[i]
        max_item_after_merge = len(result) + 1 + (total_item_num - 1 - i)
        if cur_length + current_concatenated_length <= max_length and (
            max_size == -1 or len(current_list) < max_size
        ) and max_item_after_merge >= min_item_num:
            current_concatenated_length += cur_length
            current_list.append(i)
        else:
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    assert sum([len(indices) for indices in result]) == len(lengths)
    return result
    

def pack_with_min_item_num(
    lengths: List[int], max_length: int, min_item_num: int = -1
) -> List[List[int]]:
    if min_item_num == -1:
        print("Pack as much as we can as min_item_num is not set", flush=True)
        return pack_data_points_by_length(lengths, max_length, -1)
    
    greedy_packed_groups = pack_data_points_by_length(lengths, max_length, -1)
    if len(greedy_packed_groups) >= min_item_num:
        print(f"max as much as we can still > min_item_num={min_item_num}, size={len(greedy_packed_groups)}", flush=True)
        return greedy_packed_groups
    
    no_packed_groups = [[index] for index in range(len(lengths))]
    if len(no_packed_groups) <= min_item_num:
        print(f"number of original items ({len(lengths)}) is already smaller than: {min_item_num}", flush=True)
        return no_packed_groups
    
    for i in range(2, 51):
        groups = pack_data_points_by_length(lengths, max_length, max_size=i)
        print(f"number of items if max_size={i} is: {len(groups)}", flush=True)
        if len(groups) < min_item_num:
            result = merge_intervals(lengths, max_length, max_size=i, min_item_num=min_item_num)
            print(f"THE BEST MAX GROUP SIZE IN PACKING IS: {i}, size={len(groups)}; number of final groups: {len(result)}", flush=True)
            return result
    return groups


class PackedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Any,
        max_input_length: int,
        pack_length: int = -1,
        max_packed_size: int = -1,
        min_item_num: int = -1,
        seed: int = 42,
        use_fa: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.use_fa = use_fa
        self.isolated = bool(use_fa and _ISOLATION_ACTIVE)

        self.pack_length = pack_length
        if pack_length == -1:
            self.pack_length = self.max_input_length

        self.lengths = []
        self.data_points = []
        size = len(dataset.eval_dataset)
        for i in range(size):
            dp = dataset.eval_dataset[i]
            input_length = len(dp["input_ids"])
            assert input_length == len(dp["attention_mask"]) == len(dp["labels"])
            self.data_points.append(dp)
            self.lengths.append(input_length)
        self.groups =  pack_with_min_item_num(
            self.lengths, self.pack_length, min_item_num
        )
        random.seed(seed)
        random.shuffle(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        group_data_points = [self.data_points[index] for index in group]
        if not self.use_fa:
            return pack_data_points_naive(
                group_data_points, self.tokenizer, self.max_input_length
            )
        return pack_data_points_FA(
            group_data_points, self.tokenizer, self.max_input_length,
            isolated=self.isolated,
        )

    def stat_dict(self) -> Dict[str, float]:
        
        packed = [sum(self.lengths[i] for i in g) for g in self.groups]
        avg_packed = sum(packed) / len(packed) if packed else 0.0
        cap = float(self.max_input_length or 0)
        ordered = sorted(packed)

        def pct(q):
            if not ordered:
                return None
            return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

        return {
            "n_original": len(self.data_points),
            "n_groups": len(self.groups),
            "avg_original_len": (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0,
            "avg_packed_len": avg_packed,
            "max_input_length": cap,
            "packing_efficiency": (avg_packed / cap) if cap else None,
            "pad_waste": (1.0 - avg_packed / cap) if cap else None,
            "fill_p10": pct(0.10), "fill_p50": pct(0.50), "fill_p90": pct(0.90),
            "use_fa": bool(self.use_fa),
            "isolated": bool(self.isolated),
        }

    def stat(self) -> str:
        result_str = ""
        result_str += f"number of original data points:{len(self.data_points)}; packed to: {len(self.groups)} data points"
        original_avg_length = sum(self.lengths) / len(self.lengths)
        result_str += f"original avg length: {original_avg_length}; "
        original_avg_length = sum(self.lengths) / len(self.lengths)

        packed_lengths = []
        for group in self.groups:
            lengths = [self.lengths[index] for index in group]
            packed_lengths.append(sum(lengths))

        avg_packed_length = sum(packed_lengths) / len(packed_lengths)
        result_str += f"original avg length: {original_avg_length}; avg packed length: {avg_packed_length}"
        return result_str
