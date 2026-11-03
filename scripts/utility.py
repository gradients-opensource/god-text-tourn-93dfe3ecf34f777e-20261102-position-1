from typing import Dict, Any, Optional
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Callable
import torch
import logging
from datetime import datetime
import sys
import wandb
import random
import json
import requests
import os
import shutil
from transformers.trainer_utils import is_main_process
from lfm2_posid_conv import segment_ids
from token_bounds import check_ids_in_range

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def log_info(message: str, event_name: str = "print"):
    if is_main_process(LOCAL_RANK):
        logger.info(f"{event_name}: {message}")


def pad_sequence(sequence: list[int], pad_value: int, max_length: int, padding_side: str) -> list[int]:
    if padding_side == "left":
        return [pad_value] * (max_length - len(sequence)) + sequence
    else:
        return sequence + [pad_value] * (max_length - len(sequence))


def smart_truncate(input_dict: dict, max_length: int) -> dict:
    
    input_ids = input_dict["input_ids"]
    if len(input_ids) <= max_length:
        return input_dict

    labels = input_dict["labels"]

    comp_start = len(labels)
    for i, l in enumerate(labels):
        if l != -100:
            comp_start = i
            break

    comp_len = len(input_ids) - comp_start

    if comp_len >= max_length:
        trim_start = comp_start
    else:
        prompt_budget = max_length - comp_len
        trim_start = comp_start - prompt_budget

    return {
        key: val[trim_start : trim_start + max_length]
        for key, val in input_dict.items()
    }


def apply_prompt_loss_weight(samples: list[dict], plw: float = 0.05, seed: int = 42) -> list[dict]:
    
    import random as _rng
    _rng.seed(seed)
    n_unmasked = 0
    for sample in samples:
        input_ids = sample.get("input_ids", [])
        labels = sample.get("labels", [])
        new_labels = list(labels)
        for i, (tok, lab) in enumerate(zip(input_ids, labels)):
            if lab == -100 and _rng.random() < plw:
                new_labels[i] = tok
                n_unmasked += 1
        sample["labels"] = new_labels
    n_total_prompt = sum(1 for s in samples for l in s.get("labels", []) if l == -100)
    print(
        f"[sn56][plw] Unmasked {n_unmasked} prompt tokens (plw={plw}, "
        f"~{n_unmasked / max(1, n_unmasked + n_total_prompt):.1%} of prompt)",
        flush=True,
    )
    return samples


def gaussian_subsample(samples: list[dict], target_size: int, seed: int = 42) -> list[dict]:
    
    import numpy as _np

    n = len(samples)
    if target_size >= n:
        return samples

    diffs = _np.array([
        sum(1 for l in s.get("labels", []) if l != -100) for s in samples
    ], dtype=_np.float64)

    median = _np.median(diffs)
    std = max(_np.std(diffs), 1.0)

    weights = _np.exp(-0.5 * ((diffs - median) / std) ** 2)
    weights /= weights.sum()

    rng = _np.random.RandomState(seed)
    indices = rng.choice(n, size=target_size, replace=False, p=weights)
    indices.sort()

    print(
        f"[sn56][gauss] {n} -> {target_size} samples "
        f"(median_comp_tokens={median:.0f}, std={std:.0f})",
        flush=True,
    )
    return [samples[i] for i in indices]


def pad_inputs(tokenizer: AutoTokenizer, input_dict: dict, max_length: int, padding_side: str) -> dict:
    assert padding_side in ["left", "right"]
    if max_length <= 0:
        return input_dict
    input_dict = smart_truncate(input_dict, max_length)
    result = {
        "input_ids": pad_sequence(input_dict["input_ids"], tokenizer.pad_token_id, max_length, padding_side),
        "attention_mask": pad_sequence(input_dict["attention_mask"], 0, max_length, padding_side),
        "labels": pad_sequence(input_dict["labels"], -100, max_length, padding_side),
    }
    return result


DYNAMIC_PADDING_ENV = "SN56_DYNAMIC_PADDING"
DYNAMIC_PAD_MULTIPLE = 8
_PAD_VALUES = {"attention_mask": 0, "labels": -100, "position_ids": 0}


def dynamic_padding_enabled() -> bool:
    
    return os.getenv(DYNAMIC_PADDING_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def truncate_inputs(tokenizer: AutoTokenizer, input_dict: dict, max_length: int) -> dict:
    
    if max_length <= 0:
        return input_dict
    input_dict = smart_truncate(input_dict, max_length)
    return {
        "input_ids": list(input_dict["input_ids"]),
        "attention_mask": list(input_dict["attention_mask"]),
        "labels": list(input_dict["labels"]),
    }


class DynamicPaddingCollator:
    

    def __init__(
        self,
        pad_token_id: int,
        padding_side: str = "right",
        label_pad_token_id: int = -100,
        pad_to_multiple_of: Optional[int] = DYNAMIC_PAD_MULTIPLE,
        max_length: Optional[int] = None,
    ) -> None:
        assert padding_side in ("left", "right"), padding_side
        if pad_token_id is None:
            raise ValueError("DynamicPaddingCollator needs a pad_token_id")
        self.pad_token_id = int(pad_token_id)
        self.padding_side = padding_side
        self.label_pad_token_id = int(label_pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length
        self.stats = {"batches": 0, "rows": 0, "slots": 0, "real": 0, "supervised": 0,
                      "max_batch_len": 0}
        self.with_seq_idx = False
        self.vocab_limit: Optional[int] = None

    def _pad_value(self, key: str) -> int:
        if key == "input_ids":
            return self.pad_token_id
        if key == "labels":
            return self.label_pad_token_id
        return _PAD_VALUES.get(key, 0)

    def __call__(self, features):
        if not features:
            return {}
        keys = [k for k in features[0] if all(k in f for f in features)]
        lengths = [len(f["input_ids"]) for f in features]
        target = max(lengths)
        if min(lengths) != target:
            if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
                m = int(self.pad_to_multiple_of)
                target = ((target + m - 1) // m) * m
            if self.max_length and self.max_length > 0:
                target = min(target, max(self.max_length, max(lengths)))

        batch = {}
        for key in keys:
            col = [f[key] for f in features]
            first = col[0]
            if not torch.is_tensor(first) and not isinstance(first, (list, tuple)):
                batch[key] = torch.tensor(col)
                continue
            rows = []
            for value in col:
                seq = value.tolist() if torch.is_tensor(value) else list(value)
                gap = target - len(seq)
                if gap < 0:
                    raise ValueError(f"row longer than the batch max on {key!r}")
                if gap:
                    fill = [self._pad_value(key)] * gap
                    seq = fill + seq if self.padding_side == "left" else seq + fill
                rows.append(seq)
            batch[key] = torch.tensor(rows, dtype=torch.long)
        if self.with_seq_idx and "position_ids" in batch and "seq_idx" not in batch:
            batch["seq_idx"] = segment_ids(batch["position_ids"])
        if self.vocab_limit and "input_ids" in batch:
            check_ids_in_range(batch["input_ids"], self.vocab_limit, "DynamicPaddingCollator")

        st = self.stats
        st["batches"] += 1
        st["rows"] += len(features)
        st["slots"] += target * len(features)
        st["real"] += sum(lengths)
        st["max_batch_len"] = max(st["max_batch_len"], target)
        if "labels" in batch:
            st["supervised"] += int((batch["labels"] != self.label_pad_token_id).sum())
        return batch

    def fill_stats(self) -> dict:
        
        st = dict(self.stats)
        slots = st["slots"]
        st["token_fill"] = round(st["real"] / slots, 5) if slots else None
        st["supervised_fill"] = round(st["supervised"] / slots, 5) if slots else None
        st["mean_padded_len"] = round(slots / st["rows"], 2) if st["rows"] else None
        return st


class MyDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str,
        max_length: int,
        dynamic_padding: Optional[bool] = None,
    ) -> None:
        super().__init__()
        with open(data_path, 'r') as file:
            self.eval_dataset = json.load(file)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dynamic_padding = (
            dynamic_padding_enabled() if dynamic_padding is None else bool(dynamic_padding)
        )
        print("padding_side: ", self.tokenizer.padding_side)
        print("dynamic_padding: ", self.dynamic_padding)

    def __len__(self):
        return len(self.eval_dataset)

    @property
    def lengths(self) -> list[int]:
        
        cap = self.max_length if self.max_length and self.max_length > 0 else None
        return [
            (min(len(dp["input_ids"]), cap) if cap else len(dp["input_ids"]))
            for dp in self.eval_dataset
        ]

    def __getitem__(self, idx):
        dp = self.eval_dataset[idx]
        if self.dynamic_padding:
            input_dict = truncate_inputs(self.tokenizer, dp, self.max_length)
        else:
            input_dict = pad_inputs(self.tokenizer, dp, self.max_length, self.tokenizer.padding_side)
        input_dict = dict(input_dict)
        for key in input_dict:
            input_dict[key] = torch.tensor(input_dict[key])
        return input_dict
