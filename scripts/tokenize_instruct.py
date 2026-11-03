import quiet_mode
import host_cpu
from typing import Dict, Any
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Callable
import native_tokenize
from native_tokenize import tokenize_chat_dataset, tokenize_instruct_dataset
import torch
import logging
from datetime import datetime
import sys
import time
import random
import json
import requests
import os
import shutil
import typer


def _process_custom_dataset_fields(custom_type_dict: dict) -> dict:
    if not custom_type_dict.get("field_output"):
        return {
            "type": "completion",
            "field": custom_type_dict.get("field_instruction"),
        }

    processed_dict = custom_type_dict.copy()
    processed_dict.setdefault("no_input_format", "{instruction}")
    if processed_dict.get("field_input"):
        processed_dict.setdefault("format", "{instruction} {input}")
    else:
        processed_dict.setdefault("format", "{instruction}")

    return {"format": "custom", "type": processed_dict}


def _process_chat_template_dataset_fields(dataset_dict: dict) -> dict:
    processed_dict = {}

    processed_dict["chat_template"] = dataset_dict["chat_template"]
    processed_dict["type"] = "chat_template"
    processed_dict["field_messages"] = dataset_dict["chat_column"]
    processed_dict["message_field_role"] = dataset_dict["chat_role_field"]
    processed_dict["message_field_content"] = dataset_dict["chat_content_field"]
    processed_dict["roles"] = {
        "assistant": [dataset_dict["chat_assistant_reference"]],
        "user": [dataset_dict["chat_user_reference"]],
    }

    processed_dict["message_property_mappings"] = {
        "role": dataset_dict["chat_role_field"],
        "content": dataset_dict["chat_content_field"],
    }

    return processed_dict


def create_dataset_entry(
    data_path: str,
    dataset_type: Dict,
    file_format: str,
) -> dict:
    dataset_entry = {"path": data_path}
    custom_type_dict = {
        key: value for key, value in dataset_type.items() if value is not None
    }
    if "chat_template" in dataset_type:
        print("Processing chat template dataset type")
        dataset_entry.update(_process_chat_template_dataset_fields(dataset_type))
    else:
        print("Processing instruct dataset type")
        dataset_entry.update(_process_custom_dataset_fields(custom_type_dict))

    dataset_entry["ds_type"] = file_format
    dataset_entry["data_files"] = [data_path]
    return dataset_entry


def _load_evaluation_dataset_native(
    data_path: str,
    dataset_type: Dict,
    tokenizer: AutoTokenizer,
    sequence_len: int,
    overlength: str,
) -> list:
    
    with open(data_path, "r") as file:
        rows = json.load(file)

    if "chat_template" in dataset_type and dataset_type.get("chat_template"):
        native_type = _process_chat_template_dataset_fields(dataset_type)
        return _tokenize_pooled(native_tokenize.KIND_CHAT, tokenizer, rows, native_type,
                                sequence_len, overlength, data_path)

    custom_type_dict = {k: v for k, v in dataset_type.items() if v is not None}
    entry = _process_custom_dataset_fields(custom_type_dict)
    if entry.get("type") == "completion":
        raise NotImplementedError(
            "completion-type instruct datasets (no field_output) are not ported off axolotl yet; "
            "they were never exercised by a real task here, so porting them blind would be "
            "unverified. Add a reference and a matrix row before enabling."
        )
    return _tokenize_pooled(native_tokenize.KIND_INSTRUCT, tokenizer, rows, entry["type"],
                            sequence_len, overlength, data_path)


def _tokenize_pooled(kind: str, tokenizer, rows: list, native_type: dict, sequence_len: int,
                     overlength: str, data_path: str) -> list:
    out, _ = native_tokenize.tokenize_dataset_pooled(
        kind, tokenizer, rows, native_type, sequence_len, overlength=overlength)
    return out


def remove_empty_output_items(items: list):
    result = []
    for item in items:
        if "output" in item and not item["output"]:
            continue
        if "input" in item and "instruct" in item:
            if not item["instruct"] and not item["input"]:
                continue
        if "output" in item and type(item["output"]) is not str:
            continue

        if (
            "instruct" in item
            and type(item["instruct"]) is not str
            and item["instruct"] is not None
        ):
            continue
        if (
            "input" in item
            and type(item["input"]) is not str
            and item["input"] is not None
        ):
            continue
        result.append(item)
    return result


def replace_wrong_token_in_item(item: dict):
    for key in item:
        if type(item[key]) is str:
            item[key] = item[key].replace("[PAD]", "")
    return item

def _instruct_output_length(item: dict) -> float:
    
    output = item.get("output", "")
    if not isinstance(output, str):
        return 0.0
    return float(len(output))


def dev_random_split() -> bool:
    return os.environ.get("SN56_DEV_RANDOM_SPLIT", "1") != "0"


def split_dataset(
    total_data_path: str,
    train_data_path: str,
    dev_data_path: str,
    seed: int = 42,
    dev_size: int = 200,
    max_data_size: int = -1,
    hours_to_complete: float = 2.0,
):
    
    from val_split import stratified_split, concat_all_text

    with open(total_data_path, "r") as file:
        data = json.load(file)

    random.seed(seed)
    random.shuffle(data)
    if max_data_size > 0:
        data = data[:max_data_size]

    before_len = len(data)
    data = remove_empty_output_items(data)
    n_removed = before_len - len(data)
    if n_removed:
        print(f"Removed {n_removed} empty output items before split")

    dev_size = min(1000, max(min(400, len(data) // 5), len(data) // 33))
    print(f"Escolhendo {dev_size} pro teste (dados={len(data)})")

    dev_items, train_items = stratified_split(
        items=data,
        dev_size=dev_size,
        text_fn=concat_all_text,
        difficulty_fn=_instruct_output_length,
        seed=seed,
        near_dedup=not dev_random_split(),
    )

    with open(train_data_path, "w") as file:
        json.dump(train_items, file, ensure_ascii=False)

    with open(dev_data_path, "w") as file:
        json.dump(dev_items, file, ensure_ascii=False)

    print(
        f"split {total_data_path} ({len(data)} items) into {train_data_path} ({len(train_items)} items) and {dev_data_path} ({len(dev_items)} items)"
    )


def data_stat(items: list):
    for item in items:
        pass


def tokenize_dataset(
    tokenizer: AutoTokenizer,
    data_path: str,
    dataset_type: Dict,
    config_path: str,
    output_path: str,
    split: str,
    max_length: int = -1,
):
    
    with open(config_path, "r") as file:
        sequence_len = yaml.safe_load(file)["sequence_len"]
    if max_length > 0:
        sequence_len = max_length

    native_tokenize.reset_drop_stats()
    overlength = native_tokenize.overlength_policy_for_split(split)
    eval_dataset = _load_evaluation_dataset_native(
        data_path, dataset_type, tokenizer, sequence_len, overlength
    )

    original_length = len(eval_dataset)
    eval_dataset = [
        sample
        for sample in eval_dataset
        if any(label != -100 for label in sample["labels"])
    ]
    try:
        pass
    except Exception:
        pass
    print(
        f"Filtered out {original_length - len(eval_dataset)} samples with empty outputs"
    )
    print(f"Loaded dataset with {len(eval_dataset)} samples")

    result = list(eval_dataset)
    print(f"Dumped {len(result)} samples to {output_path}")

    with open(output_path, "w") as file:
        json.dump(result, file, ensure_ascii=False)


def main(training_request_path: str):
    host_cpu.apply("tokenize_instruct")
    t1 = datetime.now()
    with open(training_request_path, "r") as file:
        training_request = json.load(file)

    task_id = training_request["train_request"]["task_id"]
    total_path = training_request["train_request"]["dataset"]
    train_path = f"datasets/train_{task_id}.json"
    dev_path = f"datasets/dev_{task_id}.json"
    max_data_size = training_request["train_request"].get("max_data_size", -1)
    if max_data_size > 0:
        print(
            f"Max data size is {max_data_size}, so we will only extract {max_data_size} samples randomly"
        )

    hours = training_request["train_request"].get("hours_to_complete", 2)
    split_dataset(
        total_path,
        train_path,
        dev_path,
        max_data_size=max_data_size,
        hours_to_complete=hours,
    )

    config_path = "test_axolotl.yml"
    from model_utility import load_tokenizer
    tokenizer = load_tokenizer(training_request["train_request"]["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = -1

    if "max_length" in training_request["train_request"]:
        max_length = training_request["train_request"]["max_length"]

    print(f"max_length={max_length}")

    tokenize_dataset(
        tokenizer,
        train_path,
        training_request["train_request"]["dataset_type"],
        config_path,
        f"datasets/train_tokenized_{task_id}.json",
        split=native_tokenize.SPLIT_TRAIN,
        max_length=max_length,
    )

    tokenize_dataset(
        tokenizer,
        dev_path,
        training_request["train_request"]["dataset_type"],
        config_path,
        f"datasets/dev_tokenized_{task_id}.json",
        split=native_tokenize.SPLIT_DEV,
        max_length=max_length,
    )

    t2 = datetime.now()
    print(f"Tokenization completed in {(t2 - t1).seconds} seconds")


if __name__ == "__main__":
    typer.run(main)
