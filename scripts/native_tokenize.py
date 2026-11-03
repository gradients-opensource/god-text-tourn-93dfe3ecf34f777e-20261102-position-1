
import time
from typing import Any, Optional

from transformers import AutoTokenizer

import host_cpu

DROP_STATS = {"rows_in": 0, "rows_out": 0, "dropped_overlength": 0,
              "dropped_all_masked": 0, "dropped_empty": 0,
              "truncated_overlength": 0, "dropped_completion_exceeds": 0,
              "tokens_in": 0, "tokens_out": 0,
              "tokens_in_truncated_rows": 0, "tokens_recovered": 0}


def reset_drop_stats():
    for key in DROP_STATS:
        DROP_STATS[key] = 0


IGNORE_INDEX = -100

MIN_PROMPT_TOKENS_KEPT = 1
OVERLENGTH_TRUNCATE = "truncate"
OVERLENGTH_DROP = "drop"

SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
OVERLENGTH_BY_SPLIT = {
    SPLIT_TRAIN: OVERLENGTH_TRUNCATE,
    SPLIT_DEV: OVERLENGTH_DROP,
}


def overlength_policy_for_split(split: str) -> str:
    
    return OVERLENGTH_BY_SPLIT.get(split, OVERLENGTH_DROP)


def left_truncate(input_ids: list[int], labels: list[int], sequence_len: int):
    
    return input_ids[-sequence_len:], labels[-sequence_len:]


DEFAULT_FORMAT = "{instruction} {input}"
DEFAULT_NO_INPUT_FORMAT = "{instruction}"


def _field(item: dict, name: str | None) -> str:
    
    if not name:
        return ""
    value = item.get(name)
    return value if isinstance(value, str) else ""


def build_prompt(item: dict, dataset_type: dict) -> str:
    
    instruction = _field(item, dataset_type.get("field_instruction"))
    model_input = _field(item, dataset_type.get("field_input"))

    if model_input:
        template = dataset_type.get("format") or DEFAULT_FORMAT
        return template.format(instruction=instruction, input=model_input)

    template = dataset_type.get("no_input_format") or DEFAULT_NO_INPUT_FORMAT
    return template.format(instruction=instruction)


def tokenize_row(
    tokenizer: AutoTokenizer,
    item: dict,
    dataset_type: dict,
    sequence_len: int,
    overlength: str = OVERLENGTH_TRUNCATE,
) -> dict[str, list[int]] | None:
    
    prompt = build_prompt(item, dataset_type)
    completion = _field(item, dataset_type.get("field_output"))

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    DROP_STATS["tokens_in"] += len(input_ids)

    if sequence_len > 0 and len(input_ids) > sequence_len:
        if overlength != OVERLENGTH_TRUNCATE:
            DROP_STATS["dropped_overlength"] += 1
            return None
        if len(completion_ids) + MIN_PROMPT_TOKENS_KEPT > sequence_len:
            DROP_STATS["dropped_completion_exceeds"] += 1
            return None
        DROP_STATS["truncated_overlength"] += 1
        DROP_STATS["tokens_in_truncated_rows"] += len(input_ids)
        DROP_STATS["tokens_recovered"] += sequence_len
        input_ids, labels = left_truncate(input_ids, labels, sequence_len)

    if all(label == IGNORE_INDEX for label in labels):
        DROP_STATS["dropped_all_masked"] += 1
        return None

    DROP_STATS["tokens_out"] += len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def tokenize_instruct_dataset(
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
    dataset_type: dict,
    sequence_len: int,
    overlength: str = OVERLENGTH_TRUNCATE,
) -> list[dict[str, list[int]]]:
    
    out = []
    for item in rows:
        tokenized = tokenize_row(tokenizer, item, dataset_type, sequence_len, overlength)
        if tokenized is not None:
            out.append(tokenized)
    DROP_STATS["rows_in"] += len(rows)
    DROP_STATS["rows_out"] += len(out)
    return out


DUMMY_MESSAGE = "[[dummy_message]]"

DEFAULT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
}


def chat_role_map(dataset_type: dict) -> dict:
    
    roles = dataset_type.get("roles")
    if not roles:
        return dict(DEFAULT_ROLE_MAP)
    return {source: canonical for canonical, sources in roles.items() for source in sources}


def to_turns(row: dict, dataset_type: dict) -> list[dict]:
    
    mappings = dataset_type.get("message_property_mappings") or {
        "role": dataset_type.get("message_field_role", "role"),
        "content": dataset_type.get("message_field_content", "content"),
    }
    role_map = chat_role_map(dataset_type)

    turns = []
    for message in row[dataset_type["field_messages"]]:
        turn = {}
        for key, source in mappings.items():
            if message.get(source) is not None:
                turn[key] = message[source]
        if "role" in turn:
            turn["role"] = role_map.get(turn["role"], turn["role"])
        turns.append(turn)
    return turns


def _render(tokenizer, turns: list[dict], chat_template: str | None) -> list[int]:
    
    kwargs = {"add_generation_prompt": False}
    if chat_template and chat_template != "tokenizer_default":
        kwargs["chat_template"] = chat_template

    out = tokenizer.apply_chat_template(turns, **kwargs)
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def find_turn(tokenizer, turns: list[dict], turn_idx: int, chat_template: str | None) -> tuple[int, int]:
    
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if turn_idx == 0 and turns[0].get("role") == "system" and ("mistral" in name or "gemma" in name):
        return -1, -1

    empty = {"role": turns[turn_idx].get("role"), "content": DUMMY_MESSAGE}
    dummy_ids = _render(tokenizer, turns[:turn_idx] + [empty], chat_template)
    full_ids = _render(tokenizer, turns[: turn_idx + 1], chat_template)
    if not full_ids or not dummy_ids:
        return -1, -1

    min_len = min(len(dummy_ids), len(full_ids))

    start_idx = None
    for i in range(min_len):
        if dummy_ids[i] != full_ids[i]:
            start_idx = i
            break
    if start_idx is None:
        return -1, -1

    end_idx = None
    for i in range(min_len):
        if dummy_ids[len(dummy_ids) - 1 - i] != full_ids[len(full_ids) - 1 - i]:
            end_idx = len(full_ids) - 1 - i + 1
            break
    if end_idx is None or end_idx <= start_idx:
        return -1, -1

    return start_idx, end_idx


def _find_first(input_ids: list[int], token_ids: set[int], start_idx: int) -> int:
    for i in range(start_idx, len(input_ids)):
        if input_ids[i] in token_ids:
            return i
    return -1


def tokenize_chat_row(
    tokenizer,
    row: dict,
    dataset_type: dict,
    sequence_len: int,
    roles_to_train: tuple[str, ...] = ("assistant",),
    train_on_eos: str = "turn",
    train_on_inputs: bool = False,
) -> dict[str, list[int]] | None:
    chat_template = dataset_type.get("chat_template")
    turns = to_turns(row, dataset_type)
    if not turns:
        DROP_STATS["dropped_empty"] += 1
        return None

    input_ids = _render(tokenizer, turns, chat_template)
    if not input_ids:
        DROP_STATS["dropped_empty"] += 1
        return None

    labels = [IGNORE_INDEX] * len(input_ids)

    role_map = chat_role_map(dataset_type)
    trainable = {role_map.get(r, r) for r in roles_to_train}

    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    last_eos_idx = -1

    for index, turn in enumerate(turns):
        should_train = train_on_inputs or turn.get("role") in trainable
        start_idx, end_idx = find_turn(tokenizer, turns, index, chat_template)

        if should_train and start_idx != -1 and end_idx != -1:
            labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

        token_idx = _find_first(input_ids, eos_ids, end_idx)
        if token_idx != -1 and abs(token_idx - end_idx) <= 3:
            last_eos_idx = token_idx
            if train_on_eos == "all" or (train_on_eos == "turn" and should_train):
                labels[token_idx] = input_ids[token_idx]

    if train_on_eos == "last" and last_eos_idx != -1:
        labels[last_eos_idx] = input_ids[last_eos_idx]

    DROP_STATS["tokens_in"] += len(input_ids)
    if sequence_len > 0 and len(input_ids) > sequence_len:
        DROP_STATS["dropped_overlength"] += 1
        return None
    if all(label == IGNORE_INDEX for label in labels):
        DROP_STATS["dropped_all_masked"] += 1
        return None

    DROP_STATS["tokens_out"] += len(input_ids)
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def tokenize_chat_dataset(tokenizer, rows, dataset_type: dict, sequence_len: int):
    out = []
    for row in rows:
        tokenized = tokenize_chat_row(tokenizer, row, dataset_type, sequence_len)
        if tokenized is not None:
            out.append(tokenized)
    DROP_STATS["rows_in"] += len(rows)
    DROP_STATS["rows_out"] += len(out)
    return out


KIND_INSTRUCT = "instruct"
KIND_CHAT = "chat"
POOL_MIN_ROWS = 1000

_POOL_JOB: Optional[tuple] = None


def _serial(kind: str, tokenizer, rows, dataset_type: dict, sequence_len: int, overlength: str):
    if kind == KIND_CHAT:
        return tokenize_chat_dataset(tokenizer, rows, dataset_type, sequence_len)
    return tokenize_instruct_dataset(tokenizer, rows, dataset_type, sequence_len,
                                     overlength=overlength)


def _tokenize_range(span: tuple) -> tuple:
    kind, tokenizer, rows, dataset_type, sequence_len, overlength = _POOL_JOB
    lo, hi = span
    reset_drop_stats()
    out = _serial(kind, tokenizer, rows[lo:hi], dataset_type, sequence_len, overlength)
    return out, dict(DROP_STATS)


def tokenize_dataset_pooled(kind: str, tokenizer, rows: list, dataset_type: dict,
                            sequence_len: int, overlength: str = OVERLENGTH_TRUNCATE,
                            workers: Optional[int] = None, min_rows: int = POOL_MIN_ROWS):
    global _POOL_JOB
    w = int(workers if workers is not None else host_cpu.current().tokenize_workers)
    if w <= 1 or len(rows) < min_rows:
        t0 = time.perf_counter()
        out = _serial(kind, tokenizer, rows, dataset_type, sequence_len, overlength)
        return out, host_cpu.PoolRun(mode="serial", workers=1, n_items=len(rows), n_chunks=1,
                                     secs=round(time.perf_counter() - t0, 3))
    spans = [(c[0], c[-1] + 1) for c in host_cpu.chunk(range(len(rows)), w * 4) if c]
    before = dict(DROP_STATS)
    _POOL_JOB = (kind, tokenizer, rows, dataset_type, sequence_len, overlength)
    try:
        parts, run = host_cpu.pool_map(_tokenize_range, spans, workers=w, n_items=len(rows))
    finally:
        _POOL_JOB = None
    for key in DROP_STATS:
        DROP_STATS[key] = before.get(key, 0) + sum(int(st.get(key, 0)) for _, st in parts)
    out = [row for part, _ in parts for row in part]
    return out, run
