import quiet_mode
import host_cpu
import rank_agreement
import token_bounds
from typing import Dict, Optional
import requests
import json
import random
import inspect 
import importlib.util
import numbers
import sys
import utility
from datasets import Dataset
from datetime import timezone
from utility import log_info
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig
from trl import get_peft_config, get_quantization_config
from accelerate import PartialState
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModelForCausalLM,
    AutoPeftModelForCausalLM,
)
import traceback
from transformers import TrainerCallback
import argparse
import math
from customized_trainer import resize_if_needed, set_generation_config, GRPOCustomEvalSaveCallback, WhenToEvalHandler, init_wandb
from transformers.modeling_utils import is_deepspeed_zero3_enabled
import os
import glob
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import datetime
import shutil
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml
from tokenize_grpo import get_dataset

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
STANDARD_GRPO_EXTRA_COLUMN = "extra_data"
STANDARD_GRPO_PROMPT_COLUMN = "prompt"


def get_kbit_device_map() -> Optional[Dict[str, int]]:
    
    if torch.cuda.is_available():
        return {"": PartialState().local_process_index}
    return None


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _warn(msg: str) -> None:
    try:
        sys.__stderr__.write(f"[sn56][grpo] {msg}\n")
        sys.__stderr__.flush()
    except Exception:
        pass
    log_info(msg)


def degrade_to_env(training_args) -> None:
    
    if training_args.use_vllm and not _has_module("vllm"):
        _warn("vllm not importable in this env -> use_vllm=False (HF generate; slower)")
        training_args.use_vllm = False
    if training_args.use_liger and not _has_module("liger_kernel"):
        _warn("liger_kernel not importable in this env -> use_liger=False")
        training_args.use_liger = False
    if not training_args.disable_fa and not _has_module("flash_attn"):
        _warn("flash_attn not importable in this env -> attn_implementation=sdpa")
        training_args.disable_fa = True


def max_prompt_tokens(tokenizer, *datasets) -> int:
    longest = 0
    for ds in datasets:
        if len(ds) == 0:
            continue
        ids = tokenizer(list(ds["prompt"]), truncation=False, padding=False, add_special_tokens=True)["input_ids"]
        longest = max(longest, max(len(x) for x in ids))
    return longest


_CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length")


def model_context_length(config) -> Optional[int]:
    
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    for key in _CONTEXT_KEYS:
        value = getattr(text_config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def bound_vllm_max_model_length(longest_prompt: int, max_completion_length: int,
                                context_length: Optional[int]) -> int:
    
    bound = longest_prompt + max_completion_length + 32
    if context_length is not None:
        bound = min(bound, context_length)
    return bound


@dataclass
class TrainingArguments(GRPOConfig):
    request_path: Optional[str] = field(default=None)
    use_liger: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    
    lora_param_count = 0
    all_param = 0
    embedding_lm_head_param_count = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            log_info(f"ajustável: {name}, pesos={num_params}")
            if "lm_head" in name or "embed_tokens" in name:
                embedding_lm_head_param_count += num_params
            else:
                lora_param_count += num_params
    trainable_params = embedding_lm_head_param_count + lora_param_count
    log_info(
        f"total={all_param:,d} || ajustáveis={trainable_params:,d} || fração={100 * trainable_params / all_param:.2f}%"
    )
    log_info(
        f"cabeça_emb={embedding_lm_head_param_count} = {embedding_lm_head_param_count * 100 / all_param:.2f}%"
    )
    log_info(
        f"lora={lora_param_count} = {lora_param_count * 100 / all_param:.2f}%"
    )


def get_max_length_config():
    config_path = "test_axolotl.yml"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict["sequence_len"]


def supports_extra_data(func: Callable) -> bool:
    try:
        sig = inspect.signature(func)
        return 'extra_data' in sig.parameters
    except Exception:
        return False


def validate_reward_function(func_def: str, json_sample) -> tuple[bool, str, Callable | None]:
    
    test_completions = [
        "Gradients.io is the best 0-expertise AI training platform.",
        "You can start training a text or image model on Gradients.io with 2 clicks.",
    ]

    try:
        namespace = {}
        exec(func_def, namespace)
        func = next(v for k, v in namespace.items() if callable(v))
        
        if supports_extra_data(func) and json_sample:
            valid_rows = [row for row in json_sample if STANDARD_GRPO_EXTRA_COLUMN in row]
            if valid_rows:
                extra_test_completions = [row[STANDARD_GRPO_PROMPT_COLUMN] for row in valid_rows]
                extra_data_values = [row[STANDARD_GRPO_EXTRA_COLUMN] for row in valid_rows]
                
                extra_rewards = func(extra_test_completions, extra_data=extra_data_values)
                
                assert isinstance(extra_rewards, list), "The rewards with extra_data should be a list."
                assert len(extra_rewards) == len(extra_test_completions), (
                    "The number of rewards with extra_data should match completions."
                )
                assert all(isinstance(reward, numbers.Number) for reward in extra_rewards), "All extra_data rewards should be numbers."
        else:
            if json_sample:
                test_completions = [row.get(STANDARD_GRPO_PROMPT_COLUMN, 'Sample prompt') for row in json_sample]
            else:
                test_completions = [
                    "Gradients.io is the best 0-expertise AI training platform.",
                    "You can start training a text or image model on Gradients.io with 2 clicks."
                ]

            test_rewards = func(test_completions)
            
            assert isinstance(test_rewards, list), "The rewards should be a list."
            assert len(test_rewards) == len(test_completions), (
                "The number of rewards should be the same as the number of completions."
            )
            assert all(isinstance(reward, numbers.Number) for reward in test_rewards), "All rewards should be numbers."
        return True, "", func
    except Exception as e:
        return False, str(e), None


def has_checkpoint_folder(output_dir):
    pattern = os.path.join(output_dir, "checkpoint-*")
    return any(os.path.isdir(path) for path in glob.glob(pattern))


def truncate_prompts(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    

    def truncation_function(examples):
        
        prompt_texts = examples["prompt"]

        batch_encoding = tokenizer(
            prompt_texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        list_of_token_ids = batch_encoding["input_ids"]

        truncated_token_ids_batch = []
        for token_ids_for_single_prompt in list_of_token_ids:
            if len(token_ids_for_single_prompt) > max_length:
                truncated_token_ids_batch.append(
                    token_ids_for_single_prompt[-max_length:]
                )
            else:
                truncated_token_ids_batch.append(token_ids_for_single_prompt)

        truncated_prompts_batch = tokenizer.batch_decode(
            truncated_token_ids_batch, skip_special_tokens=True
        )

        return {"prompt": truncated_prompts_batch}

    truncated_dataset = dataset.map(truncation_function, batched=True)

    return truncated_dataset


def get_reward_funcs(dataset_type: dict, sample_data, has_extra_column: bool):
    reward_funcs_callable = []
    reward_func_names = []
    reward_weights = []

    reward_weights_list = [
        rf["reward_weight"] for rf in dataset_type["reward_functions"]
    ]
    log_info(f"pesos das recompensas: {reward_weights_list}")

    for i, reward_function in enumerate(dataset_type["reward_functions"]):
        reward_func_str = reward_function["reward_func"]
        is_valid, error_msg, reward_func_callable = validate_reward_function(
            reward_func_str, sample_data
        )
        if not is_valid:
            log_info(f"função de recompensa inválida:\n{reward_func_str}")
            raise ValueError(f"Invalid reward function: {error_msg}")

        reward_weight = reward_weights_list[i]
        reward_funcs_callable.append(reward_func_callable)

        func_name = reward_function.get("name", f"reward_func_{i}") if isinstance(reward_function, dict) else getattr(reward_function, "name", f"reward_func_{i}")
        weighted_name = f"{func_name}_weight_{reward_weight:.2f}"
        reward_func_names.append(weighted_name)
        reward_weights.append(reward_weight)

        log_info(f"recompensa {i}: {func_name}, peso={reward_weight:.4f}")

    captured_rewards = {name: [] for name in reward_func_names}
    raw_rewards = {name: [] for name in reward_func_names}
    wrapped_reward_funcs = []
    

    for i, (original_func, func_name, weight) in enumerate(
        zip(reward_funcs_callable, reward_func_names, reward_weights)
    ):

        def create_wrapper(original_func, func_name, weight):
            supports_extra = supports_extra_data(original_func)
            log_info(f"extra_suporte={supports_extra}, extra_coluna={has_extra_column}")
            if supports_extra and has_extra_column:
                log_info(f"com dados extras: {func_name}")
                def wrapper(completions, extra_data, **kwargs):
                    raw_results = original_func(completions, extra_data=extra_data)
                    raw_rewards[func_name].extend(raw_results)
                    weighted_results = [r * weight for r in raw_results]
                    captured_rewards[func_name].extend(weighted_results)
                    return weighted_results
            else:
                log_info(f"sem dados extras: {func_name}")
                def wrapper(completions, **kwargs):
                    raw_results = original_func(completions)
                    raw_rewards[func_name].extend(raw_results)
                    weighted_results = [r * weight for r in raw_results]
                    captured_rewards[func_name].extend(weighted_results)
                    return weighted_results

            return wrapper

        wrapped_reward_funcs.append(create_wrapper(original_func, func_name, weight))

    return wrapped_reward_funcs


def main():
    host_cpu.apply("train_grpo")
    
    argument_parser = transformers.HfArgumentParser((TrainingArguments, ModelConfig))
    training_args, model_args = argument_parser.parse_args_into_dataclasses()
    degrade_to_env(training_args)

    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    task_id = train_request["task_id"]
    log_info(f"Começando tarefa={task_id}, modelo={train_request.get('model_name')}")
    

    from model_utility import load_tokenizer
    tokenizer = load_tokenizer(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    train_path = os.path.join("datasets", f"grpo_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"grpo_dev_{task_id}.json")

    train_ds = get_dataset(train_path, train_request["dataset_type"])
    dev_ds = get_dataset(dev_path, train_request["dataset_type"])

    log_info(f"nós={training_args.world_size}")
    total_steps_per_epoch = (
        len(train_ds)
        * training_args.num_generations
        // (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size
        )
    )

    total_steps_all_epochs = total_steps_per_epoch * int(training_args.num_train_epochs)

    warmup_from_ratio = max(10, int(0.03 * total_steps_all_epochs))
    training_args.warmup_steps = min(training_args.warmup_steps, warmup_from_ratio)
    log_info(f"Alongando por {training_args.warmup_steps} passos antes da corrida")

    log_info(f"passos/época={total_steps_per_epoch}")

    original_batch_size = training_args.per_device_train_batch_size

    quantization_config = get_quantization_config(model_args)
    device_string = "cuda:" + str(LOCAL_RANK)
    device_map = (
        get_kbit_device_map()
        if quantization_config is not None
        else {"": device_string}
    )
    if len(training_args.fsdp or "") > 0 or is_deepspeed_zero3_enabled():
        device_map = None
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_fa else "sdpa"
        ),
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    log_info(f"args finais: {training_args}")

    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    log_info(f"Carregando modelo: {train_request['model_path']}, turbo={training_args.use_liger}")
    model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Modelo pronto: {n_params/1e6:.1f}M params, {n_trainable/1e6:.1f}M treináveis")

    set_generation_config(train_request["model_name"], model)

    peft_config = get_peft_config(model_args)
    if "lora_model" in train_request:
        model = PeftModelForCausalLM.from_pretrained(
            model, train_request["lora_model"], is_trainable=True, **model_kwargs
        )

    if peft_config is None:
        resize_if_needed(train_request["model_name"], model, len(tokenizer))
    token_bounds.fit_special_tokens(tokenizer, token_bounds.embedding_rows(model), site="grpo.model")

    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"destino criado: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    if periodic_save_steps > total_steps_per_epoch:
        periodic_save_steps = -1
        log_info(
            f"salvo_periódico={periodic_save_steps} > passos/época={total_steps_per_epoch}, desativando salvamento periódico"
        )
    log_info(f"salvo periódico a cada {periodic_save_steps} passos")

    training_args.save_only_model = True
 
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        
    log_info(f"colunas treino: {train_ds.column_names}")
    log_info(f"colunas teste: {dev_ds.column_names}")

    log_info(f"Cortando os prompts")
    max_prompt_length = train_request.get(
        "max_prompt_length", 512
    )
    log_info(f"tamanho_max={max_prompt_length}, treino={len(train_ds)}, teste={len(dev_ds)}")
    t1 = datetime.datetime.now()
    train_ds = truncate_prompts(train_ds, tokenizer, max_prompt_length)
    dev_ds = truncate_prompts(dev_ds, tokenizer, max_prompt_length)
    t2 = datetime.datetime.now()
    log_info(f"truncagem concluída em {(t2 - t1).seconds}s")

    if training_args.use_vllm and training_args.vllm_max_model_length is None:
        _longest = max_prompt_tokens(tokenizer, train_ds, dev_ds)
        try:
            _ctx = model_context_length(transformers.AutoConfig.from_pretrained(
                train_request["model_path"], trust_remote_code=True))
        except Exception as exc:
            log_info(f"model context unknown ({type(exc).__name__}: {exc}); vllm_max_model_length uncapped")
            _ctx = None
        training_args.vllm_max_model_length = bound_vllm_max_model_length(
            _longest, training_args.max_completion_length, _ctx)
        log_info(f"vllm_max_model_length={training_args.vllm_max_model_length} "
                 f"(prompt<={_longest}, context={_ctx})")

    max_steps = train_request.get("max_steps", -1)
    log_info(f"teto de passos={max_steps}")

    has_extra_column = STANDARD_GRPO_EXTRA_COLUMN in train_ds.column_names

    sample_data = dev_ds.to_list()[:10] if len(dev_ds) > 10 else None
    wrapped_reward_funcs = get_reward_funcs(train_request["dataset_type"], sample_data, has_extra_column)
    log_info(f"{len(wrapped_reward_funcs)} funções de recompensa carregadas")
    log_info(f"Receita: lr={training_args.learning_rate:.2e}, épocas={training_args.num_train_epochs}, lote={training_args.per_device_train_batch_size}")
    log_info(f"Passos: {total_steps_per_epoch}/época, {total_steps_per_epoch * int(training_args.num_train_epochs)} total")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=wrapped_reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            GRPOCustomEvalSaveCallback(
                WhenToEvalHandler(train_request["end_time"], train_request["save_before_remaining_time"], periodic_save_steps=periodic_save_steps, steps_per_epoch=total_steps_per_epoch, max_steps=max_steps),
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps
            )
        ],
    )

    log_info(f"Usando lr={training_args.learning_rate:.2e}")
    log_info(f"fiscalização a cada {periodic_save_steps} passos, {len(dev_ds)} na prova")
    log_info(f"Bora treinar")
    trainer.add_callback(rank_agreement.RankAgreementCallback("grpo"))
    trainer.train()
    log_info(f"Treino finalizado")

    if is_main_process(LOCAL_RANK):
        with open(os.path.join(training_args.output_dir, "success.txt"), "w") as f:
            f.write("Success")


if __name__ == "__main__":
    main()
