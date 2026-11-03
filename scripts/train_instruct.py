import host_cpu
import save_guard as sg
import run_state
import quiet_mode
import async_save
import member_guard
import planner_clock
import rank_agreement
import train_step_clock
import token_bounds
from typing import Dict, Optional
import requests
import json
import random
from utility import log_info, MyDataset, DynamicPaddingCollator, dynamic_padding_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from transformers.trainer_utils import is_main_process
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from dataclasses import asdict, dataclass, field
from transformers import Trainer
from customized_trainer import MIS_MATCH_VOCAB_SIZE_MODELS, resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb, restate_eval_gate_horizon, restate_eval_cadence, CADENCE_HORIZON_FRACTION, eval_gate_step
from checkpoint_avg_callback import AdaptiveTrainingCallback, stop_rule_mode
import lr_search
import model_sanity
from lr_search import run_lr_search
import second_trajectory
from kl_trainer import KLRegularizedTrainer, load_base_model
import lfm2_posid_conv
import ssm_kernels
from granite_flce import enable_granite_flce, wants_granite_flce
from row_weighted_loss import RowWeightedTrainer, row_weight_alpha, row_loss_cap, cap_for_record, make_probe_loss_fn
import batch_policy
import clock_hygiene
from batched_eval import with_batched_eval
from dev_pass import DEV_PASS_MICRO_BATCH
import phi_rtn_cosine
import precise_update
import length_grouping
import logit_calib
import wsd_a0
import a0_cosine
import early_evals
import packed_fill
import stop_to_decay
import depth_first
import token_budget

from transformers import (
    Trainer,
    TrainingArguments,
)

import os
import datetime
import shutil
import time
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml
from state_manager import get_state, set_state

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    request_path: Optional[str] = field(default=None)
    packing: Optional[bool] = field(default=False)
    max_packed_size: Optional[int] = field(default=-1)
    use_liger: Optional[bool] = field(default=False)
    use_lora: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)
    use_attn_implementation: Optional[str] = field(default="")

@dataclass
class LoraArguments:
    lora_r: int = 128
    lora_alpha: int = 512
    lora_dropout: float = 0.1
    lora_target_modules: str = "all"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    
    
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
        f"total={all_param:,d} || ajustáveis={trainable_params:,d} || fração={100 * trainable_params / all_param:.1f}%"
    )
    log_info(
        f"cabeça_emb={embedding_lm_head_param_count} ({embedding_lm_head_param_count * 100 / all_param:.1f}%)"
    )
    log_info(
        f"lora={lora_param_count} ({lora_param_count * 100 / all_param:.1f}%)"
    )
    

ATTN_ERROR_MARKERS = (
    "does not support an attention implementation",
    "scaled_dot_product_attention yet",
    "is not supported when `use_flash_attn` is true",
    "does not support flash attention",
    "not support flash_attention_2",
)


def _is_attn_incompatibility(exc: Exception) -> bool:
    
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return False
    return any(m in str(exc).lower() for m in ATTN_ERROR_MARKERS)


def load_with_attn_fallback(model_class, model_path: str, preferred: str, **kwargs):
    
    ladder = [preferred] + [a for a in ("sdpa", "eager") if a != preferred]
    last_exc = None
    for attn in ladder:
        try:
            model = model_class.from_pretrained(
                model_path, attn_implementation=attn, **kwargs)
            if attn != preferred:
                log_info(f"attn fallback: {preferred} -> {attn}")
            else:
                pass
            return model
        except Exception as exc:
            if not _is_attn_incompatibility(exc):
                raise
            log_info(f"attn {attn} rejected by architecture: {str(exc)[:200]}")
            last_exc = exc
    raise last_exc


def probe_attn_implementation(model_path: str, preferred: str) -> str:
    
    if not torch.cuda.is_available():
        return preferred
    ladder = [preferred] + [a for a in ("sdpa", "eager") if a != preferred]
    device = f"cuda:{LOCAL_RANK}"
    for attn in ladder:
        model = None
        try:
            cfg = transformers.AutoConfig.from_pretrained(model_path)
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                if isinstance(getattr(cfg, attr, None), int):
                    setattr(cfg, attr, min(2, getattr(cfg, attr)))
            model = transformers.AutoModelForCausalLM.from_config(
                cfg, attn_implementation=attn)
            model = model.to(device=device, dtype=torch.bfloat16).eval()
            ids = torch.ones((1, 8), dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
            return attn
        except Exception as exc:
            if not _is_attn_incompatibility(exc):
                log_info(f"attn probe inconclusive on {attn}: {str(exc)[:200]}")
                return attn
            log_info(f"attn probe: {attn} rejected by architecture: {str(exc)[:200]}")
        finally:
            del model
            torch.cuda.empty_cache()
    return ladder[-1]


def load_lora_model(training_args: TrainingArguments, model_path: str, lora_args: LoraArguments, token_nums: int):
    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    preferred = (training_args.use_attn_implementation
                 or ("flash_attention_2" if not training_args.disable_fa else "sdpa"))
    model = load_with_attn_fallback(
        model_class,
        model_path,
        preferred,
        torch_dtype=torch.bfloat16,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2" if not training_args.disable_fa else "sdpa",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if lora_args.q_lora
            else None
        ),
    )
    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        modules = lora_args.lora_target_modules.split(" ")
        target_modules = [mod.strip() for mod in modules if len(mod.strip()) > 0]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        _reentrant_checkpointing_for_zero3(training_args, model)

    model.config.use_cache = False
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)

    print_trainable_parameters(model)
    return model


def _reentrant_checkpointing_for_zero3(training_args: TrainingArguments, model) -> None:
    
    if not is_deepspeed_zero3_enabled():
        return
    kw = {"use_reentrant": True}
    training_args.gradient_checkpointing_kwargs = kw
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)
    log_info("[sn56] ZeRO-3: gradient checkpointing set to reentrant")


def resolve_attn_implementation(training_args: TrainingArguments) -> str:
    
    if training_args.use_attn_implementation:
        return training_args.use_attn_implementation
    return "flash_attention_2" if not training_args.disable_fa else "sdpa"


def load_model(training_args: TrainingArguments, model_path: str, token_nums: int):
    model_class = transformers.AutoModelForCausalLM
    
    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        log_info("--- turbo ativo ---")
        model_class = AutoLigerKernelForCausalLM
    
    attn_implementation = resolve_attn_implementation(training_args)
    if training_args.use_attn_implementation:
        log_info(f"atenção via {attn_implementation}")
    log_info(f"atenção: {attn_implementation}")
    
    model = load_with_attn_fallback(
        model_class,
        model_path,
        attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    if wants_granite_flce(model, training_args.use_liger, is_deepspeed_zero3_enabled()):
        enable_granite_flce(model)
        log_info("granite: FLCE em h/logits_scaling")
    return model


def get_max_length_config():
    config_path = "test_axolotl.yml"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict["sequence_len"]


def _emit_epoch_plan_restated(facts, t_per_step_search, inputs) -> None:
    try:
        clk = facts.clock_stats if facts is not None else None
        if clk is None or not inputs or not t_per_step_search:
            return
    except Exception:
        pass


def _devfit_collective_timeout(facts, training_args) -> Optional[float]:
    if int(getattr(training_args, "world_size", 1) or 1) <= 1:
        return None
    clk = getattr(facts, "clock_stats", None) if facts is not None else None
    t_step = clk.t_step if clk is not None else getattr(facts, "t_step_realised", None)
    t_point = clk.t_point if clk is not None else getattr(facts, "eval_runtime", None)
    return member_guard.collective_timeout_secs(t_step, t_point)


def main():
    
    host_cpu.apply("train_instruct")
    argument_parser = transformers.HfArgumentParser((TrainingArguments, LoraArguments))
    (training_args, lora_args) = argument_parser.parse_args_into_dataclasses()
    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    task_id = train_request["task_id"]

    use_kl = os.environ.get("USE_KL") == "1"
    _kl_coef_env = os.environ.get("KL_COEF")
    kl_coef = 0.0
    if use_kl and _kl_coef_env:
        try:
            kl_coef = float(_kl_coef_env)
        except (ValueError, TypeError):
            log_info(f"[sn56][kl] KL_COEF inválido ({_kl_coef_env!r}), desligando KL")
            kl_coef = 0.0
    use_kl = use_kl and kl_coef > 0
    if use_kl:
        log_info(f"[sn56][kl] USE_KL=1, kl_coef={kl_coef} — termo KL ativo")

    from model_utility import load_tokenizer
    tokenizer = load_tokenizer(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _emb_rows = token_bounds.config_embedding_rows(
        train_request["model_path"],
        resize_to=(len(tokenizer) if (train_request["model_name"] in MIS_MATCH_VOCAB_SIZE_MODELS
                                      and not training_args.use_lora) else None))
    token_bounds.fit_special_tokens(tokenizer, _emb_rows, site="instruct.tokenizer")
    
        
    max_length = get_max_length_config()
    if "max_length" in train_request:
        max_length = train_request["max_length"]

    train_ds = MyDataset(
        tokenizer,
        f"datasets/train_tokenized_{task_id}.json",
        max_length
    )

    dev_ds = MyDataset(
        tokenizer,
        f"datasets/dev_tokenized_{task_id}.json",
        max_length
    )
    log_info(f"treino={len(train_ds)}, teste={len(dev_ds)}")
    token_bounds.check_rows_in_range(train_ds.eval_dataset, _emb_rows, "train dataset")
    token_bounds.check_rows_in_range(dev_ds.eval_dataset, _emb_rows, "dev dataset")

    baseline_stats = train_request.get("baseline_stats")
    near_dup_rate = 0.0
    if baseline_stats:
        near_dup_rate = baseline_stats.get("dataset", {}).get("near_duplicate_rate", 0.0)
    if near_dup_rate > 0.2:
        from data_filter import deduplicate_samples
        before = len(train_ds.eval_dataset)
        train_ds.eval_dataset = deduplicate_samples(train_ds.eval_dataset)
        log_info(f"[sn56][dedup] near_dup_rate={near_dup_rate:.2f}, deduped {before} -> {len(train_ds.eval_dataset)}")

    if baseline_stats:
        _ds = baseline_stats.get("dataset", {})
        _prompt_tok = _ds.get("prompt_tokens", 0)
        _comp_tok = _ds.get("completion_tokens", 1)
        _pc_ratio = _prompt_tok / max(_comp_tok, 1)
        if _pc_ratio > 5:
            _plw = 0.05 / max(1.0, _pc_ratio / 5.0)
            from utility import apply_prompt_loss_weight
            train_ds.eval_dataset = apply_prompt_loss_weight(train_ds.eval_dataset, plw=_plw)
            log_info(f"[sn56][plw] prompt:comp={_pc_ratio:.1f}:1, PLW={_plw:.4f}")

    donot_pack = False
    original_train_size = len(train_ds)
    original_steps = original_train_size // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    if original_steps < train_request["min_steps"]:
        donot_pack = True
        log_info(f"passos={original_steps} < mínimo={train_request['min_steps']}, sem compactação")

    min_data_size_num = (
        train_request["min_steps"]
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    
        
    log_info(f"limiar={min_data_size_num}, janela={max_length}")
    packing_mode = train_request.get("packing_mode", "fa")
    _preferred_attn = (training_args.use_attn_implementation
                       or ("flash_attention_2" if not training_args.disable_fa else "sdpa"))
    ssm_kernels.apply(train_request["model_path"])
    resolved_attn = probe_attn_implementation(train_request["model_path"], _preferred_attn)
    if resolved_attn != _preferred_attn:
        log_info(f"escada de atenção resolvida antes da compactação: "
                 f"{_preferred_attn} -> {resolved_attn}")
    training_args.use_attn_implementation = resolved_attn
    use_fa_packing = packing_mode == "fa" and resolved_attn == "flash_attention_2"
    log_info(f"máscara de fronteira: {'sim (cu_seqlens)' if use_fa_packing else 'não'}")
    _packing_off_by_attn = False
    if not use_fa_packing:
        donot_pack = True
        _packing_off_by_attn = True
        log_info(f"atenção não-FA ({packing_mode}), sem compactação — costura ingênua evitada")
    _conv_seg = False
    _gd = lfm2_posid_conv.resolve_guard(train_request["model_path"],
                                        train_request.get("model_name", ""))
    _mt, _guard = _gd.model_type, _gd.guard
    _pack_wanted = bool(training_args.packing and not donot_pack)
    log_info(f"packing_guard model_type={_mt or '?'} source={_gd.source} "
             f"decision={_guard} reason={_gd.reason} packing_requested={_pack_wanted}")
    if _gd.source == "unresolved":
        log_info("AVISO packing_guard: model_type não resolvido - compactação segue sem guarda")
    if _pack_wanted:
        if _guard == "no_pack":
            donot_pack = True
            log_info(f"estado SSM vaza entre segmentos ({_mt}, {_gd.reason}), sem compactação")
        elif _guard == "segconv":
            _conv_seg = lfm2_posid_conv.patch_short_conv(_mt)
            if _conv_seg:
                log_info(f"conv curta segmentada por position_ids instalada ({_mt})")
            else:
                donot_pack = True
                log_info(f"conv segmentada indisponível para {_mt}, sem compactação")
    _raw_train_ds = train_ds
    _train_is_packed = False
    _bp_on = batch_policy.enabled() and getattr(training_args, "deepspeed", None) is None
    _bp_plan = None
    _tb = None
    _st_step_ladder = None
    _bp_tokens_table = None
    _bp_cmd_micro = int(training_args.per_device_train_batch_size)
    _bp_cmd_accum = int(training_args.gradient_accumulation_steps)
    _bp_pack_floor = -1 if _bp_on else min_data_size_num
    if training_args.packing and not donot_pack:
        from monkeypatch import monkey_patch_packing_for_model, PackedDataset
        if use_fa_packing:
            log_info("costurando sequências com atenção rápida")
            monkey_patch_packing_for_model(
                train_request["model_path"], resolve_attn_implementation(training_args)
            )
        else:
            log_info("costura ingênua, posições reiniciadas por sequência")

        t1 = datetime.datetime.now()
        train_ds = PackedDataset(
            train_ds,
            tokenizer,
            max_input_length=max_length,
            max_packed_size=training_args.max_packed_size,
            min_item_num=_bp_pack_floor,
            use_fa=use_fa_packing,
        )
        _train_is_packed = True
        t2 = datetime.datetime.now()
        log_info(f"costura treino: {(t2 - t1).total_seconds()}s")
        if use_fa_packing:
            t1 = datetime.datetime.now()
            dev_ds = PackedDataset(
                dev_ds,
                tokenizer,
                max_input_length=max_length,
                max_packed_size=training_args.max_packed_size,
                use_fa=True,
            )
            t2 = datetime.datetime.now()
            log_info(f"costura teste: {(t2 - t1).total_seconds()}s")
        else:
            log_info("teste sem costura — isolamento necessário")
        log_info(f"treino compactado: {train_ds.stat()}")
        if hasattr(dev_ds, 'stat'):
            log_info(f"teste compactado: {dev_ds.stat()}")
        else:
            log_info(f"teste solto: {len(dev_ds)} amostras")

    if _bp_on:
        _bp_lengths = list(train_ds.lengths)
        _bp_total_tokens = float(sum(_bp_lengths))
        _bp_table_rows = _bp_cmd_micro * _bp_cmd_accum * training_args.world_size
        if _train_is_packed:
            from monkeypatch import pack_with_min_item_num
            _bp_table_groups = len(pack_with_min_item_num(_bp_lengths, train_ds.pack_length,
                                                          min_data_size_num))
            _bp_tokens_table = _bp_total_tokens * _bp_table_rows / max(1, _bp_table_groups)
        else:
            _bp_table_groups = len(train_ds)
            _bp_tokens_table = _bp_total_tokens * _bp_table_rows / max(1, len(train_ds))
        _bp_row_tokens = _bp_total_tokens / max(1, len(train_ds))
        _bp_n_rows = len(train_ds)
        if batch_policy.mode() == "tokens":
            _bp_plan = batch_policy.plan_rows(
                n_rows=_bp_n_rows, total_tokens=_bp_total_tokens,
                tokens_target=batch_policy.target_tokens(), min_steps=train_request["min_steps"],
                world_size=training_args.world_size,
                row_capacity=_bp_row_tokens,
                micro_tokens_floor=batch_policy.min_micro_tokens(),
            )
            training_args.per_device_train_batch_size = _bp_plan.micro
            training_args.gradient_accumulation_steps = 1
        else:
            _bp_plan = batch_policy.plan_from_micro(
                _bp_cmd_micro * _bp_cmd_accum, training_args.world_size, _bp_n_rows, _bp_row_tokens)
        batch_policy.emit("plan", _bp_plan, packed=_train_is_packed,
                          table_micro=_bp_cmd_micro, table_accum=_bp_cmd_accum,
                          table_rows=_bp_table_rows, table_groups=_bp_table_groups,
                          table_tokens_est=_bp_tokens_table, total_tokens=_bp_total_tokens,
                          provisional=(batch_policy.mode() != "tokens"))
        log_info(
            f"[sn56][lote] modo={batch_policy.mode()}: linhas {_bp_table_rows} -> "
            f"{_bp_plan.rows} ({_bp_plan.row_tokens:.0f} tok/linha, ~{_bp_plan.tokens_est:.0f} tok/passo, "
            f"passos/época {_bp_table_groups // max(1, _bp_table_rows)} -> {_bp_plan.steps_per_epoch}, "
            f"piso={'sim' if _bp_plan.floor_bound else 'não'})"
        )

    _packed_now = bool(training_args.packing and not donot_pack)
    _dyn_pad = dynamic_padding_enabled()
    packed_fill.arm(_packed_now)
    data_collator = DynamicPaddingCollator(
        pad_token_id=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
        max_length=max_length,
    )
    data_collator.with_seq_idx = bool(_conv_seg and _packed_now)
    data_collator.vocab_limit = _emb_rows
    _grp_on = bool(length_grouping.enabled() and _dyn_pad and not _packed_now and not use_kl and _bp_on)
    _grp_apply = bool(_grp_on and length_grouping.when_mode() == "always")
    _grp_sources = (None, None)
    if _grp_on:
        length_grouping.set_lengths(_bp_lengths)
        try:
            pass
        except Exception:
            pass
    else:
        pass
    log_info(
        f"preenchimento dinâmico: {'sim' if _dyn_pad else 'não'}"
        f"{' (compactado — inerte)' if _packed_now else ''}"
    )

    log_info(f"nós={training_args.world_size}")
    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    log_info(f"passos/época={total_steps_per_epoch}")
    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = int(max_batch_size_theory)
    if max_batch_size_theory == 0:
        max_batch_size_theory = 1

    original_batch_size = training_args.per_device_train_batch_size
    if training_args.per_device_train_batch_size > max_batch_size_theory:
        if train_request.get("adjust_batch_size", True):
            log_info(
                f"lote grande demais ({training_args.per_device_train_batch_size}), cortando para {max_batch_size_theory}"
            )
            training_args.per_device_train_batch_size = max_batch_size_theory
            total_steps_per_epoch = len(train_ds) // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
            log_info(f"passos/época recalculados={total_steps_per_epoch}")

    _a0_load_t0 = time.time()
    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        resize_if_needed(train_request["model_name"], model, len(tokenizer))
    _a0_load_secs = time.time() - _a0_load_t0
    _emb_rows = token_bounds.reconcile(model, tokenizer, data_collator, _emb_rows,
                                       site="instruct.model")
    model_sanity.check_model(model, tokenizer)
    phi_rtn_cosine.arm_from_model(model)
    ssm_kernels.fast_path_status(model)
    
    try:
        model.config.use_cache = False
    except:
        pass
    
    set_generation_config(train_request["model_name"], model)

    _bp_fp32_relaunch = False
    if _bp_plan is not None:
        _bp_saved = get_state().get("batch_policy") or {}
        _bp_relaunch_oom = bool(_bp_saved.get("micro")) and _bp_cmd_micro < int(_bp_saved.get("cmd_micro", 0) or 0)
        _bp_fp32_relaunch = precise_update.fp32_oom_relaunch_active(_bp_saved)
        if _bp_fp32_relaunch:
            precise_update.fp32_master_refuse("oom_relaunch", micro=int(_bp_saved["micro"]),
                                              accum=int(_bp_saved.get("accum", 1) or 1),
                                              path="same_plan")
        _bp_fp32_failsoft, _bp_relaunch_micro = precise_update.oom_relaunch_plan(
            _bp_saved, _bp_relaunch_oom)
        if _bp_fp32_failsoft:
            precise_update.fp32_master_refuse("oom_relaunch", micro=int(_bp_saved["micro"]),
                                              accum=int(_bp_saved.get("accum", 1) or 1))
        _bp_micro_max = _bp_relaunch_micro if _bp_relaunch_oom else _bp_plan.micro
        _bp_micro_cap = _bp_micro_max
        if torch.cuda.is_available():
            model = model.to(torch.device(f"cuda:{LOCAL_RANK}"))
        _bp_alpha = row_weight_alpha()
        _bp_loss_fn = make_probe_loss_fn(_bp_alpha, row_loss_cap()) if (_bp_alpha > 0.0 and not use_kl) else None
        _bp_probe = None
        _bp_probe_kw = dict(
            seq_len=int(max_length), vocab=int(model.get_input_embeddings().num_embeddings),
            pad_id=int(tokenizer.pad_token_id or 0), loss_fn=_bp_loss_fn,
            with_position_ids=bool(_packed_now and use_fa_packing),
            reserve_bytes=batch_policy.trainable_bytes(model),
        )
        if batch_policy.probe_enabled():
            try:
                if batch_policy.mode() in ("throughput", "v3"):
                    _bp_ref = host_cpu.reference_threads("batch_probe")
                    _bp_dev = next(model.parameters()).device
                    _bp_iid = batch_policy.dataset_rows(train_ds, data_collator, _bp_dev)
                    _bp_probe_common = dict(
                        max_micro=(_bp_micro_max if _bp_relaunch_oom else batch_policy.LADDER_MAX_MICRO),
                        training_args=training_args, row_tokens=_bp_row_tokens,
                        accum_rungs=(batch_policy.accum_rungs() if batch_policy.mode() == "v3" else ()),
                        **_bp_probe_kw)
                    with _bp_ref:
                        _bp_probe = batch_policy.probe_throughput(model, row_source=_bp_iid, **_bp_probe_common)
                    batch_policy.emit_throughput(_bp_probe, relaunch_oom=_bp_relaunch_oom, grouped=False)
                    if _grp_on:
                        _bp_pick = next((t for t in _bp_probe.tries if t.get("fits") and t.get("secs")
                                         and int(t.get("micro", 0)) == int(_bp_probe.micro)
                                         and int(t.get("accum", 1) or 1) == 1), None)
                        _bp_fixed = bool(_bp_pick is not None and batch_policy.fixed_cost_bound(_bp_pick))
                        _grp_apply = bool(length_grouping.when_mode() == "always" or _bp_fixed)
                        _grp_repick = bool(_grp_apply and length_grouping.repick_mode() != "none")
                        if _grp_repick:
                            _grp_sources = length_grouping.grouped_row_sources(
                                train_ds, data_collator, _bp_dev, _bp_lengths, _bp_row_tokens,
                                batch_policy.LADDER_MAX_MICRO, batch_policy.LADDER_ROW_SEED)
                            _grp_repick = _grp_sources[0] is not None
                        if _grp_repick:
                            with _bp_ref:
                                _bp_probe = batch_policy.probe_throughput(
                                    model, row_source=_grp_sources[0], fit_source=_grp_sources[1],
                                    **_bp_probe_common)
                            batch_policy.emit_throughput(_bp_probe, relaunch_oom=_bp_relaunch_oom,
                                                         grouped=True)
                else:
                    _bp_probe = batch_policy.probe_micro_batch(model, _bp_micro_max, **_bp_probe_kw)
                    batch_policy.emit_probe(_bp_probe, relaunch_oom=_bp_relaunch_oom)
                _bp_micro_max = min(_bp_micro_max, _bp_probe.micro)
                _st_step_ladder = second_trajectory.step_ladder_from_tries(
                    getattr(_bp_probe, "tries", None))
            except Exception as _bp_exc:
                log_info(f"[sn56][lote] sonda falhou ({_bp_exc}), mantendo lote planejado")
        if torch.distributed.is_initialized():
            _bp_t = torch.tensor([_bp_micro_max], device=next(model.parameters()).device)
            torch.distributed.all_reduce(_bp_t, op=torch.distributed.ReduceOp.MIN)
            _bp_micro_max = int(_bp_t.item())
            if _grp_on:
                _grp_t = torch.tensor([int(_grp_apply)], device=next(model.parameters()).device)
                torch.distributed.all_reduce(_grp_t, op=torch.distributed.ReduceOp.MIN)
                _grp_apply = bool(int(_grp_t.item()))
        _bp_v3 = None
        if batch_policy.mode() == "v3":
            _bp_tries = [t for t in (_bp_probe.tries if _bp_probe is not None else []) if t.get("fits")
                         and t.get("secs") and int(t.get("micro", 0)) <= _bp_micro_max]
            if _bp_tries:
                _bp_v3 = batch_policy.plan_v3(
                    _bp_tries, training_args.world_size, _bp_n_rows, _bp_row_tokens,
                    total_secs=float(train_request["hours_to_complete"]) * 3600.0,
                    epochs_plan=batch_policy.clock_epochs(float(training_args.num_train_epochs)),
                    ladder=[t for t in _bp_probe.tries if t.get("fits") and t.get("secs")])
                _bp_plan = batch_policy.plan_from_micro(
                    _bp_v3.micro, training_args.world_size, _bp_n_rows, _bp_row_tokens)
                _bp_plan.accum = _bp_v3.accum
                _bp_plan.rows = _bp_v3.rows
                _bp_plan.tokens_est = _bp_v3.tokens_est
                _bp_plan.steps_per_epoch = _bp_v3.steps_per_epoch
                if token_budget.enabled():
                    _tb_plan_kw = dict(
                        world_size=training_args.world_size, n_rows=_bp_n_rows,
                        row_tokens=_bp_row_tokens,
                        total_secs=float(train_request["hours_to_complete"]) * 3600.0,
                        epochs_plan=batch_policy.clock_epochs(float(training_args.num_train_epochs)))
                    _tb_gate = token_budget.gate(
                        switch=True, mode=batch_policy.mode(),
                        probe_fits=bool(token_budget.fitting_tries(_bp_probe.tries)),
                        dyn_pad=_dyn_pad, packed=_packed_now, use_kl=use_kl,
                        world_size=int(training_args.world_size),
                        deepspeed=getattr(training_args, "deepspeed", None) is not None,
                        r0=int(_bp_v3.rows),
                        r1=token_budget.rows_uncut(_bp_probe.tries, micro_cap=_bp_micro_cap,
                                                   **_tb_plan_kw))
                    if _tb_gate.engaged:
                        _tb = token_budget.engage(
                            model, gate=_tb_gate, probe=_bp_probe, micro_cap=_bp_micro_cap,
                            probe_kw=_bp_probe_kw, plan_kw=_tb_plan_kw, train_ds=train_ds,
                            collator=data_collator, lengths=_bp_lengths, row_tokens=_bp_row_tokens,
                            max_length=int(max_length), padding_side=tokenizer.padding_side,
                            training_args=training_args, relaunch_oom=_bp_relaunch_oom,
                            log=log_info)
                        if _tb is not None:
                            _bp_v3, _bp_probe = _tb.plan, _tb.probe
                            _bp_plan = batch_policy.plan_from_micro(
                                _bp_v3.micro, training_args.world_size, _bp_n_rows, _bp_row_tokens)
                            _st_step_ladder = second_trajectory.step_ladder_from_tries(_bp_probe.tries)
                            _tb = _tb.budget
                            member_guard.TOKEN_BUDGET = int(_tb.tokens)
            else:
                _bp_plan = batch_policy.plan_from_micro(
                    _bp_micro_max, training_args.world_size, _bp_n_rows, _bp_row_tokens)
        elif batch_policy.mode() == "throughput":
            _bp_plan = batch_policy.plan_from_micro(
                _bp_micro_max, training_args.world_size, _bp_n_rows, _bp_row_tokens)
        else:
            batch_policy.split_rows(_bp_plan, _bp_micro_max)
        if precise_update.fp32_master_wanted() and not training_args.use_lora:
            _fp32_fit = precise_update.fp32_master_fit_batch(
                _bp_probe, _bp_plan.micro, _bp_plan.accum, int(training_args.world_size or 1),
                sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
                precise_update.n_fp32_moments(model),
                sharded=bool(getattr(training_args, "deepspeed", None) is not None
                             or len(getattr(training_args, "fsdp", "") or "") > 0),
                task_id=task_id,
                worst_row_tokens=(int(max_length) if (_dyn_pad and not _packed_now) else None))
            _bp_plan.micro, _bp_plan.accum = _fp32_fit.micro, _fp32_fit.accum
        if _bp_fp32_relaunch:
            _fp32_snap = get_state().get(precise_update.FP32_SEARCH_SNAPSHOT) or None
            _rl_micro, _rl_accum, _ = precise_update.relaunch_plan_target(_bp_saved, _fp32_snap)
            _rl_v3 = (_fp32_snap or {}).get("v3")
            if _rl_v3 and _bp_v3 is not None:
                try:
                    _bp_v3 = batch_policy.V3Plan(**_rl_v3)
                except TypeError:
                    pass
            _rl_ws = max(1, int(training_args.world_size or 1))
            _bp_plan.micro, _bp_plan.accum = _rl_micro, _rl_accum
            _bp_plan.rows = _rl_micro * _rl_accum * _rl_ws
            _bp_plan.tokens_est = (float(_bp_v3.tokens_est) if (_rl_v3 and _bp_v3 is not None)
                                   else _bp_plan.rows * float(_bp_plan.row_tokens))
            _bp_plan.steps_per_epoch = max(1, int(_bp_plan.n_rows) // max(1, _bp_plan.rows))
            if _bp_v3 is not None:
                _bp_v3.micro, _bp_v3.accum, _bp_v3.rows = _rl_micro, _rl_accum, _bp_plan.rows
        training_args.per_device_train_batch_size = _bp_plan.micro
        training_args.gradient_accumulation_steps = _bp_plan.accum
        _bp_lr_before = float(training_args.learning_rate)
        _bp_scale, _bp_scale_reason = 1.0, "lr_found"
        if get_state().get("best_lr_found") is None and _bp_tokens_table:
            if batch_policy.mode() == "v3":
                _bp_scale, _bp_scale_reason = batch_policy.lr_prior_scale_fit(
                    _bp_tokens_table, _bp_plan.tokens_est)
            else:
                _bp_scale, _bp_scale_reason = batch_policy.lr_scale_for_plan(
                    _bp_table_rows, _bp_plan.rows, _bp_tokens_table, _bp_plan.tokens_est)
            training_args.learning_rate = _bp_lr_before * _bp_scale
        if _bp_v3 is not None:
            batch_policy.emit_v3("v3", _bp_v3, relaunch_oom=_bp_relaunch_oom,
                                 probe_reason=(_bp_probe.reason if _bp_probe else None),
                                 lr_scale=_bp_scale, lr_scale_reason=_bp_scale_reason,
                                 lr_before=_bp_lr_before, lr_after=float(training_args.learning_rate),
                                 table_tokens_est=_bp_tokens_table, table_rows=_bp_table_rows,
                                 **({"token_budget": _tb.tokens, "chunked": True,
                                     "rows_plan_rowsized": _tb_gate.rows_plan_rowsized,
                                     "rows_target_uncut": _tb_gate.rows_target_uncut}
                                    if _tb is not None else {}))
            log_info(
                f"[sn56][lote] v3: piso B_thr={_bp_v3.b_thr_micro} ({_bp_v3.b_thr_tok_s:.0f} tok/s), "
                f"relógio c={_bp_v3.c_thr:.2f} (máx {_bp_v3.clock_max:.2f}) -> ramo {_bp_v3.branch}: "
                f"micro={_bp_v3.micro} x accum={_bp_v3.accum} = {_bp_v3.rows} linhas; "
                f"teto B_noise medido na busca"
            )
            for _bp_fd in (getattr(_bp_probe, "rung_read", None) or {}, _bp_v3.rung_read_thr, _bp_v3.rung_read_clock):
                if _bp_fd:
                    log_info(
                        f"[sn56][lote] leitura do degrau ({_bp_fd.get('reason')}, ref={_bp_fd.get('ref_mode')}): "
                        f"accum={_bp_fd.get('allow_accum')} pico {_bp_fd.get('ref_micro')}x{_bp_fd.get('ref_accum')} "
                        f"{_bp_fd.get('ref_tok_s')} tok/s -> {_bp_fd.get('micro')}x{_bp_fd.get('accum')} razão "
                        f"{_bp_fd.get('ratio')}±{_bp_fd.get('ratio_sd')} (margem {_bp_fd.get('margin_sd')} dp, "
                        f"z={_bp_fd.get('z')}, abaixo {_bp_fd.get('ratio_below')}); leitura v3.4 "
                        f"{_bp_fd.get('raw_micro')}x{_bp_fd.get('raw_accum')}"
                    )
        if is_main_process(LOCAL_RANK):
            _bp_s = get_state()
            _bp_s["batch_policy"] = {"rows": _bp_plan.rows, "micro": _bp_plan.micro,
                                     "accum": _bp_plan.accum, "cmd_micro": _bp_cmd_micro,
                                     "fp32_master": bool(precise_update.fp32_master_on())}
            set_state(_bp_s)
        batch_policy.emit("final", _bp_plan, relaunch_oom=_bp_relaunch_oom,
                          probe_micro=(_bp_probe.micro if _bp_probe else None),
                          probe_reason=(_bp_probe.reason if _bp_probe else None),
                          lr_scale=_bp_scale, lr_scale_reason=_bp_scale_reason, lr_before=_bp_lr_before,
                          lr_after=float(training_args.learning_rate),
                          table_tokens_est=_bp_tokens_table, table_rows=_bp_table_rows)
        log_info(
            f"[sn56][lote] final: micro={_bp_plan.micro} x accum={_bp_plan.accum} x nós={_bp_plan.world_size} "
            f"= {_bp_plan.rows} linhas (~{_bp_plan.tokens_est:.0f} tok/passo, "
            f"passos/época {_bp_plan.steps_per_epoch}), lr x{_bp_scale:.3f} ({_bp_scale_reason}) "
            f"-> {training_args.learning_rate:.2e}"
            f"{' (relançamento pós-OOM)' if _bp_relaunch_oom else ''}"
        )
        if int(_bp_plan.micro) == 1:
            _bp_m1 = ((getattr(_bp_probe, "rung_read", None) or {}).get("micro_one")
                      or ("relaunch_oom" if _bp_relaunch_oom else "no_probe" if _bp_probe is None else "plan"))
            log_info(f"[sn56][lote] AVISO: micro=1 por dispositivo ({_bp_m1}); "
                     f"accum={_bp_plan.accum}, maior degrau que cabe")

    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"destino criado: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    log_info(f"salvo periódico a cada {periodic_save_steps} passos")
    training_args.save_only_model = True
    
    max_steps = train_request.get("max_steps", -1)
    log_info(f"teto de passos={max_steps}")
    
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    if "train" not in state:
        state["train"] = {}
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)
        
    total_steps_per_epoch = len(train_ds) // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
    
    total_steps_all_epochs = total_steps_per_epoch * training_args.num_train_epochs
    log_info(f"passos/época={total_steps_per_epoch}, total={total_steps_all_epochs}")

    warmup_from_ratio = max(10, int(0.03 * total_steps_all_epochs))
    training_args.warmup_steps = min(training_args.warmup_steps, warmup_from_ratio)
    log_info(f"Alongando por {training_args.warmup_steps} passos antes da corrida")

    _cm = train_request.get("checking_mode")
    is_final_run = _cm in ("none", None)
    log_info(f"checking_mode={_cm!r}, is_final_run={is_final_run}")
    if is_final_run:
        hours = train_request.get("hours_to_complete", 2)
        max_evals = max(3, int(hours * 4))
        adaptive_eval_steps = max(30, total_steps_all_epochs // min(10, max_evals))
        training_args.eval_strategy = "steps"
        training_args.eval_steps = adaptive_eval_steps
        training_args.save_strategy = "steps"
        training_args.save_steps = adaptive_eval_steps
        log_info(f"Olhando no espelho a cada {adaptive_eval_steps} passos")

    success_file = os.path.join(training_args.output_dir, "success.txt")
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)
    if is_main_process(LOCAL_RANK):
        sg.ship_marker_clear(training_args.output_dir)
    
    checking_step = train_request["checking_step"]
    if checking_step >= total_steps_per_epoch:
        checking_step = total_steps_per_epoch - 2
    
    _neft_alpha = 1
    if baseline_stats:
        grad_noise = baseline_stats.get("training", {}).get("gradient_noise_scale", 0.0)
        if grad_noise > 1.0:
            _neft_alpha = 5
            log_info(f"[sn56][neftune] grad_noise={grad_noise:.2f} > 1.0, alpha=5")
    training_args.neftune_noise_alpha = _neft_alpha
    log_info(f"[sn56][neftune] alpha={_neft_alpha}")

    _shard_ds = getattr(training_args, "deepspeed", None) is not None
    _sharded = _shard_ds or len(training_args.fsdp or "") > 0
    _trainable_bytes = sum(p.numel() for p in model.parameters() if p.requires_grad) * 2
    _avg_mode = "disk" if (_sharded or 6 * _trainable_bytes > 100e9) else "ram"
    log_info(
        f"[sn56][caldo] modo={_avg_mode} (treináveis={_trainable_bytes / 1e9:.0f}GB, "
        f"sharded={_sharded})"
    )
    ckpt_avg = (
        AdaptiveTrainingCallback(
            window=3, averaging_mode=_avg_mode, output_dir=training_args.output_dir
        )
        if is_final_run else None
    )
    if ckpt_avg is not None:
        ckpt_avg._submission_dir = train_request["submission_dir"]
        ckpt_avg.end_time = train_request["end_time"]

    _trainer_cls = KLRegularizedTrainer if use_kl else Trainer
    _row_alpha = row_weight_alpha()
    _row_cap = row_loss_cap()
    _use_row_loss = bool(_row_alpha > 0.0 and not use_kl)
    if _use_row_loss:
        _trainer_cls = RowWeightedTrainer
    _trainer_cls = with_batched_eval(_trainer_cls)
    _trainer_cls = precise_update.with_precise_update(_trainer_cls)
    if _grp_apply:
        _trainer_cls = length_grouping.with_length_grouping(_trainer_cls)
    if _tb is not None:
        _trainer_cls = token_budget.with_token_budget(_trainer_cls)
    log_info(f"[sn56][rowloss] objetivo={'por linha' if _use_row_loss else 'padrão'} alpha={_row_alpha} cap={_row_cap}")
    when_to_eval_handler = WhenToEvalHandler(
        train_request["end_time"],
        train_request["save_before_remaining_time"],
        periodic_save_steps=periodic_save_steps,
        steps_per_epoch=total_steps_per_epoch,
        max_steps=max_steps,
    )
    eval_save_cb = CustomEvalSaveCallback(
        when_to_eval_handler,
        train_request["submission_dir"],
        training_args.output_dir,
        train_request["model_name"],
        max_steps,
        checking_step=checking_step,
        total_steps_all_epochs=total_steps_all_epochs,
        end_time=train_request["end_time"],
        checking_mode=train_request.get("checking_mode", "none"),
        has_dev=bool(dev_ds is not None and len(dev_ds) > 0),
    )
    _trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        callbacks=[
            *([] if ckpt_avg is None else [ckpt_avg]),
            eval_save_cb,
        ],
    )
    if use_kl:
        _trainer_kwargs["kl_coef"] = kl_coef
        _trainer_kwargs["use_lora_base"] = bool(training_args.use_lora)
    trainer = _trainer_cls(**_trainer_kwargs)
    trainer.processing_class = tokenizer
    if _tb is not None:
        trainer.add_callback(token_budget.StepsCallback(trainer))

    if ckpt_avg is not None:
        ckpt_avg.trainer = trainer
    member_guard.install_a0_timeout(trainer, float(getattr(training_args, "ddp_timeout", 1800) or 1800))
    trainer.tokenizer = tokenizer

    _param_nums = sum(p.numel() for p in model.parameters())
    _est_step_time = 4.0 + _param_nums / 1e9 * 3.0
    _hours = train_request.get("hours_to_complete", 2)
    _eff_bs = max(1, training_args.per_device_train_batch_size
                  * training_args.gradient_accumulation_steps
                  * training_args.world_size)
    _steps_per_epoch = len(train_ds) // _eff_bs
    _est_total_steps = int(_hours * 3600 / _est_step_time)
    _est_coverage = _est_total_steps / max(1, _steps_per_epoch)
    log_info(f"[sn56][cobertura] steps/epoch={_steps_per_epoch}, est_total={_est_total_steps}, coverage~{_est_coverage:.1%} (heurística)")

    _use_deepspeed = getattr(training_args, "deepspeed", None) is not None
    t_per_step = None
    _search_pad_to = (max_length if (_dyn_pad and not _packed_now and _tb is None) else None)

    _state_cache = get_state()
    _cached_lr = _state_cache.get("best_lr_found")
    _cached_bs = _state_cache.get("best_lr_batch_size")
    _lr_pin = precise_update.lr_pin(task_id)
    if _lr_pin is not None and (_cached_lr is not None or _use_deepspeed
                                or precise_update.lr_pin_skips_search(task_id)):
        training_args.learning_rate = _lr_pin
        if _bp_plan is not None and _bp_v3 is not None and _bp_v3.t_step_final > 0:
            t_per_step = float(_bp_v3.t_step_final)
        log_info(f"[sn56][farejando] LR fixado por SN56_LR_PIN_ABS={_lr_pin:.2e}, sem busca")
    elif _cached_lr is not None:
        import math
        _current_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if _cached_bs is not None and _cached_bs != _current_bs:
            _scale = math.sqrt(_current_bs / _cached_bs)
            _cached_lr = _cached_lr * _scale
            log_info(f"[sn56][farejando] LR cached ajustado pra bs {_cached_bs}->{_current_bs}: {_cached_lr:.2e}")
        training_args.learning_rate = _cached_lr
        log_info(f"[sn56][farejando] Usando LR cached de tentativa anterior: {_cached_lr:.2e}")
        _fp32_snap = (get_state().get(precise_update.FP32_SEARCH_SNAPSHOT) or None) if _bp_fp32_relaunch else None
        if _fp32_snap:
            t_per_step = _fp32_snap.get("t_per_step")
            lr_search.LAST_SELECTION = dict(_fp32_snap.get("last_selection") or {})
            lr_search.LAST_GNS = dict(_fp32_snap.get("last_gns") or {})
            packed_fill.LR_RESCALE = float(_fp32_snap.get("lr_rescale", 1.0) or 1.0)
    elif _use_deepspeed:
        log_info(f"[sn56][farejando] Pulando (DeepSpeed ativo)")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{LOCAL_RANK}")
            model = model.to(device)
            log_info(f"[sn56][farejando] Modelo movido pra {device}")

        search_loader = trainer.get_train_dataloader()

        def _halve_batch_dataloader():
            cur = training_args.per_device_train_batch_size
            new_bs = max(1, cur // 2)
            training_args.per_device_train_batch_size = new_bs
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][farejando] Cozinha quente, lote {cur} -> {new_bs}")
            return trainer.get_train_dataloader()

        original_bs = training_args.per_device_train_batch_size

        def _live_rows_per_step():
            return max(1, training_args.per_device_train_batch_size
                       * training_args.gradient_accumulation_steps
                       * training_args.world_size)

        def _live_steps_per_epoch():
            return len(train_ds) // _live_rows_per_step()

        def _bp_recentre(log_center: float, gns_res: dict):
            new, _diag = batch_policy.recentre_log_lr(
                log_center, _bp_scale, _bp_tokens_table,
                _live_rows_per_step() * _bp_row_tokens, gns_res.get("b_noise_tokens"))
            return new

        def _remaining_secs():
            _end = datetime.datetime.strptime(
                train_request["end_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            return (_end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()

        _ext_max_epochs = train_step_clock.max_epochs_for(len(train_ds))

        _search_opt_cls, _search_opt_kw, _search_opt = precise_update.search_optimizer(
            training_args, model)
        lr_search.CARRY_LR_CAL_SUPPRESSED = (
            "fp32_master" if (_search_opt.on and _search_opt.mode == "master")
            else "phi_rtn_cosine" if phi_rtn_cosine.armed() else None)
        if not _search_opt.on:
            log_info(f"[sn56][farejando] otimizador da busca: torch AdamW ({_search_opt.reason}"
                     f"{':' + _search_opt.plan_reason if _search_opt.plan_reason else ''})")
        else:
            log_info(f"[sn56][farejando] otimizador da busca: PreciseAdamW "
                     f"{_search_opt.mode}/{_search_opt.writeback}")
        best_lr, t_per_step = run_lr_search(
            model=model,
            train_dataloader=search_loader,
            initial_lr=training_args.learning_rate,
            **({"optimizer_cls": _search_opt_cls, "optimizer_kwargs": _search_opt_kw}
               if _search_opt.on else {}),
            hours_to_complete=train_request["hours_to_complete"],
            grad_accum_steps=training_args.gradient_accumulation_steps,
            max_grad_norm=training_args.max_grad_norm,
            dataloader_factory=_halve_batch_dataloader,
            steps_per_epoch=_steps_per_epoch,
            steps_per_epoch_fn=_live_steps_per_epoch,
            restate_steps_per_epoch=_packing_off_by_attn,
            warmup_pad_to=_search_pad_to,
            warmup_pad_token_id=tokenizer.pad_token_id or 0,
            warmup_padding_side=tokenizer.padding_side,
            neftune_noise_alpha=getattr(training_args, "neftune_noise_alpha", None),
            probe_loss_fn=(make_probe_loss_fn(_row_alpha, _row_cap) if _use_row_loss else None),
            probe_horizon=(dict(n_rows=len(train_ds), rows_per_step=_live_rows_per_step(),
                                frac=batch_policy.probe_epoch_frac()) if _bp_v3 is not None else None),
            gns_cfg=(dict(world_size=training_args.world_size, rungs=batch_policy.gns_rungs(),
                          min_steps=batch_policy.gns_min_steps())
                     if (_bp_v3 is not None and batch_policy.gns_enabled()) else None),
            recentre_fn=None,
            recentre_min_dex=batch_policy.recentre_min_dex(),
            t_per_step_hint=(_bp_v3.t_step_final if (_bp_v3 is not None and _bp_v3.t_step_final > 0) else None),
            prune_rows_per_step=(_live_rows_per_step() if _bp_v3 is not None else None),
            centre_inputs=(dict(finder=_bp_lr_before, rows=_bp_plan.rows, table_rows=_bp_table_rows,
                                tokens_ratio=(_bp_plan.tokens_est / _bp_tokens_table
                                              if _bp_tokens_table else None),
                                scale=_bp_scale, scale_reason=_bp_scale_reason,
                                exponent=batch_policy.LR_CENTRE_EXPONENT)
                           if _bp_plan is not None else None),
            remaining_secs_fn=_remaining_secs,
            max_epochs_cap=_ext_max_epochs,
            chunker=(token_budget.Chunker(_tb.tokens, _tb.pad_multiple, tokenizer.padding_side)
                     if _tb is not None else None),
        )

        if torch.distributed.is_initialized():
            _bs_t = torch.tensor(
                [training_args.per_device_train_batch_size],
                device=next(model.parameters()).device,
            )
            torch.distributed.all_reduce(_bs_t, op=torch.distributed.ReduceOp.MIN)
            _min_bs = int(_bs_t.item())
            if _min_bs < training_args.per_device_train_batch_size:
                log_info(
                    f"[sn56][farejando] OOM em outro rank: lote "
                    f"{training_args.per_device_train_batch_size} -> {_min_bs} (sincronizado)"
                )
                training_args.per_device_train_batch_size = _min_bs

        if training_args.per_device_train_batch_size != original_bs:
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][farejando] Lote reduzido durante busca: {original_bs} -> {training_args.per_device_train_batch_size} (carries to training)")

        if _bp_v3 is not None:
            _gns_s1 = dict(lr_search.LAST_GNS.get("s1") or {})
            _bp_multi = bool(torch.distributed.is_initialized()
                             and int(training_args.world_size) > 1)
            if _bp_multi:
                _gns_all = [None] * int(training_args.world_size)
                torch.distributed.all_gather_object(_gns_all, _gns_s1 or None)
                _gns_pooled = lr_search.pool_gns(_gns_all, int(training_args.world_size))
                _gns_s1 = dict(_gns_pooled or {})
            if int(training_args.per_device_train_batch_size) == 1 and int(_bp_v3.micro) > 1:
                log_info(f"[sn56][lote] AVISO: micro {_bp_v3.micro} -> 1 por OOM na busca de LR")
            _bp_v3.micro = int(training_args.per_device_train_batch_size)
            _bp_v3.accum = int(training_args.gradient_accumulation_steps)
            _bp_v3.rows = _bp_v3.micro * _bp_v3.accum * training_args.world_size
            _gns_ok, _gns_why = ((lr_search.gns_reliable(_gns_s1, batch_policy.gns_min_steps()))
                                 if batch_policy.noise_cap_enabled() else (False, "disabled"))
            _bn_rows = _gns_s1.get("b_noise_rows") if _gns_ok else None
            batch_policy.cap_to_noise(_bp_v3, _bn_rows, reliable=_gns_ok)
            if torch.distributed.is_initialized():
                _cap_t = torch.tensor([_bp_v3.micro, _bp_v3.accum], device=next(model.parameters()).device)
                torch.distributed.all_reduce(_cap_t, op=torch.distributed.ReduceOp.MIN)
                _bp_v3.micro, _bp_v3.accum = int(_cap_t[0].item()), int(_cap_t[1].item())
                _bp_v3.rows = _bp_v3.micro * _bp_v3.accum * training_args.world_size
            if _bp_multi:
                _plan_box = [{"micro": _bp_v3.micro, "accum": _bp_v3.accum, "rows": _bp_v3.rows,
                              "cap_applied": bool(_bp_v3.cap_applied),
                              "cap_reason": _bp_v3.cap_reason,
                              "rows_before_cap": _bp_v3.rows_before_cap,
                              "floor_rows": _bp_v3.floor_rows,
                              "steps_per_epoch": _bp_v3.steps_per_epoch,
                              "tokens_est": _bp_v3.tokens_est,
                              "b_noise_rows": _bp_v3.b_noise_rows,
                              "b_noise_tokens": _bp_v3.b_noise_tokens,
                              "gns_ok": bool(_gns_ok), "gns_why": _gns_why}
                             if is_main_process(LOCAL_RANK) else None]
                torch.distributed.broadcast_object_list(_plan_box, src=0)
                _p0 = _plan_box[0]
                for _k in ("micro", "accum", "rows", "cap_applied", "cap_reason", "rows_before_cap",
                           "floor_rows", "steps_per_epoch", "tokens_est", "b_noise_rows",
                           "b_noise_tokens"):
                    setattr(_bp_v3, _k, _p0[_k])
                _gns_ok, _gns_why = _p0["gns_ok"], _p0["gns_why"]
                _bn_rows = _bp_v3.b_noise_rows if _gns_ok else None
                training_args.per_device_train_batch_size = int(_bp_v3.micro)
                training_args.gradient_accumulation_steps = int(_bp_v3.accum)
                trainer._train_batch_size = training_args.train_batch_size
            if _bp_v3.cap_applied and t_per_step and _bp_v3.rows_before_cap:
                _tps_before = float(t_per_step)
                t_per_step = _tps_before * float(_bp_v3.rows) / float(_bp_v3.rows_before_cap)
            if _bp_v3.cap_applied and packed_fill.active():
                _pf_scale = packed_fill.noise_cap_lr_scale(_bp_v3.rows_before_cap, _bp_v3.rows)
                if _pf_scale != 1.0:
                    _pf_lr_before = float(best_lr)
                    best_lr = _pf_lr_before * _pf_scale
                    packed_fill.LR_RESCALE = float(_pf_scale)
                    log_info(f"[sn56][lote] lr {_pf_lr_before:.2e} x ({_bp_v3.rows}/"
                             f"{_bp_v3.rows_before_cap})^{packed_fill.LR_BATCH_EXP} = {best_lr:.2e}")
            if _bp_v3.cap_applied:
                training_args.per_device_train_batch_size = _bp_v3.micro
                training_args.gradient_accumulation_steps = _bp_v3.accum
                trainer._train_batch_size = training_args.train_batch_size
                if is_main_process(LOCAL_RANK):
                    _bp_s = get_state()
                    _bp_s["batch_policy"] = {"rows": _bp_v3.rows, "micro": _bp_v3.micro,
                                             "accum": _bp_v3.accum, "cmd_micro": _bp_cmd_micro,
                                             "fp32_master": bool(precise_update.fp32_master_on())}
                    set_state(_bp_s)
                log_info(
                    f"[sn56][lote] teto de ruído: B_noise={_bn_rows:.1f} linhas < {_bp_v3.rows_before_cap} "
                    f"(piso {_bp_v3.floor_rows}) -> micro={_bp_v3.micro} x accum={_bp_v3.accum} = "
                    f"{_bp_v3.rows} linhas ("
                    f"{'lr x%.3f' % packed_fill.LR_RESCALE if packed_fill.LR_RESCALE != 1.0 else 'lr mantido'}"
                    f", {_bp_v3.cap_reason})"
                )
            else:
                log_info(f"[sn56][lote] teto de ruído não aplicado: {_bp_v3.cap_reason} ({_gns_why}), "
                         f"{_bp_v3.rows} linhas mantidas")
            batch_policy.emit_v3("noise_cap", _bp_v3, gns_status=_gns_s1.get("status"),
                                 gns_steps=_gns_s1.get("n_steps"), n_steps=_gns_s1.get("n_steps"),
                                 gns_pruned=_gns_s1.get("pruned"), gns_reason=_gns_why,
                                 gns_min_steps=batch_policy.gns_min_steps(),
                                 rows_before=_bp_v3.rows_before_cap, rows_after=_bp_v3.rows,
                                 applied=_bp_v3.cap_applied, reason=_bp_v3.cap_reason,
                                 b_noise_rows_first=_gns_s1.get("b_noise_rows_first"),
                                 b_noise_rows_warmup=(lr_search.LAST_GNS.get("warmup") or {}).get("b_noise_rows"),
                                 noise_cap_enabled=batch_policy.noise_cap_enabled(),
                                 lr_selected=float(best_lr),
                                 **({"lr_searched": float(best_lr) / packed_fill.LR_RESCALE,
                                     "packed_fill_lr_scale": packed_fill.LR_RESCALE}
                                    if packed_fill.LR_RESCALE != 1.0 else {}))

        if torch.distributed.is_initialized():
            _sync = torch.tensor([best_lr, t_per_step or 0.0],
                                 device=next(model.parameters()).device)
            torch.distributed.broadcast(_sync, src=0)
            best_lr = _sync[0].item()
            t_per_step = _sync[1].item() if _sync[1].item() > 0 else None
            for p in model.parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)
            log_info(f"[sn56][farejando] Sincronizando lr={best_lr:.2e} + weights to all ranks")

        if _lr_pin is not None:
            log_info(f"[sn56][farejando] LR da busca {best_lr:.2e} substituído por "
                     f"SN56_LR_PIN_ABS={_lr_pin:.2e}")
            best_lr = _lr_pin
        best_lr, _pin, _lr_unpinned = precise_update.apply_lr_pin(best_lr)
        if _pin != 1.0:
            log_info(f"[sn56][farejando] LR {_lr_unpinned:.2e} x pino {_pin} = {best_lr:.2e}")
        trainer.args.learning_rate = best_lr
        log_info(f"[sn56][farejando] Usando lr={best_lr:.2e}")
        _s = get_state()
        _s["best_lr_found"] = best_lr
        _s["best_lr_batch_size"] = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if precise_update.fp32_oom_fix_enabled() and precise_update.fp32_master_on():
            _s[precise_update.FP32_SEARCH_SNAPSHOT] = precise_update.search_snapshot(
                t_per_step=t_per_step, micro=training_args.per_device_train_batch_size,
                accum=training_args.gradient_accumulation_steps, rows=_s["best_lr_batch_size"],
                v3=(asdict(_bp_v3) if _bp_v3 is not None else None),
                last_selection=lr_search.LAST_SELECTION, last_gns=lr_search.LAST_GNS,
                lr_rescale=packed_fill.LR_RESCALE, search_mode=_search_opt.mode)
        if is_main_process(LOCAL_RANK):
            set_state(_s)

    if use_kl and t_per_step is not None:
        _kl_overhead = 1.4
        t_per_step = t_per_step * _kl_overhead
        log_info(f"[sn56][kl] t_per_step inflado x{_kl_overhead} p/ overhead do termo KL: {t_per_step:.3f}s")

    _cur_eff_bs = max(1, training_args.per_device_train_batch_size
                      * training_args.gradient_accumulation_steps
                      * training_args.world_size)
    _t_cov, _cov_clock_src, _cov_hint = t_per_step, "search", None
    _cov_on = bool(t_per_step is not None and batch_policy.coverage_clock_enabled())
    if (_cov_on and _bp_plan is not None and _bp_v3 is not None
            and float(_bp_v3.t_step_final) > 0):
        _cov_hint = float(_bp_v3.t_step_final)
        if _bp_v3.cap_applied and _bp_v3.rows_before_cap:
            _cov_hint = _cov_hint * float(_bp_v3.rows) / float(_bp_v3.rows_before_cap)
        _t_cov, _cov_clock_src = batch_policy.coverage_clock(
            t_per_step, _cov_hint, padded=bool(_search_pad_to), slack=lr_search._T_HINT_SLACK,
            kl_overhead=(1.4 if use_kl else 1.0))
    if t_per_step is not None:
        _cur_spe = max(1, len(train_ds) // _cur_eff_bs)
        _coverage = _hours * 3600 * 0.85 / (_cur_spe * _t_cov)
        _cov_src = "medido"
    else:
        _coverage = _est_coverage
        _cov_src = "heurística"
    _n_raw = len(_raw_train_ds.eval_dataset)
    if t_per_step is not None and _coverage < 0.5 and _n_raw > 2000:
        from utility import gaussian_subsample
        _target = max(2000, int(_coverage * _n_raw))
        if _target < _n_raw:
            _raw_train_ds.eval_dataset = gaussian_subsample(
                _raw_train_ds.eval_dataset, _target
            )
            if _train_is_packed:
                from monkeypatch import PackedDataset
                _t1 = datetime.datetime.now()
                train_ds = PackedDataset(
                    _raw_train_ds,
                    tokenizer,
                    max_input_length=max_length,
                    max_packed_size=training_args.max_packed_size,
                    min_item_num=_bp_pack_floor,
                    use_fa=use_fa_packing,
                )
                trainer.train_dataset = train_ds
                _t2 = datetime.datetime.now()
                log_info(f"[sn56][gauss] re-costura: {(_t2 - _t1).total_seconds():.1f}s")
            else:
                train_ds = _raw_train_ds
                trainer.train_dataset = train_ds
            log_info(
                f"[sn56][gauss] coverage={_coverage:.0%}<50% ({_cov_src}, relógio {_cov_clock_src}), "
                f"núcleo de {_target}/{_n_raw} amostras (~1 época no orçamento; "
                f"steps/epoch agora {len(train_ds) // _cur_eff_bs})"
            )

    _epoch_plan_inputs = None
    if t_per_step is not None and not _use_deepspeed:
        _epoch_info = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=next(model.parameters()).device)

        if is_main_process(LOCAL_RANK):
            _now = datetime.datetime.now(datetime.timezone.utc)
            _end_dt = datetime.datetime.strptime(
                train_request["end_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            remaining_secs = (_end_dt - _now).total_seconds()
            achievable_steps = int(remaining_secs * 0.85 / t_per_step)
            _current_eff_bs = max(1,
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size)
            steps_per_epoch = len(train_ds) // _current_eff_bs

            if steps_per_epoch > 0:
                _ep = train_step_clock.plan_epochs(
                    remaining_secs=remaining_secs, t_per_step=t_per_step,
                    steps_per_epoch=steps_per_epoch, n_train=len(train_ds),
                    warmup_steps_arg=training_args.warmup_steps,
                    eval_steps_arg=training_args.eval_steps, is_final_run=is_final_run,
                    source=train_step_clock.SOURCE_SEARCH)
                _max_epochs = _ep.max_epochs
                optimal_epochs = _ep.optimal_epochs
                new_total_steps = _ep.new_total_steps
                warmup = _ep.warmup
                eval_steps = _ep.eval_steps
                _epoch_info[0] = optimal_epochs
                _epoch_info[1] = float(warmup)
                _epoch_info[2] = eval_steps
                _epoch_info[3] = float(new_total_steps)
                _epoch_info[4] = float(steps_per_epoch)
                _epoch_info[5] = float(achievable_steps)
                _epoch_plan_inputs = dict(remaining_secs=float(remaining_secs),
                                          steps_per_epoch=int(steps_per_epoch),
                                          max_epochs=float(_max_epochs),
                                          optimal_epochs=float(optimal_epochs),
                                          achievable_steps=int(achievable_steps))

                log_info(
                    f"[sn56][relogio] Epochs {training_args.num_train_epochs} -> {optimal_epochs} "
                    f"(t_per_step={t_per_step:.3f}s, remaining={remaining_secs:.0f}s, "
                    f"achievable={achievable_steps}, steps/epoch={steps_per_epoch}, "
                    f"warmup={warmup}, eval_steps={int(eval_steps)})"
                )

        if torch.distributed.is_initialized():
            torch.distributed.broadcast(_epoch_info, src=0)

        if _epoch_info[0] > 0:
            training_args.num_train_epochs = _epoch_info[0].item()
            training_args.warmup_steps = int(_epoch_info[1].item())
            if is_final_run:
                training_args.eval_steps = int(_epoch_info[2].item())
                training_args.save_steps = int(_epoch_info[2].item())
            _gate = restate_eval_gate_horizon(
                eval_save_cb, int(_epoch_info[3].item()), training_args.num_train_epochs
            )
            if is_main_process(LOCAL_RANK):
                log_info(
                    f"[sn56][gate] eval gate {_gate['gate_step_before']} -> "
                    f"{_gate['gate_step_after']} (total {_gate['total_steps_all_epochs_before']} -> "
                    f"{_gate['total_steps_all_epochs_after']}, replanned "
                    f"{_gate['total_steps_all_epochs_replanned']}, clamped={_gate['clamped']})"
                )

            _hf_eval_steps = (
                int(training_args.eval_steps)
                if str(getattr(training_args, "eval_strategy", "no")).lower().endswith("steps")
                else 0
            )
            _cad = restate_eval_cadence(
                when_to_eval_handler,
                int(_epoch_info[4].item()),
                gate_step=_gate["gate_step_after"],
                horizon=int(CADENCE_HORIZON_FRACTION
                            * min(int(_epoch_info[5].item()), int(_epoch_info[3].item()))),
                total_steps=int(_epoch_info[3].item()),
                hf_eval_steps=_hf_eval_steps,
            )
            if is_main_process(LOCAL_RANK):
                log_info(
                    f"[sn56][cadencia] passos/época {_cad['steps_per_epoch_before']} -> "
                    f"{_cad['steps_per_epoch_after']} (replanejado "
                    f"{_cad['steps_per_epoch_replanned']}, clamped={_cad['clamped']}, "
                    f"floored={_cad['floored']}), avaliações "
                    f"{_cad['n_evals_before']} -> {_cad['n_evals_after']} "
                    f"em (porta {_cad['gate_step']}, horizonte {_cad['horizon']}]"
                )

    _dev_pass_secs = None
    if is_final_run and _avg_mode == "ram" and len(dev_ds) > 0 and t_per_step is not None:
        _micro_cost = t_per_step / max(
            1,
            training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size,
        )
        _blocks_per_rank = len(dev_ds) / max(1, training_args.world_size)
        _dev_pass_secs = 2.0 * _blocks_per_rank * _micro_cost + 120
        _extra_min = min(15.0, max(2.0, _dev_pass_secs / 60))
        when_to_eval_handler.save_before_remaining_time = (
            train_request["save_before_remaining_time"] + _extra_min
        )
        log_info(
            f"[sn56][devfit] orçamento ~{_dev_pass_secs:.0f}s "
            f"({len(dev_ds)} blocos dev @ micro={_micro_cost:.3f}s/rank); "
            f"parada final em T-{when_to_eval_handler.save_before_remaining_time:.1f}min"
        )

    if use_kl and not training_args.use_lora:
        _kl_attn = "flash_attention_2" if not training_args.disable_fa else "sdpa"
        if training_args.use_attn_implementation:
            _kl_attn = training_args.use_attn_implementation
        _kl_device = next(model.parameters()).device
        trainer.base_model = load_base_model(train_request["model_path"], _kl_attn, _kl_device)
        log_info(f"[sn56][kl] modelo base congelado carregado em {_kl_device} (full-FT)")
    elif use_kl:
        log_info("[sn56][kl] usando disable_adapter do LoRA como base (sem cópia)")

    log_info(f"fiscalização a cada {training_args.eval_steps} passos, {len(dev_ds)} na prova")

    _ch_logging_saved = None
    _ch_on = bool(clock_hygiene.enabled() and is_final_run)
    if _ch_on and training_args.gradient_checkpointing and not training_args.use_lora \
            and not use_kl and max_length and int(max_length) > 0 \
            and not is_deepspeed_zero3_enabled() and training_args.world_size == 1:
        try:
            _ch_probe = clock_hygiene.gc_off_fits(
                model, int(training_args.per_device_train_batch_size), int(max_length or 0),
                vocab=int(model.get_input_embeddings().num_embeddings),
                pad_id=int(tokenizer.pad_token_id or 0),
                loss_fn=(make_probe_loss_fn(_row_alpha, _row_cap) if _use_row_loss else None),
                with_position_ids=bool(_packed_now and use_fa_packing),
                training_args=training_args)
        except Exception as _ch_exc:
            _ch_probe = clock_hygiene.GcProbe(gc_off=False, reason=f"probe_raise:{type(_ch_exc).__name__}",
                                              micro=int(training_args.per_device_train_batch_size),
                                              seq_len=int(max_length or 0))
        clock_hygiene.apply_gc_decision(model, training_args, _ch_probe)
        if not _ch_probe.gc_off:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=(training_args.gradient_checkpointing_kwargs
                                               or {"use_reentrant": False}))
        log_info(f"[sn56][relógio] checkpointing de gradiente {'DESLIGADO' if _ch_probe.gc_off else 'mantido'} "
                 f"({_ch_probe.reason}, pico {_ch_probe.peak_gb}GB / orçamento {_ch_probe.budget_gb}GB)")
    pass
    if _ch_on:
        _ch_keep_steps = bool(ckpt_avg is not None and stop_rule_mode() == "vot"
                              and second_trajectory.members_on_world(
                                  int(training_args.world_size)))
        _ch_logging_saved = clock_hygiene.a0_logging(training_args, keep_steps=_ch_keep_steps)

    if _dyn_pad and not _packed_now and max_length and max_length > 0:
        import transformers.integrations.neftune as _neft

        def _pinned_neftune_hook(module, input, output, _cap=int(max_length)):
            if module.training:
                dims = torch.tensor(_cap * output.size(2))
                mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
                output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output

        _neft.neftune_post_forward_hook = _pinned_neftune_hook

    for _k in data_collator.stats:
        data_collator.stats[_k] = 0

    sg.install_save_guard(
        trainer,
        stop_reason=run_state.STOP_REASON,
        submission_dir=train_request["submission_dir"],
        base_model_path=train_request["model_path"],
    )
    if is_final_run:
        async_save.install(
            trainer,
            avg_mode=_avg_mode,
            submission_dir=train_request["submission_dir"],
            base_model_path=train_request["model_path"],
        )
    def _st_static_facts() -> dict:
        _ds = trainer.train_dataset
        return dict(
            steps_per_epoch=max(1, len(_ds) // max(1,
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size)),
            num_train_epochs=float(training_args.num_train_epochs),
            eval_steps=int(training_args.eval_steps or 0),
            t_per_step_search=t_per_step,
            learning_rate=float(trainer.args.learning_rate),
            seed=int(training_args.seed),
            argmin_lr=packed_fill.scaled_argmin(lr_search.LAST_SELECTION.get("argmin_lr")),
            pool_tier=lr_search.LAST_SELECTION.get("pool_tier"),
            world_size=int(training_args.world_size),
            avg_mode=_avg_mode,
            is_final_run=bool(is_final_run),
            warmup_steps=int(getattr(training_args, "warmup_steps", 0) or 0),
            min_lr_rate=float((training_args.lr_scheduler_kwargs or {}).get(
                "min_lr_rate", 0.0) or 0.0),
            lr_scheduler_type=str(getattr(training_args, "lr_scheduler_type", "")),
            total_secs=max(0.0, float(train_request.get("hours_to_complete", 2))
                           * 3600.0 - 180.0),
            t_load_a0=float(_a0_load_secs),
        )

    _devfit_ok = bool(is_final_run and _avg_mode == "ram"
                      and dev_ds is not None and len(dev_ds) > 0)
    if ckpt_avg is not None:
        try:
            ckpt_avg.stop_rule_facts = _st_static_facts()
            ckpt_avg.stop_rule_cadences = lambda: (
                int(when_to_eval_handler.steps_per_epoch),
                int(when_to_eval_handler.periodic_save_steps),
                int(training_args.eval_steps or 0))
            ckpt_avg.stop_rule_devpass_reserve = (
                max(second_trajectory._DEVPASS_RESERVE_MIN_SECS,
                    float(_dev_pass_secs or 0.0)) if _devfit_ok else 0.0)
            ckpt_avg.std_guard_secs = float(
                getattr(when_to_eval_handler, "save_before_remaining_time", 0) or 0) * 60.0
            ckpt_avg.std_gate_step = lambda: eval_gate_step(
                eval_save_cb.total_steps_all_epochs, training_args.num_train_epochs)
        except Exception:
            ckpt_avg.stop_rule_facts = None
    _st_rng_state = None
    try:
        _st_rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    except Exception:
        _st_rng_state = None
    _ts_clock_cb = None
    _depth = None
    if t_per_step is None and train_step_clock.enabled():
        _ts_clock_cb = train_step_clock.TrainStepClockCallback(
            source=(train_step_clock.SOURCE_DS if _use_deepspeed else train_step_clock.SOURCE_TRAIN),
            end_time=train_request["end_time"],
            n_train_fn=lambda: len(trainer.train_dataset),
            is_final_run=bool(is_final_run),
            eval_save_cb=eval_save_cb,
            when_to_eval_handler=when_to_eval_handler,
            cadence_horizon_fraction=CADENCE_HORIZON_FRACTION,
            restate_gate=restate_eval_gate_horizon,
            restate_cadence=restate_eval_cadence,
            log=log_info,
        )
        trainer.add_callback(_ts_clock_cb)
    elif (t_per_step is not None and train_step_clock.enabled() and train_step_clock.extend_enabled()
          and packed_fill.extension_reason(_dyn_pad, _packed_now, _use_deepspeed) is not None
          and (_tb is None or depth_first.should_arm(_dyn_pad, _packed_now))):
        _ts_end_time = train_request["end_time"]
        _pf_extend = (packed_fill.extension_reason(_dyn_pad, _packed_now, _use_deepspeed)
                      == packed_fill.EXT_PACKED_FILL)
        _ts_gate = train_step_clock.no_member_fits_gate(
            remaining_fn=lambda: train_step_clock.remaining_secs_until(_ts_end_time),
            planned_steps_fn=lambda: int(
                (_depth.legacy_h if (_depth is not None and _depth.legacy_h)
                 else getattr(trainer.state, "max_steps", 0)) or 0),
            **({"reserve_fn": lambda: second_trajectory.measured_tail_secs(ckpt_avg)}
               if _pf_extend else {}))
        if depth_first.should_arm(_dyn_pad, _packed_now):
            _depth = depth_first.DepthFirst(
                trainer=trainer, ckpt_avg=ckpt_avg, handler=when_to_eval_handler,
                eval_save_cb=eval_save_cb, restate_gate=restate_eval_gate_horizon, log=log_info)
        _ts_clock_cb = train_step_clock.TrainStepClockCallback(
            source=train_step_clock.SOURCE_TRAIN,
            end_time=_ts_end_time,
            n_train_fn=lambda: len(trainer.train_dataset),
            is_final_run=False,
            eval_save_cb=eval_save_cb,
            when_to_eval_handler=when_to_eval_handler,
            cadence_horizon_fraction=CADENCE_HORIZON_FRACTION,
            restate_gate=restate_eval_gate_horizon,
            restate_cadence=restate_eval_cadence,
            log=log_info,
            extend_only=True,
            measure=train_step_clock.extend_measure_steps(t_per_step),
            apply_gate=(_ts_gate if _depth is None else _depth.gate(_ts_gate)),
            remeasure_at=(train_step_clock.extend_remeasure_at() or None),
            **({"depth": _depth} if _depth is not None else {}),
        )
        trainer.add_callback(_ts_clock_cb)
        if _depth is not None:
            _depth.clock_cb = _ts_clock_cb
            trainer.add_callback(_depth)
            _depth_loop = depth_first.install_loop(trainer, _depth.cap)
            if ckpt_avg is not None:
                ckpt_avg.depth_first = _depth
            second_trajectory.DEPTH_MEMBER_MIN_FRAC = depth_first.MEMBER_MIN_FRAC
    if ckpt_avg is not None and _ts_clock_cb is not None:
        ckpt_avg.std_clock_cb = _ts_clock_cb
    _a0_clock_cb = second_trajectory.A0ClockCallback()
    trainer.add_callback(_a0_clock_cb)
    if ckpt_avg is not None:
        ckpt_avg.planner_clock = _a0_clock_cb
    _a0_mem_cb = member_guard.A0MemoryCallback()
    trainer.add_callback(_a0_mem_cb)
    trainer.add_callback(rank_agreement.RankAgreementCallback("a0"))
    _wsd_info = None
    if ((a0_cosine.enabled() or wsd_a0.enabled())
            and "cosine_with_min_lr" in str(training_args.lr_scheduler_type).lower()):
        _wsd_kw = training_args.lr_scheduler_kwargs or {}
        if isinstance(_wsd_kw, str):
            try:
                _wsd_kw = json.loads(_wsd_kw)
            except (json.JSONDecodeError, ValueError):
                _wsd_kw = {}
        try:
            _wsd_min = float(_wsd_kw.get("min_lr_rate", 0.25))
        except (AttributeError, TypeError, ValueError):
            _wsd_min = 0.25
        if a0_cosine.enabled():
            a0_cosine.install(trainer, min_lr_rate=_wsd_min)
        else:
            _wsd_info = wsd_a0.install(trainer, min_lr_rate=_wsd_min)
            if ckpt_avg is not None and isinstance(ckpt_avg.stop_rule_facts, dict):
                ckpt_avg.stop_rule_facts["lr_scheduler_type"] = "warmup_stable_decay"
    _early = None
    if early_evals.enabled() and is_final_run and ckpt_avg is not None and dev_ds is not None \
            and len(dev_ds) > 0:
        _ee_gate = eval_gate_step(eval_save_cb.total_steps_all_epochs, training_args.num_train_epochs)
        if not early_evals.single_process():
            pass
        else:
            _ee_rows = max(1, training_args.per_device_train_batch_size
                           * training_args.gradient_accumulation_steps)
            _early = early_evals.EarlyEvals(
                steps_per_epoch=max(1, len(trainer.train_dataset) // _ee_rows),
                budget_secs=float(train_request.get("hours_to_complete", 2) or 2) * 3600.0,
                unit_prior=early_evals.unit_prior(
                    n_dev=len(dev_ds), rows_per_step=_ee_rows, t_per_step=t_per_step,
                    trainable_bytes=float(_trainable_bytes)),
                t_step_prior=t_per_step)
            if _early.decide(int(_ee_gate))["ok"]:
                eval_save_cb.early_evals = _early
                trainer.add_callback(_early)
                log_info(f"[sn56][w] avaliações precoces a cada {_early.cad} passos antes da porta "
                         f"{_ee_gate}")
            else:
                _early = None
    try:
        pass
    except Exception:
        pass
    if _grp_apply:
        length_grouping.activate(True)
    _a0_clock_cb.clock.train_call(time.time())
    _train_output = trainer.train()
    if _depth is not None:
        second_trajectory.DEPTH_MEMBERS = not bool(_depth.disarmed)
    if _early is not None:
        eval_save_cb.early_evals = None
        try:
            trainer.remove_callback(_early)
        except Exception:
            pass
    _mem_proof = None
    try:
        _mem_rows_max = member_guard.row_len_max(
            (getattr(trainer, "train_dataset", None), getattr(trainer, "eval_dataset", None)),
            max_length)
    except Exception:
        _mem_rows_max = None
    _mem_proof = member_guard.agree(
        _a0_mem_cb, training_args=training_args, max_length=max_length,
        pad_multiple=int(getattr(data_collator, "pad_to_multiple_of", None) or 1),
        row_len=_mem_rows_max)
    if is_final_run and is_main_process(LOCAL_RANK):
        _own0 = sg.submission_owner(training_args.output_dir)
        _rec0 = _own0.get("record") or {}
        sg.ship_marker_init(training_args.output_dir, train_request["submission_dir"],
                            shipped=(f"attempt{_own0['attempt']}_soup" if _own0["owns"]
                                     else f"attempt{_rec0.get('attempt')}_step{_rec0.get('step')}"),
                            shipped_loss=((getattr(ckpt_avg, "best_loss", None)
                                           if ckpt_avg is not None else None)
                                          if _own0["owns"] else _rec0.get("loss")))


    _st_out = second_trajectory.Outcome()
    _st_facts = None
    try:
        _st_train_ds = trainer.train_dataset
        _st_eval_ds = trainer.eval_dataset
        _st_base_model = getattr(trainer, "base_model", None)
        _st_collator = trainer.data_collator
        _st_metrics = getattr(_train_output, "metrics", None) or {}
        _st_eval_runtime = None
        for _entry in reversed(getattr(trainer.state, "log_history", []) or []):
            if "eval_runtime" in _entry:
                _st_eval_runtime = _entry.get("eval_runtime")
                break
        _st_facts = second_trajectory.Attempt0Facts(
            **_st_static_facts(),
            best_step=getattr(ckpt_avg, "best_step", None) if ckpt_avg is not None else None,
            best_loss=getattr(ckpt_avg, "best_loss", None) if ckpt_avg is not None else None,
            steps_run=int(getattr(trainer.state, "global_step", 0) or 0),
            n_evals=sum(1 for e in (getattr(trainer.state, "log_history", []) or [])
                        if "eval_loss" in e),
            train_runtime=_st_metrics.get("train_runtime"),
            eval_runtime=_st_eval_runtime,
            horizon_steps=int(getattr(trainer.state, "max_steps", 0) or 0),
            stop_reason=(run_state.STOP_REASON.get("reason") or "epochs_complete"),
            clock=_a0_clock_cb.stats_dict(),
            memory=_mem_proof.model_dump() if _mem_proof is not None else None,
            owns_submission=sg.submission_owner(training_args.output_dir)["owns"],
            soup_secs_per_cand=(getattr(ckpt_avg, "soup_secs_per_cand", None)
                                if ckpt_avg is not None else None),
            step_ladder=_st_step_ladder,
            n_dev=len(_st_eval_ds) if _st_eval_ds else 0,
            a0_trial_secs=(list(getattr(ckpt_avg, "soup_trial_secs", None) or [])
                           if ckpt_avg is not None else None),
            a0_soup_steps=([int(e["step"]) for e in (getattr(ckpt_avg, "soup_accepted", None) or [])
                            if isinstance(e, dict) and e.get("step") is not None
                            and e.get("src", "a0") == "a0"]
                           if ckpt_avg is not None else None),
            a0_stop_to_decay=bool(getattr(ckpt_avg, "std_active", False)
                                  if ckpt_avg is not None else False),
        )
        if _wsd_info is not None:
            _st_facts.lr_scheduler_type = "warmup_stable_decay"
        _emit_epoch_plan_restated(_st_facts, t_per_step, _epoch_plan_inputs)

        def _st_build_model():
            
            if training_args.use_lora:
                m = load_lora_model(training_args, train_request["model_path"],
                                    lora_args, len(tokenizer))
            else:
                m = load_model(training_args, train_request["model_path"], len(tokenizer))
                resize_if_needed(train_request["model_name"], m, len(tokenizer))
            try:
                m.config.use_cache = False
            except Exception:
                pass
            set_generation_config(train_request["model_name"], m)
            return m

        def _st_build_trainer(_model, _args):
            
            _kw = dict(
                model=_model,
                processing_class=tokenizer,
                args=_args,
                train_dataset=_st_train_ds,
                eval_dataset=_st_eval_ds,
                data_collator=_st_collator,
                callbacks=([]),
            )
            if use_kl:
                _kw["kl_coef"] = kl_coef
                _kw["use_lora_base"] = bool(training_args.use_lora)
            _t = _trainer_cls(**_kw)
            _t.precise_phase = "member"
            _t.processing_class = tokenizer
            _t.tokenizer = tokenizer
            if _st_base_model is not None:
                _t.base_model = _st_base_model
            return _t

        _min_lr_rate = 0.25
        _kw = training_args.lr_scheduler_kwargs or {}
        if isinstance(_kw, str):
            try:
                _kw = json.loads(_kw)
            except (json.JSONDecodeError, ValueError):
                _kw = {}
        try:
            _min_lr_rate = float(_kw.get("min_lr_rate", 0.25))
        except (AttributeError, TypeError, ValueError):
            _min_lr_rate = 0.25
        _devfit_plan = second_trajectory.DevfitPlan(
            enabled=_devfit_ok,
            n_dev=len(dev_ds),
            min_lr=trainer.args.learning_rate * _min_lr_rate,
            min_lr_rate=_min_lr_rate,
            max_grad_norm=training_args.max_grad_norm,
            train_per_device=training_args.per_device_train_batch_size,
            train_grad_accum=training_args.gradient_accumulation_steps,
            local_rank=LOCAL_RANK,
            budget_secs_a0=_dev_pass_secs,
            eval_runtime=_st_eval_runtime,
            skip_reason=("" if _devfit_ok else
                         f"final={is_final_run},mode={_avg_mode},dev={len(dev_ds)}"),
            memory=_mem_proof.model_dump() if _mem_proof is not None else None,
            collective_timeout=_devfit_collective_timeout(_st_facts, training_args),
            ddp_timeout=float(getattr(training_args, "ddp_timeout", 1800) or 1800),
            dev_micro_batch=DEV_PASS_MICRO_BATCH,
        )

        model = None
        clock_hygiene.restore_trajectory_logging(training_args, _ch_logging_saved)
        _st_out = second_trajectory.run(
            ckpt_avg=ckpt_avg, trainer=trainer, training_args=training_args,
            train_request=train_request, facts=_st_facts,
            model_builder=_st_build_model, trainer_builder=_st_build_trainer,
            rng_state=_st_rng_state, devfit=_devfit_plan, log=log_info,
        )
    except Exception as _e:
        log_info(f"[sn56][traj2] wiring falhou ({_e})")
    _st_mutated = bool(getattr(_st_out, "mutated", False))
    _st_devfit = getattr(_st_out, "devfit", None)

    if _st_devfit is not None:
        log_info(
            f"[sn56][devfit] decidido pós-soup: ran={_st_devfit.get('ran')} "
            f"({_st_devfit.get('reason')}), submissão={_st_devfit.get('shipped')}"
        )
    else:
        try:
            _dev_n = len(dev_ds)
            _dev_window_ok = True
            _dev_skip_reason = None
            _dev_budget = _dev_pass_secs
            if _dev_budget is None and is_final_run and _avg_mode == "ram" and _dev_n > 0:
                _clk_dp = _st_facts.clock_stats if _st_facts is not None else None
                if _clk_dp is not None:
                    _dev_budget = planner_clock.tail_reserve(
                        _clk_dp, None, None, devfit_enabled=True,
                        devpass_legacy=0.0).devpass_reserve
                else:
                    _dev_window_ok, _dev_skip_reason = False, "no_budget_estimate"
            if _dev_budget is not None:
                _end_dt = datetime.datetime.strptime(
                    train_request["end_time"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=datetime.timezone.utc)
                _remaining = (_end_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                _dev_window_ok = _remaining > _dev_budget
                if not _dev_window_ok:
                    _dev_skip_reason = "no_window"
                    log_info(
                        f"[sn56][devfit] sem janela ({_remaining:.0f}s restantes < "
                        f"~{_dev_budget:.0f}s necessários), pulando"
                    )
            _dev_own = sg.submission_owner(training_args.output_dir)
            if _dev_window_ok and not _dev_own["owns"]:
                _dev_window_ok, _dev_skip_reason = False, "shipped_model_not_in_memory"
                log_info(f"[sn56][devfit] submissão é da tentativa "
                         f"{(_dev_own['record'] or {}).get('attempt')}, não da memória; pulando")
            _dev_window_ok, _dev_skip_reason = second_trajectory._r0(
                (bool(_dev_window_ok), _dev_skip_reason))
            if logit_calib.enabled() and is_final_run and _avg_mode == "ram" and _dev_n > 0 \
                    and _dev_window_ok and not _st_mutated:
                logit_calib.calibrate_a0_ship(
                    trainer, ckpt_avg, train_request, _st_facts, _dev_budget,
                    second_trajectory._multi(), is_main_process(LOCAL_RANK), log_info)
            if is_final_run and _avg_mode == "ram" and _dev_n > 0 and _dev_window_ok \
                    and not _st_mutated:
                from dev_pass import run_dev_pass
                _min_lr_rate = 0.25
                _kw = training_args.lr_scheduler_kwargs or {}
                if isinstance(_kw, str):
                    try:
                        _kw = json.loads(_kw)
                    except (json.JSONDecodeError, ValueError):
                        _kw = {}
                try:
                    _min_lr_rate = float(_kw.get("min_lr_rate", 0.25))
                except (AttributeError, TypeError, ValueError):
                    pass
                _min_lr = trainer.args.learning_rate * _min_lr_rate
                log_info(
                    f"[sn56][devfit] {_dev_n} amostras de dev, 1 época @ lr={_min_lr:.2e}"
                )
                _dp_res = run_dev_pass(
                    trainer,
                    submission_dir=train_request["submission_dir"],
                    min_lr=_min_lr,
                    max_grad_norm=training_args.max_grad_norm,
                    train_per_device=training_args.per_device_train_batch_size,
                    train_grad_accum=training_args.gradient_accumulation_steps,
                    local_rank=LOCAL_RANK, log=log_info,
                    micro_batch=DEV_PASS_MICRO_BATCH,
                    memory_proof=(_mem_proof.model_dump() if _mem_proof is not None else None),
                    collective_timeout=_devfit_collective_timeout(_st_facts, training_args),
                    restore_timeout=float(getattr(training_args, "ddp_timeout", 1800) or 1800),
                ) or {}
            else:
                log_info(
                    f"[sn56][devfit] pulando (final={is_final_run}, modo={_avg_mode}, "
                    f"dev={_dev_n})"
                )
        except Exception as _e:
            log_info(f"[sn56][devfit] dev-pass falhou ({_e}), mantendo melhor checkpoint")

    if is_main_process(LOCAL_RANK):
        success_file = os.path.join(training_args.output_dir, "success.txt")
        with open(success_file, "w") as f:
            f.write("Success")
        sg.ship_marker_update(stage="done", activity="done")
    log_info("missão cumprida", "finish")

if __name__ == "__main__":
    main()
