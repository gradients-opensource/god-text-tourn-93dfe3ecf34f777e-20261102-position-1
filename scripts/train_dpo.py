import quiet_mode
import host_cpu
import rank_agreement
import token_bounds
from typing import Dict, Optional
import requests
import json
import random
import utility
from datasets import Dataset
from utility import log_info
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from trl import DPOTrainer, DPOConfig, ModelConfig, ScriptArguments, TrlParser
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
from transformers import TrainerCallback
import argparse
import math
import os
import sys
from customized_trainer import resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb
from checkpoint_avg_callback import AdaptiveTrainingCallback
from state_manager import get_state, set_state
import member_guard

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
import datetime
import shutil
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
import yaml
from tokenize_dpo import get_dataset
from lr_search import run_lr_search
import lr_search
import model_sanity
import second_trajectory
import run_state
from collections import defaultdict
from transformers.modeling_utils import is_deepspeed_zero3_enabled


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def get_kbit_device_map() -> Optional[Dict[str, int]]:
    
    if torch.cuda.is_available():
        return {"": PartialState().local_process_index}
    return None


def _warn(msg: str) -> None:
    try:
        sys.__stderr__.write(f"[sn56][dpo] {msg}\n")
        sys.__stderr__.flush()
    except Exception:
        pass
    log_info(msg)


def warn_padding_free_noop(training_args) -> bool:
    
    if not training_args.padding_free:
        return False
    _warn(
        "padding_free=True is a NO-OP in trl 1.10 (falls back to standard padding). "
        f"per_device_train_batch_size={training_args.per_device_train_batch_size} was tuned "
        "on 0.18 padding-free: expect lower throughput and possibly an OOM backoff."
    )
    return True


def load_reference_model(model_path: str, model_kwargs: dict, model_name: str, n_tokens: int):
    
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    resize_if_needed(model_name, ref_model, n_tokens)
    ref_model.requires_grad_(False)
    ref_model.eval()
    return ref_model


@dataclass
class TrainingArguments(DPOConfig):
    request_path: Optional[str] = field(default=None)
    use_liger: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)
    use_attn_implementation: Optional[str] = field(default="")


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


def dpo_max_epochs(n_train: int) -> float:
    return 4.0 if n_train < 10000 else 3.0


def min_lr_rate_of(training_args, default: float = 0.25) -> float:
    kw = training_args.lr_scheduler_kwargs or {}
    if isinstance(kw, str):
        try:
            kw = json.loads(kw)
        except (json.JSONDecodeError, ValueError):
            kw = {}
    try:
        return float(kw.get("min_lr_rate", default))
    except (AttributeError, TypeError, ValueError):
        return default


def remaining_secs_until(end_time: str) -> float:
    end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=datetime.timezone.utc)
    return (end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()


_DPO_ID_COLUMNS = ("prompt_ids", "chosen_ids", "rejected_ids")


def live_null_rows(zero_ids: list, *datasets) -> list:
    want = set(int(i) for i in zero_ids)
    seen: set = set()
    for ds in datasets:
        if ds is None or not want - seen:
            continue
        cols = [c for c in _DPO_ID_COLUMNS if c in (getattr(ds, "column_names", None) or [])]
        for col in cols:
            for row in ds[col]:
                seen |= want.intersection(row)
                if not want - seen:
                    break
    return sorted(seen)


@dataclass
class NullRowPin:
    ids: list = field(default_factory=list)
    pinned: bool = False
    reason: str = ""
    handle: Optional[object] = None


def pin_null_embedding_rows(model, ids: list) -> NullRowPin:
    if not ids:
        return NullRowPin(reason="no_zero_rows")
    try:
        w = model.get_input_embeddings().weight
    except Exception as exc:
        return NullRowPin(ids=list(ids), reason=f"no_embedding:{type(exc).__name__}")
    if not w.requires_grad:
        return NullRowPin(ids=list(ids), reason="embedding_frozen")
    idx = torch.tensor(list(ids), dtype=torch.long, device=w.device)

    def _zero_rows_grad(grad):
        return grad.index_fill(0, idx.to(grad.device), 0)

    return NullRowPin(ids=list(ids), pinned=True, reason="pinned",
                      handle=w.register_hook(_zero_rows_grad))


_LAYER_STACK_NAMES = ("layers", "h", "blocks", "layer")


def decoder_layer_stack(model):
    best_name, best = None, None
    for name, mod in model.named_modules():
        if (isinstance(mod, torch.nn.ModuleList) and len(mod) > 0
                and name.rsplit(".", 1)[-1] in _LAYER_STACK_NAMES
                and (best is None or len(mod) > len(best))):
            best_name, best = name, mod
    if best is None:
        return None, []
    owner = model.get_submodule(best_name.rsplit(".", 1)[0]) if "." in best_name else model
    return owner, list(best)


class NullPositionDetach:

    def __init__(self, ids, stage: str = "a0"):
        self.ids = sorted(int(i) for i in ids)
        self.stage = stage
        self.handles: list = []
        self.n_layers = 0
        self.mask = None
        self.has = None
        self.n_fallback = 0
        self.first_counts: Optional[list] = None
        self._counts: dict = {}
        self._ids_t = None

    def install(self, model) -> bool:
        if not self.ids:
            return False
        try:
            emb = model.get_input_embeddings()
        except Exception:
            emb = None
        owner, layers = decoder_layer_stack(model)
        if emb is None or owner is None or not layers:
            return False
        self._ids_t = torch.tensor(self.ids, dtype=torch.long)
        self.handles.append(owner.register_forward_pre_hook(self._clear))
        self.handles.append(emb.register_forward_pre_hook(self._tokens, with_kwargs=True))
        for i, layer in enumerate(layers):
            self.handles.append(layer.register_forward_pre_hook(
                self._make_layer_hook(i), with_kwargs=True))
        self.n_layers = len(layers)
        return True

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def _clear(self, module, args):
        self.mask, self.has = None, None

    def _tokens(self, module, args, kwargs):
        ids = args[0] if args else kwargs.get("input")
        if not isinstance(ids, torch.Tensor) or ids.is_floating_point():
            self.mask, self.has = None, None
            return None
        if self._ids_t.device != ids.device:
            self._ids_t = self._ids_t.to(ids.device)
        self.mask = torch.isin(ids, self._ids_t)
        self.has = bool(self.mask.any())
        return None

    def _make_layer_hook(self, idx: int):
        def hook(module, args, kwargs):
            if self.has is False or not torch.is_grad_enabled():
                return None
            hs = args[0] if args else kwargs.get("hidden_states")
            if not isinstance(hs, torch.Tensor) or not hs.requires_grad or hs.dim() < 2:
                return None
            null = (hs == 0).all(dim=-1)
            tok = self.mask
            if tok is not None and tuple(tok.shape) == tuple(null.shape):
                null = null & tok.to(null.device)
            else:
                self.n_fallback += 1
            out = torch.where(null.unsqueeze(-1), hs.detach(), hs)
            if self.first_counts is None:
                self._counts[idx] = int(null.sum())
                if len(self._counts) == self.n_layers:
                    self.first_counts = [self._counts[i] for i in range(self.n_layers)]
            if args:
                return (out,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = out
            return args, kwargs
        return hook


def install_null_position_detach(model, ids, stage: str = "a0") -> Optional[NullPositionDetach]:
    if not ids:
        return None
    d = NullPositionDetach(ids, stage=stage)
    return d if d.install(model) else None


_PEFT_PREFIXES = ("base_model.model.", "module.", "_orig_mod.")
_EMBED_KEYS = ("model.embed_tokens.weight", "transformer.wte.weight", "gpt_neox.embed_in.weight",
               "transformer.word_embeddings.weight", "model.decoder.embed_tokens.weight",
               "transformer.embedding.word_embeddings.weight", "embed_tokens.weight")


@dataclass
class CheckpointZeroRows:
    ids: list = field(default_factory=list)
    ok: bool = False
    key: Optional[str] = None
    file: Optional[str] = None
    n_rows: Optional[int] = None
    error: Optional[str] = None


def _input_embedding_param_name(model) -> Optional[str]:
    try:
        w = model.get_input_embeddings().weight
    except Exception:
        return None
    for name, p in model.named_parameters():
        if p is w:
            for pre in _PEFT_PREFIXES:
                while name.startswith(pre):
                    name = name[len(pre):]
            return name
    return None


def _checkpoint_tensor_index(model_path: str) -> dict:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(model_path, index_name)
        if os.path.isfile(p):
            with open(p) as fh:
                return {k: os.path.join(model_path, f) for k, f in json.load(fh)["weight_map"].items()}
    p = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(p):
        from safetensors import safe_open
        with safe_open(p, framework="pt", device="cpu") as fh:
            return {k: p for k in fh.keys()}
    p = os.path.join(model_path, "pytorch_model.bin")
    if os.path.isfile(p):
        sd = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
        return {k: p for k in sd.keys()}
    return {}


def _pick_embedding_key(keys, live_name: Optional[str], tied: bool) -> Optional[str]:
    keys = set(keys)
    if live_name and live_name in keys:
        return live_name
    if live_name:
        tail = ".".join(live_name.split(".")[-2:])
        hits = sorted((k for k in keys if k == tail or k.endswith("." + tail)), key=len)
        if hits:
            return hits[0]
    for k in _EMBED_KEYS:
        if k in keys:
            return k
    if tied and "lm_head.weight" in keys:
        return "lm_head.weight"
    return None


@torch.no_grad()
def checkpoint_embedding_zero_rows(model_path: str, model=None) -> CheckpointZeroRows:
    out = CheckpointZeroRows()
    try:
        if not model_path or not os.path.isdir(model_path):
            out.error = "not_a_local_dir"
            return out
        index = _checkpoint_tensor_index(model_path)
        if not index:
            out.error = "no_weight_files"
            return out
        tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
        key = _pick_embedding_key(index.keys(), _input_embedding_param_name(model) if model is not None
                                  else None, tied)
        if key is None:
            out.error = "embedding_key_not_found"
            return out
        path = index[key]
        out.key, out.file = key, os.path.basename(path)
        ids: list = []
        if path.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as fh:
                sl = fh.get_slice(key)
                shape = sl.get_shape()
                if len(shape) != 2:
                    out.error = f"not_2d:{shape}"
                    return out
                for start in range(0, shape[0], model_sanity._CHUNK_ROWS):
                    blk = sl[start:start + model_sanity._CHUNK_ROWS]
                    ids.extend(int(i) + start for i in
                               torch.nonzero((blk == 0).all(dim=1)).flatten().tolist())
                out.n_rows = int(shape[0])
        else:
            w = torch.load(path, map_location="cpu", mmap=True, weights_only=True)[key]
            ids = model_sanity.zero_row_ids(w)
            out.n_rows = int(w.shape[0]) if w.dim() == 2 else None
        out.ids, out.ok = ids, True
    except Exception as exc:
        out.error = f"{type(exc).__name__}: {exc}"[:300]
    return out


def live_embedding_materialised(model) -> bool:
    try:
        w = model.get_input_embeddings().weight
    except Exception:
        return False
    return w is not None and w.dim() == 2 and w.numel() > 0 and w.device.type != "meta"


@dataclass
class ZeroRowDetection:
    ids: list = field(default_factory=list)
    source: str = "none"
    ckpt: Optional[CheckpointZeroRows] = None
    live_ids: Optional[list] = None


def detect_zero_rows(model_path: str, model) -> ZeroRowDetection:
    ck = checkpoint_embedding_zero_rows(model_path, model)
    live = (model_sanity.input_embedding_zero_rows(model)
            if live_embedding_materialised(model) else None)
    if ck.ok:
        ids = sorted(set(ck.ids) | set(live or []))
        return ZeroRowDetection(ids=ids, source="checkpoint", ckpt=ck, live_ids=live)
    if live is not None:
        return ZeroRowDetection(ids=list(live), source="live", ckpt=ck, live_ids=live)
    return ZeroRowDetection(ids=[], source="none", ckpt=ck, live_ids=None)


def params_sharded(training_args=None) -> bool:
    try:
        if is_deepspeed_zero3_enabled():
            return True
    except Exception:
        pass
    return bool(training_args is not None and len(getattr(training_args, "fsdp", None) or "") > 0)


_NONFINITE_EMIT_CAP = 20


class NonfiniteGradGuard:

    def __init__(self, model, stage: str, drop_grads: bool = True, engine_steps: bool = False):
        self.names = {id(p): n for n, p in model.named_parameters()}
        self.stage = stage
        self.drop_grads = drop_grads
        self.engine_steps = engine_steps
        self.n_events = 0
        self.n_calls = 0
        self.step_fn = None

    def wrap(self, orig):
        guard = self

        def clip_grad_norm_(parameters, max_norm, *args, **kwargs):
            guard.n_calls += 1
            params = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
            grads = [p.grad for p in params if p.grad is not None]
            norms = None
            if grads and not guard.engine_steps:
                try:
                    norms = torch._foreach_norm(grads)
                except Exception:
                    norms = [g.norm() for g in grads]
            total = orig(params, max_norm, *args, **kwargs)
            if total is None:
                return total
            t = total if isinstance(total, torch.Tensor) else torch.tensor(float(total))
            if bool(torch.isfinite(t)):
                return total
            guard._on_nonfinite(params, norms, float(t))
            return total
        clip_grad_norm_._sn56_guard = guard
        return clip_grad_norm_

    def _on_nonfinite(self, params, norms, total: float) -> None:
        bad: list = []
        if norms is not None:
            with_grad = [p for p in params if p.grad is not None]
            fin = torch.isfinite(torch.stack([n.float().reshape(()) for n in norms]).cpu())
            bad = [self.names.get(id(p), "?") for p, ok in zip(with_grad, fin.tolist()) if not ok]
        if self.drop_grads and not self.engine_steps:
            for p in params:
                p.grad = None
        self.n_events += 1
        if self.n_events <= _NONFINITE_EMIT_CAP or self.n_events % 100 == 0:
            try:
                step = self.step_fn() if self.step_fn is not None else self.n_calls
            except Exception:
                pass
        log_info(f"[sn56][dpo] grad não-finito ({self.stage}, norma={total}): "
                 f"{len(bad)} parâmetro(s) {bad[:4]} — passo pulado")


def install_nonfinite_guard(trainer, stage: str) -> Optional[NonfiniteGradGuard]:
    try:
        acc = trainer.accelerator
        orig = acc.clip_grad_norm_
        if getattr(orig, "_sn56_guard", None) is not None:
            return orig._sn56_guard
        guard = NonfiniteGradGuard(trainer.model, stage,
                                   engine_steps=bool(getattr(trainer, "is_deepspeed_enabled", False)))

        def _step():
            st = getattr(trainer, "state", None)
            return int(getattr(st, "global_step", 0) or 0) + 1

        guard.step_fn = _step
        wrapped = guard.wrap(orig)

        def clip_with_stage(parameters, max_norm, *a, **kw):
            guard.stage = stage if getattr(trainer, "optimizer", None) is not None else f"{stage}_devfit"
            return wrapped(parameters, max_norm, *a, **kw)
        clip_with_stage._sn56_guard = guard
        acc.clip_grad_norm_ = clip_with_stage
        return guard
    except Exception:
        return None


@dataclass
class NullRowProtection:
    detach: Optional[NullPositionDetach] = None
    pin: NullRowPin = field(default_factory=NullRowPin)
    guard: Optional[NonfiniteGradGuard] = None

    def fields(self) -> dict:
        return dict(detach_hooks=self.detach is not None,
                    detach_n_layers=self.detach.n_layers if self.detach is not None else 0,
                    param_pin=bool(self.pin.pinned), pinned=bool(self.pin.pinned),
                    pin_reason=self.pin.reason,
                    nonfinite_guard=(None if self.guard is None else
                                     ("report_only" if self.guard.engine_steps else "skip")))


def protect_null_rows(trainer, ids, *, stage: str, sharded: bool) -> NullRowProtection:
    prot = NullRowProtection(guard=install_nonfinite_guard(trainer, stage))
    if not ids:
        prot.pin = NullRowPin(reason="no_zero_rows")
        return prot
    prot.detach = install_null_position_detach(trainer.model, ids, stage=stage)
    if sharded:
        prot.pin = NullRowPin(ids=list(ids), reason="sharded")
    else:
        prot.pin = pin_null_embedding_rows(trainer.model, ids)
    return prot


def dpo_trajectory_builders(*, model_class, model_kwargs, train_request, tokenizer, peft_config,
                            ref_model, train_ds, dev_ds, callbacks_fn, pin_row_ids=(),
                            sharded=False):

    def build_model():
        m = model_class.from_pretrained(train_request["model_path"], **model_kwargs)
        if "lora_model" in train_request:
            m = PeftModelForCausalLM.from_pretrained(
                m, train_request["lora_model"], is_trainable=True, **model_kwargs)
        if peft_config is None:
            resize_if_needed(train_request["model_name"], m, len(tokenizer))
        try:
            m.config.use_cache = False
        except Exception:
            pass
        set_generation_config(train_request["model_name"], m)
        return m

    def build_trainer(model, args):
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=callbacks_fn(),
        )
        ids = list(pin_row_ids or ())
        if ids and live_embedding_materialised(trainer.model):
            zero = set(model_sanity.input_embedding_zero_rows(trainer.model))
            ids = [i for i in ids if i in zero]
        token_bounds.guard_trainer_collator(trainer, "dpo member collator")
        prot = protect_null_rows(trainer, ids, stage="member", sharded=sharded)
        return trainer

    return build_model, build_trainer


def run_dpo_lr_search(trainer, training_args, train_request, steps_per_epoch,
                      remaining_secs_fn=None, max_epochs_cap=None):
    search_model = trainer.model
    search_loader = trainer.get_train_dataloader()

    def _halve_batch_dataloader():
        cur = training_args.per_device_train_batch_size
        new_bs = max(1, cur // 2)
        training_args.per_device_train_batch_size = new_bs
        trainer._train_batch_size = training_args.train_batch_size
        log_info(f"[sn56][farejando] Cozinha quente, lote {cur} -> {new_bs}")
        return trainer.get_train_dataloader()

    _orig_gather = trainer.accelerator.gather_for_metrics
    _orig_gather_all = trainer.accelerator.gather
    trainer.accelerator.gather_for_metrics = lambda tensors, *a, **kw: tensors
    trainer.accelerator.gather = lambda tensors, *a, **kw: tensors
    trainer.current_gradient_accumulation_steps = training_args.gradient_accumulation_steps
    _orig_clip = torch.nn.utils.clip_grad_norm_
    _search_guard = NonfiniteGradGuard(search_model, "lr_search", drop_grads=False)
    torch.nn.utils.clip_grad_norm_ = _search_guard.wrap(_orig_clip)
    try:
        return run_lr_search(
            model=search_model,
            train_dataloader=search_loader,
            initial_lr=training_args.learning_rate,
            hours_to_complete=train_request["hours_to_complete"],
            grad_accum_steps=training_args.gradient_accumulation_steps,
            max_grad_norm=training_args.max_grad_norm,
            dataloader_factory=_halve_batch_dataloader,
            trainer=trainer,
            steps_per_epoch=steps_per_epoch,
            eval_dataloader=trainer.get_eval_dataloader(),
            neftune_noise_alpha=getattr(training_args, "neftune_noise_alpha", None),
            remaining_secs_fn=remaining_secs_fn,
            max_epochs_cap=max_epochs_cap,
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = _orig_clip
        trainer.accelerator.gather_for_metrics = _orig_gather
        trainer.accelerator.gather = _orig_gather_all
        trainer._stored_metrics = defaultdict(lambda: defaultdict(list))


def cached_lr_for_batch(cached_lr, cached_bs, current_bs):
    try:
        lr = float(cached_lr)
    except (TypeError, ValueError):
        return None, f"cached lr {cached_lr!r} not a number"
    if not math.isfinite(lr) or lr <= 0:
        return None, f"cached lr {lr!r} not finite/positive"

    def _ok(b):
        return isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b) \
            and b >= 1 and float(b) == int(b)

    if cached_bs is None or cached_bs == current_bs:
        return lr, "same batch"
    if not (_ok(cached_bs) and _ok(current_bs)):
        return lr, f"batch {cached_bs!r}->{current_bs!r} invalid, LR unscaled"
    return lr * math.sqrt(current_bs / cached_bs), f"bs {int(cached_bs)}->{int(current_bs)}"


def get_max_length_config():
    config_path = "test_axolotl.yml"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict["sequence_len"]


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (TrainingArguments, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo", help="Run the DPO training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


def main():
    host_cpu.apply("train_dpo")
    
    parser = make_parser()
    training_args, model_args = parser.parse_args_and_config()
    warn_padding_free_noop(training_args)
    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]

    task_id = train_request["task_id"]
    

    output_dir = training_args.output_dir
    from model_utility import load_tokenizer
    tokenizer = load_tokenizer(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    train_path = os.path.join("datasets", f"dpo_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"dpo_dev_{task_id}.json")

    train_ds = get_dataset(train_path, train_request["dataset_type"])
    dev_ds = get_dataset(dev_path, train_request["dataset_type"])

    log_info(f"nós={training_args.world_size}")
    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )

    log_info(f"passos/época={total_steps_per_epoch}")

    original_batch_size = training_args.per_device_train_batch_size

    quantization_config = get_quantization_config(model_args)
    device_string = "cuda:" + str(LOCAL_RANK)
    
    device_map=(
            get_kbit_device_map()
            if quantization_config is not None
            else {"": device_string}
        )
    if len(training_args.fsdp or "") > 0 or is_deepspeed_zero3_enabled():
        device_map = None


    attn_implementation="flash_attention_2" if not training_args.disable_fa else "sdpa"
    if training_args.use_attn_implementation:
        attn_implementation = training_args.use_attn_implementation
        log_info(f"atenção via {attn_implementation}")
        
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map
    )
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    log_info(f"args finais: {training_args}")

    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)
    if len(training_args.fsdp or "") > 0 or is_deepspeed_zero3_enabled():
        log_info("checkpointing ativado com reentrada para deepspeed")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})


    set_generation_config(train_request["model_name"], model)

    ref_model = None
    if "ref_model" in train_request:
        ref_model = model_class.from_pretrained(
            train_request["ref_model"], **model_kwargs
        )

    peft_config = get_peft_config(model_args)
    if "lora_model" in train_request:
        model = PeftModelForCausalLM.from_pretrained(
            model, train_request["lora_model"], is_trainable=True, **model_kwargs
        )

    if peft_config is None:
        resize_if_needed(train_request["model_name"], model, len(tokenizer))
        if ref_model is None and not training_args.precompute_ref_log_probs:
            ref_model = load_reference_model(
                train_request["model_path"], model_kwargs, train_request["model_name"], len(tokenizer)
            )
            log_info(f"ref_model: {train_request['model_path']} dtype={ref_model.dtype}")

    _emb_rows = token_bounds.embedding_rows(model)
    token_bounds.fit_special_tokens(tokenizer, _emb_rows, site="dpo.model")
    model_sanity.check_model(model, tokenizer)
    _zero_det = detect_zero_rows(train_request["model_path"], model)
    _zero_row_ids = _zero_det.ids
    _null_row_ids: list = []
    _sharded_params = params_sharded(training_args)


    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"destino criado: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    log_info(f"salvo periódico a cada {periodic_save_steps} passos")

    training_args.save_only_model = True

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    log_info(f"colunas: {train_ds.column_names}")

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
    log_info(f"modo={_cm!r}, final={is_final_run}")
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
    
    checking_step = train_request["checking_step"]
    if checking_step >= total_steps_per_epoch:
        checking_step = total_steps_per_epoch - 2

    _neft_alpha = 1
    baseline_stats = train_request.get("baseline_stats")
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
    log_info(f"[sn56][caldo] modo={_avg_mode} (treináveis={_trainable_bytes / 1e9:.0f}GB, sharded={_sharded})")
    ckpt_avg = (
        AdaptiveTrainingCallback(
            window=3, averaging_mode=_avg_mode, output_dir=training_args.output_dir
        )
        if is_final_run else None
    )
    if ckpt_avg is not None:
        ckpt_avg._submission_dir = train_request["submission_dir"]
        ckpt_avg.end_time = train_request["end_time"]

    when_to_eval_handler = WhenToEvalHandler(
        train_request["end_time"], train_request["save_before_remaining_time"],
        periodic_save_steps=periodic_save_steps, steps_per_epoch=total_steps_per_epoch,
        max_steps=max_steps)

    def _callbacks():
        return ([])

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            *_callbacks(),
            *([] if ckpt_avg is None else [ckpt_avg]),
            CustomEvalSaveCallback(
                when_to_eval_handler,
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps,
                checking_step=checking_step,
                total_steps_all_epochs=total_steps_all_epochs,
                end_time=train_request["end_time"],
                checking_mode=train_request.get("checking_mode", "none"),
                use_reward_accuracy=False,
            )
        ],
    )
    token_bounds.guard_trainer_collator(trainer, "dpo collator", _emb_rows)

    if ckpt_avg is not None:
        ckpt_avg.trainer = trainer
    member_guard.install_a0_timeout(trainer, float(getattr(training_args, "ddp_timeout", 1800) or 1800))

    if _zero_row_ids:
        _null_row_ids = live_null_rows(_zero_row_ids, trainer.train_dataset, trainer.eval_dataset)
    if _null_row_ids:
        training_args.neftune_noise_alpha = None
        trainer.neftune_noise_alpha = None
        log_info(f"[sn56][neftune] {len(_null_row_ids)} zero embedding row(s) in the inputs "
                 f"{_null_row_ids[:8]}: NEFTune off (was alpha={_neft_alpha}), rows pinned")
    _null_prot = protect_null_rows(trainer, _null_row_ids, stage="a0", sharded=_sharded_params)

    _use_deepspeed = getattr(training_args, "deepspeed", None) is not None
    _state_cache = get_state()
    _cached_lr = _state_cache.get("best_lr_found")
    _cached_bs = _state_cache.get("best_lr_batch_size")
    if _cached_lr is not None:
        _current_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        _cached_lr, _note = cached_lr_for_batch(_cached_lr, _cached_bs, _current_bs)
        log_info(f"[sn56][farejando] LR cached ({_note}): {_cached_lr}")
    _t_per_step = None
    if _cached_lr is not None:
        training_args.learning_rate = _cached_lr
        log_info(f"[sn56][farejando] Usando LR cached de tentativa anterior: {_cached_lr:.2e}")
    elif _use_deepspeed:
        log_info(f"[sn56][farejando] Pulando (DeepSpeed ativo)")
    else:
        model = trainer.model
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{LOCAL_RANK}")
            model = model.to(device)
            log_info(f"[sn56][farejando] Modelo movido pra {device}")

        original_bs = training_args.per_device_train_batch_size
        best_lr, _t_per_step = run_dpo_lr_search(
            trainer, training_args, train_request, total_steps_per_epoch,
            remaining_secs_fn=lambda: remaining_secs_until(train_request["end_time"]),
            max_epochs_cap=dpo_max_epochs(len(train_ds)),
        )

        if torch.distributed.is_initialized():
            _bs_t = torch.tensor(
                [training_args.per_device_train_batch_size],
                device=next(model.parameters()).device,
            )
            torch.distributed.all_reduce(_bs_t, op=torch.distributed.ReduceOp.MIN)
            _min_bs = int(_bs_t.item())
            if not 1 <= _min_bs <= training_args.per_device_train_batch_size:
                raise RuntimeError(
                    f"[sn56][farejando] all_reduce(MIN) of the per-device batch returned "
                    f"{_min_bs} (local {training_args.per_device_train_batch_size}): "
                    f"ranks' collectives are out of step after the LR search"
                )
            if _min_bs < training_args.per_device_train_batch_size:
                log_info(
                    f"[sn56][farejando] OOM em outro rank: lote "
                    f"{training_args.per_device_train_batch_size} -> {_min_bs} (sincronizado)"
                )
                training_args.per_device_train_batch_size = _min_bs

        if training_args.per_device_train_batch_size != original_bs:
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][farejando] Lote reduzido durante busca: {original_bs} -> {training_args.per_device_train_batch_size} (carries to training)")

        if torch.distributed.is_initialized():
            lr_tensor = torch.tensor([best_lr], device=next(model.parameters()).device)
            torch.distributed.broadcast(lr_tensor, src=0)
            best_lr = lr_tensor.item()
            for p in model.parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)
            log_info(f"[sn56][farejando] Sincronizando lr={best_lr:.2e} + weights to all ranks")

        trainer.args.learning_rate = best_lr
        log_info(f"[sn56][farejando] Usando lr={best_lr:.2e}")
        _s = get_state()
        _s["best_lr_found"] = best_lr
        _s["best_lr_batch_size"] = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if is_main_process(LOCAL_RANK):
            set_state(_s)

        _epoch_info = torch.tensor([0.0, 0.0, 0.0], device=next(model.parameters()).device)
        if is_main_process(LOCAL_RANK) and _t_per_step is not None:
            import datetime as _dt_mod
            _now = _dt_mod.datetime.now(_dt_mod.timezone.utc)
            _end_dt = _dt_mod.datetime.strptime(
                train_request["end_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=_dt_mod.timezone.utc)
            _remaining = (_end_dt - _now).total_seconds() * 0.85
            _achievable = int(_remaining / _t_per_step)
            _spe = len(train_ds) // max(1,
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
            if _spe > 0:
                _max_ep = dpo_max_epochs(len(train_ds))
                _opt_epochs = round(max(1.3, min(_max_ep, 1.25 * _achievable / _spe)), 2)
                _new_total = int(_spe * _opt_epochs)
                _warmup = min(training_args.warmup_steps, max(10, int(0.03 * _new_total)))
                _eval_s = float(training_args.eval_steps)
                if is_final_run:
                    _eval_s = float(max(30, _spe // 5))
                _epoch_info[0] = _opt_epochs
                _epoch_info[1] = float(_warmup)
                _epoch_info[2] = _eval_s
                log_info(
                    f"[sn56][relogio] Epochs {training_args.num_train_epochs} -> {_opt_epochs} "
                    f"(t/step={_t_per_step:.3f}s, steps/ep={_spe}, achievable={_achievable})"
                )

        if torch.distributed.is_initialized():
            torch.distributed.broadcast(_epoch_info, src=0)

        if _epoch_info[0] > 0:
            training_args.num_train_epochs = _epoch_info[0].item()
            training_args.warmup_steps = int(_epoch_info[1].item())
            if is_final_run:
                training_args.eval_steps = int(_epoch_info[2].item())
                training_args.save_steps = int(_epoch_info[2].item())

    _dev_pass_secs = None
    _devfit_ok = bool(is_final_run and _avg_mode == "ram" and len(dev_ds) > 0)
    if _devfit_ok and _t_per_step is not None:
        _micro_cost = _t_per_step / max(
            1, training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
        _dev_pass_secs = 2.0 * (len(dev_ds) / max(1, training_args.world_size)) * _micro_cost + 120
        when_to_eval_handler.save_before_remaining_time = (
            train_request["save_before_remaining_time"] + min(15.0, max(2.0, _dev_pass_secs / 60)))
        log_info(f"[sn56][devfit] orçamento ~{_dev_pass_secs:.0f}s; parada final em "
                 f"T-{when_to_eval_handler.save_before_remaining_time:.1f}min")

    def _st_static_facts() -> dict:
        return dict(
            steps_per_epoch=max(1, len(train_ds) // max(1,
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size)),
            num_train_epochs=float(training_args.num_train_epochs),
            eval_steps=int(training_args.eval_steps or 0),
            t_per_step_search=_t_per_step,
            learning_rate=float(trainer.args.learning_rate),
            seed=int(training_args.seed),
            argmin_lr=lr_search.LAST_SELECTION.get("argmin_lr"),
            pool_tier=lr_search.LAST_SELECTION.get("pool_tier"),
            world_size=int(training_args.world_size),
            avg_mode=_avg_mode,
            is_final_run=bool(is_final_run),
            warmup_steps=int(getattr(training_args, "warmup_steps", 0) or 0),
            min_lr_rate=min_lr_rate_of(training_args, 0.0),
            lr_scheduler_type=str(getattr(training_args, "lr_scheduler_type", "")),
            total_secs=max(0.0, float(train_request.get("hours_to_complete", 2)) * 3600.0 - 180.0),
        )

    _dpo_members_world_ok = int(training_args.world_size) == 1

    if ckpt_avg is not None and _dpo_members_world_ok:
        try:
            ckpt_avg.stop_rule_facts = _st_static_facts()
            ckpt_avg.stop_rule_cadences = lambda: (
                int(when_to_eval_handler.steps_per_epoch),
                int(when_to_eval_handler.periodic_save_steps),
                int(training_args.eval_steps or 0))
            ckpt_avg.stop_rule_devpass_reserve = (
                max(second_trajectory._DEVPASS_RESERVE_MIN_SECS, float(_dev_pass_secs or 0.0))
                if _devfit_ok else 0.0)
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

    log_info("Bora treinar")
    log_info(f"fiscalização a cada {training_args.eval_steps} passos, {len(dev_ds)} na prova")
    trainer.add_callback(rank_agreement.RankAgreementCallback("dpo"))
    _train_output = trainer.train()

    _st_out = second_trajectory.Outcome()
    try:
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
        )
        _st_build_model, _st_build_trainer = dpo_trajectory_builders(
            model_class=model_class, model_kwargs=model_kwargs, train_request=train_request,
            tokenizer=tokenizer, peft_config=peft_config,
            ref_model=(getattr(trainer, "ref_model", None) if peft_config is None else None),
            train_ds=train_ds, dev_ds=dev_ds, callbacks_fn=_callbacks,
            pin_row_ids=_null_row_ids, sharded=_sharded_params)

        _min_lr_rate = min_lr_rate_of(training_args)
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
        )
        if _dpo_members_world_ok:
            model = None
            _st_out = second_trajectory.run(
                ckpt_avg=ckpt_avg, trainer=trainer, training_args=training_args,
                train_request=train_request, facts=_st_facts,
                model_builder=_st_build_model, trainer_builder=_st_build_trainer,
                rng_state=_st_rng_state, devfit=_devfit_plan, log=log_info,
            )
        else:
            _gate_dpo = second_trajectory._gate(_st_facts, training_args)
            if _gate_dpo in (None, "no_best_step"):
                _gate_dpo = f"world_size={int(training_args.world_size)}"
            _st_out.reason = _gate_dpo
            log_info(f"[sn56][traj2] inativo ({_gate_dpo})")
    except Exception as _e:
        log_info(f"[sn56][traj2] wiring falhou ({_e})")
    _st_mutated = bool(getattr(_st_out, "mutated", False))

    try:
        _dev_n = len(dev_ds)
        if _st_mutated:
            log_info(f"[sn56][devfit] decidido pós-soup: {getattr(_st_out, 'devfit', None)}")
        elif is_final_run and _avg_mode == "ram" and _dev_n > 0:
            from dev_pass import run_dev_pass
            _min_lr = trainer.args.learning_rate * min_lr_rate_of(training_args)
            log_info(f"[sn56][devfit] {_dev_n} amostras de dev, 1 época @ lr={_min_lr:.2e}")
            _dp_res = run_dev_pass(
                trainer,
                submission_dir=train_request["submission_dir"],
                min_lr=_min_lr,
                max_grad_norm=training_args.max_grad_norm,
                train_per_device=training_args.per_device_train_batch_size,
                train_grad_accum=training_args.gradient_accumulation_steps,
                local_rank=LOCAL_RANK, log=log_info,
            ) or {}
        else:
            log_info(f"[sn56][devfit] pulando (final={is_final_run}, modo={_avg_mode}, dev={_dev_n})")
    except Exception as _e:
        log_info(f"[sn56][devfit] dev-pass falhou ({_e}), mantendo melhor checkpoint")

    if is_main_process(LOCAL_RANK):
        with open(os.path.join(training_args.output_dir, "success.txt"), "w") as f:
            f.write("Success")


if __name__ == "__main__":
    main()
