

import contextlib
import ctypes
import gc
import math
import os
import shutil
import time
from inspect import signature as _signature
from typing import Optional

import bitsandbytes as bnb
import bitsandbytes.functional as bnbf
import torch
from pydantic import BaseModel
from torch.multiprocessing.reductions import StorageWeakRef
from transformers import Trainer
from transformers.trainer_utils import is_main_process

import host_snapshot
import member_guard
import precise_update
import rank_agreement
import save_guard as sg
import spare_budget as sb

_MEM_MARGIN_FRAC = 0.01
_MEM_MARGIN_MIN = 512 << 20
_BNB_MIN_8BIT_SIZE = 4096
_BNB_BLOCK = 256
CHOICES = ("precise", "adamw", "train_optim", "skip")
WITHIN_A0_PEAK = os.environ.get("SN56_DEVFIT_WITHIN_A0_PEAK", "1").strip().lower() not in (
    "0", "false", "no", "off")


class OptimizerFootprint(BaseModel):
    name: str
    state_bytes: Optional[int] = None
    step_temp_bytes: Optional[int] = None


class DevfitOptimizerDecision(BaseModel):
    choice: str
    reason: str
    train_optim: str
    trainable_bytes: int
    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    act_bytes: Optional[int] = None
    margin_bytes: Optional[int] = None
    need_adamw: Optional[int] = None
    need_train_optim: Optional[int] = None
    need_precise: Optional[int] = None
    adamw: OptimizerFootprint
    train: OptimizerFootprint
    precise: Optional[OptimizerFootprint] = None
    alloc_bytes: Optional[int] = None
    a0_peak_bytes: Optional[int] = None
    proj_peak_adamw: Optional[int] = None
    proj_peak_train_optim: Optional[int] = None


def _nbytes(p) -> int:
    return int(p.numel()) * int(p.element_size())


def _bnb_32bit_override_ids() -> set:
    try:
        mng = bnb.optim.GlobalOptimManager.get_instance()
    except Exception:
        return set()
    ids = {pid for pid, cfg in (getattr(mng, "pid2config", {}) or {}).items()
           if int((cfg or {}).get("optim_bits", 8)) == 32}
    for module, attr, cfg in getattr(mng, "module_weight_config_triple", []) or []:
        if int((cfg or {}).get("optim_bits", 8)) == 32:
            t = getattr(module, attr, None)
            if t is not None:
                ids.add(id(t))
    return ids


def optimizer_footprint(name: str, params) -> OptimizerFootprint:
    params = list(params)
    pb = sum(_nbytes(p) for p in params)
    n = name.lower()
    if n == "adamw_devfit" or (n.startswith("adamw_torch") and "fused" not in n) or n == "adamw_hf":
        return OptimizerFootprint(name=name, state_bytes=2 * pb, step_temp_bytes=pb)
    if n.startswith("adamw_torch") and "fused" in n:
        return OptimizerFootprint(name=name, state_bytes=2 * pb, step_temp_bytes=0)
    if "8bit" in n:
        over = _bnb_32bit_override_ids()
        total = 0
        for p in params:
            k = int(p.numel())
            if k < _BNB_MIN_8BIT_SIZE or id(p) in over:
                total += 8 * k
            else:
                total += 2 * k + 2 * 4 * math.ceil(k / _BNB_BLOCK)
        return OptimizerFootprint(name=name, state_bytes=total, step_temp_bytes=0)
    if "32bit" in n and ("bnb" in n or "paged" in n):
        return OptimizerFootprint(name=name, state_bytes=8 * sum(int(p.numel()) for p in params),
                                  step_temp_bytes=0)
    if n == "adafactor":
        total = 0
        for p in params:
            if p.dim() >= 2:
                total += 4 * (int(p.shape[-1]) + int(p.numel() // p.shape[-1]))
            else:
                total += 4 * int(p.numel())
        return OptimizerFootprint(name=name, state_bytes=total, step_temp_bytes=4 * max(
            (int(p.numel()) for p in params), default=0))
    return OptimizerFootprint(name=name)


def _within_a0_peak(d: DevfitOptimizerDecision, act: int, alloc_bytes: Optional[int],
                    a0_peak_bytes: Optional[int], tolerance: int) -> DevfitOptimizerDecision:
    if not WITHIN_A0_PEAK or alloc_bytes is None or a0_peak_bytes is None:
        return d
    d.alloc_bytes, d.a0_peak_bytes = int(alloc_bytes), int(a0_peak_bytes)
    cap = int(a0_peak_bytes) + int(tolerance)
    for choice, fp in (("adamw", d.adamw), ("train_optim", d.train)):
        if fp.state_bytes is None:
            continue
        proj = int(alloc_bytes) + int(fp.state_bytes) + max(int(fp.step_temp_bytes or 0), act)
        setattr(d, f"proj_peak_{choice}", proj)
        if proj <= cap:
            d.choice, d.reason = choice, f"{d.reason},{choice}_within_a0_peak"
            return d
    return d


def precise_devpass_wanted(trainer) -> bool:
    if not isinstance(trainer, precise_update.PreciseUpdateMixin):
        return False
    try:
        return precise_update.PhasePolicy.from_env().devpass != precise_update.STOCK
    except Exception:
        return False


def choose_dev_optimizer(*, train_optim: str, params, free_bytes: Optional[int],
                         total_bytes: Optional[int], act_bytes: Optional[int],
                         alloc_bytes: Optional[int] = None, a0_peak_bytes: Optional[int] = None,
                         peak_tolerance: int = 0,
                         precise: Optional[OptimizerFootprint] = None) -> DevfitOptimizerDecision:
    params = list(params)
    pb = sum(_nbytes(p) for p in params)
    fa = optimizer_footprint("adamw_devfit", params)
    ft = optimizer_footprint(train_optim, params)
    d = DevfitOptimizerDecision(choice="adamw", reason="no_device_memory_info",
                                train_optim=train_optim, trainable_bytes=pb, adamw=fa, train=ft,
                                free_bytes=free_bytes, total_bytes=total_bytes, act_bytes=act_bytes,
                                precise=precise)
    if free_bytes is None or total_bytes is None:
        if precise is not None:
            d.choice = "precise"
        return d
    act = int(act_bytes or 0)
    margin = max(_MEM_MARGIN_MIN, int(_MEM_MARGIN_FRAC * total_bytes))
    d.margin_bytes = margin
    if precise is not None and precise.state_bytes is not None:
        d.need_precise = int(precise.state_bytes + max(precise.step_temp_bytes or 0, act) + margin)
        if d.need_precise <= free_bytes:
            d.choice, d.reason = "precise", "precise_fits"
            return d
    d.need_adamw = int(fa.state_bytes + max(fa.step_temp_bytes, act) + margin)
    if d.need_adamw <= free_bytes:
        d.choice, d.reason = "adamw", "adamw_fits"
        return d
    if ft.state_bytes is None:
        d.choice, d.reason = "skip", "adamw_over_memory,train_optim_unmodelled"
        return _within_a0_peak(d, act, alloc_bytes, a0_peak_bytes, peak_tolerance)
    d.need_train_optim = int(ft.state_bytes + max(ft.step_temp_bytes or 0, act) + margin)
    if d.need_train_optim <= free_bytes:
        d.choice, d.reason = "train_optim", "adamw_over_memory"
        return d
    d.choice, d.reason = "skip", "optimizer_state_over_memory"
    return _within_a0_peak(d, act, alloc_bytes, a0_peak_bytes, peak_tolerance)


def _device_memory(device) -> tuple:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None, None
    free, total = torch.cuda.mem_get_info(device)
    alloc = torch.cuda.memory_allocated(device)
    idle = max(0, torch.cuda.memory_reserved(device) - alloc)
    act = max(0, torch.cuda.max_memory_allocated(device) - alloc)
    return int(free + idle), int(total), int(act)


def _device_alloc(device) -> Optional[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.memory_allocated(device))


PAGED_EVICT_RAM_MARGIN = 4 * 1024 ** 3


def evict_paged_to_host(tensors: list, avail_fn=None) -> dict:
    paged = [t for t in tensors if torch.is_tensor(t) and getattr(t, "is_paged", False)]
    nbytes = int(sum(_nbytes(t) for t in paged))
    info = {"paged_bytes": nbytes, "paged_evicted_bytes": 0, "paged_evict": "none"}
    if not paged or not torch.cuda.is_available():
        return info
    try:
        avail = (avail_fn or host_snapshot.available_ram_bytes)()
    except Exception:
        avail = None
    if avail is not None and int(avail) - PAGED_EVICT_RAM_MARGIN < nbytes:
        info["paged_evict"] = "host_ram"
        info["host_avail"] = int(avail)
        return info
    info["paged_evict"] = "skipped_unsafe"
    return info


FREE_PAGED_ENV = "SN56_FREE_PAGED_STATE"


def _env_tristate(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off")


FREE_PAGED_STATE: Optional[bool] = _env_tristate(FREE_PAGED_ENV)
FREE_PAGED_MULTI_ENV = "SN56_FREE_PAGED_MULTI_GPU"
FREE_PAGED_MULTI: bool = _env_tristate(FREE_PAGED_MULTI_ENV) is not False


class PagedFreeSetting(BaseModel):
    on: bool
    world: int
    guard: bool
    reason: str


def paged_free_setting(world: int, guard: bool,
                       explicit: Optional[bool] = None,
                       multi: Optional[bool] = None) -> PagedFreeSetting:
    world = max(1, int(world or 1))
    guard = bool(guard) and world > 1
    if explicit is not None:
        return PagedFreeSetting(on=bool(explicit), world=world, guard=guard,
                                reason=f"env:{FREE_PAGED_ENV}={int(bool(explicit))}")
    if world <= 1:
        return PagedFreeSetting(on=True, world=world, guard=False, reason="single_gpu")
    if guard:
        if FREE_PAGED_MULTI if multi is None else bool(multi):
            return PagedFreeSetting(on=True, world=world, guard=True, reason="multi_gpu_free")
        return PagedFreeSetting(on=False, world=world, guard=True, reason="multi_gpu_guard")
    return PagedFreeSetting(on=True, world=world, guard=False, reason="multi_gpu_unguarded")


def current_paged_free() -> PagedFreeSetting:
    world = (int(torch.distributed.get_world_size())
             if torch.distributed.is_available() and torch.distributed.is_initialized() else 1)
    return paged_free_setting(world, sb.multi_gpu_guard(world).on, FREE_PAGED_STATE)


class PagedBuffer(BaseModel):
    ptr: int
    nbytes: int


def _paged_buffers(tensors: list) -> list:
    out, seen = [], set()
    for t in tensors:
        if not (torch.is_tensor(t) and getattr(t, "is_paged", False)):
            continue
        st = t.untyped_storage()
        ptr = int(st.data_ptr())
        if ptr == 0 or ptr in seen:
            continue
        seen.add(ptr)
        out.append((PagedBuffer(ptr=ptr, nbytes=int(st.nbytes())), StorageWeakRef(st)))
    return out


_LIBCUDA = None


def _cu_mem_free(ptr: int) -> int:
    global _LIBCUDA
    if _LIBCUDA is None:
        _LIBCUDA = ctypes.CDLL("libcuda.so.1")
        _LIBCUDA.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        _LIBCUDA.cuMemFree_v2.restype = ctypes.c_int
    return int(_LIBCUDA.cuMemFree_v2(ctypes.c_uint64(ptr)))


def free_unreferenced_paged(buffers: list, free_fn=None) -> dict:
    info = {"paged_freed": 0, "paged_freed_bytes": 0, "paged_live": 0, "paged_free_failed": 0}
    if not buffers:
        return info
    dead = [b for b, ref in buffers if ref.expired()]
    info["paged_live"] = len(buffers) - len(dead)
    if not dead:
        return info
    if free_fn is None:
        if not torch.cuda.is_available():
            return info
        torch.cuda.synchronize()
        free_fn = _cu_mem_free
    for b in dead:
        try:
            rc = free_fn(b.ptr)
        except Exception as exc:
            info["paged_free_failed"] += 1
            info["paged_free_error"] = f"{type(exc).__name__}: {exc}"[:160]
            break
        if rc != 0:
            info["paged_free_failed"] += 1
            info["paged_free_error"] = f"CUresult={rc}"
            continue
        info["paged_freed"] += 1
        info["paged_freed_bytes"] += b.nbytes
    return info


def release_optimizer_state(optimizer, free_fn=None) -> dict:
    info = {"state_bytes": 0, "paged_released": 0}
    inner = getattr(optimizer, "optimizer", optimizer)
    state = getattr(inner, "state", None)
    if not state:
        return info
    tensors = [v for st in list(state.values()) for v in (st or {}).values() if torch.is_tensor(v)]
    ids = {id(v) for v in tensors}
    info["state_bytes"] = int(sum(_nbytes(v) for v in tensors))
    try:
        info.update(evict_paged_to_host(tensors))
    except Exception as exc:
        info["paged_evict"] = f"failed:{type(exc).__name__}"[:120]
    buffers = _paged_buffers(tensors) if current_paged_free().on else []
    del tensors
    try:
        mng = bnbf.GlobalPageManager.get_instance()
        before = len(mng.paged_tensors)
        mng.paged_tensors = [t for t in mng.paged_tensors if id(t) not in ids]
        info["paged_released"] = before - len(mng.paged_tensors)
    except Exception:
        pass
    try:
        state.clear()
    except Exception:
        pass
    if buffers:
        gc.collect()
        try:
            info.update(free_unreferenced_paged(buffers, free_fn=free_fn))
        except Exception as exc:
            info["paged_free_error"] = f"{type(exc).__name__}: {exc}"[:160]
    return info


def _train_optim_name(args) -> str:
    o = getattr(args, "optim", "") or ""
    return str(getattr(o, "value", o))


def _build_train_optim(trainer, unwrapped, min_lr: float):
    cls, kw = Trainer.get_optimizer_cls_and_kwargs(trainer.args, unwrapped)
    kw = dict(kw)
    if any(k in kw for k in ("params", "model", "optimizer_dict")):
        raise ValueError("training optimizer builds its own parameter set")
    kw["lr"] = min_lr
    kw.pop("weight_decay", None)
    decay = set(trainer.get_decay_parameter_names(unwrapped))
    named = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in named if n in decay], "weight_decay": 0.0},
              {"params": [p for n, p in named if n not in decay], "weight_decay": 0.0}]
    return cls(groups, **kw)


def _sharded(trainer) -> str:
    if getattr(trainer, "is_deepspeed_enabled", False):
        return "deepspeed"
    if getattr(trainer, "is_fsdp_enabled", False):
        return "fsdp"
    return ""


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _agree_max(value, device):
    
    if not torch.distributed.is_initialized():
        return value
    try:
        t = torch.tensor([float(value)], device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return float(t.item())
    except Exception:
        return value


DEV_PASS_MICRO_BATCH = 1


def dev_loader(trainer, micro_batch: int):
    stock = getattr(trainer, "stock_eval_dataloader", None)
    if stock is not None:
        return stock(micro_batch)
    args = trainer.args
    saved = args.per_device_eval_batch_size
    args.per_device_eval_batch_size = int(micro_batch)
    try:
        return trainer.get_eval_dataloader()
    finally:
        args.per_device_eval_batch_size = saved


def run_dev_pass(
    trainer,
    *,
    submission_dir,
    min_lr,
    max_grad_norm,
    train_per_device,
    train_grad_accum,
    local_rank,
    log,
    deadline=None,
    save_secs_est=0.0,
    probe_micro=4,
    overshoot_ok=0.0,
    memory_proof=None,
    collective_timeout=None,
    restore_timeout=1800.0,
    micro_batch=None,
):
    
    t_enter = time.time()
    result = {
        "ran": False, "started": False, "saved": False, "reason": "",
        "steps_in_epoch": 0, "micro_steps": 0, "opt_steps": 0,
        "accum": 0, "per_device_eval": 0, "effective_batch": 0,
        "projected_secs": None, "budget_secs_left": None,
        "save_secs_est": round(float(save_secs_est or 0.0), 2),
        "secs": 0.0,
    }
    if deadline is not None:
        result["budget_secs_left"] = round(deadline - t_enter, 1)

    def _done(reason):
        result["reason"] = reason
        result["secs"] = round(time.time() - t_enter, 2)
        try:
            tr = [float(x) for x in (result.pop("loss_trace", None) or [])]
        except Exception:
            tr = []
        tr = [x for x in tr if x == x]
        if tr:
            q = max(1, len(tr) // 4)
            result["loss_first_q"] = round(sum(tr[:q]) / q, 6)
            result["loss_last_q"] = round(sum(tr[-q:]) / q, 6)
            result["loss_mean"] = round(sum(tr) / len(tr), 6)
            result["loss_n"] = len(tr)
        return result

    accelerator = trainer.accelerator
    ddp_model = getattr(trainer, "model_wrapped", None) or trainer.model
    unwrapped = _unwrap(ddp_model)
    device = next(unwrapped.parameters()).device

    _sh = _sharded(trainer)
    if _sh:
        log(f"[sn56][devfit] parâmetros fragmentados ({_sh}), pulando")
        return _done(f"sharded:{_sh}")

    trainable = [p for p in unwrapped.parameters() if p.requires_grad]
    if not trainable:
        log("[sn56][devfit] sem parâmetros treináveis, pulando")
        return _done("no_trainable_params")

    try:
        if getattr(trainer, "optimizer", None) is not None:
            _released = release_optimizer_state(trainer.optimizer)
    except Exception:
        pass
    try:
        trainer.optimizer = None
    except Exception:
        pass
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dev_opt = None
    result["optimizer"] = None
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    per_device_eval = max(1, int(micro_batch if micro_batch is not None else
                                 (getattr(trainer.args, "per_device_eval_batch_size", 1) or 1)))
    loader = dev_loader(trainer, per_device_eval)
    accum = max(1, round(train_per_device * train_grad_accum / per_device_eval))

    steps_in_epoch = len(loader)
    if torch.distributed.is_initialized():
        _t = torch.tensor([steps_in_epoch], device=device)
        torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MIN)
        steps_in_epoch = int(_t.item())
    result["steps_in_epoch"] = steps_in_epoch
    result["accum"] = accum
    result["per_device_eval"] = per_device_eval
    if steps_in_epoch == 0:
        log("[sn56][devfit] dev loader vazio, pulando")
        del dev_opt
        return _done("empty_dev_loader")

    if rank_agreement.agree("devfit", {"per_device_eval": per_device_eval, "accum": accum,
                                       "steps_in_epoch": steps_in_epoch,
                                       "min_lr": float(min_lr)}):
        log("[sn56][devfit] ranks discordam do plano do dev-pass, pulando")
        return _done("rank_disagreement")

    _a0_peak = None
    if memory_proof is not None:
        _proof = (memory_proof if isinstance(memory_proof, member_guard.MemoryProof)
                  else member_guard.MemoryProof(**memory_proof))
        _a0_peak = _proof.a0_peak_bytes
        _world = (int(torch.distributed.get_world_size())
                  if torch.distributed.is_initialized() else 1)
        if sb.multi_gpu_guard(_world).on:
            _a0_peak = None
        _shape = member_guard.devfit_shape(eval_micro=per_device_eval, proof=_proof)
        if not _shape.ok:
            log(f"[sn56][devfit] forma de memória não provada pelo a0 ({_shape.reason}), pulando")
            return _done(_shape.reason)

    try:
        trainer.current_gradient_accumulation_steps = accum
    except Exception:
        pass
    _ts_kwargs = ({"num_items_in_batch": None}
                  if "num_items_in_batch" in _signature(trainer.training_step).parameters
                  else {})

    probe_at = 0
    if deadline is not None:
        probe_at = max(1, min(int(probe_micro or 1), steps_in_epoch))

    def _abandon(reason, projected):
        
        result["projected_secs"] = round(projected, 1)
        unwrapped.zero_grad(set_to_none=True)
        log(f"[sn56][devfit] abortado ({reason}): projetado {projected:.0f}s, "
            f"restam {deadline - time.time():.0f}s — submissão intacta")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    sg.ship_marker_update(activity="devfit")
    with member_guard.short_collective_timeout(collective_timeout, restore_timeout, "devfit"):
        ddp_model.train()
        result["started"] = True
        t_loop = time.time()
        _dl: list = []
        result["loss_trace"] = _dl
        n_micro = 0
        n_opt = 0
        over = None
        for step, inputs in enumerate(loader):
            if step >= steps_in_epoch:
                break
            do_sync = ((step + 1) % accum == 0) or ((step + 1) == steps_in_epoch)
            accelerator.gradient_state._set_sync_gradients(do_sync)
            sync_ctx = (contextlib.nullcontext() if do_sync
                        else accelerator.no_sync(model=ddp_model))
            with sync_ctx:
                _l = trainer.training_step(ddp_model, inputs, **_ts_kwargs)
            n_micro += 1
            try:
                if _l is not None:
                    _dl.append(_l.detach().float().reshape(-1)[0])
            except Exception:
                pass

            if deadline is not None and (n_micro == probe_at or do_sync):
                rate = (time.time() - t_loop) / n_micro
                finish_at = _agree_max(
                    t_loop + steps_in_epoch * rate + float(save_secs_est or 0.0), device)
                result["projected_secs"] = round(finish_at - t_loop, 1)
                if finish_at > deadline + float(overshoot_ok or 0.0):
                    over = "probe_over_budget" if n_micro == probe_at else "overrun"
                    _abandon(over, finish_at - t_loop)
                    break

            if do_sync and dev_opt is None:
                _free_b, _total_b, _act_b = _device_memory(device)
                _pl = None
                _pfp = None
                if precise_devpass_wanted(trainer):
                    _pl = precise_update.plan(trainer.args, unwrapped, phase="devpass",
                                              free_bytes=_free_b, total_bytes=_total_b)
                if _pl is not None and _pl.on:
                    _fp = precise_update.footprint(_pl.mode, _pl.n_trainable, _pl.chunk,
                                                   _pl.n_fp32_moments, _pl.writeback)
                    _pfp = OptimizerFootprint(name=f"precise_{_fp.name}",
                                              state_bytes=_fp.state_bytes,
                                              step_temp_bytes=_fp.step_temp_bytes)
                _dec = choose_dev_optimizer(train_optim=_train_optim_name(trainer.args),
                                            params=trainable, free_bytes=_free_b,
                                            total_bytes=_total_b, act_bytes=_act_b,
                                            alloc_bytes=_device_alloc(device),
                                            a0_peak_bytes=_a0_peak,
                                            peak_tolerance=member_guard.PROVEN_PEAK_TOLERANCE,
                                            precise=_pfp)
                _idx = int(_agree_max(float(CHOICES.index(_dec.choice)), device))
                if CHOICES[_idx] != _dec.choice:
                    _dec.choice, _dec.reason = CHOICES[_idx], f"peer:{CHOICES[_idx]}"
                if _dec.choice == "train_optim":
                    try:
                        dev_opt = _build_train_optim(trainer, unwrapped, min_lr)
                        _ok = 1.0
                    except Exception as _exc:
                        _dec.reason = f"train_optim_build:{type(_exc).__name__}"
                        _ok = 0.0
                    if _agree_max(1.0 - _ok, device) > 0.5:
                        dev_opt = None
                        _dec.choice = "skip"
                elif _dec.choice == "precise":
                    dev_opt = precise_update.build(trainer.args, unwrapped, phase="devpass",
                                                   lr=min_lr, weight_decay=0.0, mode=_pl.mode)
                elif _dec.choice == "adamw":
                    dev_opt = torch.optim.AdamW(trainable, lr=min_lr, weight_decay=0.0)
                result["optimizer"] = _dec.choice
                result["optimizer_reason"] = _dec.reason
                if dev_opt is None:
                    over = "optimizer_memory"
                    unwrapped.zero_grad(set_to_none=True)
                    log(f"[sn56][devfit] otimizador não cabe ({_dec.reason}): "
                        f"precisa adamw={_dec.need_adamw} / treino={_dec.need_train_optim} bytes, "
                        f"livres {_dec.free_bytes} — submissão intacta")
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    break

            if do_sync:
                if max_grad_norm and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
                dev_opt.step()
                ddp_model.zero_grad(set_to_none=True)
                n_opt += 1

    result["micro_steps"] = n_micro
    result["opt_steps"] = n_opt
    eff = per_device_eval * accum * max(1, getattr(accelerator, "num_processes", 1))
    result["effective_batch"] = eff

    del dev_opt
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if over is not None:
        return _done(over)

    log(f"[sn56][devfit] {steps_in_epoch} micro / {n_opt} opt-steps sobre dev "
        f"@ lr={min_lr:.2e} (lote efetivo ~{eff}, igual ao treino)")

    saved = True
    if is_main_process(local_rank):
        saved = _save_weights_only(unwrapped, submission_dir, log)
        if saved:
            sg.ship_marker_update(shipped_devfit=True, shipped_loss=None,
                                  activity="devfit_written")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    result["saved"] = bool(saved)
    result["ran"] = bool(saved)
    return _done("ok" if saved else "save_failed")


def _is_weight_file(name):
    return (name.endswith(".safetensors")
            or name == "model.safetensors.index.json"
            or name.endswith(".bin"))


def _save_weights_only(unwrapped, submission_dir, log):
    
    if not submission_dir or not os.path.isdir(submission_dir):
        log(f"[sn56][devfit] submission_dir ausente ({submission_dir}), não salvo")
        return False
    base = submission_dir.rstrip("/")
    staging, backup = base + ".devfit_new", base + ".devfit_old"
    kept = base + ".pre_devfit"
    for d in (staging, backup, kept):
        if os.path.exists(d):
            shutil.rmtree(d)
    try:
        os.makedirs(staging, exist_ok=True)
        unwrapped.save_pretrained(staging, safe_serialization=True)
        for f in os.listdir(submission_dir):
            src = os.path.join(submission_dir, f)
            if os.path.isfile(src) and not _is_weight_file(f):
                shutil.copy2(src, os.path.join(staging, f))
        host_snapshot.drop_written_cache(staging, site="submission")
        os.rename(submission_dir, backup)
        os.rename(staging, submission_dir)
        try:
            os.rename(backup, kept)
        except Exception as _exc:
            shutil.rmtree(backup, ignore_errors=True)
            log(f"[sn56][devfit] não consegui guardar o pré-devfit ({_exc})")
        log("[sn56][devfit] submissão trocada pelo modelo do dev-pass "
            "(config/tokenizer/loss.txt preservados; pré-devfit guardado)")
        return True
    except Exception as e:
        log(f"[sn56][devfit] falha ao salvar ({e}), restaurando melhor")
        if not os.path.isdir(submission_dir) and os.path.isdir(backup):
            try:
                os.rename(backup, submission_dir)
            except Exception:
                pass
        for d in (staging, backup):
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
        return False
