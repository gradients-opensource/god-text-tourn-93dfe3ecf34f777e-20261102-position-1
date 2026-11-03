import run_state
import save_guard as sg
import async_save
from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import inspect
import types
import math
import os
import transformers.optimization as _hf_optimization
from typing import Callable, Optional, Dict
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
from state_manager import get_state, set_state
from pydantic import BaseModel, ValidationError
MAX_TRIES = 9


def _cosine_lr_lambda_clamped(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    progress = min(1.0, progress)
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


_HF_COSINE_LAMBDA = "_get_cosine_schedule_with_warmup_lr_lambda"
_HF_COSINE_LAMBDA_PARAMS = [
    "current_step", "num_warmup_steps", "num_training_steps", "num_cycles", "min_lr_rate",
]


def _install_cosine_progress_clamp():
    
    stock = getattr(_hf_optimization, _HF_COSINE_LAMBDA, None)
    if stock is None:
        raise RuntimeError(
            f"cosine-progress-clamp: transformers.optimization has no {_HF_COSINE_LAMBDA}; "
            "the cosine lambda moved or was renamed — re-derive the clamp against this version"
        )
    params = list(inspect.signature(stock).parameters)
    if params != _HF_COSINE_LAMBDA_PARAMS:
        raise RuntimeError(
            f"cosine-progress-clamp: {_HF_COSINE_LAMBDA} signature changed "
            f"({params} != {_HF_COSINE_LAMBDA_PARAMS}) — re-derive the clamp against this version"
        )
    setattr(_hf_optimization, _HF_COSINE_LAMBDA, _cosine_lr_lambda_clamped)
    for sched in ("cosine", "cosine_with_min_lr"):
        builder = next(
            (fn for k, fn in _hf_optimization.TYPE_TO_SCHEDULER_FUNCTION.items()
             if getattr(k, "value", k) == sched),
            None,
        )
        code = getattr(builder, "__code__", None)
        if code is None or _HF_COSINE_LAMBDA not in code.co_names:
            raise RuntimeError(
                f"cosine-progress-clamp: get_scheduler({sched!r}) no longer resolves "
                f"{_HF_COSINE_LAMBDA} at call time — the rebind would be a silent no-op"
            )
        if builder.__globals__.get(_HF_COSINE_LAMBDA) is not _cosine_lr_lambda_clamped:
            raise RuntimeError(
                f"cosine-progress-clamp: get_scheduler({sched!r}) is built from a different "
                "module namespace — the rebind would be a silent no-op"
            )
    return stock


_install_cosine_progress_clamp()


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"posto={LOCAL_RANK}", flush=True)
    
def eval_gate_step(total_steps_all_epochs, num_train_epochs) -> int:
    
    if not total_steps_all_epochs or total_steps_all_epochs <= 0:
        return 0
    if num_train_epochs <= 1:
        return 0
    return int(0.75 * (total_steps_all_epochs / max(1, num_train_epochs)))


def restate_eval_gate_horizon(callback, new_total_steps, num_train_epochs) -> dict:
    
    before = callback.total_steps_all_epochs
    fresh = new_total_steps
    clamped = bool(before and before > 0 and fresh > before)
    after = before if clamped else fresh
    callback.total_steps_all_epochs = after
    return {
        "total_steps_all_epochs_before": before,
        "total_steps_all_epochs_after": after,
        "total_steps_all_epochs_replanned": fresh,
        "num_train_epochs": float(num_train_epochs),
        "gate_step_before": eval_gate_step(before, num_train_epochs),
        "gate_step_after": eval_gate_step(after, num_train_epochs),
        "clamped": clamped,
    }


EVAL_COUNT_FLOOR = 8

CADENCE_HORIZON_FRACTION = 0.5


def eval_steps_in_window(*, cadences, gate_step, horizon) -> list:
    
    fired = set()
    for cadence in cadences:
        if not cadence or cadence <= 0:
            continue
        step = cadence
        if step <= 1:
            step = cadence * 2
        while step <= horizon:
            if step >= gate_step:
                fired.add(step)
            step += cadence
    return sorted(fired)


def restate_eval_cadence(handler, realised_steps_per_epoch, *, gate_step, horizon,
                         hf_eval_steps, total_steps=None,
                         floor: int = EVAL_COUNT_FLOOR) -> dict:
    
    before = int(handler.steps_per_epoch)
    replanned = int(realised_steps_per_epoch)
    periodic = int(handler.periodic_save_steps)
    hf_eval_steps = int(hf_eval_steps or 0)
    horizon = int(horizon)
    total_steps = int(horizon if total_steps is None else total_steps)

    def _schedule(spe, upto=None):
        return eval_steps_in_window(
            cadences=(spe, periodic, hf_eval_steps),
            gate_step=gate_step, horizon=int(horizon if upto is None else upto),
        )

    sched_before = _schedule(before)
    sched_replanned = _schedule(replanned)
    n_before, n_replanned = len(sched_before) + 1, len(sched_replanned) + 1

    sched_before_full = _schedule(before, upto=total_steps)
    sched_replanned_full = _schedule(replanned, upto=total_steps)
    n_before_full = len(sched_before_full) + 1
    n_replanned_full = len(sched_replanned_full) + 1

    disabled = before <= 0
    clamped = bool(not disabled and 0 < replanned < before)
    window_empty = bool(not disabled and not clamped and replanned != before
                        and not sched_before and not sched_replanned)
    floored = bool(not disabled and not clamped and not window_empty
                   and replanned != before
                   and n_replanned < floor and n_replanned < n_before)
    would_add = bool(not disabled and not clamped and not window_empty and not floored
                     and (n_replanned > n_before or n_replanned_full > n_before_full))

    refused = ("disabled" if disabled else
               "clamped" if clamped else
               "window_empty" if window_empty else
               "floored" if floored else
               "would_add" if would_add else
               "" if replanned != before else "identity")
    adopted = (not (disabled or clamped or window_empty or floored or would_add)
               and replanned != before)
    after = replanned if adopted else before
    handler.steps_per_epoch = after

    schedule = sched_replanned if adopted else sched_before
    return {
        "steps_per_epoch_before": before,
        "steps_per_epoch_after": after,
        "steps_per_epoch_replanned": replanned,
        "adopted": adopted,
        "refused": refused,
        "disabled": disabled,
        "clamped": clamped,
        "window_empty": window_empty,
        "floored": floored,
        "would_add": would_add,
        "floor": int(floor),
        "n_evals_before": n_before,
        "n_evals_replanned": n_replanned,
        "total_steps": total_steps,
        "n_evals_before_full_span": n_before_full,
        "n_evals_replanned_full_span": n_replanned_full,
        "n_evals_after": len(schedule) + 1,
        "horizon": horizon,
        "horizon_fraction": CADENCE_HORIZON_FRACTION,
        "gate_step": int(gate_step),
        "hf_eval_steps": hf_eval_steps,
        "periodic_save_steps": periodic,
        "scheduled_eval_steps": schedule[:48],
        "scheduled_eval_steps_truncated": len(schedule) > 48,
    }


SUBMISSION_RECORD_FILE = sg.SUBMISSION_RECORD_FILE


class SubmissionRecord(BaseModel):
    loss: Optional[float]
    step: int
    attempt: int


def submission_record_path(output_dir: str) -> str:
    return os.path.join(output_dir, SUBMISSION_RECORD_FILE)


def read_submission_record(output_dir: str) -> Optional[SubmissionRecord]:
    path = submission_record_path(output_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return SubmissionRecord.model_validate_json(fh.read())
    except (OSError, ValidationError):
        return None


def write_submission_record(output_dir: str, record: SubmissionRecord) -> None:
    path = submission_record_path(output_dir)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(record.model_dump_json())
        os.replace(tmp, path)
    except OSError:
        pass


def beats_shipped(prior: Optional[SubmissionRecord], loss: Optional[float]) -> bool:
    
    if prior is None or prior.loss is None:
        return True
    if loss is None:
        return False
    return loss < prior.loss


def _attempt_index() -> int:
    try:
        return int(os.environ.get("SN56_ATTEMPT", "0"))
    except ValueError:
        return 0


class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none",
        use_reward_accuracy: bool = False,
        has_dev: bool = False,
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self._original_config = None
        base_config_path = os.path.join(original_model_name, "config.json")
        if os.path.exists(base_config_path):
            try:
                with open(base_config_path) as f:
                    self._original_config = json.load(f)
            except Exception:
                pass
        self.end_time = end_time
        self.use_reward_accuracy = use_reward_accuracy
        self.has_dev = has_dev
        self.early_evals = None

    def _patch_submission_architectures(self):
        
        if not self._original_config:
            return
        config_path = os.path.join(self.submission_dir, "config.json")
        if not os.path.exists(config_path):
            return
        orig_arch = self._original_config.get("architectures")
        if not orig_arch:
            return
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("architectures") != orig_arch:
                cfg["architectures"] = orig_arch
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def compute_loss(self, state: TrainerState, metrics):
        if self.use_reward_accuracy:
            acc = metrics.get("eval_rewards/accuracies")
            if acc is not None:
                return -acc
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            my_state = get_state()
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(my_state["train"]["start_train_time"], "%Y-%m-%d %H:%M:%S")
            
            log_content = f"verificando passo={state.global_step}"
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\npreparação={preparation_time}s"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\ndecorrido={time_so_far}s"
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f"\ntempo/passo={time_for_one_step}s"
            log_content += f"\npassos_total={self.total_steps_all_epochs}"
            total_remaining_training_time = time_for_one_step * (self.total_steps_all_epochs - state.global_step)
            log_content += f"\ntreino_restante={total_remaining_training_time}s"
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\ntempo_restante={total_remaining_time}s"
            
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far + total_remaining_training_time + 12 * 60)) / (time_so_far + max_var_time_sofar)
            n = int(n)
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "end_time": self.end_time,
            }
            if n > 0:
                log_content += f"\npassos_estimados={n}"
                control.should_training_stop = True
                run_state.STOP_REASON["reason"] = run_state.STOP_REASON["reason"] or "end_time_or_max_steps"
                control.should_save = False
                args.save_strategy = "no"
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\ntentativas={n + 1}"
                my_state["next_runs"] = n + 1
            else:
                print(f"sem tempo, terminando", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time":
            log_content = f"segunda verificação, passo={state.global_step}"            
            my_state = get_state()
            current_loss = state.log_history[-1]["loss"]
            my_state["train"]["current_loss"] = current_loss
                
            control.should_training_stop = True
            run_state.STOP_REASON["reason"] = run_state.STOP_REASON["reason"] or "end_time_or_max_steps"

            current_is_the_best = False
            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
            if current_loss <= current_min_loss:
                if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                    print(f"perda={my_state['train']['current_loss']} < mínimo={current_min_loss}, melhor até agora", flush=True)
                    current_is_the_best = True
                    
            if current_is_the_best:
                control.should_training_stop = False
                my_state["mode"] = "finish"
            else:
                control.should_save = False
                args.save_strategy = "no"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
        
            
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            print(f"avaliando passo={state.global_step}, motivo={when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                async_save.drain("custom_eval_save.end_time")
                control.should_training_stop = True
                run_state.STOP_REASON["reason"] = run_state.STOP_REASON["reason"] or "end_time_or_max_steps"
                if not self.has_checkpoint:
                    print(f"sem checkpoint, só salvando no passo={state.global_step}", flush=True)
                    self.save_only = True
                    if not self.has_dev:
                        control.should_evaluate = False

        early = self.early_evals
        if early is not None and when_to_eval["reason"] != "end_time":
            _gate_now = eval_gate_step(self.total_steps_all_epochs, args.num_train_epochs)
            if state.global_step < _gate_now and early.allow(state.global_step, _gate_now):
                control.should_evaluate = True
                control.should_save = True
                return control

        if control.should_evaluate and when_to_eval["reason"] != "end_time":
            if state.global_step < eval_gate_step(
                self.total_steps_all_epochs, args.num_train_epochs
            ):
                control.should_evaluate = False
                control.should_save = False

        return control


    @sg.nonfatal("custom_eval_save.on_evaluate")
    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return 
        print(f"examinando passo={state.global_step}", flush=True)
        if eval_loss is None:
            print(f"perda nula no passo={state.global_step}, ignorando", flush=True)
            return
        self.save_only = False
        if self.best_checkpoint_info is None or eval_loss < self.best_checkpoint_info["loss"]:
            print(f"novo melhor no passo={state.global_step}, perda={eval_loss}", flush=True)
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "step": state.global_step
            }
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f"passo={state.global_step}: perda={eval_loss} >= melhor={self.best_checkpoint_info['loss']}, sem atualização", flush=True)

    @sg.nonfatal("custom_eval_save.on_save")
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        writer = async_save.active()
        if writer is not None:
            if not (step == self.max_steps and self.max_steps != -1):
                view = types.SimpleNamespace(
                    save_only=self.save_only,
                    update_best_checkpoint=self.update_best_checkpoint,
                    best_checkpoint_info=(dict(self.best_checkpoint_info)
                                          if self.best_checkpoint_info is not None else None),
                    has_checkpoint=False)
                self.save_only = False
                self.update_best_checkpoint = False

                def body(view=view):
                    self._save_body(args, step, None, view)
                    if view.has_checkpoint and sg.is_usable_checkpoint(
                            os.path.join(self.output_dir, f"checkpoint-{step}")):
                        self.has_checkpoint = True

                writer.defer(body, label="custom_eval_save.on_save.deferred", step=step)
                return
            writer.drain("custom_eval_save.max_steps")
        self._save_body(args, step, control, self)

    def _save_body(self, args, step: int, control: Optional[TrainerControl], view):
        if sg.save_failed_at(step):
            print(f"salvamento falhou no passo={step}, seguindo", flush=True)
            return

        
        if step == self.max_steps and self.max_steps != -1 and control is not None:
            print(f"teto atingido: {self.max_steps} passos", flush=True)
            control.should_training_stop = True
            run_state.STOP_REASON["reason"] = run_state.STOP_REASON["reason"] or "end_time_or_max_steps"

        view.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK):
            return 
            
        if view.save_only:
            print(f"só salvando, sem avaliação, passo={step}", flush=True)
            current_step = step
            src = os.path.join(self.output_dir, f"checkpoint-{current_step}")
            if not sg.is_usable_checkpoint(src):
                print(f"checkpoint {current_step} incompleto, não enviado", flush=True)
                view.save_only = False
                return
            _prior = read_submission_record(self.output_dir)
            if not beats_shipped(_prior, None):
                print(f"checkpoint {current_step} sem avaliação; mantendo submissão "
                      f"avaliada (tentativa {_prior.attempt}, perda={_prior.loss})", flush=True)
                view.save_only = False
                return
            sg.stage_and_swap(src, self.submission_dir,
                              extra_files={"loss.txt": f"{current_step},no_eval"})
            self._patch_submission_architectures()
            write_submission_record(self.output_dir, SubmissionRecord(
                loss=None, step=current_step, attempt=_attempt_index()))
            view.update_best_checkpoint = False

            view.save_only = False
            return 
            
        if (
            view.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"enviando melhor checkpoint, passo={step}", flush=True)
            best_eval_loss = view.best_checkpoint_info["loss"]
            best_step = view.best_checkpoint_info["step"]
            src = os.path.join(self.output_dir, f"checkpoint-{best_step}")
            if not sg.is_usable_checkpoint(src):
                print(f"melhor checkpoint {best_step} incompleto, submissão mantida",
                      flush=True)
                return
            _prior = read_submission_record(self.output_dir)
            if not beats_shipped(_prior, best_eval_loss):
                print(f"melhor checkpoint {best_step} (perda={best_eval_loss}) não supera a "
                      f"submissão da tentativa {_prior.attempt} (perda={_prior.loss}), mantida",
                      flush=True)
                view.update_best_checkpoint = False
                return
            sg.stage_and_swap(src, self.submission_dir,
                              extra_files={"loss.txt": f"{best_step},{best_eval_loss}"})
            self._patch_submission_architectures()
            write_submission_record(self.output_dir, SubmissionRecord(
                loss=float(best_eval_loss), step=best_step, attempt=_attempt_index()))
            view.update_best_checkpoint = False


VALIDATOR_BETA_GRPO = 0.5


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_reward, eval_kl = self._extract_metrics(state, metrics)
        if eval_reward is None:
            print("recompensa não encontrada nas métricas", flush=True)
            return None

        if eval_kl is not None:
            score = eval_reward - VALIDATOR_BETA_GRPO * eval_kl
            print(
                f"pontuação: recompensa({eval_reward:.4f}) - {VALIDATOR_BETA_GRPO}*kl({eval_kl:.4f}) = {score:.4f}",
                flush=True,
            )
        else:
            score = eval_reward
            print(
                f"pontuação sem kl: recompensa({eval_reward:.4f})",
                flush=True,
            )

        return -score

    @staticmethod
    def _extract_metrics(state: TrainerState, metrics):
        sources = []
        if metrics:
            sources.append(metrics)
        if state.log_history:
            sources.append(state.log_history[-1])

        eval_reward, eval_kl = None, None
        kl_keys = ("eval_kl", "eval_completions/mean_kl", "eval/kl", "eval_mean_kl")
        for src in sources:
            if eval_reward is None:
                eval_reward = src.get("eval_reward")
            if eval_kl is None:
                for k in kl_keys:
                    if k in src:
                        eval_kl = src[k]
                        break
        return eval_reward, eval_kl
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"agora={now} fim={end_time} diff={time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"tempo acabando, avaliando e salvando", flush=True)
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"teto atingido: {self.max_steps} passos", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"falha ao configurar geração para {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"falha ao redimensionar tokens para {model_name}")
        pass


def init_wandb(train_request: Dict):
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True
