

import quiet_mode
import host_cpu
import release_defaults
import save_guard as sg
import argparse
import asyncio
import json
import os
import shutil
import copy
import subprocess
import sys
import uuid
import re
import time 
from datetime import datetime, timezone, timedelta
from typing import Optional

import yaml
from transformers import AutoTokenizer

from state_manager import get_state, set_state
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
import training_paths as train_paths
from instruct_config import get_training_json as get_instruct_training_json
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
from grpo_config import GRPO_VENV_TORCHRUN
import pathlib
from transformers import AutoConfig
import lr_utils
import restore_scaling

def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None):
    with open(log_file_path, "w") as log_file:
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

        log_file.write(f"\nProcess completed with return code: {return_code}\n")

    return return_code


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)([^\s]+))(\s+)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        return None


def extract_value_from_cmd(cmd: str, arg_name: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)(?P<value>[^\s]+))(\s+)", cmd)
    if match:
        return match.group("value")
    else:
        return None


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"


def patch_submission_config(model_path: str, submission_dir: str):
    
    sub_config_path = os.path.join(submission_dir, "config.json")
    base_config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(sub_config_path) or not os.path.exists(base_config_path):
        return
    try:
        with open(base_config_path) as f:
            base_cfg = json.load(f)
        with open(sub_config_path) as f:
            sub_cfg = json.load(f)
        orig_arch = base_cfg.get("architectures")
        if orig_arch and sub_cfg.get("architectures") != orig_arch:
            print(f"[config-patch] Restoring architectures: {sub_cfg.get('architectures')} -> {orig_arch}", flush=True)
            sub_cfg["architectures"] = orig_arch
            with open(sub_config_path, "w") as f:
                json.dump(sub_cfg, f, indent=2)
    except Exception as e:
        print(f"[config-patch] Warning: {e}", flush=True)


def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == "gptossforcausallm":
        return True
    return False


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None


_TRACEBACK_MARKER = "Traceback (most recent call last):"

_LAUNCHER_EXCEPTIONS = frozenset({
    "ChildFailedError", "ProcessRaisedException", "ProcessExitedException",
    "SubprocessError", "CalledProcessError",
})


def _exception_line(chunk: str):
    
    for line in chunk.splitlines()[1:]:
        if not line or line[0].isspace():
            continue
        head, _, message = line.partition(":")
        return head.strip()[:200], message.strip()[:2000]
    return None, None


def _tail(path, nbytes=4096):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
            return fh.read().decode("utf-8", "replace")
    except Exception:
        return None


def exception_from_log(log_path: str, max_bytes: int = 65536):
    
    try:
        text = _tail(log_path, max_bytes) or ""
        launcher = None
        end = len(text)
        while True:
            idx = text.rfind(_TRACEBACK_MARKER, 0, end)
            if idx < 0:
                return None, None, launcher
            head, message = _exception_line(text[idx:end])
            if head is None:
                return None, None, launcher
            if head.rsplit(".", 1)[-1] in _LAUNCHER_EXCEPTIONS:
                launcher = head
                end = idx
                continue
            return head, message, launcher
    except Exception:
        return None, None, None


def extract_output_dir(train_cmd: str) -> str:
    match = re.search(r"--output_dir\s+(.*?)\s+", train_cmd)
    if match:
        return match.group(1)
    else:
        return None


def grpo_main_env_without_vllm(train_cmd: str) -> str:
    
    return train_cmd.replace(GRPO_VENV_TORCHRUN, "torchrun", 1)


_FP32_OOM_FIX_ENV = "SN56_FP32_OOM_FIX"
_FP32_OOM_RELAUNCH_ENV = "SN56_FP32_OOM_RELAUNCH"


def fp32_oom_failsoft(state: dict) -> Optional[dict]:
    if release_defaults.raw(_FP32_OOM_FIX_ENV).strip().lower() in ("0", "false", "no", "off"):
        return None
    bp = (state or {}).get("batch_policy") or {}
    return bp if (bp.get("fp32_master") and bp.get("micro")) else None


def run_training(
    train_cmd: str,
    log_path: str,
    task_id: str,
    retries: int,
    task_type: str,
    expected_repo_name: str,
):
    _log_dir = os.path.dirname(log_path) or "."
    _log_stem, _log_ext = os.path.splitext(os.path.basename(log_path))

    def _attempt_log(index):
        return os.path.join(_log_dir, f"{_log_stem}_a{index}{_log_ext}")

    for i in range(retries):
        os.environ["SN56_ATTEMPT"] = str(i)
        _fp32_relaunch = False
        attempt_log_path = _attempt_log(i)
        prev_log_path = _attempt_log(i - 1) if i else log_path
        if not os.path.exists(prev_log_path):
            prev_log_path = log_path
        print(
            f"************* Training attempt {i+1}/{retries} for task {task_id}*************",
            flush=True,
        )
        if i > 0:
            if os.path.exists(prev_log_path):
                error_type = get_error_type(prev_log_path)
                _fp32_saved = fp32_oom_failsoft(get_state()) if error_type == OOM_ERROR else None
                _fp32_relaunch = _fp32_saved is not None
                if _fp32_relaunch:
                    print(f"[sn56][fp32] OOM under the fp32 master rule: relaunch on the base rule, "
                          f"same batch ({_fp32_saved['micro']} x {_fp32_saved.get('accum', 1)})",
                          flush=True)
                if error_type == OOM_ERROR and not _fp32_relaunch:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        new_batch_size = current_batch_size // 2
                        print(
                            f"Reducing batch size from {current_batch_size} to {new_batch_size}",
                            flush=True,
                        )
                        train_cmd = replace_args_in_cmd(
                            train_cmd,
                            "per_device_train_batch_size",
                            str(new_batch_size),
                        )
                    else:
                        request_path = extract_value_from_cmd(train_cmd, "request_path")
                        _did_reduce_maxlen = False
                        if request_path and os.path.exists(request_path):
                            try:
                                with open(request_path) as _f:
                                    _req = json.load(_f)
                                cur_maxlen = _req.get("train_request", {}).get("max_length")
                                if cur_maxlen is not None and cur_maxlen > 2048:
                                    new_maxlen = max(2048, cur_maxlen // 2)
                                    new_maxlen = ((new_maxlen + 63) // 64) * 64
                                    _req["train_request"]["max_length"] = new_maxlen
                                    with open(request_path, "w") as _f:
                                        json.dump(_req, _f, indent=4, ensure_ascii=False)
                                    print(
                                        f"[sn56][oom-fallback] max_length {cur_maxlen} -> {new_maxlen}",
                                        flush=True,
                                    )
                                    _did_reduce_maxlen = True
                            except Exception as e:
                                print(f"[sn56][oom-fallback] Failed to reduce max_length: {e}", flush=True)

                        if not _did_reduce_maxlen:
                            print(f"batch size is 1, cannot reduce further", flush=True)
                            if task_type == TaskType.GRPOTASK.value:
                                train_cmd = replace_args_in_cmd(
                                    train_cmd, "use_vllm", "False"
                                )
                                train_cmd = grpo_main_env_without_vllm(train_cmd)
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print(f"VLLM OOM error, disable VLLM", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")
                        train_cmd = grpo_main_env_without_vllm(train_cmd)
                elif (task_type == TaskType.GRPOTASK.value
                      and GRPO_VENV_TORCHRUN in train_cmd):
                    print(f"[sn56] GRPO venv attempt failed (not OOM); retrying in main env", flush=True)
                    train_cmd = train_cmd.replace(GRPO_VENV_TORCHRUN, "torchrun", 1)

        if os.path.exists(attempt_log_path):
            with open(attempt_log_path, "w") as f:
                f.write("STARTING TRAINING")
        _attempt_out = extract_value_from_cmd(train_cmd, "output_dir")
        if _attempt_out:
            sg.ship_marker_clear(_attempt_out)

        training_env_vars = {
            "SN56_ATTEMPT": str(i),
            "WANDB_MODE": "offline",
            "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
            "WANDB_NAME": f"{task_id}_{expected_repo_name}",
        }
        if _fp32_relaunch:
            training_env_vars[_FP32_OOM_RELAUNCH_ENV] = "1"
        if task_type == TaskType.GRPOTASK.value:
            training_env_vars["TORCH_CUDNN_SDPA_DEPRIORITIZED"] = "1"
            training_env_vars["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

        run_cmd_with_log(train_cmd, attempt_log_path, env_vars=training_env_vars)

        try:
            shutil.copyfile(attempt_log_path, log_path)
        except Exception:
            pass

        output_dir = extract_value_from_cmd(train_cmd, "output_dir")
        ok = os.path.exists(os.path.join(output_dir, "success.txt"))
        if ok:
            return True
        try:
            _rec = sg.recover_after_child_failure(output_dir) if output_dir else {"shippable": False}
        except Exception as exc:
            _rec = {"shippable": False, "reason": f"recover_raise:{type(exc).__name__}"}
        if _rec.get("shippable"):
            _, _, _ = exception_from_log(attempt_log_path)
            try:
                _err_type = get_error_type(attempt_log_path)
            except OSError:
                pass
            print(f"[sn56] filho morreu após o a0 ({_rec.get('activity')}); "
                  f"enviando {_rec.get('shipped')} sem refazer o treino", flush=True)
            return True
        time.sleep(5)
    return False


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def delete_poor_checkpoints(train_runs: list[dict]):
    lowest_loss = min([run["current_loss"] for run in train_runs])
    for run in train_runs:
        if run["current_loss"] > lowest_loss:
            if os.path.exists(run["output_dir"]):
                print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                shutil.rmtree(run["output_dir"])


def get_log_scale(task_type: str):
    log_scale_map = {
        TaskType.INSTRUCTTEXTTASK.value: 0.18,
        TaskType.DPOTASK.value: 0.18,
        TaskType.GRPOTASK.value: 0.2,
        TaskType.CHATTASK.value: 0.18,
    }
    return log_scale_map[task_type]


def _run_tag(args) -> str:
    
    return (getattr(args, "expected_repo_name", "") or "").rsplit("-", 1)[-1] or "run"


def _run_root(args) -> str:
    return os.path.join(train_cst.OUTPUT_CHECKPOINTS_PATH, args.task_id, _run_tag(args))


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument("--retries", type=int, help="Number of retries", default=8)
    parser.add_argument(
        "--min-steps", type=int, help="Min steps to use for training", default=100
    )

    parser.add_argument(
        "--reg-ratio", type=float, help="Reg ratio to use for training", default=1.0
    )
    parser.add_argument(
        "--baseline-stats",
        type=str,
        default=None,
        help="JSON string of baseline statistics from model-prep, or path to JSON file",
    )

    args = parser.parse_args()

    os.environ.setdefault("SN56_RUN_ROOT", _run_root(args))
    original_model_name = args.model
    original_task_type = args.task_type
    host_cpu.apply("text_trainer")

    baseline_stats = None
    baseline_stats_raw = args.baseline_stats or os.environ.get("BASELINE_STATS_PATH") or os.environ.get("BASELINE_STATS")
    if baseline_stats_raw:
        try:
            if os.path.isfile(baseline_stats_raw):
                with open(baseline_stats_raw, "r") as f:
                    baseline_stats = json.load(f)
            else:
                baseline_stats = json.loads(baseline_stats_raw)
            print(f"Loaded baseline stats (task_type={baseline_stats.get('task_type', 'unknown')})", flush=True)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: failed to parse baseline stats: {e}", flush=True)

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    print(f"submission_dir: {submission_dir}", flush=True)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)
    _stale_record = os.path.join(output_dir, sg.SUBMISSION_RECORD_FILE)
    if os.path.exists(_stale_record):
        os.remove(_stale_record)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))

    is_openai = False
    if is_openai_model(original_model_name):
        print("Upgrading python packages for openai model", flush=True)
        run_cmd_with_log(
            "pip uninstall -y transformers && pip install transformers==4.55.0",
            os.path.join(ds_folder, f"upgrade_transformers.log"),
        )
        run_cmd_with_log(
            "pip uninstall -y deepspeed && pip install deepspeed==0.17.4",
            os.path.join(ds_folder, f"upgrade_deepspeed.log"),
        )
        run_cmd_with_log(
            "pip install kernels==0.9.0", os.path.join(ds_folder, f"install_kernel.log")
        )
        is_openai = True

    train_info = {
        "model_name": original_model_name,
        "model_path": model_path,
        "task_id": args.task_id,
        "dataset": dataset_path,
        "hours_to_complete": args.hours_to_complete,
        "expected_repo_name": args.expected_repo_name,
        "end_time": end_time,
        "dataset_type": dataset_type_dict,
        "submission_dir": submission_dir,
        "output_dir": output_dir,
        "adjust_batch_size": True,
        "request_path": request_path,
        "max_data_size": args.max_data_size,
        "max_steps": args.max_steps,
        "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
        "min_steps": args.min_steps,
        "is_openai": is_openai,
        "reg_ratio": args.reg_ratio,
        "find_lk_lr": True,
        "checking_mode": "first_time",
        "baseline_stats": baseline_stats,
    }

    if (
        args.task_type == TaskType.INSTRUCTTEXTTASK.value
        or args.task_type == TaskType.CHATTASK.value
    ):
        train_info = get_instruct_training_json(train_info)
        tokenize_cmd = (
            f"python tokenize_instruct.py {request_path}"
        )
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.DPOTASK.value:
        train_info = get_dpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_dpo.py {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.GRPOTASK.value:
        train_info = get_grpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_grpo.py {request_path}"
        train_cmd = train_info["run_cmd"]
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    
    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    run_cmd_with_log(
        tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log")
    )

    if restore_scaling.enabled() and args.task_type in (
            TaskType.INSTRUCTTEXTTASK.value, TaskType.CHATTASK.value):
        train_info = restore_scaling.apply_for_training(
            train_info,
            dev_path=os.path.join(ds_folder, f"dev_tokenized_{args.task_id}.json"),
            log=lambda m: print(m, flush=True))

    original_train_cmd = train_cmd
    train_success = False

    original_lr = extract_value_from_cmd(original_train_cmd, "learning_rate")
    print(f"[sn56] Bora. lr={original_lr}, tipo={args.task_type}", flush=True)

    c_train_info = copy.deepcopy(train_info)
    c_train_info["train_request"]["checking_mode"] = "none"

    run_output_dir = output_dir
    train_cmd = replace_args_in_cmd(original_train_cmd, "output_dir", run_output_dir)

    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    with open(request_path, "w") as f:
        json.dump(c_train_info, f, indent=4, ensure_ascii=False)

    train_cmd = replace_args_in_cmd(train_cmd, "request_path", request_path)

    cmd_lr = extract_value_from_cmd(train_cmd, "learning_rate")
    print(f"[sn56] Treino completo: lr={cmd_lr}", flush=True)

    state = {}
    state["mode"] = "finish"
    state["train"] = {
        "train_cmd": train_cmd,
        "log_path": os.path.join(ds_folder, f"train_{args.task_id}.log"),
        "lr": cmd_lr,
        "output_dir": run_output_dir,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    set_state(state)

    log_path = state["train"]["log_path"]
    success = run_training(
        train_cmd,
        log_path,
        args.task_id,
        args.retries,
        args.task_type,
        args.expected_repo_name,
    )

    if not success:
        print(f"[sn56] Treino FALHOU", flush=True)
    else:
        print(f"[sn56] Treino finalizado com sucesso", flush=True)

    _n_files = len(os.listdir(submission_dir)) if os.path.exists(submission_dir) else 0
    _has_weights = sg.has_weights(submission_dir) if os.path.exists(submission_dir) else False
    if _n_files < 2 or not _has_weights:
        print(f"Training failed for task {args.task_id}", flush=True)
    else:
        print(f"Training successfully done for task {args.task_id}", flush=True)
        train_success = True

    if not train_success:
        _exc_type, _exc_message, _ = exception_from_log(log_path)
        _fallback_reason = (
            f"{_exc_type}: {_exc_message}" if _exc_type
            else ("no weights in submission dir" if success else "training run failed")
        )
        try:
            sys.__stderr__.write(
                f"[sn56][NOISE_FALLBACK] task={args.task_id} type={args.task_type} "
                f"train_run_success={bool(success)} has_weights={_has_weights} "
                f"reason={_fallback_reason[:500]!r} -> shipping base model + noise\n"
            )
            sys.__stderr__.flush()
        except Exception:
            pass
        print(f"Training failed for task {args.task_id}", flush=True)
        add_noise_cmd = f"python add_random_noise.py {model_path} {submission_dir}"
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )

    patch_submission_config(model_path, submission_dir)

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
