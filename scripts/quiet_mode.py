
import builtins
import functools
import logging
import os
import sys


_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")

_real_print = builtins.print


def _quiet_print(*args, **kwargs):
    if kwargs.get("file", None) is sys.stderr:
        return _real_print(*args, **kwargs)
    return None


try:
    functools.update_wrapper(_quiet_print, _real_print)
except Exception:
    pass


def _is_quiet() -> bool:
    if os.environ.get("SN56_VERBOSE", "").strip().lower() in _TRUTHY:
        return False
    if os.environ.get("SN56_QUIET", "").strip().lower() in _FALSY:
        return False
    return True


def _silence_libraries() -> None:
    env_defaults = {
        "TQDM_DISABLE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "DATASETS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_LOGGING_LEVEL": "ERROR",
        "BITSANDBYTES_NOWELCOME": "1",
        "ACCELERATE_LOG_LEVEL": "error",
        "DATASETS_DISABLE_PROGRESS_BARS": "1",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)

    logging.disable(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR)
    for name in (
        "transformers",
        "datasets",
        "accelerate",
        "deepspeed",
        "vllm",
        "axolotl",
        "torch",
        "huggingface_hub",
        "filelock",
        "urllib3",
    ):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass


def _silence_print() -> None:
    builtins.print = _quiet_print


def _main() -> None:
    if not _is_quiet():
        return
    try:
        _silence_libraries()
    except Exception:
        pass
    try:
        _silence_print()
    except Exception:
        pass


_main()
