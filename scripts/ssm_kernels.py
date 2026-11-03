import importlib
import importlib.util
import json
import os
import pathlib
import sys
from types import ModuleType
from typing import Optional

_LEGACY_IMPORT = (
    "from transformers.generation import GreedySearchDecoderOnlyOutput, "
    "SampleDecoderOnlyOutput, TextStreamer"
)
_COMPAT_IMPORT = (
    "try:\n"
    "    from transformers.generation import GreedySearchDecoderOnlyOutput, "
    "SampleDecoderOnlyOutput, TextStreamer\n"
    "except ImportError:\n"
    "    from transformers.generation import GenerateDecoderOnlyOutput as GreedySearchDecoderOnlyOutput\n"
    "    from transformers.generation import GenerateDecoderOnlyOutput as SampleDecoderOnlyOutput\n"
    "    from transformers.generation import TextStreamer"
)
_EXTENSIONS = ("torch", "selective_scan_cuda", "causal_conv1d_cuda")
_STATUS_FILE = pathlib.Path(__file__).resolve().parent / ".ssm_kernels_status"
_BANNER = "=" * 72
_SYMBOLS = (
    ("mamba_ssm", "__version__"),
    ("mamba_ssm.ops.triton.selective_state_update", "selective_state_update"),
    ("mamba_ssm.ops.triton.ssd_combined", "mamba_chunk_scan_combined"),
    ("mamba_ssm.ops.triton.ssd_combined", "mamba_split_conv1d_scan_combined"),
    ("causal_conv1d", "causal_conv1d_fn"),
    ("causal_conv1d", "causal_conv1d_update"),
)
_SSM_MODULE_SUFFIX = "modeling_falcon_h1"
_SSM_MODEL_TYPES = frozenset({"falcon_h1"})
_SSM_PROBES = ("is_mamba_ssm_available", "is_mamba_2_ssm_available")
_SSM_KERNEL_KEYS = ("mamba-ssm", "falcon_mamba-ssm")
_saved_probes: dict = {}


def installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def generation_file() -> Optional[pathlib.Path]:
    try:
        spec = importlib.util.find_spec("mamba_ssm")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(list(spec.submodule_search_locations)[0]) / "utils" / "generation.py"


def patch_generation(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    if _COMPAT_IMPORT in text:
        return "already"
    if _LEGACY_IMPORT not in text:
        return "unexpected"
    path.write_text(text.replace(_LEGACY_IMPORT, _COMPAT_IMPORT, 1), encoding="utf-8")
    return "patched"


def selfcheck() -> list:
    problems = []
    for ext in _EXTENSIONS:
        try:
            importlib.import_module(ext)
        except Exception as exc:
            problems.append(f"{ext}: {type(exc).__name__}: {exc}")
    for mod, name in _SYMBOLS:
        try:
            if getattr(importlib.import_module(mod), name, None) is None:
                problems.append(f"{mod}.{name}: absent")
        except Exception as exc:
            problems.append(f"{mod}.{name}: {type(exc).__name__}: {exc}")
    return problems


def build_status() -> str:
    try:
        return _STATUS_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return "unknown"


def write_build_status(text: str) -> None:
    try:
        _STATUS_FILE.write_text(text.strip() + "\n", encoding="utf-8")
    except Exception:
        pass


def model_type_of(model_path) -> str:
    try:
        with open(os.path.join(str(model_path), "config.json"), encoding="utf-8") as fh:
            return str(json.load(fh).get("model_type") or "").strip().lower()
    except Exception:
        pass
    try:
        from transformers import AutoConfig
        return str(getattr(AutoConfig.from_pretrained(str(model_path)), "model_type", "") or "").strip().lower()
    except Exception:
        return ""


def _never(*_args, **_kwargs) -> bool:
    return False


def _probe_module():
    return importlib.import_module("transformers.utils.import_utils")


def _kernel_cache() -> dict:
    return importlib.import_module("transformers.integrations.hub_kernels")._KERNEL_MODULE_MAPPING


def hide_ssm_kernels() -> None:
    mod = _probe_module()
    for name in _SSM_PROBES:
        fn = getattr(mod, name, None)
        if fn is None or fn is _never:
            continue
        _saved_probes.setdefault(name, fn)
        setattr(mod, name, _never)
    cache = _kernel_cache()
    for key in _SSM_KERNEL_KEYS:
        if isinstance(cache.get(key), ModuleType):
            cache.pop(key)


def expose_ssm_kernels() -> None:
    mod = _probe_module()
    for name, fn in _saved_probes.items():
        setattr(mod, name, fn)
    cache = _kernel_cache()
    for key in _SSM_KERNEL_KEYS:
        if key in cache and cache[key] is None:
            cache.pop(key)


def ssm_kernels_hidden() -> bool:
    return getattr(_probe_module(), _SSM_PROBES[0], None) is _never


def apply(model_path=None, model_type: Optional[str] = None) -> dict:
    status = {
        "mode": "error",
        "model_type": (model_type or "").strip().lower(),
        "mamba_ssm": installed("mamba_ssm"),
        "causal_conv1d": installed("causal_conv1d"),
        "build_status": build_status(),
    }
    try:
        if not status["model_type"] and model_path is not None:
            status["model_type"] = model_type_of(model_path)
        if status["model_type"] in _SSM_MODEL_TYPES:
            expose_ssm_kernels()
            status["mode"] = "exposed" if status["mamba_ssm"] else "absent"
        else:
            hide_ssm_kernels()
            status["mode"] = "hidden"
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    print(f"[ssm_kernels] model_type={status['model_type'] or '?'} mode={status['mode']} "
          f"mamba_ssm={status['mamba_ssm']} causal_conv1d={status['causal_conv1d']} "
          f"build={status['build_status']}", flush=True)
    return status


def fast_path_status(model) -> dict:
    status = {
        "mamba_ssm": installed("mamba_ssm"),
        "causal_conv1d": installed("causal_conv1d"),
        "build_status": build_status(),
        "hidden": ssm_kernels_hidden(),
        "ssm_model": False,
        "fast_path": None,
    }
    try:
        for sub in model.modules():
            modname = type(sub).__module__
            if modname.endswith(_SSM_MODULE_SUFFIX):
                status["ssm_model"] = True
                mod = sys.modules.get(modname)
                status["fast_path"] = bool(getattr(mod, "is_fast_path_available", False))
                break
    except Exception:
        pass
    return status


def main(argv: list) -> int:
    if "--install" in argv:
        path = generation_file()
        if path is None or not path.exists():
            print("ssm kernels: mamba_ssm not installed")
            return 1
        result = patch_generation(path)
        print(f"ssm kernels: generation shim {result} ({path})")
        return 0 if result in ("patched", "already") else 1
    if "--selfcheck" in argv:
        problems = selfcheck()
        if problems:
            print(_BANNER)
            print("ssm kernels: SELF-CHECK FAILED - mamba-ssm will be removed, Mamba layers run the torch path")
            for line in problems:
                print(f"ssm kernels:   {line}")
            print(_BANNER)
            write_build_status("absent: " + "; ".join(problems))
            return 1
        print("ssm kernels: self-check ok")
        write_build_status("present")
        return 0
    print("usage: ssm_kernels.py --install | --selfcheck")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
