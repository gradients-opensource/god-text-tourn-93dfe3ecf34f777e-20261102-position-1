
import math
import time
from typing import Optional

from transformers import AutoConfig

import release_defaults


_TASK_CONFIG = {
    "InstructTextTask": {
        "eta_target": 0.003,
        "ref_loss": 5.5,
        "clamp": (1e-6, 1e-3),
    },
    "ChatTask": {
        "eta_target": 0.003,
        "ref_loss": 5.5,
        "clamp": (1e-6, 1e-3),
    },
    "DpoTask": {
        "eta_target": 0.001,
        "ref_loss": 2.5,
        "task_correction": 0.2,
        "clamp": (5e-7, 3e-5),
    },
    "GrpoTask": {
        "eta_target": 0.001,
        "ref_loss": 3.0,
        "clamp": (1e-6, 1.5e-5),
    },
    "EnvTask": {
        "eta_target": 0.002,
        "ref_loss": 3.0,
        "clamp": (1e-6, 2e-3),
    },
}

_TRAINABLE_GROUPS = [
    "attention_qkv",
    "attention_output",
    "ffn_up",
    "ffn_down",
]

_REF_PARAMS = 1_000_000_000

_MUP_RMS_ENV = "SN56_LR_MUP_RMS"
_FALSY = ("0", "false", "no", "off", "")


def mup_rms_enabled() -> bool:
    return release_defaults.raw(_MUP_RMS_ENV).strip().lower() not in _FALSY


def finder_model_config(model_path):
    if not mup_rms_enabled() or not model_path:
        return None
    try:
        return AutoConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"[sn56][palpite] RMS por grupo: config ilegivel ({type(e).__name__}); sem escala", flush=True)
        return None


def _cfg_get(config, name):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _mult(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) and v > 0 else None


def _dim(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


def _pooled_factor(members: list) -> float:
    n = sum(size for size, _ in members)
    return math.sqrt(sum(size * m * m for size, m in members) / n)


def mup_group_members(model_config) -> dict:
    groups: dict = {}
    mlp = _cfg_get(model_config, "mlp_multipliers")
    if isinstance(mlp, (list, tuple)) and len(mlp) == 2:
        gate, down = _mult(mlp[0]), _mult(mlp[1])
        if gate is not None:
            groups["ffn_up"] = [("gate_proj", 1, gate), ("up_proj", 1, 1.0)]
        if down is not None:
            groups["ffn_down"] = [("down_proj", 1, down)]

    attn_in = _mult(_cfg_get(model_config, "attention_in_multiplier"))
    key = _mult(_cfg_get(model_config, "key_multiplier"))
    attn_out = _mult(_cfg_get(model_config, "attention_out_multiplier"))
    ssm_out = _mult(_cfg_get(model_config, "ssm_out_multiplier"))

    n_heads = _dim(_cfg_get(model_config, "num_attention_heads"))
    n_kv = _dim(_cfg_get(model_config, "num_key_value_heads")) or n_heads
    hidden = _dim(_cfg_get(model_config, "hidden_size"))
    head_dim = _dim(_cfg_get(model_config, "head_dim"))
    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads
    q_width = n_heads * head_dim if n_heads and head_dim else None
    kv_width = n_kv * head_dim if n_kv and head_dim else None

    if attn_in is not None or key is not None:
        a_in = attn_in if attn_in is not None else 1.0
        k_m = key if key is not None else 1.0
        if q_width and kv_width:
            sizes = (q_width, kv_width, kv_width)
        else:
            sizes = (1, 1, 1)
        groups["attention_qkv"] = [("q_proj", sizes[0], a_in), ("k_proj", sizes[1], a_in * k_m),
                                   ("v_proj", sizes[2], a_in)]

    if attn_out is not None or ssm_out is not None:
        members = []
        d_ssm = _dim(_cfg_get(model_config, "mamba_d_ssm"))
        if d_ssm is None:
            expand = _cfg_get(model_config, "mamba_expand")
            if hidden and isinstance(expand, (int, float)) and not isinstance(expand, bool) and expand > 0:
                d_ssm = int(expand * hidden)
        o_size, s_size = (q_width, d_ssm) if (q_width and d_ssm) else (1, 1)
        members.append(("o_proj", o_size, attn_out if attn_out is not None else 1.0))
        if ssm_out is not None:
            members.append(("ssm_out_proj", s_size, ssm_out))
        groups["attention_output"] = members
    return groups


def mup_group_factors(model_config) -> dict:
    factors = {}
    for group, members in mup_group_members(model_config).items():
        f = _pooled_factor([(size, m) for _, size, m in members])
        if f != 1.0:
            factors[group] = f
    return factors


def _aggregate_weight_rms(weights: dict, factors: Optional[dict] = None) -> float:
    by_group = weights.get("by_group", {})
    rms_values = []
    for group in _TRAINABLE_GROUPS:
        if group in by_group:
            rms = by_group[group].get("weight_rms")
            if rms is not None and rms > 0:
                if factors and group in factors:
                    rms = rms * factors[group]
                rms_values.append(rms)

    if not rms_values:
        for group_stats in by_group.values():
            rms = group_stats.get("weight_rms")
            if rms is not None and rms > 0:
                rms_values.append(rms)

    if not rms_values:
        return 0.02

    rms_values.sort()
    n = len(rms_values)
    if n % 2 == 1:
        return rms_values[n // 2]
    return (rms_values[n // 2 - 1] + rms_values[n // 2]) / 2


def _get_init_loss(training: dict, task_type: str) -> float:
    if task_type in ("InstructTextTask", "ChatTask"):
        masked = training.get("masked_completion_loss")
        if masked is not None and masked > 0:
            return masked

    init_loss = training.get("init_loss")
    if init_loss is not None and init_loss > 0:
        return init_loss

    return 3.0


def compute_lr_from_stats(
    baseline_stats: dict,
    task_type: str,
    param_count: int,
    effective_batch_size: int,
    model_config=None,
) -> float:
    t_start = time.perf_counter()
    cfg = _TASK_CONFIG.get(task_type, _TASK_CONFIG["InstructTextTask"])
    training = baseline_stats.get("training", {})
    weights = baseline_stats.get("weights", {})

    if task_type == "GrpoTask":
        lr = _compute_grpo_lr(param_count, cfg)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        print(
            f"[sn56][palpite] lr={lr:.2e} (GRPO) in {elapsed_ms:.1f}ms "
            f"(params={param_count/1e9:.2f}B, scale={math.sqrt(_REF_PARAMS / max(param_count, 1)):.3f})",
            flush=True,
        )
        return lr


    mup_factors = mup_group_factors(model_config) if mup_rms_enabled() else {}
    if mup_factors:
        weight_rms_raw = _aggregate_weight_rms(weights)
        weight_rms = _aggregate_weight_rms(weights, mup_factors)
    else:
        weight_rms = _aggregate_weight_rms(weights)
    lr_base = cfg["eta_target"] * weight_rms

    init_loss = _get_init_loss(training, task_type)
    curvature_factor = 1.0

    gns = training.get("gradient_noise_scale", 0.0)
    if gns > 0 and effective_batch_size > 0:
        noise_factor = math.sqrt(
            effective_batch_size / (effective_batch_size + gns)
        )
    else:
        noise_factor = 1.0

    if param_count > 0:
        scale_factor = math.sqrt(_REF_PARAMS / param_count)
    else:
        scale_factor = 1.0

    task_correction = cfg.get("task_correction", 1.0)

    lr = lr_base * curvature_factor * noise_factor * scale_factor * task_correction

    lo, hi = cfg["clamp"]
    if mup_factors:
        lr_unscaled = max(lo, min(hi, lr * weight_rms_raw / weight_rms))
    lr = max(lo, min(hi, lr))

    if mup_factors:
        _emit_mup_rms(weights, model_config, mup_factors, weight_rms_raw, weight_rms, lr,
                      lr_unscaled=lr_unscaled, task_type=task_type)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    print(
        f"[sn56][palpite] lr={lr:.2e} computed in {elapsed_ms:.1f}ms "
        f"(base={lr_base:.2e}, curvature={curvature_factor:.3f}, "
        f"noise={noise_factor:.3f}, scale={scale_factor:.3f}, "
        f"task_corr={task_correction:.3f}, "
        f"weight_rms={weight_rms:.4f}, loss={init_loss:.3f}, ref_loss={cfg['ref_loss']:.1f})",
        flush=True,
    )

    return lr


def _emit_mup_rms(weights, model_config, factors, rms_raw, rms_eff, lr, lr_unscaled, task_type):
    print(
        f"[sn56][palpite] RMS efetivo dos pesos: mediana {rms_raw:.4f} -> {rms_eff:.4f} "
        f"(x{rms_eff / rms_raw:.3f}; factors {', '.join(f'{g}={f:.3f}' for g, f in factors.items())}); "
        f"lr {lr_unscaled:.2e} -> {lr:.2e}",
        flush=True,
    )


_GRPO_BASE_LR = 1e-5


def _compute_grpo_lr(param_count: int, cfg: dict) -> float:
    scale = math.sqrt(_REF_PARAMS / max(param_count, 1))
    lr = _GRPO_BASE_LR * scale
    lo, hi = cfg["clamp"]
    return max(lo, min(hi, lr))


def estimate_starting_lr(
    baseline_stats: Optional[dict],
    task_type: str,
    param_count: int,
    effective_batch_size: int,
    fallback_lr: Optional[float] = None,
    model_config=None,
) -> Optional[float]:
    if baseline_stats is None:
        if fallback_lr is not None:
            print(
                f"[sn56][palpite] No baseline stats, using fallback lr={fallback_lr:.2e}",
                flush=True,
            )
        return fallback_lr

    training = baseline_stats.get("training")
    weights = baseline_stats.get("weights")

    if not training or not weights:
        print(
            "[sn56][palpite] baseline_stats present but missing training/weights, "
            f"using fallback lr={fallback_lr}",
            flush=True,
        )
        return fallback_lr

    computed_lr = compute_lr_from_stats(
        baseline_stats, task_type, param_count, effective_batch_size, model_config=model_config
    )

    if fallback_lr is not None and fallback_lr > 0 and computed_lr > 0:
        ratio = computed_lr / fallback_lr
        print(
            f"[sn56][palpite] computed={computed_lr:.2e} vs bucket={fallback_lr:.2e} "
            f"({ratio:.1f}x)",
            flush=True,
        )

    return computed_lr
