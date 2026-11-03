from typing import Optional

import release_defaults

ENV = "SN56_PHI_RTN_COSINE"
MODEL_TYPE = "phi"
WRITEBACK = "rtn"
_FALSY = ("", "0", "false", "no", "off")

_ARMED: Optional[dict] = None


def enabled() -> bool:
    return release_defaults.raw(ENV).strip().lower() not in _FALSY


def applies(model_type) -> bool:
    return enabled() and isinstance(model_type, str) and model_type == MODEL_TYPE


def model_type_of(model) -> Optional[str]:
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None)
    return mt if isinstance(mt, str) else None


def arm(model_type) -> Optional[dict]:
    global _ARMED
    if not applies(model_type):
        return None
    if _ARMED is None:
        _ARMED = {"model_type": model_type, "writeback": WRITEBACK, "fp32_master": False,
                  "a0_schedule": "cosine_a0", "carry_lr_cal": "suppressed"}
        print(f"[sn56][precisao] model_type={model_type}: writeback rtn, sem fp32 master, "
              f"a0 cosseno", flush=True)
    return dict(_ARMED)


def arm_from_model(model) -> Optional[dict]:
    return arm(model_type_of(model))


def armed() -> bool:
    return _ARMED is not None


def reset() -> None:
    global _ARMED
    _ARMED = None
