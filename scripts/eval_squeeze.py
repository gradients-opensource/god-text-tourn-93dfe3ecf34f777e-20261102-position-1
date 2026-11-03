import os
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


ENV_MIDEVAL = "SN56_SKIP_MEMBER_MIDEVAL"
ENV_REUSE = "SN56_REUSE_BASE_EVAL"


def _flag(name: str) -> str:
    return os.environ.get(name, "1").strip().lower()


def skip_member_mideval_enabled() -> bool:
    return _flag(ENV_MIDEVAL) not in ("0", "false", "no", "off")


def reuse_base_mode() -> str:
    v = _flag(ENV_REUSE)
    if v in ("0", "false", "no", "off"):
        return "off"
    if v == "verify":
        return "verify"
    return "on"


class EvalConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset: Any = None
    plan: Any = None
    batched: bool = False
    eval_batch_size: int = 0

    def same_as(self, other: "EvalConfig") -> bool:
        return (self.dataset is not None and self.dataset is other.dataset
                and self.plan is other.plan and self.batched == other.batched
                and self.eval_batch_size == other.eval_batch_size)


def eval_config(trainer) -> Optional[EvalConfig]:
    if trainer is None:
        return None
    try:
        return EvalConfig(dataset=getattr(trainer, "eval_dataset", None),
                          plan=getattr(trainer, "_evb_plan", None),
                          batched=bool(getattr(trainer, "_evb_on", False)),
                          eval_batch_size=int(trainer.args.per_device_eval_batch_size or 0))
    except Exception:
        return None


class SoupEvalRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    state_id: int
    loss: float
    config: Optional[EvalConfig] = None
    secs_per_cand: Optional[float] = None


def base_reuse_check(record: Optional[SoupEvalRecord], n: int, trainer, world: int) -> str:
    if record is None:
        return "no_record"
    if world != 1:
        return "world_size"
    if int(n) != 1:
        return "acc_not_base"
    cfg = eval_config(trainer)
    if record.config is None or cfg is None or not record.config.same_as(cfg):
        return "eval_config"
    return ""


class _Saved(BaseModel):
    n: int = 0
    secs: float = 0.0


class Ledger(BaseModel):
    member_mideval: _Saved = _Saved()
    base_eval: _Saved = _Saved()

    def add(self, what: str, secs: Optional[float], n: int = 1) -> None:
        s = getattr(self, what)
        s.n += int(n)
        s.secs += float(secs or 0.0)

    def emit(self) -> None:
        pass


LEDGER = Ledger()
