from __future__ import annotations

import datetime
import math
import os
import time
from typing import Callable, Optional

import torch
from pydantic import BaseModel

import dev_pass
import length_grouping
import planner_clock
import save_guard as sg
import spare_budget as sb
import release_defaults

_FALSY = ("0", "false", "no", "off")
GRID = (0.8, 0.85, 0.9, 0.95, 1.05)
REL_FLOOR = 1e-3
PROBE_T = 0.9
PROBE_TOL = 1e-2
NORM_NAMES = ("norm", "ln_f", "final_layernorm", "final_layer_norm", "norm_f", "final_norm")


def enabled() -> bool:
    if not length_grouping.dealcal_enabled():
        return False
    return (release_defaults.raw("SN56_LOGIT_CALIB") or "0").strip().lower() not in _FALSY


def n_evals() -> int:
    return len(GRID) + 1


def price_secs(t_eval: Optional[float], t_write: Optional[float]) -> float:
    if not t_eval or t_eval <= 0 or not math.isfinite(float(t_eval)):
        return 0.0
    return n_evals() * float(t_eval) + max(0.0, float(t_write or 0.0))


def price_tail(tail, t_eval: Optional[float], t_write: Optional[float]):
    if not enabled():
        return tail
    extra = price_secs(t_eval, t_write)
    if extra <= 0:
        return tail
    return tail.model_copy(update={"reserve": float(tail.reserve) + extra})


class CalibTarget(BaseModel):
    kind: str
    form: str
    names: list[str]
    params: list = []
    one_plus: list = []

    model_config = {"arbitrary_types_allowed": True}


class CalibResult(BaseModel):
    site: str
    t: float = 1.0
    kept: bool = False
    reason: str = ""
    dev_before: Optional[float] = None
    dev_after: Optional[float] = None
    se: Optional[float] = None
    threshold: Optional[float] = None
    target: Optional[str] = None
    form: Optional[str] = None
    names: list[str] = []
    grid: dict = {}
    need_secs: Optional[float] = None
    budget_secs: Optional[float] = None
    secs: Optional[float] = None


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _softcapped(model) -> bool:
    cfg = getattr(model, "config", None)
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is not None and getattr(c, "final_logit_softcapping", None):
            return True
    return False


def _tied(model, head) -> bool:
    try:
        emb = model.get_input_embeddings()
    except Exception:
        emb = None
    w = getattr(head, "weight", None)
    ew = getattr(emb, "weight", None)
    if w is None or ew is None:
        return False
    return w is ew or (w.data_ptr() == ew.data_ptr() and w.shape == ew.shape)


def _final_norm(model):
    base = getattr(model, "base_model", None) or model
    for holder, prefix in ((base, ""), (getattr(base, "model", None), "model.")):
        if holder is None:
            continue
        for n in NORM_NAMES:
            m = getattr(holder, n, None)
            if isinstance(m, torch.nn.Module) and getattr(m, "weight", None) is not None:
                return prefix + n, m
    return None, None


def _param_name(model, p) -> str:
    for n, q in model.named_parameters():
        if q is p:
            return n
    return "?"


@torch.no_grad()
def apply_scale(target: CalibTarget, t: float) -> None:
    for p, op in zip(target.params, target.one_plus):
        if op:
            p.copy_(((1.0 + p.float()) * float(t) - 1.0).to(p.dtype))
        else:
            p.mul_(float(t))


def snapshot(target: CalibTarget) -> list:
    return [p.detach().to("cpu", copy=True) for p in target.params]


@torch.no_grad()
def restore(target: CalibTarget, saved: list) -> None:
    for p, s in zip(target.params, saved):
        p.copy_(s.to(p.device))


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    den = float(b.norm())
    return float((a - b).norm()) / den if den > 0 else float((a - b).norm())


@torch.no_grad()
def resolve_target(model, probe_ids: Optional[torch.Tensor] = None) -> tuple:
    m = _unwrap(model)
    if _softcapped(m):
        return None, "softcapped_head"
    try:
        head = m.get_output_embeddings()
    except Exception:
        head = None
    if head is None or getattr(head, "weight", None) is None:
        return None, "no_output_projection"
    hb = getattr(head, "bias", None)
    if not _tied(m, head):
        params = [head.weight] + ([hb] if hb is not None else [])
        return CalibTarget(kind="lm_head", form="mul",
                           names=[_param_name(m, p) for p in params], params=params,
                           one_plus=[False] * len(params)), ""
    nname, norm = _final_norm(m)
    if norm is None:
        return None, "tied_no_final_norm"
    nb = getattr(norm, "bias", None)
    if probe_ids is None:
        dev = next(m.parameters()).device
        vocab = int(head.weight.shape[0])
        probe_ids = (torch.arange(4, device=dev) % max(1, vocab - 1) + 1).unsqueeze(0)
    got: dict = {}

    def _norm_hook(_mod, args, out):
        got["x"] = args[0].detach()
        got["y"] = (out[0] if isinstance(out, tuple) else out).detach()

    def _head_hook(_mod, args):
        got["z"] = args[0].detach()

    h1 = norm.register_forward_hook(_norm_hook)
    h2 = head.register_forward_pre_hook(_head_hook)
    was_training = m.training
    try:
        m.eval()
        m(input_ids=probe_ids)
    except Exception as exc:
        return None, f"probe_forward_failed:{type(exc).__name__}"
    finally:
        h1.remove()
        h2.remove()
        if was_training:
            m.train()
    if "y" not in got or "z" not in got or got["y"].shape != got["z"].shape \
            or _rel_err(got["z"], got["y"]) > PROBE_TOL:
        return None, "norm_not_feeding_head"
    params = [norm.weight] + ([nb] if nb is not None else []) + ([hb] if hb is not None else [])
    names = [f"{nname}.weight"] + ([f"{nname}.bias"] if nb is not None else []) + \
        ([_param_name(m, hb)] if hb is not None else [])
    want = got["y"].float() * PROBE_T
    for form in ("mul", "one_plus"):
        op = [form == "one_plus"] + [False] * (len(params) - 1)
        tgt = CalibTarget(kind="final_norm", form=form, names=names, params=params, one_plus=op)
        saved = snapshot(tgt)
        try:
            apply_scale(tgt, PROBE_T)
            out = norm(got["x"])
            y2 = out[0] if isinstance(out, tuple) else out
            ok = _rel_err(y2, want) <= PROBE_TOL
        finally:
            restore(tgt, saved)
        if ok:
            return tgt, ""
    return None, "norm_not_linear_in_scale"


def trainer_evaluator(trainer, ckpt_avg=None) -> Callable[[], tuple]:
    def _ev():
        tr = trainer if trainer is not None else getattr(ckpt_avg, "trainer", None)
        if ckpt_avg is not None:
            ckpt_avg._evaluating = True
        try:
            met = tr.evaluate() or {}
        finally:
            if ckpt_avg is not None:
                ckpt_avg._evaluating = False
        dev = met.get("eval_dev_macro", met.get("eval_loss"))
        rows = getattr(tr, "last_eval_rows", None)
        return (float(dev) if dev is not None else float("nan"),
                [float(x) for x in rows] if rows else None)
    return _ev


def decide(dev0: float, rows0: Optional[list], trials: dict) -> tuple:
    fin = {t: v for t, v in trials.items() if v[0] == v[0] and math.isfinite(v[0])}
    if not fin:
        return 1.0, False, "grid_eval_failed", None, None, None
    tb = min(fin, key=lambda t: fin[t][0])
    dv, rows = fin[tb]
    se = sb.paired_se(rows0, rows)
    thr = max(REL_FLOOR * abs(dev0), se or 0.0)
    if se is None:
        return 1.0, False, "no_paired_se", dv, None, thr
    if dev0 - dv > thr:
        return float(tb), True, "improves", dv, se, thr
    return 1.0, False, "within_noise", dv, se, thr


def calibrate(model, evaluate: Callable[[], tuple], *, site: str, budget_secs: Optional[float],
              t_eval: Optional[float], t_write: Optional[float], multi: bool = False,
              log: Callable[[str], None] = print) -> Optional[CalibResult]:
    if not enabled():
        return None
    t0 = time.time()
    res = CalibResult(site=site)
    need = price_secs(t_eval, t_write)
    res.need_secs = round(need, 2)
    res.budget_secs = None if budget_secs is None else round(float(budget_secs), 2)

    def _done(reason: str) -> CalibResult:
        res.reason = res.reason or reason
        res.secs = round(time.time() - t0, 2)
        log(f"[sn56][calib] {site}: t={res.t} kept={res.kept} ({res.reason})")
        return res

    if multi:
        return _done("multi_gpu")
    if need <= 0:
        return _done("no_eval_time")
    if budget_secs is None or float(budget_secs) < need:
        return _done("no_time")
    try:
        tgt, why = resolve_target(model)
    except Exception as exc:
        tgt, why = None, f"probe_failed:{type(exc).__name__}"
    if tgt is None:
        return _done(why)
    res.target, res.form, res.names = tgt.kind, tgt.form, list(tgt.names)
    deadline = t0 + float(budget_secs) - max(0.0, float(t_write or 0.0))
    saved = snapshot(tgt)
    try:
        dev0, rows0 = evaluate()
        res.dev_before = dev0
        if not (dev0 == dev0 and math.isfinite(dev0)):
            return _done("eval_failed")
        trials: dict = {}
        for t in GRID:
            if time.time() + float(t_eval) > deadline:
                res.reason = "clock_cut_grid"
                break
            apply_scale(tgt, t)
            try:
                trials[t] = evaluate()
            finally:
                restore(tgt, saved)
        res.grid = {str(t): round(v[0], 6) for t, v in trials.items()}
        t_best, kept, reason, dv, se, thr = decide(dev0, rows0, trials)
        res.dev_after, res.se, res.threshold = dv, se, thr
        res.reason = reason if not res.reason else f"{res.reason}:{reason}"
        if kept:
            apply_scale(tgt, t_best)
            res.t, res.kept = float(t_best), True
    except Exception as exc:
        restore(tgt, saved)
        res.t, res.kept = 1.0, False
        res.reason = f"failed:{type(exc).__name__}: {exc}"[:300]
    return _done(res.reason)


def _remaining(end_time: Optional[str]) -> Optional[float]:
    if not end_time:
        return None
    end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=datetime.timezone.utc)
    return (end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()


def calibrate_a0_ship(trainer, ckpt_avg, train_request: dict, facts, devpass_budget: Optional[float],
                      multi: bool, is_main: bool, log: Callable[[str], None] = print
                      ) -> Optional[CalibResult]:
    try:
        left = _remaining(train_request.get("end_time"))
        t_eval = getattr(facts, "eval_runtime", None) if facts is not None else None
        t_write = planner_clock.write_secs(getattr(facts, "clock_stats", None)) \
            if facts is not None else 0.0
        res = calibrate(trainer.model, trainer_evaluator(trainer, ckpt_avg), site="a0_soup",
                        budget_secs=(None if left is None else left - float(devpass_budget or 0.0)),
                        t_eval=t_eval, t_write=t_write, multi=multi, log=log)
        if res is not None and res.kept and is_main:
            sg.ship_marker_update(activity="logit_calib_write")
            if dev_pass._save_weights_only(_unwrap(trainer.model),
                                           train_request["submission_dir"], log):
                sg.ship_marker_update(activity="logit_calib_written")
        return res
    except Exception:
        return None
