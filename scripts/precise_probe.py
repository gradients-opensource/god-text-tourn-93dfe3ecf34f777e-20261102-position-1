import math
import os
import re
import time
from typing import Optional

import torch


CLASSES = ("embed", "attn", "mlp", "norm", "lm_head", "other")
SEED = 1234
_FALSY = ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


EVERY = max(1, _env_int("SN56_PRECISE_PROBE_EVERY", 25))
ELEMS = max(16, _env_int("SN56_PRECISE_PROBE_ELEMS", 4096))
TENSORS = max(1, _env_int("SN56_PRECISE_PROBE_TENSORS", 8))
SHADOW_EVERY_STEP = os.environ.get("SN56_PRECISE_PROBE_SHADOW", "1").strip().lower() not in _FALSY

_PATTERNS = (
    ("lm_head", re.compile(r"lm_head|embed_out|(^|\.)output\.weight$")),
    ("norm", re.compile(r"norm|(^|\.)ln_|(^|\.)ln\d")),
    ("embed", re.compile(r"embed|(^|\.)wte|(^|\.)wpe|tok_emb|word_emb")),
    ("attn", re.compile(r"attn|attention|q_proj|k_proj|v_proj|o_proj|qkv|out_proj|(^|\.)w[qkvo](\.|$)")),
    ("mlp", re.compile(r"mlp|ffn|feed_forward|gate_proj|up_proj|down_proj|fc1|fc2|dense_h_to_4h"
                       r"|dense_4h_to_h|(^|\.)w[123](\.|$)|c_fc|c_proj")),
)


def param_class(name: str) -> str:
    n = (name or "").lower()
    for cls, pat in _PATTERNS:
        if pat.search(n):
            return cls
    return "other"


def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1).index_select(0, idx)


def _rtn(x32: torch.Tensor) -> torch.Tensor:
    return x32.to(torch.bfloat16).to(torch.float32)


def _ulp(x: torch.Tensor) -> torch.Tensor:
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
    return torch.ldexp(torch.ones_like(x), (e - 7).to(torch.int32))


def _rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt()) if x.numel() else 0.0


def _ratio(a: float, b: float) -> Optional[float]:
    return a / b if b > 0 else None


def _frac(cond: torch.Tensor, where: torch.Tensor) -> Optional[float]:
    n = int(where.sum())
    return float(cond[where].float().mean()) if n else None


def _proj(real: torch.Tensor, intended: torch.Tensor) -> Optional[float]:
    den = float((intended * intended).sum())
    return float((real * intended).sum()) / den if den > 0 else None


class Sample:

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _or_w(s):
    return s.w1 if s.master1 is None else s.master1


METRICS = (
    ("n", lambda s: int(s.w0.numel())),
    ("survival", lambda s: _frac(s.w1 != s.w0, s.delta != 0)),
    ("survival_shadow_step", lambda s: _frac(_rtn(s.w0 + s.delta) != s.w0, s.delta != 0)),
    ("survival_shadow_cum", lambda s: _frac(s.shadow1 != s.shadow0, s.delta != 0)),
    ("flush_rate", lambda s: float((s.w1 != s.w0).float().mean())),
    ("step_rms_intended", lambda s: _rms(s.delta)),
    ("step_rms_real", lambda s: _rms(s.w1 - s.w0)),
    ("step_ratio", lambda s: _ratio(_rms(s.w1 - s.w0), _rms(s.delta))),
    ("step_ratio_shadow", lambda s: _ratio(_rms(_rtn(s.w0 + s.delta) - s.w0), _rms(s.delta))),
    ("step_ratio_master", lambda s: None if s.master1 is None
     else _ratio(_rms(s.master1 - s.master0), _rms(s.delta))),
    ("delta_over_ulp_median", lambda s: float((s.delta.abs() / s.ulp0).median())),
    ("wd_intended_rms", lambda s: _rms(s.wd0)),
    ("wd_realised_ratio", lambda s: _proj(s.w1 - s.w0, s.wd0)),
    ("drift_shadow_rms_rel", lambda s: _ratio(_rms(s.shadow1 - s.w1), _rms(s.w1))),
    ("master_resid_rms_over_ulp", lambda s: None if s.master1 is None
     else _ratio(_rms(s.master1 - s.w1), _rms(s.ulp0))),
    ("master_shadow_rms_rel", lambda s: None if s.master1 is None
     else _ratio(_rms(s.shadow1 - s.master1), _rms(s.master1))),
    ("resid_abs_mean_over_ulp", lambda s: None if s.master1 is None
     else float(((s.master1 - s.w1).abs() / s.ulp1).mean())),
    ("resid_sat_frac", lambda s: None if s.master1 is None
     else float((((s.master1 - s.w1).abs() / s.ulp1) >= 127.0 / 256.0).float().mean())),
    ("cum_disp_rms_rel", lambda s: _ratio(_rms(_or_w(s) - s.init), _rms(s.init))),
    ("shadow_cum_disp_ratio", lambda s: _ratio(_rms(s.shadow1 - s.init), _rms(_or_w(s) - s.init))),
    ("frozen_frac_shadow_cum", lambda s: float((s.shadow1 == s.init).float().mean())),
    ("frozen_frac", lambda s: float((s.w1 == s.init).float().mean())),
    ("zero_frac", lambda s: float((s.w1 == 0).float().mean())),
    ("w_rms", lambda s: _rms(s.w1)),
)


def memory_snapshot(stage: str, phase: str, model=None, device=None, **extra) -> dict:
    f = {"stage": stage, "phase": phase, **extra}
    dev = device
    if model is not None and dev is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = None
    if dev is not None and dev.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(dev)
        f.update(alloc_bytes=int(torch.cuda.memory_allocated(dev)),
                 reserved_bytes=int(torch.cuda.memory_reserved(dev)),
                 peak_bytes=int(torch.cuda.max_memory_allocated(dev)),
                 free_bytes=int(free), total_bytes=int(total))
    return f


class _Pick:
    def __init__(self, cls, gi, p, idx):
        self.cls, self.gi, self.p, self.idx = cls, gi, p, idx
        self.init = _gather(p, idx).to(torch.float32)
        self.shadow = self.init.clone()
        self.w0 = None
        self.master0 = None
        self.lost = 0.0
        self.mass = 0.0


class Probe:
    def __init__(self, opt):
        self.opt = opt
        self.picks: list = []
        self.timed = False
        self.times: dict = {}
        self._t_last = 0.0
        self.step_idx = 0
        self.first_done = False
        self.lr_prev: Optional[float] = None
        self.lr_max = 0.0
        by_cls: dict = {c: [] for c in CLASSES}
        for gi, group in enumerate(opt.param_groups):
            for p in group["params"]:
                by_cls[param_class(opt.names.get(id(p), ""))].append((gi, p))
        g = torch.Generator().manual_seed(SEED)
        layout = {}
        for c, items in by_cls.items():
            if not items:
                continue
            if len(items) > TENSORS:
                keep = sorted(torch.randperm(len(items), generator=g)[:TENSORS].tolist())
                items = [items[i] for i in keep]
            tot = sum(int(p.numel()) for _, p in items)
            n_el = 0
            for gi, p in items:
                k = max(1, min(int(p.numel()), int(round(ELEMS * p.numel() / max(1, tot)))))
                idx = torch.randint(0, int(p.numel()), (k,), generator=g).unique().to(p.device)
                self.picks.append(_Pick(c, gi, p, idx))
                n_el += int(idx.numel())
            zeros = sum(int((pk.init == 0).sum()) for pk in self.picks if pk.cls == c)
            layout[c] = {"n_tensors": len(items), "n_elems": n_el,
                         "zero_frac_init": zeros / max(1, n_el),
                         "names": [opt.names.get(id(p), "?") for _, p in items][:TENSORS]}

    def _sync(self):
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def _hp(self, gi: int):
        g = self.opt.param_groups[gi]
        b1, b2 = g["betas"]
        return float(g["lr"]), float(b1), float(b2), float(g["eps"]), float(g["weight_decay"])

    def _intended(self, pick: _Pick, w_for_wd: torch.Tensor) -> torch.Tensor:
        st = self.opt.state.get(pick.p) or {}
        if "step" not in st:
            return torch.zeros_like(w_for_wd)
        mv = self.opt.moments_at(pick.p, pick.idx)
        if mv is None:
            if "exp_avg" not in st or "exp_avg_sq" not in st:
                return torch.zeros_like(w_for_wd)
            mv = (_gather(st["exp_avg"], pick.idx), _gather(st["exp_avg_sq"], pick.idx))
        lr, b1, b2, eps, wd = self._hp(pick.gi)
        t = float(st["step"])
        m, v = mv[0].to(torch.float32), mv[1].to(torch.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        adam = (m / max(bc1, 1e-30)) / ((v / max(bc2, 1e-30)).sqrt() + eps)
        return -lr * adam - lr * wd * w_for_wd

    def _lr_trend(self, lr: float) -> str:
        prev, self.lr_prev = self.lr_prev, lr
        self.lr_max = max(self.lr_max, lr)
        if prev is None:
            return "start"
        if lr > prev * (1 + 1e-6):
            return "rising"
        if lr < prev * (1 - 1e-6):
            return "falling"
        return "flat_max" if lr >= self.lr_max * (1 - 1e-6) else "flat_min"

    def begin_step(self):
        self.step_idx = int(self.opt.n_steps)
        self.timed = (self.step_idx % EVERY == 0)
        if not self.timed:
            return
        self._sync()
        self.times = {}
        self._t_last = time.perf_counter()
        for pick in self.picks:
            pick.w0 = _gather(pick.p, pick.idx).to(torch.float32)
            m = self.opt.master_at(pick.p, pick.idx)
            pick.master0 = m.to(torch.float32) if m is not None else None

    def mark(self, name: str):
        if not self.timed:
            return
        self._sync()
        now = time.perf_counter()
        self.times[name] = self.times.get(name, 0.0) + (now - self._t_last)
        self._t_last = now

    @torch.no_grad()
    def end_step(self):
        if not (self.timed or SHADOW_EVERY_STEP):
            return
        per_cls: dict = {}
        lost: dict = {}
        for pick in self.picks:
            d_shadow = self._intended(pick, pick.shadow)
            shadow0 = pick.shadow
            pick.shadow = _rtn(shadow0 + d_shadow)
            pick.lost += float((pick.shadow - shadow0 - d_shadow).abs().sum())
            pick.mass += float(d_shadow.abs().sum())
            if not self.timed or pick.w0 is None:
                continue
            lr, _b1, _b2, _eps, wd = self._hp(pick.gi)
            w1 = _gather(pick.p, pick.idx).to(torch.float32)
            m = self.opt.master_at(pick.p, pick.idx)
            master1 = m.to(torch.float32) if (m is not None and pick.master0 is not None) else None
            delta = self._intended(pick, pick.w0)
            b = per_cls.setdefault(pick.cls, {"w0": [], "w1": [], "delta": [], "shadow0": [],
                                              "shadow1": [], "master0": [], "master1": [],
                                              "init": [], "wd0": [], "ulp0": [], "ulp1": []})
            b["w0"].append(pick.w0)
            b["w1"].append(w1)
            b["delta"].append(delta)
            b["shadow0"].append(shadow0)
            b["shadow1"].append(pick.shadow)
            b["master0"].append(pick.master0)
            b["master1"].append(master1)
            b["init"].append(pick.init)
            b["wd0"].append(-lr * wd * pick.w0)
            b["ulp0"].append(_ulp(pick.w0))
            b["ulp1"].append(_ulp(w1))
            acc = lost.setdefault(pick.cls, [0.0, 0.0])
            acc[0] += pick.lost
            acc[1] += pick.mass
            pick.lost = pick.mass = 0.0
            pick.w0 = pick.master0 = None
        if not self.timed:
            return
        classes = {}
        for cls, b in per_cls.items():
            fields = {}
            for k, parts in b.items():
                fields[k] = None if any(x is None for x in parts) else torch.cat(parts)
            s = Sample(**fields)
            row = {}
            for name, fn in METRICS:
                try:
                    v = fn(s)
                except Exception as exc:
                    v = f"error:{type(exc).__name__}"
                row[name] = v
            lo, ma = lost.get(cls, (0.0, 0.0))
            row["shadow_lost_mass"] = lo / ma if ma > 0 else None
            classes[cls] = row
        self.mark("probe")
        timing = {k: round(v * 1e3, 3) for k, v in self.times.items()}
        timing["total"] = round(sum(self.times.values()) * 1e3, 3)
        lr_now = float(self.opt.param_groups[0]["lr"])
        self._lr_trend(lr_now)
        if not self.first_done:
            self.first_done = True
            first = next((p for g_ in self.opt.param_groups for p in g_["params"]), None)
            memory_snapshot("after_first_step", phase=self.opt.phase,
                            device=(first.device if first is not None else None))
