import math
import os
import statistics
from typing import Optional

from pydantic import BaseModel, Field

import compute_affordance
import release_defaults

T_SEED = "t_seed"
T_LR = "t_lr"
T_DIV_BATCH = "t_div_batch"
T_DIV_LR = "t_div_lr"
T_PERTURB = "t_perturb"
T_BAG = "t_bag"
T_NEFT = "t_neft"
BORROW_SEED_KINDS = (T_PERTURB, "t_div_lr", T_BAG, T_NEFT)
SEED_PRIOR_KINDS = ("t_seed", T_PERTURB, "t_lr", T_BAG, T_NEFT)
RUNG_SEP = "@"


def div_kind(d: int) -> str:
    return f"{T_DIV_BATCH}{RUNG_SEP}{int(d)}"


def base_kind(kind: Optional[str]) -> Optional[str]:
    return str(kind).partition(RUNG_SEP)[0] if kind else kind


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


DIV_RUNGS = (2, 4, 8)
DIV_TARGET_UPDATES = int(_env_float("SN56_SB_DIV_TARGET_UPDATES", 400))
AT_TARGET_RUNG = int(_env_float("SN56_SB_AT_TARGET_RUNG", 4))
DIV_LR_ON = bool(int(_env_float("SN56_SB_DIV_LR", 1)))
G_MIN = _env_float("SN56_SB_G_MIN", 0.002)
G_IDLE = _env_float("SN56_SB_G_IDLE", 0.0005)
REJECT_RUN = int(_env_float("SN56_SB_REJECT_RUN", 3))
MAX_DIVERGED = int(_env_float("SN56_SB_MAX_DIVERGED", 2))
MAX_MEMBERS = int(_env_float("SN56_SB_MAX_MEMBERS", 999))
PERTURB_AFTER = int(_env_float("SN56_SB_PERTURB_AFTER", 3))
PRIOR_N = _env_float("SN56_SB_PRIOR_N", 2.0)
PRIOR_G_N = _env_float("SN56_SB_PRIOR_G_N", 1.0)
GAIN_WINDOW = int(_env_float("SN56_SB_GAIN_WINDOW", 2))
TAIL_FILL = bool(int(_env_float("SN56_SB_TAIL_FILL", 1)))
TAIL_SAFETY = _env_float("SN56_SB_TAIL_SAFETY", 1.1)
TAIL_STOPS = ("time", "quality")
CONTINUATION = bool(int(_env_float("SN56_SB_CONTINUATION", 1)))
CONTINUATION_KINDS: list = []
CONTINUATION_PRIORS: dict = {}
_ACCEPT_BAR_ENV = os.environ.get("SN56_SB_ACCEPT_BAR", "shadow").strip().lower()
ACCEPT_BAR = _ACCEPT_BAR_ENV not in ("0", "false", "no", "off")
ACCEPT_BAR_SHADOW = _ACCEPT_BAR_ENV == "shadow"
ACCEPT_ALPHA = _env_float("SN56_SB_ACCEPT_ALPHA", 0.5)
SHADOW_ALL_TRIALS = bool(int(_env_float("SN56_SB_BAR_SHADOW_ALL", 1)))
TIE_SEED = bool(int(_env_float("SN56_SB_TIE_SEED", 1)))
SEED_WHEN_ONE = bool(int(_env_float("SN56_SB_SEED_WHEN_ONE", 1)))
DLR_INBAND_PRIOR = bool(int(_env_float("SN56_SB_DLR_INBAND_PRIOR", 1)))

REJECT_RESCUE = bool(int(float(release_defaults.raw("SN56_SB_REJECT_RESCUE"))))
RESCUE_Z = _env_float("SN56_SB_RESCUE_Z", 2.0)
RESCUE_POOL_MIN = 2
RESCUE_IDLE_FRAC = _env_float("SN56_SB_RESCUE_IDLE_FRAC", 0.10)


def noise_level(raw_gain: Optional[float], se: Optional[float], z: Optional[float] = None) -> bool:
    z = RESCUE_Z if z is None else float(z)
    if raw_gain is None or se is None or not math.isfinite(float(se)) or float(se) <= 0:
        return False
    return float(raw_gain) >= -z * float(se)


def accept_z(n: int, alpha: Optional[float] = None) -> float:
    a = ACCEPT_ALPHA if alpha is None else float(alpha)
    n = max(1, int(n))
    an = min(0.5, a * 6.0 / (math.pi ** 2 * n * n))
    if an >= 0.5:
        return 0.0
    return float(statistics.NormalDist().inv_cdf(1.0 - an))


def paired_se(before: Optional[list], after: Optional[list]) -> Optional[float]:
    if not before or not after or len(before) != len(after):
        return None
    d = [float(b) - float(a) for b, a in zip(before, after)
         if b is not None and a is not None and math.isfinite(float(b)) and math.isfinite(float(a))]
    if len(d) < 2:
        return None
    return float(statistics.stdev(d) / math.sqrt(len(d)))


MULTI_GPU_ENV = "SN56_SB_MULTI_GPU"
MULTI_GPU = os.environ.get(MULTI_GPU_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


class MultiGpuGuard(BaseModel):
    on: bool
    world: int
    reason: str


def multi_gpu_guard(world: int, allow: Optional[bool] = None) -> MultiGpuGuard:
    world = max(1, int(world or 1))
    allow = MULTI_GPU if allow is None else bool(allow)
    if world <= 1:
        return MultiGpuGuard(on=False, world=world, reason="single_gpu")
    if allow:
        return MultiGpuGuard(on=False, world=world, reason=f"{MULTI_GPU_ENV}=1")
    return MultiGpuGuard(on=True, world=world, reason="multi_gpu_unvalidated")


class KindPrior(BaseModel):
    p_accept: float
    gain: float


PRIORS_LEGACY = {
    T_SEED: KindPrior(p_accept=0.59, gain=0.0028),
    T_DIV_BATCH: KindPrior(p_accept=0.54, gain=0.0065),
    T_PERTURB: KindPrior(p_accept=0.59, gain=0.0028),
    T_LR: KindPrior(p_accept=0.59, gain=0.0028),
    T_DIV_LR: KindPrior(p_accept=0.59, gain=0.0028),
    T_BAG: KindPrior(p_accept=0.59, gain=0.0028),
    T_NEFT: KindPrior(p_accept=0.59, gain=0.0028),
}
SOUP_BETA = 1.773
U_SEED = 0.0366
P_SEED = 0.84
R_A, R_B = 5.139, -0.924
P_DB = 0.86
R_DLR = 1.63
P_DLR = 0.45
PRIORS_REFIT = {
    T_SEED: KindPrior(p_accept=P_SEED, gain=U_SEED * 7 ** -SOUP_BETA),
    T_DIV_BATCH: KindPrior(p_accept=P_DB, gain=U_SEED * math.exp(R_A) * 234 ** R_B * 7 ** -SOUP_BETA),
    T_PERTURB: KindPrior(p_accept=P_SEED, gain=U_SEED * 7 ** -SOUP_BETA),
    T_LR: KindPrior(p_accept=0.59, gain=0.0028),
    T_DIV_LR: KindPrior(p_accept=P_DLR, gain=R_DLR * U_SEED * 7 ** -SOUP_BETA),
    T_BAG: KindPrior(p_accept=P_SEED, gain=U_SEED * 7 ** -SOUP_BETA),
    T_NEFT: KindPrior(p_accept=P_SEED, gain=U_SEED * 7 ** -SOUP_BETA),
}
PRIOR_REFIT = bool(int(_env_float("SN56_SB_PRIOR_REFIT", 1)))
PRIORS = PRIORS_REFIT if PRIOR_REFIT else PRIORS_LEGACY


def rows_d_ratio(a0_updates_per_epoch: int) -> float:
    upe = max(1, int(a0_updates_per_epoch or 0))
    return math.exp(R_A) * upe ** R_B


def prior_for(kind: Optional[str], priors: Optional[dict] = None) -> KindPrior:
    pr = PRIORS if priors is None else priors
    if kind in pr:
        return pr[kind]
    return pr.get(base_kind(kind), pr[T_SEED])


RATE_CHOICE = bool(int(_env_float("SN56_SB_RATE_CHOICE", 1)))
G_RATE = _env_float("SN56_SB_G_RATE", 0.0)


class KindStats(BaseModel):
    kind: str
    tried: int = 0
    accepted: int = 0
    gains: list = Field(default_factory=list)
    diverged: int = 0
    values: list = Field(default_factory=list)
    run: int = 0
    rejects: list = Field(default_factory=list)


def expected_gain(st: KindStats, prior: KindPrior, prior_n: float = PRIOR_N,
                  prior_g_n: float = PRIOR_G_N, window: int = GAIN_WINDOW) -> tuple:
    p = (prior.p_accept * prior_n + st.accepted) / (prior_n + st.tried)
    recent = [float(g) for g in st.gains[-max(1, int(window)):]] if st.gains else []
    g = (prior.gain * prior_g_n + sum(recent)) / (prior_g_n + len(recent))
    return p, g, p * g


def div_rung(a0_updates_per_epoch: int, rows: int, target: int = DIV_TARGET_UPDATES,
             rungs: tuple = DIV_RUNGS) -> int:
    upe = max(0, int(a0_updates_per_epoch or 0))
    rows = max(1, int(rows or 1))
    if upe >= target or rows < 2 or upe <= 0:
        return 1
    d = next((int(r) for r in rungs if upe * int(r) >= target), int(rungs[-1]))
    while d > 1 and d > rows:
        d //= 2
    return max(1, d)


def at_target_rung(rows: int, rung: Optional[int] = None, rungs: tuple = DIV_RUNGS) -> int:
    rows = max(1, int(rows or 1))
    d = int(AT_TARGET_RUNG if rung is None else (rung or 0))
    if d < 2 or d not in tuple(int(r) for r in rungs) or rows < 2:
        return 1
    while d > 1 and d > rows:
        d //= 2
    return max(1, d)


def rung_down(micro: int, accum: int, div: int) -> tuple:
    micro, accum = max(1, int(micro)), max(1, int(accum))
    target = max(1, (micro * accum) // max(1, int(div)))
    if target <= micro:
        return target, 1
    return micro, max(1, target // micro)


def rung_rows(micro: int, accum: int, div: int) -> int:
    m, a = rung_down(micro, accum, div)
    return int(m) * int(a)


class RungNote(BaseModel):
    d: int
    offered: bool
    reason: str
    rows: Optional[int] = None


def deeper_rung(d: int, micro: int, accum: int, rungs: tuple = DIV_RUNGS) -> RungNote:
    d, d2 = int(d), 2 * int(d)
    rows = max(1, int(micro)) * max(1, int(accum))
    if d <= 1:
        return RungNote(d=d2, offered=False, reason="no_batch_member")
    if d2 not in tuple(int(r) for r in rungs):
        return RungNote(d=d2, offered=False, reason="beyond_ladder")
    if d2 > rows:
        return RungNote(d=d2, offered=False, reason="rows_floor")
    r1, r2 = rung_rows(micro, accum, d), rung_rows(micro, accum, d2)
    if r2 >= r1:
        return RungNote(d=d2, offered=False, reason="not_distinct", rows=r2)
    return RungNote(d=d2, offered=True, reason="deeper_rung", rows=r2)


class Choice(BaseModel):
    kind: Optional[str] = None
    stop: Optional[str] = None
    ev: dict = Field(default_factory=dict)
    rate: dict = Field(default_factory=dict)
    continuation: Optional[dict] = None
    prices: dict = Field(default_factory=dict)
    need: Optional[float] = None
    left: Optional[float] = None
    skipped: dict = Field(default_factory=dict)
    bar: Optional[str] = None
    g_bar: Optional[float] = None
    cut: list = Field(default_factory=list)
    cut_lifted: bool = False
    rule: str = "bars"
    inputs: dict = Field(default_factory=dict)
    seed_when_one: bool = False
    rescue: Optional[dict] = None


class TailOption(BaseModel):
    horizon: int
    price: float
    fixed: float
    tail_fixed: Optional[float] = None
    min_steps: int = 1


class TailChoice(BaseModel):
    kind: Optional[str] = None
    steps: Optional[int] = None
    horizon: Optional[int] = None
    frac: Optional[float] = None
    price: Optional[float] = None
    ev: Optional[float] = None
    rate: Optional[float] = None
    spendable: Optional[float] = None
    g_bar: Optional[float] = None
    stop: Optional[str] = None
    options: dict = Field(default_factory=dict)
    cut: list = Field(default_factory=list)
    cut_lifted: bool = False


def tail_steps(opt: TailOption, spendable: float, safety: float = None) -> int:
    safety = TAIL_SAFETY if safety is None else float(safety)
    fixed_t = float(opt.fixed if opt.tail_fixed is None else opt.tail_fixed)
    var = max(0.0, float(opt.price) - float(opt.fixed)) * max(1.0, safety)
    room = float(spendable) - fixed_t
    if room <= 0 or opt.horizon <= 0:
        return 0
    if var <= 0:
        return int(opt.horizon)
    return max(0, int(math.floor(opt.horizon * room / var)))


def tail_price(opt: TailOption, steps: int, safety: float = None) -> float:
    safety = TAIL_SAFETY if safety is None else float(safety)
    fixed_t = float(opt.fixed if opt.tail_fixed is None else opt.tail_fixed)
    var = max(0.0, float(opt.price) - float(opt.fixed)) * max(1.0, safety)
    return fixed_t + var * float(steps) / float(max(1, opt.horizon))


class MemberPlanner(BaseModel):
    d: int = 1
    rungs: list = Field(default_factory=list)
    rung_notes: list[RungNote] = Field(default_factory=list)
    a0_updates_per_epoch: int = 0
    rotation: list = Field(default_factory=list)
    perturb: bool = False
    stats: dict = Field(default_factory=dict)
    turn: int = 0
    n_launched: int = 0
    reject_run: int = 0
    n_diverged: int = 0
    g_min: float = G_MIN
    g_idle: float = Field(default_factory=lambda: G_IDLE)
    last_bar: Optional[str] = None
    n_idle: int = 0
    n_tail: int = 0
    reject_limit: int = REJECT_RUN
    max_diverged: int = MAX_DIVERGED
    max_members: int = MAX_MEMBERS
    member_cap: Optional[int] = None
    online: bool = True
    generate: bool = True
    issued: dict = Field(default_factory=dict)
    n_fit: Optional[int] = None
    cut_div: bool = False
    cut_reason: str = ""
    rate_choice: bool = Field(default_factory=lambda: RATE_CHOICE)
    value_model: bool = Field(default_factory=lambda: PRIOR_REFIT)
    soup_n: int = 1
    explore: list = Field(default_factory=list)
    continuation_on: bool = Field(default_factory=lambda: CONTINUATION)
    cont: bool = False
    cont_excluded: list = Field(default_factory=list)
    unpriced_out: list = Field(default_factory=list)
    run_kinds: list = Field(default_factory=list)
    cont_n0: Optional[int] = None
    cont_left0: Optional[float] = None
    cont_kinds: list = Field(default_factory=list)

    p_dlr: Optional[float] = None
    rows_gate_upe: Optional[float] = None
    rescue_on: bool = Field(default_factory=lambda: REJECT_RESCUE)
    rescue: bool = False
    rescue_done: bool = False
    left_start: Optional[float] = None
    rescue_kinds: list = Field(default_factory=list)
    rescue_info: Optional[dict] = None
    trial_secs: float = 0.0

    @classmethod
    def build(cls, a0_updates_per_epoch: int, rows: int, perturb: bool = False,
              online: bool = True, member_cap: Optional[int] = None,
              micro: Optional[int] = None, accum: Optional[int] = None,
              generate: bool = True, div_lr: Optional[bool] = None,
              at_target: Optional[int] = None, n_fit: Optional[int] = None,
              extra_kinds: Optional[list] = None,
              rows_gate_upe: Optional[float] = None,
              p_dlr: Optional[float] = None) -> "MemberPlanner":
        _div_cut = compute_affordance.member_cuts(n_fit)[0]
        div_lr = DIV_LR_ON if div_lr is None else bool(div_lr)
        at_target = AT_TARGET_RUNG if at_target is None else int(at_target)
        rows = max(1, int(rows or 1))
        if not micro:
            micro, accum = rows, 1
        micro, accum = max(1, int(micro)), max(1, int(accum or 1))
        upe = int(a0_updates_per_epoch or 0)
        d = div_rung(upe, rows)
        notes, rungs = [], []
        if rows_gate_upe is not None and not (0 < upe < float(rows_gate_upe)):
            d = 1
            notes.append(RungNote(d=1, offered=False, reason="rows_gate_upe"))
        elif d > 1:
            rungs.append(d)
            notes.append(RungNote(d=d, offered=True, reason="div_rung",
                                  rows=rung_rows(micro, accum, d)))
            nxt = deeper_rung(d, micro, accum)
            notes.append(nxt)
            if nxt.offered:
                rungs.append(nxt.d)
        elif upe >= DIV_TARGET_UPDATES and at_target_rung(rows, at_target) > 1:
            d = at_target_rung(rows, at_target)
            rungs.append(d)
            notes.append(RungNote(d=d, offered=True, reason="offered_at_target",
                                  rows=rung_rows(micro, accum, d)))
        else:
            why = ("a0_updates_at_target" if upe >= DIV_TARGET_UPDATES
                   else "no_update_count" if upe <= 0 else "rows_floor")
            notes.append(RungNote(d=1, offered=False, reason=why))
        rot = [div_kind(r) for r in rungs] + ([T_DIV_LR] if div_lr else []) + [T_SEED]
        explore = []
        for k in (extra_kinds or []):
            if k and k not in rot:
                rot.append(str(k))
                explore.append(str(k))
        return cls(d=d, rungs=rungs, rung_notes=notes, a0_updates_per_epoch=upe, rotation=rot,
                   perturb=bool(perturb), online=bool(online), member_cap=member_cap,
                   generate=bool(generate), explore=explore,
                   n_fit=(int(n_fit) if n_fit is not None else None),
                   cut_div=bool(_div_cut.active and rungs), cut_reason=_div_cut.reason,
                   p_dlr=(float(p_dlr) if p_dlr is not None else None),
                   rows_gate_upe=(float(rows_gate_upe) if rows_gate_upe is not None else None))

    def _stats(self, kind: str) -> KindStats:
        if kind not in self.stats:
            self.stats[kind] = KindStats(kind=kind)
        st = self.stats[kind]
        if isinstance(st, dict):
            st = KindStats(**st)
            self.stats[kind] = st
        return st

    def _u_post(self, kind: str) -> tuple:
        st = self._stats(kind)
        b = base_kind(kind)
        if (b == T_DIV_LR and st.tried == 0 and self.p_dlr is not None
                and DLR_INBAND_PRIOR):
            seed_u = self._u_post(T_SEED)[1]
            pp, pu = float(self.p_dlr), R_DLR * seed_u
            p = (pp * PRIOR_N + st.accepted) / (PRIOR_N + st.tried)
            return p, pu, pp, pu
        if b in BORROW_SEED_KINDS and st.tried == 0:
            return self._u_post(T_SEED)
        if b in SEED_PRIOR_KINDS:
            pp, pu = P_SEED, U_SEED
        elif b not in (T_DIV_BATCH, T_DIV_LR):
            pr = CONTINUATION_PRIORS.get(kind) or CONTINUATION_PRIORS.get(b)
            pp, pu = ((float(pr.p_accept), float(pr.gain)) if pr is not None else (P_SEED, U_SEED))
        else:
            seed_u = self._u_post(T_SEED)[1]
            if b == T_DIV_BATCH:
                pp, pu = P_DB, rows_d_ratio(self.a0_updates_per_epoch) * seed_u
            else:
                pp = P_DLR if self.p_dlr is None else float(self.p_dlr)
                pu = R_DLR * seed_u
        p = (pp * PRIOR_N + st.accepted) / (PRIOR_N + st.tried)
        recent = [float(v) for v in st.values[-max(1, int(GAIN_WINDOW)):]]
        u = (pu * PRIOR_G_N + sum(recent)) / (PRIOR_G_N + len(recent))
        return p, u, pp, pu

    def ev(self, kind: str) -> float:
        st = self._stats(kind)
        if self.value_model:
            p, u, _, _ = self._u_post(kind)
            return p * u * float(self.soup_n + 1) ** -SOUP_BETA
        if kind in BORROW_SEED_KINDS and st.tried == 0:
            return self.ev(T_SEED)
        return expected_gain(st, prior_for(kind))[2]

    def ev_inputs(self, kind: str) -> dict:
        st = self._stats(kind)
        if self.value_model:
            p, u, pp, pu = self._u_post(kind)
            return {"model": "value", "prior_p": round(pp, 4), "prior_u": round(pu, 6),
                    "tried": st.tried, "accepted": st.accepted,
                    "recent_u": [round(float(x), 6) for x in st.values[-max(1, int(GAIN_WINDOW)):]],
                    "p": round(p, 4), "u": round(u, 6), "soup_n": int(self.soup_n),
                    "beta": SOUP_BETA, "ev": round(self.ev(kind), 6),
                    "r_upe": (round(rows_d_ratio(self.a0_updates_per_epoch), 4)
                              if base_kind(kind) == T_DIV_BATCH else None)}
        if kind in BORROW_SEED_KINDS and st.tried == 0:
            d = self.ev_inputs(T_SEED)
            d["borrowed"] = T_SEED
            return d
        pr = prior_for(kind)
        p, g, e = expected_gain(st, pr)
        return {"prior_p": pr.p_accept, "prior_g": pr.gain, "tried": st.tried,
                "accepted": st.accepted, "recent": [round(float(x), 6) for x in
                                                    st.gains[-max(1, int(GAIN_WINDOW)):]],
                "p": round(p, 4), "g": round(g, 6), "ev": round(e, 6), "borrowed": None}

    def active_rotation(self) -> list:
        rot = list(self.rotation)
        if self.perturb and self.n_launched >= PERTURB_AFTER and T_PERTURB not in rot:
            rot.append(T_PERTURB)
        if self.cont:
            rot = [k for k in rot if k not in self.cont_excluded and k not in self.unpriced_out]
            rot += [k for k in list(self.cont_kinds) + list(CONTINUATION_KINDS)
                    if k not in rot and k not in self.cont_excluded and k not in self.unpriced_out]
        return rot

    def kinds_in_order(self) -> list:
        rot = self.active_rotation()
        if not rot:
            return []
        i = self.turn % len(rot)
        return rot[i:] + rot[:i]

    def record(self, kind: str, accepted: Optional[bool], gain: Optional[float] = None,
               diverged: bool = False, soup_n: Optional[int] = None,
               bar_blocked: bool = False, raw_gain: Optional[float] = None,
               se: Optional[float] = None) -> None:
        st = self._stats(kind)
        if accepted:
            self.soup_n = int(soup_n) if soup_n else int(self.soup_n) + 1
            st.values.append(max(0.0, float(gain or 0.0)) * float(self.soup_n) ** SOUP_BETA)
        elif soup_n:
            self.soup_n = int(soup_n)
        self.n_launched += 1
        if diverged:
            st.diverged += 1
            st.tried += 1
            self.n_diverged += 1
            self.reject_run += 1
            self.run_kinds.append(kind)
            st.run += 1
            return
        if accepted is None:
            return
        st.tried += 1
        if accepted:
            st.accepted += 1
            st.gains.append(max(0.0, float(gain or 0.0)))
            self.reject_run = 0
            self.run_kinds = []
            st.run = 0
        else:
            self.reject_run += 1
            self.run_kinds.append(kind)
            st.rejects.append([None if raw_gain is None else float(raw_gain),
                               None if se is None else float(se)])
            if not bar_blocked:
                st.run += 1

    def stop_reason(self) -> Optional[str]:
        if not self.generate:
            return "no_online_soup"
        if self.n_diverged >= self.max_diverged:
            return "diverged"
        if self.online and self.reject_run >= self.reject_limit:
            return "rejects"
        if self.n_launched >= self.max_members:
            return "max_members"
        if self.member_cap is not None and self.n_launched >= self.member_cap:
            return "pool_ram_cap"
        return None

    def enter_continuation(self, left: Optional[float], prices: Optional[dict] = None) -> Optional[str]:
        for k in list(self.stats):
            if self._stats(k).run >= self.reject_limit and k not in self.cont_excluded:
                self.cont_excluded.append(k)
        if not self.cont:
            self.cont, self.cont_n0 = True, int(self.n_launched)
            self.cont_left0 = round(float(left), 1) if left is not None else None
            _first = [T_DIV_LR] if T_DIV_LR in self.active_rotation() else []
            for k in _first + list(self.cont_kinds) + list(CONTINUATION_KINDS):
                if k not in self.explore and self._stats(k).tried == 0:
                    self.explore.append(k)
        if prices is not None:
            live = self.active_rotation()
            if live and all(prices.get(k) is None for k in live):
                self.unpriced_out.extend(k for k in live if k not in self.unpriced_out)
        self.reject_run, self.run_kinds = 0, []
        if not self.active_rotation():
            self.reject_run = self.reject_limit
            return "rejects"
        return self.stop_reason()

    def run_se(self) -> Optional[float]:
        ses = [float(r[1]) for k in list(self.stats) for r in self._stats(k).rejects
               if r[1] is not None and math.isfinite(float(r[1])) and float(r[1]) > 0]
        return float(statistics.median(ses)) if ses else None

    def kind_noise(self, kind: str) -> dict:
        med = self.run_se()
        best = None
        for g, se in self._stats(kind).rejects:
            if g is None:
                continue
            se_u = se if (se is not None and se > 0) else med
            if se_u is None or se_u <= 0:
                continue
            zb = float(g) / float(se_u)
            if best is None or zb > best[0]:
                best = (zb, float(g), float(se_u))
        if best is None:
            return {"kind": kind, "noise": False, "z_best": None, "why": "no_measured_reject"}
        return {"kind": kind, "noise": bool(best[0] >= -RESCUE_Z), "z_best": round(best[0], 3),
                "raw_gain": round(best[1], 7), "se": round(best[2], 7)}

    def rescue_decision(self, prices: dict, left: Optional[float], tail: float) -> dict:
        info: dict = {"action": "stop", "soup_n": int(self.soup_n),
                      "left": (round(float(left), 1) if left is not None else None),
                      "tail": round(float(tail), 1), "trial_secs": round(float(self.trial_secs), 2),
                      "z": RESCUE_Z, "pool_min": RESCUE_POOL_MIN, "n_launched": int(self.n_launched),
                      "run_se": (round(self.run_se(), 7) if self.run_se() is not None else None)}
        self.rescue_done = True
        self.rescue_info = info
        kinds = [self.kind_noise(k) for k in list(self.cont_excluded)]
        info["kinds"] = kinds
        noise = [d["kind"] for d in kinds if d["noise"]]
        if not noise:
            info["why"] = "decisive"
            return info
        priced = {k: float(prices[k]) for k in noise
                  if prices.get(k) is not None and float(prices[k]) > 0}
        if not priced:
            info["why"] = "unpriced"
            return info
        price = min(priced.values())
        unit = price + float(self.trial_secs)
        n_need = max(RESCUE_POOL_MIN, int(self.soup_n))
        need = n_need * unit + float(tail)
        info.update(price=round(price, 1), need=round(need, 1), n_need=n_need,
                    n_affordable=(int(max(0.0, float(left) - float(tail)) // unit)
                                  if left is not None and unit > 0 else None))
        if left is None or float(left) < need:
            info["why"] = "budget"
            return info
        for k in noise:
            if k in self.cont_excluded:
                self.cont_excluded.remove(k)
            self._stats(k).run = 0
            if (k not in self.rotation and k not in self.cont_kinds
                    and k not in CONTINUATION_KINDS):
                self.rotation.append(k)
        self.rescue, self.rescue_kinds = True, list(noise)
        self.reject_run, self.run_kinds = 0, []
        info.update(action="readmit", readmitted=list(noise), why="noise_level")
        self.rescue_info = info
        return info

    def _apply_cut(self, eligible: list, out: Choice, skipped: dict) -> list:
        keep, cut, lifted = compute_affordance.div_kinds_cut(
            self.kinds_in_order(), eligible, self.cut_div,
            lambda k: base_kind(k) == T_DIV_BATCH)
        for k in cut:
            skipped[k] = "cut_low_affordance"
        out.cut = sorted(set(out.cut) | set(cut))
        out.cut_lifted = bool(out.cut_lifted or lifted)
        return keep

    def rescue_rearm(self, left: Optional[float], tail: float) -> bool:
        if not self.rescue_done or left is None or self.left_start is None:
            return False
        if float(left) <= RESCUE_IDLE_FRAC * float(self.left_start) + float(tail):
            return False
        self.rescue_done = False
        self.unpriced_out = []
        return True

    def choose(self, prices: dict, left: Optional[float], tail: float) -> Choice:
        out = Choice(left=left)
        if self.left_start is None and left is not None:
            self.left_start = float(left)
        stop = self.stop_reason()
        if self.continuation_on and self.online and (stop == "rejects" or self.cont):
            stop = self.enter_continuation(left, prices)
            if (stop == "rejects" and self.rescue_on and not self.active_rotation()
                    and (not self.rescue_done or self.rescue_rearm(left, tail))):
                out.rescue = self.rescue_decision(prices, left, tail)
                if out.rescue["action"] == "readmit":
                    stop = self.stop_reason()
            elif stop == "rejects" and self.rescue_done and out.rescue is None:
                out.rescue = {"action": "final", "why": "rescue_spent",
                              "readmitted": list(self.rescue_kinds)}
            if stop == "rejects" and self.unpriced_out:
                stop = "exhausted"
        if self.cont:
            out.continuation = {"excluded": list(self.cont_excluded), "n0": self.cont_n0,
                                "left0": self.cont_left0, "kinds": self.active_rotation()}
        order = self.kinds_in_order()
        for k in order:
            out.ev[k] = round(self.ev(k), 6)
            if k in prices and prices[k] is not None:
                out.prices[k] = round(float(prices[k]), 1)
                if float(prices[k]) > 0:
                    out.rate[k] = round(self.ev(k) / float(prices[k]), 9)
        if stop:
            out.stop = stop
            out.bar = self.last_bar
            out.g_bar = (G_RATE if self.last_bar == "rate" else self.g_idle if self.last_bar == "idle"
                         else self.g_min if self.last_bar else None)
            return out
        def _fits(kinds: list, skipped: dict) -> list:
            got = []
            for k in kinds:
                price = prices.get(k)
                if price is None:
                    skipped[k] = "unpriced"
                    continue
                need = float(price) + float(tail)
                if left is not None and left < need:
                    skipped[k] = "time"
                    continue
                got.append((k, need, float(price)))
            return got

        if self.online:
            for k in order:
                out.inputs[k] = self.ev_inputs(k)
        if self.online and self.rate_choice:
            out.rule, out.bar, out.g_bar = "rate", "rate", G_RATE
            eligible = [k for k in order if self.ev(k) > G_RATE]
            for k in order:
                if k not in eligible:
                    out.skipped[k] = "quality"
            eligible = self._apply_cut(eligible, out, out.skipped)
            if not eligible:
                out.stop = "quality"
                return out
            fits = _fits(eligible, out.skipped)
            if not fits:
                out.stop = "time"
                return out
            return self._issue(out, fits)
        if self.online:
            out.bar, out.g_bar = "quality", self.g_min
        eligible = [k for k in order if (not self.online) or self.ev(k) >= self.g_min]
        for k in order:
            if k not in eligible:
                out.skipped[k] = "quality"
        eligible = self._apply_cut(eligible, out, out.skipped)
        if not eligible:
            idle = [k for k in order if self.ev(k) >= self.g_idle]
            sk: dict = {k: "quality" for k in order if k not in idle}
            idle = self._apply_cut(idle, out, sk)
            fits = _fits(idle, sk) if self.online else []
            if not fits:
                out.stop = "quality"
                return out
            out.bar, out.g_bar, out.skipped = "idle", self.g_idle, sk
        else:
            fits = _fits(eligible, out.skipped)
            if not fits and TAIL_FILL:
                idle = [k for k in order if k not in eligible and k not in out.cut
                        and self.ev(k) >= self.g_idle]
                sk = dict(out.skipped)
                if idle:
                    _kept = set(self._apply_cut(eligible + idle, out, sk))
                    idle = [k for k in idle if k in _kept]
                ifits = _fits(idle, sk)
                if ifits:
                    for k, _, _ in ifits:
                        sk.pop(k, None)
                    fits = ifits
                    out.bar, out.g_bar, out.skipped = "idle", self.g_idle, sk
        if not fits:
            out.stop = "time"
            return out
        return self._issue(out, fits)

    def _issue(self, out: Choice, fits: list) -> Choice:
        if self.online:
            best = min(range(len(fits)),
                       key=lambda i: (-(self.ev(fits[i][0]) / max(fits[i][2], 1e-9)),
                                      (0 if (TIE_SEED and fits[i][0] == T_SEED) else 1), i))
            untried = [i for i, (k, _, _) in enumerate(fits)
                       if k in self.explore and self._stats(k).tried == 0]
            if untried:
                best = untried[0]
        else:
            best = 0
        if SEED_WHEN_ONE and self.n_fit is not None and int(self.n_fit) <= 1:
            seed = [i for i, (k, _, _) in enumerate(fits) if k == T_SEED]
            if seed and seed[0] != best:
                best, out.seed_when_one = seed[0], True
        k, need, _ = fits[best]
        out.kind, out.need = k, round(need, 1)
        rot = self.active_rotation()
        self.rotation = rot
        self.turn = (rot.index(k) + 1) % len(rot)
        self.issued[k] = int(self.issued.get(k, 0)) + 1
        self.last_bar = out.bar
        if out.bar == "idle":
            self.n_idle += 1
        return out

    def tail_fill(self, stop: Optional[str], options: dict, spendable: Optional[float],
                  g_bar: Optional[float] = None) -> TailChoice:
        bar = self.g_idle if g_bar is None else float(g_bar)
        out = TailChoice(spendable=(round(float(spendable), 1) if spendable is not None
                                    else None), g_bar=bar)
        if not TAIL_FILL:
            out.stop = "off"
            return out
        if stop not in TAIL_STOPS:
            out.stop = f"stop:{stop}"
            return out
        if not self.online or not self.generate:
            out.stop = "no_online_soup"
            return out
        if spendable is None or spendable <= 0:
            out.stop = "no_spendable"
            return out
        best = None
        _order = self.kinds_in_order()
        _kept, _cut, _lifted = compute_affordance.div_kinds_cut(
            _order, [k for k in _order if options.get(k) is not None], self.cut_div,
            lambda k: base_kind(k) == T_DIV_BATCH)
        out.cut, out.cut_lifted = sorted(_cut), bool(_lifted)
        for i, k in enumerate(_order):
            opt = options.get(k)
            if opt is None:
                out.options[k] = {"skip": "unpriced"}
                continue
            if k in _cut:
                out.options[k] = {"skip": "cut_low_affordance"}
                continue
            if isinstance(opt, dict):
                opt = TailOption(**opt)
            ev_full = self.ev(k)
            steps = min(tail_steps(opt, spendable), int(opt.horizon) - 1)
            rec = {"horizon": int(opt.horizon), "min_steps": int(opt.min_steps),
                   "steps": int(max(0, steps)), "ev_full": round(ev_full, 6),
                   "price_full": round(float(opt.price), 1)}
            if steps < max(1, int(opt.min_steps)):
                rec["skip"] = "min_steps"
                out.options[k] = rec
                continue
            frac = steps / float(opt.horizon)
            ev = ev_full * frac
            price = tail_price(opt, steps)
            rec.update(frac=round(frac, 4), ev=round(ev, 6), price=round(price, 1),
                       rate=round(ev / max(price, 1e-9), 9))
            if ev < bar:
                rec["skip"] = "quality"
                out.options[k] = rec
                continue
            out.options[k] = rec
            key = (-(ev / max(price, 1e-9)), (0 if (TIE_SEED and k == T_SEED) else 1), i)
            if best is None or key < best[0]:
                best = (key, k, steps, opt, frac, ev, price)
        if best is None:
            out.stop = "none_fits"
            return out
        _, k, steps, opt, frac, ev, price = best
        out.kind, out.steps, out.horizon = k, int(steps), int(opt.horizon)
        out.frac, out.ev = round(frac, 4), round(ev, 6)
        out.price, out.rate = round(price, 1), round(ev / max(price, 1e-9), 9)
        out.stop = ""
        rot = self.active_rotation()
        self.rotation = rot
        if k in rot:
            self.turn = (rot.index(k) + 1) % len(rot)
        self.issued[k] = int(self.issued.get(k, 0)) + 1
        self.n_tail += 1
        return out
