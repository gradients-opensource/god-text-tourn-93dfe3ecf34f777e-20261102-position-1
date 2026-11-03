from __future__ import annotations

import os
from typing import Callable, Sequence

from pydantic import BaseModel

import spare_budget as sb

_FALSY = ("0", "false", "no", "off")


def _flag(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in _FALSY


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def price_on() -> bool:
    return _flag("SN56_MP_PRICE")


def launch_on() -> bool:
    return _flag("SN56_MP_LAUNCH")


def stop_on() -> bool:
    return _flag("SN56_MP_STOP")


def stop_v2_on() -> bool:
    return _flag("SN56_MP_STOP_V2")


def late_probe_on() -> bool:
    return _flag("SN56_MP_LATE_PROBE")


def anyrel_on() -> bool:
    return _flag("SN56_MP_STOP_ANYREL")


def noturn_on() -> bool:
    return os.environ.get("SN56_MP_STOP_NOTURN", "0").strip().lower() not in _FALSY


def loop_pool_mix_on() -> bool:
    return _flag("SN56_MP_STOP_POOLMIX")


def launch_pool_on() -> bool:
    return _flag("SN56_MP_LAUNCH_POOL")


LEGACY_SAFETY = 1.1
PRICE_B = 0.01
PRICE_A = 18.4
PRE_C = 9.54
PRE_K = 5.0
PRE2_C = 9.54
PRE2_K_POOL = 0.5
SOUP_EVALS_PER_ENTRY = 2.0
P_RECOVER_UB = 3.0 / 1705.0
STOP_ERR_PER_WRITE_Q21 = (-22.36, -14.2, -12.38, -10.83, -5.28, -4.69, -3.83, -2.86, -0.48, -0.33,
                          0.16, 1.48, 1.95, 2.33, 2.86, 4.85, 5.79, 6.5, 7.37, 18.99, 24.19)
STOP_ERR_A = 9.88
STOP_ERR_B = 6.045
STOP_ERR_RESID_Q21 = (-81.0, -51.2, -29.1, -25.8, -21.8, -19.2, -17.8, -13.1, -9.7, -3.7, 0.0, 8.3,
                      24.8, 37.4, 45.8, 52.2, 60.0, 80.4, 111.5, 183.7, 421.2)
MEMBER_A = 0.025056
MEMBER_BETA = 1.4564
class RiseClass(BaseModel):
    v_late_per_eval: float
    rec: float
    rec_ub: float = 0.0


RISE_CLASSES = {
    (1, True, 1): RiseClass(v_late_per_eval=1.50244e-3, rec=2.40717e-2),
    (1, True, 2): RiseClass(v_late_per_eval=1.083e-5, rec=0.0, rec_ub=3.0 / 34),
    (1, False, 1): RiseClass(v_late_per_eval=6.1749e-4, rec=3.71796e-2),
    (1, False, 2): RiseClass(v_late_per_eval=7.97e-5, rec=8.118e-4),
    (5, True, 1): RiseClass(v_late_per_eval=1.1434e-3, rec=6.1e-6, rec_ub=4.74 / 450),
    (5, True, 2): RiseClass(v_late_per_eval=1.616e-5, rec=0.0, rec_ub=3.0 / 357),
    (5, False, 1): RiseClass(v_late_per_eval=6.1534e-4, rec=5.095e-3),
    (5, False, 2): RiseClass(v_late_per_eval=2.824e-5, rec=1.1937e-3),
}
LOOP_POOL_MIX = ((315.0 / 502.0, 2), (183.0 / 502.0, 3), (4.0 / 502.0, 4))
REL_FLOOR = 0.01


def rise_key(rel: float, turn: bool, turn_epoch: int | None) -> tuple:
    band = 5 if float(rel) >= 0.05 else 1
    k = 1 if turn_epoch is None or int(turn_epoch) <= 1 else 2
    return band, bool(turn), k


def stop_errs(write_secs: float) -> tuple:
    w = max(0.0, float(write_secs or 0.0))
    if stop_v2_on():
        return tuple(STOP_ERR_A + STOP_ERR_B * w + r for r in STOP_ERR_RESID_Q21)
    return tuple(q * w for q in STOP_ERR_PER_WRITE_Q21) if w > 0 else (0.0,)


def price_b() -> float:
    return _num("SN56_MP_PRICE_B", PRICE_B)


def price_a() -> float:
    return _num("SN56_MP_PRICE_A", PRICE_A)


def priced_core(core: float, safety: float = LEGACY_SAFETY) -> float:
    core = max(0.0, float(core or 0.0))
    old = float(safety) * core
    if not price_on():
        return old
    return min(old, (1.0 + price_b()) * core + price_a())


def prelaunch_reserve(write_secs: float, pool_size: int | None = None) -> float:
    if not launch_on():
        return 0.0
    w = max(0.0, float(write_secs or 0.0))
    if launch_pool_on() and pool_size is not None:
        return (_num("SN56_MP_PRE2_C", PRE2_C) + w
                + _num("SN56_MP_PRE2_K", PRE2_K_POOL) * max(0, int(pool_size)) * w)
    return _num("SN56_MP_PRE_C", PRE_C) + _num("SN56_MP_PRE_K", PRE_K) * w


def launch_tail(n_keep: int, t_eval: float, soup_over: float, write_secs: float) -> float:
    if not launch_on():
        return 0.0
    unit = max(0.0, float(t_eval or 0.0)) + max(0.0, float(soup_over or 0.0))
    return (max(0, int(n_keep)) + 4) * unit + 2.0 * max(0.0, float(write_secs or 0.0))


def a0_soup_secs(pool_size: int, t_eval: float) -> float:
    n = int(pool_size or 0)
    if n <= 1:
        return 0.0
    per = SOUP_EVALS_PER_ENTRY if stop_on() else 1.0
    return n * per * max(0.0, float(t_eval or 0.0))


class PivotalValue(BaseModel):
    n_now: int
    n_p3: int
    ev_now: float
    ev_p3: float
    ev_freed: float
    ev_continue: float
    stop: bool


def member_ev(position: int) -> float:
    return float(sb.P_SEED) * float(sb.U_SEED) * float(max(2, int(position))) ** (-float(sb.SOUP_BETA))


def members_ev(n: int, ncap: int) -> float:
    return sum(member_ev(k + 1) for k in range(1, min(int(n), int(ncap)) + 1))


def ev_continue() -> float:
    return P_RECOVER_UB * member_ev(2)


def pivotal(n_of: Callable[[float], int], r_now: float, r_p3: float, ncap: int,
            errs: Sequence[float] = (0.0,)) -> PivotalValue:
    errs = tuple(errs) or (0.0,)
    ev_now = sum(members_ev(n_of(r_now - e), ncap) for e in errs) / len(errs)
    ev_p3 = sum(members_ev(n_of(r_p3 - e), ncap) for e in errs) / len(errs)
    evc = ev_continue()
    return PivotalValue(n_now=int(n_of(r_now)), n_p3=int(n_of(r_p3)), ev_now=round(ev_now, 6),
                        ev_p3=round(ev_p3, 6), ev_freed=round(ev_now - ev_p3, 6),
                        ev_continue=round(evc, 7), stop=bool(ev_now - ev_p3 > evc))


def member_ev_v2(position: int) -> float:
    return float(MEMBER_A) * float(max(2, int(position))) ** (-float(MEMBER_BETA))


def members_ev_v2(n: int, ncap: int) -> float:
    return sum(member_ev_v2(k + 1) for k in range(1, min(int(n), int(ncap)) + 1))


def v_late(rel: float, turn: bool, turn_epoch: int | None, h: int) -> float:
    return float(RISE_CLASSES[rise_key(rel, turn, turn_epoch)].v_late_per_eval) * max(0, int(h))


def ev_continue_v2(rel: float, turn: bool, turn_epoch: int | None) -> float:
    c = RISE_CLASSES[rise_key(rel, turn, turn_epoch)]
    return max(float(c.rec), float(c.rec_ub) * member_ev_v2(2))


class PivotalValueV2(BaseModel):
    n_now: int
    n_p3: int
    p_extra: float
    ev_now: float
    ev_p3: float
    ev_freed: float
    v_late: float
    ev_continue: float
    turn_epoch: int | None
    rise: str
    h: int
    stop: bool


def pivotal_v2(n_of: Callable[..., int], r_now: float, r_p3: float, ncap: int,
               errs: Sequence[float], turn_epoch: int | None, h: int, rel: float = 0.05,
               turn: bool = True, scenarios: Sequence = ((1.0, None),)) -> PivotalValueV2:
    errs = tuple(errs) or (0.0,)
    m = len(errs)
    scen = [(float(w), sc) for w, sc in (scenarios or ((1.0, None),))]
    wsum = sum(w for w, _ in scen) or 1.0
    ev_now = ev_p3 = p_extra = 0.0
    for w, sc in scen:
        f = (lambda r: n_of(r)) if sc is None else (lambda r, sc=sc: n_of(r, sc))
        now = [f(r_now - e) for e in errs]
        p3 = [f(r_p3 - e) for e in errs]
        ev_now += w / wsum * sum(members_ev_v2(n, ncap) for n in now) / m
        ev_p3 += w / wsum * sum(members_ev_v2(n, ncap) for n in p3) / m
        p_extra += w / wsum * sum(1 for a, b in zip(now, p3) if min(a, ncap) > min(b, ncap)) / m
    vl = v_late(rel, turn, turn_epoch, h)
    evc = ev_continue_v2(rel, turn, turn_epoch)
    rk = rise_key(rel, turn, turn_epoch)
    freed = ev_now - ev_p3
    return PivotalValueV2(n_now=int(n_of(r_now)), n_p3=int(n_of(r_p3)), p_extra=round(p_extra, 4),
                          ev_now=round(ev_now, 7), ev_p3=round(ev_p3, 7), ev_freed=round(freed, 7),
                          v_late=round(vl, 7), ev_continue=round(evc, 7),
                          turn_epoch=(int(turn_epoch) if turn_epoch is not None else None),
                          rise="rel%d_%s_k%d" % (rk[0], "turn" if rk[1] else "noturn", rk[2]),
                          h=int(h), stop=bool(freed > vl + evc))


def v_late_probe(gain: float, h: int, pool_n: int) -> float:
    g = max(0.0, float(gain or 0.0))
    pos = max(2, int(pool_n or 0) + 1)
    return g * (pos / 2.0) ** (-float(MEMBER_BETA)) * max(0, int(h))


def probe_cost_est(t_eval: float, write_secs: float) -> float:
    return max(0.0, float(t_eval or 0.0)) + 2.0 * max(0.0, float(write_secs or 0.0))
