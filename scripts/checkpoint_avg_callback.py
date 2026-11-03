import save_guard as sg
import run_state
"""
Adaptive training callback: checkpoint averaging + overfitting rollback.

Combines three mechanisms:
1. Sliding-window checkpoint averaging (last 3 eval-point snapshots)
2. Overfitting detection (eval loss >5% above best for 2 consecutive evals)
3. Rollback + escalating response (restore best weights, cut LR, bump NEFTune)

On overfitting detection:
- Restores model to best checkpoint weights
- Cuts LR by 50%
- Bumps NEFTune noise alpha (5 -> 10 -> 15) if NEFTune is active
- Continues training with more regularization

The model gets a second chance from the best point with gentler settings,
instead of wasting remaining time on a diverging trajectory.
"""

import dataclasses
import datetime
import gc
import inspect
import os
import shutil
import statistics
import time
from collections import deque
from typing import Callable, Optional

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import is_main_process

import async_save
import eval_squeeze
import free_candidates as fc
import gpu_soup
import host_snapshot
import member_pricing as mp
import planner_clock
import precise_update
import packed_fill
import second_trajectory as traj2
import stop_to_decay as std
import train_step_clock as tsc

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

_CGROUP_MEMORY_FILES = (
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
     "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
)
_CGROUP_UNLIMITED = 1 << 62

_OVERFIT_THRESHOLD = 0.05
_OVERFIT_PATIENCE = 3
_MAX_ROLLBACKS = 2

_STOP_RULE_ENV = "TURN_STOP_RULE"
_TURN_TRAIN_DROP = 0.1
_TURN_DEV_MARGIN = 0.05
_TURN_WINDOW = 50
_TURN_MIN_LOGS = 5
_STOP_RULE_NCAP = 4
_FULL_STRIDE_FRAC = 0.8

_REL_ENV = "SN56_TURN_RELATIVE"
_REL_LR_MAX = 0.5
_REL_DIP_FRAC = 0.25
_REL_POOL_MIN = 8


def turn_relative_enabled() -> bool:
    return os.getenv(_REL_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def boundary_dip(evals, best_step: int, best_loss: float, steps_per_epoch: int,
                 frac: float = _REL_DIP_FRAC) -> Optional[float]:
    spe = int(steps_per_epoch or 0)
    if spe <= 0 or best_step is None or best_step < 1:
        return None
    b = (int((best_step - 1) // spe) + 1) * spe
    before = [(s, l) for s, l in evals if s < b and s != best_step]
    if not before:
        return None
    win = [l for s, l in before if s >= b - frac * spe]
    ref = max(win) if win else max(before, key=lambda x: x[0])[1]
    return max(0.0, float(ref) - float(best_loss))


def turned_relative(step: int, best_step: int, steps_per_epoch: int, stride: int,
                    lr_frac: Optional[float], dev: float, pool_losses, in_pool: bool,
                    dip: Optional[float], pool_min: int = _REL_POOL_MIN) -> tuple:
    spe = int(steps_per_epoch or 0)
    if spe <= 0 or int(stride or 0) <= 0 or best_step is None or best_step < 1:
        return False, "no_cadence"
    b = (int((best_step - 1) // spe) + 1) * spe
    if step < b + int(stride):
        return False, "within_stride_of_boundary"
    if lr_frac is None:
        return False, "no_lr"
    if lr_frac >= _REL_LR_MAX:
        return False, "lr_high"
    pool_losses = [float(x) for x in pool_losses if x is not None and x == x]
    if len(pool_losses) < pool_min:
        return False, "pool_not_full"
    if in_pool:
        return False, "entered_pool"
    if dip is None:
        return False, "no_dip_reference"
    if dev - dip <= max(pool_losses):
        return False, "boundary_dip_could_reenter"
    return True, "turned"


def relative_trade(n_now: int, n_end: int) -> bool:
    return min(int(n_now), _STOP_RULE_NCAP) > min(int(n_end), _STOP_RULE_NCAP)


def stop_rule_mode() -> str:
    v = os.getenv(_STOP_RULE_ENV, "vot").strip().lower()
    return "patience" if v in ("patience", "0", "off", "false", "no") else "vot"


def turn_signature(log_history, best_step: int, step: int,
                   steps_per_epoch: Optional[int]) -> dict:
    out = {"ctx": "mid", "boundary_step": None, "train_drop": None,
           "n_window": 0, "turn": False}
    spe = int(steps_per_epoch or 0)
    if spe <= 0 or best_step is None or best_step < 1:
        return out
    bstep = (int((best_step - 1) // spe) + 1) * spe
    out["boundary_step"] = bstep
    if step <= bstep:
        return out
    out["ctx"] = "post"
    pre: list[float] = []
    post: list[float] = []
    for e in log_history or ():
        if "eval_loss" in e or "loss" not in e:
            continue
        s, loss = e.get("step"), e.get("loss")
        if s is None or loss is None:
            continue
        try:
            loss = float(loss)
        except (TypeError, ValueError):
            continue
        if loss != loss:
            continue
        if s <= bstep:
            pre.append(loss)
        elif s <= step:
            post.append(loss)
    pre = pre[-_TURN_WINDOW:]
    post = post[:_TURN_WINDOW]
    n = min(len(pre), len(post), _TURN_WINDOW)
    out["n_window"] = n
    if n < _TURN_MIN_LOGS:
        return out
    drop = statistics.median(pre[-n:]) - statistics.median(post[:n])
    out["train_drop"] = float(drop)
    if drop > _TURN_TRAIN_DROP:
        out["ctx"], out["turn"] = "turn", True
    return out


def stop_rule_decide(rel: float, turn: bool, n_now: int, n_p3: int) -> tuple:
    if not (turn and rel >= _TURN_DEV_MARGIN):
        return False, "no_turn_signature"
    if min(n_now, _STOP_RULE_NCAP) > min(n_p3, _STOP_RULE_NCAP):
        return True, "turn_time_stop"
    return False, "not_pivotal"


_NEFTUNE_LEVELS = [5, 10, 15]


class AdaptiveTrainingCallback(TrainerCallback):
    

    def __init__(
        self,
        window: int = 3,
        device: str = "cpu",
        use_reward_accuracy: bool = False,
        averaging_mode: str = "ram",
        output_dir: str | None = None,
        disk_members: int = 4,
        soup_max: int = 8,
    ):
        self.window = window
        self.device = device
        self.use_reward_accuracy = use_reward_accuracy

        self.averaging_mode = averaging_mode
        self.output_dir = output_dir
        self.disk_members = disk_members

        self.snapshots: deque[dict[str, torch.Tensor]] = deque(maxlen=window)
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_loss: float = float("inf")
        self.best_source: str = "none"
        self.best_step: int = -1

        self.soup_max = soup_max
        self.soup_n_members: int = 1
        self.soup_accepted: list = []
        self.pool: list[dict] = []
        self.pool_tag: Optional[str] = None
        self.pool_pin: Optional[Callable[[dict], bool]] = None
        self.soup_dropped_freed: int = 0
        self.online_resident: bool = False
        self._snap_bytes: Optional[int] = None
        self._numel: Optional[tuple] = None
        self.on_soup_pass1: Optional[Callable] = None
        self._submission_dir: str | None = None
        self.end_time: str = ""

        self.overfit_counter = 0
        self.rollback_count = 0
        self.neftune_level_idx = 0

        self.stop_rule_facts: Optional[dict] = None
        self.stop_rule_devpass_reserve: float = 0.0
        self.stop_rule_cadences: Optional[Callable[[], tuple]] = None
        self.planner_clock = None
        self._sr_t0: Optional[float] = None
        self._sr_evals: list[tuple[int, float]] = []
        self._sr_devs: list[tuple[int, float]] = []
        self.depth_first = None

        self.std_active: bool = False
        self.std_handoff: Optional[dict] = None
        self.std_guard_secs: float = 0.0
        self.std_gate_step: Optional[Callable[[], int]] = None
        self.std_clock_cb = None
        self.decay_started: Optional[dict] = None

        self._evaluating = False
        self.trainer = None

    def _get_metric(self, metrics):
        
        if self.use_reward_accuracy:
            acc = metrics.get("eval_rewards/accuracies")
            if acc is not None:
                return -acc
        return metrics.get("eval_loss")

    @staticmethod
    def _unwrap(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    @torch.no_grad()
    def _snapshot(self, model, *, step: Optional[int] = None,
                  consumer: str = "ckpt_avg") -> dict[str, torch.Tensor]:
        unwrapped = self._unwrap(model)
        named = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]
        params = {n: p.data for n, p in named}
        if precise_update.soup_fp32():
            hi = precise_update.high_precision(unwrapped)
            if hi:
                params = {n: hi.get(n, t) for n, t in params.items()}
        tr = getattr(self, "trainer", None)
        sharded = bool(getattr(tr, "is_deepspeed_enabled", False)
                       or getattr(tr, "is_fsdp_enabled", False)
                       or any(hasattr(p, "ds_id") or hasattr(p, "_is_flat_param") for _, p in named))
        if not host_snapshot.enabled() or sharded:
            return {n: t.cpu().clone() for n, t in params.items()}
        return host_snapshot.snapshot(
            params, consumer=consumer,
            share_with=((unwrapped, int(step)) if step is not None else None))

    def _restore(self, model, state: dict[str, torch.Tensor]):
        host_snapshot.weights_changed("ckpt_avg.restore")
        unwrapped = self._unwrap(model)
        with torch.no_grad():
            host_snapshot.restore_params(unwrapped.named_parameters(), state)

    def _soup_engine(self, model, op: str, *, n_acc: int = 1, cands: Optional[list] = None,
                     acc_host: bool = False) -> "gpu_soup.SoupEngine":
        numel, biggest = self._trainable_numel(model)
        return gpu_soup.SoupEngine.build(
            op, list(self._unwrap(model).named_parameters()), trainer=self.trainer, n_acc=n_acc,
            cand_bytes=(int(self._trainable_snapshot_bytes(model)) if cands else 0),
            n_cands=len(cands or []), numel=numel, max_numel=biggest, acc_host=acc_host)

    @torch.no_grad()
    def _compute_avg(self) -> dict[str, torch.Tensor]:
        avg = {}
        n = len(self.snapshots)
        for name in self.snapshots[0]:
            avg[name] = sum(s[name] for s in self.snapshots) / n
        return avg

    def _bcast_params(self, model):
        if torch.distributed.is_initialized():
            for p in self._unwrap(model).parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)

    def _bcast_flag(self, model, value: bool) -> bool:
        if not torch.distributed.is_initialized():
            return value
        t = torch.tensor([1.0 if value else 0.0], device=next(model.parameters()).device)
        torch.distributed.broadcast(t, src=0)
        return t.item() > 0.5

    def _bcast_scalar(self, model, value) -> float:
        if not torch.distributed.is_initialized():
            return value if value is not None else float("inf")
        t = torch.tensor([value if value is not None else float("inf")],
                         device=next(model.parameters()).device)
        torch.distributed.broadcast(t, src=0)
        return t.item()

    def _agreed(self, model, value: bool) -> bool:
        if traj2._multi():
            return self._bcast_flag(model, bool(value))
        return bool(value)

    def _restore_best(self, model):
        if is_main_process(LOCAL_RANK) and self.best_state is not None:
            self._restore(model, self.best_state)
        self._bcast_params(model)

    def _trainable_snapshot_bytes(self, model) -> int:
        
        if self._snap_bytes is None:
            self._snap_bytes = sum(
                p.numel() * (4 if precise_update.soup_fp32() else p.element_size())
                for p in self._unwrap(model).parameters() if p.requires_grad
            )
        return self._snap_bytes

    @staticmethod
    def _available_ram_bytes() -> Optional[int]:
        return host_snapshot.available_ram_bytes(_CGROUP_MEMORY_FILES, _CGROUP_UNLIMITED)

    def _trainable_numel(self, model) -> tuple:
        if getattr(self, "_numel", None) is None and model is None:
            return (int(self._trainable_snapshot_bytes(model)) // 2, 0)
        if getattr(self, "_numel", None) is None:
            ns = [p.numel() for p in self._unwrap(model).parameters() if p.requires_grad]
            self._numel = (int(sum(ns)), int(max(ns) if ns else 0))
        return self._numel

    def _soup_peak_bytes(self, model) -> int:
        numel, biggest = self._trainable_numel(model)
        if getattr(self, "online_resident", False):
            return int(self._trainable_snapshot_bytes(model)) + 2 * 4 * int(biggest)
        return host_snapshot.soup_phase_peak_bytes(
            self._trainable_snapshot_bytes(model), numel, biggest)

    def _ram_budget(self) -> "host_snapshot.RamBudget":
        b = host_snapshot.ram_budget(_CGROUP_MEMORY_FILES, _CGROUP_UNLIMITED)
        return b.model_copy(update={"avail": self._available_ram_bytes()})

    def _live_pool(self) -> list:
        return [e for e in self.pool if e.get("state") is not None]

    def _ram_pool_cap(self, model, budget=None) -> tuple:
        b = budget if budget is not None else self._ram_budget()
        snap = self._trainable_snapshot_bytes(model)
        cap = host_snapshot.pool_cap_from_ram(
            limit=b.limit, usage=b.usage, pool_bytes=len(self._live_pool()) * snap,
            snap_bytes=snap, soup_peak=self._soup_peak_bytes(model),
            reserve=host_snapshot.host_reserve_bytes(b.limit))
        return cap, b

    def _evict_worst(self) -> Optional[dict]:
        live = self._live_pool()
        if not live:
            return None
        _evictable = ([e for e in live if not self.pool_pin(e)]
                      if self.pool_pin is not None else live) or live
        worst = max(_evictable, key=lambda e: e["loss"])
        self.pool.remove(worst)
        worst["state"] = None
        return worst

    def _host_release(self) -> dict:
        info = host_snapshot.release_host_memory()
        self.host_release_secs = float(getattr(self, "host_release_secs", 0.0) or 0.0) + info["secs"]
        return info

    def _trim_pool_to(self, cap: int, why: str, budget=None) -> list:
        dropped = []
        while len(self._live_pool()) > max(0, int(cap)):
            w = self._evict_worst()
            if w is None:
                break
            dropped.append((w.get("step"), w.get("src"), w.get("loss")))
        if dropped:
            self._host_release()
        return dropped

    def _can_admit_snapshot(self, model, budget=None) -> bool:
        
        snap = self._trainable_snapshot_bytes(model)
        b = budget if budget is not None else self._ram_budget()
        avail = b.avail
        if avail is None:
            return len(self.pool) < 2
        margin = max(2 * 1024**3, 3 * snap,
                     self._soup_peak_bytes(model) + host_snapshot.host_reserve_bytes(b.limit))
        return (avail - margin) >= snap

    def _consider_for_pool(self, model, loss, step: int) -> None:
        
        if loss is None or loss != loss:
            return
        src = self.pool_tag or "a0"
        ram_cap, budget = self._ram_pool_cap(model)
        cap = int(self.soup_max)
        if ram_cap is not None and ram_cap < cap:
            cap = ram_cap
            if self._trim_pool_to(cap, "admission", budget) and host_snapshot.trim_enabled():
                ram_cap, budget = self._ram_pool_cap(model)
                cap = min(int(self.soup_max), int(ram_cap)) if ram_cap is not None else cap
        admit = len(self.pool) < cap and self._can_admit_snapshot(model)
        if not admit and len(self.pool) < cap:
            self._trainable_snapshot_bytes(model)
            self._soup_peak_bytes(model)
        elif ram_cap is not None and ram_cap < int(self.soup_max):
            self._trainable_snapshot_bytes(model)
            self._soup_peak_bytes(model)
        if admit:
            self.pool.append({"loss": loss, "step": step, "src": src,
                              "state": self._snapshot(model, step=step,
                                                      consumer="ckpt_avg.pool")})
        elif self.pool:
            _evictable = ([e for e in self.pool if not self.pool_pin(e)]
                          if self.pool_pin is not None else self.pool) or self.pool
            worst = max(_evictable, key=lambda e: e["loss"])
            if loss < worst["loss"]:
                self.pool.remove(worst)
                worst["state"] = None
                del worst
                self._host_release()
                self.pool.append({"loss": loss, "step": step, "src": src,
                                  "state": self._snapshot(model, step=step,
                                                          consumer="ckpt_avg.pool")})

        self.pool.sort(key=lambda e: e["loss"])
        snap_gb = self._trainable_snapshot_bytes(model) / 1e9
        avail = self._available_ram_bytes()
        avail_str = f"{avail / 1e9:.0f}GB" if avail is not None else "?"
        print(
            f"[sn56][soup] pool={len(self.pool)}/{self.soup_max} "
            f"(snap~{snap_gb:.2f}GB/ea, RAM livre {avail_str})",
            flush=True,
        )

    def _fit_soup_phase_to_ram(self, model, cands: list) -> int:
        b = self._ram_budget()
        if b.avail is None:
            return 0
        snap = self._trainable_snapshot_bytes(model)
        need = self._soup_peak_bytes(model) + host_snapshot.host_reserve_bytes(b.limit)
        room = int(b.avail)
        freed = []
        for e in reversed(cands[1:]):
            if room >= need:
                break
            if e.get("state") is None:
                continue
            if self.pool_pin is not None and self.pool_pin(e) and any(
                    not self.pool_pin(x) and x.get("state") is not None for x in cands[1:]):
                continue
            e["state"] = None
            room += snap
            freed.append((e.get("step"), e.get("src")))
        if freed:
            gc.collect()
        return len(freed)

    def _stage_ram(self, model, stage: str) -> tuple:
        numel, biggest = self._trainable_numel(model)
        need = host_snapshot.post_pass1_stage_need(
            stage, snap_bytes=self._trainable_snapshot_bytes(model), numel=numel,
            max_tensor_numel=biggest)
        b = self._ram_budget()
        room = host_snapshot.stage_room(b)
        info = {"stage": stage, "need": int(need), "room": room,
                "reserve": int(host_snapshot.host_reserve_bytes(b.limit)), **b.fields()}
        return (room is None or room >= need), info

    def _fit_pass2_to_ram(self, model, cands: list, rejected: list) -> tuple:
        fits, _ = self._stage_ram(model, "pass2")
        if fits:
            return rejected, 0
        order = sorted(rejected, key=lambda i: cands[i]["loss"], reverse=True)
        if self.pool_pin is not None:
            order = ([i for i in order if not self.pool_pin(cands[i])]
                     + [i for i in order if self.pool_pin(cands[i])])
        freed = []
        for i in order:
            if cands[i].get("state") is None:
                continue
            cands[i]["state"] = None
            freed.append(i)
            gc.collect()
            fits, _ = self._stage_ram(model, "pass2")
            if fits:
                break
        live = [i for i in rejected if cands[i].get("state") is not None]
        skipped = not live
        if skipped:
            print("[sn56][soup] 2a passada pulada: RAM do container", flush=True)
        return (live if not skipped else []), len(freed)

    def _soup_remaining_secs(self) -> Optional[float]:
        if not self.end_time:
            return None
        try:
            end = datetime.datetime.strptime(
                self.end_time, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            return (end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        except Exception:
            return None

    @torch.no_grad()
    def _load_soup_into_model(self, model, soup_sum: dict, n: int, eng=None) -> None:
        
        host_snapshot.weights_changed("ckpt_avg.load_soup")
        inv = 1.0 / n
        if eng is not None:
            eng.load_scaled(soup_sum, inv)
            return
        for name, p in self._unwrap(model).named_parameters():
            if name in soup_sum:
                p.data.copy_((soup_sum[name] * inv).to(p.dtype))

    @torch.no_grad()
    def _load_weighted_into_model(self, model, base_sum: dict, extra: dict, w: float,
                                  n, eng=None) -> None:
        host_snapshot.weights_changed("ckpt_avg.load_soup")
        inv = 1.0 / n
        if eng is not None:
            eng.load_weighted(base_sum, extra, w, inv)
            return
        for name, p in self._unwrap(model).named_parameters():
            if name in base_sum:
                t = base_sum[name]
                if name in extra:
                    t = t + w * extra[name].float()
                p.data.copy_((t * inv).to(p.dtype))
                del t

    def _greedy_soup(self, model, second_pass: bool = False,
                     second_pass_max_secs: Optional[float] = None) -> bool:
        engs: list = []
        try:
            return self._greedy_soup_impl(model, second_pass, second_pass_max_secs, engs)
        finally:
            for e in engs:
                e.close()

    @torch.no_grad()
    def _greedy_soup_impl(self, model, second_pass: bool, second_pass_max_secs: Optional[float],
                          engs: list) -> bool:
        
        is_main = is_main_process(LOCAL_RANK)
        self.soup_dropped_freed = 0
        if is_main:
            _sorted = sorted(self.pool, key=lambda e: e["loss"])
            cands = [e for e in _sorted if e.get("state") is not None]
            self.soup_dropped_freed = len(_sorted) - len(cands)
        else:
            cands = None
        k = int(self._bcast_scalar(model, float(len(cands) if is_main else 0)))
        if k <= 1:
            return False

        eng = None
        if is_main:
            self._fit_soup_phase_to_ram(model, cands)
            eng = self._soup_engine(model, "greedy", cands=cands)
            engs.append(eng)
        soup_sum = eng.init_acc(cands[0]["state"]) if is_main else None
        if is_main:
            self._restore(model, cands[0]["state"])
        self._bcast_params(model)

        t0 = time.perf_counter()
        seed_loss = self._eval_synced(model)
        if seed_loss != seed_loss or seed_loss == float("inf"):
            return False
        per_eval = max(1.0, time.perf_counter() - t0)
        remaining = self._soup_remaining_secs()
        budget = (remaining * 0.5) if remaining is not None else float("inf")

        best, n = seed_loss, 1
        accepted_idx: list = [0]
        rejected: list = []
        _tried = 1
        for i in range(1, k):
            if self._agreed(model, (time.perf_counter() - t0) + per_eval > budget):
                if is_main:
                    print(f"[sn56][soup] tempo esgotado após {i} cand(s)", flush=True)
                break
            freed = int(self._bcast_scalar(
                model, 1.0 if (is_main and cands[i].get("state") is None) else 0.0))
            if freed:
                if is_main:
                    self.soup_dropped_freed += 1
                continue
            if is_main:
                inv = 1.0 / (n + 1)
                st = cands[i]["state"]
                host_snapshot.weights_changed("ckpt_avg.soup_trial")
                eng.load_trial(soup_sum, st, inv)
            self._bcast_params(model)
            _t_ev = time.perf_counter()
            cand_loss = self._eval_synced(model)
            _t_ev = time.perf_counter() - _t_ev
            _tried += 1
            accept = cand_loss == cand_loss and cand_loss < best - 1e-6
            if is_main:
                eng.trial_event(index=i, eval_secs=_t_ev, accepted=bool(accept))
            if accept:
                if is_main:
                    st = cands[i]["state"]
                    eng.add(soup_sum, st)
                    accepted_idx.append(i)
                n += 1
                best = cand_loss
            elif is_main:
                self._load_soup_into_model(model, soup_sum, n, eng)
                rejected.append(i)
        if _tried >= 2:
            self.soup_secs_per_cand = (time.perf_counter() - t0) / _tried
        if is_main and eng is not None:
            self.soup_trial_secs = [float(t) for t in (getattr(eng, "trial_secs", None) or [])]
        if is_main and getattr(self, "on_soup_pass1", None) is not None and n >= 2:
            try:
                self.on_soup_pass1(model, best, n)
            except Exception:
                pass
        if is_main and second_pass and n >= 2 and rejected:
            rejected, _ = self._fit_pass2_to_ram(model, cands, rejected)
        n2 = int(self._bcast_scalar(
            model, float(len(rejected) if (is_main and second_pass and n >= 2) else 0)))
        t_pass2 = time.perf_counter()
        for j in range(n2):
            if self._agreed(model, (
                    (time.perf_counter() - t0) + per_eval > budget
                    or (second_pass_max_secs is not None
                        and (time.perf_counter() - t_pass2) + per_eval > second_pass_max_secs))):
                if is_main:
                    print(f"[sn56][soup] tempo esgotado na 2a passada após {j} cand(s)",
                          flush=True)
                break
            i = rejected[j] if is_main else -1
            freed = int(self._bcast_scalar(
                model, 1.0 if (is_main and cands[i].get("state") is None) else 0.0))
            if freed:
                continue
            if is_main:
                inv = 1.0 / (n + 1)
                st = cands[i]["state"]
                host_snapshot.weights_changed("ckpt_avg.soup_trial")
                eng.load_trial(soup_sum, st, inv)
            self._bcast_params(model)
            _t_ev = time.perf_counter()
            cand_loss = self._eval_synced(model)
            _t_ev = time.perf_counter() - _t_ev
            accept = cand_loss == cand_loss and cand_loss < best - 1e-6
            if is_main:
                eng.trial_event(index=i, eval_secs=_t_ev, accepted=bool(accept), pass_no=2)
            if accept:
                if is_main:
                    st = cands[i]["state"]
                    eng.add(soup_sum, st)
                    accepted_idx.append(i)
                n += 1
                best = cand_loss
            elif is_main:
                self._load_soup_into_model(model, soup_sum, n, eng)
        if is_main:
            self._load_soup_into_model(model, soup_sum, n, eng)
        soup_sum = None
        self._bcast_params(model)

        if is_main:
            self.best_state = self._snapshot(model)
            self.best_loss = best
            self.soup_eval_record = (
                eval_squeeze.SoupEvalRecord(state_id=id(self.best_state), loss=float(best),
                                            config=eval_squeeze.eval_config(self.trainer),
                                            secs_per_cand=getattr(self, "soup_secs_per_cand",
                                                                  None))
                if best == best and best != float("inf") else None)
            self.best_source = f"soup(n={n}/{k})"
            self.soup_n_members = int(n)
            self.soup_accepted = [cands[i] for i in accepted_idx]
            print(f"[sn56][soup] n={n}/{k}, loss {seed_loss:.4f} -> {best:.4f}", flush=True)
        self.best_loss = self._bcast_scalar(model, best if is_main else best)
        return True

    @torch.no_grad()
    def _online_trial(self, model, soup_sum: Optional[dict], n: int, state: Optional[dict],
                      eng=None, index=None) -> float:
        is_main = is_main_process(LOCAL_RANK)
        own = None
        try:
            if is_main and soup_sum is not None and state is not None:
                if eng is None:
                    eng = own = self._soup_engine(model, "online_trial", acc_host=True)
                inv = 1.0 / (int(n) + 1)
                host_snapshot.weights_changed("ckpt_avg.online_trial")
                eng.load_trial(soup_sum, state, inv)
            self._bcast_params(model)
            _t_ev = time.perf_counter()
            loss = self._eval_synced(model)
            if eng is not None:
                eng.trial_event(index=index, eval_secs=time.perf_counter() - _t_ev,
                                label="online")
            return loss
        finally:
            if own is not None:
                own.close()

    @torch.no_grad()
    def _online_add(self, model, soup_sum: dict, state: dict, eng=None) -> None:
        own = None
        try:
            if eng is None:
                eng = own = self._soup_engine(model, "online_add", acc_host=True)
            eng.add(soup_sum, state)
        finally:
            if own is not None:
                own.close()

    @torch.no_grad()
    def _online_eval_soup(self, model, soup_sum: Optional[dict], n: int) -> float:
        if is_main_process(LOCAL_RANK) and soup_sum is not None:
            eng = self._soup_engine(model, "online_eval", acc_host=True)
            try:
                self._load_soup_into_model(model, soup_sum, int(n), eng)
            finally:
                eng.close()
        self._bcast_params(model)
        return self._eval_synced(model)

    def _greedy_extend(self, model, soup_sum: Optional[dict], n: int, loss: float,
                       srcs: Optional[list] = None, second_pass_max_secs=None,
                       extra: Optional[list] = None,
                       extra_fits: Optional[Callable[[float], bool]] = None) -> bool:
        engs: list = []
        try:
            return self._greedy_extend_impl(model, soup_sum, n, loss, srcs,
                                            second_pass_max_secs, engs, extra, extra_fits)
        finally:
            for e in engs:
                e.close()

    @torch.no_grad()
    def _greedy_extend_impl(self, model, soup_sum: Optional[dict], n: int, loss: float,
                            srcs: Optional[list], second_pass_max_secs, engs: list,
                            extra: Optional[list] = None,
                            extra_fits: Optional[Callable[[float], bool]] = None) -> bool:
        is_main = is_main_process(LOCAL_RANK)
        self.soup_dropped_freed = 0
        cands = (sorted([e for e in self.pool if e.get("state") is not None
                         and e.get("src") != "a0_soup"], key=lambda e: e["loss"])
                 if is_main else None)
        k = int(self._bcast_scalar(model, float(len(cands) if is_main else 0)))
        n = int(self._bcast_scalar(model, float(n if is_main else 0)))
        best = float(self._bcast_scalar(model, loss if is_main else 0.0))
        if n < 1 or best != best or best == float("inf"):
            return False
        eng = None
        if is_main and soup_sum is not None:
            eng = self._soup_engine(model, "extend", cands=cands)
            engs.append(eng)
            eng.adopt(soup_sum)
        t0 = time.perf_counter()
        remaining = self._soup_remaining_secs()
        budget = (remaining * 0.5) if remaining is not None else float("inf")
        if second_pass_max_secs is not None:
            budget = min(budget, float(second_pass_max_secs) * 2.0)
        per_eval = 1.0
        n_acc = 0
        taken: list = []
        for i in range(k):
            if self._agreed(model, (time.perf_counter() - t0) + per_eval > budget):
                if is_main:
                    print(f"[sn56][soup] extensão: tempo esgotado após {i} cand(s)", flush=True)
                break
            t1 = time.perf_counter()
            cand_loss = self._online_trial(model, soup_sum, n,
                                           cands[i]["state"] if is_main else None,
                                           eng=eng, index=i + 1)
            per_eval = max(per_eval, time.perf_counter() - t1)
            accept = cand_loss == cand_loss and cand_loss < best - 1e-6
            if accept:
                if is_main:
                    st = cands[i]["state"]
                    eng.add(soup_sum, st)
                    taken.append(cands[i])
                n += 1
                n_acc += 1
                best = cand_loss
        if k:
            self.soup_secs_per_cand = (time.perf_counter() - t0) / max(1, k)
        if extra is not None:
            n, best, n_acc, _ = self._free_extend(
                model, soup_sum, n, best, extra, extra_fits, taken, per_eval, n_acc,
                eng=eng, index0=k)
        if is_main:
            self._load_soup_into_model(model, soup_sum, n, eng)
        self._bcast_params(model)
        if is_main and getattr(self, "on_soup_pass1", None) is not None and n >= 2:
            try:
                self.on_soup_pass1(model, best, n)
            except Exception:
                pass
        self.soup_n_members = int(n)
        if is_main:
            self.best_state = self._snapshot(model)
            self.best_loss = best
            self.best_source = f"soup(online,n={n})"
            self.soup_accepted = list(taken)
            print(f"[sn56][soup] online n={n}, loss {loss:.4f} -> {best:.4f}", flush=True)
        self.best_loss = self._bcast_scalar(model, best)
        return True

    @torch.no_grad()
    def _free_extend(self, model, soup_sum: Optional[dict], n: int, best: float,
                     extra: list, extra_fits: Optional[Callable[[float], bool]],
                     taken: list, per_eval: float, n_acc: int, eng=None,
                     index0: int = 0) -> tuple:
        is_main = is_main_process(LOCAL_RANK)
        cands = (sorted([e for e in (extra or []) if e.get("state") is not None],
                        key=lambda e: e["loss"]) if is_main else None)
        k2 = int(self._bcast_scalar(model, float(len(cands) if is_main else 0)))
        info = {"n_held": k2, "n_tried": 0, "n_accepted": 0, "n_skipped_time": 0,
                "dev_before": best, "dev_after": best, "accepted": [], "rejected": [],
                "skipped": []}
        per = max(1.0, float(per_eval or 1.0))
        for j in range(k2):
            fits = True
            if is_main and extra_fits is not None:
                try:
                    fits = bool(extra_fits(per))
                except Exception:
                    fits = False
            if not self._agreed(model, fits):
                if is_main:
                    for e in cands[j:]:
                        info["skipped"].append(fc.describe(e))
                        e["state"] = None
                info["n_skipped_time"] = k2 - j
                break
            e = cands[j] if is_main else None
            t1 = time.perf_counter()
            cand_loss = self._online_trial(model, soup_sum, n, e["state"] if is_main else None,
                                           eng=eng, index=index0 + j + 1)
            per = max(per, time.perf_counter() - t1)
            info["n_tried"] += 1
            accept = cand_loss == cand_loss and cand_loss < best - 1e-6
            if is_main:
                (info["accepted"] if accept else info["rejected"]).append(fc.describe(e))
            if accept:
                if is_main:
                    st = e["state"]
                    if eng is not None:
                        eng.add(soup_sum, st)
                    else:
                        for name in soup_sum:
                            if name in st:
                                soup_sum[name] += st[name].float()
                    taken.append(e)
                n += 1
                n_acc += 1
                info["n_accepted"] += 1
                best = cand_loss
            if is_main:
                e["state"] = None
        if is_main and cands:
            gc.collect()
        info["dev_after"] = best
        return n, best, n_acc, info

    @torch.no_grad()
    def _a0_weight_probe_sum(self, model, soup_sum: Optional[dict], n: int,
                             a0_state: Optional[dict], margin: float = 0.0) -> dict:
        is_main = is_main_process(LOCAL_RANK)
        info = {"ran": False, "reason": "", "incumbent": self.best_loss,
                "cand_half": None, "cand_zero": None, "adopted": "greedy",
                "weight": "1/n", "n_members": int(n), "margin": margin, "online": True}
        incumbent_state, incumbent_loss = self.best_state, self.best_loss
        eligible = bool(is_main and soup_sum is not None and a0_state is not None
                        and int(n) >= 2 and incumbent_state is not None)
        ram_ok = True
        if eligible:
            ram_ok, _ = self._stage_ram(model, "pass2")
        ok = int(self._bcast_scalar(model, float(1 if (eligible and ram_ok) else 0)))
        if not ok:
            info["reason"] = ("host_ram" if (is_main and eligible and not ram_ok)
                              else "no_soup")
            if is_main and incumbent_state is not None:
                self._restore(model, incumbent_state)
            self._bcast_params(model)
            return info

        eng = self._soup_engine(model, "a0_probe_online", acc_host=True) if is_main else None

        def _load(w: float) -> None:
            host_snapshot.weights_changed("ckpt_avg.load_soup")
            inv = 1.0 / ((int(n) - 1) + w)
            eng.load_minus(soup_sum, a0_state, w, inv)

        try:
            return self._a0_weight_probe_sum_run(model, _load, eng, info, incumbent_state,
                                                 incumbent_loss, margin, n)
        finally:
            if eng is not None:
                eng.close()

    def _a0_weight_probe_sum_run(self, model, _load, eng, info, incumbent_state, incumbent_loss,
                                 margin, n) -> dict:
        is_main = is_main_process(LOCAL_RANK)
        results = {}
        for key, w in (("cand_half", 0.5), ("cand_zero", 0.0)):
            if is_main:
                _load(w)
            self._bcast_params(model)
            _t_ev = time.perf_counter()
            results[key] = info[key] = self._eval_synced(model)
            if eng is not None:
                eng.trial_event(index=key, eval_secs=time.perf_counter() - _t_ev,
                                label="a0_probe")
        best_key = min((k for k in results if results[k] == results[k]),
                       key=lambda k: results[k], default=None)
        adopt = (best_key is not None
                 and results[best_key] < incumbent_loss - max(0.0, margin))
        adopt = bool(self._bcast_scalar(model, 1.0 if adopt else 0.0) > 0.5)
        if adopt:
            w = 0.5 if best_key == "cand_half" else 0.0
            if is_main:
                _load(w)
                self.best_state = self._snapshot(model)
                self.best_loss = results[best_key]
                self.best_source = f"soup(online,n={n},a0w={'1/2n' if w else '0'})"
            info.update(adopted=best_key, weight=("1/(2n)" if w else "0"))
            self._bcast_params(model)
        else:
            if is_main:
                self._restore(model, incumbent_state)
            self._bcast_params(model)
        self.best_loss = self._bcast_scalar(model, self.best_loss)
        info["ran"] = True
        return info

    @torch.no_grad()
    def _a0_weight_probe(self, model, margin: float = 0.0) -> dict:
        
        is_main = is_main_process(LOCAL_RANK)
        info = {"ran": False, "reason": "", "incumbent": self.best_loss,
                "cand_half": None, "cand_zero": None, "adopted": "greedy",
                "weight": "1/n", "n_members": 0, "margin": margin}
        incumbent_state, incumbent_loss = self.best_state, self.best_loss
        acc = list(self.soup_accepted or []) if is_main else []
        a0 = next((e for e in acc if e.get("src") == "a0_soup"
                   and e.get("state") is not None), None) if is_main else None
        eligible = bool(is_main and a0 is not None and len(acc) >= 2
                        and incumbent_state is not None
                        and all(e.get("state") is not None for e in acc))
        ram_ok = True
        if eligible:
            ram_ok, _ = self._stage_ram(model, "a0_probe")
        ok = int(self._bcast_scalar(model, float(1 if (eligible and ram_ok) else 0)))
        if not ok:
            info["reason"] = ("host_ram" if (is_main and eligible and not ram_ok)
                              else "a0_soup_not_in_soup" if is_main and a0 is None
                              else "no_soup")
            if is_main and incumbent_state is not None:
                self._restore(model, incumbent_state)
            self._bcast_params(model)
            return info
        n = len(acc) if is_main else 0
        info["n_members"] = n
        others_sum = None
        eng = None
        if is_main:
            eng = self._soup_engine(model, "a0_probe")
            others = [e for e in acc if e is not a0]
            others_sum = eng.init_acc(others[0]["state"])
            for e in others[1:]:
                eng.add(others_sum, e["state"])
        try:
            return self._a0_weight_probe_run(model, eng, others_sum, a0, n, info,
                                             incumbent_state, incumbent_loss, margin)
        finally:
            others_sum = None
            if eng is not None:
                eng.close()

    def _a0_weight_probe_run(self, model, eng, others_sum, a0, n, info, incumbent_state,
                             incumbent_loss, margin) -> dict:
        is_main = is_main_process(LOCAL_RANK)
        results = {}
        for key, w in (("cand_half", 0.5), ("cand_zero", 0.0)):
            if is_main:
                self._load_weighted_into_model(model, others_sum, a0["state"], w, (n - 1) + w,
                                               eng)
            self._bcast_params(model)
            _t_ev = time.perf_counter()
            loss = self._eval_synced(model)
            if eng is not None:
                eng.trial_event(index=key, eval_secs=time.perf_counter() - _t_ev,
                                label="a0_probe")
            results[key] = loss
            info[key] = loss
        best_key = min((k for k in results if results[k] == results[k]),
                       key=lambda k: results[k], default=None)
        adopt = (best_key is not None
                 and results[best_key] < incumbent_loss - max(0.0, margin))
        adopt = bool(self._bcast_scalar(model, 1.0 if adopt else 0.0) > 0.5)
        if adopt:
            w = 0.5 if best_key == "cand_half" else 0.0
            if is_main:
                self._load_weighted_into_model(model, others_sum, a0["state"], w, (n - 1) + w,
                                               eng)
                others_sum = None
                self.best_state = self._snapshot(model)
                self.best_loss = results[best_key]
                self.best_source = f"soup(n={n},a0w={'1/2n' if w else '0'})"
                info.update(adopted=best_key, weight=("1/(2n)" if w else "0"))
            self._bcast_params(model)
        elif is_main:
            self._restore(model, incumbent_state)
            self._bcast_params(model)
        else:
            self._bcast_params(model)
        self.best_loss = self._bcast_scalar(model, self.best_loss if is_main else self.best_loss)
        info["ran"] = True
        return info

    @torch.no_grad()
    def _eval_uniform_soup(self, model, states) -> float:
        
        is_main = is_main_process(LOCAL_RANK)
        n = int(self._bcast_scalar(
            model, float(len(states) if (is_main and states) else 0)))
        if n < 1:
            return float("inf")
        if is_main:
            eng = self._soup_engine(model, "uniform")
            try:
                acc = eng.init_acc(states[0])
                for st in states[1:]:
                    eng.add(acc, st)
                self._load_soup_into_model(model, acc, len(states), eng)
                acc = None
            finally:
                eng.close()
        self._bcast_params(model)
        return self._eval_synced(model)

    def _greedy_soup_over(self, model, entries, eng=None) -> tuple:
        own = None
        if eng is None and is_main_process(LOCAL_RANK):
            live = [e for e in (entries or []) if e.get("state") is not None]
            eng = own = self._soup_engine(model, "greedy_over", cands=live)
        try:
            out = self._greedy_soup_over_impl(model, entries, eng)
            if own is not None and out[0] is not None:
                own.home(out[0])
            return out
        finally:
            if own is not None:
                own.close()

    @torch.no_grad()
    def _greedy_soup_over_impl(self, model, entries, eng) -> tuple:
        
        is_main = is_main_process(LOCAL_RANK)
        cands = (sorted([e for e in entries if e.get("state") is not None],
                        key=lambda e: e["loss"]) if is_main else None)
        k = int(self._bcast_scalar(model, float(len(cands) if is_main else 0)))
        if k < 1:
            return None, 0, float("inf")
        soup_sum = eng.init_acc(cands[0]["state"]) if is_main else None
        if is_main:
            self._restore(model, cands[0]["state"])
        self._bcast_params(model)
        best = self._eval_synced(model)
        if best != best or best == float("inf"):
            return None, 0, float("inf")
        n = 1
        for i in range(1, k):
            if is_main:
                inv = 1.0 / (n + 1)
                st = cands[i]["state"]
                host_snapshot.weights_changed("ckpt_avg.soup_trial")
                eng.load_trial(soup_sum, st, inv)
            self._bcast_params(model)
            _t_ev = time.perf_counter()
            cand_loss = self._eval_synced(model)
            _t_ev = time.perf_counter() - _t_ev
            accept = cand_loss == cand_loss and cand_loss < best - 1e-6
            if is_main:
                eng.trial_event(index=i, eval_secs=_t_ev, accepted=bool(accept),
                                label="greedy_over")
            if accept:
                if is_main:
                    st = cands[i]["state"]
                    eng.add(soup_sum, st)
                n += 1
                best = cand_loss
            elif is_main:
                self._load_soup_into_model(model, soup_sum, n, eng)
        return soup_sum, n, best

    @torch.no_grad()
    def _soup_of_soups(self, model, margin: float = 0.0) -> dict:
        
        is_main = is_main_process(LOCAL_RANK)
        info = {"ran": False, "reason": "", "incumbent": self.best_loss,
                "candidate": None, "adopted": "greedy", "margin": margin,
                "members": None, "total_members": 0, "n_soups": 0}
        incumbent_state = self.best_state
        incumbent_loss = self.best_loss
        if is_main and incumbent_state is None:
            info["reason"] = "no_incumbent"
            return info

        groups: dict = {}
        a0_entry = None
        if is_main:
            a0_entry = next((e for e in self.pool if e.get("src") == "a0_soup"
                             and e.get("state") is not None), None)
            for e in self.pool:
                if e.get("state") is None:
                    continue
                src = e.get("src", "a0")
                if src in ("a0", "a0_soup"):
                    continue
                groups.setdefault(src, []).append(e)

        ram_skip = False
        if is_main and groups:
            _fits, _ = self._stage_ram(model, "blind_soup")
            if not _fits:
                groups, ram_skip = {}, True
        n_groups = int(self._bcast_scalar(model, float(len(groups) if is_main else 0)))
        if n_groups < 1:
            info["reason"] = "host_ram" if ram_skip else "no_extra_trajectories"
            if is_main:
                self._restore(model, incumbent_state)
            self._bcast_params(model)
            return info

        eng = None
        if is_main:
            eng = self._soup_engine(
                model, "soup_of_soups", n_acc=2,
                cands=[e for g in groups.values() for e in g if e.get("state") is not None])
        try:
            return self._soup_of_soups_run(model, eng, groups, a0_entry, n_groups, info,
                                           incumbent_state, incumbent_loss, margin)
        finally:
            if eng is not None:
                eng.close()

    def _soup_of_soups_run(self, model, eng, groups, a0_entry, n_groups, info, incumbent_state,
                           incumbent_loss, margin) -> dict:
        is_main = is_main_process(LOCAL_RANK)
        total_sum = None
        total_n = 0
        members: dict = {}
        n_soups = 0
        if is_main and a0_entry is not None:
            m0 = max(1, int(a0_entry.get("members", 1) or 1))
            total_sum = eng.init_scaled(a0_entry["state"], m0)
            total_n = m0
            members["a0_soup"] = m0
            n_soups += 1

        src_order = sorted(groups.keys()) if is_main else None
        for gi in range(n_groups):
            entries = groups[src_order[gi]] if is_main else None
            s_sum, s_n, s_loss = self._greedy_soup_over(model, entries, eng=eng)
            usable = int(self._bcast_scalar(
                model, float(1 if (is_main and s_sum is not None and s_n >= 1
                                   and s_loss == s_loss and s_loss != float("inf"))
                             else 0)))
            if usable and is_main:
                if total_sum is None:
                    total_sum = s_sum
                else:
                    eng.add_acc(total_sum, s_sum)
                    eng.release(s_sum)
                total_n += s_n
                members[src_order[gi]] = s_n
                n_soups += 1
            elif is_main and s_sum is not None:
                eng.release(s_sum)
            s_sum = None

        have = int(self._bcast_scalar(
            model, float(1 if (is_main and total_sum is not None and total_n > 0) else 0)))
        if not have:
            info["reason"] = "no_candidate"
            if is_main:
                self._restore(model, incumbent_state)
            self._bcast_params(model)
            return info

        if is_main:
            self._load_soup_into_model(model, total_sum, total_n, eng)
            total_sum = None
        self._bcast_params(model)
        cand_loss = self._eval_synced(model)
        info.update(candidate=cand_loss, members=members,
                    total_members=total_n, n_soups=n_soups)

        adopt = (cand_loss == cand_loss
                 and cand_loss < incumbent_loss - max(0.0, margin))
        if is_main:
            if adopt:
                self.best_state = self._snapshot(model)
                self.best_loss = cand_loss
                self.best_source = f"soup_of_soups(w={total_n},n={n_soups})"
                info["adopted"] = "soup_of_soups"
            else:
                self._restore(model, incumbent_state)
        self._bcast_params(model)
        self.best_loss = self._bcast_scalar(
            model, self.best_loss if is_main else self.best_loss)
        info.update(ran=True,
                    delta_vs_greedy=((incumbent_loss - cand_loss)
                                     if (isinstance(incumbent_loss, (int, float))
                                         and isinstance(cand_loss, (int, float)))
                                     else None))
        return info

    def _eval_synced(self, model) -> float:
        
        self._evaluating = True
        try:
            loss = self._get_metric(self.trainer.evaluate())
        except Exception as e:
            print(f"[sn56][soup] eval falhou ({e})", flush=True)
            loss = None
        finally:
            self._evaluating = False
        return self._bcast_scalar(model, loss)

    @torch.no_grad()
    def _disk_average_and_submit(self, model):
        
        import glob
        from safetensors.torch import safe_open

        is_main = is_main_process(LOCAL_RANK)
        avg_dir = None
        if is_main and self.output_dir:
            ckpts = sorted(
                glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
                key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
            )
            ckpts = [c for c in ckpts
                     if glob.glob(os.path.join(c, "*.safetensors"))
                     and sg.is_usable_checkpoint(c)]
            ckpts = ckpts[-self.disk_members:]
            if len(ckpts) >= 2:
                avg_dir = self._stream_average_checkpoints(ckpts)

        if not self._bcast_flag(model, avg_dir is not None):
            return

        loaded = True
        if is_main:
            try:
                full = {}
                for sf in glob.glob(os.path.join(avg_dir, "*.safetensors")):
                    with safe_open(sf, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            full[k] = f.get_tensor(k)
                host_snapshot.weights_changed("ckpt_avg.disk_avg")
                self._unwrap(model).load_state_dict(full, strict=False)
            except Exception as e:
                print(f"[sn56][disco] carga da média falhou ({e}), mantendo melhor", flush=True)
                loaded = False
        if not self._bcast_flag(model, loaded):
            return

        self._bcast_params(model)
        self._evaluating = True
        try:
            avg_loss = self._get_metric(self.trainer.evaluate())
        except Exception as e:
            print(f"[sn56][disco] avaliação da média falhou ({e})", flush=True)
            avg_loss = None
        finally:
            self._evaluating = False
        avg_loss = self._bcast_scalar(model, avg_loss)

        if is_main:
            print(f"[sn56][disco] média={avg_loss:.4f} vs melhor={self.best_loss:.4f}", flush=True)
        if avg_loss < self.best_loss - 0.002 * abs(self.best_loss):
            if is_main and avg_dir:
                self._submit_dir(avg_dir, avg_loss)
                print("[sn56][disco] submissão atualizada com a média", flush=True)

    @torch.no_grad()
    def _stream_average_checkpoints(self, ckpts: list[str]) -> str | None:
        
        import glob
        from safetensors.torch import safe_open, save_file
        try:
            opens = []
            for c in ckpts:
                files = glob.glob(os.path.join(c, "*.safetensors"))
                opens.append({os.path.basename(f): f for f in files})
            keys = None
            for o in opens:
                ks = set()
                for b, p in o.items():
                    with safe_open(p, framework="pt", device="cpu") as h:
                        ks |= {(b, k) for k in h.keys()}
                keys = ks if keys is None else (keys & ks)
            n = len(ckpts)
            merged: dict[str, torch.Tensor] = {}
            for (b, k) in keys:
                acc = None
                src_dtype = None
                for o in opens:
                    with safe_open(o[b], framework="pt", device="cpu") as h:
                        t = h.get_tensor(k)
                        src_dtype = t.dtype
                        t = t.float()
                    acc = t if acc is None else acc + t
                merged[k] = (acc / n).to(src_dtype)
            out = (self._submission_dir or ckpts[-1]).rstrip("/") + ".disk_avg"
            if os.path.exists(out):
                shutil.rmtree(out)
            os.makedirs(out, exist_ok=True)
            save_file(merged, os.path.join(out, "model.safetensors"))
            for fn in os.listdir(ckpts[-1]):
                src = os.path.join(ckpts[-1], fn)
                if not fn.endswith(".safetensors") and os.path.isfile(src):
                    shutil.copy2(src, os.path.join(out, fn))
            return out
        except Exception as e:
            print(f"[sn56][disco] média em disco falhou ({e})", flush=True)
            return None

    def _submit_dir(self, src_dir: str, metric: float):
        
        sub = getattr(self, "_submission_dir", None)
        if not sub:
            return
        with open(os.path.join(src_dir, "loss.txt"), "w") as f:
            f.write(f"disk_avg,{metric}")
        if os.path.exists(sub):
            shutil.rmtree(sub)
        os.rename(src_dir, sub)

    @torch.no_grad()
    def _drop_memorized_samples(self, model):
        
        dataset = getattr(self.trainer.train_dataset, "eval_dataset", None)
        if dataset is None or len(dataset) < 100:
            return

        try:
            from data_filter import compute_sample_losses
            device = str(next(model.parameters()).device)
            losses = compute_sample_losses(model, dataset, batch_size=128, device=device)
        except ImportError:
            return

        import numpy as np
        nonzero = [l for l in losses if l > 0]
        if len(nonzero) < 100:
            return

        median_loss = float(np.median(nonzero))

        n_masked = 0
        for sample, loss in zip(dataset, losses):
            if 0 < loss < median_loss:
                labels = sample.get("labels", [])
                if isinstance(labels, torch.Tensor):
                    sample["labels"] = torch.full_like(labels, -100)
                elif isinstance(labels, list):
                    sample["labels"] = [-100] * len(labels)
                n_masked += 1

        if n_masked > 0:
            print(
                f"Escondi {n_masked}/{len(dataset)} memorized samples "
                f"(loss < {median_loss:.4f}), zero gradient contribution",
                flush=True,
            )

    def _rollback_and_respond(self, model, state):
        
        self.rollback_count += 1

        _restored_from_disk = False
        if is_main_process(LOCAL_RANK):
            submission_dir = getattr(self, "_submission_dir", None)
            if submission_dir and os.path.exists(submission_dir):
                try:
                    from safetensors.torch import load_file
                    import glob
                    safetensor_files = glob.glob(os.path.join(submission_dir, "*.safetensors"))
                    if safetensor_files:
                        unwrapped = self._unwrap(model)
                        full_state = {}
                        for sf in safetensor_files:
                            full_state.update(load_file(sf, device=str(next(unwrapped.parameters()).device)))
                        host_snapshot.weights_changed("ckpt_avg.rollback")
                        unwrapped.load_state_dict(full_state, strict=True)
                        _restored_from_disk = True
                        print(f"pesos recuperados do disco ({submission_dir})", flush=True)
                except Exception as e:
                    print(f"disco falhou ({e}), tentando memória", flush=True)
                    _restored_from_disk = False

            if not _restored_from_disk and self.best_state is not None:
                self._restore(model, self.best_state)

        if torch.distributed.is_initialized():
            _unwrapped = self._unwrap(model)
            for p in _unwrapped.parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)
            for b in _unwrapped.buffers():
                torch.distributed.broadcast(b.data, src=0)

        if self.trainer is not None and self.trainer.optimizer is not None:
            for pg in self.trainer.optimizer.param_groups:
                old_lr = pg["lr"]
                pg["lr"] = old_lr * 0.75
            if hasattr(self.trainer, "lr_scheduler") and self.trainer.lr_scheduler is not None:
                sched = self.trainer.lr_scheduler
                if hasattr(sched, "base_lrs"):
                    sched.base_lrs = [lr * 0.75 for lr in sched.base_lrs]
            new_lr = self.trainer.optimizer.param_groups[0]["lr"]
            for pg in self.trainer.optimizer.param_groups:
                for p in pg["params"]:
                    opt_state = self.trainer.optimizer.state.get(p)
                    if opt_state:
                        if "exp_avg" in opt_state:
                            opt_state["exp_avg"].zero_()
                        if "exp_avg_sq" in opt_state:
                            opt_state["exp_avg_sq"].zero_()
            print(
                f"Não não não, voltando pro melhor momento #{self.rollback_count}: "
                f"restored best weights (loss={self.best_loss:.4f}), "
                f"LR cut to {new_lr:.2e}, optimizer state reset",
                flush=True,
            )

        if self.trainer is not None:
            model_to_check = self._unwrap(self.trainer.model)
            emb = model_to_check.get_input_embeddings() if hasattr(model_to_check, "get_input_embeddings") else None
            if emb is not None and hasattr(emb, "neftune_noise_alpha") and emb.neftune_noise_alpha is not None:
                self.neftune_level_idx = min(
                    self.neftune_level_idx + 1, len(_NEFTUNE_LEVELS) - 1
                )
                new_alpha = _NEFTUNE_LEVELS[self.neftune_level_idx]
                emb.neftune_noise_alpha = new_alpha
                print(
                    f"Sacudindo mais um pouco, caos={new_alpha}",
                    flush=True,
                )

        try:
            self._drop_memorized_samples(model)
        except Exception as e:
            print(f"Filtragem falhou: {e}", flush=True)

        self.overfit_counter = 0
        self.snapshots.clear()
        if is_main_process(LOCAL_RANK) and self.best_state is not None:
            self.snapshots.append({k: v.clone() for k, v in self.best_state.items()})

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        host_snapshot.end_share("train_begin")
        self._sr_t0 = self._clock()
        self._sr_evals = []
        self._sr_devs = []
        if model is None:
            return
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mb = n_params * 4 / 1e6 * self.window
        print(
            f"Rede de segurança pronta: window={self.window}, "
            f"overfit_threshold={_OVERFIT_THRESHOLD:.0%}, patience={_OVERFIT_PATIENCE}, "
            f"max_rollbacks={_MAX_ROLLBACKS}, "
            f"{n_params/1e6:.1f}M params (~{mb:.0f}MB for snapshots)",
            flush=True,
        )

    def on_step_begin(self, args, state, control, **kwargs):
        host_snapshot.end_share("step_begin")

    @sg.nonfatal("ckpt_avg.on_evaluate")
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if self._evaluating or model is None or self.trainer is None:
            return
        is_main = is_main_process(LOCAL_RANK)

        base_loss = self._get_metric(metrics) if metrics else None
        if base_loss is None:
            return

        is_new_best = base_loss < self.best_loss
        if is_new_best:
            self.best_loss = base_loss
            self.best_step = state.global_step
            if is_main:
                try:
                    self.best_state = self._snapshot(model, step=state.global_step,
                                                     consumer="ckpt_avg.best_state")
                    self.best_source = f"base@step{state.global_step}"
                except Exception:
                    pass
            self.overfit_counter = 0

        if is_main and self.averaging_mode == "ram":
            try:
                self._consider_for_pool(model, base_loss, state.global_step)
            except Exception:
                pass

        if torch.distributed.is_initialized():
            _stop_req = 1.0 if (is_main and sg.take_stop_request()) else 0.0
            bl_tensor = torch.tensor([self.best_loss, _stop_req],
                                     device=next(model.parameters()).device)
            torch.distributed.broadcast(bl_tensor, src=0)
            self.best_loss = bl_tensor[0].item()
            if bl_tensor[1].item() > 0.5:
                control.should_training_stop = True

        delta_pct = (base_loss - self.best_loss) / self.best_loss * 100 if self.best_loss > 0 else 0
        if is_main:
            print(
                f"passo={state.global_step}: "
                f"atual={base_loss:.4f} (pool {len(self.pool)}/{self.soup_max}), "
                f"melhor={self.best_loss:.4f} de {self.best_source}, "
                f"delta={delta_pct:+.1f}%",
                flush=True,
            )

        _pre_stop = bool(control.should_training_stop)
        _rule_stop = False
        if self.std_active:
            return
        if not is_new_best and base_loss > self.best_loss * (1 + _OVERFIT_THRESHOLD):
            self.overfit_counter += 1
            print(
                f"Hmm ficando confortável demais... {delta_pct:+.1f}% fora "
                f"(contador={self.overfit_counter}/{_OVERFIT_PATIENCE})",
                flush=True,
            )

            if self.overfit_counter >= _OVERFIT_PATIENCE:
                print(
                    "Overfit confirmado (3x), parando cedo — melhor sera "
                    "restaurado e o dev-pass roda no fim",
                    flush=True,
                )
                control.should_training_stop = True
                _rule_stop = True
                run_state.STOP_REASON["reason"] = "overfit"
        elif not is_new_best:
            self.overfit_counter = 0

        if self.depth_first is not None and hasattr(self.depth_first, "turn_gate"):
            try:
                self.depth_first.turn_gate(args, state, model, float(metrics.get("eval_loss", base_loss)))
            except Exception:
                pass

        if (stop_rule_mode() == "vot" and self.stop_rule_facts is not None
                and traj2.members_on_world(int(self.stop_rule_facts.get("world_size", 1) or 1))
                and self._stop_rule_check(
                args, state, model, metrics, base_loss, is_new_best,
                    patience_fired=bool(control.should_training_stop))):
            control.should_training_stop = True
            _rule_stop = True
            run_state.STOP_REASON["reason"] = "overfit"

        if _rule_stop and not _pre_stop and std.enabled():
            self._stop_to_decay(args, state, control)

    def _stop_to_decay(self, args, state, control) -> bool:
        try:
            if self.decay_started is not None:
                return False
            if not std.single_process():
                return False
            sched = getattr(self.trainer, "lr_scheduler", None) if self.trainer is not None else None
            kw = std.wsd_keywords(sched)
            lam = tsc.lambda_scheduler(sched)
            lr_frac = float(lam.lr_lambdas[0](int(state.global_step))) if lam is not None else None
            stats = None
            if self.planner_clock is not None:
                stats = self.planner_clock.clock.stats()
            remaining = self._stop_rule_remaining()
            tail_secs = None
            if packed_fill.active():
                try:
                    tail_secs = traj2.measured_tail_secs(self)
                except Exception:
                    pass
            elif self.stop_rule_facts is not None:
                try:
                    _f = traj2.Attempt0Facts(**self.stop_rule_facts)
                    self._attach_clock(_f)
                    _tail = traj2.tail_reserves(
                        _f, devfit_enabled=self.stop_rule_devpass_reserve > 0,
                        devpass_legacy=self.stop_rule_devpass_reserve,
                        measured_devpass=self._measured_devpass(_f))
                    tail_secs = float(_tail.reserve) + float(_tail.devpass_reserve)
                except Exception:
                    pass
            hold = max(float(self.std_guard_secs or 0.0), float(tail_secs or 0.0))
            avail = None if remaining is None else float(remaining) - hold
            cads = [int(getattr(state, "eval_steps", 0) or getattr(args, "eval_steps", 0) or 0)]
            if self.stop_rule_cadences is not None:
                try:
                    cads += [int(c or 0) for c in self.stop_rule_cadences()]
                except Exception:
                    pass
            cads = [c for c in cads if c > 0]
            gate = 0
            if self.std_gate_step is not None:
                try:
                    gate = int(self.std_gate_step() or 0)
                except Exception:
                    pass
            h = std.plan_handoff(step=int(state.global_step), kw=kw, avail_secs=avail,
                                 t_step=(stats.t_step if stats is not None else None),
                                 t_point=(stats.t_point if stats is not None else None),
                                 eval_steps=cads, lr_frac=lr_frac, gate_step=gate)
            out = {"applied": False}
            if h.ok:
                out = std.apply_handoff(sched, state, h)
            ok = bool(h.ok and out.get("applied"))
            if not ok:
                return False
            self.claim_decay("stop_to_decay", int(state.global_step))
            if self.std_clock_cb is not None and hasattr(self.std_clock_cb, "horizon"):
                self.std_clock_cb.horizon = int(h.horizon)
            control.should_training_stop = False
            self.std_active = True
            self.std_handoff = h.model_dump()
            self.overfit_counter = 0
            if is_main_process(LOCAL_RANK):
                print(f"[sn56][plano] parada na fase plana (passo {h.step}): decaimento até "
                      f"{h.horizon} ({h.decay} passos{', encurtado' if h.reason else ''})",
                      flush=True)
            return True
        except Exception:
            return False

    def claim_decay(self, by: str, step: int) -> bool:
        if self.decay_started is not None:
            return False
        self.decay_started = {"by": str(by), "step": int(step)}
        return True

    _clock = staticmethod(time.time)

    def _stop_rule_remaining(self) -> Optional[float]:
        return traj2._remaining_secs(self.end_time)

    @staticmethod
    def _planner_n(facts, remaining: float, devpass_reserve: float,
                   measured_devpass: bool = False) -> tuple:
        if traj2._gate(facts):
            return 0, 0.0
        tail = traj2.tail_reserves(facts, devfit_enabled=devpass_reserve > 0,
                                   devpass_legacy=devpass_reserve,
                                   measured_devpass=bool(measured_devpass))
        d = traj2.plan(dataclasses.replace(facts), remaining, reserve=tail.reserve,
                    devpass_reserve=tail.devpass_reserve, remainder=False)
        return int(d.n_fit or 0), float(d.t_traj or 0.0)

    def _plan_n(self, facts, remaining: float, measured_devpass: bool) -> tuple:
        if measured_devpass and len(inspect.signature(self._planner_n).parameters) >= 4:
            return self._planner_n(facts, remaining, self.stop_rule_devpass_reserve, True)
        return self._planner_n(facts, remaining, self.stop_rule_devpass_reserve)

    def _measured_devpass(self, facts) -> bool:
        if not mp.stop_on() or not planner_clock.devpass_measured_on():
            return False
        if int(getattr(facts, "world_size", 1) or 1) != 1:
            return False
        return self.trainer is None or traj2.sft_objective(self.trainer)

    def _attach_clock(self, facts) -> None:
        if self.planner_clock is None:
            return
        try:
            facts.clock = self.planner_clock.stats_dict()
        except Exception:
            facts.clock = None

    def _eval_cadence(self) -> Optional[tuple]:
        ev = self._sr_evals
        gaps = [(ev[i + 1][0] - ev[i][0], ev[i + 1][1] - ev[i][1]) for i in range(len(ev) - 1)]
        gaps = [g for g in gaps if g[0] > 0]
        if not gaps:
            return None
        stride = max(g[0] for g in gaps)
        full = [g[1] for g in gaps if g[0] >= _FULL_STRIDE_FRAC * stride]
        return (float(statistics.median(full)), int(stride)) if full else None

    @staticmethod
    def _earliest_patience_secs(step: int, h: int, cadences, max_steps: int,
                                stride: int, c: float, t_eval: float) -> tuple:
        nxt = set()
        for cad in cadences:
            cad = int(cad or 0)
            if cad > 0:
                k = step // cad + 1
                nxt.update((k + j) * cad for j in range(h))
        grid = sorted(nxt)
        target = grid[h - 1] if len(grid) >= h else step + h * stride
        n_evals = h
        if max_steps and step < max_steps < target:
            target = max_steps
            n_evals = sum(1 for g in grid if g < max_steps) + 1
        t_step = max(0.0, c - t_eval) / stride
        return (target - step) * t_step + n_evals * t_eval, target

    def _stop_rule_inputs(self, args, state, metrics, base_loss) -> Optional[dict]:
        rel = base_loss / self.best_loss - 1.0 if self.best_loss > 0 else 0.0
        cadences = tuple(self.stop_rule_cadences() if self.stop_rule_cadences else ())
        spe = int(cadences[0] if cadences and (cadences[0] or 0) > 0 else
                  (self.stop_rule_facts or {}).get("steps_per_epoch") or 0)
        epoch = float(getattr(state, "epoch", 0.0) or 0.0)
        if spe <= 0 and epoch > 0:
            spe = int(round(state.global_step / epoch))
        sig = turn_signature(state.log_history, self.best_step, state.global_step, spe)
        info = {"step": state.global_step, "epoch": epoch, "dev": base_loss,
                "best": self.best_loss, "best_step": self.best_step, "rel": rel,
                "steps_per_epoch": spe, "overfit_counter": self.overfit_counter, **sig}
        _anyrel = bool(mp.stop_on() and mp.stop_v2_on() and mp.anyrel_on()
                       and int(state.global_step) > int(self.best_step or 0)
                       and rel >= mp.REL_FLOOR
                       and (sig["turn"] or mp.noturn_on()))
        info["mp_anyrel"] = _anyrel
        if not ((sig["turn"] and rel >= _TURN_DEV_MARGIN) or _anyrel):
            return info
        remaining = self._stop_rule_remaining()
        cad = self._eval_cadence()
        if self.stop_rule_facts is None or remaining is None or cad is None:
            info["skip"] = ("no_planner_facts" if self.stop_rule_facts is None else
                            "no_end_time" if remaining is None else "no_interval")
            return info
        c, stride = cad
        t_eval = (metrics or {}).get("eval_runtime")
        n_evals = sum(1 for e in state.log_history or () if "eval_loss" in e)
        facts = traj2.Attempt0Facts(**self.stop_rule_facts)
        facts.best_step = int(self.best_step)
        facts.best_loss = float(self.best_loss)
        facts.steps_run = int(state.global_step)
        facts.n_evals = n_evals
        facts.train_runtime = ((self._clock() - self._sr_t0)
                              if self._sr_t0 is not None else None)
        facts.eval_runtime = float(t_eval) if t_eval else facts.eval_runtime
        facts.horizon_steps = int(getattr(state, "max_steps", 0) or facts.horizon_steps or 0)
        facts.stop_reason = "overfit"
        self._attach_clock(facts)
        traj2.set_pool_facts(facts, self)
        t_eval = float(facts.eval_runtime or 0.0)
        t_a0_soup = mp.a0_soup_secs(len(self.pool), t_eval)
        _mdp = self._measured_devpass(facts)
        if rel >= _OVERFIT_THRESHOLD or not _anyrel:
            h = max(1, _OVERFIT_PATIENCE - self.overfit_counter)
        else:
            _ms = int(getattr(state, "max_steps", 0) or 0)
            h = max(1, -(-(max(0, _ms - int(state.global_step))) // max(1, stride)) + 2)
        t_p3, p3_step = self._earliest_patience_secs(
            int(state.global_step), h, (spe,) + cadences[1:] if cadences else (spe, stride),
            int(getattr(state, "max_steps", 0) or 0), stride, c, t_eval)
        r_now = remaining - t_a0_soup
        r_p3 = r_now - t_p3
        n_now, t_traj = self._plan_n(facts, r_now, _mdp)
        n_p3, _ = self._plan_n(facts, r_p3, _mdp)
        _tail = traj2.tail_reserves(facts, devfit_enabled=self.stop_rule_devpass_reserve > 0,
                                    devpass_legacy=self.stop_rule_devpass_reserve,
                                    measured_devpass=_mdp)
        info.update(mp_stop=mp.stop_on(), mp_price=mp.price_on(), mp_launch=mp.launch_on(),
                    measured_devpass=_mdp)
        if mp.stop_on():
            _pv = mp.pivotal(lambda r: self._plan_n(facts, r, _mdp)[0], r_now, r_p3,
                             _STOP_RULE_NCAP,
                             errs=mp.stop_errs(planner_clock.write_secs(facts.clock_stats)))
            if mp.stop_v2_on():
                _bstep = sig.get("boundary_step")
                _tep = (int(_bstep // spe) if (_bstep and spe > 0) else None)
                def _n_of(r, pool_loop=None):
                    f = facts
                    if pool_loop is not None and facts.pool_size_loop is not None:
                        f = dataclasses.replace(
                            facts, pool_size_loop=max(int(facts.pool_size_loop), int(pool_loop)))
                    return self._plan_n(f, r, _mdp)[0]

                _pv = mp.pivotal_v2(_n_of, r_now, r_p3, _STOP_RULE_NCAP,
                                    scenarios=(mp.LOOP_POOL_MIX if mp.loop_pool_mix_on()
                                               else ((1.0, None),)),
                                    errs=mp.stop_errs(planner_clock.write_secs(facts.clock_stats)),
                                    turn_epoch=_tep, h=h, rel=rel, turn=bool(sig["turn"]))
                info.update(mp_v2=True, p_extra=_pv.p_extra, v_late=_pv.v_late, rise=_pv.rise,
                            v_late_source="prior", turn_epoch=_tep, pool_n=len(self.pool), t_eval=t_eval,
                            write_secs=planner_clock.write_secs(facts.clock_stats))
            info.update(ev_now=_pv.ev_now, ev_p3=_pv.ev_p3, ev_freed=_pv.ev_freed,
                        ev_continue=_pv.ev_continue, mp_fire=_pv.stop)
        info.update(remaining=remaining, t_a0_soup=t_a0_soup, c=c, stride=stride, h=h,
                    p3_step_earliest=p3_step, t_to_p3=t_p3,
                    remaining_now=r_now, remaining_p3=r_p3, n_now=n_now, n_p3=n_p3,
                    t_traj=t_traj, devpass_reserve=self.stop_rule_devpass_reserve,
                    tail_reserve=_tail.reserve, tail_devpass=_tail.devpass_reserve,
                    tail_source=_tail.source,
                    clock_source=("measured" if facts.clock_stats is not None else "legacy"),
                    world_size=facts.world_size, gate=traj2._gate(facts))
        return info

    @staticmethod
    def _lr_frac(state, facts) -> Optional[float]:
        lrs = [e.get("learning_rate") for e in (state.log_history or ())
               if e.get("learning_rate") is not None and "eval_loss" not in e]
        lrs = [float(x) for x in lrs if x == x]
        if lrs and max(lrs) > 0:
            return lrs[-1] / max(lrs)
        sched = str((facts or {}).get("lr_scheduler_type", "")).lower()
        if "cosine" in sched:
            return traj2.cosine_lr_frac(int(state.global_step), int(getattr(state, "max_steps", 0) or 0),
                                        int((facts or {}).get("warmup_steps", 0) or 0),
                                        float((facts or {}).get("min_lr_rate", 0.0) or 0.0))
        return None

    def _relative_inputs(self, args, state, metrics, base_loss) -> Optional[dict]:
        if not turn_relative_enabled() or self.stop_rule_facts is None:
            return None
        cadences = tuple(self.stop_rule_cadences() if self.stop_rule_cadences else ())
        spe = int(cadences[0] if cadences and (cadences[0] or 0) > 0 else
                  (self.stop_rule_facts or {}).get("steps_per_epoch") or 0)
        cad = self._eval_cadence()
        stride = int((self.stop_rule_facts or {}).get("eval_steps") or 0) or (cad[1] if cad else 0)
        step = int(state.global_step)
        pool = [e for e in (self.pool or []) if e.get("src", "a0") == "a0"]
        in_pool = any(int(e.get("step", -1)) == step for e in pool)
        dip = boundary_dip(self._sr_devs, self.best_step, self.best_loss, spe)
        lr_frac = self._lr_frac(state, self.stop_rule_facts)
        turned, why = turned_relative(step, self.best_step, spe, stride, lr_frac, float(base_loss),
                                      [e.get("loss") for e in pool], in_pool, dip,
                                      pool_min=min(_REL_POOL_MIN, int(self.soup_max or _REL_POOL_MIN)))
        info = {"rel_turned": turned, "rel_why": why, "rel_lr_frac": lr_frac, "rel_dip": dip,
                "rel_pool_worst": (max(e["loss"] for e in pool) if pool else None),
                "rel_stride": stride}
        if not turned:
            return info
        remaining = self._stop_rule_remaining()
        if remaining is None or cad is None:
            info["rel_why"] = "no_end_time" if remaining is None else "no_interval"
            return info
        c, stride_c = cad
        t_eval = float((metrics or {}).get("eval_runtime") or 0.0)
        facts = traj2.Attempt0Facts(**self.stop_rule_facts)
        facts.best_step = int(self.best_step)
        facts.best_loss = float(self.best_loss)
        facts.steps_run = step
        facts.n_evals = sum(1 for e in state.log_history or () if "eval_loss" in e)
        facts.train_runtime = ((self._clock() - self._sr_t0) if self._sr_t0 is not None else None)
        facts.eval_runtime = t_eval or facts.eval_runtime
        facts.horizon_steps = int(getattr(state, "max_steps", 0) or facts.horizon_steps or 0)
        facts.stop_reason = "overfit"
        self._attach_clock(facts)
        traj2.set_pool_facts(facts, self)
        t_eval = float(facts.eval_runtime or 0.0)
        t_a0_soup = mp.a0_soup_secs(len(self.pool), t_eval)
        _mdp = self._measured_devpass(facts)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        rel = base_loss / self.best_loss - 1.0 if self.best_loss > 0 else 0.0
        if rel >= _OVERFIT_THRESHOLD:
            h = max(1, _OVERFIT_PATIENCE - self.overfit_counter)
            end_kind = "patience"
        else:
            h = max(1, -(-(max(0, max_steps - step)) // max(1, stride_c)) + 2)
            end_kind = "horizon"
        t_end, end_step = self._earliest_patience_secs(
            step, h, (spe,) + cadences[1:] if cadences else (spe, stride_c), max_steps,
            stride_c, c, t_eval)
        r_now = remaining - t_a0_soup
        r_end = r_now - t_end
        n_now, t_traj = self._plan_n(facts, r_now, _mdp)
        n_end, _ = self._plan_n(facts, r_end, _mdp)
        rel_fire = relative_trade(n_now, n_end)
        info.update(rel_end_kind=end_kind, rel_end_step=end_step, rel_t_to_end=t_end,
                    rel_n_now=n_now, rel_n_end=n_end, rel_remaining_now=r_now,
                    rel_t_traj=t_traj, rel_fire=rel_fire,
                    rel_clock_source=("measured" if facts.clock_stats is not None else "legacy"))
        return info

    def _late_probe_verdict(self, state, model, info: dict) -> None:
        try:
            if not (float(info.get("ev_freed") or 0.0) > float(info.get("ev_continue") or 0.0)):
                return
            if int(state.global_step) <= int(self.best_step or 0):
                return
            key = int(self.best_step or 0)
            cache = getattr(self, "_late_probe_cache", None)
            if cache is None:
                cache = self._late_probe_cache = {}
            if key not in cache:
                cache[key] = {"ran": False, "why": "error"}
                cache[key] = self._late_probe(state, model, info)
            pr = cache[key]
            info.update(probe={k: pr.get(k) for k in ("ran", "why", "gain", "secs", "est_secs",
                                                       "soup_loss", "best_loss", "step")})
            if not pr.get("ran"):
                return
            v = mp.v_late_probe(pr["gain"], int(info.get("h") or 0), int(info.get("pool_n") or 0))
            info.update(v_late_prior=info.get("v_late"), v_late=round(v, 7), v_late_source="probe",
                        mp_fire=bool(float(info["ev_freed"]) > v + float(info["ev_continue"])))
        except Exception:
            pass

    def _late_probe(self, state, model, info: dict) -> dict:
        est = mp.probe_cost_est(float(info.get("t_eval") or 0.0) or float(
            (state.log_history[-1] if state.log_history else {}).get("eval_runtime") or 0.0),
            float(info.get("write_secs") or 0.0))
        out = self.soup_pair_trial(state, model, est_secs=est,
                                   budget_secs=float(info.get("t_to_p3") or 0.0))
        return out

    def soup_trial_write_secs(self) -> float:
        try:
            if self.stop_rule_facts is None:
                return 0.0
            facts = traj2.Attempt0Facts(**self.stop_rule_facts)
            self._attach_clock(facts)
            return float(planner_clock.write_secs(facts.clock_stats) or 0.0)
        except Exception:
            return 0.0

    def soup_pair_trial(self, state, model, *, est_secs: float, budget_secs: float) -> dict:
        out = {"ran": False, "why": "", "gain": None, "secs": None, "step": int(state.global_step)}
        tr = self.trainer
        est = float(est_secs)
        out["est_secs"] = round(est, 2)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            out["why"] = "multi_gpu"
            return out
        if not is_main_process(LOCAL_RANK) or tr is None or model is None:
            out["why"] = "no_trainer"
            return out
        if self.best_state is None:
            out["why"] = "no_best_state"
            return out
        out["budget_secs"] = round(float(budget_secs), 2)
        if not est < float(budget_secs):
            out["why"] = "time"
            return out
        try:
            fits, _ri = self._stage_ram(model, "pass2")
        except Exception:
            fits = False
        if not fits:
            out["why"] = "host_ram"
            return out
        unwrapped = self._unwrap(model)
        named = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]
        t0 = time.perf_counter()
        was_training = bool(getattr(model, "training", True))
        saved = {k: getattr(tr, k) for k in list(vars(tr)) if k.startswith("last_eval")}
        orig_log, orig_on_eval = tr.log, tr.callback_handler.on_evaluate
        cur = None
        try:
            with torch.no_grad():
                cur = {n: p.data.detach().to("cpu", copy=True) for n, p in named}
            tr.log = lambda *a, **k: None
            tr.callback_handler.on_evaluate = lambda args, st, control, *a, **k: control
            loss = self._eval_uniform_soup(model, [self.best_state, cur])
        finally:
            tr.log, tr.callback_handler.on_evaluate = orig_log, orig_on_eval
            for k, v in saved.items():
                setattr(tr, k, v)
            if cur is not None:
                host_snapshot.weights_changed("ckpt_avg.late_probe_restore")
                with torch.no_grad():
                    for n, p in named:
                        p.data.copy_(cur[n].to(p.device))
                    _n0, _p0 = named[0]
                    _n1, _p1 = named[-1]
                    if not (torch.equal(_p0.data.cpu(), cur[_n0]) and torch.equal(_p1.data.cpu(), cur[_n1])):
                        raise RuntimeError("late_probe restore mismatch")
            if was_training:
                model.train()
        cur = None
        out.update(ran=True, why="measured", soup_loss=loss, best_loss=float(self.best_loss),
                   gain=(float(self.best_loss) - float(loss)) if loss == loss and loss != float("inf") else None,
                   secs=round(time.perf_counter() - t0, 2))
        if out["gain"] is None:
            out["ran"], out["why"] = False, "eval_failed"
        return out

    def _stop_rule_check(self, args, state, model, metrics, base_loss, is_new_best,
                         patience_fired: bool) -> bool:
        fire, why, info = False, "", None
        try:
            _held = bool(self.depth_first is not None and hasattr(self.depth_first, "holds_stop")
                         and self.depth_first.holds_stop(int(state.global_step)))
        except Exception:
            _held = False
        try:
            self._sr_evals.append((int(state.global_step), self._clock()))
            self._sr_devs.append((int(state.global_step), float(base_loss)))
            if not is_new_best and not patience_fired:
                info = self._stop_rule_inputs(args, state, metrics, base_loss)
                if info is not None and "n_now" in info:
                    fire, why = stop_rule_decide(info["rel"], info["turn"],
                                                 info["n_now"], info["n_p3"])
                    _priced = why in ("turn_time_stop", "not_pivotal") or (
                        info.get("mp_anyrel") and info.get("mp_v2"))
                    if "mp_fire" in info and _priced:
                        _turn_clause = why in ("turn_time_stop", "not_pivotal")
                        if info.get("mp_v2") and mp.late_probe_on() and not _held:
                            self._late_probe_verdict(state, model, info)
                        fire = bool(info["mp_fire"])
                        why = (("turn_time_stop" if fire else "not_pivotal") if _turn_clause
                               else ("priced_stop" if fire else "not_priced"))
                    if fire and int(state.global_step) <= int(self.best_step or 0):
                        fire, why = False, "before_best"
                elif info is not None:
                    why = info.get("skip") or ("no_turn_signature")
        except Exception as e:
            print(f"[sn56][stop_rule] falhou ({e}); só paciência", flush=True)
            fire, why, info = False, "error", None
        _v2_priced = bool(info is not None and info.get("mp_anyrel") and info.get("mp_v2")
                          and "mp_fire" in info)
        if (not fire and info is not None and not is_new_best and not patience_fired
                and not _v2_priced):
            try:
                rinfo = self._relative_inputs(args, state, metrics, base_loss)
                if rinfo is not None:
                    info.update(rinfo)
                    if rinfo.get("rel_fire"):
                        fire, why = True, "turn_relative_stop"
            except Exception as e:
                print(f"[sn56][stop_rule] relativa falhou ({e})", flush=True)
                info["rel_why"] = "error"

        if fire and self.decay_started is not None:
            fire, why = False, ("depth_first_decay" if self.decay_started.get("by") == "depth_first"
                                else "stop_to_decay_decay")
        elif (fire and self.depth_first is not None and hasattr(self.depth_first, "holds_stop")
              and self.depth_first.holds_stop(int(state.global_step))):
            fire, why = False, "depth_first_flat"

        if torch.distributed.is_initialized():
            t = torch.tensor([1.0 if fire else 0.0], device=next(model.parameters()).device)
            torch.distributed.broadcast(t, src=0)
            fire = bool(t.item() > 0.5)
            if fire and not why:
                why = "turn_time_stop"

        if fire and is_main_process(LOCAL_RANK):
            print(
                f"Virada de época com tempo em jogo: dev {base_loss:.4f} vs melhor "
                f"{self.best_loss:.4f}"
                + (f" ({info['rel']:+.1%}), membros {info['n_now']} agora vs "
                   f"{info['n_p3']} na paciência — parando já" if info and 'n_now' in info
                   else " (rank 0) — parando já"),
                flush=True,
            )
        return fire

    @sg.nonfatal("ckpt_avg.on_save")
    def on_save(self, args, state, control, model=None, **kwargs):
        if not is_main_process(LOCAL_RANK):
            return
        if self.best_state is None or "avg" not in self.best_source:
            return

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        async_save.drain("ckpt_avg.on_save")
        if not sg.is_usable_checkpoint(checkpoint_dir):
            return

        sg.poison_checkpoint(checkpoint_dir)
        current_state = self._snapshot(model)
        self._restore(model, self.best_state)
        try:
            unwrapped = self._unwrap(model)
            if hasattr(unwrapped, "save_pretrained"):
                unwrapped.save_pretrained(checkpoint_dir)
        finally:
            self._restore(model, current_state)
        sg.unpoison_checkpoint(checkpoint_dir)

        print(
            f"Salvei o checkpoint com os pesos misturados "
            f"(source={self.best_source})",
            flush=True,
        )

    @sg.nonfatal("ckpt_avg.on_train_end")
    def on_train_end(self, args, state, control, model=None, **kwargs):
        host_snapshot.end_share("train_end")
        if self.averaging_mode == "disk":
            async_save.drain("ckpt_avg.on_train_end.disk")
        if self.averaging_mode == "disk" and self.trainer is not None:
            try:
                self._disk_average_and_submit(model)
            except Exception as e:
                print(f"[sn56][disco] falhou ({e}), mantendo melhor", flush=True)
            print(
                f"Forma final: {self.best_source} "
                f"(nota={self.best_loss:.4f}, voltas={self.rollback_count})",
                flush=True,
            )
            return

        if self.averaging_mode == "ram" and self.trainer is not None:
            try:
                self._greedy_soup(model)
            except Exception as e:
                print(f"[sn56][soup] falhou ({e}), usando melhor único", flush=True)

        async_save.drain("ckpt_avg.on_train_end")
        if self.best_state is None and not torch.distributed.is_initialized():
            print("Nenhuma foto tirada", flush=True)
            return
        self._restore_best(model)

        if is_main_process(LOCAL_RANK) and getattr(self, "_submission_dir", None):
            _ok, _take, _own = sg.may_write_from_memory(args.output_dir, self.best_loss)
            if not _ok:
                _rec = _own.get("record") or {}
                print(f"[sn56][soup] não supera a submissão da tentativa {_rec.get('attempt')} "
                      f"(perda={_rec.get('loss')}), mantida", flush=True)
            else:
                try:
                    from dev_pass import _save_weights_only
                    _wrote = _save_weights_only(
                        self._unwrap(model), self._submission_dir,
                        lambda m: print(m, flush=True),
                    )
                    if _wrote and _take:
                        sg.write_submission_record_dict(
                            args.output_dir, loss=self.best_loss,
                            step=int(self.best_step or 0), attempt=_own["attempt"])
                except Exception as e:
                    print(f"[sn56][soup] persistência falhou ({e})", flush=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print(
            f"Forma final: {self.best_source} "
            f"(nota={self.best_loss:.4f}, voltas={self.rollback_count})",
            flush=True,
        )
