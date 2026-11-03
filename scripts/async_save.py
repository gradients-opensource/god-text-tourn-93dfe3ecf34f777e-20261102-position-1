import copy
import dataclasses
import functools
import json
import os
import shutil
import queue
import threading
import time
import traceback
from typing import Callable, Optional

import torch
from transformers import TrainerCallback
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer import TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_callback import ExportableState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, rotate_checkpoints

import host_snapshot
import save_guard as sg

ENV = "SN56_ASYNC_SAVE"
ENV_INFLIGHT = "SN56_ASYNC_SAVE_INFLIGHT"
ENV_SCOPE = "SN56_ASYNC_SAVE_SCOPE"
SITE = "trainer._save_checkpoint"


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def max_inflight() -> int:
    try:
        return max(1, int(os.environ.get(ENV_INFLIGHT, "2")))
    except ValueError:
        return 2


def scope() -> str:
    v = os.environ.get(ENV_SCOPE, "needed").strip().lower()
    return v if v in ("needed", "all") else "needed"


def _eval_save_callback(trainer):
    for cb in trainer.callback_handler.callbacks:
        if hasattr(cb, "update_best_checkpoint") and hasattr(cb, "save_only") \
                and hasattr(cb, "submission_dir"):
            return cb
    return None


def active() -> Optional["AsyncCheckpointWriter"]:
    return sg.ASYNC_WRITER


def drain(site: str) -> None:
    w = sg.ASYNC_WRITER
    if w is not None:
        w.drain(site)


class _Job:
    __slots__ = ("fn", "kind", "step", "label", "t_enq", "done", "is_write", "group",
                 "supersedable", "cancelled", "started")

    def __init__(self, fn, kind, step, label, is_write, group=None, supersedable=False):
        self.fn, self.kind, self.step, self.label = fn, kind, step, label
        self.t_enq = time.time()
        self.done = threading.Event()
        self.is_write = is_write
        self.group = group
        self.supersedable = supersedable
        self.cancelled = False
        self.started = False


class AsyncCheckpointWriter:
    def __init__(self, trainer, *, submission_dir=None, base_model_path=None,
                 inflight: Optional[int] = None):
        self.trainer = trainer
        self.submission_dir = submission_dir
        self.base_model_path = base_model_path
        self.inflight = inflight or max_inflight()
        self._q: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self._writes: list = []
        self._pending: list = []
        self._actions: list = []
        self._lock = threading.Lock()
        self._closed = False
        self.stats = {"writes": 0, "write_secs": 0.0, "stall_secs": 0.0, "drain_secs": 0.0,
                      "failed": 0, "deferred": 0, "superseded": 0, "skipped": 0}
        self.write_secs: list = []
        self.avail_fn = None
        self._thread = threading.Thread(target=self._run, name="sn56-async-save", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            job = self._q.get()
            if job is None:
                return
            with self._lock:
                job.started = not job.cancelled
                if job in self._pending:
                    self._pending.remove(job)
            if job.cancelled:
                job.fn = None
                job.done.set()
                continue
            t0 = time.time()
            try:
                job.fn()
            except BaseException as exc:
                self._fail(job, exc)
            finally:
                if job.is_write:
                    self.stats["write_secs"] += time.time() - t0
                job.fn = None
                job.done.set()
                job = None

    def _fail(self, job, exc):
        self.stats["failed"] += 1
        try:
            action = sg._record(job.label, exc, step=job.step, control=None,
                                **({"async_save": True}))
        except Exception:
            action = "raise" if sg.is_unrecoverable(exc) else "continue"
        if action != "continue":
            with self._lock:
                self._actions.append((action, exc, job.label, job.step))

    def submit_write(self, fn: Callable[[], None], step: int, *, supersedable: bool = False) -> float:
        t0 = time.time()
        if supersedable:
            with self._lock:
                gone = {j.group for j in self._pending if j.is_write and j.supersedable}
                for j in self._pending:
                    if j.group in gone:
                        j.cancelled = True
                if gone:
                    self.stats["superseded"] += len(gone)
        while True:
            with self._lock:
                self._writes = [j for j in self._writes
                                if not j.done.is_set() and not j.cancelled]
                oldest = self._writes[0] if len(self._writes) >= self.inflight else None
            if oldest is None:
                break
            oldest.done.wait()
        job = _Job(fn, "write", step, SITE, True, group=step, supersedable=supersedable)
        with self._lock:
            self._writes.append(job)
            self._pending.append(job)
        self._q.put(job)
        return time.time() - t0

    def defer(self, fn: Callable[[], None], *, label: str, step: Optional[int] = None) -> None:
        self.stats["deferred"] += 1
        traced = fn

        def run():
            traced()
            site = sg._site(label)
            site.consecutive = 0
            site.n_ok += 1

        job = _Job(run, "deferred", step, label, False, group=step)
        with self._lock:
            self._pending.append(job)
        self._q.put(job)

    def _wait_queue(self) -> int:
        n = self._q.qsize()
        if self._thread.is_alive():
            marker = _Job(lambda: None, "marker", None, "drain", False)
            self._q.put(marker)
            marker.done.wait()
        return n

    def drain(self, site: str, *, raise_errors: bool = True) -> float:
        t0 = time.time()
        n = self._wait_queue()
        waited = time.time() - t0
        self.stats["drain_secs"] += waited
        self.surface(raise_errors=raise_errors, drained=True)
        return waited

    def surface(self, raise_errors: bool = True, drained: bool = False) -> None:
        with self._lock:
            actions, self._actions = self._actions, []
        if not actions:
            return
        if not drained and any(a[0] == "stop" for a in actions):
            self._wait_queue()
            with self._lock:
                actions += self._actions
                self._actions = []
        to_raise = None
        for action, exc, label, step in actions:
            if action == "stop":
                sg.request_stop(getattr(self.trainer, "control", None), label)
                if label == SITE:
                    sg.seed_submission(self.submission_dir, self.base_model_path)
            elif action == "raise" and to_raise is None:
                to_raise = exc
        if to_raise is not None and raise_errors:
            raise to_raise

    def close(self, raise_errors: bool = True) -> None:
        if self._closed:
            return
        try:
            self.drain("close", raise_errors=raise_errors)
        finally:
            self._closed = True
            self._q.put(None)
            self._thread.join(timeout=60)


def _eligible(trainer, avg_mode: str) -> str:
    args = trainer.args
    if not enabled():
        return "disabled_env"
    if avg_mode != "ram":
        return f"avg_mode_{avg_mode}"
    if getattr(trainer, "is_deepspeed_enabled", False) or getattr(trainer, "is_fsdp_enabled", False):
        return "sharded"
    if not getattr(args, "should_save", True):
        return "non_main_rank"
    if not getattr(args, "save_only_model", False):
        return "not_save_only_model"
    if getattr(args, "push_to_hub", False):
        return "push_to_hub"
    model = trainer.model
    if not isinstance(model, PreTrainedModel):
        return f"model_type_{type(model).__name__}"
    if getattr(model, "_hf_peft_config_loaded", False) or getattr(model, "hf_quantizer", None) is not None:
        return "peft_or_quantized"
    if getattr(model, "_tp_size", None) is not None:
        return "tensor_parallel"
    if any(p.device.type == "meta" for p in model.parameters()):
        return "offloaded"
    return ""


def _state_json(trainer) -> str:
    for cb in [cb for cb in trainer.callback_handler.callbacks + [trainer.control]
               if isinstance(cb, ExportableState)]:
        cb_name = cb.__class__.__name__
        cb_state = cb.state()
        if isinstance(trainer.state.stateful_callbacks[cb_name], list):
            trainer.state.stateful_callbacks[cb_name].append(cb_state)
        else:
            trainer.state.stateful_callbacks[cb_name] = cb_state
    return json.dumps(dataclasses.asdict(trainer.state), indent=2, sort_keys=True) + "\n"


def _world_size() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except Exception:
        pass
    return 1


def _stock_save_without_collectives(trainer, sync_save: Callable, model, trial):
    orig = trainer.__dict__.get("store_flos")
    trainer.store_flos = lambda: None
    try:
        return sync_save(model, trial)
    finally:
        if orig is None:
            del trainer.store_flos
        else:
            trainer.store_flos = orig


def make_async_save_checkpoint(trainer, writer: AsyncCheckpointWriter, sync_save: Callable):

    def _save_checkpoint(model, trial=None, *rest, **kwargs):
        writer.surface()
        if (trial is not None or rest or kwargs or trainer.hp_search_backend is not None
                or trainer.state.best_global_step or not trainer.args.should_save):
            return sync_save(model, trial, *rest, **kwargs)
        t0 = time.time()
        step = int(trainer.state.global_step)
        supersedable = False
        if scope() == "needed":
            cb = _eval_save_callback(trainer)
            if cb is not None and not cb.save_only:
                if not cb.update_best_checkpoint:
                    trainer.store_flos()
                    writer.stats["skipped"] += 1
                    return None
                supersedable = True
        run_dir = trainer._get_output_dir(trial=trial)
        out_dir = os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}")
        trainer.store_flos()

        def _inline_stock(reason: str, **info):
            writer.stats["inline_fallback"] = writer.stats.get("inline_fallback", 0) + 1
            return _stock_save_without_collectives(trainer, sync_save, model, trial)

        target = unwrap_model(trainer.model)
        try:
            dev_sd = target.state_dict()
            layout = host_snapshot.layout_of(dev_sd)
        except Exception as exc:
            writer.drain("layout_fallback")
            return _inline_stock("layout", exc=f"{type(exc).__name__}: {exc}"[:300])
        share_key = (target, step)
        shared = host_snapshot.shared_bytes_available(dev_sd, share_key)
        ok, adm = host_snapshot.writer_admit(layout, writer.avail_fn, shared=shared)
        if not ok:
            writer.drain("ram_admission")
            ok, adm = host_snapshot.writer_admit(layout, writer.avail_fn, shared=shared)
        if not ok:
            writer.stats["ram_fallback"] = writer.stats.get("ram_fallback", 0) + 1
            return _inline_stock("ram", **{k: v for k, v in adm.items() if k in ("need", "avail")})
        try:
            host_sd = host_snapshot.snapshot(dev_sd, consumer="async_save.writer",
                                             share_with=share_key)
        except Exception as exc:
            writer.drain("snapshot_fallback")
            return _inline_stock("snapshot", exc=f"{type(exc).__name__}: {exc}"[:300])
        state_json = _state_json(trainer)
        args_copy = copy.deepcopy(trainer.args)
        save_total_limit = trainer.args.save_total_limit
        best_ckpt = trainer.state.best_model_checkpoint
        processing = trainer.processing_class
        if processing is None:
            processing = getattr(getattr(trainer, "data_collator", None), "tokenizer", None)

        def _write():
            tw = time.time()
            try:
                os.makedirs(out_dir, exist_ok=True)
                target.save_pretrained(out_dir, state_dict=host_sd)
                if processing is not None:
                    processing.save_pretrained(out_dir)
                torch.save(args_copy, os.path.join(out_dir, TRAINING_ARGS_NAME))
                with open(os.path.join(out_dir, TRAINER_STATE_NAME), "w", encoding="utf-8") as fh:
                    fh.write(state_json)
                rotate_checkpoints(output_dir=run_dir, save_total_limit=save_total_limit,
                                   best_model_checkpoint=best_ckpt, use_mtime=True)
                host_snapshot.drop_written_cache(out_dir, site="async_save")
            except BaseException:
                sg.FAILED_STEPS.add(step)
                sg.quarantine_checkpoint(out_dir)
                raise
            site = sg._site(SITE)
            site.consecutive = 0
            site.n_ok += 1
            sg.FAILED_STEPS.discard(step)
            sg.mark_checkpoint_complete(out_dir)
            writer.stats["writes"] += 1
            _ws = time.time() - tw
            writer.write_secs.append(_ws)

        wait = writer.submit_write(_write, step, supersedable=supersedable)
        stall = time.time() - t0
        writer.stats["stall_secs"] += stall
        return None

    return _save_checkpoint


class AsyncSaveCallback(TrainerCallback):

    def __init__(self, writer: AsyncCheckpointWriter):
        self.writer = writer

    def on_step_end(self, args, state, control, **kwargs):
        if self.writer._actions:
            self.writer.surface()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.drain("on_train_end")


def install(trainer, *, avg_mode: str, submission_dir=None, base_model_path=None) -> Optional[AsyncCheckpointWriter]:
    why = _eligible(trainer, avg_mode)
    if why:
        return None
    writer = AsyncCheckpointWriter(trainer, submission_dir=submission_dir,
                                   base_model_path=base_model_path)
    sync_save = trainer._save_checkpoint
    async_save = make_async_save_checkpoint(trainer, writer, sync_save)
    orig_train = trainer.train
    cb = AsyncSaveCallback(writer)

    @functools.wraps(orig_train)
    def train(*a, **k):
        host_snapshot.end_share("async_save.train_begin")
        sg.ASYNC_WRITER = writer
        trainer._save_checkpoint = async_save
        trainer.add_callback(cb)
        ok = False
        try:
            out = orig_train(*a, **k)
            ok = True
            return out
        finally:
            try:
                writer.close(raise_errors=ok)
            finally:
                host_snapshot.end_share("async_save.train_end")
                sg.ASYNC_WRITER = None
                trainer._save_checkpoint = sync_save
                trainer.train = orig_train
                try:
                    trainer.remove_callback(cb)
                except Exception:
                    pass
                host_snapshot.ram_budget()

    trainer.train = train
    return writer


ENV_INTERIM = "SN56_INTERIM_SHIP"
INTERIM_SUFFIX = ".interim"
INTERIM_DISK_MARGIN = 5 * 1024 ** 3


def interim_enabled(world_size: int) -> bool:
    v = os.environ.get(ENV_INTERIM, "multi").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on", "always"):
        return True
    return int(world_size or 1) > 1


def _meta_skeleton(target):
    cfg = copy.deepcopy(target.config)
    with torch.device("meta"):
        skel = type(target)(cfg)
    gen = getattr(target, "generation_config", None)
    if gen is not None:
        skel.generation_config = copy.deepcopy(gen)
    return skel


class InterimWriter:
    def __init__(self, submission_dir: str, *, avail_fn=None, disk_free_fn=None):
        self.submission_dir = submission_dir
        self.path = submission_dir.rstrip("/") + INTERIM_SUFFIX
        self._tmp = self.path + ".tmp"
        self.avail_fn = avail_fn
        self.disk_free_fn = disk_free_fn or (lambda d: shutil.disk_usage(d).free)
        self._cv = threading.Condition()
        self._pending = None
        self._busy = False
        self._closed = False
        self._discard = False
        self.stats = {"offered": 0, "queued": 0, "written": 0, "superseded": 0, "skipped": 0,
                      "failed": 0, "stall_secs": 0.0, "write_secs": 0.0}
        self._thread = threading.Thread(target=self._run, name="sn56-interim", daemon=True)
        self._thread.start()

    def _skip(self, reason: str, **kw) -> str:
        self.stats["skipped"] += 1
        return reason

    def offer(self, model, *, loss: float, label: str, n_members: int) -> str:
        self.stats["offered"] += 1
        t0 = time.time()
        if self._closed:
            return self._skip("closed")
        target = unwrap_model(model)
        if not isinstance(target, PreTrainedModel) or getattr(target, "_hf_peft_config_loaded", False) \
                or getattr(target, "hf_quantizer", None) is not None:
            return self._skip("model_type", model=type(target).__name__)
        try:
            dev_sd = target.state_dict()
            layout = host_snapshot.layout_of(dev_sd)
        except Exception as exc:
            return self._skip("layout", exc=f"{type(exc).__name__}: {exc}"[:300])
        need = int(sum(layout))
        try:
            parent = os.path.dirname(self.path.rstrip("/")) or "."
            free = int(self.disk_free_fn(parent))
            if free - INTERIM_DISK_MARGIN < 2 * need:
                return self._skip("disk", free=free, need=need)
        except Exception as exc:
            return self._skip("disk_probe", exc=f"{type(exc).__name__}: {exc}"[:300])
        ok, adm = host_snapshot.writer_admit(layout, self.avail_fn)
        if not ok:
            with self._cv:
                dropped = self._pending is not None
                self._pending = None
            if dropped:
                self.stats["superseded"] += 1
            ok, adm = host_snapshot.writer_admit(layout, self.avail_fn)
        if not ok:
            return self._skip("ram", **adm)
        try:
            skel = _meta_skeleton(target)
            host_sd = host_snapshot.snapshot(dev_sd, consumer="interim")
        except Exception as exc:
            return self._skip("snapshot", exc=f"{type(exc).__name__}: {exc}"[:300])
        job = {"sd": host_sd, "skel": skel, "loss": float(loss), "label": str(label),
               "n_members": int(n_members), "t": time.time(), "nbytes": need}
        with self._cv:
            if self._pending is not None:
                self.stats["superseded"] += 1
            self._pending = job
            self._cv.notify_all()
        stall = time.time() - t0
        self.stats["queued"] += 1
        self.stats["stall_secs"] += stall
        return "queued"

    def _write(self, job) -> None:
        tw = time.time()
        if os.path.exists(self._tmp):
            shutil.rmtree(self._tmp, ignore_errors=True)
        os.makedirs(self._tmp, exist_ok=True)
        job["skel"].save_pretrained(self._tmp, state_dict=job["sd"], safe_serialization=True)
        for f in os.listdir(self.submission_dir):
            src = os.path.join(self.submission_dir, f)
            if os.path.isfile(src) and not _is_weight_name(f):
                shutil.copy2(src, os.path.join(self._tmp, f))
        host_snapshot.drop_written_cache(self._tmp, site="interim")
        old = self.path + ".old"
        if os.path.exists(old):
            shutil.rmtree(old, ignore_errors=True)
        if os.path.isdir(self.path):
            os.rename(self.path, old)
        os.rename(self._tmp, self.path)
        shutil.rmtree(old, ignore_errors=True)
        secs = time.time() - tw
        with self._cv:
            if self._discard:
                return
            sg.ship_marker_update(interim={"path": self.path, "loss": job["loss"],
                                           "label": job["label"], "n_members": job["n_members"],
                                           "t": time.time()})
        self.stats["written"] += 1
        self.stats["write_secs"] += secs

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._closed:
                    self._cv.wait()
                if self._pending is None and self._closed:
                    break
                job, self._pending = self._pending, None
                self._busy = True
            try:
                self._write(job)
            except BaseException:
                self.stats["failed"] += 1
                shutil.rmtree(self._tmp, ignore_errors=True)
            finally:
                job = None
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()
        self._finish()

    def _finish(self) -> None:
        if self._discard:
            for d in (self._tmp, self.path, self.path + ".old"):
                shutil.rmtree(d, ignore_errors=True)

    def close(self, *, discard: bool = True, wait: Optional[float] = None) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._discard = bool(discard)
            dropped = self._pending is not None
            self._pending = None
            if discard:
                sg.ship_marker_update(interim=None)
            self._cv.notify_all()
        if dropped:
            self.stats["superseded"] += 1
        if wait:
            self._thread.join(timeout=wait)


def _is_weight_name(name: str) -> bool:
    return (name.endswith(".safetensors") or name == "model.safetensors.index.json"
            or name.endswith(".bin"))
