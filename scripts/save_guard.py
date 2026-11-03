

import errno
import functools
import json
import os
import shutil
import threading
import time
import traceback

import torch

import host_snapshot


MAX_CONSECUTIVE_FAILURES = 3

COMPLETE_MARKER = ".sn56_save_complete"
_END_OF_WRITE_FILES = (COMPLETE_MARKER, "trainer_state.json")

POISON_MARKER = ".sn56_save_failed"

FAILED_STEPS = set()

STOP_REASON = None

ASYNC_WRITER = None

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")

_FATAL_ERRNOS = frozenset(
    v for v in (
        getattr(errno, "ENOSPC", None),
        getattr(errno, "EROFS", None),
        getattr(errno, "EDQUOT", None),
        getattr(errno, "EFBIG", None),
    ) if v is not None
)

_FATAL_SUBSTRINGS = (
    "cuda error",
    "cuda out of memory",
    "out of memory",
    "nccl",
    "device-side assert",
    "no space left on device",
    "read-only file system",
    "disk quota exceeded",
)


def is_unrecoverable(exc) -> bool:
    
    if isinstance(exc, (MemoryError, SystemError)):
        return True
    if isinstance(exc, OSError) and exc.errno in _FATAL_ERRNOS:
        return True
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return any(s in msg for s in _FATAL_SUBSTRINGS)


def is_usable_checkpoint(path) -> bool:
    
    if not path or not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, POISON_MARKER)):
        return False
    return any(os.path.exists(os.path.join(path, f)) for f in _END_OF_WRITE_FILES)


def mark_checkpoint_complete(path) -> None:
    
    try:
        if os.path.isdir(path):
            with open(os.path.join(path, COMPLETE_MARKER), "w") as fh:
                fh.write(str(time.time()))
    except OSError:
        pass


def poison_checkpoint(path) -> None:
    
    try:
        if os.path.isdir(path):
            with open(os.path.join(path, POISON_MARKER), "w") as fh:
                fh.write(str(time.time()))
    except OSError:
        pass


def unpoison_checkpoint(path) -> None:
    try:
        os.remove(os.path.join(path, POISON_MARKER))
    except OSError:
        pass


def quarantine_checkpoint(path):
    
    if not path or not os.path.isdir(path):
        return {"existed": False}
    listing = {}
    try:
        for name in sorted(os.listdir(path))[:64]:
            fp = os.path.join(path, name)
            listing[name] = os.path.getsize(fp) if os.path.isfile(fp) else -1
    except OSError:
        pass
    if is_usable_checkpoint(path):
        return {"existed": True, "removed": False, "complete": True, "files": listing}
    try:
        shutil.rmtree(path)
    except OSError:
        poison_checkpoint(path)
        return {"existed": True, "removed": False, "complete": False,
                "poisoned": True, "files": listing}
    return {"existed": True, "removed": True, "complete": False, "files": listing}


def has_weights(directory) -> bool:
    
    try:
        return any(n.endswith(_WEIGHT_SUFFIXES) for n in os.listdir(directory))
    except OSError:
        return False


SUBMISSION_RECORD_FILE = "submission_record.json"


def stage_and_swap(src, dest, extra_files=None) -> None:
    
    base = dest.rstrip("/")
    staging, backup = base + ".incoming", base + ".previous"
    for d in (staging, backup):
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    try:
        shutil.copytree(src, staging, copy_function=host_snapshot.copy_file_nocache)
        for name, content in (extra_files or {}).items():
            with open(os.path.join(staging, name), "w") as fh:
                fh.write(content)
        if os.path.isdir(dest):
            os.rename(dest, backup)
        os.rename(staging, dest)
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        if not os.path.isdir(dest) and os.path.isdir(backup):
            try:
                os.rename(backup, dest)
            except OSError:
                pass
        for d in (staging, backup):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        raise


def seed_submission(submission_dir, base_model_path) -> bool:
    
    if not submission_dir or not base_model_path:
        return False
    try:
        if not os.path.isdir(base_model_path):
            return False
        if os.path.isdir(submission_dir) and has_weights(submission_dir):
            return False
        os.makedirs(submission_dir, exist_ok=True)
        copied = 0
        for name in os.listdir(base_model_path):
            src = os.path.join(base_model_path, name)
            if os.path.isfile(src) and not name.endswith(_WEIGHT_SUFFIXES):
                shutil.copy2(src, os.path.join(submission_dir, name))
                copied += 1
        return copied > 0
    except OSError:
        return False


class _Site:
    __slots__ = ("name", "consecutive", "n_failed", "n_ok")

    def __init__(self, name):
        self.name = name
        self.consecutive = 0
        self.n_failed = 0
        self.n_ok = 0


_SITES = {}
_GIVE_UP = {}


AGREED_STOP = {"pending": False, "why": ""}


def _world_size() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except Exception:
        pass
    return 1


def request_stop(control, why: str = "") -> None:
    if _world_size() > 1:
        AGREED_STOP["pending"] = True
        AGREED_STOP["why"] = AGREED_STOP["why"] or why
        return
    if control is not None:
        try:
            control.should_training_stop = True
        except AttributeError:
            pass


def take_stop_request() -> bool:
    v = bool(AGREED_STOP["pending"])
    AGREED_STOP["pending"] = False
    return v


def _site(name) -> _Site:
    site = _SITES.get(name)
    if site is None:
        site = _SITES[name] = _Site(name)
    return site


def save_failed_at(step) -> bool:
    
    return step in FAILED_STEPS


def reset():
    
    _SITES.clear()
    _GIVE_UP.clear()
    FAILED_STEPS.clear()


def _record(name, exc, step=None, control=None, **extra):
    
    site = _site(name)
    site.n_failed += 1
    site.consecutive += 1
    fatal = is_unrecoverable(exc)
    giving_up = (not fatal) and site.consecutive >= MAX_CONSECUTIVE_FAILURES
    if giving_up:
        _GIVE_UP[name] = True
        request_stop(control, name)
        if STOP_REASON is not None and not STOP_REASON.get("reason"):
            STOP_REASON["reason"] = "save_failures"
    if fatal:
        return "raise"
    return "stop" if giving_up else "continue"


def _find(args, kwargs, attr):
    for value in list(kwargs.values()) + list(args):
        if hasattr(value, attr):
            return value
    return None


def nonfatal(name):
    
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                out = fn(*args, **kwargs)
            except Exception as exc:
                control = _find(args, kwargs, "should_training_stop")
                state = _find(args, kwargs, "global_step")
                action = _record(name, exc, step=getattr(state, "global_step", None),
                                 control=control)
                if action == "raise":
                    raise
                return None
            _site(name).consecutive = 0
            _site(name).n_ok += 1
            return out
        return wrapper
    return deco


def install_save_guard(trainer, stop_reason=None, submission_dir=None,
                       base_model_path=None):
    
    global STOP_REASON
    if stop_reason is not None:
        STOP_REASON = stop_reason
    original = trainer._save_checkpoint
    name = "trainer._save_checkpoint"

    @functools.wraps(original)
    def guarded(model, trial=None, *rest, **kwargs):
        step = getattr(getattr(trainer, "state", None), "global_step", None)
        output_dir = getattr(getattr(trainer, "args", None), "output_dir", None)
        ckpt_dir = (os.path.join(output_dir, f"checkpoint-{step}")
                    if output_dir and step is not None else None)
        t0 = time.time()
        try:
            out = original(model, trial, *rest, **kwargs)
        except Exception as exc:
            if step is not None:
                FAILED_STEPS.add(step)
            action = _record(
                name, exc, step=step,
                control=getattr(trainer, "control", None),
                secs=round(time.time() - t0, 3),
                checkpoint=quarantine_checkpoint(ckpt_dir),
                )
            if action == "raise":
                raise
            if action == "stop":
                seed_submission(submission_dir, base_model_path)
            return None
        site = _site(name)
        site.consecutive = 0
        site.n_ok += 1
        FAILED_STEPS.discard(step)
        mark_checkpoint_complete(ckpt_dir)
        return out

    trainer._save_checkpoint = guarded
    return guarded


SHIP_MARKER = "shippable.json"
_SUBMISSION_BACKUPS = (".previous", ".devfit_old", ".pre_interim")
_MARKER_LOCK = threading.Lock()
_MARKER = {"path": None, "state": None}


def ship_marker_path(output_dir) -> str:
    return os.path.join(output_dir, SHIP_MARKER)


def _write_json_atomic(path, obj) -> None:
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def ship_marker_clear(output_dir) -> None:
    with _MARKER_LOCK:
        _MARKER["path"], _MARKER["state"] = None, None
        try:
            os.remove(ship_marker_path(output_dir))
        except FileNotFoundError:
            pass
        except OSError:
            pass


def ship_marker_init(output_dir, submission_dir, *, shipped, shipped_loss=None, stage="a0") -> bool:
    if not output_dir or not submission_dir or not has_weights(submission_dir):
        return False
    state = {"version": 1, "stage": stage, "shipped": shipped,
             "shipped_loss": (float(shipped_loss) if shipped_loss is not None
                              and shipped_loss == shipped_loss else None),
             "submission_dir": submission_dir, "activity": stage, "interim": None,
             "t": time.time(), "pid": os.getpid()}
    with _MARKER_LOCK:
        _MARKER["path"], _MARKER["state"] = ship_marker_path(output_dir), state
        try:
            _write_json_atomic(_MARKER["path"], state)
        except OSError:
            _MARKER["path"], _MARKER["state"] = None, None
            return False
    return True


def ship_marker_update(**fields) -> None:
    with _MARKER_LOCK:
        if _MARKER["path"] is None:
            return
        _MARKER["state"].update(fields)
        _MARKER["state"]["t"] = time.time()
        try:
            _write_json_atomic(_MARKER["path"], _MARKER["state"])
        except OSError:
            pass


def ship_marker_state() -> dict:
    with _MARKER_LOCK:
        return dict(_MARKER["state"] or {})


def ship_marker_read(output_dir):
    try:
        with open(ship_marker_path(output_dir)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def repair_submission(submission_dir) -> str:
    if has_weights(submission_dir):
        return "ok"
    base = submission_dir.rstrip("/")
    for suffix in _SUBMISSION_BACKUPS:
        cand = base + suffix
        if has_weights(cand):
            try:
                if os.path.isdir(submission_dir):
                    shutil.rmtree(submission_dir, ignore_errors=True)
                os.rename(cand, submission_dir)
                return f"restored:{suffix}"
            except OSError:
                pass
    return "missing"


def install_interim(submission_dir, interim) -> bool:
    if not has_weights(interim):
        return False
    base = submission_dir.rstrip("/")
    backup = base + ".pre_interim"
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)
    try:
        if os.path.isdir(submission_dir):
            os.rename(submission_dir, backup)
        os.rename(interim, submission_dir)
        return True
    except OSError:
        if not os.path.isdir(submission_dir) and os.path.isdir(backup):
            os.rename(backup, submission_dir)
        return False


def _persisted_candidates(m: dict) -> list:
    out = []
    for it in [m.get("interim")] + list(m.get("persisted") or []):
        if not isinstance(it, dict):
            continue
        try:
            loss = float(it.get("loss"))
        except (TypeError, ValueError):
            continue
        if loss == loss and it.get("path"):
            out.append(dict(it, loss=loss))
    out.sort(key=lambda e: e["loss"])
    return out


def recover_after_child_failure(output_dir) -> dict:
    m = ship_marker_read(output_dir)
    if not m:
        return {"shippable": False, "reason": "no_marker"}
    sub = m.get("submission_dir") or ""
    info = {"shippable": False, "stage": m.get("stage"), "activity": m.get("activity"),
            "shipped": m.get("shipped"), "shipped_loss": m.get("shipped_loss"),
            "submission_dir": sub, "interim_used": False}
    info["repair"] = repair_submission(sub) if sub else "no_submission_dir"
    base_loss = m.get("shipped_loss")
    cands = _persisted_candidates(m)
    info["n_persisted"] = len(cands)
    if base_loss is not None and sub:
        for it in cands:
            if not (it["loss"] < float(base_loss)):
                break
            if not has_weights(it["path"]):
                continue
            if install_interim(sub, it["path"]):
                info.update(interim_used=True, shipped=it.get("label") or "interim",
                            shipped_loss=float(it["loss"]), interim_members=it.get("n_members"),
                            recovered_from=it["path"])
                break
    info["shippable"] = bool(sub) and has_weights(sub)
    if not info["shippable"]:
        info["reason"] = "submission_invalid"
    return info


def _attempt_now() -> int:
    try:
        return int(os.environ.get("SN56_ATTEMPT", "0"))
    except ValueError:
        return 0


def read_submission_record_dict(output_dir):
    try:
        with open(os.path.join(output_dir, SUBMISSION_RECORD_FILE)) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def write_submission_record_dict(output_dir, *, loss, step, attempt) -> None:
    path = os.path.join(output_dir, SUBMISSION_RECORD_FILE)
    try:
        _write_json_atomic(path, {"loss": (None if loss is None else float(loss)),
                                  "step": int(step), "attempt": int(attempt)})
    except OSError:
        pass


def submission_owner(output_dir, attempt=None) -> dict:
    cur = _attempt_now() if attempt is None else int(attempt)
    rec = read_submission_record_dict(output_dir) if output_dir else None
    owns = rec is None or int(rec.get("attempt", -1)) == cur
    return {"owns": bool(owns), "record": rec, "attempt": cur}


def may_write_from_memory(output_dir, loss, attempt=None) -> tuple:
    own = submission_owner(output_dir, attempt)
    if own["owns"]:
        return True, False, own
    rec = own["record"] or {}
    prior = rec.get("loss")
    if loss is not None and (prior is None or float(loss) < float(prior)):
        return True, True, own
    return False, False, own
