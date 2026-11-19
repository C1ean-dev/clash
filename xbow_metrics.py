"""Append local training progress as it becomes available."""

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from heapq import merge
import json
import os
from pathlib import Path
from uuid import uuid4


STOP_DEFERRED = ContextVar("stop_deferred", default=False)
CONTINUOUS_EVENTS = 1024


class TrainingStopped(KeyboardInterrupt):
    pass


def check_training_stop(*, at_boundary=False):
    requested = os.environ.get("XBOW_STOP_FILE")
    if (at_boundary or not STOP_DEFERRED.get()) and requested and Path(requested).is_file():
        raise TrainingStopped()


@contextmanager
def defer_training_stop(enabled):
    token = STOP_DEFERRED.set(enabled)
    try:
        yield
    finally:
        STOP_DEFERRED.reset(token)


def read_training_journal(path):
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                try:
                    entry = json.loads(line, parse_constant=lambda _: None)
                except ValueError:
                    continue
                if isinstance(entry, dict) and "attempt" in entry:
                    yield entry
    except (OSError, UnicodeError):
        return


def read_run_journal(output):
    return merge(
        read_training_journal(output / "progress.jsonl"),
        read_training_journal(output / "progress.live.jsonl"),
        key=lambda entry: entry.get("time", ""),
    )


@contextmanager
def training_journal(output, deck, *, continuous=False):
    output.mkdir(parents=True, exist_ok=True)
    attempt = uuid4().hex
    path = output / ("progress.live.jsonl" if continuous else "progress.jsonl")
    recent = deque(maxlen=CONTINUOUS_EVENTS) if continuous else None
    pending = 0

    def compact():
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text("".join(recent), encoding="utf-8")
            temporary.replace(path)
        except PermissionError:
            return False
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def encode(entry):
        return json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"

    if continuous:
        recent.extend(encode(entry) for entry in read_training_journal(path))
        if path.exists():
            compact()

    def report(event, **details):
        nonlocal pending
        entry = {
            "event": event, "attempt": attempt, "deck": deck,
            "time": datetime.now(timezone.utc).isoformat(), **details,
        }
        line = encode(entry)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
        if recent is not None:
            recent.append(line)
            pending += 1
            if pending >= 128 and compact():
                pending = 0

    report("started")
    try:
        yield report
    except KeyboardInterrupt:
        report("stopped")
        raise
    except Exception as exc:
        report("failed", message=str(exc))
        raise
    else:
        report("finished")
