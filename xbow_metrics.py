"""Append local training progress as it becomes available."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4


class TrainingStopped(KeyboardInterrupt):
    pass


def check_training_stop():
    requested = os.environ.get("XBOW_STOP_FILE")
    if requested and Path(requested).is_file():
        raise TrainingStopped()


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


@contextmanager
def training_journal(output, deck):
    output.mkdir(parents=True, exist_ok=True)
    attempt = uuid4().hex
    path = output / "progress.jsonl"

    def report(event, **details):
        entry = {
            "event": event, "attempt": attempt, "deck": deck,
            "time": datetime.now(timezone.utc).isoformat(), **details,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

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
