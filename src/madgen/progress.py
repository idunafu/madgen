"""Live progress log: timestamped lines appended to a file, plus a heartbeat every
HEARTBEAT_SEC so a long silent step (model loading, VAD, one big batch) still shows the
process is alive.

    tail -f work/corpus.sqlite.log
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

HEARTBEAT_SEC = 20.0


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"


class Progress:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._file = None
        self._t0 = time.monotonic()
        self._stage = "starting"
        self._stage_t0 = self._t0
        self._done = 0.0
        self._total: float | None = None
        self._last_report: float | None = None
        self._reported_done: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)
        self._t0 = time.monotonic()
        self._stage, self._total, self._done = "starting", None, 0.0
        self._stage_t0 = self._t0
        self._last_report = self._reported_done = None
        self._write(f"=== {' '.join(sys.argv)}")
        print(f"progress log: {path}", file=sys.stderr, flush=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def close(self, status: str = "done") -> None:
        if self._file is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.log(f"finished: {status} (total {_fmt(time.monotonic() - self._t0)})")
        with self._lock:
            self._file.close()
            self._file = None

    def _write(self, line: str) -> None:
        with self._lock:
            if self._file is not None:
                self._file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")

    def _status(self) -> str:
        elapsed = time.monotonic() - self._stage_t0
        if self._total:
            pct = 100.0 * self._done / self._total
            eta = f", eta {_fmt(elapsed / self._done * (self._total - self._done))}" if self._done else ""
            done = f"{self._done:.0f}" if float(self._done).is_integer() else f"{self._done:.1f}"
            return (f"{self._stage}: {done}/{self._total:g} ({pct:.1f}%), "
                    f"stage {_fmt(elapsed)}{eta}, total {_fmt(time.monotonic() - self._t0)}")
        return f"{self._stage}: stage {_fmt(elapsed)}, total {_fmt(time.monotonic() - self._t0)}"

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SEC):
            with self._lock:
                self._write(f"alive | {self._status()}")

    def log(self, msg: str) -> None:
        self._write(msg)

    def stage(self, name: str, total: float | None = None) -> None:
        with self._lock:
            self._stage, self._total, self._done = name, total, 0.0
            self._stage_t0 = time.monotonic()
            self._last_report = self._reported_done = None
            self._write(f"stage | {self._status()}")

    def update(self, done: float) -> None:
        """Set the progress of the current stage (in the units of its total)."""
        with self._lock:
            self._done = done

    def report(self, done: float, *, terminal: bool = False) -> None:
        """Record the first, final, and at most one intermediate update every five seconds."""
        with self._lock:
            self._done = done
            now = time.monotonic()
            finished = self._total is not None and done >= self._total
            if self._last_report is None or now - self._last_report >= 5 or (
                finished and done != self._reported_done
            ):
                self._last_report, self._reported_done = now, done
                status = self._status()
                self._write(f"progress | {status}")
                if terminal:
                    print(status, file=sys.stderr, flush=True)

    @contextmanager
    def phase(self, name: str):
        """Temporarily show blocking work without losing the enclosing inference progress."""
        with self._lock:
            previous = (self._stage, self._total, self._done, self._stage_t0,
                        self._last_report, self._reported_done)
            self.stage(name)
        try:
            yield
        finally:
            with self._lock:
                (self._stage, self._total, self._done, self._stage_t0,
                 self._last_report, self._reported_done) = previous
                self._write(f"resume | {self._status()}")


progress = Progress()
