"""Shared advisory lock so CPU-heavy eval/diversity work on this machine
doesn't run concurrently and starve itself of cores.

Motivation (2026-09-29): a `bench.eval` pass that normally takes ~110s took
over 3x longer under an artificial 12-core CPU stress test, directly
demonstrating that this project's execution sandboxes (wall-clock
``subprocess.run(timeout=...)``) are sensitive to machine-wide CPU
contention -- not just to timeout values being too short. Any script doing
bulk pass@k evaluation (``bench.eval``'s CLI) or bulk diversity computation
(self-CodeBLEU, AST tree-edit-distance via ``zss`` in ``fingerprint.py``)
should acquire this lock before starting its CPU-heavy work, so the two
classes of job serialize instead of racing for the same cores.

Deliberately blocking, not refuse-and-exit (contrast with
``scripts/auto_sync_eval_lcb_v6.py``'s own lock, which refuses a second
concurrent instance outright): the point here is to make concurrent heavy
jobs wait their turn, not to prevent a second one from ever running.

**Scope, as of 2026-09-29 (caught by independent review — narrower than the
module's own original claim, do not assume this is project-wide)**: this is
currently wired into ``bench/eval/__main__.py`` ONLY. Five other scripts call
``evaluate_all``/``evaluate_all_apps``/``evaluate_all_lcb`` directly without
going through that CLI, and do **not** acquire this lock yet:
``bench/eval/entropy_history.py``, ``eval_full_precision_humaneval.py``,
``eval_full_mbpp.py``, ``consolidated_eval.py``, ``eval_temperature_sweep.py``.
Running any of those concurrently with a locked job (or with each other)
still races for CPU exactly as before this module existed. Wiring those five
in is a real, not-yet-done follow-up, not a "someday" — do not treat this
lock as project-wide protection until they're covered.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

LOCK_FILE = Path("/tmp/pless_cpu_heavy.lock")


@contextlib.contextmanager
def cpu_heavy_lock(label: str = "", warn_after: float = 60.0):
    """Block until exclusive access to the shared CPU-heavy-work lock is
    acquired, then yield; release on exit (including on exception).

    No hard timeout — a CPU-heavy job may legitimately run for hours (this
    project's own eval sweeps do) and forcing a wait to give up would just
    trade "blocked" for "silently racing," the exact problem this lock
    exists to prevent. Instead, a non-blocking attempt is tried first; if it
    fails, a warning is printed every ``warn_after`` seconds while waiting
    (naming whoever currently holds the lock, read from the lock file) so a
    long wait is visible and diagnosable rather than a silent hang."""
    lock_fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR)
    try:
        waited = 0.0
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
            if waited == 0.0:
                holder = LOCK_FILE.read_text().strip() if LOCK_FILE.exists() else "(unknown)"
                print(f"cpu_heavy_lock: waiting for {LOCK_FILE} (held by: {holder})...", flush=True)
            time.sleep(1.0)
            waited += 1.0
            if waited > 0 and waited % warn_after == 0:
                holder = LOCK_FILE.read_text().strip() if LOCK_FILE.exists() else "(unknown)"
                print(f"cpu_heavy_lock: still waiting after {waited:.0f}s (held by: {holder})...",
                      flush=True)
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, f"{os.getpid()} {label}\n".encode())
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
