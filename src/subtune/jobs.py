"""Background jobs for the UI: each runs `python -m subtune ...` with output to a log file.

Running jobs as subprocesses keeps GPU memory and long network loops out of the web server,
and lets you cancel a job by stopping its process.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT, get_paths


@dataclass
class Job:
    name: str
    proc: subprocess.Popen
    log_path: Path
    started: dt.datetime

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def status(self) -> str:
        code = self.proc.poll()
        if code is None:
            return f"{self.name}: running (started {self.started:%H:%M:%S})"
        return f"{self.name}: {'finished' if code == 0 else f'FAILED (exit {code})'}"


class JobManager:
    """At most one job per slot ("data" for scans/downloads, "train" for training)."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}

    def start(self, slot: str, name: str, args: list[str]) -> Job:
        current = self.jobs.get(slot)
        if current and current.running:
            raise RuntimeError(f"A {slot} job is already running: {current.name}")
        logs = get_paths().logs
        log_path = logs / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{slot}.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT / "src")}
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"$ subtune {' '.join(args)}\n")
            f.flush()
            proc = subprocess.Popen([sys.executable, "-m", "subtune", *args], stdout=f, stderr=subprocess.STDOUT,
                                    cwd=PROJECT_ROOT, env=env)
        job = Job(name, proc, log_path, dt.datetime.now())
        self.jobs[slot] = job
        return job

    def stop(self, slot: str) -> str:
        job = self.jobs.get(slot)
        if not job or not job.running:
            return "Nothing running."
        job.proc.terminate()
        return f"Stopped {job.name}."

    def view(self, slot: str, lines: int = 60) -> tuple[str, str]:
        job = self.jobs.get(slot)
        if not job:
            return "No job yet.", ""
        try:
            text = job.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        tail = "\n".join(text.splitlines()[-lines:])
        return job.status(), tail
