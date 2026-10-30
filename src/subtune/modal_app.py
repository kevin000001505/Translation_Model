"""Run a training job on a Modal cloud GPU (https://modal.com).

The run's dataset is uploaded to the ``subtune`` Modal volume, training runs in a container with
the chosen GPU, and the finished weights are downloaded back into ``outputs/models/<tag>``.
Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in .env.

Billing stops when the job ends. The job runs in an *ephemeral* app (``app.run()``), which Modal
stops as soon as this script exits, fails, or loses its connection. On top of that:

- an idle container is released after Modal's default ``scaledown_window`` (60 s);
- ``timeout`` (the run's ``max_hours``) is a hard cap: Modal kills the container at that point
  even if the local script is gone;
- ``retries`` is left at 0, so a crash never starts a second paid attempt.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import modal

from .gguf_export import LLAMA_CPP_REPO, LLAMA_CPP_TAG, LLAMA_CPP_TOOLS

APP_NAME = "subtune"
VOLUME_ROOT = "/vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("subtune", create_if_missing=True)
hf_cache = modal.Volume.from_name("subtune-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # llama.cpp for exporting fine-tuned models to GGUF (converters + tools), built once per image.
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        f"git clone -q --depth 1 --branch {LLAMA_CPP_TAG} {LLAMA_CPP_REPO} /opt/llama.cpp",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF "
        "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF",
        f"cmake --build /opt/llama.cpp/build --target {' '.join(LLAMA_CPP_TOOLS)} -j 8",
    )
    .pip_install(
        "torch==2.8.0",
        "transformers>=5.6",
        "peft>=0.13",
        "accelerate>=1.0",
        "bitsandbytes>=0.45",
        "sacrebleu>=2.4",
        "sentencepiece",
        "protobuf",
        "gguf",
        "python-dotenv",
        "numpy",
    )
    # Converted GGUF models live on the cache volume, so a GGUF is converted only once.
    .env({"HF_HOME": "/hf-cache", "HF_HUB_ENABLE_HF_TRANSFER": "0", "SUBTUNE_GGUF_CACHE": "/hf-cache/gguf-converted",
          "SUBTUNE_LLAMA_CPP": "/opt/llama.cpp"})
    .add_local_python_source("subtune")
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=8 * 3600,  # replaced per run with the run's max_hours
    volumes={VOLUME_ROOT: volume, "/hf-cache": hf_cache},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN") or ""})],
)
def train_remote(cfg_dict: dict, run_name: str, tag: str, log_queue=None) -> dict:
    from subtune.runconfig import TrainConfig
    from subtune.train import run_training

    def log(message: str) -> None:
        # The training log goes back to the local job through the queue (shown in the app's
        # Training log); without one, it goes to Modal's own log stream.
        if log_queue is None:
            print(message, flush=True)
            return
        try:
            log_queue.put(message)
        except Exception:
            print(message, flush=True)

    run_dir = Path(VOLUME_ROOT) / "runs" / run_name
    out_dir = Path(VOLUME_ROOT) / "models" / tag
    metrics = run_training(TrainConfig.from_dict(cfg_dict), run_dir, out_dir, log)
    volume.commit()
    hf_cache.commit()
    return metrics


def _memory_mib(cfg) -> int | None:
    """RAM to reserve when a run holds the whole bf16 model in CPU memory: converting a GGUF base
    model, or merging the adapter for a GGUF export. Other runs keep Modal's default (memory is
    billed by the amount reserved)."""
    if not (cfg.gguf_file or cfg.gguf_export):
        return None
    from .models import by_id

    spec = by_id(cfg.model_id)
    params_b = spec.params_b if spec else 14
    return int((params_b * 3 + 8) * 1024)


class _relay:
    """Copy log lines from the cloud job's queue into the local log while the job runs, and say
    so when the cloud side has been quiet for a while (image build, container start, downloads)."""

    QUIET_SECONDS = 120

    def __init__(self, queue, log: Callable[[str], None]):
        self.queue, self.log = queue, log
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.start = self.last = time.time()
        self.seen = False

    def _drain(self, block: bool) -> None:
        for line in self.queue.get_many(100, block=block, timeout=5 if block else None):
            self.log(str(line))
            self.last, self.seen = time.time(), True

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                self._drain(block=True)
            except Exception as exc:  # keep relaying; a transient queue error must not end the job
                self.log(f"(could not read cloud log: {exc})")
                self.stop.wait(5)
            quiet = time.time() - self.last
            if quiet >= self.QUIET_SECONDS:
                state = "running" if self.seen else "starting (building the image / waiting for a GPU)"
                self.log(f"... cloud job {state}, no new log for {quiet / 60:.0f} min "
                         f"({(time.time() - self.start) / 60:.0f} min in total)")
                self.last = time.time()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=15)
        try:
            self._drain(block=False)  # lines sent just before the job returned
        except Exception:
            pass
        return False


def run_on_modal(cfg, run_dir: Path, tag: str, out_dir: Path, log: Callable[[str], None]) -> dict:
    run_name = run_dir.name
    log(f"Uploading dataset to Modal volume 'subtune' (runs/{run_name}).")
    with volume.batch_upload(force=True) as batch:
        for name in ("train.jsonl", "eval.jsonl"):
            batch.put_file(str(run_dir / name), f"/runs/{run_name}/{name}")
    timeout = max(60, min(int(cfg.max_hours * 3600), 24 * 3600))  # Modal allows up to 24 h
    log(f"Starting Modal job on {cfg.gpu}, hard time limit {timeout / 3600:.1f} h. Container logs follow.")
    try:
        # enable_output streams image builds, warnings and tracebacks; the training log itself
        # comes back through the queue, so it shows up in the job log line by line.
        with modal.enable_output(), app.run(), modal.Queue.ephemeral() as log_queue:
            try:
                log(f"Modal dashboard for this run: {app.get_dashboard_url()}")
            except Exception:
                pass
            fn = train_remote.with_options(gpu=cfg.gpu, timeout=timeout, memory=_memory_mib(cfg))
            with _relay(log_queue, log):
                metrics = fn.remote(cfg.to_dict(), run_name, tag, log_queue)
    finally:
        # Leaving app.run() stops the ephemeral app and every container in it, also on errors and on
        # Stop (SIGTERM is turned into SystemExit by the CLI), so no GPU keeps running.
        log("Modal app stopped; the cloud GPU is released.")
    log("Downloading weights from Modal.")
    remote_root = f"models/{tag}"
    for entry in volume.listdir(remote_root, recursive=True):
        if getattr(entry.type, "name", "") != "FILE":
            continue
        rel = Path(entry.path.lstrip("/")).relative_to(remote_root)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
    # The local copy is now the one that counts: remove this run from the volume. The Hugging Face
    # cache volume is kept so the next run does not download the base model again.
    for path in (remote_root, f"runs/{run_name}"):
        volume.remove_file(path, recursive=True)
    log(f"Weights saved to {out_dir}; run files removed from the Modal volume.")
    return metrics
