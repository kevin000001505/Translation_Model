"""Run a training job on a Modal cloud GPU (https://modal.com).

The run's dataset is uploaded to the ``subtune`` Modal volume, training runs in a container with
the chosen GPU, and the finished weights are downloaded back into ``outputs/models/<tag>``.
Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in .env.

Billing stops when the job ends. The job runs in an *ephemeral* app (``app.run()``), which Modal
stops as soon as this script exits, fails, or loses its connection. On top of that:

- ``scaledown_window=2`` releases the GPU container 2 s after the function returns (Modal's
  default keeps an idle container, and bills it, for up to 60 s);
- ``timeout`` (the run's ``max_hours``) is a hard cap: Modal kills the container at that point
  even if the local script is gone;
- ``retries`` is left at 0, so a crash never starts a second paid attempt.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import modal

APP_NAME = "subtune"
VOLUME_ROOT = "/vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("subtune", create_if_missing=True)
hf_cache = modal.Volume.from_name("subtune-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers>=5.6",
        "peft>=0.13",
        "accelerate>=1.0",
        "bitsandbytes>=0.45",
        "sacrebleu>=2.4",
        "sentencepiece",
        "protobuf",
        "python-dotenv",
        "numpy",
    )
    .env({"HF_HOME": "/hf-cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("subtune")
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=8 * 3600,  # replaced per run with the run's max_hours
    scaledown_window=2,  # the minimum Modal allows: free the GPU right after the job returns
    volumes={VOLUME_ROOT: volume, "/hf-cache": hf_cache},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN") or ""})],
)
def train_remote(cfg_dict: dict, run_name: str, tag: str) -> dict:
    from subtune.runconfig import TrainConfig
    from subtune.train import run_training

    run_dir = Path(VOLUME_ROOT) / "runs" / run_name
    out_dir = Path(VOLUME_ROOT) / "models" / tag
    metrics = run_training(TrainConfig.from_dict(cfg_dict), run_dir, out_dir, lambda m: print(m, flush=True))
    volume.commit()
    return metrics


def run_on_modal(cfg, run_dir: Path, tag: str, out_dir: Path, log: Callable[[str], None]) -> dict:
    run_name = run_dir.name
    log(f"Uploading dataset to Modal volume 'subtune' (runs/{run_name}).")
    with volume.batch_upload(force=True) as batch:
        for name in ("train.jsonl", "eval.jsonl"):
            batch.put_file(str(run_dir / name), f"/runs/{run_name}/{name}")
    timeout = max(60, min(int(cfg.max_hours * 3600), 24 * 3600))  # Modal allows up to 24 h
    log(f"Starting Modal job on {cfg.gpu}, hard time limit {timeout / 3600:.1f} h. Container logs follow.")
    try:
        with modal.enable_output(), app.run():
            metrics = train_remote.with_options(gpu=cfg.gpu, timeout=timeout).remote(cfg.to_dict(), run_name, tag)
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
