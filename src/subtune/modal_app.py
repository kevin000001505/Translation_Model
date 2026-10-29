"""Run a training job on a Modal cloud GPU (https://modal.com).

The run's dataset is uploaded to the ``subtune`` Modal volume, training runs in a container with
the chosen GPU, and the finished weights are downloaded back into ``outputs/models/<tag>``.
Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (and HF_TOKEN for gated models) in .env.
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
        "transformers>=4.56",
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
    timeout=24 * 3600,
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
    log(f"Starting Modal job on {cfg.gpu}. Container logs follow.")
    with modal.enable_output(), app.run():
        metrics = train_remote.with_options(gpu=cfg.gpu).remote(cfg.to_dict(), run_name, tag)
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
    log(f"Weights saved to {out_dir} (also kept in the Modal volume at {remote_root}).")
    return metrics
