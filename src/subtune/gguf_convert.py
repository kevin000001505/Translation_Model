"""Turn a GGUF model (llama.cpp format) into a regular checkpoint that can be fine-tuned.

Transformers reads GGUF files for the common architectures (Llama, Mistral, Qwen2/3, Gemma 2/3, ...).
The weights are dequantized to bf16 once and saved; later runs reuse the converted folder. The
GGUF's chat template travels with the tokenizer, so prompts match the original model.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable

import torch

from .progress import dir_size_gb, heartbeat


def cache_dir() -> Path:
    env = os.environ.get("SUBTUNE_GGUF_CACHE")
    if env:
        return Path(env)
    from .config import get_paths

    return get_paths().data / "gguf-converted"


def converted_path(repo: str, gguf_file: str) -> Path:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{repo}__{gguf_file.removesuffix('.gguf')}")
    return cache_dir() / name


def download_gguf(repo: str, gguf_file: str, log: Callable[[str], None] = print) -> Path:
    """Download (or find in the Hugging Face cache) one GGUF file, logging its size and progress."""
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE

    token = os.environ.get("HF_TOKEN") or None
    cached = try_to_load_from_cache(repo, gguf_file)
    if isinstance(cached, str):
        return Path(cached)
    try:
        size = HfApi().get_paths_info(repo, [gguf_file], token=token)[0].size / 1e9
        log(f"Downloading {gguf_file} ({size:.1f} GB) from {repo}; this is the slowest step of a first run.")
    except Exception:
        size = 0.0
        log(f"Downloading {gguf_file} from {repo}.")
    repo_cache = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
    total = f" of {size:.1f}" if size else ""
    with heartbeat(log, "downloading", detail=lambda: f"{dir_size_gb(repo_cache):.1f}{total} GB on disk"):
        return Path(hf_hub_download(repo, gguf_file, token=token))


def ensure_converted(repo: str, gguf_file: str, log: Callable[[str], None] = print) -> Path:
    """Return a local folder with the dequantized model, converting it on first use."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, GgufConfig

    out = converted_path(repo, gguf_file)
    if (out / "config.json").exists():
        log(f"Using converted GGUF at {out}")
        return out
    token = os.environ.get("HF_TOKEN") or None
    gguf_path = download_gguf(repo, gguf_file, log)
    log("Converting the GGUF to a bf16 checkpoint (one time; needs RAM for the whole model).")
    with heartbeat(log, "converting the GGUF"):
        tokenizer = AutoTokenizer.from_pretrained(repo, gguf_file=gguf_file, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            repo, gguf_file=gguf_file, token=token, dtype=torch.bfloat16,
            quantization_config=GgufConfig(gguf_file=gguf_file, dequantize=True),
        )
    from .gemma3 import fix_rope, gguf_rope_scaling

    if fix_rope(model.config, gguf_rope_scaling(gguf_path)):
        log("Restored the GGUF's linear RoPE scaling, which the conversion drops for Gemma 3.")
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    tmp.rename(out)  # only a complete conversion is ever picked up
    log(f"Converted {type(model).__name__} saved to {out}")
    del model
    return out
