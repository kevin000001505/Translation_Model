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


def cache_dir() -> Path:
    env = os.environ.get("SUBTUNE_GGUF_CACHE")
    if env:
        return Path(env)
    from .config import get_paths

    return get_paths().data / "gguf-converted"


def converted_path(repo: str, gguf_file: str) -> Path:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{repo}__{gguf_file.removesuffix('.gguf')}")
    return cache_dir() / name


def ensure_converted(repo: str, gguf_file: str, log: Callable[[str], None] = print) -> Path:
    """Return a local folder with the dequantized model, converting it on first use."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, GgufConfig

    out = converted_path(repo, gguf_file)
    if (out / "config.json").exists():
        log(f"Using converted GGUF at {out}")
        return out
    log(f"Converting {repo}/{gguf_file} to a bf16 checkpoint (one time; needs RAM for the full model)...")
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(repo, gguf_file=gguf_file, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        repo, gguf_file=gguf_file, token=token, dtype=torch.bfloat16,
        quantization_config=GgufConfig(gguf_file=gguf_file, dequantize=True),
    )
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    tmp.rename(out)  # only a complete conversion is ever picked up
    log(f"Converted {type(model).__name__} saved to {out}")
    del model
    return out
