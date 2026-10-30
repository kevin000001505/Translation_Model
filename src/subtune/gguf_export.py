"""Export a fine-tuned model as a GGUF file for llama.cpp, Ollama and LM Studio.

Two routes, both using llama.cpp's own tools (pinned release, fetched and built on first use; needs
git, cmake and a C++ compiler; the Docker and Modal images have them prebuilt):

- Base model from a regular Hugging Face repo: merge the LoRA adapter into the bf16 base with PEFT,
  then ``convert_hf_to_gguf.py`` writes the GGUF (chat template embedded).
- Base model from a GGUF file (e.g. TranslateGemma): ``convert_lora_to_gguf.py`` turns the adapter
  into a GGUF LoRA, and ``llama-export-lora`` merges it into the *original* GGUF. That keeps the
  original file's tokenizer and metadata exactly, instead of rebuilding them from tokenizer.json.

Q4_K_M (and every type on the GGUF route) is produced from an F16/BF16 intermediate with
``llama-quantize``.
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

Log = Callable[[str], None]

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_TAG = "b11989"  # pinned so conversions are reproducible
LLAMA_CPP_TOOLS = ("llama-quantize", "llama-export-lora")

# UI choice -> type written directly by convert_hf_to_gguf.py, or None when llama-quantize makes it.
EXPORT_TYPES = {"Q4_K_M": None, "Q8_0": "q8_0", "F16": "f16", "BF16": "bf16"}


def llama_cpp_dir() -> Path:
    env = os.environ.get("SUBTUNE_LLAMA_CPP")
    if env:
        return Path(env)
    from .config import get_paths

    return get_paths().data / "tools" / f"llama.cpp-{LLAMA_CPP_TAG}"


def _run(cmd: list, log: Log, env: dict | None = None) -> None:
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        log(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} {Path(cmd[1]).name} failed (exit {proc.returncode})")


def ensure_llama_cpp(log: Log) -> Path:
    root = llama_cpp_dir()
    if (root / "convert_hf_to_gguf.py").exists():
        return root
    log(f"Fetching llama.cpp {LLAMA_CPP_TAG} (one time)...")
    tmp = root.with_name(root.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "-q", "--depth", "1", "--branch", LLAMA_CPP_TAG, LLAMA_CPP_REPO, tmp], log)
    tmp.rename(root)
    return root


def tool(root: Path, name: str, log: Log) -> Path:
    """Path to a llama.cpp binary, building the tools the first time one is needed."""
    path = root / "build" / "bin" / name
    if path.exists():
        return path
    log("Building llama.cpp tools (one time, a few minutes)...")
    _run(["cmake", "-S", root, "-B", root / "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
          "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF"], log)
    _run(["cmake", "--build", root / "build", "--target", *LLAMA_CPP_TOOLS, "-j", os.cpu_count() or 4], log)
    return path


def _python_env(root: Path) -> dict:
    # llama.cpp's converters need its own (newer) gguf-py, ahead of any installed gguf package.
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "gguf-py"), os.environ.get("PYTHONPATH", "")])}


def _merged_checkpoint(source: str, weights_dir: Path, work_dir: Path, trust_remote_code: bool, log: Log) -> Path:
    """Base model + LoRA adapter merged into one bf16 checkpoint (on CPU, to leave the GPU alone)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("Merging the adapter into the bf16 base model...")
    base = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, device_map={"": "cpu"},
                                                trust_remote_code=trust_remote_code)
    merged = PeftModel.from_pretrained(base, weights_dir).merge_and_unload()
    shutil.rmtree(work_dir, ignore_errors=True)
    merged.save_pretrained(work_dir)
    AutoTokenizer.from_pretrained(weights_dir).save_pretrained(work_dir)
    del base, merged
    gc.collect()
    return work_dir


def _from_hf(kind: str, root: Path, source: str, weights_dir: Path, is_adapter: bool, dest: Path, work: Path,
             trust_remote_code: bool, log: Log) -> None:
    direct = EXPORT_TYPES[kind]
    quantize = None if direct else tool(root, "llama-quantize", log)  # fail before the slow merge
    model_dir = _merged_checkpoint(source, weights_dir, work / "merged", trust_remote_code, log) if is_adapter \
        else weights_dir
    converted = dest if direct else work / "model-bf16.gguf"
    _run([sys.executable, root / "convert_hf_to_gguf.py", model_dir, "--outfile", converted,
          "--outtype", direct or "bf16"], log, _python_env(root))
    if quantize:
        _run([quantize, converted, dest, kind], log)


def _from_gguf(kind: str, root: Path, base_gguf: Path, converted_dir: str, weights_dir: Path, dest: Path,
               work: Path, log: Log) -> None:
    export_lora = tool(root, "llama-export-lora", log)
    quantize = tool(root, "llama-quantize", log)
    lora = work / "adapter-f16.gguf"
    _run([sys.executable, root / "convert_lora_to_gguf.py", "--base", converted_dir, "--outtype", "f16",
          "--outfile", lora, weights_dir], log, _python_env(root))
    merged = dest if kind == "F16" else work / "merged-f16.gguf"  # export-lora always writes F16
    _run([export_lora, "-m", base_gguf, "--lora", lora, "-o", merged], log)
    if kind != "F16":
        _run([quantize, merged, dest, kind], log)


def export_gguf(kind: str, source: str, weights_dir: Path, is_adapter: bool, out_dir: Path, work: Path,
                trust_remote_code: bool, log: Log, base_gguf: Path | None = None) -> Path:
    """Write ``out_dir/gguf/<tag>.<kind>.gguf`` and return its path.

    ``source`` is the Hugging Face checkpoint training started from (for a GGUF base: its converted
    folder); ``base_gguf`` is the original GGUF file when the base model came from one."""
    if kind not in EXPORT_TYPES:
        raise ValueError(f"Unknown GGUF type {kind}; choose one of {', '.join(EXPORT_TYPES)}")
    if base_gguf and not is_adapter:
        raise ValueError("GGUF export of a full fine-tune of a GGUF base model is not supported (its tokenizer "
                         "would have to be rebuilt); fine-tune it with LoRA or QLoRA instead.")
    root = ensure_llama_cpp(log)
    dest = out_dir / "gguf" / f"{out_dir.name}.{kind}.gguf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        if base_gguf:
            _from_gguf(kind, root, base_gguf, source, weights_dir, dest, work, log)
        else:
            _from_hf(kind, root, source, weights_dir, is_adapter, dest, work, trust_remote_code, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)  # merged checkpoints and intermediates are large
    log(f"GGUF written: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    return dest
