"""Curated base models offered in the UI, and training defaults per method."""

from __future__ import annotations

from dataclasses import dataclass

# Every curated model is ungated on Hugging Face: it downloads without a token or an access request.
# GGUF models (llama.cpp format) are converted once to a regular checkpoint before training (gguf.py).
#
# Prompt families (see prompts.py):
#   translategemma  TranslateGemma's chat template: structured content with language codes
#   hunyuan_mt      chat template, Hunyuan-MT's own instruction wording
#   hy_mt2          chat template, Hy-MT2's own instruction wording
#   seedx           raw prompt ending in a target-language tag, no chat template
#   chat            any chat model: generic instruction through its chat template
#   plain           base model without a chat template: "Source: ...\nTranslation:"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    family: str
    params_b: float
    trust_remote_code: bool = False
    note: str = ""
    gguf_file: str = ""  # set for GGUF repos: which .gguf file in the repo to use


_GEMMA_NOTE = ("Ungated GGUF copy by mradermacher; Google's original repo is gated. The Gemma Terms of Use "
               "still apply to the fine-tuned model. Converting the GGUF needs about {ram} GB of RAM once; "
               "the non-GGUF TranslateGemma entry is the same model without that step.")
_GEMMA_ST_NOTE = ("Original bf16 weights (ungated copy by Infomaniak-AI; Google's own repo is gated). "
                  "The Gemma Terms of Use apply to the fine-tuned model.")

MODELS: list[ModelSpec] = [
    ModelSpec("Infomaniak-AI/vllm-translategemma-4b-it", "TranslateGemma 4B", "translategemma", 4.3,
              note=_GEMMA_ST_NOTE),
    ModelSpec("Infomaniak-AI/vllm-translategemma-12b-it", "TranslateGemma 12B", "translategemma", 12.2,
              note=_GEMMA_ST_NOTE),
    ModelSpec("Infomaniak-AI/vllm-translategemma-27b-it", "TranslateGemma 27B", "translategemma", 27.4,
              note=_GEMMA_ST_NOTE),
    ModelSpec("mradermacher/translategemma-4b-it-GGUF", "TranslateGemma 4B (GGUF f16)", "translategemma", 3.9,
              gguf_file="translategemma-4b-it.f16.gguf", note=_GEMMA_NOTE.format(ram=16)),
    ModelSpec("mradermacher/translategemma-12b-it-GGUF", "TranslateGemma 12B (GGUF Q8_0)", "translategemma", 11.8,
              gguf_file="translategemma-12b-it.Q8_0.gguf", note=_GEMMA_NOTE.format(ram=40)),
    ModelSpec("mradermacher/translategemma-27b-it-GGUF", "TranslateGemma 27B (GGUF Q8_0)", "translategemma", 27.0,
              gguf_file="translategemma-27b-it.Q8_0.gguf", note=_GEMMA_NOTE.format(ram=80)),
    ModelSpec("tencent/Hy-MT2-7B", "Hy-MT2 7B", "hy_mt2", 7.5, note="Apache-2.0. Needs transformers 5.6 or newer."),
    ModelSpec("tencent/Hunyuan-MT-7B", "Hunyuan-MT 7B", "hunyuan_mt", 7.5,
              note="Tencent Hunyuan Community License (check its territory restrictions)."),
    ModelSpec("ByteDance-Seed/Seed-X-PPO-7B", "Seed-X PPO 7B", "seedx", 7.5,
              note="OpenMDW license. No chat template; prompts end with a target-language tag."),
    ModelSpec("Qwen/Qwen3-4B", "Qwen3 4B", "chat", 4.0),
    ModelSpec("Qwen/Qwen3-8B", "Qwen3 8B", "chat", 8.2),
    ModelSpec("Qwen/Qwen3-14B", "Qwen3 14B", "chat", 14.8),
]

CUSTOM_LABEL = "Custom Hugging Face model…"
METHODS = ("LoRA", "QLoRA", "Full")
TARGETS = ("Local (AMD R9700, ROCm)", "Modal cloud GPU")
MODAL_GPUS = ("L40S", "A100-40GB", "A100-80GB", "H100", "H200", "B200")
LOCAL_VRAM_GB = 32  # AMD Radeon AI PRO R9700


def by_label(label: str) -> ModelSpec | None:
    return next((m for m in MODELS if m.label == label), None)


def by_id(model_id: str) -> ModelSpec | None:
    return next((m for m in MODELS if m.model_id == model_id), None)


def check_downloadable(model_id: str) -> tuple[bool, str]:
    """A model is allowed only if anyone can download it: public and not gated on Hugging Face."""
    from huggingface_hub import model_info
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if "/" not in model_id:
        return False, f"'{model_id}' is not a Hugging Face model id (expected org/name)."
    try:
        info = model_info(model_id, token=False)
    except GatedRepoError:
        return False, f"{model_id} is gated (needs an access request); pick an ungated model."
    except RepositoryNotFoundError:
        return False, f"{model_id} does not exist or is private."
    except Exception as exc:  # network errors
        return False, f"Could not check {model_id} on Hugging Face: {exc}"
    if info.gated:
        return False, f"{model_id} is gated (needs an access request); pick an ungated model."
    return True, f"{model_id} is public and ungated."


def gguf_files(model_id: str) -> list[str]:
    """Model .gguf files in a repo (vision projector files left out); [] for a regular repo."""
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(model_id, token=False)
    except Exception:
        return []
    return sorted(f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower())


def custom_spec(model_id: str, gguf_file: str = "") -> ModelSpec:
    # family "auto" resolves to translategemma, chat or plain once the tokenizer is loaded.
    return ModelSpec(model_id, model_id, "auto", 0.0, gguf_file=gguf_file)


def defaults(method: str, params_b: float) -> dict:
    """Starting hyperparameters. Subtitle lines are short, so sequences stay small."""
    big = params_b >= 20
    return {
        "epochs": 1,
        "learning_rate": 2e-4 if method != "Full" else 1e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "batch_size": 8 if not big else 4,
        "grad_accum": 4 if not big else 8,
        "max_seq_len": 256,
        "eval_sentences": 200,
    }


def estimate_vram_gb(method: str, params_b: float) -> float:
    """Rough peak memory for training with gradient checkpointing (weights + optimizer + slack)."""
    if params_b <= 0:
        return 0.0
    if method == "QLoRA":
        return params_b * 0.6 + 4
    if method == "LoRA":
        return params_b * 2.1 + 5
    return params_b * 16 + 6  # full fine-tune, AdamW: bf16 weights + fp32 master/moments


def suggest_gpu(method: str, params_b: float) -> str:
    need = estimate_vram_gb(method, params_b) or 40
    for gpu, mem in (("L40S", 48), ("A100-80GB", 80), ("H200", 141), ("B200", 180)):
        if need <= mem * 0.9:
            return gpu
    return "B200"
