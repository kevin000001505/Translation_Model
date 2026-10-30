"""Restore Gemma 3's RoPE scaling where our model sources lose it.

Gemma 3 4B/12B/27B (and TranslateGemma) scale RoPE linearly by 8 on their full-attention layers;
the sliding-window layers are unscaled. Transformers drops that setting in two cases we hit:

- GGUF conversion: its gemma3 GGUF reader ignores ``rope.scaling.*`` keys;
- vLLM-style copies (e.g. Infomaniak-AI/vllm-translategemma-*), whose flattened
  ``rope_parameters: {factor, rope_type}`` Transformers 5 does not apply to any layer type.

Short subtitle lines translate about the same either way, but long inputs and anything exported to
GGUF (llama.cpp reads ``rope_parameters["full_attention"]``) need the real setting.
"""

from __future__ import annotations


def text_config(config):
    return getattr(config, "text_config", None) or config


def fix_rope(config, scaling: dict | None = None) -> bool:
    """Put linear RoPE scaling back on the full-attention layers of a Gemma 3 config, from a
    flattened entry in the config itself or from `scaling` ({"rope_type", "factor"}, e.g. read from a
    GGUF). Returns True if the config was changed."""
    text = text_config(config)
    if getattr(text, "model_type", "") != "gemma3_text":
        return False
    params = dict(getattr(text, "rope_parameters", None) or {})
    full = params.get("full_attention")
    if not isinstance(full, dict):
        return False
    if params.get("rope_type") not in (None, "default") and params.get("factor"):
        scaling = {"rope_type": params.pop("rope_type"), "factor": float(params.pop("factor"))}
    if not scaling or (full.get("rope_type") == scaling["rope_type"] and full.get("factor") == scaling["factor"]):
        return False
    params["full_attention"] = {**full, **scaling}
    text.rope_parameters = params
    return True


def gguf_rope_scaling(gguf_path) -> dict | None:
    """{"rope_type", "factor"} from a GGUF file's metadata, or None if it has no RoPE scaling."""
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))

    def value(key):
        field = reader.fields.get(key)
        if field is None:
            return None
        raw = field.parts[field.data[0]]
        return bytes(raw).decode() if field.types and field.types[0] == gguf.GGUFValueType.STRING else raw[0]

    arch = value("general.architecture")
    kind, factor = value(f"{arch}.rope.scaling.type"), value(f"{arch}.rope.scaling.factor")
    if not kind or kind == "none" or not factor:
        return None
    return {"rope_type": kind, "factor": float(factor)}
