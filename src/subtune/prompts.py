"""Turn a (source, target) sentence pair into model input ids, per prompt family.

Training and evaluation share these functions, so the fine-tuned model sees the same prompt
it will be scored with.
"""

from __future__ import annotations

from .languages import HUNYUAN_CHINESE_NAMES, base_code, iso_code, prompt_name


def resolve_family(family: str, tokenizer) -> str:
    if family != "auto":
        return family
    return "chat" if getattr(tokenizer, "chat_template", None) else "plain"


def _messages(family: str, src_lang: str, tgt_lang: str, text: str) -> list[dict]:
    if family == "translategemma":
        content = [{"type": "text", "source_lang_code": iso_code(src_lang),
                    "target_lang_code": iso_code(tgt_lang), "text": text}]
        return [{"role": "user", "content": content}]
    if family == "hunyuan_mt":
        if "zh" in (base_code(src_lang), base_code(tgt_lang)) and tgt_lang in HUNYUAN_CHINESE_NAMES:
            instruction = f"把下面的文本翻译成{HUNYUAN_CHINESE_NAMES[tgt_lang]}，不要额外解释。"
        else:
            instruction = f"Translate the following segment into {prompt_name(tgt_lang)}, without additional explanation."
        return [{"role": "user", "content": f"{instruction}\n\n{text}"}]
    instruction = (f"Translate the following {prompt_name(src_lang)} text into {prompt_name(tgt_lang)}. "
                   "Output only the translation.")
    return [{"role": "user", "content": f"{instruction}\n\n{text}"}]


def _raw_prompt(family: str, src_lang: str, tgt_lang: str, text: str) -> str:
    if family == "seedx":
        # Seed-X model card: "Translate the following English sentence into Chinese:\n<text> <zh>"
        return (f"Translate the following {prompt_name(src_lang)} sentence into {prompt_name(tgt_lang)}:\n"
                f"{text} <{base_code(tgt_lang)}>")
    return f"{prompt_name(src_lang)}: {text}\n{prompt_name(tgt_lang)}:"


def _template(tokenizer, messages, **kw) -> str:
    # Qwen3 thinks before answering unless told not to; other templates ignore the flag.
    return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False, **kw)


def _end_marker(tokenizer) -> str:
    vocab = tokenizer.get_vocab()
    for tok in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        if tok in vocab:
            return tok
    return tokenizer.eos_token or ""


def build_texts(tokenizer, family: str, src_lang: str, tgt_lang: str, src: str, tgt: str | None = None) -> tuple[str, str]:
    """Return (prompt text, completion text). Completion is "" when tgt is None (inference)."""
    if family in ("seedx", "plain"):
        prompt = _raw_prompt(family, src_lang, tgt_lang, src)
        completion = "" if tgt is None else (" " if family == "plain" else "") + tgt + (tokenizer.eos_token or "")
        return prompt, completion
    messages = _messages(family, src_lang, tgt_lang, src)
    prompt = _template(tokenizer, messages, add_generation_prompt=True)
    if tgt is None:
        return prompt, ""
    try:
        full = _template(tokenizer, messages + [{"role": "assistant", "content": tgt}])
    except Exception:  # template refuses an assistant turn
        full = ""
    if full.startswith(prompt) and tgt in full[len(prompt):]:
        completion = full[len(prompt):]
    else:
        completion = tgt + _end_marker(tokenizer)
    return prompt, completion


def encode(tokenizer, family: str, src_lang: str, tgt_lang: str, src: str,
           tgt: str | None = None) -> tuple[list[int], list[int]]:
    """Token ids for prompt and completion. Raw prompts get the tokenizer's BOS; chat templates
    already contain their special tokens, so nothing is added to them."""
    prompt, completion = build_texts(tokenizer, family, src_lang, tgt_lang, src, tgt)
    raw = family in ("seedx", "plain")
    p_ids = tokenizer(prompt, add_special_tokens=raw)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] if completion else []
    return p_ids, c_ids
