"""Turn a (source, target) sentence pair into model input ids, per prompt family.

Training and evaluation share these functions, so the fine-tuned model sees the same prompt
it will be scored with.
"""

from __future__ import annotations

from .languages import HUNYUAN_CHINESE_NAMES, base_code, iso_code, prompt_name


def resolve_family(family: str, tokenizer) -> str:
    if family != "auto":
        return family
    template = getattr(tokenizer, "chat_template", None) or ""
    if "source_lang_code" in template:  # a TranslateGemma-style template (e.g. from a custom GGUF)
        return "translategemma"
    return "chat" if template else "plain"


def _chinese_prompt(src_lang: str, tgt_lang: str) -> bool:
    """Tencent's models use a Chinese instruction when either side is Chinese."""
    return "zh" in (base_code(src_lang), base_code(tgt_lang)) and tgt_lang in HUNYUAN_CHINESE_NAMES


def _messages(family: str, src_lang: str, tgt_lang: str, text: str) -> list[dict]:
    if family == "translategemma":
        # TranslateGemma's template takes one structured item with ISO codes (e.g. en -> zh-TW).
        content = [{"type": "text", "source_lang_code": iso_code(src_lang),
                    "target_lang_code": iso_code(tgt_lang), "text": text}]
        return [{"role": "user", "content": content}]
    if family == "hy_mt2":  # wording from the Hy-MT2 model card ("Default Translation")
        if _chinese_prompt(src_lang, tgt_lang):
            instruction = f"将以下文本翻译为{HUNYUAN_CHINESE_NAMES[tgt_lang]}，注意只需要输出翻译后的结果，不要额外解释："
        else:
            instruction = (f"Translate the following text into {prompt_name(tgt_lang)}. Note that you should only "
                           "output the translated result without any additional explanation:")
        return [{"role": "user", "content": f"{instruction}\n\n{text}"}]
    if family == "hunyuan_mt":
        if _chinese_prompt(src_lang, tgt_lang):
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


def end_marker(tokenizer) -> str:
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
    if family == "translategemma" and "<<<source>>>" in (getattr(tokenizer, "chat_template", None) or ""):
        # vLLM-style TranslateGemma copies take the codes inside a plain string; same rendered prompt.
        messages = [{"role": "user",
                     "content": f"<<<source>>>{iso_code(src_lang)}<<<target>>>{iso_code(tgt_lang)}<<<text>>>{src}"}]
    prompt = _template(tokenizer, messages, add_generation_prompt=True)
    if tgt is None:
        return prompt, ""
    try:
        full = _template(tokenizer, messages + [{"role": "assistant", "content": tgt}])
    except Exception:  # template refuses an assistant turn
        full = ""
    if full.startswith(prompt) and tgt in full[len(prompt):]:
        completion = full[len(prompt):]
        # Drop template whitespace before the answer (TranslateGemma's template renders "\n        "
        # there), so training targets start exactly where generation does.
        k = completion.find(tgt)
        if k > 0 and not completion[:k].strip():
            completion = completion[k:]
    else:
        completion = tgt + end_marker(tokenizer)
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


_PLACEHOLDER = "\ue000SUBTUNE_SOURCE_TEXT\ue000"


def plain_text_template(tokenizer, family: str, src_lang: str, tgt_lang: str) -> str:
    """A Jinja chat template for the exported model: the last user message is the text to
    translate, wrapped in exactly the prompt the model was fine-tuned with. Callers can then use a
    plain OpenAI-style request ({"role": "user", "content": "Where are you going?"})."""
    prompt, _ = build_texts(tokenizer, family, src_lang, tgt_lang, _PLACEHOLDER)
    head, tail = prompt.split(_PLACEHOLDER)
    bos = ""
    if tokenizer.bos_token and head.startswith(tokenizer.bos_token):
        head, bos = head[len(tokenizer.bos_token):], "{{ bos_token }}"  # let the runtime place BOS
    for part in (head, tail):
        if any(t in part for t in ("{{", "{%", "{#")):
            raise ValueError("prompt text contains Jinja syntax; cannot build a plain-text template")
    # Literal prompt text is kept outside tags (no escaping needed); no "-" trims touch it.
    return (
        '{%- set ns = namespace(text="") -%}'
        "{%- for message in messages -%}"
        '{%- if message["role"] == "user" -%}'
        '{%- if message["content"] is string -%}{%- set ns.text = message["content"] -%}'
        '{%- else -%}{%- set ns.text = message["content"][0]["text"] -%}{%- endif -%}'
        "{%- endif -%}"
        "{%- endfor %}"
        f"{bos}{head}{{{{ ns.text | trim }}}}{tail}"
        # Jinja drops a single trailing newline of the template source; end on an expression so
        # the generation prompt keeps its final "\n".
        "{{ '' }}"
    )
