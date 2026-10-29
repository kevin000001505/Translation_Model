import datetime as dt
import json

from subtune.models import MODELS, estimate_vram_gb, suggest_gpu
from subtune.prompts import build_texts
from subtune.registry import list_models, make_tag, write_metadata
from subtune.runconfig import TrainConfig


class FakeChatTokenizer:
    """Mimics a ChatML template with Qwen3's think block."""

    chat_template = "chatml"
    eos_token = "<|im_end|>"

    def get_vocab(self):
        return {"<|im_end|>": 1}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        text = ""
        for m in messages:
            content = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"], ensure_ascii=False)
            if m["role"] == "assistant":
                text += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n"
            else:
                text += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        return text


def test_chat_prompt_and_completion_split_cleanly():
    tok = FakeChatTokenizer()
    prompt, completion = build_texts(tok, "chat", "en", "zh-tw", "Hello.", "你好。")
    assert "Traditional Chinese" in prompt and prompt.endswith("</think>\n\n")
    assert completion == "你好。<|im_end|>\n"


def test_translategemma_uses_language_codes():
    tok = FakeChatTokenizer()
    prompt, _ = build_texts(tok, "translategemma", "en", "zh-tw", "Hello.")
    assert '"source_lang_code": "en"' in prompt and '"target_lang_code": "zh-TW"' in prompt


def test_hunyuan_uses_chinese_instruction_for_chinese_pairs():
    tok = FakeChatTokenizer()
    prompt, _ = build_texts(tok, "hunyuan_mt", "en", "zh-tw", "Hello.")
    assert "把下面的文本翻译成繁体中文，不要额外解释。" in prompt


def test_seedx_prompt_ends_with_language_tag():
    tok = FakeChatTokenizer()
    prompt, completion = build_texts(tok, "seedx", "en", "zh-tw", "Hello.", "你好。")
    assert prompt.endswith("Hello. <zh>") and completion == "你好。<|im_end|>"


def test_tag_metadata_and_listing(tmp_path):
    cfg = TrainConfig(model_id="google/translategemma-4b-it", family="translategemma", src="en", tgt="zh-tw")
    tag = make_tag(tmp_path, cfg, dt.date(2026, 10, 29))
    assert tag == "translategemma-4b__en-zh-TW__lora__2026-10-29"
    (tmp_path / tag).mkdir()
    assert make_tag(tmp_path, cfg, dt.date(2026, 10, 29)).endswith("-2")
    metrics = {"base": {"chrf": 20.0, "bleu": 10.0, "n": 5}, "tuned": {"chrf": 30.0, "bleu": 15.0, "n": 5}}
    meta = write_metadata(tmp_path / tag, tag, cfg, {"train_titles": 3, "train_pairs": 100}, metrics)
    assert meta["language_pair"] == "en-zh-TW"
    card = (tmp_path / tag / "README.md").read_text()
    assert "- zh-TW" in card and "translation" in card
    df = list_models(tmp_path)
    assert df.loc[0, "Tag"] == tag and df.loc[0, "chrF base → tuned"] == "20.0 → 30.0"


def test_curated_models_and_gpu_suggestions():
    assert {m.family for m in MODELS} == {"translategemma", "chat", "seedx", "hunyuan_mt"}
    assert estimate_vram_gb("QLoRA", 27.4) < 32 < estimate_vram_gb("LoRA", 27.4)
    assert suggest_gpu("LoRA", 4.3) == "L40S"
    assert suggest_gpu("LoRA", 27.4) == "A100-80GB"
