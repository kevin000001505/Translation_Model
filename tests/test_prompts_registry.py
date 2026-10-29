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


def test_template_whitespace_before_answer_is_dropped():
    class IndentingTokenizer(FakeChatTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
            text = "<start_of_turn>user\nhi<end_of_turn>\n"
            if len(messages) > 1:  # like TranslateGemma's template: indentation leaks before the answer
                text += "<start_of_turn>model\n\n        " + messages[1]["content"] + "<end_of_turn>\n"
            return text + ("<start_of_turn>model\n" if add_generation_prompt else "")

    _, completion = build_texts(IndentingTokenizer(), "chat", "en", "zh-tw", "Hi.", "你好。")
    assert completion == "你好。<end_of_turn>\n"


def test_auto_family_detects_translategemma_template():
    from subtune.prompts import resolve_family

    tok = FakeChatTokenizer()
    tok.chat_template = "{% set source_lang_code = content['source_lang_code'] %}"
    assert resolve_family("auto", tok) == "translategemma"
    tok.chat_template = "chatml"
    assert resolve_family("auto", tok) == "chat"


def test_hy_mt2_uses_its_documented_prompts():
    tok = FakeChatTokenizer()
    prompt, _ = build_texts(tok, "hy_mt2", "en", "zh-tw", "Hello.")
    assert "将以下文本翻译为繁体中文，注意只需要输出翻译后的结果，不要额外解释：\n\nHello." in prompt
    prompt, _ = build_texts(tok, "hy_mt2", "en", "ja", "Hello.")
    assert "Translate the following text into Japanese. Note that you should only output" in prompt


def test_hunyuan_uses_chinese_instruction_for_chinese_pairs():
    tok = FakeChatTokenizer()
    prompt, _ = build_texts(tok, "hunyuan_mt", "en", "zh-tw", "Hello.")
    assert "把下面的文本翻译成繁体中文，不要额外解释。" in prompt


def test_seedx_prompt_ends_with_language_tag():
    tok = FakeChatTokenizer()
    prompt, completion = build_texts(tok, "seedx", "en", "zh-tw", "Hello.", "你好。")
    assert prompt.endswith("Hello. <zh>") and completion == "你好。<|im_end|>"


def test_tag_metadata_and_listing(tmp_path):
    cfg = TrainConfig(model_id="tencent/Hy-MT2-7B", family="hy_mt2", src="en", tgt="zh-tw")
    tag = make_tag(tmp_path, cfg, dt.date(2026, 10, 29))
    assert tag == "hy-mt2-7b__en-zh-TW__lora__2026-10-29"
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
    assert {m.family for m in MODELS} == {"translategemma", "hy_mt2", "hunyuan_mt", "seedx", "chat"}
    assert not any(m.model_id.startswith("google/") for m in MODELS)  # Google's own repos are gated
    gguf = [m for m in MODELS if m.gguf_file]
    assert [m.label for m in gguf] == ["TranslateGemma 4B (GGUF f16)", "TranslateGemma 12B (GGUF Q8_0)",
                                       "TranslateGemma 27B (GGUF Q8_0)"]
    assert all(m.gguf_file.endswith((".f16.gguf", ".Q8_0.gguf")) for m in gguf)
    assert estimate_vram_gb("QLoRA", 14.8) < 32 < estimate_vram_gb("LoRA", 14.8)
    assert suggest_gpu("LoRA", 7.5) == "L40S"
    assert suggest_gpu("Full", 7.5) == "H200"


def test_gguf_names_and_file_listing(monkeypatch):
    import huggingface_hub

    from subtune.models import gguf_files
    from subtune.registry import model_short_name

    assert model_short_name("mradermacher/translategemma-4b-it-GGUF") == "translategemma-4b"
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo, token=None: [
        "README.md", "m.Q4_K_M.gguf", "m.f16.gguf", "m.mmproj-f16.gguf"])
    assert gguf_files("org/m-GGUF") == ["m.Q4_K_M.gguf", "m.f16.gguf"]  # vision projector left out


def test_only_ungated_models_are_allowed(monkeypatch):
    import types

    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    from subtune.models import check_downloadable

    def fake_info(model_id, token=None):
        assert token is False  # checked anonymously, the way a user without a token would see it
        if model_id == "org/missing":
            raise RepositoryNotFoundError("nope", response=types.SimpleNamespace(headers={}, status_code=404))
        return types.SimpleNamespace(gated="manual" if model_id == "google/translategemma-4b-it" else False)

    monkeypatch.setattr(huggingface_hub, "model_info", fake_info)
    assert check_downloadable("Qwen/Qwen3-4B")[0]
    ok, msg = check_downloadable("google/translategemma-4b-it")
    assert not ok and "gated" in msg
    assert not check_downloadable("org/missing")[0]
    assert not check_downloadable("not-an-id")[0]
