"""Gemma 3 RoPE fix, plain-text GGUF chat template, validation summary, publish + deploy."""

import types
from pathlib import Path

import pytest

from subtune import publish
from subtune.evaluate import compare
from subtune.gemma3 import fix_rope


def _gemma3(rope_parameters):
    text = types.SimpleNamespace(model_type="gemma3_text", rope_parameters=rope_parameters)
    return types.SimpleNamespace(model_type="gemma3", text_config=text)


PER_LAYER = {"full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
             "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}


def test_flattened_vllm_rope_entry_moves_to_full_attention_layers():
    config = _gemma3({"factor": 8.0, "rope_type": "linear", **PER_LAYER})  # Infomaniak-AI copies
    assert fix_rope(config)
    assert config.text_config.rope_parameters == {
        "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},  # sliding layers stay unscaled
    }
    assert not fix_rope(config)  # idempotent


def test_gguf_rope_scaling_is_restored_and_other_models_are_left_alone():
    converted = types.SimpleNamespace(model_type="gemma3_text", rope_parameters=dict(PER_LAYER))
    assert fix_rope(converted, {"rope_type": "linear", "factor": 8.0})
    assert converted.rope_parameters["full_attention"]["factor"] == 8.0
    assert not fix_rope(types.SimpleNamespace(model_type="gemma3_text", rope_parameters=dict(PER_LAYER)))
    qwen = types.SimpleNamespace(model_type="qwen3", rope_parameters={"rope_type": "default"})
    assert not fix_rope(qwen, {"rope_type": "linear", "factor": 8.0})


def test_plain_text_template_renders_the_exact_training_prompt():
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from subtune.prompts import build_texts, plain_text_template

    tk = Tokenizer(models.WordLevel({"<unk>": 0, "<bos>": 1, "<eos>": 2, "<end_of_turn>": 3}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    # Shaped like TranslateGemma's: BOS, structured user content, generation prompt ending in "\n"
    tok.chat_template = (
        "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'user' %}"
        "{{ '<start_of_turn>user\\nTranslate ' + m['content'][0]['source_lang_code'] + ' to ' "
        "+ m['content'][0]['target_lang_code'] + ':\\n\\n\\n' + m['content'][0]['text'] | trim }}"
        "{{ '<end_of_turn>\\n' }}{% else %}{{ '<start_of_turn>model\\n' + m['content'] | trim }}"
        "{{ '<end_of_turn>\\n' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
    )
    trained, _ = build_texts(tok, "translategemma", "en", "zh-tw", "Where are you going?")
    tok.chat_template = plain_text_template(tok, "translategemma", "en", "zh-tw")
    for messages in (
        [{"role": "user", "content": "Where are you going?"}],
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "  Where are you going?\n"}],
        [{"role": "user", "content": [{"type": "text", "text": "Where are you going?"}]}],
    ):
        assert tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) == trained
    assert trained.endswith("<start_of_turn>model\n")


def test_validation_summary_counts():
    rows = [{"src": f"s{i}", "tgt": "你好嗎朋友"} for i in range(3)]
    details, summary = compare(rows, ["你好", "再見", "你好嗎朋友"], ["你好嗎朋友", "再見", "你好嗎朋友"])
    assert summary["sentences"] == 3 and summary["changed"] == 1 and summary["improved"] == 1
    assert summary["worse"] == 0 and details[0]["delta"] > 0


class FakeHf:
    def __init__(self):
        self.calls = []

    def whoami(self):
        return {"name": "kevin000001505"}

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create", repo_id, kw))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
        self.calls.append(("upload", path_in_repo, repo_id))


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("LLM_MANAGER_URL", "http://manager:8000/")
    monkeypatch.delenv("SUBTUNE_HF_NAMESPACE", raising=False)
    tag = "translategemma-12b__en-zh-TW__lora__2026-10-30"
    out = tmp_path / tag
    (out / "gguf").mkdir(parents=True)
    (out / "gguf" / f"{tag}.Q4_K_M.gguf").write_bytes(b"GGUF")
    (out / "README.md").write_text("card")
    meta = {"tag": tag, "gguf": {"type": "Q4_K_M", "file": f"gguf/{tag}.Q4_K_M.gguf"}}
    return out, meta


def test_publish_uploads_privately_then_deploys(run_dir, monkeypatch):
    out, meta = run_dir
    hf, posts = FakeHf(), []
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: hf)

    def post(url, json, timeout):
        posts.append((url, json))
        return FakeResp(200, {"status": "Provisioning started."})

    monkeypatch.setattr(publish.requests, "post", post)
    result = publish.publish(out, meta, lambda m: None)

    repo = "kevin000001505/translategemma-12b__en-zh-TW__lora__2026-10-30-GGUF"
    assert hf.calls[0] == ("create", repo, {"private": True, "exist_ok": True, "repo_type": "model"})
    assert ("upload", "translategemma-12b__en-zh-TW__lora__2026-10-30.Q4_K_M.gguf", repo) in hf.calls
    url, body = posts[0]
    assert url == "http://manager:8000/api/setup"
    assert body["hf_repo"] == repo and body["quant"] == "Q4_K_M" and body["backend"] == "llama"
    assert result["deployed"]["model_id"] == "translategemma-12b_en-zh-TW_lora_2026-10-30-Q4_K_M"


def test_publish_records_failures_instead_of_raising(run_dir, monkeypatch):
    out, meta = run_dir
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: FakeHf())
    monkeypatch.setattr(publish.requests, "post",
                        lambda url, json, timeout: FakeResp(400, {"detail": "Insufficient Storage."}))
    result = publish.publish(out, meta, lambda m: None)
    assert result["repo"].endswith("-GGUF") and "Insufficient Storage" in result["deploy_error"]

    class Broken(FakeHf):
        def create_repo(self, *a, **kw):
            raise RuntimeError("401 Unauthorized: token has no write access")

    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: Broken())
    assert "write access" in publish.publish(out, meta, lambda m: None)["error"]


def test_repo_and_config_names_are_valid():
    assert publish.repo_name("a__b--c..d") == "a__b-c.d-GGUF"
    assert len(publish.repo_name("x" * 200)) <= 96
    assert publish.config_name("m__en-zh-TW__lora__2026-10-30") == "m_en-zh-TW_lora_2026-10-30"
