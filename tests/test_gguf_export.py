"""GGUF export: which llama.cpp steps run for each route, and what is left behind."""

from pathlib import Path

import pytest

from subtune import gguf_export


@pytest.fixture
def fake_llama(monkeypatch, tmp_path):
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("")
    monkeypatch.setenv("SUBTUNE_LLAMA_CPP", str(root))
    calls = []

    def fake_run(cmd, log, env=None):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if cmd[0] == "cmake" and "--build" in cmd:
            for name in gguf_export.LLAMA_CPP_TOOLS:
                (root / "build" / "bin" / name).write_text("")
        # create whatever file the step writes
        for flag in ("--outfile", "-o"):
            if flag in cmd:
                Path(cmd[cmd.index(flag) + 1]).write_bytes(b"gguf")
        if Path(cmd[0]).name == "llama-quantize":
            positional = [a for a in cmd[1:] if not a.startswith("--")]
            Path(positional[1]).write_bytes(b"gguf")
        if env is not None:
            assert env["PYTHONPATH"].startswith(str(root / "gguf-py"))  # llama.cpp's own gguf-py first

    def fake_merge(source, weights_dir, work_dir, trust_remote_code, log):
        work_dir.mkdir(parents=True)
        return work_dir

    monkeypatch.setattr(gguf_export, "_run", fake_run)
    monkeypatch.setattr(gguf_export, "_merged_checkpoint", fake_merge)
    return root, calls


def _names(calls):
    """Tool per step: the script for `python script.py ...`, else the binary."""
    return [Path(c[1]).name if Path(c[0]).name.startswith("python") else Path(c[0]).name for c in calls]


def _export(tmp_path, kind, is_adapter=True, base_gguf=None):
    out_dir = tmp_path / "models" / "my-tag"
    work = tmp_path / "work"
    path = gguf_export.export_gguf(kind, "/src/model", tmp_path / "adapter", is_adapter, out_dir, work, False,
                                   lambda m: None, base_gguf=base_gguf)
    assert path == out_dir / "gguf" / f"my-tag.{kind}.gguf" and path.exists()
    assert not work.exists()  # merged checkpoint and intermediates are removed
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
    return path


def test_hf_base_q4_k_m_converts_then_quantizes(fake_llama, tmp_path):
    root, calls = fake_llama
    _export(tmp_path, "Q4_K_M")
    assert _names(calls) == ["cmake", "cmake", "convert_hf_to_gguf.py", "llama-quantize"]
    convert = calls[2]
    assert convert[convert.index("--outtype") + 1] == "bf16"
    assert calls[3][-1] == "Q4_K_M"


def test_hf_base_q8_0_needs_no_compiled_tools(fake_llama, tmp_path):
    root, calls = fake_llama
    _export(tmp_path, "Q8_0")
    assert _names(calls) == ["convert_hf_to_gguf.py"]
    assert calls[0][calls[0].index("--outtype") + 1] == "q8_0"


def test_full_fine_tune_is_converted_without_merging(fake_llama, tmp_path, monkeypatch):
    root, calls = fake_llama
    monkeypatch.setattr(gguf_export, "_merged_checkpoint", lambda *a: pytest.fail("no merge for full weights"))
    _export(tmp_path, "F16", is_adapter=False)
    assert calls[0][2] == str(tmp_path / "adapter")  # the saved full model folder is converted directly


def test_gguf_base_merges_lora_into_original_file(fake_llama, tmp_path):
    root, calls = fake_llama
    base = tmp_path / "translategemma-4b-it.f16.gguf"
    base.write_bytes(b"orig")
    _export(tmp_path, "Q4_K_M", base_gguf=base)
    assert _names(calls) == ["cmake", "cmake", "convert_lora_to_gguf.py", "llama-export-lora", "llama-quantize"]
    lora_cmd, merge_cmd = calls[2], calls[3]
    assert lora_cmd[lora_cmd.index("--base") + 1] == "/src/model" and lora_cmd[-1] == str(tmp_path / "adapter")
    assert merge_cmd[merge_cmd.index("-m") + 1] == str(base)  # original file: its tokenizer is kept
    # export-lora keeps untouched tensors in the base file's Q8_0: re-quantizing them must be allowed
    assert "--allow-requantize" in calls[4]


def test_gguf_base_f16_skips_quantize(fake_llama, tmp_path):
    root, calls = fake_llama
    base = tmp_path / "b.gguf"
    base.write_bytes(b"orig")
    _export(tmp_path, "F16", base_gguf=base)
    assert _names(calls)[-1] == "llama-export-lora"


def test_full_fine_tune_of_gguf_base_is_refused(fake_llama, tmp_path):
    with pytest.raises(ValueError, match="LoRA or QLoRA"):
        gguf_export.export_gguf("Q8_0", "/src", tmp_path / "m", False, tmp_path / "o", tmp_path / "w", False,
                                lambda m: None, base_gguf=tmp_path / "b.gguf")


def test_tools_are_built_once(fake_llama, tmp_path):
    root, calls = fake_llama
    _export(tmp_path, "Q4_K_M")
    calls.clear()
    _export(tmp_path, "Q4_K_M")
    assert "cmake" not in _names(calls)


def test_dockerfile_pins_the_same_llama_cpp_release():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    assert f"ARG LLAMA_CPP_TAG={gguf_export.LLAMA_CPP_TAG}" in dockerfile
