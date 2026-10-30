"""The cloud GPU must stop when a run ends, fails, or is stopped from the UI."""

import signal
import subprocess
import sys
import textwrap
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

modal_app = pytest.importorskip("subtune.modal_app")
from subtune.runconfig import TrainConfig  # noqa: E402


class FakeModal:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.app_running = False
        self.options = {}
        self.removed = []

    @contextmanager
    def run(self):
        self.app_running = True
        try:
            yield
        finally:
            self.app_running = False  # leaving app.run() is what stops the ephemeral app

    def with_options(self, **kw):
        self.options = kw
        return types.SimpleNamespace(remote=self.remote)

    def remote(self, cfg, run_name, tag):
        assert self.app_running
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return {"tuned": {"chrf": 1.0}}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    def make(fail=False):
        f = FakeModal(fail)
        monkeypatch.setattr(modal_app.app, "run", f.run)
        monkeypatch.setattr(modal_app.train_remote, "with_options", f.with_options)

        @contextmanager
        def batch_upload(force=False):
            yield types.SimpleNamespace(put_file=lambda *a: None)

        entry = types.SimpleNamespace(path="models/t/adapter/w.bin", type=types.SimpleNamespace(name="FILE"))
        monkeypatch.setattr(modal_app.volume, "batch_upload", batch_upload)
        monkeypatch.setattr(modal_app.volume, "listdir", lambda path, recursive=False: [entry])
        monkeypatch.setattr(modal_app.volume, "read_file", lambda path: iter([b"weights"]))
        monkeypatch.setattr(modal_app.volume, "remove_file", lambda path, recursive=False: f.removed.append(path))
        monkeypatch.setattr(modal_app.modal, "enable_output", contextmanager(lambda: (yield)))
        return f
    return make


def _cfg(**kw):
    return TrainConfig(model_id="Qwen/Qwen3-4B", family="chat", src="en", tgt="zh-tw", target="modal",
                       gpu="L40S", **kw)


def test_finished_run_stops_app_caps_time_and_cleans_volume(fake, tmp_path):
    f = fake()
    logs = []
    modal_app.run_on_modal(_cfg(max_hours=3), tmp_path / "run1", "t", tmp_path / "out", logs.append)
    assert not f.app_running
    assert f.options == {"gpu": "L40S", "timeout": 3 * 3600, "memory": None}  # default RAM, not billed extra
    assert (tmp_path / "out" / "adapter" / "w.bin").read_bytes() == b"weights"
    assert f.removed == ["models/t", "runs/run1"]
    assert any("cloud GPU is released" in m for m in logs)


def test_failed_run_still_stops_app(fake, tmp_path):
    f = fake(fail=True)
    logs = []
    with pytest.raises(RuntimeError):
        modal_app.run_on_modal(_cfg(), tmp_path / "run1", "t", tmp_path / "out", logs.append)
    assert not f.app_running
    assert any("cloud GPU is released" in m for m in logs)


def test_gguf_runs_reserve_ram_for_conversion(fake, tmp_path):
    f = fake()
    cfg = _cfg()
    cfg.model_id, cfg.gguf_file = "mradermacher/translategemma-27b-it-GGUF", "translategemma-27b-it.Q8_0.gguf"
    modal_app.run_on_modal(cfg, tmp_path / "run1", "t", tmp_path / "out", lambda m: None)
    assert f.options["memory"] >= 27 * 2 * 1024  # bf16 copy of 27B parameters


def test_gguf_export_reserves_ram_for_the_merge(fake, tmp_path):
    f = fake()
    modal_app.run_on_modal(_cfg(gguf_export="Q4_K_M"), tmp_path / "run1", "t", tmp_path / "out", lambda m: None)
    assert f.options["memory"] >= 4 * 2 * 1024  # bf16 copy of Qwen3 4B


def test_time_limit_is_clamped_to_modal_maximum(fake, tmp_path):
    f = fake()
    modal_app.run_on_modal(_cfg(max_hours=100), tmp_path / "run1", "t", tmp_path / "out", lambda m: None)
    assert f.options["timeout"] == 24 * 3600


def test_stop_button_sigterm_runs_cleanup(tmp_path):
    """Stop sends SIGTERM to the job process; `with`/`finally` cleanup (stopping the Modal app) must run."""
    script = textwrap.dedent("""
        import sys, time
        sys.path.insert(0, %r)
        import subtune.cli as cli
        import subtune.opensubtitles as os_mod
        def slow_quota(self):
            try:
                print("RUNNING", flush=True)
                time.sleep(60)
            finally:
                print("CLEANUP", flush=True)
        os_mod.OpenSubtitlesClient.quota = slow_quota
        cli.main(["quota"])
    """) % str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                            env={"SUBTUNE_DATA_DIR": str(tmp_path / "d"), "SUBTUNE_OUTPUT_DIR": str(tmp_path / "o")})
    assert proc.stdout.readline().strip() == "RUNNING"
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=20)
    assert "CLEANUP" in out
    assert proc.returncode == 143
