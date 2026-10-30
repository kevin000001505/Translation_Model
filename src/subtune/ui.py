"""Gradio app: pick a language pair and titles, prepare data, train, and browse tagged models."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

import gradio as gr
import pandas as pd

from . import languages, models
from .catalog import Catalog
from .config import PROJECT_ROOT, OpenSubtitlesSettings, get_paths, modal_configured
from .jobs import JobManager
from .listing import listing_status
from .opensubtitles import OpenSubtitlesClient, OpenSubtitlesError
from .gguf_export import EXPORT_TYPES
from .registry import list_models
from .runconfig import TrainConfig
from .selection import TABLE_COLUMNS, readiness, titles_table

MAX_ROWS = 1000  # rows rendered in the table; filtering/search narrows the view
JOBS = JobManager()
_client: OpenSubtitlesClient | None = None
_qlora_local: tuple[bool, str] | None = None


def _catalog() -> Catalog:
    return Catalog(get_paths().db)


def _api() -> OpenSubtitlesClient:
    global _client
    if _client is None:
        _client = OpenSubtitlesClient(OpenSubtitlesSettings.from_env())
    return _client


def _env_warnings() -> str:
    missing = OpenSubtitlesSettings.from_env().missing()
    if missing:
        return f"⚠️ Missing in `.env`: {', '.join(missing)}. Copy `.env.example` to `.env` and fill it in."
    return ""


# --- data tab -------------------------------------------------------------------------------

def load_listing(src: str, tgt: str):
    if src == tgt:
        return pd.DataFrame(columns=TABLE_COLUMNS), {}, "Pick two different languages."
    cat = _catalog()
    df, members = titles_table(cat, src, tgt)
    status = listing_status(cat, src, tgt)
    if df.empty:
        text = f"**No titles cached for {src} → {tgt} yet.** Click *Build / resume list*.\n\n{status}"
    else:
        movies = int((df["Type"] == "Movie").sum())
        series = df[df["Type"] == "TV"]
        text = (f"### {movies:,} movies and {len(series):,} TV series ({int(series['Episodes'].sum()):,} episodes) "
                f"have both {languages.LANGUAGES[src]} and {languages.LANGUAGES[tgt]} subtitles\n\n{status}")
    warn = _env_warnings()
    return df, members, (warn + "\n\n" + text) if warn else text


def _filtered(df: pd.DataFrame, kind: str, search: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    view = df
    if kind == "Movies":
        view = view[view["Type"] == "Movie"]
    elif kind == "TV":
        view = view[view["Type"] == "TV"]
    if search and search.strip():
        q = search.strip().lower()
        view = view[view["Title"].fillna("").str.lower().str.contains(q, regex=False)
                    | view["IMDb"].str.lower().str.contains(q, regex=False)]
    return view


def render_table(df, selected: list[str], kind: str, search: str) -> pd.DataFrame:
    view = _filtered(df, kind, search).head(MAX_ROWS).copy()
    view["Select"] = view["Key"].isin(set(selected or []))
    return view


def selection_summary(df, selected: list[str]) -> str:
    if not selected:
        return "Nothing selected."
    sel = df[df["Key"].isin(set(selected))] if df is not None and not df.empty else df
    movies = int((sel["Type"] == "Movie").sum())
    tv = sel[sel["Type"] == "TV"]
    return (f"**{len(selected)} titles selected**: {movies} movies, {len(tv)} series "
            f"({int(tv['Episodes'].sum())} episodes).")


def on_table_edit(view: pd.DataFrame, selected: list[str], df):
    chosen = set(selected or [])
    for _, row in view.iterrows():
        if bool(row["Select"]):
            chosen.add(row["Key"])
        else:
            chosen.discard(row["Key"])
    order = {k: i for i, k in enumerate(df["Key"])} if df is not None and not df.empty else {}
    new = sorted(chosen, key=lambda k: order.get(k, 1 << 30))
    return new, selection_summary(df, new)


def select_top(df, kind: str, search: str, n: int):
    keys = list(_filtered(df, kind, search)["Key"].head(int(n)))
    return keys, render_table(df, keys, kind, search), selection_summary(df, keys)


def check_readiness(src: str, tgt: str, selected: list[str], members: dict):
    if not selected:
        return "Select some titles first."
    r = readiness(_catalog(), src, tgt, selected, members)
    try:
        q = _api().quota()
    except OpenSubtitlesError as exc:
        return r.describe() + f"\n\n(Could not read quota: {exc})"
    return r.describe(q.remaining, q.allowed)


def _feature_ids(selected: list[str], members: dict) -> list[int]:
    return [f for k in selected for f in members.get(k, [])]


def start_scan(src: str, tgt: str, refresh: bool):
    if src == tgt:
        return "Pick two different languages."
    if _env_warnings():
        return _env_warnings()
    args = ["scan", "--src", src, "--tgt", tgt] + (["--refresh"] if refresh else [])
    try:
        JOBS.start("data", f"scan {src}→{tgt}", args)
    except RuntimeError as exc:
        return str(exc)
    return "Scan started. Progress shows below; the table reloads when it finishes."


def start_download(src: str, tgt: str, selected: list[str], members: dict):
    if not selected:
        return "Select some titles first."
    path = get_paths().runs / f"features-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(_feature_ids(selected, members)))
    try:
        JOBS.start("data", f"download {src}→{tgt}", ["prepare", "--src", src, "--tgt", tgt,
                                                       "--features-file", str(path)])
    except RuntimeError as exc:
        return str(exc)
    return "Download started (most popular selected titles first)."


LIVE_RELOAD_SECONDS = 30


def poll_data(was_running: bool, last_reload: float, src: str, tgt: str, selected, kind, search):
    status, log = JOBS.view("data")
    job = JOBS.jobs.get("data")
    running = bool(job and job.running)
    now = time.time()
    # Refresh the list when a job just finished, and when a scan of these languages (started here
    # or with `subtune scan` on the command line) has saved progress since the last refresh: at most
    # every 30 s while it runs, right away once it has finished.
    saved_at, scanning = _catalog().scan_marker(src, tgt) if src != tgt else (0.0, False)
    progressed = saved_at > last_reload and (not scanning or now - last_reload >= LIVE_RELOAD_SECONDS)
    if (was_running and not running) or progressed:
        df, members, text = load_listing(src, tgt)
        return status, log, running, now, df, members, text, render_table(df, selected, kind, search)
    return status, log, running, last_reload, gr.skip(), gr.skip(), gr.skip(), gr.skip()


# --- train tab ------------------------------------------------------------------------------

def _spec(model_label: str, custom_id: str, gguf_file: str = "") -> models.ModelSpec | None:
    if model_label == models.CUSTOM_LABEL:
        if not custom_id or not custom_id.strip():
            return None
        return models.custom_spec(custom_id.strip(), gguf_file or "")
    return models.by_label(model_label)


def model_changed(model_label: str, method: str, target: str, custom_id: str, gguf_file: str):
    spec = _spec(model_label, custom_id, gguf_file)
    d = models.defaults(method, spec.params_b if spec else 0)
    custom = model_label == models.CUSTOM_LABEL
    return (
        gr.update(visible=custom),
        gr.update(visible=custom and bool(gguf_file)),
        gr.update(visible=target == models.TARGETS[1], value=models.suggest_gpu(method, spec.params_b if spec else 8)),
        gr.update(visible=target == models.TARGETS[1]),
        estimate_text(model_label, method, target, custom_id, gguf_file),
        d["learning_rate"], d["lora_r"], d["lora_alpha"], d["batch_size"], d["grad_accum"],
    )


def custom_repo_changed(model_label: str, method: str, target: str, custom_id: str):
    """List the repo's GGUF files (if it is a GGUF repo) and re-check it."""
    files = models.gguf_files(custom_id.strip()) if custom_id and custom_id.strip() else []
    choice = files[0] if files else ""
    preferred = [f for f in files if re.search(r"(bf16|f16|f32|q8_0)", f, re.I)]
    if preferred:
        choice = preferred[0]  # fine-tune from the most precise file available
    return (gr.update(choices=files, value=choice or None, visible=bool(files)),
            estimate_text(model_label, method, target, custom_id, choice))


def estimate_text(model_label: str, method: str, target: str, custom_id: str, gguf_file: str = "") -> str:
    spec = _spec(model_label, custom_id, gguf_file)
    if not spec:
        return "Enter a Hugging Face model id."
    lines = []
    if model_label == models.CUSTOM_LABEL:
        ok, msg = models.check_downloadable(spec.model_id)
        lines.append(("✅ " if ok else "⛔ ") + msg)
        if spec.gguf_file:
            lines.append(f"GGUF file `{spec.gguf_file}` is converted to a regular checkpoint once before training "
                         "(needs RAM for the full model). Prefer f16/bf16/Q8_0 files: lower-bit quantization "
                         "errors carry into the fine-tune.")
    if spec.note:
        lines.append(f"ℹ️ {spec.note}")
    if spec.params_b:
        need = models.estimate_vram_gb(method, spec.params_b)
        lines.append(f"Estimated training memory: ~{need:.0f} GB.")
        if target == models.TARGETS[0] and need > models.LOCAL_VRAM_GB:
            lines.append(f"⚠️ Likely too big for the local {models.LOCAL_VRAM_GB} GB GPU: use QLoRA or a Modal GPU.")
    if target == models.TARGETS[0] and method == "QLoRA":
        if _qlora_local is None:
            lines.append("QLoRA on ROCm depends on bitsandbytes support for your GPU; it is checked when you start.")
        elif not _qlora_local[0]:
            lines.append(f"⚠️ QLoRA is not available locally ({_qlora_local[1]}). Use LoRA or a Modal GPU.")
    if target == models.TARGETS[1]:
        lines.append("The cloud GPU is billed per second while the job runs, plus Modal's 60 s idle window. It is "
                     "shut down when training ends or fails, when you press Stop, and at the latest after "
                     "*Max cloud hours*.")
        if not modal_configured():
            lines.append("⚠️ MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are not set in `.env`.")
    return "\n\n".join(lines)


def _check_qlora_local() -> tuple[bool, str]:
    global _qlora_local
    if _qlora_local is None:
        try:
            out = subprocess.run([sys.executable, "-m", "subtune", "check-qlora"], capture_output=True, text=True,
                                 timeout=600, cwd=PROJECT_ROOT,
                                 env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
            result = json.loads(out.stdout.strip().splitlines()[-1])
            _qlora_local = (bool(result["ok"]), result["message"])
        except Exception as exc:
            _qlora_local = (False, f"check failed: {exc}")
    return _qlora_local


def publish_status() -> str:
    from .publish import missing_settings, settings

    missing = missing_settings()
    if missing:
        return "Needs in `.env`: " + ", ".join(missing) + "."
    return f"Deploys to llm-model-manager at {settings()['manager_url']}."


def _build_config(src, tgt, selected, members, model_label, custom_id, gguf_file, method, target, gpu, max_hours,
                  epochs, max_pairs, gguf_export, publish, lr, lora_r, lora_alpha, batch_size, grad_accum,
                  max_seq_len, eval_sentences) -> TrainConfig:
    spec = _spec(model_label, custom_id, gguf_file)
    return TrainConfig(
        model_id=spec.model_id, family=spec.family, gguf_file=spec.gguf_file, src=src, tgt=tgt, method=method,
        target="modal" if target == models.TARGETS[1] else "local", gpu=gpu or "",
        max_hours=float(max_hours or 8), trust_remote_code=spec.trust_remote_code,
        epochs=float(epochs), max_pairs=int(max_pairs or 0), learning_rate=float(lr), lora_r=int(lora_r),
        lora_alpha=int(lora_alpha), batch_size=int(batch_size), grad_accum=int(grad_accum),
        max_seq_len=int(max_seq_len), eval_sentences=int(eval_sentences),
        gguf_export="" if gguf_export == NO_EXPORT else gguf_export,
        publish=bool(publish),
        titles={k: members.get(k, []) for k in selected},
    )


HIDE = gr.update(visible=False)
NO_EXPORT = "No GGUF export"
EXPORT_CHOICES = list(EXPORT_TYPES)  # Q4_K_M, Q8_0, F16, BF16


def start_training(*inputs):
    """Check everything; start right away if all data is ready, else ask what to do."""
    src, tgt, selected, members, model_label, custom_id, gguf_file, method, target = inputs[:9]
    no_choice = (HIDE, "", HIDE, HIDE, HIDE)
    if not selected:
        return ("Select titles on the Data tab first.", *no_choice)
    if not _spec(model_label, custom_id, gguf_file):
        return ("Enter a Hugging Face model id.", *no_choice)
    if model_label == models.CUSTOM_LABEL:
        ok, msg = models.check_downloadable(custom_id.strip())
        if not ok:
            return (msg, *no_choice)
    if target == models.TARGETS[1] and not modal_configured():
        return ("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env first.", *no_choice)
    cfg = _build_config(*inputs)
    if cfg.publish:
        from .publish import missing_settings

        if not cfg.gguf_export:
            return ("Uploading needs a GGUF: pick an export type, or untick the upload box.", *no_choice)
        if missing_settings():
            return ("To upload and deploy, set " + ", ".join(missing_settings()) + " in .env (or untick the box).",
                    *no_choice)
    if cfg.method == "Full" and cfg.gguf_file and cfg.gguf_export:
        return ("GGUF export of a *full* fine-tune of a GGUF base model is not supported (its tokenizer would "
                "have to be rebuilt). Use LoRA or QLoRA, or choose 'No GGUF export'.", *no_choice)
    if target == models.TARGETS[0] and method == "QLoRA":
        ok, msg = _check_qlora_local()
        if not ok:
            return (f"QLoRA is not available on the local GPU: {msg}. Use LoRA locally, or QLoRA on Modal.",
                    *no_choice)
    r = readiness(_catalog(), src, tgt, selected, members)
    if r.pending_features == 0:
        if r.ready_features == 0:
            return ("None of the selected titles can be used (alignment failed or no subtitle file).", *no_choice)
        return (_launch_training(_build_config(*inputs), prepare_first=False), *no_choice)
    remaining = None
    try:
        q = _api().quota()
        remaining, allowed = q.remaining, q.allowed
        text = r.describe(remaining, allowed)
    except OpenSubtitlesError as exc:
        text = r.describe() + f"\n\n(Could not read quota: {exc})"
    fits = remaining is not None and r.downloads_needed <= remaining
    text = "### Not all selected titles are downloaded yet\n\n" + text
    return (
        "Waiting for your choice below.",
        gr.update(visible=True),
        text,
        gr.update(visible=r.ready_features > 0,
                  value=f"Train now on the {r.ready_features} ready movies/episodes"),
        gr.update(visible=True),
        gr.update(visible=fits, value=f"Download {r.downloads_needed} files, then train on everything"),
    )


def _launch_training(cfg: TrainConfig, prepare_first: bool) -> str:
    cfg.prepare_first = prepare_first
    path = get_paths().runs / f"run-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    cfg.save(path)
    try:
        JOBS.start("train", f"train {cfg.model_id} {cfg.src}→{cfg.tgt} ({cfg.method}, {cfg.target})",
                   ["train", "--config", str(path)])
    except RuntimeError as exc:
        return str(exc)
    return "Training job started. Follow the log below; the result appears on the Models tab."


def train_ready_now(*inputs):
    return _launch_training(_build_config(*inputs), prepare_first=False), HIDE


def download_then_train(*inputs):
    return _launch_training(_build_config(*inputs), prepare_first=True), HIDE


def download_only(src, tgt, selected, members):
    return start_download(src, tgt, selected, members) + " Progress is on the Data tab.", HIDE


def poll_train():
    return JOBS.view("train", lines=80)


# --- models tab -----------------------------------------------------------------------------

def refresh_models():
    df = list_models(get_paths().models)
    tags = list(df["Tag"]) if not df.empty else []
    return df, gr.update(choices=tags, value=tags[0] if tags else None)


SENTENCE_COLUMNS = ["Source", "Reference", "Base model", "Fine-tuned", "chrF base", "chrF tuned", "Δ chrF"]


def validation_view(tag: str):
    """Summary, loss curves, base-vs-tuned scores and per-sentence results for one tagged model."""
    empty = (pd.DataFrame(columns=["step", "loss", "series"]), pd.DataFrame(columns=["bar", "score", "model"]),
             pd.DataFrame(columns=SENTENCE_COLUMNS))
    if not tag:
        return ("", *empty)
    folder = get_paths().models / tag
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) \
        if (folder / "metadata.json").exists() else {}
    if not meta:
        return ("No metadata for this tag.", *empty)
    v = meta.get("validation") or {}
    scores = meta.get("scores") or {}
    base, tuned = scores.get("base") or {}, scores.get("fine_tuned") or {}

    lines = [f"### {tag}"]
    if base and tuned:
        lines.append(f"**chrF** {base.get('chrf')} → {tuned.get('chrf')} ({tuned.get('chrf', 0) - base.get('chrf', 0):+.2f})"
                     f" · **BLEU** {base.get('bleu')} → {tuned.get('bleu')} ({tuned.get('bleu', 0) - base.get('bleu', 0):+.2f})"
                     f" on {tuned.get('n')} held-out sentences")
    if v.get("loss_before") is not None:
        lines.append(f"**Validation loss** {v['loss_before']:.4f} → {v['loss_after']:.4f}")
    if v.get("sentences"):
        lines.append(f"**Per sentence:** {v['improved']} better, {v['worse']} worse, "
                     f"{v['sentences'] - v['changed']} unchanged of {v['sentences']}")
        if v.get("changed") == 0:
            lines.append("⚠️ The fine-tuned model translated every validation sentence exactly like the base "
                         "model: this fine-tune had no effect.")
    elif not v:
        lines.append("This model was trained before validation reports existed; only the scores are available.")

    curves = meta.get("curves") or {}
    loss_rows = [{"step": s_, "loss": l_, "series": name}
                 for key, name in (("train", "training"), ("validation", "validation"))
                 for s_, l_ in curves.get(key, [])]
    loss_df = pd.DataFrame(loss_rows, columns=["step", "loss", "series"])
    score_rows = [{"bar": f"{name} · {label}", "score": d.get(m), "model": label}
                  for m, name in (("chrf", "chrF"), ("bleu", "BLEU"))
                  for label, d in (("base", base), ("fine-tuned", tuned)) if d.get(m) is not None]
    score_df = pd.DataFrame(score_rows, columns=["bar", "score", "model"])

    sentences = pd.DataFrame(columns=SENTENCE_COLUMNS)
    path = folder / "validation.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        sentences = pd.DataFrame([[r["src"], r["ref"], r["base"], r["tuned"], r["base_chrf"], r["tuned_chrf"], r["delta"]]
                                  for r in rows], columns=SENTENCE_COLUMNS).sort_values("Δ chrF", ascending=False)
    return "\n\n".join(lines), loss_df, score_df, sentences


def show_model(tag: str):
    if not tag:
        return ({}, *validation_view(""))
    path = get_paths().models / tag / "metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return (meta, *validation_view(tag))


# --- layout ---------------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    lang_choices = languages.choices()
    model_choices = [m.label for m in models.MODELS] + [models.CUSTOM_LABEL]
    first = models.MODELS[0]
    d = models.defaults("LoRA", first.params_b)

    with gr.Blocks(title="subtune: subtitle translation fine-tuning") as demo:
        gr.Markdown("# subtune\nFine-tune translation models on OpenSubtitles parallel subtitles.")
        st_df = gr.State(pd.DataFrame(columns=TABLE_COLUMNS))
        st_members = gr.State({})
        st_selected = gr.State([])
        st_data_running = gr.State(False)
        st_last_reload = gr.State(0.0)

        with gr.Row():
            src = gr.Dropdown(lang_choices, value="en", label="Source language")
            tgt = gr.Dropdown(lang_choices, value="zh-tw", label="Target language")

        with gr.Tab("1 · Data"):
            listing_md = gr.Markdown()
            with gr.Row():
                scan_btn = gr.Button("Build / resume list", variant="primary")
                refresh_cb = gr.Checkbox(label="Rescan for new subtitles", value=False)
                reload_btn = gr.Button("Reload table")
            with gr.Row():
                kind = gr.Radio(["Both", "Movies", "TV"], value="Both", label="Show")
                search = gr.Textbox(label="Search title or IMDb id", placeholder="e.g. inception or tt1375666")
            with gr.Row():
                top_n = gr.Slider(1, 3000, value=100, step=1, label="Use top N titles by popularity")
                top_btn = gr.Button("Select top N")
                clear_btn = gr.Button("Clear selection")
            table = gr.Dataframe(
                value=pd.DataFrame(columns=TABLE_COLUMNS),
                headers=TABLE_COLUMNS,
                datatype=["bool", "str", "str", "number", "str", "number", "number", "number", "number", "str", "str"],
                interactive=True,
                static_columns=list(range(1, len(TABLE_COLUMNS))),
                max_height=520,
                label=f"Titles with both languages (tick Select to add or remove; shows up to {MAX_ROWS} rows)",
            )
            sel_md = gr.Markdown("Nothing selected.")
            with gr.Row():
                ready_btn = gr.Button("Check readiness & quota")
                dl_btn = gr.Button("Download selected (uses quota)")
                stop_data = gr.Button("Stop data job", variant="stop")
            ready_md = gr.Markdown()
            data_status = gr.Markdown()
            data_log = gr.Textbox(label="Data job log", lines=12, max_lines=12, autoscroll=True)

        with gr.Tab("2 · Train"):
            with gr.Row():
                model_dd = gr.Dropdown(model_choices, value=first.label, label="Base model")
                custom_id = gr.Textbox(label="Hugging Face model id", visible=False,
                                       placeholder="org/model, or a GGUF repo such as org/model-GGUF")
                gguf_dd = gr.Dropdown([], label="GGUF file", visible=False)
            with gr.Row():
                method = gr.Radio(list(models.METHODS), value="LoRA", label="Method")
                target = gr.Radio(list(models.TARGETS), value=models.TARGETS[0], label="Run on")
                gpu = gr.Dropdown(list(models.MODAL_GPUS), value=models.suggest_gpu("LoRA", first.params_b),
                                  label="Modal GPU", visible=False)
                max_hours = gr.Number(value=8, minimum=0.1, maximum=24, label="Max cloud hours (hard stop)",
                                      visible=False)
            estimate_md = gr.Markdown(estimate_text(first.label, "LoRA", models.TARGETS[0], ""))
            with gr.Row():
                epochs = gr.Number(value=d["epochs"], label="Epochs", minimum=0.1, step=0.5)
                max_pairs = gr.Number(value=0, label="Max training pairs (0 = all)", minimum=0, precision=0)
                gguf_export = gr.Dropdown([NO_EXPORT, *EXPORT_CHOICES], value="Q4_K_M",
                                          label="Also export as GGUF (llama.cpp / Ollama / LM Studio)")
            with gr.Row():
                publish_cb = gr.Checkbox(value=False, label="Upload the GGUF to my Hugging Face (private repo) and "
                                                            "deploy it with llm-model-manager")
                publish_md = gr.Markdown(publish_status())
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    lr = gr.Number(value=d["learning_rate"], label="Learning rate")
                    lora_r = gr.Number(value=d["lora_r"], label="LoRA rank", precision=0)
                    lora_alpha = gr.Number(value=d["lora_alpha"], label="LoRA alpha", precision=0)
                with gr.Row():
                    batch_size = gr.Number(value=d["batch_size"], label="Batch size", precision=0)
                    grad_accum = gr.Number(value=d["grad_accum"], label="Gradient accumulation", precision=0)
                    max_seq_len = gr.Number(value=d["max_seq_len"], label="Max sequence length", precision=0)
                    eval_sentences = gr.Number(value=d["eval_sentences"], label="Evaluation sentences", precision=0)
            start_btn = gr.Button("Start training", variant="primary")
            train_msg = gr.Markdown()
            with gr.Group(visible=False) as choice_box:
                choice_md = gr.Markdown()
                with gr.Row():
                    train_now_btn = gr.Button("Train now on ready data", variant="primary")
                    dl_only_btn = gr.Button("Download only (continue tomorrow)")
                    dl_train_btn = gr.Button("Download, then train")
            with gr.Row():
                train_status = gr.Markdown()
                stop_train = gr.Button("Stop training", variant="stop")
            train_log = gr.Textbox(label="Training log", lines=18, max_lines=18, autoscroll=True)

        with gr.Tab("3 · Models"):
            models_btn = gr.Button("Refresh")
            models_df = gr.Dataframe(interactive=False, label="Fine-tuned models (by tag)")
            model_pick = gr.Dropdown([], label="Details for tag")
            val_md = gr.Markdown()
            with gr.Row():
                loss_plot = gr.LinePlot(x="step", y="loss", color="series", title="Loss (validation = held-out titles)",
                                        x_title="training step", y_title="loss", height=300)
                score_plot = gr.BarPlot(x="bar", y="score", color="model", title="Validation scores",
                                        x_title="", y_title="score", height=300)
            sentence_df = gr.Dataframe(interactive=False, wrap=True, max_height=420,
                                       label="Validation sentences, most improved first")
            with gr.Accordion("metadata.json", open=False):
                model_json = gr.JSON()

        # --- wiring ---
        listing_outputs = [st_df, st_members, listing_md]
        view_inputs = [st_df, st_selected, kind, search]

        def reload(s, t, sel, k, q):
            df, members, text = load_listing(s, t)
            keys = set(df["Key"]) if not df.empty else set()
            sel = [x for x in (sel or []) if x in keys]
            return df, members, text, sel, render_table(df, sel, k, q), selection_summary(df, sel)

        reload_outputs = [st_df, st_members, listing_md, st_selected, table, sel_md]
        for trigger in (demo.load, src.change, tgt.change, reload_btn.click):
            trigger(reload, [src, tgt, st_selected, kind, search], reload_outputs)
        for comp in (kind, search):
            comp.change(render_table, view_inputs, table)
        table.input(on_table_edit, [table, st_selected, st_df], [st_selected, sel_md])
        top_btn.click(select_top, [st_df, kind, search, top_n], [st_selected, table, sel_md])
        clear_btn.click(lambda df, k, q: ([], render_table(df, [], k, q), "Nothing selected."),
                        [st_df, kind, search], [st_selected, table, sel_md])
        scan_btn.click(start_scan, [src, tgt, refresh_cb], data_status)
        ready_btn.click(check_readiness, [src, tgt, st_selected, st_members], ready_md)
        dl_btn.click(start_download, [src, tgt, st_selected, st_members], data_status)
        stop_data.click(lambda: JOBS.stop("data"), None, data_status)

        model_inputs = [model_dd, method, target, custom_id, gguf_dd]
        model_outputs = [custom_id, gguf_dd, gpu, max_hours, estimate_md, lr, lora_r, lora_alpha, batch_size,
                         grad_accum]
        for comp in (model_dd, method, target):
            comp.change(model_changed, model_inputs, model_outputs)
        for event in (custom_id.submit, custom_id.blur):  # checks Hugging Face, so not on every keystroke
            event(custom_repo_changed, model_inputs[:4], [gguf_dd, estimate_md])
        gguf_dd.input(estimate_text, model_inputs, estimate_md)

        train_inputs = [src, tgt, st_selected, st_members, model_dd, custom_id, gguf_dd, method, target, gpu, max_hours,
                        epochs, max_pairs, gguf_export, publish_cb, lr, lora_r, lora_alpha, batch_size, grad_accum,
                        max_seq_len, eval_sentences]
        start_btn.click(start_training, train_inputs,
                        [train_msg, choice_box, choice_md, train_now_btn, dl_only_btn, dl_train_btn])
        train_now_btn.click(train_ready_now, train_inputs, [train_msg, choice_box])
        dl_train_btn.click(download_then_train, train_inputs, [train_msg, choice_box])
        dl_only_btn.click(download_only, [src, tgt, st_selected, st_members], [train_msg, choice_box])
        stop_train.click(lambda: JOBS.stop("train"), None, train_status)

        timer = gr.Timer(2.0)
        timer.tick(poll_data, [st_data_running, st_last_reload, src, tgt, st_selected, kind, search],
                   [data_status, data_log, st_data_running, st_last_reload, st_df, st_members, listing_md, table])
        timer.tick(poll_train, None, [train_status, train_log])

        demo.load(refresh_models, None, [models_df, model_pick])
        models_btn.click(refresh_models, None, [models_df, model_pick])
        model_pick.change(show_model, model_pick, [model_json, val_md, loss_plot, score_plot, sentence_df])
    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    build_ui().queue().launch(server_name=host, server_port=port, share=share)
