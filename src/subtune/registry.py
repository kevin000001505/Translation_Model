"""Tagged fine-tuned models: naming, metadata.json, model card, and listing."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

from .languages import LANGUAGES, iso_code
from .runconfig import TrainConfig


def model_short_name(model_id: str) -> str:
    name = model_id.rstrip("/").split("/")[-1].lower()
    return re.sub(r"-(it|instruct)$", "", name)


def make_tag(models_dir: Path, cfg: TrainConfig, date: dt.date | None = None) -> str:
    date = date or dt.date.today()
    base = f"{model_short_name(cfg.model_id)}__{iso_code(cfg.src)}-{iso_code(cfg.tgt)}__{cfg.method.lower()}__{date:%Y-%m-%d}"
    tag, n = base, 2
    while (models_dir / tag).exists():
        tag, n = f"{base}-{n}", n + 1
    return tag


def write_metadata(out_dir: Path, tag: str, cfg: TrainConfig, data: dict, metrics: dict) -> dict:
    src, tgt = iso_code(cfg.src), iso_code(cfg.tgt)
    meta = {
        "tag": tag,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "base_model": cfg.model_id,
        "method": cfg.method,
        "language_pair": f"{src}-{tgt}",
        "source_language": {"code": src, "name": LANGUAGES.get(cfg.src, cfg.src)},
        "target_language": {"code": tgt, "name": LANGUAGES.get(cfg.tgt, cfg.tgt)},
        "trained_on": cfg.target if cfg.target == "local" else f"modal:{cfg.gpu}",
        "data": data,
        "hyperparameters": cfg.hyperparameters(),
        "scores": {"base": _strip(metrics.get("base")), "fine_tuned": _strip(metrics.get("tuned"))},
        "samples": (metrics.get("tuned") or {}).get("samples", []),
        "train_loss": metrics.get("train_loss"),
        "train_seconds": metrics.get("train_seconds"),
        "prompt_family": metrics.get("prompt_family", cfg.family),
        "weights": "adapter" if cfg.method != "Full" else "model",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "README.md").write_text(model_card(meta), encoding="utf-8")
    return meta


def _strip(scores: dict | None) -> dict | None:
    return {k: v for k, v in scores.items() if k != "samples"} if scores else None


def model_card(meta: dict) -> str:
    src, tgt = meta["source_language"], meta["target_language"]
    base, tuned = meta["scores"]["base"] or {}, meta["scores"]["fine_tuned"] or {}
    tags = ["translation", "opensubtitles", "subtitles", src["code"], tgt["code"], meta["method"].lower(), "subtune"]
    lines = [
        "---",
        f"base_model: {meta['base_model']}",
        "library_name: peft" if meta["weights"] == "adapter" else "library_name: transformers",
        "pipeline_tag: translation",
        "language:",
        f"- {src['code'].split('-')[0]}",
        f"- {tgt['code'].split('-')[0]}",
        "tags:",
        *[f"- {t}" for t in tags],
        "---",
        "",
        f"# {meta['tag']}",
        "",
        f"{meta['method']} fine-tune of `{meta['base_model']}` for **{src['name']} → {tgt['name']}** "
        f"subtitle translation, trained on OpenSubtitles.",
        "",
        "| | chrF | BLEU |",
        "|---|---|---|",
        f"| Base model | {base.get('chrf', '–')} | {base.get('bleu', '–')} |",
        f"| Fine-tuned | {tuned.get('chrf', '–')} | {tuned.get('bleu', '–')} |",
        "",
        f"Scored on {tuned.get('n', 0)} held-out sentences from titles not used in training.",
        "",
        "## Data",
        "",
        "```json",
        json.dumps(meta["data"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Hyperparameters",
        "",
        "```json",
        json.dumps(meta["hyperparameters"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def list_models(models_dir: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(models_dir.glob("*/metadata.json")):
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        base, tuned = m["scores"].get("base") or {}, m["scores"].get("fine_tuned") or {}
        rows.append({
            "Tag": m["tag"],
            "Pair": m["language_pair"],
            "Base model": m["base_model"],
            "Method": m["method"],
            "Titles": m["data"].get("train_titles"),
            "Pairs": m["data"].get("train_pairs"),
            "chrF base → tuned": f"{base.get('chrf', '–')} → {tuned.get('chrf', '–')}",
            "BLEU base → tuned": f"{base.get('bleu', '–')} → {tuned.get('bleu', '–')}",
            "Trained on": m.get("trained_on"),
            "Created": m.get("created"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Created", ascending=False).reset_index(drop=True) if not df.empty else df
