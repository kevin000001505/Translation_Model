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
    name = re.sub(r"[-_.]gguf$", "", name)
    return re.sub(r"-(it|instruct)$", "", name)


def make_tag(models_dir: Path, cfg: TrainConfig, date: dt.date | None = None) -> str:
    date = date or dt.date.today()
    base = f"{model_short_name(cfg.model_id)}__{iso_code(cfg.src)}-{iso_code(cfg.tgt)}__{cfg.method.lower()}__{date:%Y-%m-%d}"
    tag, n = base, 2
    while (models_dir / tag).exists():
        tag, n = f"{base}-{n}", n + 1
    return tag


def save_metadata(out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "README.md").write_text(model_card(meta), encoding="utf-8")


def write_metadata(out_dir: Path, tag: str, cfg: TrainConfig, data: dict, metrics: dict) -> dict:
    src, tgt = iso_code(cfg.src), iso_code(cfg.tgt)
    meta = {
        "tag": tag,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "base_model": cfg.model_id,
        "base_model_gguf_file": cfg.gguf_file or None,
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
        # Validation set (held-out titles): loss before/after, per-sentence summary, loss curves.
        "validation": metrics.get("validation") or {},
        "curves": metrics.get("curves") or {},
        "train_seconds": metrics.get("train_seconds"),
        "prompt_family": metrics.get("prompt_family", cfg.family),
        "weights": "adapter" if cfg.method != "Full" else "model",
        "gguf": metrics.get("gguf"),  # {"type", "file"} or {"type", "error"}; None when not exported
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(out_dir, meta)
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
        f"{meta['method']} fine-tune of `{meta['base_model']}`"
        + (f" (GGUF file `{meta['base_model_gguf_file']}`)" if meta.get("base_model_gguf_file") else "")
        + f" for **{src['name']} → {tgt['name']}** subtitle translation, trained on OpenSubtitles.",
        "",
        "| | chrF | BLEU |",
        "|---|---|---|",
        f"| Base model | {base.get('chrf', '–')} | {base.get('bleu', '–')} |",
        f"| Fine-tuned | {tuned.get('chrf', '–')} | {tuned.get('bleu', '–')} |",
        "",
        f"Scored on {tuned.get('n', 0)} held-out sentences from titles not used in training.",
        "",
        *_validation_section(meta),
        *_gguf_section(meta),
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


def _gguf_section(meta: dict) -> list[str]:
    gguf = meta.get("gguf")
    if not gguf or not gguf.get("file"):
        return []
    lines = ["## GGUF", ""]
    if gguf.get("plain_text_template"):
        lines.append(f"`{gguf['file']}` ({gguf['type']}) runs in llama.cpp, Ollama and LM Studio. Its chat template "
                     f"wraps the user message in the exact prompt used for fine-tuning, so send just the text to "
                     f"translate ({meta['source_language']['name']}) as the user message; the reply is the "
                     f"{meta['target_language']['name']} translation.")
    else:
        lines.append(f"`{gguf['file']}` ({gguf['type']}) runs in llama.cpp, Ollama and LM Studio, with the base "
                     f"model's chat template (subtune prompt family `{meta['prompt_family']}`).")
    published = meta.get("published") or {}
    if published.get("repo"):
        lines += ["", f"Published to the private repo [{published['repo']}]({published['url']})."]
    deployed = published.get("deployed") or {}
    if deployed.get("model_id"):
        from .publish import api_example

        lines += ["", f"Served by llm-model-manager / llama-swap as `{deployed['model_id']}`:", "", "```bash",
                  api_example(meta), "```"]
    return lines + [""]


def _loss_cell(v: dict) -> str:
    if v.get("loss_before") is None or v.get("loss_after") is None:
        return ""
    return f"{v['loss_before']:.3f} → {v['loss_after']:.3f}"


def _validation_section(meta: dict) -> list[str]:
    v = meta.get("validation") or {}
    if not v.get("sentences"):
        return []
    lines = ["## Validation", ""]
    if v.get("loss_before") is not None:
        lines.append(f"Validation loss: {v['loss_before']:.4f} before training, {v['loss_after']:.4f} after.")
    lines += [
        f"Of {v['sentences']} held-out sentences, {v['improved']} improved, {v['worse']} got worse and "
        f"{v['sentences'] - v['changed']} were translated exactly as before (sentence chrF, ±0.5).",
        "Every sentence with both translations is in `validation.jsonl`.",
        "",
    ]
    return lines


def _gguf_cell(gguf: dict | None) -> str:
    if not gguf:
        return ""
    return gguf["type"] if gguf.get("file") else f"{gguf['type']} (failed)"


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
            "Val. loss before → after": _loss_cell(m.get("validation") or {}),
            "GGUF": _gguf_cell(m.get("gguf")),
            "Serving as": ((m.get("published") or {}).get("deployed") or {}).get("model_id", ""),
            "Trained on": m.get("trained_on"),
            "Created": m.get("created"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Created", ascending=False).reset_index(drop=True) if not df.empty else df
