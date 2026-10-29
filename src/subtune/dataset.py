"""Assemble train/eval JSONL files for a run from cached aligned pairs.

The evaluation set is held out by *title* (a movie, or a whole series), so the model is scored on
films it never saw in training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .catalog import Catalog
from .languages import pair_key


@dataclass
class DatasetInfo:
    train_pairs: int
    eval_pairs: int
    train_titles: int
    eval_titles: int
    train_features: int
    warning: str = ""


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(
    catalog: Catalog,
    src: str,
    tgt: str,
    title_features: dict[str, list[int]],
    out_dir: Path,
    max_pairs: int | None = None,
    eval_fraction: float = 0.01,
    eval_max: int = 500,
    seed: int = 42,
) -> DatasetInfo:
    """`title_features` maps title key -> feature ids (only ready ones are used)."""
    aligned = catalog.alignments(pair_key(src, tgt))
    by_title: dict[str, list[dict]] = {}
    features_used: dict[str, int] = {}
    for key, fids in title_features.items():
        rows = []
        for fid in fids:
            a = aligned.get(fid)
            if a is None or a["status"] != "ok" or not a["path"] or not Path(a["path"]).exists():
                continue
            rows.extend({"src": r["src"], "tgt": r["tgt"], "feature_id": fid} for r in _load(Path(a["path"])))
            features_used[key] = features_used.get(key, 0) + 1
        if rows:
            by_title[key] = rows
    if not by_title:
        raise ValueError("None of the selected titles have aligned pairs yet. Download them first.")

    rng = random.Random(seed)
    keys = sorted(by_title)
    rng.shuffle(keys)
    total = sum(len(v) for v in by_title.values())
    eval_target = max(1, min(eval_max, int(total * eval_fraction)))
    warning = ""

    eval_rows: list[dict] = []
    eval_keys: list[str] = []
    if len(keys) >= 2:
        for k in keys:
            if len(eval_rows) >= eval_target or len(eval_keys) >= len(keys) - 1:
                break
            eval_keys.append(k)
            eval_rows.extend(by_title[k])
        train_rows = [r for k in keys if k not in eval_keys for r in by_title[k]]
    else:
        warning = "Only one title selected: evaluation pairs come from the same title as training."
        rows = by_title[keys[0]][:]
        rng.shuffle(rows)
        eval_rows, train_rows = rows[:eval_target], rows[eval_target:]

    rng.shuffle(eval_rows)
    eval_rows = eval_rows[:eval_max]
    rng.shuffle(train_rows)
    if max_pairs:
        train_rows = train_rows[:max_pairs]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train_rows), ("eval.jsonl", eval_rows)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    train_titles = len(keys) - len(eval_keys) if eval_keys else len(keys)
    return DatasetInfo(
        train_pairs=len(train_rows),
        eval_pairs=len(eval_rows),
        train_titles=train_titles,
        eval_titles=len(eval_keys),
        train_features=sum(features_used[k] for k in keys if k not in eval_keys),
        warning=warning,
    )
