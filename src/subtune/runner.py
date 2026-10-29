"""One training job end to end: (optional downloads) → dataset → train → tag."""

from __future__ import annotations

import datetime as dt
import shutil
from typing import Callable

from .catalog import Catalog
from .config import OpenSubtitlesSettings, get_paths
from .dataset import build_dataset
from .opensubtitles import OpenSubtitlesClient
from .prepare import prepare
from .registry import make_tag, write_metadata
from .runconfig import TrainConfig

Log = Callable[[str], None]


def run_job(cfg: TrainConfig, log: Log = print) -> str:
    paths = get_paths()
    catalog = Catalog(paths.db)
    if cfg.prepare_first:
        feature_ids = [f for fids in cfg.titles.values() for f in fids]
        client = OpenSubtitlesClient(OpenSubtitlesSettings.from_env())
        prepare(client, catalog, paths, cfg.src, cfg.tgt, feature_ids, log)

    tag = make_tag(paths.models, cfg)
    run_dir = paths.runs / f"{dt.datetime.now():%Y%m%d-%H%M%S}__{tag}"
    info = build_dataset(
        catalog, cfg.src, cfg.tgt, cfg.titles, run_dir,
        max_pairs=cfg.max_pairs or None, eval_max=max(cfg.eval_sentences, 1), seed=cfg.seed,
    )
    if info.warning:
        log("Warning: " + info.warning)
    log(f"Dataset: {info.train_pairs:,} training pairs from {info.train_titles} titles "
        f"({info.train_features} movies/episodes); {info.eval_pairs} evaluation pairs from "
        f"{info.eval_titles} held-out titles.")
    cfg.save(run_dir / "run.json")

    out_dir = paths.models / tag
    try:
        if cfg.target == "modal":
            from .modal_app import run_on_modal

            metrics = run_on_modal(cfg, run_dir, tag, out_dir, log)
        else:
            from .train import run_training

            metrics = run_training(cfg, run_dir, out_dir, log)
    except BaseException:
        if out_dir.exists() and not (out_dir / "metadata.json").exists():
            shutil.rmtree(out_dir, ignore_errors=True)  # don't leave an untagged half-run behind
        raise

    data = {
        "source": "OpenSubtitles",
        "train_titles": info.train_titles,
        "train_movies_and_episodes": info.train_features,
        "train_pairs": info.train_pairs,
        "eval_titles": info.eval_titles,
        "eval_pairs": info.eval_pairs,
        "selected_titles": len(cfg.titles),
        "title_keys": sorted(cfg.titles),
    }
    meta = write_metadata(out_dir, tag, cfg, data, metrics)
    base, tuned = meta["scores"]["base"] or {}, meta["scores"]["fine_tuned"] or {}
    log(f"DONE. Tag: {tag}")
    log(f"chrF {base.get('chrf')} → {tuned.get('chrf')}, BLEU {base.get('bleu')} → {tuned.get('bleu')}")
    return tag
