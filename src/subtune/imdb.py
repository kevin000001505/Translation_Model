"""Candidate titles from IMDb's public datasets (https://developer.imdb.com/non-commercial-datasets/).

Used when OpenSubtitles refuses broad searches: each candidate is then looked up by IMDb id, which
the API always accepts. Titles are ordered by IMDb votes, so the most-watched come first.
The datasets are free for personal and non-commercial use.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path
from typing import Callable

import requests

DATASETS_URL = "https://datasets.imdbws.com/"
MAX_AGE_DAYS = 30
MOVIE_TYPES = {"movie": "movie", "tvMovie": "movie", "tvSeries": "series", "tvMiniSeries": "series"}


def min_votes() -> int:
    return int(os.environ.get("SUBTUNE_IMDB_MIN_VOTES") or 1000)


def _download(name: str, dest_dir: Path, log: Callable[[str], None]) -> Path:
    dest = dest_dir / name
    if dest.exists() and time.time() - dest.stat().st_mtime < MAX_AGE_DAYS * 86400:
        return dest
    log(f"Downloading IMDb dataset {name}...")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with requests.get(DATASETS_URL + name, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def candidates(data_dir: Path, threshold: int, log: Callable[[str], None] = print) -> list[list]:
    """[[imdb_id (int), "movie"|"series", votes, title, year], ...], most votes first."""
    cache_dir = data_dir / "imdb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"candidates-{threshold}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < MAX_AGE_DAYS * 86400:
        return json.loads(cache.read_text())

    votes: dict[str, int] = {}
    with gzip.open(_download("title.ratings.tsv.gz", cache_dir, log), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            tconst, _, n = line.rstrip("\n").split("\t")
            if int(n) >= threshold:
                votes[tconst] = int(n)

    log("Reading IMDb title list...")
    rows = []
    with gzip.open(_download("title.basics.tsv.gz", cache_dir, log), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            kind = MOVIE_TYPES.get(p[1])
            if kind and p[0] in votes:
                year = int(p[5]) if p[5].isdigit() else None
                rows.append([int(p[0][2:]), kind, votes[p[0]], p[2], year])
    rows.sort(key=lambda r: -r[2])
    cache.write_text(json.dumps(rows))
    log(f"{len(rows)} IMDb titles with at least {threshold} votes.")
    return rows
