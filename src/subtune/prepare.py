"""Download the subtitles a selection needs (within the quota) and align them into pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .align import align
from .catalog import Catalog
from .config import Paths
from .languages import pair_key
from .opensubtitles import OpenSubtitlesClient, QuotaExceeded
from .selection import PENDING, feature_states
from .srt import load_cues

Progress = Callable[[str], None]


def pairs_path(paths: Paths, src: str, tgt: str, feature_id: int) -> Path:
    return paths.pairs / pair_key(src, tgt) / f"{feature_id}.jsonl"


def align_feature(catalog: Catalog, paths: Paths, src: str, tgt: str, feature_id: int,
                  src_file_id: int, tgt_file_id: int) -> tuple[bool, str]:
    src_path, tgt_path = catalog.file_path(src_file_id), catalog.file_path(tgt_file_id)
    result = align(load_cues(src_path.read_bytes(), src), load_cues(tgt_path.read_bytes(), tgt), src, tgt)
    out = None
    if result.ok:
        out = pairs_path(paths, src, tgt, feature_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for s, t, start in result.pairs:
                f.write(json.dumps({"src": s, "tgt": t, "start": round(start, 2)}, ensure_ascii=False) + "\n")
    catalog.record_alignment(pair_key(src, tgt), feature_id, src_file_id, tgt_file_id,
                             "ok" if result.ok else "failed", len(result.pairs) if result.ok else 0,
                             result.coverage, result.message, out)
    return result.ok, result.message


def prepare(
    client: OpenSubtitlesClient | None,
    catalog: Catalog,
    paths: Paths,
    src: str,
    tgt: str,
    feature_ids: list[int],
    progress: Progress = print,
) -> dict:
    """Download missing files for `feature_ids` in order, until done or out of quota; align each
    feature as soon as both of its files are present. Returns a summary dict."""
    states = feature_states(catalog, src, tgt, feature_ids)
    todo = [states[f] for f in feature_ids if states[f].status == PENDING]
    progress(f"{len(todo)} movies/episodes to prepare; "
             f"{sum(len(s.missing_files) for s in todo)} downloads needed.")
    downloaded = aligned = failed = 0
    quota_hit = None
    for n, st in enumerate(todo, 1):
        if st.missing_files and quota_hit is None:
            if client is None:
                raise RuntimeError("Downloads needed but no OpenSubtitles client configured")
            try:
                for file_id, subtitle_id, lang in st.missing_files:
                    raw, quota = client.download(file_id)
                    dest = paths.subtitles / lang / f"{file_id}.srt"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)
                    catalog.record_file(file_id, subtitle_id, lang, dest)
                    downloaded += 1
                    if quota.remaining <= 0:
                        quota_hit = quota.reset_time_utc or "unknown"
                st.missing_files = []
            except QuotaExceeded as exc:
                quota_hit = exc.reset_time_utc or "unknown"
                progress(f"Download quota reached ({exc}).")
        if st.missing_files:
            continue
        ok, message = align_feature(catalog, paths, src, tgt, st.feature_id, st.src_file_id, st.tgt_file_id)
        aligned += ok
        failed += not ok
        if n % 10 == 0 or n == len(todo) or not ok:
            progress(f"[{n}/{len(todo)}] feature {st.feature_id}: {message}")
    remaining = sum(1 for s in todo if s.missing_files)
    summary = {"downloaded": downloaded, "aligned": aligned, "failed": failed,
               "still_pending": remaining, "quota_reset_utc": quota_hit}
    progress(f"Prepared: {summary}")
    if quota_hit and remaining:
        progress(f"{remaining} movies/episodes still need downloads; quota resets at {quota_hit}. "
                 "Run Download again after that to continue.")
    return summary
