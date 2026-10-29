"""Titles table for a language pair, readiness of a selection, and best-subtitle choice.

A *title* is the unit you select in the UI: a movie, or a TV series (all its episodes that have
both languages). A *feature* is one movie or one episode; downloads and alignment work per feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .catalog import Catalog, SubtitleRow
from .languages import pair_key

READY, FAILED, UNAVAILABLE, PENDING = "ready", "failed", "unavailable", "pending"


def best_subtitle(cands: list[SubtitleRow]) -> SubtitleRow | None:
    """Most downloaded single-file subtitle that is a full, human-made translation."""
    usable = [
        c for c in cands
        if c.file_id and c.n_files == 1 and not c.ai_translated and not c.machine_translated
    ]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda c: (c.foreign_parts_only, c.hearing_impaired, not c.from_trusted, -c.download_count),
    )[0]


@dataclass
class FeatureState:
    feature_id: int
    status: str  # READY / FAILED / UNAVAILABLE / PENDING
    missing_files: list[tuple[int, str, str]] = field(default_factory=list)  # (file_id, subtitle_id, lang)
    src_file_id: int | None = None
    tgt_file_id: int | None = None
    n_pairs: int = 0


def feature_states(catalog: Catalog, src: str, tgt: str, feature_ids: list[int]) -> dict[int, FeatureState]:
    pair = pair_key(src, tgt)
    aligned = catalog.alignments(pair)
    downloaded = catalog.downloaded_file_ids()
    out: dict[int, FeatureState] = {}
    for fid in feature_ids:
        a = aligned.get(fid)
        if a is not None and a["status"] == "ok":
            out[fid] = FeatureState(fid, READY, n_pairs=a["n_pairs"] or 0)
            continue
        if a is not None:
            out[fid] = FeatureState(fid, FAILED)
            continue
        s = best_subtitle(catalog.candidates(fid, src))
        t = best_subtitle(catalog.candidates(fid, tgt))
        if not s or not t:
            out[fid] = FeatureState(fid, UNAVAILABLE)
            continue
        missing = [(c.file_id, c.subtitle_id, lang) for c, lang in ((s, src), (t, tgt)) if c.file_id not in downloaded]
        out[fid] = FeatureState(fid, PENDING, missing, s.file_id, t.file_id)
    return out


def title_key(row) -> str:
    if row["feature_type"] == "episode":
        return f"series:{row['parent_feature_id'] or row['parent_title']}"
    return f"movie:{row['feature_id']}"


TABLE_COLUMNS = ["Select", "Type", "Title", "Year", "IMDb", "Episodes", "Src subs", "Tgt subs",
                 "Popularity", "Status", "Key"]


def titles_table(catalog: Catalog, src: str, tgt: str) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """One row per title, most popular first, plus the mapping title key -> feature ids."""
    rows = catalog.pair_features(src, tgt)
    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS), {}
    states = feature_states(catalog, src, tgt, [r["feature_id"] for r in rows])
    groups: dict[str, dict] = {}
    members: dict[str, list[int]] = {}
    for r in rows:
        key = title_key(r)
        st = states[r["feature_id"]]
        is_ep = r["feature_type"] == "episode"
        g = groups.setdefault(key, {
            "Type": "TV" if is_ep else "Movie",
            "Title": (r["parent_title"] or r["title"]) if is_ep else r["title"],
            "Year": r["year"],
            "IMDb": f"tt{int(r['parent_imdb_id'] if is_ep else r['imdb_id']):07d}"
                    if (r["parent_imdb_id"] if is_ep else r["imdb_id"]) else "",
            "Episodes": 0, "Src subs": 0, "Tgt subs": 0, "Popularity": 0,
            "_ready": 0, "_failed": 0, "_unavail": 0, "_downloads": 0,
        })
        members.setdefault(key, []).append(r["feature_id"])
        g["Episodes"] += 1
        g["Src subs"] += r["n_src"]
        g["Tgt subs"] += r["n_tgt"]
        g["Popularity"] += r["popularity"] or 0
        if r["year"] and (not g["Year"] or r["year"] < g["Year"]):
            g["Year"] = r["year"]
        g["_ready"] += st.status == READY
        g["_failed"] += st.status == FAILED
        g["_unavail"] += st.status == UNAVAILABLE
        g["_downloads"] += len(st.missing_files)

    records = []
    for key, g in groups.items():
        n = g["Episodes"]
        if g["_ready"] == n:
            status = "ready"
        elif g["_failed"] + g["_unavail"] == n:
            status = "not usable"
        else:
            parts = []
            if g["_ready"]:
                parts.append(f"{g['_ready']}/{n} ready")
            if g["_downloads"]:
                parts.append(f"needs {g['_downloads']} downloads")
            elif g["_ready"] + g["_failed"] + g["_unavail"] < n:
                parts.append("needs alignment")
            if g["_failed"]:
                parts.append(f"{g['_failed']} failed")
            status = ", ".join(parts)
        records.append({"Select": False, **{k: v for k, v in g.items() if not k.startswith("_")},
                        "Status": status, "Key": key})
    df = pd.DataFrame.from_records(records, columns=TABLE_COLUMNS)
    df["Year"] = df["Year"].astype("Int64")
    df = df.sort_values("Popularity", ascending=False, kind="stable").reset_index(drop=True)
    return df, members


@dataclass
class Readiness:
    titles: int
    features: int
    ready_features: int
    ready_titles: int
    pending_features: int
    downloads_needed: int
    unusable_features: int
    ready_pairs: int

    def describe(self, remaining_quota: int | None = None, daily_quota: int = 1000) -> str:
        lines = [
            f"{self.ready_titles} of {self.titles} selected titles are fully ready "
            f"({self.ready_features}/{self.features} movies/episodes, {self.ready_pairs:,} sentence pairs).",
        ]
        if self.pending_features:
            lines.append(f"{self.pending_features} movies/episodes still need {self.downloads_needed} downloads"
                         " (plus alignment).")
        if self.unusable_features:
            lines.append(f"{self.unusable_features} cannot be used (alignment failed or no single-file subtitle).")
        if remaining_quota is not None and self.downloads_needed:
            shortfall = max(self.downloads_needed - remaining_quota, 0)
            days = -(-shortfall // max(daily_quota, 1))
            lines.append(f"Download quota left today: {remaining_quota}."
                         + (f" About {days} more day(s) at {daily_quota}/day after today." if days else ""))
        return "\n".join(lines)


def readiness(catalog: Catalog, src: str, tgt: str, keys: list[str], members: dict[str, list[int]]) -> Readiness:
    fids = [fid for k in keys for fid in members.get(k, [])]
    states = feature_states(catalog, src, tgt, fids)
    ready_titles = sum(all(states[f].status == READY for f in members.get(k, [])) for k in keys)
    return Readiness(
        titles=len(keys),
        features=len(fids),
        ready_features=sum(s.status == READY for s in states.values()),
        ready_titles=ready_titles,
        pending_features=sum(s.status == PENDING for s in states.values()),
        downloads_needed=sum(len(s.missing_files) for s in states.values()),
        unusable_features=sum(s.status in (FAILED, UNAVAILABLE) for s in states.values()),
        ready_pairs=sum(s.n_pairs for s in states.values()),
    )
