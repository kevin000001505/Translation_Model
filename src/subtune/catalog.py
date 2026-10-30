"""SQLite catalog: what OpenSubtitles has, what is downloaded, and what is aligned.

Everything fetched from the API is recorded here so it is never fetched twice:

- ``features``     movies and episodes seen in search results
- ``subtitles``    subtitle candidates per feature and language (from search, no quota)
- ``presence``     how many usable subtitles a feature has in a language (search-time count)
- ``lang_scans``   resumable state of the full scan of one language's subtitles
- ``files``        downloaded .srt files (quota spent once, kept forever)
- ``alignments``   result of aligning a feature's two subtitles for a language pair
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    feature_id INTEGER PRIMARY KEY,
    feature_type TEXT NOT NULL,          -- 'movie' or 'episode'
    title TEXT,
    year INTEGER,
    imdb_id INTEGER,
    tmdb_id INTEGER,
    parent_feature_id INTEGER,
    parent_title TEXT,
    parent_imdb_id INTEGER,
    season_number INTEGER,
    episode_number INTEGER,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS subtitles (
    subtitle_id TEXT PRIMARY KEY,
    feature_id INTEGER NOT NULL,
    language TEXT NOT NULL,
    file_id INTEGER,
    n_files INTEGER,
    download_count INTEGER,
    hearing_impaired INTEGER,
    foreign_parts_only INTEGER,
    ai_translated INTEGER,
    machine_translated INTEGER,
    from_trusted INTEGER,
    fps REAL,
    release TEXT,
    upload_date TEXT
);
CREATE INDEX IF NOT EXISTS subtitles_feature_lang ON subtitles(feature_id, language);
CREATE TABLE IF NOT EXISTS presence (
    feature_id INTEGER NOT NULL,
    language TEXT NOT NULL,
    n_subs INTEGER NOT NULL,
    checked_at REAL,
    PRIMARY KEY (feature_id, language)
);
CREATE INDEX IF NOT EXISTS presence_lang ON presence(language, n_subs);
CREATE TABLE IF NOT EXISTS lang_scans (
    language TEXT PRIMARY KEY,
    status TEXT NOT NULL,                -- 'running' or 'done'
    state TEXT NOT NULL,                 -- JSON, see listing.py
    total_count INTEGER,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS pair_checks (
    primary_lang TEXT NOT NULL,
    secondary_lang TEXT NOT NULL,
    status TEXT NOT NULL,                -- 'running' or 'done'
    updated_at REAL,
    PRIMARY KEY (primary_lang, secondary_lang)
);
CREATE TABLE IF NOT EXISTS files (
    file_id INTEGER PRIMARY KEY,
    subtitle_id TEXT,
    language TEXT,
    path TEXT NOT NULL,
    downloaded_at REAL
);
CREATE TABLE IF NOT EXISTS alignments (
    pair TEXT NOT NULL,                  -- languages.pair_key(src, tgt)
    feature_id INTEGER NOT NULL,
    src_file_id INTEGER,
    tgt_file_id INTEGER,
    status TEXT NOT NULL,                -- 'ok' or 'failed'
    n_pairs INTEGER,
    coverage REAL,
    message TEXT,
    path TEXT,
    created_at REAL,
    PRIMARY KEY (pair, feature_id)
);
"""


def _bool(v: Any) -> int:
    return 1 if v else 0


@dataclass
class SubtitleRow:
    subtitle_id: str
    feature_id: int
    language: str
    file_id: int | None
    n_files: int
    download_count: int
    hearing_impaired: bool
    foreign_parts_only: bool
    ai_translated: bool
    machine_translated: bool
    from_trusted: bool


class Catalog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # --- search results -------------------------------------------------------------------

    def record_search_item(self, item: dict) -> tuple[int, str] | None:
        """Store one /subtitles result item; returns (feature_id, language) or None if unusable."""
        attrs = item.get("attributes") or {}
        fd = attrs.get("feature_details") or {}
        feature_id = fd.get("feature_id")
        language = (attrs.get("language") or "").lower()
        if not feature_id or not language:
            return None
        ftype = "episode" if (fd.get("feature_type") or "").lower() == "episode" else "movie"
        now = time.time()
        files = attrs.get("files") or []
        self.conn.execute(
            """INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(feature_id) DO UPDATE SET
                 feature_type=excluded.feature_type, title=excluded.title, year=excluded.year,
                 imdb_id=excluded.imdb_id, tmdb_id=excluded.tmdb_id,
                 parent_feature_id=excluded.parent_feature_id, parent_title=excluded.parent_title,
                 parent_imdb_id=excluded.parent_imdb_id, season_number=excluded.season_number,
                 episode_number=excluded.episode_number, updated_at=excluded.updated_at""",
            (
                int(feature_id),
                ftype,
                fd.get("title") or fd.get("movie_name"),
                fd.get("year"),
                fd.get("imdb_id"),
                fd.get("tmdb_id"),
                fd.get("parent_feature_id"),
                fd.get("parent_title"),
                fd.get("parent_imdb_id"),
                fd.get("season_number"),
                fd.get("episode_number"),
                now,
            ),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO subtitles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(attrs.get("subtitle_id") or item.get("id")),
                int(feature_id),
                language,
                files[0].get("file_id") if files else None,
                len(files),
                int(attrs.get("download_count") or 0),
                _bool(attrs.get("hearing_impaired")),
                _bool(attrs.get("foreign_parts_only")),
                _bool(attrs.get("ai_translated")),
                _bool(attrs.get("machine_translated")),
                _bool(attrs.get("from_trusted")),
                attrs.get("fps"),
                attrs.get("release"),
                attrs.get("upload_date"),
            ),
        )
        return int(feature_id), language

    def set_presence(self, feature_id: int, language: str, n_subs: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO presence VALUES (?,?,?,?)", (feature_id, language, n_subs, time.time())
        )

    def recount_presence(self, language: str, feature_ids: list[int] | None = None) -> None:
        """Set presence for a language from the subtitles stored for it (all features, or some)."""
        now = time.time()
        only = ""
        args: list = [now, language]
        if feature_ids is not None:
            if not feature_ids:
                return
            only = f" AND feature_id IN ({','.join('?' * len(feature_ids))})"
            args += list(feature_ids)
        with self.tx() as c:
            c.execute(
                f"""INSERT OR REPLACE INTO presence
                   SELECT feature_id, language, COUNT(*), ? FROM subtitles
                   WHERE language=? AND ai_translated=0 AND machine_translated=0{only}
                   GROUP BY feature_id""",
                args,
            )

    def scan_running(self, *languages: str) -> bool:
        return self.scan_marker(*languages)[1]

    def scan_marker(self, *languages: str) -> tuple[float, bool]:
        """(last time a scan of these languages saved progress, whether one is running)."""
        marks = ",".join("?" * len(languages))
        row = self.conn.execute(
            f"SELECT MAX(updated_at), MAX(status='running') FROM lang_scans WHERE language IN ({marks})",
            languages,
        ).fetchone()
        return float(row[0] or 0), bool(row[1])

    # --- scan state -----------------------------------------------------------------------

    def get_lang_scan(self, language: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM lang_scans WHERE language=?", (language,)).fetchone()
        if not row:
            return None
        return {"status": row["status"], "state": json.loads(row["state"]), "total_count": row["total_count"],
                "updated_at": row["updated_at"]}

    def save_lang_scan(self, language: str, status: str, state: dict, total_count: int | None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO lang_scans VALUES (?,?,?,?,?)",
                (language, status, json.dumps(state), total_count, time.time()),
            )

    def reset_lang_scan(self, language: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM lang_scans WHERE language=?", (language,))

    def get_pair_check(self, primary: str, secondary: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM pair_checks WHERE primary_lang=? AND secondary_lang=?", (primary, secondary)
        ).fetchone()
        return row["status"] if row else None

    def set_pair_check(self, primary: str, secondary: str, status: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO pair_checks VALUES (?,?,?,?)", (primary, secondary, status, time.time()))

    def features_with(self, language: str) -> list[int]:
        rows = self.conn.execute("SELECT feature_id FROM presence WHERE language=? AND n_subs>0", (language,))
        return [r[0] for r in rows]

    def unchecked_features(self, primary: str, secondary: str) -> list[int]:
        rows = self.conn.execute(
            """SELECT p.feature_id FROM presence p
               LEFT JOIN presence s ON s.feature_id=p.feature_id AND s.language=?
               WHERE p.language=? AND p.n_subs>0 AND s.feature_id IS NULL
               ORDER BY p.feature_id""",
            (secondary, primary),
        )
        return [r[0] for r in rows]

    # --- listing --------------------------------------------------------------------------

    def pair_features(self, src: str, tgt: str) -> list[sqlite3.Row]:
        """Every feature that has usable subtitles in both languages, with popularity."""
        return self.conn.execute(
            """SELECT f.*, ps.n_subs AS n_src, pt.n_subs AS n_tgt,
                      (SELECT COALESCE(SUM(download_count),0) FROM subtitles s
                        WHERE s.feature_id=f.feature_id AND s.language IN (?,?)) AS popularity
               FROM features f
               JOIN presence ps ON ps.feature_id=f.feature_id AND ps.language=? AND ps.n_subs>0
               JOIN presence pt ON pt.feature_id=f.feature_id AND pt.language=? AND pt.n_subs>0""",
            (src, tgt, src, tgt),
        ).fetchall()

    # --- subtitles, files and alignments --------------------------------------------------

    def candidates(self, feature_id: int, language: str) -> list[SubtitleRow]:
        rows = self.conn.execute(
            """SELECT subtitle_id, feature_id, language, file_id, n_files, download_count, hearing_impaired,
                      foreign_parts_only, ai_translated, machine_translated, from_trusted
               FROM subtitles WHERE feature_id=? AND language=?""",
            (feature_id, language),
        ).fetchall()
        return [SubtitleRow(**dict(r)) for r in rows]

    def file_path(self, file_id: int) -> Path | None:
        row = self.conn.execute("SELECT path FROM files WHERE file_id=?", (file_id,)).fetchone()
        if row and Path(row["path"]).exists():
            return Path(row["path"])
        return None

    def downloaded_file_ids(self) -> set[int]:
        return {r[0] for r in self.conn.execute("SELECT file_id FROM files")}

    def record_file(self, file_id: int, subtitle_id: str, language: str, path: Path) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?)",
                (file_id, subtitle_id, language, str(path), time.time()),
            )

    def alignment(self, pair: str, feature_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM alignments WHERE pair=? AND feature_id=?", (pair, feature_id)
        ).fetchone()

    def alignments(self, pair: str) -> dict[int, sqlite3.Row]:
        rows = self.conn.execute("SELECT * FROM alignments WHERE pair=?", (pair,)).fetchall()
        return {r["feature_id"]: r for r in rows}

    def record_alignment(
        self,
        pair: str,
        feature_id: int,
        src_file_id: int,
        tgt_file_id: int,
        status: str,
        n_pairs: int,
        coverage: float,
        message: str,
        path: Path | None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO alignments VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pair, feature_id, src_file_id, tgt_file_id, status, n_pairs, coverage, message,
                 str(path) if path else None, time.time()),
            )
