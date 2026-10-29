"""Find every movie/episode that has subtitles in both languages of a pair.

The API cannot filter on "has subtitles in A and B", so we:

1. Pick the rarer language as *primary* and page through all of its subtitles (no quota cost).
   This gives every feature that has the primary language.
2. For each of those features, ask whether it has subtitles in the *secondary* language
   (one search per feature; also stores the candidates we may later download).

Both steps are resumable and cached in the catalog, and are shared between language pairs:
a scan of zh-tw done for en→zh-tw is reused for ja→zh-tw.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Callable

from .catalog import Catalog
from .opensubtitles import OpenSubtitlesClient

log = logging.getLogger(__name__)

Progress = Callable[[str], None]

# Search filters applied to every query: we never want AI or machine translated subtitles.
QUALITY_FILTERS = {"ai_translated": "exclude", "machine_translated": "exclude"}
FIRST_YEAR = 1900


def _is_capped(total_pages: int, per_page: int, total_count: int) -> bool:
    return total_count > 0 and total_pages * max(per_page, 1) < total_count * 0.98


def _year_slices() -> list[dict]:
    last = dt.date.today().year + 1
    return [
        {"params": {"type": ftype, "year": year}, "next_page": 1, "total_pages": None}
        for ftype in ("movie", "episode")
        for year in range(FIRST_YEAR, last + 1)
    ]


def language_total(client: OpenSubtitlesClient, language: str) -> int:
    page = client.search_subtitles(languages=language, page=1, **QUALITY_FILTERS)
    return int(page.get("total_count") or 0)


def choose_primary(client: OpenSubtitlesClient, catalog: Catalog, src: str, tgt: str, progress: Progress) -> tuple[str, str]:
    """Primary = the language with fewer subtitles (cheaper to scan in full)."""
    for lang, other in ((src, tgt), (tgt, src)):
        scan = catalog.get_lang_scan(lang)
        if scan and scan["status"] == "done":
            # A finished scan can be reused if it is the rarer language (or the other is unknown).
            other_scan = catalog.get_lang_scan(other)
            if not other_scan or (scan["total_count"] or 0) <= (other_scan["total_count"] or 0):
                progress(f"Reusing finished scan of '{lang}' subtitles.")
                return lang, other
    counts = {lang: language_total(client, lang) for lang in (src, tgt)}
    progress(f"Subtitle counts: {counts}")
    primary = min(counts, key=counts.get)
    return primary, (tgt if primary == src else src)


def scan_language(
    client: OpenSubtitlesClient, catalog: Catalog, language: str, progress: Progress, refresh: bool = False
) -> None:
    """Page through every subtitle in `language`, storing features and candidates."""
    existing = catalog.get_lang_scan(language)
    if existing and existing["status"] == "done" and not refresh:
        progress(f"Scan of '{language}' already complete ({existing['total_count']} subtitles).")
        return
    if existing and existing["status"] == "running" and not refresh:
        state = existing["state"]
        total_count = existing["total_count"]
        progress(f"Resuming scan of '{language}'.")
    else:
        state = {"mode": "all", "slices": [{"params": {}, "next_page": 1, "total_pages": None}], "fetched": 0}
        total_count = None

    i = 0
    while i < len(state["slices"]):
        sl = state["slices"][i]
        while sl["total_pages"] is None or sl["next_page"] <= sl["total_pages"]:
            params = {
                "languages": language,
                "order_by": "upload_date",
                "order_direction": "asc",
                "page": sl["next_page"],
                **QUALITY_FILTERS,
                **sl["params"],
            }
            page = client.search_subtitles(**params)
            total_pages = int(page.get("total_pages") or 0)
            data = page.get("data") or []
            if sl["total_pages"] is None:
                sl["total_pages"] = total_pages
                if state["mode"] == "all":
                    total_count = int(page.get("total_count") or 0)
                    if _is_capped(total_pages, int(page.get("per_page") or len(data)), total_count):
                        progress(
                            f"Search results for '{language}' are capped at {total_pages} pages; "
                            "switching to per-year scanning."
                        )
                        state = {"mode": "years", "slices": _year_slices(), "fetched": state["fetched"]}
                        i = -1
                        break
                elif _is_capped(total_pages, int(page.get("per_page") or len(data)), int(page.get("total_count") or 0)):
                    progress(f"Warning: results for {sl['params']} are capped; some titles may be missed.")
            if not data and sl["next_page"] <= total_pages and state["mode"] == "all":
                progress(f"Empty page {sl['next_page']}/{total_pages}; switching to per-year scanning.")
                state = {"mode": "years", "slices": _year_slices(), "fetched": state["fetched"]}
                i = -1
                break
            with catalog.tx():
                for item in data:
                    catalog.record_search_item(item)
            state["fetched"] += len(data)
            sl["next_page"] += 1
            catalog.save_lang_scan(language, "running", state, total_count)
            if sl["next_page"] % 20 == 0 or sl["next_page"] > total_pages:
                where = f" ({sl['params']})" if sl["params"] else ""
                progress(f"'{language}'{where}: page {sl['next_page'] - 1}/{total_pages}, "
                         f"{state['fetched']} subtitles so far")
            if total_pages == 0:
                break
        i += 1

    catalog.recount_presence(language)
    catalog.save_lang_scan(language, "done", state, total_count)
    progress(f"Scan of '{language}' complete: {state['fetched']} subtitles, "
             f"{len(catalog.features_with(language))} titles/episodes.")


def check_secondary(
    client: OpenSubtitlesClient, catalog: Catalog, primary: str, secondary: str, progress: Progress
) -> None:
    """For each feature with `primary` subtitles, look up its `secondary` subtitles."""
    todo = catalog.unchecked_features(primary, secondary)
    catalog.set_pair_check(primary, secondary, "running")
    progress(f"Checking {len(todo)} titles/episodes for '{secondary}' subtitles.")
    found = 0
    for n, feature_id in enumerate(todo, 1):
        page = client.search_subtitles(
            id=feature_id, languages=secondary, order_by="download_count", order_direction="desc", **QUALITY_FILTERS
        )
        with catalog.tx():
            for item in page.get("data") or []:
                catalog.record_search_item(item)
            count = int(page.get("total_count") or 0)
            catalog.set_presence(feature_id, secondary, count)
        found += count > 0
        if n % 100 == 0 or n == len(todo):
            progress(f"Checked {n}/{len(todo)}; {found} have '{secondary}' so far.")
    catalog.set_pair_check(primary, secondary, "done")


def build_listing(
    client: OpenSubtitlesClient,
    catalog: Catalog,
    src: str,
    tgt: str,
    progress: Progress = print,
    refresh: bool = False,
) -> None:
    primary, secondary = choose_primary(client, catalog, src, tgt, progress)
    progress(f"Primary (scanned in full): '{primary}'. Secondary (checked per title): '{secondary}'.")
    scan_language(client, catalog, primary, progress, refresh=refresh)
    check_secondary(client, catalog, primary, secondary, progress)
    progress(f"Done: {len(catalog.pair_features(src, tgt))} titles/episodes have both '{src}' and '{tgt}'.")


def listing_status(catalog: Catalog, src: str, tgt: str) -> str:
    """One-line description of how complete the cached listing for a pair is."""
    parts = []
    for primary, secondary in ((src, tgt), (tgt, src)):
        scan = catalog.get_lang_scan(primary)
        if scan:
            check = catalog.get_pair_check(primary, secondary) or "not started"
            when = dt.datetime.fromtimestamp(scan["updated_at"]).strftime("%Y-%m-%d %H:%M")
            parts.append(f"scan of '{primary}': {scan['status']} ({when}); check of '{secondary}': {check}")
    return "; ".join(parts) if parts else "Not scanned yet."
