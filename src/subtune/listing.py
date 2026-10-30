"""Find every movie/episode that has subtitles in both languages of a pair.

The API cannot filter on "has subtitles in A and B", so we:

1. Pick the rarer language as *primary* and collect every feature that has subtitles in it
   (searches cost no quota).
2. For each of those features, ask whether it has subtitles in the *secondary* language
   (one search per feature; also stores the candidates we may later download).

Step 1 uses the broadest search the API accepts. The API answers "Not enough parameters" to a
search by language alone, so the scan falls back automatically:

- ``all``    language only, paged (if the API ever allows it);
- ``years``  language + year + movie/episode, paged per year (covers the whole catalog);
- ``imdb``   language + IMDb id, for every IMDb title above a vote threshold (imdb.py), most-voted
             first; the secondary language is checked right away, so the list fills while it runs.

Both steps are resumable and cached in the catalog, and are shared between language pairs:
a scan of zh-tw done for en→zh-tw is reused for ja→zh-tw.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Callable

from . import imdb
from .catalog import Catalog
from .config import get_paths
from .opensubtitles import OpenSubtitlesClient, OpenSubtitlesError

log = logging.getLogger(__name__)

Progress = Callable[[str], None]

# Search filters applied to every query: we never want AI or machine translated subtitles.
QUALITY_FILTERS = {"ai_translated": "exclude", "machine_translated": "exclude"}
FIRST_YEAR = 1900


def _is_capped(total_pages: int, per_page: int, total_count: int) -> bool:
    return total_count > 0 and total_pages * max(per_page, 1) < total_count * 0.98


def _year_slices() -> list[dict]:
    """Newest years first, so recent (most-watched) titles show up early while the scan runs."""
    last = dt.date.today().year + 1
    return [
        {"params": {"type": ftype, "year": year}, "next_page": 1, "total_pages": None}
        for year in range(last, FIRST_YEAR - 1, -1)
        for ftype in ("movie", "episode")
    ]


def _slice_done(sl: dict) -> bool:
    return sl["total_pages"] is not None and sl["next_page"] > sl["total_pages"]


def _newest_first(state: dict) -> dict:
    """Reorder a resumed year scan (older versions went oldest-first): finished slices first, so
    they are skipped, then the remaining ones from the newest year down."""
    if state.get("mode") == "years":
        state["slices"].sort(key=lambda sl: (not _slice_done(sl), -sl["params"]["year"], sl["params"]["type"]))
    return state


def _refused(exc: OpenSubtitlesError) -> bool:
    """The API's answer to a search it considers too broad."""
    return "HTTP 400" in str(exc) and "parameters" in str(exc).lower()


def language_total(client: OpenSubtitlesClient, language: str) -> int:
    page = client.search_subtitles(languages=language, page=1, **QUALITY_FILTERS)
    return int(page.get("total_count") or 0)


def choose_primary(client: OpenSubtitlesClient, catalog: Catalog, src: str, tgt: str, progress: Progress) -> tuple[str, str]:
    """Primary = the language with fewer subtitles (cheaper to scan)."""
    for lang, other in ((src, tgt), (tgt, src)):
        scan = catalog.get_lang_scan(lang)
        if scan and scan["status"] == "done":
            # A finished scan can be reused if it is the rarer language (or the other is unknown).
            other_scan = catalog.get_lang_scan(other)
            if not other_scan or (scan["total_count"] or 0) <= (other_scan["total_count"] or 0):
                progress(f"Reusing finished scan of '{lang}' subtitles.")
                return lang, other
    try:
        counts = {lang: language_total(client, lang) for lang in (src, tgt)}
    except OpenSubtitlesError as exc:
        if not _refused(exc):
            raise
        # Totals per language are not available: English is by far the largest language, so the
        # other one is the rarer; between two non-English languages, use the target.
        primary = tgt if src == "en" or tgt != "en" else src
        return primary, (tgt if primary == src else src)
    progress(f"Subtitle counts: {counts}")
    primary = min(counts, key=counts.get)
    return primary, (tgt if primary == src else src)


def _accepts_year_search(client: OpenSubtitlesClient, language: str) -> bool:
    try:
        client.search_subtitles(languages=language, type="movie", year=2015, page=1, **QUALITY_FILTERS)
        return True
    except OpenSubtitlesError as exc:
        if _refused(exc):
            return False
        raise


def scan_language(
    client: OpenSubtitlesClient,
    catalog: Catalog,
    language: str,
    progress: Progress,
    refresh: bool = False,
    secondary: str | None = None,
) -> None:
    """Collect every feature with subtitles in `language`, storing features and candidates."""
    existing = catalog.get_lang_scan(language)
    if existing and existing["status"] == "done" and not refresh:
        progress(f"Scan of '{language}' already complete ({existing['total_count']} subtitles).")
        return
    if existing and existing["status"] == "running" and not refresh:
        state = _newest_first(existing["state"])
        total_count = existing["total_count"]
        progress(f"Resuming scan of '{language}'.")
        # Titles found before this resume (older versions counted them only at the very end).
        catalog.recount_presence(language)
        if secondary:
            check_secondary(client, catalog, language, secondary, progress)
    else:
        state = {"mode": "all", "slices": [{"params": {}, "next_page": 1, "total_pages": None}], "fetched": 0}
        total_count = None

    while state["mode"] != "imdb":
        try:
            state, total_count = _scan_paged(client, catalog, language, state, total_count, progress, secondary)
            break
        except OpenSubtitlesError as exc:
            if not _refused(exc):
                raise
            if state["mode"] == "all" and _accepts_year_search(client, language):
                progress("The API refuses searches by language alone; scanning year by year instead.")
                state = {"mode": "years", "slices": _year_slices(), "fetched": state["fetched"]}
            else:
                state = {"mode": "imdb", "next": 0, "min_votes": imdb.min_votes(), "fetched": state["fetched"],
                         "found": 0}
                progress("The API refuses broad searches; looking titles up by IMDb id instead "
                         f"(IMDb titles with at least {state['min_votes']} votes, most-voted first).")
            catalog.save_lang_scan(language, "running", state, total_count)
    if state["mode"] == "imdb":
        state = _scan_imdb(client, catalog, language, secondary, state, progress)

    catalog.recount_presence(language)
    catalog.save_lang_scan(language, "done", state, total_count)
    progress(f"Scan of '{language}' complete: {state['fetched']} subtitles, "
             f"{len(catalog.features_with(language))} titles/episodes.")


def _scan_paged(client, catalog, language, state, total_count, progress, secondary=None):
    """Page through `language` subtitles slice by slice ("all" or "years" mode). Each title found is
    counted and checked for `secondary` right away, so the list fills while the scan runs."""
    checked = _checked(catalog, secondary)
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
                found = {f[0] for item in data if (f := catalog.record_search_item(item))}
            catalog.recount_presence(language, sorted(found))
            _check_new(client, catalog, found, secondary, checked)
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
    return state, total_count


def _checked(catalog: Catalog, secondary: str | None) -> set[int]:
    if not secondary:
        return set()
    return {f for f, in catalog.conn.execute("SELECT feature_id FROM presence WHERE language=?", (secondary,))}


def _check_new(client, catalog, feature_ids, secondary, checked) -> int:
    """Check `secondary` for features not checked yet; returns how many have it."""
    found = 0
    if secondary:
        for feature_id in sorted(feature_ids):
            if feature_id not in checked:
                found += _check_feature(client, catalog, feature_id, secondary) > 0
                checked.add(feature_id)
    return found


def _check_feature(client: OpenSubtitlesClient, catalog: Catalog, feature_id: int, language: str) -> int:
    """Store `language` subtitle candidates of one feature; returns how many it has."""
    page = client.search_subtitles(
        id=feature_id, languages=language, order_by="download_count", order_direction="desc", **QUALITY_FILTERS
    )
    with catalog.tx():
        for item in page.get("data") or []:
            catalog.record_search_item(item)
        count = int(page.get("total_count") or 0)
        catalog.set_presence(feature_id, language, count)
    return count


def _scan_imdb(client, catalog, language, secondary, state, progress) -> dict:
    """Look up `language` subtitles for each IMDb candidate title, resuming at state["next"]."""
    titles = imdb.candidates(get_paths().data, state["min_votes"], progress)
    state["total"] = len(titles)
    checked = _checked(catalog, secondary)
    for n in range(state["next"], len(titles)):
        imdb_id, kind, _votes, _title, _year = titles[n]
        id_param = {"imdb_id": imdb_id} if kind == "movie" else {"parent_imdb_id": imdb_id}
        counts: dict[int, int] = {}
        page_no, total_pages = 1, 1
        while page_no <= total_pages:
            page = client.search_subtitles(languages=language, page=page_no, **id_param, **QUALITY_FILTERS)
            total_pages = int(page.get("total_pages") or 0)
            with catalog.tx():
                for item in page.get("data") or []:
                    found = catalog.record_search_item(item)
                    if found:
                        counts[found[0]] = counts.get(found[0], 0) + 1
            state["fetched"] += len(page.get("data") or [])
            page_no += 1
        with catalog.tx():
            for feature_id, count in counts.items():
                catalog.set_presence(feature_id, language, count)
        state["found"] += _check_new(client, catalog, counts, secondary, checked)
        state["next"] = n + 1
        if state["next"] % 20 == 0 or state["next"] == len(titles):
            catalog.save_lang_scan(language, "running", state, None)
        if state["next"] % 200 == 0 or state["next"] == len(titles):
            both = f"; {state['found']} movies/episodes have both languages" if secondary else ""
            progress(f"IMDb titles checked: {state['next']}/{len(titles)}{both}. "
                     "Reload the table to see them while the scan continues.")
    return state


def check_secondary(
    client: OpenSubtitlesClient, catalog: Catalog, primary: str, secondary: str, progress: Progress
) -> None:
    """For each feature with `primary` subtitles, look up its `secondary` subtitles."""
    todo = catalog.unchecked_features(primary, secondary)
    catalog.set_pair_check(primary, secondary, "running")
    progress(f"Checking {len(todo)} titles/episodes for '{secondary}' subtitles.")
    found = 0
    for n, feature_id in enumerate(todo, 1):
        found += _check_feature(client, catalog, feature_id, secondary) > 0
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
    progress(f"Primary (scanned): '{primary}'. Secondary (checked per title): '{secondary}'.")
    scan_language(client, catalog, primary, progress, refresh=refresh, secondary=secondary)
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
            st = scan["state"]
            how = (f", by IMDb id: {st.get('next', 0)}/{st.get('total', '?')} titles with ≥{st['min_votes']} votes"
                   if st.get("mode") == "imdb" else "")
            parts.append(f"scan of '{primary}': {scan['status']}{how} ({when}); check of '{secondary}': {check}")
    return "; ".join(parts) if parts else "Not scanned yet."
