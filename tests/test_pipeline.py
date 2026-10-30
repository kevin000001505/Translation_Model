"""Scan → list → select → download within quota → align → dataset, against the fake API."""

import json

import pytest

from subtune.catalog import Catalog
from subtune.config import OpenSubtitlesSettings, Paths
from subtune.dataset import build_dataset
from subtune.listing import build_listing, listing_status
from subtune.opensubtitles import OpenSubtitlesClient
from subtune.prepare import prepare
from subtune.selection import readiness, titles_table

from fake_api import FakeOpenSubtitles, chinese, english, make_srt

N = 60


def _world(quota: int, refuse: str | None = None):
    features = {
        1: {"feature_type": "movie", "title": "Alpha", "year": 2001},
        2: {"feature_type": "movie", "title": "Beta (English only)", "year": 2002},
        3: {"feature_type": "movie", "title": "Gamma (Chinese only)", "year": 2003},
        4: {"feature_type": "movie", "title": "Delta", "year": 2004},
        10: {"feature_type": "episode", "title": "Pilot", "parent_feature_id": 100, "parent_title": "Show"},
        11: {"feature_type": "episode", "title": "Second", "parent_feature_id": 100, "parent_title": "Show"},
    }
    subs = []
    sid = 1000

    def add(fid, lang, downloads, srt):
        nonlocal sid
        sid += 1
        subs.append({"subtitle_id": sid, "feature_id": fid, "language": lang, "file_id": sid * 10,
                     "download_count": downloads, "srt": srt})

    for fid, pop in ((1, 500), (4, 50), (10, 200), (11, 100)):
        add(fid, "en", pop, make_srt(english(N, f" f{fid}")))
        add(fid, "en", 1, make_srt(english(N, " worse")))  # less popular duplicate: never downloaded
        add(fid, "zh-tw", pop, make_srt(chinese(N, f"{fid}"), offset=1.5))
    add(2, "en", 900, make_srt(english(N)))
    add(3, "zh-tw", 900, make_srt(chinese(N)))
    # English has more subtitles than Chinese, so Chinese is scanned in full.
    for fid in (1, 2):
        add(fid, "en", 1, make_srt(english(N)))
    return FakeOpenSubtitles(features, subs, quota=quota, refuse=refuse)


@pytest.fixture
def env(tmp_path):
    def make(quota=100, refuse=None):
        fake = _world(quota, refuse)
        settings = OpenSubtitlesSettings("key", "user", "pw", "subtune-test v0", rate_limit=1000)
        client = OpenSubtitlesClient(settings, session=fake, sleep=lambda s: None)
        paths = Paths(tmp_path / "data", tmp_path / "out").ensure()
        return fake, client, Catalog(paths.db), paths
    return make


def test_listing_finds_titles_with_both_languages(env):
    fake, client, cat, paths = env()
    build_listing(client, cat, "en", "zh-tw", progress=lambda m: None)
    df, members = titles_table(cat, "en", "zh-tw")
    assert list(df["Title"]) == ["Alpha", "Show", "Delta"]  # by popularity; series grouped
    assert list(df["Type"]) == ["Movie", "TV", "Movie"]
    assert members["series:100"] == [10, 11]
    assert df.loc[df["Title"] == "Show", "Episodes"].item() == 2
    # Search spends no quota.
    assert fake.downloads == []
    # Every zh-tw feature was checked for English exactly once; a second run re-checks nothing.
    n_calls = len(fake.calls)
    build_listing(client, cat, "en", "zh-tw", progress=lambda m: None)
    assert len(fake.calls) - n_calls <= 3


def test_prepare_respects_quota_and_never_redownloads(env):
    fake, client, cat, paths = env(quota=3)
    build_listing(client, cat, "en", "zh-tw", progress=lambda m: None)
    df, members = titles_table(cat, "en", "zh-tw")
    keys = list(df["Key"])
    fids = [f for k in keys for f in members[k]]

    r = readiness(cat, "en", "zh-tw", keys, members)
    assert (r.pending_features, r.downloads_needed, r.ready_features) == (4, 8, 0)

    summary = prepare(client, cat, paths, "en", "zh-tw", fids, progress=lambda m: None)
    assert summary["aligned"] == 1 and summary["still_pending"] == 3
    assert len(fake.downloads) == 3  # quota spent, then stopped

    fake.remaining = 100  # next day
    prepare(client, cat, paths, "en", "zh-tw", fids, progress=lambda m: None)
    assert len(fake.downloads) == 8 == len(set(fake.downloads))  # each file exactly once
    r = readiness(cat, "en", "zh-tw", keys, members)
    assert r.ready_features == 4 and r.downloads_needed == 0
    assert "ready" in set(titles_table(cat, "en", "zh-tw")[0]["Status"])

    # A second language pair sharing English reuses the downloaded English files.
    prepare(client, cat, paths, "en", "zh-tw", fids, progress=lambda m: None)
    assert len(fake.downloads) == 8


def test_dataset_holds_out_whole_titles(env, tmp_path):
    fake, client, cat, paths = env()
    build_listing(client, cat, "en", "zh-tw", progress=lambda m: None)
    df, members = titles_table(cat, "en", "zh-tw")
    fids = [f for k in df["Key"] for f in members[k]]
    prepare(client, cat, paths, "en", "zh-tw", fids, progress=lambda m: None)

    out = tmp_path / "run"
    info = build_dataset(cat, "en", "zh-tw", members, out, eval_fraction=0.2, eval_max=500)
    train = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    evals = [json.loads(l) for l in (out / "eval.jsonl").read_text().splitlines()]
    assert info.eval_titles >= 1 and info.train_titles >= 1
    assert not {r["feature_id"] for r in train} & {r["feature_id"] for r in evals}
    series = set(members["series:100"])
    train_f = {r["feature_id"] for r in train}
    assert series <= train_f or not (series & train_f)  # a series is never split
    assert all(r["src"] and r["tgt"] for r in train + evals)

    info = build_dataset(cat, "en", "zh-tw", members, tmp_path / "run2", max_pairs=10)
    assert info.train_pairs == 10


@pytest.mark.parametrize("refuse", ["language", "broad"])
def test_listing_works_when_the_api_refuses_broad_searches(env, monkeypatch, tmp_path, refuse):
    """The real API answers 'Not enough parameters' to a search by language alone."""
    from subtune import imdb

    monkeypatch.setenv("SUBTUNE_DATA_DIR", str(tmp_path / "data"))
    # IMDb candidates of the fake world, most votes first: movies by IMDb id, the show by its parent id
    monkeypatch.setattr(imdb, "candidates", lambda data_dir, threshold, log=print: [
        [10, "movie", 900, "Alpha", 2001], [1000, "series", 800, "Show", 2000], [20, "movie", 700, "Beta", 2002],
        [30, "movie", 600, "Gamma", 2003], [40, "movie", 500, "Delta", 2004]])
    fake, client, cat, paths = env(refuse=refuse)
    messages = []
    build_listing(client, cat, "en", "zh-tw", progress=messages.append)
    df, members = titles_table(cat, "en", "zh-tw")
    assert list(df["Title"]) == ["Alpha", "Show", "Delta"]
    assert members["series:100"] == [10, 11]
    mode = "year by year" if refuse == "language" else "by IMDb id"
    assert any(mode in m for m in messages)
    assert cat.get_lang_scan("zh-tw")["status"] == "done"
    assert "Not scanned" not in listing_status(cat, "en", "zh-tw")


def _interrupt_at_year(fake, year):
    real = fake.request

    def request(method, url, params=None, **kw):
        if dict(params or []).get("year") == str(year):
            raise KeyboardInterrupt  # the scan is stopped (or still running) at this point
        return real(method, url, params=params, **kw)

    fake.request = request
    return real


def test_titles_appear_while_the_scan_is_still_running(env, monkeypatch, tmp_path):
    """Regression: the list stayed empty until the whole year-by-year scan had finished."""
    monkeypatch.setenv("SUBTUNE_DATA_DIR", str(tmp_path / "data"))
    fake, client, cat, paths = env(refuse="language")
    real = _interrupt_at_year(fake, 2003)  # newest years first: 2004 (Delta) is already done
    with pytest.raises(KeyboardInterrupt):
        build_listing(client, cat, "en", "zh-tw", progress=lambda m: None)
    assert list(titles_table(cat, "en", "zh-tw")[0]["Title"]) == ["Delta"]
    assert cat.scan_running("zh-tw")

    fake.request = real  # resume
    build_listing(client, cat, "en", "zh-tw", progress=lambda m: None)
    assert list(titles_table(cat, "en", "zh-tw")[0]["Title"]) == ["Alpha", "Show", "Delta"]
    assert not cat.scan_running("zh-tw")


def test_resuming_an_oldest_first_scan_from_the_previous_version(env, monkeypatch, tmp_path):
    from subtune.listing import _year_slices

    monkeypatch.setenv("SUBTUNE_DATA_DIR", str(tmp_path / "data"))
    fake, client, cat, paths = env(refuse="language")
    # Saved by the previous version: years ascending, movies done up to 2001, Alpha (2001) stored
    # but never counted (presence was only computed at the very end).
    slices = sorted(_year_slices(), key=lambda sl: (sl["params"]["type"], sl["params"]["year"]))
    for sl in slices:
        if sl["params"]["type"] == "movie" and sl["params"]["year"] <= 2001:
            sl["total_pages"], sl["next_page"] = 1, 2
    with cat.tx():
        for s in fake.subs:
            if s["feature_id"] == 1 and s["language"] == "zh-tw":
                cat.record_search_item(fake._item(s))
    cat.save_lang_scan("zh-tw", "running", {"mode": "years", "slices": slices, "fetched": 1}, None)

    real = _interrupt_at_year(fake, 2003)
    with pytest.raises(KeyboardInterrupt):
        build_listing(client, cat, "en", "zh-tw", progress=lambda m: None)
    assert "Alpha" in set(titles_table(cat, "en", "zh-tw")[0]["Title"])  # found before the resume

    fake.request = real
    build_listing(client, cat, "en", "zh-tw", progress=lambda m: None)
    assert list(titles_table(cat, "en", "zh-tw")[0]["Title"]) == ["Alpha", "Show", "Delta"]
