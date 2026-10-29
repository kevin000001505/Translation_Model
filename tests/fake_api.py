"""In-memory stand-in for the OpenSubtitles API, used by the tests."""

from __future__ import annotations

import json
from urllib.parse import urlparse


def make_srt(lines: list[str], offset: float = 0.0, scale: float = 1.0, gap: float = 3.0) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    out = []
    for i, text in enumerate(lines):
        start = (10 + i * gap) * scale + offset
        end = start + 2.0 * scale
        out.append(f"{i + 1}\n{ts(start)} --> {ts(end)}\n{text}\n")
    return "\n".join(out)


def english(n: int, tag: str = "") -> list[str]:
    return [f"This is line number {i}{tag}, my friend." for i in range(n)]


def chinese(n: int, tag: str = "") -> list[str]:
    return [f"這是第{i}句台詞{tag}，朋友。" for i in range(n)]


class _Resp:
    def __init__(self, status: int, body=None, content: bytes = b"", headers=None):
        self.status_code = status
        self._body = body
        self.content = content
        self.text = json.dumps(body) if body is not None else content.decode("utf-8", "replace")
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeOpenSubtitles:
    """features: {feature_id: dict(feature_type, title, year, parent_feature_id, parent_title)}
    subs: list of dict(subtitle_id, feature_id, language, file_id, download_count, srt)"""

    def __init__(self, features: dict, subs: list[dict], quota: int = 100, per_page: int = 2):
        self.features = features
        self.subs = subs
        self.remaining = quota
        self.per_page = per_page
        self.downloads: list[int] = []
        self.calls: list[tuple[str, str, dict]] = []

    def _item(self, s: dict) -> dict:
        f = self.features[s["feature_id"]]
        return {
            "id": str(s["subtitle_id"]),
            "type": "subtitle",
            "attributes": {
                "subtitle_id": str(s["subtitle_id"]),
                "language": s["language"],
                "download_count": s.get("download_count", 10),
                "hearing_impaired": False,
                "ai_translated": False,
                "machine_translated": False,
                "from_trusted": False,
                "foreign_parts_only": False,
                "files": [{"file_id": s["file_id"], "cd_number": 1, "file_name": "x.srt"}],
                "feature_details": {
                    "feature_id": s["feature_id"],
                    "feature_type": "Episode" if f["feature_type"] == "episode" else "Movie",
                    "title": f["title"],
                    "year": f.get("year", 2000),
                    "imdb_id": s["feature_id"] * 10,
                    "parent_feature_id": f.get("parent_feature_id"),
                    "parent_title": f.get("parent_title"),
                    "parent_imdb_id": f.get("parent_feature_id") and f["parent_feature_id"] * 10,
                },
            },
        }

    # requests.Session interface used by the client
    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        path = urlparse(url).path.replace("/api/v1", "")
        q = dict(params or [])
        self.calls.append((method, path, q))
        assert "Api-Key" in headers
        if params:
            assert [k for k, _ in params] == sorted(k for k, _ in params), "params must be sorted"
        if path == "/login":
            return _Resp(200, {"token": "tok", "base_url": "api.opensubtitles.com", "user": {"vip": True}})
        if path == "/infos/user":
            return _Resp(200, {"data": {"allowed_downloads": 1000, "remaining_downloads": self.remaining,
                                        "vip": True}})
        if path == "/subtitles":
            langs = set(q["languages"].split(","))
            rows = [s for s in self.subs if s["language"] in langs]
            if "id" in q:
                rows = [s for s in rows if str(s["feature_id"]) == q["id"]]
            if "type" in q:
                rows = [s for s in rows if self.features[s["feature_id"]]["feature_type"] == q["type"]]
            page = int(q.get("page", 1))
            pages = max(1, -(-len(rows) // self.per_page))
            chunk = rows[(page - 1) * self.per_page: page * self.per_page]
            return _Resp(200, {"total_pages": pages if rows else 0, "total_count": len(rows),
                               "per_page": self.per_page, "page": page, "data": [self._item(s) for s in chunk]})
        if path == "/download":
            if self.remaining <= 0:
                return _Resp(406, {"message": "quota exceeded", "reset_time_utc": "2026-10-30T00:00:00Z"})
            self.remaining -= 1
            self.downloads.append(json["file_id"])
            return _Resp(200, {"link": f"https://dl/{json['file_id']}", "requests": len(self.downloads),
                               "remaining": self.remaining, "reset_time_utc": "2026-10-30T00:00:00Z"})
        return _Resp(404, {"message": "not found"})

    def get(self, url, headers=None, timeout=None):
        file_id = int(url.rsplit("/", 1)[1])
        sub = next(s for s in self.subs if s["file_id"] == file_id)
        return _Resp(200, content=sub["srt"].encode("utf-8"))
