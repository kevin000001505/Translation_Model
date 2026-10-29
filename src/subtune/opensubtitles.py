"""Minimal OpenSubtitles REST API client (https://opensubtitles.stoplight.io).

Covers the calls this project needs: login, subtitle search, user quota, and download.
Searches are not quota-limited; only POST /download counts against the daily download quota.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .config import OpenSubtitlesSettings

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.opensubtitles.com/api/v1"


class OpenSubtitlesError(RuntimeError):
    pass


class QuotaExceeded(OpenSubtitlesError):
    def __init__(self, message: str, reset_time_utc: str | None = None):
        super().__init__(message)
        self.reset_time_utc = reset_time_utc


@dataclass
class Quota:
    allowed: int
    remaining: int
    reset_time_utc: str | None = None
    vip: bool = False


class _RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / max(per_second, 0.1)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._next > now:
                time.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


class OpenSubtitlesClient:
    def __init__(
        self,
        settings: OpenSubtitlesSettings,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.session = session or requests.Session()
        self.base_url = DEFAULT_BASE_URL
        self.token: str | None = None
        self.user: dict[str, Any] = {}
        self._limiter = _RateLimiter(settings.rate_limit)
        self._sleep = sleep

    # --- plumbing -------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.settings.api_key:
            raise OpenSubtitlesError("OPENSUBTITLES_API_KEY is not set in .env")
        headers = {
            "Api-Key": self.settings.api_key,
            "User-Agent": self.settings.user_agent,
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> dict:
        # The API answers non-canonical query strings with a redirect: send parameters sorted
        # alphabetically, with lowercase values.
        query = None
        if params:
            query = [(k, str(v).lower()) for k, v in sorted(params.items()) if v is not None]
        for attempt in range(6):
            self._limiter.wait()
            try:
                resp = self.session.request(
                    method, self.base_url + path, params=query, json=json, headers=self._headers(), timeout=60
                )
            except requests.RequestException as exc:
                if attempt == 5:
                    raise OpenSubtitlesError(f"{method} {path} failed: {exc}") from exc
                self._sleep(2**attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = float(resp.headers.get("ratelimit-reset") or resp.headers.get("retry-after") or 2**attempt)
                log.warning("%s %s -> HTTP %s, retrying in %.0fs", method, path, resp.status_code, wait)
                self._sleep(max(wait, 1.0))
                continue
            return self._parse(resp, method, path)
        raise OpenSubtitlesError(f"{method} {path} kept failing (rate limit or server error)")

    @staticmethod
    def _parse(resp: requests.Response, method: str, path: str) -> dict:
        try:
            body = resp.json()
        except ValueError:
            body = {"message": resp.text[:300]}
        if resp.status_code == 406 and path == "/download":
            raise QuotaExceeded(body.get("message", "Download quota exceeded"), body.get("reset_time_utc"))
        if resp.status_code >= 400:
            message = body.get("message") or body.get("errors") or body
            raise OpenSubtitlesError(f"{method} {path} -> HTTP {resp.status_code}: {message}")
        return body

    # --- endpoints ------------------------------------------------------------------------

    def login(self) -> dict:
        """Log in to get a token; VIP accounts are then served from vip-api with a higher quota."""
        if self.token:
            return self.user
        missing = self.settings.missing()
        if missing:
            raise OpenSubtitlesError(f"Missing in .env: {', '.join(missing)}")
        body = self._request(
            "POST", "/login", json={"username": self.settings.username, "password": self.settings.password}
        )
        self.token = body["token"]
        self.user = body.get("user", {})
        host = body.get("base_url")
        if host:
            self.base_url = f"https://{host}/api/v1"
        return self.user

    def search_subtitles(self, **params: Any) -> dict:
        """GET /subtitles. Returns the raw page: total_pages, total_count, page, data[]."""
        return self._request("GET", "/subtitles", params=params)

    def quota(self) -> Quota:
        self.login()
        data = self._request("GET", "/infos/user")["data"]
        return Quota(
            allowed=int(data.get("allowed_downloads", 0)),
            remaining=int(data.get("remaining_downloads", 0)),
            vip=bool(data.get("vip", False)),
        )

    def download(self, file_id: int) -> tuple[bytes, Quota]:
        """Request a download link (this is what spends quota) and fetch the file (always UTF-8)."""
        self.login()
        body = self._request("POST", "/download", json={"file_id": int(file_id)})
        resp = self.session.get(body["link"], headers={"User-Agent": self.settings.user_agent}, timeout=120)
        resp.raise_for_status()
        quota = Quota(
            allowed=int(body.get("requests", 0)) + int(body.get("remaining", 0)),
            remaining=int(body.get("remaining", 0)),
            reset_time_utc=body.get("reset_time_utc"),
        )
        return resp.content, quota
