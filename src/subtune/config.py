"""Environment loading and on-disk layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env once on import so every entry point (UI, CLI, Modal launcher) sees the same values.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    data: Path
    outputs: Path

    @property
    def db(self) -> Path:
        return self.data / "catalog.sqlite"

    @property
    def subtitles(self) -> Path:
        """Raw downloaded .srt files, one per OpenSubtitles file_id. Never re-downloaded."""
        return self.data / "subtitles"

    @property
    def pairs(self) -> Path:
        """Aligned sentence pairs, one JSONL per (language pair, feature)."""
        return self.data / "pairs"

    @property
    def runs(self) -> Path:
        """Working directories for training runs (datasets, logs, configs)."""
        return self.data / "runs"

    @property
    def logs(self) -> Path:
        return self.data / "logs"

    @property
    def models(self) -> Path:
        """Finished, tagged fine-tuned models."""
        return self.outputs / "models"

    def ensure(self) -> "Paths":
        for p in (self.data, self.subtitles, self.pairs, self.runs, self.logs, self.models):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    data = Path(os.environ.get("SUBTUNE_DATA_DIR") or PROJECT_ROOT / "data")
    outputs = Path(os.environ.get("SUBTUNE_OUTPUT_DIR") or PROJECT_ROOT / "outputs")
    return Paths(data=data.resolve(), outputs=outputs.resolve()).ensure()


@dataclass(frozen=True)
class OpenSubtitlesSettings:
    api_key: str
    username: str
    password: str
    user_agent: str
    rate_limit: float

    @classmethod
    def from_env(cls) -> "OpenSubtitlesSettings":
        return cls(
            api_key=os.environ.get("OPENSUBTITLES_API_KEY", "").strip(),
            username=os.environ.get("OPENSUBTITLES_USERNAME", "").strip(),
            password=os.environ.get("OPENSUBTITLES_PASSWORD", "").strip(),
            user_agent=os.environ.get("OPENSUBTITLES_USER_AGENT", "").strip() or "subtune v0.1",
            rate_limit=float(os.environ.get("OPENSUBTITLES_RATE_LIMIT") or 5),
        )

    def missing(self) -> list[str]:
        names = {
            "OPENSUBTITLES_API_KEY": self.api_key,
            "OPENSUBTITLES_USERNAME": self.username,
            "OPENSUBTITLES_PASSWORD": self.password,
        }
        return [k for k, v in names.items() if not v]


def modal_configured() -> bool:
    return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
