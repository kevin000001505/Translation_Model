"""Parse .srt subtitles into timed cues and clean their text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from charset_normalizer import from_bytes

from .languages import CJK_LANGUAGES


@dataclass
class Cue:
    start: float  # seconds
    end: float
    text: str


_TIME = r"(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
_TIMING = re.compile(_TIME + r"\s*-->\s*" + _TIME)


# Legacy encodings worth trying first for languages where detection on short files is unreliable.
_LEGACY_ENCODINGS = {"zh-tw": ("cp950", "big5hkscs", "gb18030"), "ja": ("cp932",), "ko": ("cp949",)}


def decode(raw: bytes, language: str = "") -> str:
    """OpenSubtitles serves UTF-8, but older uploads may still carry other encodings."""
    for enc in ("utf-8-sig", *_LEGACY_ENCODINGS.get(language, ())):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    best = from_bytes(raw).best()
    return str(best) if best else raw.decode("utf-8", errors="replace")


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    blocks = re.split(r"\r?\n\s*\r?\n", text.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = block.split("\n")
        for i, line in enumerate(lines):
            m = _TIMING.search(line)
            if m:
                g = m.groups()
                start, end = _seconds(*g[:4]), _seconds(*g[4:])
                body = "\n".join(lines[i + 1 :]).strip()
                if body and end > start:
                    cues.append(Cue(start, end, body))
                break
    cues.sort(key=lambda c: c.start)
    return cues


_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")  # <i>, <font ...>, {\an8}
_HEARING_IMPAIRED = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|【[^】]*】")
_SPEAKER = re.compile(r"^[A-Z][A-Z .'-]{1,30}:\s*")  # "JOHN: ..."
_LEADING_DASH = re.compile(r"^\s*[-–—‐]+\s*")
_MUSIC = re.compile(r"[♪♫#*]+")
_URL = re.compile(r"(https?://|www\.)\S+|\S+\.(com|net|org|tv)\b", re.I)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
_LATIN = re.compile(r"[A-Za-z]")

# Credit lines that subtitle groups add; dropped wholesale.
_CREDITS = re.compile(
    r"subtitle|subbed|synced|sync by|translated by|opensubtitles|字幕|翻譯|翻译|校對|校对|時間軸|时间轴|压制|壓制",
    re.I,
)


def _keep_line(line: str, language: str) -> bool:
    if not line:
        return False
    if language in CJK_LANGUAGES:
        # Bilingual Chinese subtitles often add the English line: keep only lines in the right script.
        return len(_CJK.findall(line)) >= max(1, len(_LATIN.findall(line)) // 4)
    return True


def clean_text(text: str, language: str) -> str:
    text = _TAGS.sub("", text)
    if _CREDITS.search(text) or _URL.search(text):
        return ""
    lines = []
    for line in text.split("\n"):
        line = _HEARING_IMPAIRED.sub("", line)
        line = _MUSIC.sub("", line)
        line = _LEADING_DASH.sub("", line)
        line = _SPEAKER.sub("", line).strip()
        if _keep_line(line, language):
            lines.append(line)
    # Lines of one cue (wrapped text, or one line per speaker) become one line, space separated.
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def load_cues(raw: bytes, language: str) -> list[Cue]:
    cues = []
    for cue in parse_srt(decode(raw, language)):
        text = clean_text(cue.text, language)
        if text:
            cues.append(Cue(cue.start, cue.end, text))
    return cues
