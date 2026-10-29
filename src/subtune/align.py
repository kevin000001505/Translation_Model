"""Align two subtitle tracks of the same movie into sentence pairs, using their timings.

Two subtitle files for one movie usually come from different releases, so their clocks differ by
a constant offset and sometimes by a frame-rate factor (e.g. 23.976 vs 25 fps). We first estimate
that transform by maximising the overlap of "someone is speaking" masks, then pair cues whose
time spans overlap, allowing 1-to-many merges (one long line split into two cues on one side).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .languages import CJK_LANGUAGES
from .srt import Cue, _CJK

RESOLUTION = 0.1  # seconds per mask bin
MAX_SHIFT = 90.0  # seconds; larger offsets mean a different cut of the film
FPS_RATIOS = (1.0, 25 / 23.976, 23.976 / 25, 24 / 23.976, 23.976 / 24, 25 / 24, 24 / 25)
MIN_OVERLAP = 0.5  # fraction of the shorter cue that must overlap
MAX_GROUP = 3  # max cues per side merged into one pair
MIN_COVERAGE = 0.5  # fraction of cues that must end up paired, else the files don't match
MIN_PAIRS = 30


@dataclass
class Alignment:
    pairs: list[tuple[str, str, float]]  # (src text, tgt text, start time)
    coverage: float
    scale: float
    offset: float
    ok: bool
    message: str


def _mask(cues: list[Cue], length: int, scale: float = 1.0) -> np.ndarray:
    m = np.zeros(length, dtype=np.float32)
    for c in cues:
        a, b = int(c.start * scale / RESOLUTION), int(c.end * scale / RESOLUTION)
        m[max(a, 0) : max(min(b, length), 0)] = 1.0
    return m


def estimate_transform(src: list[Cue], tgt: list[Cue]) -> tuple[float, float]:
    """Return (scale, offset) such that tgt time * scale + offset ≈ src time."""
    end = max(src[-1].end, max(c.end for c in tgt) * max(FPS_RATIOS)) + MAX_SHIFT
    n = int(end / RESOLUTION) + 1
    size = 1 << (2 * n - 1).bit_length()
    src_f = np.fft.rfft(_mask(src, n), size)
    max_lag = int(MAX_SHIFT / RESOLUTION)
    best = (-1.0, 1.0, 0.0)
    for scale in FPS_RATIOS:
        tgt_m = _mask(tgt, n, scale)
        # corr[k] = sum_i src[i + k] * tgt[i]: overlap when tgt is shifted later by k bins.
        corr = np.fft.irfft(src_f * np.conj(np.fft.rfft(tgt_m, size)), size)
        lags = np.concatenate([corr[: max_lag + 1], corr[-max_lag:]])
        shifts = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
        k = int(np.argmax(lags))
        if lags[k] > best[0]:
            best = (float(lags[k]), scale, shifts[k] * RESOLUTION)
    return best[1], best[2]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _good_pair(src: str, tgt: str, src_lang: str, tgt_lang: str) -> bool:
    if not src or not tgt or src == tgt:
        return False
    if len(src) > 400 or len(tgt) > 400:
        return False
    for text, lang in ((src, src_lang), (tgt, tgt_lang)):
        if lang in CJK_LANGUAGES and not _CJK.search(text):
            return False
    # Extremely unbalanced lengths are almost always misalignments.
    a, b = len(src), len(tgt)
    return max(a, b) <= 12 * min(a, b) + 10


def align(src: list[Cue], tgt: list[Cue], src_lang: str, tgt_lang: str) -> Alignment:
    if len(src) < MIN_PAIRS or len(tgt) < MIN_PAIRS:
        return Alignment([], 0.0, 1.0, 0.0, False, f"too few cues ({len(src)} / {len(tgt)})")
    scale, offset = estimate_transform(src, tgt)
    moved = [Cue(c.start * scale + offset, c.end * scale + offset, c.text) for c in tgt]

    ns = len(src)
    uf = _UnionFind(ns + len(moved))
    j0 = 0
    for i, s in enumerate(src):
        while j0 < len(moved) and moved[j0].end <= s.start:
            j0 += 1
        j = j0
        while j < len(moved) and moved[j].start < s.end:
            t = moved[j]
            overlap = min(s.end, t.end) - max(s.start, t.start)
            shorter = min(s.end - s.start, t.end - t.start)
            if shorter > 0 and overlap / shorter >= MIN_OVERLAP:
                uf.union(i, ns + j)
            j += 1

    groups: dict[int, tuple[list[int], list[int]]] = {}
    for i in range(ns):
        groups.setdefault(uf.find(i), ([], []))[0].append(i)
    for j in range(len(moved)):
        groups.setdefault(uf.find(ns + j), ([], []))[1].append(j)

    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    paired_src = 0
    for s_idx, t_idx in groups.values():
        if not s_idx or not t_idx or len(s_idx) > MAX_GROUP or len(t_idx) > MAX_GROUP:
            continue
        paired_src += len(s_idx)
        s_text = " ".join(src[i].text for i in sorted(s_idx))
        t_text = " ".join(moved[j].text for j in sorted(t_idx))
        if (s_text, t_text) in seen or not _good_pair(s_text, t_text, src_lang, tgt_lang):
            continue
        seen.add((s_text, t_text))
        pairs.append((s_text, t_text, src[min(s_idx)].start))
    pairs.sort(key=lambda p: p[2])

    coverage = paired_src / ns
    ok = coverage >= MIN_COVERAGE and len(pairs) >= MIN_PAIRS
    message = f"{len(pairs)} pairs, coverage {coverage:.0%}, offset {offset:+.1f}s, scale {scale:.4f}"
    if not ok:
        message = "files do not match (different cut or bad timing): " + message
    return Alignment(pairs, coverage, scale, offset, ok, message)
