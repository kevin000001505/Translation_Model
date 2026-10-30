"""Translation quality metrics (sacreBLEU chrF and BLEU)."""

from __future__ import annotations

import sacrebleu

from .languages import base_code


def bleu_tokenizer(language: str) -> str:
    code = base_code(language)
    if code == "zh":
        return "zh"
    if code in ("ja", "ko", "th", "my", "km"):
        return "char"
    return "13a"


def sentence_chrf(hypotheses: list[str], references: list[str]) -> list[float]:
    return [round(sacrebleu.sentence_chrf(h, [r]).score, 2) for h, r in zip(hypotheses, references)]


def compare(rows: list[dict], base_hyps: list[str], tuned_hyps: list[str]) -> tuple[list[dict], dict]:
    """Per-sentence results for the validation set, and a summary of how the fine-tune changed them."""
    refs = [r["tgt"] for r in rows]
    base_c, tuned_c = sentence_chrf(base_hyps, refs), sentence_chrf(tuned_hyps, refs)
    details = [
        {"src": r["src"], "ref": r["tgt"], "base": b, "tuned": t, "base_chrf": bc, "tuned_chrf": tc,
         "delta": round(tc - bc, 2)}
        for r, b, t, bc, tc in zip(rows, base_hyps, tuned_hyps, base_c, tuned_c)
    ]
    n = len(details)
    summary = {
        "sentences": n,
        "changed": sum(d["base"] != d["tuned"] for d in details),
        "improved": sum(d["delta"] > 0.5 for d in details),
        "worse": sum(d["delta"] < -0.5 for d in details),
        "mean_sentence_chrf_base": round(sum(base_c) / n, 2) if n else None,
        "mean_sentence_chrf_tuned": round(sum(tuned_c) / n, 2) if n else None,
    }
    return details, summary


def score(hypotheses: list[str], references: list[str], language: str) -> dict:
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=bleu_tokenizer(language))
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2), "n": len(hypotheses)}
