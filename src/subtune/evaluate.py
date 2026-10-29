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


def score(hypotheses: list[str], references: list[str], language: str) -> dict:
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=bleu_tokenizer(language))
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2), "n": len(hypotheses)}
