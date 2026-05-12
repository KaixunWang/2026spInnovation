"""Structural remix metrics: sentence-level Kendall τ + normalised Levenshtein."""

from __future__ import annotations

import re

import numpy as np


_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def _token_set(s: str) -> set[str]:
    return set(w.lower() for w in re.findall(r"[A-Za-z']+", s))


def sentence_alignment(src_sents: list[str], gen_sents: list[str]) -> list[int]:
    """For each gen sentence, index of the most similar src sentence (by Jaccard)."""
    alignments: list[int] = []
    src_toks = [_token_set(s) for s in src_sents]
    for gs in gen_sents:
        gt = _token_set(gs)
        if not gt or not src_toks:
            alignments.append(-1)
            continue
        scores = [
            len(gt & st) / max(1, len(gt | st)) if st else 0.0 for st in src_toks
        ]
        alignments.append(int(np.argmax(scores)))
    return alignments


def sentence_kendall_tau(src: str, gen: str) -> float:
    """1 - Kendall-tau-based disorder over sentence alignments, in [0, 1].

    1.0 means gen preserves src sentence order; 0.0 means fully reversed.
    """
    src_sents = sentences(src)
    gen_sents = sentences(gen)
    if len(src_sents) < 2 or len(gen_sents) < 2:
        return 1.0
    align = [a for a in sentence_alignment(src_sents, gen_sents) if a >= 0]
    if len(align) < 2:
        return 1.0
    from scipy.stats import kendalltau

    tau, _ = kendalltau(list(range(len(align))), align)
    if not np.isfinite(tau):
        return 1.0
    return float((tau + 1.0) / 2.0)


def normalised_levenshtein(src: str, gen: str) -> float:
    """Character-level Levenshtein / max-length, in [0, 1].

    0 = identical, 1 = completely different.
    """
    a, b = src, gen
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, 1)
