"""Style-distribution divergences (JSD / KL) for the channel-capacity story.

We estimate P(w | persona) as a smoothed unigram distribution over the
Space-L reference corpus for each persona, and compute pairwise JSD and KL.

Public API
----------
build_style_distributions(reference_corpus)        -> dict[name, distribution]
jsd(p, q)                                          -> float
kl(p, q)                                           -> float
jsd_between_personas(dists)                        -> dict[(a, b), float]
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np


_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def build_style_distribution(
    texts: Iterable[str], *, alpha: float = 1.0
) -> tuple[dict[str, float], list[str]]:
    """Smoothed unigram distribution with Laplace additive alpha.

    Returns (distribution_dict, vocab_list).
    """
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(_tokenise(t))
    vocab = sorted(counts.keys())
    if not vocab:
        return {}, []
    total = sum(counts.values())
    v = len(vocab)
    dist = {w: (counts[w] + alpha) / (total + alpha * v) for w in vocab}
    return dist, vocab


def _align(p: dict[str, float], q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    vocab = sorted(set(p) | set(q))
    if not vocab:
        return np.array([1.0]), np.array([1.0])
    pv = np.array([p.get(w, 0.0) for w in vocab], dtype=np.float64)
    qv = np.array([q.get(w, 0.0) for w in vocab], dtype=np.float64)
    # Re-normalise (in case alpha-smoothing used different vocab sizes)
    if pv.sum() > 0:
        pv = pv / pv.sum()
    if qv.sum() > 0:
        qv = qv / qv.sum()
    return pv, qv


def kl(p: dict[str, float], q: dict[str, float], *, base: float = 2.0) -> float:
    pv, qv = _align(p, q)
    mask = (pv > 0) & (qv > 0)
    if not mask.any():
        return float("inf")
    log = np.log2 if base == 2.0 else np.log
    return float(np.sum(pv[mask] * log(pv[mask] / qv[mask])))


def jsd(p: dict[str, float], q: dict[str, float], *, base: float = 2.0) -> float:
    pv, qv = _align(p, q)
    m = 0.5 * (pv + qv)
    log = np.log2 if base == 2.0 else np.log

    def _kl_arr(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return 0.0
        return float(np.sum(a[mask] * log(a[mask] / b[mask])))

    return 0.5 * _kl_arr(pv, m) + 0.5 * _kl_arr(qv, m)


def jsd_between_personas(dists: dict[str, dict[str, float]]) -> dict[tuple[str, str], float]:
    """Compute JSD for each unordered pair of personas."""
    out: dict[tuple[str, str], float] = {}
    names = sorted(dists.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[(a, b)] = jsd(dists[a], dists[b])
            out[(b, a)] = out[(a, b)]
    return out
