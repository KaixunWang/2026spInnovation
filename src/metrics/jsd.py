"""Word-level Jensen–Shannon divergence between two texts (local, no API).

Uses unigram distributions over union vocabulary with additive smoothing.
Returns value in [0, 1] (natural JSD / ln 2).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u0080-\uFFFF']+")


def tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    if not text:
        return []
    if lowercase:
        text = text.lower()
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def _normalize(counter: Counter, vocab: Iterable[str], alpha: float) -> dict[str, float]:
    """Dirichlet-smoothed multinomial over vocab."""
    vocab = list(vocab)
    n = sum(counter[w] for w in vocab)
    denom = n + alpha * len(vocab)
    out = {}
    for w in vocab:
        out[w] = (counter[w] + alpha) / denom
    return out


def jensen_shannon_divergence_unigram(
    text_a: str,
    text_b: str,
    *,
    lowercase: bool = True,
    epsilon: float = 1e-12,
) -> float:
    """Return symmetric JSD in [0, ln 2]; divide by ln(2) for [0, 1]."""
    ta = tokenize(text_a, lowercase=lowercase)
    tb = tokenize(text_b, lowercase=lowercase)
    if not ta and not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    vocab = sorted(set(ca.keys()) | set(cb.keys()))
    if not vocab:
        return 0.0
    alpha = 1.0  # Laplace-style smoothing mass per type
    pa = _normalize(ca, vocab, alpha)
    pb = _normalize(cb, vocab, alpha)
    pm = {w: 0.5 * (pa[w] + pb[w]) for w in vocab}

    def kl(p: dict[str, float], q: dict[str, float]) -> float:
        s = 0.0
        for w in vocab:
            pw, qw = p[w], q[w]
            if pw <= epsilon:
                continue
            s += pw * math.log((pw + epsilon) / (qw + epsilon))
        return s

    jsd = 0.5 * kl(pa, pm) + 0.5 * kl(pb, pm)
    return float(max(0.0, jsd))


def word_jsd_normalized(text_a: str, text_b: str, *, lowercase: bool = True) -> float:
    """JSD unigram mapped to [0, 1] via division by ln 2."""
    j = jensen_shannon_divergence_unigram(text_a, text_b, lowercase=lowercase)
    ln2 = math.log(2.0)
    return float(min(1.0, j / ln2)) if ln2 > 0 else 0.0
