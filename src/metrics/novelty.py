"""Novelty metrics with per-genre baseline normalisation.

The central methodological correction (§6.1 of the proposal): raw distinct-n
and raw embedding-distance are confounded by genre-intrinsic diversity
(poetry > narrative > essay > academic). We therefore compute per-genre
baselines over the SOURCE corpus and report every rewrite's novelty
*relative* to those baselines.

Key functions
-------------
distinct_n(text, n)                       : type-token diversity
novel_ngram_ratio(src, gen, n)            : fraction of n-grams in gen not in src
embedding_distance(src, gen, sbert_model) : 1 - cosine of SBERT embeddings
compute_genre_baselines(sources, ...)     : per-genre reference distributions
normalise_novelty(raw, baseline)          : z-score or ratio per-genre
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Iterable

import numpy as np


_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(text: str, n: int = 2) -> float:
    tokens = _tokenise(text)
    ngr = _ngrams(tokens, n)
    if not ngr:
        return 0.0
    return len(set(ngr)) / len(ngr)


def novel_ngram_ratio(src: str, gen: str, n: int = 2) -> float:
    src_set = set(_ngrams(_tokenise(src), n))
    gen_ngr = _ngrams(_tokenise(gen), n)
    if not gen_ngr:
        return 0.0
    novel = sum(1 for g in gen_ngr if g not in src_set)
    return novel / len(gen_ngr)


def embedding_distance(src: str, gen: str, sbert_model: str) -> float:
    """1 - cosine similarity of SBERT embeddings of src and gen."""
    from ..embedding_space import embed

    vecs = embed([src, gen], sbert_model)
    cos = float(np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-12))
    return float(np.clip(1.0 - cos, 0.0, 2.0))


# ---------------------------------------------------------------------------
# Per-genre baselines
# ---------------------------------------------------------------------------


@dataclass
class GenreBaseline:
    genre: str
    n: int
    distinct2_mean: float
    distinct2_std: float
    distinct3_mean: float
    distinct3_std: float
    avg_intra_distinct2_pair_overlap: float  # used for novel_ngram_ratio baseline
    avg_intra_embedding_distance: float

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        return d


def compute_genre_baselines(
    sources_by_genre: dict[str, list[str]],
    *,
    sbert_model: str | None = None,
) -> dict[str, GenreBaseline]:
    """Compute distinct-n and embedding-distance baselines *within* each genre.

    For each genre g:
      distinct2_mean/std := distinct_2 over SOURCE texts of g
      avg_intra_distinct2_pair_overlap := 1 - novel_ngram_ratio(a, b) averaged over pairs
      avg_intra_embedding_distance := pairwise 1-cos between genre members
    """
    out: dict[str, GenreBaseline] = {}
    emb_cache: dict[str, np.ndarray] = {}
    if sbert_model is not None:
        from ..embedding_space import embed

        all_texts = [t for texts in sources_by_genre.values() for t in texts]
        all_emb = embed(all_texts, sbert_model)
        idx = 0
        for g, texts in sources_by_genre.items():
            emb_cache[g] = all_emb[idx : idx + len(texts)]
            idx += len(texts)

    for g, texts in sources_by_genre.items():
        if not texts:
            continue
        d2 = [distinct_n(t, 2) for t in texts]
        d3 = [distinct_n(t, 3) for t in texts]
        # pairwise novelty (symmetric, average over unordered pairs)
        pair_overlap: list[float] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                pair_overlap.append(1.0 - novel_ngram_ratio(texts[i], texts[j], 2))
        if sbert_model is not None:
            vecs = emb_cache[g]
            dists: list[float] = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    cos = float(
                        np.dot(vecs[i], vecs[j])
                        / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-12)
                    )
                    dists.append(1.0 - cos)
            avg_emb = float(mean(dists)) if dists else 0.0
        else:
            avg_emb = 0.0
        out[g] = GenreBaseline(
            genre=g,
            n=len(texts),
            distinct2_mean=float(mean(d2)) if d2 else 0.0,
            distinct2_std=float(pstdev(d2)) if len(d2) > 1 else 0.0,
            distinct3_mean=float(mean(d3)) if d3 else 0.0,
            distinct3_std=float(pstdev(d3)) if len(d3) > 1 else 0.0,
            avg_intra_distinct2_pair_overlap=float(mean(pair_overlap)) if pair_overlap else 1.0,
            avg_intra_embedding_distance=avg_emb,
        )
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def zscore_clip(value: float, mean_: float, std_: float, *, clip: float = 3.0) -> float:
    if std_ < 1e-9:
        return 0.0
    z = (value - mean_) / std_
    return float(np.clip(z / clip, -1.0, 1.0))


def _rebound(x: float) -> float:
    """Map [-1, 1] -> [0, 1] so that novelty remains non-negative."""
    return float((x + 1.0) / 2.0)


@dataclass
class NoveltyScores:
    distinct2_raw: float
    distinct2_rel: float          # in [0, 1]
    novel_ngram_raw: float
    novel_ngram_rel: float
    embedding_distance_raw: float
    embedding_distance_rel: float
    auto_combined: float          # mean of the three rel values, in [0, 1]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normalise_novelty(
    src: str,
    gen: str,
    genre_baseline: GenreBaseline,
    *,
    sbert_model: str | None = None,
) -> NoveltyScores:
    d2 = distinct_n(gen, 2)
    nng = novel_ngram_ratio(src, gen, 2)
    if sbert_model is not None:
        ed = embedding_distance(src, gen, sbert_model)
    else:
        ed = 0.0
    # rel values via z-score clipping into [-1,1] then rebound to [0,1]
    d2_z = zscore_clip(d2, genre_baseline.distinct2_mean, genre_baseline.distinct2_std)
    # novel_ngram is in [0,1]; baseline is 1 - avg_intra_overlap (the expected novelty
    # between two SOURCE items in the same genre); we compare ratio of difference.
    base_nng = 1.0 - genre_baseline.avg_intra_distinct2_pair_overlap
    std_nng = 0.1  # heuristic spread within a genre (tunable)
    nng_z = zscore_clip(nng - base_nng, 0.0, std_nng)
    # embedding distance: baseline is avg pairwise distance WITHIN the genre;
    # rel = (ed - base) / base
    if genre_baseline.avg_intra_embedding_distance > 1e-6:
        ed_z_raw = (ed - genre_baseline.avg_intra_embedding_distance) / (
            genre_baseline.avg_intra_embedding_distance + 1e-6
        )
        ed_z = float(np.clip(ed_z_raw, -1.0, 1.0))
    else:
        ed_z = 0.0
    d2_rel = _rebound(d2_z)
    nng_rel = _rebound(nng_z)
    ed_rel = _rebound(ed_z)
    auto = (d2_rel + nng_rel + ed_rel) / 3.0
    return NoveltyScores(
        distinct2_raw=d2,
        distinct2_rel=d2_rel,
        novel_ngram_raw=nng,
        novel_ngram_rel=nng_rel,
        embedding_distance_raw=ed,
        embedding_distance_rel=ed_rel,
        auto_combined=auto,
    )
