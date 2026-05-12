"""Automatic metrics for rewrites.

All functions take plain strings and return either a float in [0, 1]
or a dataclass of floats. None of the functions call external APIs;
they rely on local HuggingFace models (NLI / sentiment) or simple
NLP (distinct-n, Levenshtein, Kendall-tau).
"""

from .content import nli_entailment, nli_entailment_batch
from .novelty import (
    distinct_n,
    novel_ngram_ratio,
    embedding_distance,
    compute_genre_baselines,
    normalise_novelty,
)
from .value import combine_value
from .coherence import perplexity, perplexity_batch
from .sentiment import sentiment_score, sentiment_score_batch, sentiment_shift, sentiment_shift_batch
from .structural import sentence_kendall_tau, normalised_levenshtein
from .jsd import word_jsd_normalized

__all__ = [
    "nli_entailment",
    "nli_entailment_batch",
    "distinct_n",
    "novel_ngram_ratio",
    "embedding_distance",
    "compute_genre_baselines",
    "normalise_novelty",
    "combine_value",
    "perplexity",
    "perplexity_batch",
    "sentiment_score",
    "sentiment_score_batch",
    "sentiment_shift",
    "sentiment_shift_batch",
    "sentence_kendall_tau",
    "normalised_levenshtein",
    "word_jsd_normalized",
]
