"""Sentiment analysis: continuous score in [-1, 1] and absolute shift."""

from __future__ import annotations

from functools import lru_cache

from ..hf_device import hf_local_files_only, hf_pipeline_device

SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_SENT_CHAR_CAP = 4000


def _clip_sent_text(s: str) -> str:
    s = (s or "").strip() or " "
    return s if len(s) <= _SENT_CHAR_CAP else s[:_SENT_CHAR_CAP]


@lru_cache(maxsize=1)
def _sent_pipe():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    dev = hf_pipeline_device()
    if hf_local_files_only():
        lo = {"local_files_only": True}
        tok = AutoTokenizer.from_pretrained(SENT_MODEL, **lo)
        mdl = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL, **lo)
        return pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            top_k=None,
            truncation=True,
            device=dev,
        )
    return pipeline(
        "text-classification",
        model=SENT_MODEL,
        tokenizer=SENT_MODEL,
        top_k=None,
        truncation=True,
        device=dev,
    )


_LABEL_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def _label_scores_from_pipe_item(raw) -> list[dict]:
    """One sample's label list from text-classification pipeline output."""
    scores = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    if isinstance(scores, dict):
        return [scores]
    if isinstance(scores, list) and scores and isinstance(scores[0], dict):
        return scores
    return []


def _sentiment_from_label_scores(scores: list[dict]) -> float:
    total = 0.0
    for item in scores:
        w = _LABEL_MAP.get(item["label"].lower(), 0.0)
        total += w * float(item["score"])
    return float(max(-1.0, min(1.0, total)))


def sentiment_score(text: str) -> float:
    """Signed scalar in [-1, 1]: weighted mean over positive / negative / neutral."""
    return sentiment_score_batch([text])[0]


def sentiment_score_batch(texts: list[str]) -> list[float]:
    """Batch sentiment scores in [-1, 1], one per input string."""
    if not texts:
        return []
    pipe = _sent_pipe()
    out: list[float] = []
    # RoBERTa + batched pipeline can mis-pad long heterogeneous texts on CUDA; run one-by-one.
    for t in texts:
        t = _clip_sent_text(t)
        raw = pipe(t, truncation=True, max_length=512, padding=True)
        rows = [raw[0]] if isinstance(raw, list) and len(raw) == 1 else [raw]
        out.append(_sentiment_from_label_scores(_label_scores_from_pipe_item(rows[0])))
    return out


def sentiment_shift(src: str, gen: str) -> float:
    """Absolute difference of sentiment scores, in [0, 2]."""
    return sentiment_shift_batch([(src, gen)])[0]


def sentiment_shift_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """Batch absolute sentiment deltas for (src, gen) pairs."""
    if not pairs:
        return []
    srcs = [p[0] for p in pairs]
    gens = [p[1] for p in pairs]
    ss = sentiment_score_batch(srcs)
    sg = sentiment_score_batch(gens)
    return [float(abs(a - b)) for a, b in zip(ss, sg)]
