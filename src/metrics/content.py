"""Content-preservation metrics: NLI entailment src -> gen.

Uses `facebook/bart-large-mnli` by default (HuggingFace, CPU-OK).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from ..hf_device import hf_local_files_only, hf_pipeline_device

NLI_MODEL = "facebook/bart-large-mnli"

# Character cap before tokenisation (BART-MNLI handles long pairs poorly on GPU batches).
_NLI_CHAR_CAP = 8000


def _clip_pair(p: str, h: str) -> tuple[str, str]:
    p = (p or "").strip() or " "
    h = (h or "").strip() or " "
    if len(p) > _NLI_CHAR_CAP:
        p = p[:_NLI_CHAR_CAP]
    if len(h) > _NLI_CHAR_CAP:
        h = h[:_NLI_CHAR_CAP]
    return p, h


@lru_cache(maxsize=1)
def _nli_pipe():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    dev = hf_pipeline_device()
    if hf_local_files_only():
        lo = {"local_files_only": True}
        tok = AutoTokenizer.from_pretrained(NLI_MODEL, **lo)
        mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, **lo)
        return pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            truncation=True,
            top_k=None,
            device=dev,
        )
    return pipeline(
        "text-classification",
        model=NLI_MODEL,
        tokenizer=NLI_MODEL,
        truncation=True,
        top_k=None,
        device=dev,
    )


def nli_entailment(premise: str, hypothesis: str) -> float:
    """Return entailment probability P(hypothesis | premise) in [0, 1]."""
    return nli_entailment_batch([(premise, hypothesis)])[0]


def nli_entailment_batch(pairs: Iterable[tuple[str, str]]) -> list[float]:
    pairs = list(pairs)
    if not pairs:
        return []
    pipe = _nli_pipe()
    results: list[float] = []
    # One pair per call: list-of-dict batching has triggered CUDA device-side asserts
    # on some Windows + long heterogeneous pairs; sequential is slower but stable.
    for p, h in pairs:
        p, h = _clip_pair(p, h)
        item_scores = pipe(
            {"text": p, "text_pair": h},
            truncation=True,
            max_length=1024,
            padding=True,
        )
        if isinstance(item_scores, list):
            label_scores = item_scores
        else:
            label_scores = [item_scores]
        ent = 0.0
        for s in label_scores:
            if s["label"].upper().startswith("ENTAIL"):
                ent = float(s["score"])
                break
        results.append(ent)
    return results
