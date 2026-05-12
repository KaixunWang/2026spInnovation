"""Perplexity-based coherence proxy.

Uses a small open-source English LM (gpt2 by default) to compute perplexity
of a text. Lower perplexity => more fluent / likely under the reference LM.
Not a full coherence signal on its own; we pair with LLM-judge coherence.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..hf_device import torch_device_str


PPL_MODEL = "gpt2"


@lru_cache(maxsize=1)
def _load_ppl():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PPL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(PPL_MODEL)
    model.eval()
    dev = torch_device_str()
    model = model.to(dev)
    return tok, model, torch, dev


def perplexity(text: str, *, max_length: int = 512) -> float:
    tok, model, torch, dev = _load_ppl()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(dev)
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return float(torch.exp(out.loss).item())


def perplexity_batch(texts: list[str]) -> list[float]:
    return [perplexity(t) for t in texts]


def perplexity_to_unit(ppl: float, *, pivot: float = 80.0) -> float:
    """Smoothly map perplexity to [0, 1] via a logistic transform.

    1 = very fluent (low PPL), 0 = incoherent (very high PPL).
    `pivot` is the PPL corresponding to 0.5; default 80 is a sensible midpoint
    for short modern literary prose under GPT-2.
    """
    if not np.isfinite(ppl):
        return 0.0
    # logistic: higher ppl -> closer to 0
    return float(1.0 / (1.0 + (ppl / max(1.0, pivot)) ** 2))
