"""Resolve device for local torch / HuggingFace metric models.

Environment
-----------
INNOVATION_METRICS_DEVICE
    - ``auto`` (default): CUDA GPU 0 if ``torch.cuda.is_available()``, else CPU
    - ``cpu``: force CPU (HF pipeline convention: device=-1)
    - ``cuda`` or ``cuda:N``: use GPU index N (default N=0)

INNOVATION_METRICS_BATCH_SIZE
    - Optional positive integer: default HF batch size for NLI + sentiment in
      ``run_experiment metrics`` (CLI ``--metrics-batch-size`` overrides).

INNOVATION_HF_LOCAL_ONLY
    - If ``1`` / ``true``: load NLI/sentiment with ``local_files_only=True`` via
      ``AutoTokenizer`` / ``AutoModelForSequenceClassification``, then build
      ``pipeline`` (avoids Hub during load and avoids passing ``tokenizer_kwargs``
      into ``pipeline``, which breaks batching on some ``transformers`` versions).

Placed under ``src/`` (not ``src.metrics``) so importing lightweight modules
like ``embedding_space`` does not execute ``metrics/__init__.py``.
"""

from __future__ import annotations

import os


def hf_pipeline_device() -> int:
    """Return HF ``pipeline(..., device=...)`` index: -1 for CPU, else GPU index."""
    raw = os.environ.get("INNOVATION_METRICS_DEVICE", "auto").strip().lower()
    if raw in ("", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return -1
    if raw == "cpu":
        return -1
    if raw.startswith("cuda"):
        if raw == "cuda":
            return 0
        part = raw.split(":", 1)
        if len(part) == 2 and part[1].isdigit():
            return int(part[1])
        return 0
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


def torch_device_str() -> str:
    """String for ``tensor.to(...)`` / ``model.to(...)``."""
    idx = hf_pipeline_device()
    if idx < 0:
        return "cpu"
    return f"cuda:{idx}"


def hf_local_files_only() -> bool:
    """True when metric pipelines should not call the Hub (cache-only load)."""
    v = os.environ.get("INNOVATION_HF_LOCAL_ONLY", "").strip().lower()
    return v in ("1", "true", "yes", "on")
