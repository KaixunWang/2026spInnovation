"""Load proprietary T3 discrete rows from ``main_metrics.jsonl`` for pooled OpenAI fits."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import PROJECT_ROOT
from .io_utils import read_jsonl
from .metrics.value import fuse_fidelity_judge_nli

MAIN_METRICS_JSONL = PROJECT_ROOT / "data" / "generated" / "main_metrics.jsonl"


def load_t3_discrete_pooled_rows(path: Path | None = None) -> pd.DataFrame:
    path = path or MAIN_METRICS_JSONL
    rows: list[dict[str, Any]] = []
    for r in read_jsonl(path):
        if r.get("condition") != "T3":
            continue
        if r.get("target_sampling") != "discrete":
            continue
        m = r.get("metrics") or {}
        if not m.get("ok"):
            continue
        nli = m.get("nli_entailment")
        coh = m.get("coherence_auto")
        ca = m.get("creativity_auto")
        if nli is None or coh is None:
            continue
        j = r.get("judge") or {}
        surprise_raw: float | None = None
        if j.get("ok"):
            pdim = j.get("per_dim_mean") or {}
            if isinstance(pdim, dict) and pdim.get("surprise") is not None:
                surprise_raw = float(pdim["surprise"])
        cjudge = float("nan")
        if j.get("ok"):
            nj, cj, fj = j.get("novelty_judge"), j.get("coherence_judge"), j.get("fidelity_judge")
            if nj is not None and cj is not None and fj is not None:
                ff = fuse_fidelity_judge_nli(float(fj), m.get("nli_entailment"), w_judge=0.5)
                value_j = float((max(0.0, ff) + max(0.0, float(cj))) / 2.0)
                cjudge = float(nj) * value_j
        rows.append(
            {
                "source_id": str(r.get("source_id", "")),
                "genre": str(r.get("genre", "unknown")),
                "d_H": float(r["d_H"]),
                "nli_entailment": float(nli),
                "coherence_auto": float(coh),
                "creativity_auto": float(ca) if ca is not None else float("nan"),
                "creativity_judge": cjudge,
                "surprise_mean": surprise_raw,
            }
        )
    return pd.DataFrame(rows)
