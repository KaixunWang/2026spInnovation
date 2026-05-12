"""Value aggregation utilities.

Primary definition (for formal runs):
  Value_arith = (Fidelity + Coherence) / 2

Sensitivity definition (reported in analysis):
  Value_geom  = sqrt(Fidelity * Coherence)
"""

from __future__ import annotations

import math


def fuse_fidelity_judge_nli(
    fidelity_judge: float | None,
    nli_entailment: float | None,
    *,
    w_judge: float = 0.5,
) -> float:
    """Fuse judge fidelity and NLI with a fixed weighted average.

    If one side is missing, fall back to the other. If both missing, return 0.
    """
    fj = None if fidelity_judge is None else max(0.0, min(1.0, float(fidelity_judge)))
    nli = None if nli_entailment is None else max(0.0, min(1.0, float(nli_entailment)))
    if fj is None and nli is None:
        return 0.0
    if fj is None:
        return float(nli)
    if nli is None:
        return float(fj)
    wj = max(0.0, min(1.0, float(w_judge)))
    return float(wj * fj + (1.0 - wj) * nli)


def combine_value_arith(entailment: float, coherence: float) -> float:
    e = max(0.0, min(1.0, entailment))
    c = max(0.0, min(1.0, coherence))
    return float((e + c) / 2.0)


def combine_value_geom(entailment: float, coherence: float) -> float:
    e = max(0.0, min(1.0, entailment))
    c = max(0.0, min(1.0, coherence))
    if e <= 0.0 or c <= 0.0:
        return 0.0
    return float(math.sqrt(e * c))


def combine_value(entailment: float, coherence: float) -> float:
    """Primary value definition used in metrics: arithmetic mean."""
    return combine_value_arith(entailment, coherence)


def combine_creativity(novelty: float, value: float) -> float:
    n = max(0.0, min(1.0, novelty))
    v = max(0.0, min(1.0, value))
    return float(n * v)


def utility(novelty: float, value: float, lam: float = 1.0) -> float:
    """Alternative composite: Utility = Novelty - λ·(1 - Value)."""
    return float(novelty - lam * (1.0 - value))
