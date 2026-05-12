"""Four-arm quasi-causal design: T0 / T1 / T2 / T3.

Each arm takes a source text and produces a rewrite to be compared.

  T0 — identity:  return source verbatim (measurement-noise floor)
  T1 — same-personality: rewrite with the persona whose vector is nearest to the
        source's Space-H vector (pure-rewrite effect, near-zero conflict)
  T2 — random-personality: rewrite with a uniformly random persona drawn from
        the persona set (placebo for "any change")
  T3 — cross-personality: rewrite with a specified target persona (the
        treatment arm; the caller picks the target via conflict bucketing)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .conflict import CoordinateScore, compute_conflict
from .corpus import SourceText
from .personalities import Persona, PersonaSet
from .rewrite import RewriteResult, rewrite_joint


@dataclass
class BaselineRecord:
    source_id: str
    genre: str
    condition: str            # T0 / T1 / T2 / T3
    target_persona: str | None  # None for T0
    target_sampling: str        # discrete | continuous
    src_vec_H: list[float]
    tgt_vec_H: list[float] | None
    d_H: float
    delta_H: list[float] | None
    mode: str                 # joint / sequential / constrained
    model: str | None         # None for T0
    prompt_variant: int
    repeat_idx: int
    text: str
    intermediate: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


# ---------------------------------------------------------------------------
# Arm constructors
# ---------------------------------------------------------------------------


def arm_T0(source: SourceText, src_vec: np.ndarray) -> BaselineRecord:
    c = compute_conflict(src_vec, src_vec, space="H")
    return BaselineRecord(
        source_id=source.id,
        genre=source.genre,
        condition="T0",
        target_persona=None,
        target_sampling="discrete",
        src_vec_H=src_vec.tolist(),
        tgt_vec_H=None,
        d_H=0.0,
        delta_H=None,
        mode="identity",
        model=None,
        prompt_variant=0,
        repeat_idx=0,
        text=source.text,
        intermediate={},
    )


def arm_T1(
    source: SourceText,
    src_vec: np.ndarray,
    personas: PersonaSet,
    *,
    model: str,
    prompt_variant: int,
    repeat_idx: int,
) -> BaselineRecord:
    """Rewrite under the persona nearest to the source vector (minimises d)."""
    nearest: Persona = personas.closest_to(src_vec)
    c = compute_conflict(src_vec, nearest.vector, space="H")
    out = rewrite_joint(
        source, nearest, model=model, prompt_variant=prompt_variant,
        temperature=0.7,
    )
    return BaselineRecord(
        source_id=source.id,
        genre=source.genre,
        condition="T1",
        target_persona=nearest.name,
        target_sampling="discrete",
        src_vec_H=src_vec.tolist(),
        tgt_vec_H=nearest.vector.tolist(),
        d_H=c.d,
        delta_H=c.delta.tolist(),
        mode=out.mode,
        model=model,
        prompt_variant=prompt_variant,
        repeat_idx=repeat_idx,
        text=out.text,
        intermediate=out.intermediate,
    )


def arm_T2(
    source: SourceText,
    src_vec: np.ndarray,
    personas: PersonaSet,
    *,
    model: str,
    prompt_variant: int,
    repeat_idx: int,
    rng: random.Random,
) -> BaselineRecord:
    """Rewrite under a uniformly random persona (placebo for 'any rewrite')."""
    p: Persona = rng.choice(list(personas.personas))
    c = compute_conflict(src_vec, p.vector, space="H")
    out = rewrite_joint(
        source, p, model=model, prompt_variant=prompt_variant, temperature=0.7,
    )
    return BaselineRecord(
        source_id=source.id,
        genre=source.genre,
        condition="T2",
        target_persona=p.name,
        target_sampling="discrete",
        src_vec_H=src_vec.tolist(),
        tgt_vec_H=p.vector.tolist(),
        d_H=c.d,
        delta_H=c.delta.tolist(),
        mode=out.mode,
        model=model,
        prompt_variant=prompt_variant,
        repeat_idx=repeat_idx,
        text=out.text,
        intermediate=out.intermediate,
    )


def arm_T3(
    source: SourceText,
    src_vec: np.ndarray,
    target: Persona,
    *,
    model: str,
    prompt_variant: int,
    repeat_idx: int,
    mode: str = "joint",
    target_sampling: str = "discrete",
) -> BaselineRecord:
    """Treatment arm: rewrite under a specified target persona."""
    c = compute_conflict(src_vec, target.vector, space="H")
    from .rewrite import rewrite as _rewrite
    out: RewriteResult = _rewrite(
        source, target, mode=mode, model=model, prompt_variant=prompt_variant, temperature=0.7,
    )
    return BaselineRecord(
        source_id=source.id,
        genre=source.genre,
        condition="T3",
        target_persona=target.name,
        target_sampling=target_sampling,
        src_vec_H=src_vec.tolist(),
        tgt_vec_H=target.vector.tolist(),
        d_H=c.d,
        delta_H=c.delta.tolist(),
        mode=out.mode,
        model=model,
        prompt_variant=prompt_variant,
        repeat_idx=repeat_idx,
        text=out.text,
        intermediate=out.intermediate,
    )
