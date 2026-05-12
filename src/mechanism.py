"""Mechanism arm (H6): compare three stage decompositions of the rewrite.

This module is a thin wrapper around :mod:`rewrite` that enumerates the
three modes (M1 joint, M2 sequential, M3 constrained) over the same
(source, target_persona, model, repeat) cells, so that `run_experiment
mechanism` can directly iterate without duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .conflict import compute_conflict
from .corpus import SourceText
from .personalities import Persona
from .rewrite import MODE_DISPATCH


MECHANISM_MODES = ("M0", "M1", "M2", "M3")


@dataclass
class MechanismRecord:
    source_id: str
    genre: str
    target_persona: str
    src_vec_H: list[float]
    tgt_vec_H: list[float]
    d_H: float
    delta_H: list[float]
    mode: str                 # M1 / M2 / M3 (joint/sequential/constrained)
    model: str
    prompt_variant: int
    repeat_idx: int
    text: str
    intermediate: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def mechanism_cells(
    source: SourceText,
    src_vec: np.ndarray,
    target: Persona,
    *,
    model: str,
    prompt_variant: int,
    repeat_idx: int,
    modes: tuple[str, ...] = MECHANISM_MODES,
    use_cache: bool = True,
) -> Iterator[MechanismRecord]:
    """Yield one :class:`MechanismRecord` per mode for a fixed cell."""
    c = compute_conflict(src_vec, target.vector, space="H")
    for mode in modes:
        fn = MODE_DISPATCH[mode]
        out = fn(
            source,
            target,
            model=model,
            prompt_variant=prompt_variant,
            temperature=0.7,
            use_cache=use_cache,
        )
        yield MechanismRecord(
            source_id=source.id,
            genre=source.genre,
            target_persona=target.name,
            src_vec_H=src_vec.tolist(),
            tgt_vec_H=target.vector.tolist(),
            d_H=c.d,
            delta_H=c.delta.tolist(),
            mode=mode,
            model=model,
            prompt_variant=prompt_variant,
            repeat_idx=repeat_idx,
            text=out.text,
            intermediate=out.intermediate,
        )
