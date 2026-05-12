"""Multi-hop evolutionary rewrite: source -> P_A -> P_B -> ... -> P_K.

Produces a trajectory of rewrites and records, at each hop, d w.r.t. the
ORIGINAL source (d_total) and w.r.t. the previous hop (d_step). Designed
to be analysed for drift, attractors, and path dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config_loader import load_experiment_config
from .conflict import compute_conflict, score_source_in_space_h, score_source_space_h_deterministic
from .corpus import SourceText
from .personalities import Persona, PersonaSet
from .rewrite import rewrite_joint


def _coord_backend() -> str:
    return str((load_experiment_config().get("coord_scoring") or {}).get("backend", "llm")).lower()


@dataclass
class HopRecord:
    source_id: str
    genre: str
    path_id: int
    hop_index: int            # 0 for source itself
    persona: str              # persona applied at THIS hop ("_source_" for hop 0)
    src_vec_H_total: list[float]  # coord of original source
    cur_vec_H: list[float] | None  # coord of text AT this hop
    d_total: float | None     # distance cur - orig, in Space-H
    d_step: float | None      # distance cur - prev (None for hop 0)
    text: str
    model: str | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def run_multi_hop(
    source: SourceText,
    *,
    path: list[str],
    personas: PersonaSet,
    model: str,
    path_id: int,
    src_vec: np.ndarray,
    score_intermediates: bool = False,
    coord_scorer_model: str | None = None,
) -> list[HopRecord]:
    """Run one persona-ordering path. Returns K+1 records (including hop 0)."""
    records: list[HopRecord] = [
        HopRecord(
            source_id=source.id,
            genre=source.genre,
            path_id=path_id,
            hop_index=0,
            persona="_source_",
            src_vec_H_total=src_vec.tolist(),
            cur_vec_H=src_vec.tolist(),
            d_total=0.0,
            d_step=None,
            text=source.text,
            model=None,
        )
    ]
    cur_text = source.text
    prev_vec = src_vec.copy()
    for h, persona_name in enumerate(path, start=1):
        p: Persona = personas[persona_name]
        # Wrap cur_text in a pseudo-SourceText so the rewrite function can reuse length
        pseudo = SourceText(
            id=f"{source.id}__hop{h-1}",
            genre=source.genre,
            source=source.source,
            license=source.license,
            length=len(cur_text.split()),
            note="multi-hop intermediate",
            text=cur_text,
            path=Path("."),
        )
        out = rewrite_joint(pseudo, p, model=model, prompt_variant=0, temperature=0.7)
        new_text = out.text
        if score_intermediates and coord_scorer_model:
            pseudo_src = SourceText(
                id=pseudo.id + "_eval",
                genre=source.genre,
                source="",
                license="",
                length=len(new_text.split()),
                note="",
                text=new_text,
                path=Path("."),
            )
            if _coord_backend() == "hf":
                cs = score_source_space_h_deterministic(pseudo_src)
            else:
                cs = score_source_in_space_h(
                    pseudo_src, personas=personas, scorer_model=coord_scorer_model, n_seeds=1
                )
            cur_vec = cs.vec()
        else:
            cur_vec = None
        if cur_vec is not None:
            d_total = float(compute_conflict(src_vec, cur_vec, space="H").d)
            d_step = float(compute_conflict(prev_vec, cur_vec, space="H").d)
        else:
            d_total = None
            d_step = None
        records.append(
            HopRecord(
                source_id=source.id,
                genre=source.genre,
                path_id=path_id,
                hop_index=h,
                persona=p.name,
                src_vec_H_total=src_vec.tolist(),
                cur_vec_H=cur_vec.tolist() if cur_vec is not None else None,
                d_total=d_total,
                d_step=d_step,
                text=new_text,
                model=model,
            )
        )
        cur_text = new_text
        if cur_vec is not None:
            prev_vec = cur_vec
    return records
