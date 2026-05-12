"""Conflict operationalisation: scalar d + directional Δ, in Space-H and Space-L.

Two main jobs:

  1. Score each SOURCE text's (S, R) coordinates in Space-H using an LLM
     coordinate-scorer (see configs/personalities.yaml > scorer_prompt).
     We call the scorer twice with different seeds and keep both mean and sigma.

  2. Given src_vec and tgt_vec, compute:
       d   : scalar conflict magnitude (Euclidean, normalised to d_max)
       Δ   : signed vector (tgt - src), decomposed per axis
       bucket : {low, mid, high} tercile assignment (based on d)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .config_loader import cache_dir, get_role_models, load_experiment_config
from .text_style_coords import get_Ps as _get_Ps_hf
from .corpus import SourceText
from .generator import generate
from .progress_util import iter_progress
from .personalities import (
    AXIS_NAMES,
    D_MAX_SPACE_H,
    Persona,
    PersonaSet,
    load_personas,
)


# ---------------------------------------------------------------------------
# Coordinate scoring (Space-H)
# ---------------------------------------------------------------------------


_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_scorer_json(text: str) -> dict | None:
    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: first top-level {...} block
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class CoordinateScore:
    source_id: str
    S_mean: float
    S_sigma: float
    R_mean: float
    R_sigma: float
    raw: list[dict] = field(default_factory=list)

    def vec(self) -> np.ndarray:
        return np.array([self.S_mean, self.R_mean], dtype=float)


def score_source_in_space_h(
    source: SourceText,
    *,
    personas: PersonaSet | None = None,
    scorer_model: str | None = None,
    n_seeds: int = 2,
) -> CoordinateScore:
    """Run the LLM coordinate-scorer n_seeds times; return normalised (S, R) in [-1, 1]."""
    if personas is None:
        personas = load_personas()
    if scorer_model is None:
        scorer_model = get_role_models("coordinate_scorer")[0]

    prompt_template = personas.scorer_prompt
    user = prompt_template.format(GENRE=source.genre, TEXT=source.text)

    S_vals: list[float] = []
    R_vals: list[float] = []
    raws: list[dict] = []
    # inject tiny perturbation in system prompt to break identical cache keys
    for s in range(n_seeds):
        sys_prompt = f"You are a careful literary evaluator. (seed {s})"
        res = generate(
            scorer_model,
            system=sys_prompt,
            user=user,
            temperature=0.0,
            max_tokens=200,
        )
        parsed = _parse_scorer_json(res.text)
        if parsed is None:
            continue
        try:
            S = float(parsed["S"]) / 5.0
            R = float(parsed["R"]) / 5.0
        except Exception:
            continue
        S = float(np.clip(S, -1.0, 1.0))
        R = float(np.clip(R, -1.0, 1.0))
        S_vals.append(S)
        R_vals.append(R)
        raws.append(parsed)

    if not S_vals:  # scorer failed; fall back to origin
        return CoordinateScore(
            source_id=source.id,
            S_mean=0.0,
            S_sigma=float("nan"),
            R_mean=0.0,
            R_sigma=float("nan"),
            raw=raws,
        )
    return CoordinateScore(
        source_id=source.id,
        S_mean=float(np.mean(S_vals)),
        S_sigma=float(np.std(S_vals, ddof=0) if len(S_vals) > 1 else 0.0),
        R_mean=float(np.mean(R_vals)),
        R_sigma=float(np.std(R_vals, ddof=0) if len(R_vals) > 1 else 0.0),
        raw=raws,
    )


def score_source_space_h_deterministic(source: SourceText) -> CoordinateScore:
    """HF-based deterministic (S, R) in [-1, 1]; sigmas set to 0."""
    s, r = _get_Ps_hf(source.text)
    return CoordinateScore(
        source_id=source.id,
        S_mean=s,
        S_sigma=0.0,
        R_mean=r,
        R_sigma=0.0,
        raw=[],
    )


def score_sources_batch(
    sources: Iterable[SourceText],
    *,
    personas: PersonaSet | None = None,
    scorer_model: str | None = None,
    cache_file: Path | None = None,
) -> list[CoordinateScore]:
    cfg = load_experiment_config()
    cs_cfg = cfg.get("coord_scoring") or {}
    backend = str(cs_cfg.get("backend", "llm")).lower()
    n_seeds = int(cs_cfg.get("n_seeds", 2))
    if cache_file is None:
        cache_file = cache_dir() / "coord_scores.jsonl"
    scores: list[CoordinateScore] = []
    sources_list = list(sources)
    for src in iter_progress(
        sources_list,
        total=len(sources_list),
        desc="[coord] Space-H",
        unit="src",
    ):
        if backend == "hf":
            cs = score_source_space_h_deterministic(src)
        else:
            cs = score_source_in_space_h(
                src, personas=personas, scorer_model=scorer_model, n_seeds=n_seeds
            )
        scores.append(cs)
    with cache_file.open("w", encoding="utf-8") as fh:
        for cs in scores:
            fh.write(
                json.dumps(
                    {
                        "source_id": cs.source_id,
                        "S_mean": cs.S_mean,
                        "S_sigma": cs.S_sigma,
                        "R_mean": cs.R_mean,
                        "R_sigma": cs.R_sigma,
                    }
                )
                + "\n"
            )
    return scores


# ---------------------------------------------------------------------------
# Scalar d and directional Δ
# ---------------------------------------------------------------------------


@dataclass
class Conflict:
    """Conflict description for a single (source, target) pair in one space."""

    space: str                 # "H" or "L"
    src_vec: np.ndarray        # (k,)
    tgt_vec: np.ndarray        # (k,)
    d: float                   # normalised scalar in [0, 1]
    delta: np.ndarray          # tgt - src, shape (k,)
    delta_unit: np.ndarray     # delta / ||delta|| (zero if d==0)
    axis_names: tuple[str, ...]

    def to_dict(self) -> dict:
        out = {
            "space": self.space,
            "src_vec": self.src_vec.tolist(),
            "tgt_vec": self.tgt_vec.tolist(),
            "d": self.d,
            "delta": self.delta.tolist(),
            "delta_unit": self.delta_unit.tolist(),
            "axis_names": list(self.axis_names),
        }
        for i, n in enumerate(self.axis_names):
            out[f"delta_{n}"] = float(self.delta[i])
        return out


def compute_conflict(
    src_vec: np.ndarray,
    tgt_vec: np.ndarray,
    *,
    space: str,
    d_max: float | None = None,
    axis_names: tuple[str, ...] | None = None,
) -> Conflict:
    src_vec = np.asarray(src_vec, dtype=float)
    tgt_vec = np.asarray(tgt_vec, dtype=float)
    delta = tgt_vec - src_vec
    raw_d = float(np.linalg.norm(delta))
    if d_max is None:
        d_max = D_MAX_SPACE_H if space == "H" else max(raw_d, 1e-9)
    d_norm = float(np.clip(raw_d / d_max, 0.0, 1.0))
    unit = delta / raw_d if raw_d > 1e-12 else np.zeros_like(delta)
    if axis_names is None:
        axis_names = AXIS_NAMES if space == "H" else tuple(f"PC{i+1}" for i in range(delta.shape[0]))
    return Conflict(
        space=space,
        src_vec=src_vec,
        tgt_vec=tgt_vec,
        d=d_norm,
        delta=delta,
        delta_unit=unit,
        axis_names=axis_names,
    )


def bucketize(values: list[float], *, labels: tuple[str, str, str] = ("low", "mid", "high")) -> list[str]:
    """Split a list of d values into terciles."""
    vals = np.asarray(values, dtype=float)
    if len(vals) < 3:
        return [labels[1]] * len(vals)
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    out: list[str] = []
    for v in vals:
        if v <= q1:
            out.append(labels[0])
        elif v <= q2:
            out.append(labels[1])
        else:
            out.append(labels[2])
    return out


# ---------------------------------------------------------------------------
# Convenience: enumerate all (source, target_persona) pairs
# ---------------------------------------------------------------------------


def enumerate_pairs(
    sources: list[SourceText],
    coord_scores: list[CoordinateScore],
    personas: PersonaSet,
) -> list[dict]:
    """For every (source, persona) pair, compute Space-H conflict descriptor."""
    score_by_id = {cs.source_id: cs for cs in coord_scores}
    out: list[dict] = []
    for src in sources:
        cs = score_by_id.get(src.id)
        if cs is None:
            continue
        src_vec = cs.vec()
        for p in personas:
            c = compute_conflict(src_vec, p.vector, space="H")
            out.append(
                {
                    "source_id": src.id,
                    "genre": src.genre,
                    "target_persona": p.name,
                    "conflict_H": c.to_dict(),
                }
            )
    return out
