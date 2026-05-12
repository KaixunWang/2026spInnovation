# Scope note: TTCW dataset (Salesforce/ttcw_creativity_eval) contains 48 narrative
# stories with binary expert annotations. Calibration is valid only for Elaboration,
# Flexibility, and Originality dimensions. Fidelity has no TTCW counterpart (TTCW
# evaluates free-form writing, not rewriting). Task structure differs: TTCW stories
# are free compositions; our judge scores rewrites against a source text. Results
# should be interpreted as exploratory alignment, not full external validation.
"""Correlate LLM judge constructs with TTCW expert aggregates (Elaboration / Flexibility / Originality only).

Input: a JSONL file with one object per line, each containing at least:
  - ``story_id``: must match ``story_metadata.story_id`` in the TTCW rows (e.g. ``0_NewYorker``).
  - ``novelty``, ``surprise``, ``imagery``: integers in 1--5 from our absolute rubric.

Optional overrides (if present, used instead of defaults):
  - ``elaboration``, ``flexibility`` (1--5 Likert) — mapped to [0,1] the same way as below.

We **do not** correlate fidelity, fluency, consistency, logical_completeness, or any
composite Creativity score against TTCW — those have no valid TTCW counterpart.

Construct mapping (our side → TTCW dimension for Pearson):
  - ``novelty_proxy`` = mean(novelty, surprise) → [0,1] via ``(mean - 1) / 4`` ↔ TTCW **Originality** aggregate.
  - ``elaboration`` defaults to ``imagery`` ↔ TTCW **Elaboration** aggregate (operational proxy: sensory / figurative richness).
  - ``flexibility`` defaults to mean pairwise |novelty−imagery|, |surprise−imagery| / 2, scaled to [0,1] by ``/ 4``
    (exploratory spread proxy; override with column ``flexibility`` if you prefer a dedicated rubric item).

TTCW side: each expert ``binary_verdict`` is Yes/No. For a Torrance **dimension** (Elaboration, Flexibility, Originality),
``dim_score = sum(passes) / n_annotations`` over **all** expert annotations pooled across every HF column whose
parsed ``test_metadata.ttcw_dimension`` matches that dimension. Raw per-annotation binaries are **not** treated as Likert.

Usage:
  python -m scripts.calibrate_judge_ttcw --judges path/to/ttcw_story_judges.jsonl

Install: ``pip install 'datasets>=2.19'`` (or ``pip install -e .[calibration]`` if you add the extra locally).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# HF column name → we filter by parsed test_metadata["ttcw_dimension"]
_TTCW_COLUMNS: list[str] = [
    "ttcw_narrative_ending",
    "ttcw_understandability_and_coherence",
    "ttcw_scene_vs_summary",
    "ttcw_narrative_pacing",
    "ttcw_language_proficiency_and_literary_devices",
    "ttcw_emotional_flexibility",
    "ttcw_structural_flexibility",
    "ttcw_perspective_and_voice_flexibility",
    "ttcw_originality_in_thought",
    "ttcw_originality_in_form_and_structure",
    "ttcw_originality_in_theme_and_content",
    "ttcw_rhetorical_complexity",
    "ttcw_world_building_and_setting",
    "ttcw_character_development",
]

_WARN_THRESHOLD = 0.15
_WARN_MSG = (
    "[WARN] Low correlation on {dim} - consistent with Chakrabarty et al. (2024) finding that LLM judges "
    "do not reliably align with human experts on TTCW. Consider reporting this as a limitation."
)


def _safe_literal_eval(blob: str) -> Any:
    blob = (blob or "").strip()
    if not blob:
        return []
    try:
        return ast.literal_eval(blob)
    except (ValueError, SyntaxError):
        return []


def _parse_cell_blocks(cell: str) -> tuple[dict | None, dict | None, list | None]:
    """Parse HF cell: either [{story},{test},{ann}] (v2 layout) or legacy [{merged}]."""
    data = _safe_literal_eval(cell)
    if not isinstance(data, list) or not data:
        return None, None, None
    if len(data) >= 3 and all(isinstance(x, dict) for x in data[:3]):
        sm = data[0].get("story_metadata") if isinstance(data[0].get("story_metadata"), dict) else None
        tm = data[1].get("test_metadata") if isinstance(data[1].get("test_metadata"), dict) else None
        ann = data[2].get("annotations") if isinstance(data[2].get("annotations"), list) else None
        return sm or {}, tm or {}, ann
    block = data[0]
    if isinstance(block, dict):
        return (
            block.get("story_metadata") if isinstance(block.get("story_metadata"), dict) else None,
            block.get("test_metadata") if isinstance(block.get("test_metadata"), dict) else None,
            block.get("annotations") if isinstance(block.get("annotations"), list) else None,
        )
    return None, None, None


def _pass_rate_from_cell(cell: str) -> tuple[int, int] | None:
    """Return (n_yes, n_total) for one HF cell, or None if unparseable."""
    _, _, ann = _parse_cell_blocks(cell)
    if not isinstance(ann, list):
        return None
    n_yes = 0
    n_tot = 0
    for a in ann:
        if not isinstance(a, dict):
            continue
        v = str(a.get("binary_verdict", "")).strip().lower()
        if v not in ("yes", "no"):
            continue
        n_tot += 1
        if v == "yes":
            n_yes += 1
    if n_tot == 0:
        return None
    return n_yes, n_tot


def _story_id_from_cell(cell: str) -> str | None:
    sm, _, _ = _parse_cell_blocks(cell)
    if not isinstance(sm, dict):
        return None
    sid = sm.get("story_id")
    return str(sid) if sid is not None else None


def _dimension_from_cell(cell: str) -> str | None:
    _, tm, _ = _parse_cell_blocks(cell)
    if not isinstance(tm, dict):
        return None
    dim = tm.get("ttcw_dimension")
    return str(dim) if dim is not None else None


def _likert_to_01(x: float) -> float:
    """Map 1--5 Likert to approximately [0, 1]."""
    return float(np.clip((float(x) - 1.0) / 4.0, 0.0, 1.0))


def _our_scores(row: dict[str, Any]) -> dict[str, float] | None:
    try:
        n = int(row["novelty"])
        s = int(row["surprise"])
        i = int(row["imagery"])
    except (KeyError, TypeError, ValueError):
        return None
    novelty_proxy = _likert_to_01((n + s) / 2.0)
    if row.get("elaboration") is not None:
        elaboration = _likert_to_01(float(row["elaboration"]))
    else:
        elaboration = _likert_to_01(float(i))
    if row.get("flexibility") is not None:
        flexibility = _likert_to_01(float(row["flexibility"]))
    else:
        spread = (abs(float(n) - float(i)) + abs(float(s) - float(i))) / 2.0
        flexibility = float(np.clip(spread / 4.0, 0.0, 1.0))
    return {
        "novelty_proxy": novelty_proxy,
        "elaboration": elaboration,
        "flexibility": flexibility,
    }


def build_ttcw_table() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Missing dependency `datasets`. Install with: pip install 'datasets>=2.19'\n"
            "or: pip install -e .[calibration]"
        ) from e

    ds = load_dataset("Salesforce/ttcw_creativity_eval", split="train")
    rows_out: list[dict[str, Any]] = []
    for idx in range(len(ds)):
        sample = ds[idx]
        story_id: str | None = None
        passes: dict[str, int] = {"Elaboration": 0, "Flexibility": 0, "Originality": 0}
        totals: dict[str, int] = {k: 0 for k in passes}

        for col in _TTCW_COLUMNS:
            cell = sample.get(col, "")
            if not isinstance(cell, str) or not cell.strip():
                continue
            if story_id is None:
                story_id = _story_id_from_cell(cell)
            dim = _dimension_from_cell(cell)
            if dim not in passes:
                continue
            pr = _pass_rate_from_cell(cell)
            if pr is None:
                continue
            n_yes, n_tot = pr
            passes[dim] += n_yes
            totals[dim] += n_tot

        if not story_id:
            continue
        row: dict[str, Any] = {"story_id": story_id, "row_index": idx}
        for d in ("Elaboration", "Flexibility", "Originality"):
            if totals[d] > 0:
                row[f"ttcw_{d.lower()}"] = passes[d] / totals[d]
            else:
                row[f"ttcw_{d.lower()}"] = float("nan")
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def load_judges(path: Path) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    df = pd.DataFrame(recs)
    if "story_id" not in df.columns:
        raise SystemExit("judges JSONL must contain 'story_id' aligned with TTCW story_metadata.story_id")
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TTCW calibration: Elaboration, Flexibility, Originality only.")
    p.add_argument(
        "--judges",
        type=Path,
        required=True,
        help="JSONL with story_id, novelty, surprise, imagery (1-5); optional elaboration, flexibility.",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional CSV path for merged table.")
    args = p.parse_args(argv)

    ttcw = build_ttcw_table()
    judges = load_judges(args.judges)

    ours: list[dict[str, Any]] = []
    for _, r in judges.iterrows():
        row = r.to_dict()
        scores = _our_scores(row)
        if scores is None:
            continue
        ours.append({"story_id": row.get("story_id"), **scores})
    jdf = pd.DataFrame(ours)
    merged = jdf.merge(ttcw, on="story_id", how="inner")
    if merged.empty:
        print("No overlapping story_id between judges JSONL and TTCW.", file=sys.stderr)
        return 1

    # Correlation table: ONLY these three rows (no fidelity / no composite Creativity).
    pairs = [
        ("elaboration", "ttcw_elaboration", "elaboration", "TTCW Elaboration (pass rate)"),
        ("flexibility", "ttcw_flexibility", "flexibility", "TTCW Flexibility (pass rate)"),
        ("novelty_proxy", "ttcw_originality", "novelty_proxy", "TTCW Originality (pass rate)"),
    ]

    print("\n=== Pearson correlation (exploratory; TTCW = sum(passes)/total binary tests per dimension) ===\n")
    print(f"{'our_construct':<16} {'TTCW_target':<28} {'r':>8} {'p':>10} {'n':>5}")
    print("-" * 72)
    for our_col, ttcw_col, dim_key, ttcw_label in pairs:
        mask = merged[our_col].notna() & merged[ttcw_col].notna()
        x = merged.loc[mask, our_col].astype(float).values
        y = merged.loc[mask, ttcw_col].astype(float).values
        if len(x) < 3:
            r, p = float("nan"), float("nan")
        else:
            r, p = pearsonr(x, y)
        print(f"{our_col:<16} {ttcw_label:<28} {r:8.3f} {p:10.4g} {len(x):5d}")
        if np.isfinite(r) and abs(r) < _WARN_THRESHOLD:
            print(_WARN_MSG.format(dim=dim_key))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out, index=False)
        print(f"\nWrote merged table to {args.out}")

    print(
        "\nNote: Fidelity, fluency, consistency, logical_completeness, and composite Creativity "
        "are intentionally excluded from this table (no valid TTCW counterpart / wrong task structure)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
