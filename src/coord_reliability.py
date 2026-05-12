"""Diagnostics for deterministic Space-H coordinates (genre separation).

With ``coord_scoring.backend: hf``, source (S, R) are reproducible; classical ICC
across two LLM scorers is replaced by:

  * per-genre mean/std of S and R
  * pairwise Euclidean distances between genre centroids
  * one-way ANOVA and Kruskal–Wallis p-values for S and R across genres

Writes ``results/tables/coord_reliability.csv`` (per-source rows + summary rows).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal

from .config_loader import PROJECT_ROOT, load_experiment_config
from .corpus import SourceText, load_sources
from .io_utils import GENERATED_DIR, RESULTS_DIR, read_jsonl
from .progress_util import iter_progress


def _resolve_coords_path() -> Path:
    cfg = load_experiment_config()
    rel = (cfg.get("coord_scoring") or {}).get("source_coords_path") or "data/source_coords.jsonl"
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_score_table(path: Path) -> dict[str, dict[str, Any]]:
    """Map source_id -> {S, R, ...}."""
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = row.get("id") or row.get("source_id")
        if sid is None:
            continue
        s_key = "S" if "S" in row else "S_mean"
        r_key = "R" if "R" in row else "R_mean"
        out[str(sid)] = {"S": float(row[s_key]), "R": float(row[r_key])}
    return out


def run_coord_reliability(
    *,
    sources: list[SourceText] | None = None,
    out_path: Path | None = None,
) -> Path:
    cfg = load_experiment_config()
    precalc_path = _resolve_coords_path()
    if not precalc_path.exists():
        fallback = GENERATED_DIR / "coord_scores.jsonl"
        if fallback.exists():
            precalc_path = fallback
        else:
            raise FileNotFoundError(
                f"coord_reliability requires {precalc_path} or {fallback}. "
                "Run: python scripts/recompute_coords.py (after build_axis_vectors.py), or python -m src.run_experiment coord."
            )

    score_by_id = _load_score_table(precalc_path)
    if sources is None:
        sources = load_sources()

    rows: list[dict[str, Any]] = []
    for src in iter_progress(sources, total=len(sources), desc="[coord_rel] sources", unit="src"):
        sc = score_by_id.get(src.id)
        if sc is None:
            rows.append(
                {
                    "row_type": "source",
                    "source_id": src.id,
                    "genre": src.genre,
                    "S": "",
                    "R": "",
                    "note": "missing_in_precalc",
                }
            )
            continue
        rows.append(
            {
                "row_type": "source",
                "source_id": src.id,
                "genre": src.genre,
                "S": sc["S"],
                "R": sc["R"],
                "note": "",
            }
        )

    detail = pd.DataFrame(rows)
    src_only = detail[detail["row_type"] == "source"].copy()
    src_only = src_only[src_only["S"] != ""]
    src_only["S"] = pd.to_numeric(src_only["S"], errors="coerce")
    src_only["R"] = pd.to_numeric(src_only["R"], errors="coerce")
    src_only = src_only.dropna(subset=["S", "R"])

    summaries: list[dict[str, Any]] = []
    genres = sorted(src_only["genre"].unique().tolist())
    for g in genres:
        grp = src_only[src_only["genre"] == g]
        summaries.append(
            {
                "row_type": "summary_genre",
                "genre": g,
                "n_sources": len(grp),
                "mean_S": float(grp["S"].mean()),
                "std_S": float(grp["S"].std(ddof=1)) if len(grp) > 1 else 0.0,
                "mean_R": float(grp["R"].mean()),
                "std_R": float(grp["R"].std(ddof=1)) if len(grp) > 1 else 0.0,
            }
        )

    # Centroid distances
    centroids: dict[str, np.ndarray] = {}
    for g in genres:
        grp = src_only[src_only["genre"] == g]
        centroids[g] = np.array([grp["S"].mean(), grp["R"].mean()], dtype=float)
    for i, g1 in enumerate(genres):
        for g2 in genres[i + 1 :]:
            d = float(np.linalg.norm(centroids[g1] - centroids[g2]))
            summaries.append(
                {
                    "row_type": "summary_centroid_dist",
                    "genre_a": g1,
                    "genre_b": g2,
                    "euclidean_d": d,
                }
            )

    # ANOVA / Kruskal–Wallis
    groups_s = [src_only.loc[src_only["genre"] == g, "S"].values for g in genres if len(src_only[src_only["genre"] == g]) > 0]
    groups_r = [src_only.loc[src_only["genre"] == g, "R"].values for g in genres if len(src_only[src_only["genre"] == g]) > 0]
    if len(groups_s) >= 2 and all(len(g) > 0 for g in groups_s):
        f_s = f_oneway(*groups_s)
        kw_s = kruskal(*groups_s)
        summaries.append(
            {
                "row_type": "summary_tests",
                "axis": "S",
                "anova_F": float(f_s.statistic),
                "anova_p": float(f_s.pvalue),
                "kruskal_H": float(kw_s.statistic),
                "kruskal_p": float(kw_s.pvalue),
            }
        )
    if len(groups_r) >= 2 and all(len(g) > 0 for g in groups_r):
        f_r = f_oneway(*groups_r)
        kw_r = kruskal(*groups_r)
        summaries.append(
            {
                "row_type": "summary_tests",
                "axis": "R",
                "anova_F": float(f_r.statistic),
                "anova_p": float(f_r.pvalue),
                "kruskal_H": float(kw_r.statistic),
                "kruskal_p": float(kw_r.pvalue),
            }
        )

    sum_df = pd.DataFrame(summaries)
    for c in sum_df.columns:
        if c not in detail.columns:
            detail[c] = np.nan
    for c in detail.columns:
        if c not in sum_df.columns:
            sum_df[c] = np.nan
    out = pd.concat([detail, sum_df], ignore_index=True)

    out_path = out_path or (RESULTS_DIR / "tables" / "coord_reliability.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        f"[coord_reliability] wrote {out_path} ({len(src_only)} coord rows + {len(summaries)} summary rows)"
    )
    return out_path
