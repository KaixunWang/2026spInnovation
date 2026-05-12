"""Exploratory structural-collapse binomial GEE on pooled proprietary T3 rows.

Companion to ``scripts/t3_surprise_regression.py`` (rubric surprise MixedLM only).
See Appendix~D in the technical report. Run::

    python scripts/t3_collapse_risk.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t3_pooled_t3_loader import MAIN_METRICS_JSONL, load_t3_discrete_pooled_rows  # noqa: E402

TABLES_DIR = ROOT / "results" / "tables"
MAIN_METRICS = MAIN_METRICS_JSONL


def _fit_gee_collapse_nogenre(df: pd.DataFrame) -> tuple[pd.Series, str, Any]:
    from statsmodels.genmod import families
    from statsmodels.genmod.cov_struct import Exchangeable, Independence
    from statsmodels.genmod.generalized_estimating_equations import GEE

    d = df.dropna(subset=["collapse", "d_H", "source_id"]).copy()
    d["d"] = d["d_H"].astype(float)
    d["d2"] = d["d"] ** 2
    for struct, label in ((Exchangeable(), "exchangeable"), (Independence(), "independence")):
        try:
            model = GEE.from_formula(
                "collapse ~ d + d2",
                groups=d["source_id"],
                family=families.Binomial(),
                cov_struct=struct,
                data=d,
            )
            fit = model.fit(maxiter=200, ddof_scale=1)
            return fit.params, label, fit
        except Exception:
            continue
    raise RuntimeError("GEE (no genre) did not converge")


def _fit_gee_collapse(df: pd.DataFrame) -> tuple[pd.Series, str, Any]:
    from statsmodels.genmod import families
    from statsmodels.genmod.cov_struct import Exchangeable, Independence
    from statsmodels.genmod.generalized_estimating_equations import GEE

    d = df.dropna(subset=["collapse", "d_H", "genre", "source_id"]).copy()
    d["d"] = d["d_H"].astype(float)
    d["d2"] = d["d"] ** 2
    d["genre"] = d["genre"].astype("category")
    for struct, label in ((Exchangeable(), "exchangeable"), (Independence(), "independence")):
        try:
            model = GEE.from_formula(
                "collapse ~ d + d2 + C(genre)",
                groups=d["source_id"],
                family=families.Binomial(),
                cov_struct=struct,
                data=d,
            )
            fit = model.fit(maxiter=200, ddof_scale=1)
            return fit.params, label, fit
        except Exception:
            continue
    raise RuntimeError("GEE did not converge with exchangeable or independence")


def _p_curve_nogenre(fit: Any, d_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_df = pd.DataFrame({"d": d_grid, "d2": d_grid**2})
    lin = fit.predict(pred_df)
    pv = np.asarray(lin, dtype=float)
    if pv.max() > 1.0 or pv.min() < -0.01:
        z = np.clip(pv, -50.0, 50.0)
        pv = 1.0 / (1.0 + np.exp(-z))
    dp = np.gradient(pv, d_grid)
    return pv, dp


def _marginal_p_curve(fit: Any, df: pd.DataFrame, d_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genres = df["genre"].astype(str).value_counts(normalize=True)
    p_acc = np.zeros_like(d_grid, dtype=float)
    for g, w in genres.items():
        pred_df = pd.DataFrame(
            {
                "d": d_grid,
                "d2": d_grid**2,
                "genre": pd.Categorical([g] * len(d_grid), categories=df["genre"].astype("category").cat.categories),
            }
        )
        lin = fit.predict(pred_df, offset=None)
        pv = np.asarray(lin, dtype=float)
        if pv.max() > 1.0 or pv.min() < -0.01:
            z = np.clip(pv, -50.0, 50.0)
            pv = 1.0 / (1.0 + np.exp(-z))
        p_acc += w * pv
    dp = np.gradient(p_acc, d_grid)
    return p_acc, dp


def _run_one_tau(df: pd.DataFrame, q: float, suffix: str) -> int:
    tau_nli = float(df["nli_entailment"].quantile(q))
    tau_coh = float(df["coherence_auto"].quantile(q))
    dfc = df.assign(
        collapse=(
            (df["nli_entailment"] < tau_nli) & (df["coherence_auto"] < tau_coh)
        ).astype(int)
    )

    thresh_row = {
        "tau_quantile": q,
        "tau_nli_entailment": tau_nli,
        "tau_coherence_auto": tau_coh,
        "n_rows": len(dfc),
        "n_sources": dfc["source_id"].nunique(),
        "collapse_rate": float(dfc["collapse"].mean()),
    }
    pd.DataFrame([thresh_row]).to_csv(TABLES_DIR / f"t3_collapse_thresholds_{suffix}.csv", index=False)

    try:
        params, cov_label, gee_fit = _fit_gee_collapse(dfc)
    except RuntimeError as e:
        print(f"[tau {q}] {e}", file=sys.stderr)
        pd.DataFrame([{"error": str(e), "cov_struct": "failed"}]).to_csv(
            TABLES_DIR / f"t3_collapse_gee_coefs_{suffix}.csv", index=False
        )
        return 1

    gee_rows = [
        {
            "cov_struct": cov_label,
            "param": str(name),
            "coef": float(val),
            "pvalue": float(gee_fit.pvalues[name]) if name in gee_fit.pvalues.index else float("nan"),
        }
        for name, val in params.items()
    ]
    pd.DataFrame(gee_rows).to_csv(TABLES_DIR / f"t3_collapse_gee_coefs_{suffix}.csv", index=False)

    d_grid = np.linspace(0.22, 0.88, 200)
    p_curve, dp = _marginal_p_curve(gee_fit, dfc, d_grid)
    i_max = int(np.nanargmax(dp))
    curve_row = {
        "d_argmax_dpdd": float(d_grid[i_max]),
        "p_marginal_at_d_lo": float(p_curve[0]),
        "p_marginal_at_d_hi": float(p_curve[-1]),
        "p_marginal_at_argmax_slope": float(p_curve[i_max]),
        "beta_d": float(params.get("d", float("nan"))),
        "beta_d2": float(params.get("d2", float("nan"))),
        "p_d2": float(gee_fit.pvalues.get("d2", float("nan"))),
    }
    pd.DataFrame([curve_row]).to_csv(TABLES_DIR / f"t3_collapse_curve_summary_{suffix}.csv", index=False)

    if suffix == "q20":
        try:
            p2, lab2, fit2 = _fit_gee_collapse_nogenre(dfc)
            rows2 = [
                {
                    "cov_struct": lab2,
                    "param": str(name),
                    "coef": float(val),
                    "pvalue": float(fit2.pvalues[name]) if name in fit2.pvalues.index else float("nan"),
                }
                for name, val in p2.items()
            ]
            pd.DataFrame(rows2).to_csv(TABLES_DIR / "t3_collapse_gee_coefs_q20_nogenre.csv", index=False)
            p_curve, dp = _p_curve_nogenre(fit2, d_grid)
            i2 = int(np.nanargmax(dp))
            pd.DataFrame(
                [
                    {
                        "d_argmax_dpdd": float(d_grid[i2]),
                        "p_at_d_lo": float(p_curve[0]),
                        "p_at_d_hi": float(p_curve[-1]),
                        "p_at_argmax_slope": float(p_curve[i2]),
                        "beta_d": float(p2.get("d", float("nan"))),
                        "beta_d2": float(p2.get("d2", float("nan"))),
                        "p_d2": float(fit2.pvalues.get("d2", float("nan"))),
                    }
                ]
            ).to_csv(TABLES_DIR / "t3_collapse_curve_summary_q20_nogenre.csv", index=False)
        except RuntimeError as e:
            pd.DataFrame([{"error": str(e)}]).to_csv(
                TABLES_DIR / "t3_collapse_gee_coefs_q20_nogenre.csv", index=False
            )
    return 0


def main() -> int:
    if not MAIN_METRICS.exists():
        print(f"missing {MAIN_METRICS}", file=sys.stderr)
        return 1
    df = load_t3_discrete_pooled_rows()
    if df.empty:
        print("no T3 discrete pooled rows", file=sys.stderr)
        return 1
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    for q, suffix in ((0.2, "q20"), (0.5, "q50")):
        if _run_one_tau(df, q, suffix) != 0:
            rc = 1
    print("wrote t3_collapse_*_{q20,q50}.csv")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
