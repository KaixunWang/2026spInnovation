"""T3-only surprise vs headline composites (Gaussian MixedLM).

Same n=1440 pooled proprietary T3 discrete rows as judge divergence tables.
Structural-collapse GEE lives in ``scripts/t3_collapse_risk.py``.

Run::

    python scripts/t3_surprise_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t3_pooled_t3_loader import MAIN_METRICS_JSONL, load_t3_discrete_pooled_rows  # noqa: E402

OUT_SURP = ROOT / "results" / "tables" / "t3_surprise_curvature_compare.csv"
MAIN_METRICS = MAIN_METRICS_JSONL


def _fit_mixedlm_quadratic(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    out: dict[str, Any] = {
        "metric": metric,
        "n": 0,
        "fit_type": "",
        "beta_d": float("nan"),
        "beta_d2": float("nan"),
        "p_d2": float("nan"),
        "d_star": float("nan"),
        "error": "",
    }
    d = df.dropna(subset=[metric, "d_H", "genre", "source_id"]).copy()
    if d.empty:
        out["error"] = "no_data"
        return out
    if float(d[metric].std(ddof=0)) < 1e-12:
        out["error"] = "zero_variance"
        return out
    d["d"] = d["d_H"].astype(float)
    d["d2"] = d["d"] ** 2
    d["genre"] = d["genre"].astype("category")
    d["source_id"] = d["source_id"].astype(str)
    formula = f"{metric} ~ d + d2 + C(genre)"
    model = smf.mixedlm(formula, data=d, groups=d["source_id"])
    last_err: Exception | None = None
    for meth in ("lbfgs", "powell", "bfgs"):
        for reml in (True, False):
            try:
                fit = model.fit(method=meth, disp=False, reml=reml)
                out["fit_type"] = f"mixedlm_{meth}_reml{int(reml)}"
                out["n"] = int(len(d))
                b1 = float(fit.params.get("d", float("nan")))
                b2 = float(fit.params.get("d2", float("nan")))
                out["beta_d"] = b1
                out["beta_d2"] = b2
                out["p_d2"] = float(fit.pvalues.get("d2", float("nan")))
                if pd.notna(b2) and abs(b2) > 1e-12:
                    out["d_star"] = float(-b1 / (2.0 * b2))
                return out
            except Exception as e:
                last_err = e
    out["error"] = repr(last_err)
    return out


def main() -> int:
    if not MAIN_METRICS.exists():
        print(f"missing {MAIN_METRICS}", file=sys.stderr)
        return 1
    df = load_t3_discrete_pooled_rows()
    sub = df.dropna(subset=["surprise_mean"]).copy()
    sub["surprise_01"] = sub["surprise_mean"].astype(float) / 5.0
    OUT_SURP.parent.mkdir(parents=True, exist_ok=True)
    comp_rows: list[dict[str, Any]] = []
    for metric in ("surprise_01", "creativity_auto", "creativity_judge"):
        r = _fit_mixedlm_quadratic(sub, metric)
        comp_rows.append(
            {
                "outcome": metric,
                "n": r["n"],
                "fit_type": r["fit_type"],
                "beta_d2": r["beta_d2"],
                "p_d2": r["p_d2"],
                "d_star": r["d_star"],
                "error": r["error"],
            }
        )
    pd.DataFrame(comp_rows).to_csv(OUT_SURP, index=False)
    print(f"wrote {OUT_SURP.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
