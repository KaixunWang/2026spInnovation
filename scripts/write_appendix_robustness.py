"""Write Appendix C robustness tables from results/flat.csv (paper pool: no Qwen, no multihop).

Outputs:
  results/tables/appendix_mixedlm_vs_ols.csv
  results/tables/appendix_binning_t3_minus_t2.csv

Run from repo root:  python scripts/write_appendix_robustness.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
FLAT = ROOT / "results" / "flat.csv"
OUT_DIR = ROOT / "results" / "tables"

PAPER_FILES = ("main_metrics", "main_continuous_metrics", "mechanism_metrics")


def _paper_pool(df: pd.DataFrame) -> pd.DataFrame:
    # load_flat assigns source_file = Path(...).stem (e.g. main_metrics).
    return df[df["source_file"].astype(str).isin(PAPER_FILES)].copy()


def _fit_both(metric: str, pool: pd.DataFrame) -> tuple[dict, dict]:
    d = pool.dropna(subset=[metric, "d_H", "source_id"]).copy()
    d["d"] = d["d_H"].astype(float)
    d["d2"] = d["d"] ** 2
    d["genre"] = d["genre"].astype("category")
    d["condition"] = d["condition"].astype("category")
    formula = f"{metric} ~ d + d2 + C(genre) + C(condition)"
    ols = smf.ols(formula, data=d).fit()
    ols_row = {
        "metric": metric,
        "model": "OLS",
        "n": int(len(d)),
        "beta_d2": float(ols.params["d2"]),
        "p_d2": float(ols.pvalues["d2"]),
    }
    model = smf.mixedlm(formula, data=d, groups=d["source_id"])
    fit = None
    meth_used = ""
    reml_used = ""
    for meth in ("lbfgs", "powell", "bfgs"):
        for reml in (True, False):
            try:
                fit = model.fit(method=meth, disp=False, reml=reml)
                meth_used, reml_used = meth, str(reml)
                break
            except Exception:
                continue
        if fit is not None:
            break
    if fit is None:
        raise RuntimeError("MixedLM did not converge")
    ml_row = {
        "metric": metric,
        "model": f"MixedLM({meth_used}, reml={reml_used})",
        "n": int(len(d)),
        "beta_d2": float(fit.params["d2"]),
        "p_d2": float(fit.pvalues["d2"]),
    }
    return ols_row, ml_row


def _binning_gaps(pool: pd.DataFrame, how: str) -> dict[str, float]:
    d = pool.dropna(subset=["creativity_auto", "d_H", "condition"]).copy()
    if how == "quantile":
        d["bucket"] = pd.qcut(d["d_H"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    else:
        d["bucket"] = pd.cut(d["d_H"], bins=3, labels=["low", "mid", "high"], duplicates="drop")
    out: dict[str, float] = {}
    for b, grp in d.groupby("bucket", observed=True):
        m3 = grp.loc[grp["condition"] == "T3", "creativity_auto"].mean()
        m2 = grp.loc[grp["condition"] == "T2", "creativity_auto"].mean()
        key = str(b)
        if pd.notna(m3) and pd.notna(m2):
            out[key] = float(m3 - m2)
        else:
            out[key] = float("nan")
    return out


def main() -> int:
    df = pd.read_csv(FLAT)
    pool = _paper_pool(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric in ("creativity_auto", "novelty_auto_combined"):
        ols_r, ml_r = _fit_both(metric, pool)
        rows.append(ols_r)
        rows.append(ml_r)
    pd.DataFrame(rows).to_csv(OUT_DIR / "appendix_mixedlm_vs_ols.csv", index=False)

    b_rows = []
    for how, label in (("quantile", "quantile_terciles"), ("cut", "equal_width_3_bins")):
        g = _binning_gaps(pool, how)
        b_rows.append(
            {
                "scheme": label,
                "low_T3_minus_T2": g.get("low", float("nan")),
                "mid_T3_minus_T2": g.get("mid", float("nan")),
                "high_T3_minus_T2": g.get("high", float("nan")),
            }
        )
    pd.DataFrame(b_rows).to_csv(OUT_DIR / "appendix_binning_t3_minus_t2.csv", index=False)
    print(f"[appendix_robustness] wrote {OUT_DIR / 'appendix_mixedlm_vs_ols.csv'}")
    print(f"[appendix_robustness] wrote {OUT_DIR / 'appendix_binning_t3_minus_t2.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
