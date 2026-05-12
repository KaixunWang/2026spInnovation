"""Qwen 4B/8B/14B: MixedLM same spirit as ``analyze.main_regression`` on T3-only rows.

For each ``main_qwen3_*_metrics.jsonl``:
  creativity_auto ~ d + d2 + C(genre)  with random intercept (source_id)

Uses statsmodels MixedLM; on failure falls back to OLS (same fixed-effects formula).

Run from repo root::

    python scripts/qwen_scale_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze import load_flat  # noqa: E402
from src.config_loader import PROJECT_ROOT  # noqa: E402

METRIC = "creativity_auto"
METRICS_FILES = [
    PROJECT_ROOT / "data" / "generated" / "main_qwen3_4b_metrics.jsonl",
    PROJECT_ROOT / "data" / "generated" / "main_qwen3_8b_metrics.jsonl",
    PROJECT_ROOT / "data" / "generated" / "main_qwen3_14b_metrics.jsonl",
    # Pooled OpenAI T3 discrete rows (720×4o + 720×mini), same convention as ``qwen_judge_scale_regression``.
    PROJECT_ROOT / "data" / "generated" / "main_metrics.jsonl",
]
OUT_CSV = PROJECT_ROOT / "results" / "tables" / "qwen_scale_regression.csv"


def _fit_one(df: pd.DataFrame, *, label: str) -> dict:
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[METRIC, "d_H", "genre", "source_id"]).copy()
    out: dict = {
        "label": label,
        "metric": METRIC,
        "n": int(len(d)),
        "fit_type": "",
        "beta_d": float("nan"),
        "beta_d2": float("nan"),
        "p_d": float("nan"),
        "p_d2": float("nan"),
        "d_star": float("nan"),
        "d2_significant_p05": False,
        "error": "",
    }
    if d.empty:
        out["error"] = "no_data_after_dropna"
        return out
    if float(d[METRIC].std(ddof=0)) < 1e-12:
        out["error"] = "near_zero_variance"
        return out

    d["d"] = d["d_H"].astype(float)
    d["d2"] = d["d"] ** 2
    d["genre"] = d["genre"].astype("category")
    d["source_id"] = d["source_id"].astype(str)

    formula = f"{METRIC} ~ d + d2 + C(genre)"
    fit_type = ""
    coefs: dict[str, float] = {}
    pvals: dict[str, float] = {}
    model = smf.mixedlm(formula, data=d, groups=d["source_id"])
    fit_ok = False
    last_err: Exception | None = None
    for meth in ("lbfgs", "powell", "bfgs"):
        for reml in (True, False):
            try:
                fit = model.fit(method=meth, disp=False, reml=reml)
                fit_type = f"mixedlm_{meth}_reml{int(reml)}"
                coefs = {k: float(v) for k, v in fit.params.items()}
                pvals = {k: float(v) for k, v in fit.pvalues.items()}
                fit_ok = True
                break
            except Exception as e:
                last_err = e
        if fit_ok:
            break
    if not fit_ok:
        fit_type = "ols"
        try:
            fit = smf.ols(formula, data=d).fit()
            coefs = {k: float(v) for k, v in fit.params.items()}
            pvals = {k: float(v) for k, v in fit.pvalues.items()}
        except Exception as e2:
            out["fit_type"] = "failed"
            out["error"] = f"mixedlm:{last_err!r}; ols:{e2!r}"
            return out

    out["fit_type"] = fit_type
    b1 = float(coefs.get("d", float("nan")))
    b2 = float(coefs.get("d2", float("nan")))
    out["beta_d"] = b1
    out["beta_d2"] = b2
    out["p_d"] = float(pvals.get("d", float("nan")))
    out["p_d2"] = float(pvals.get("d2", float("nan")))
    if pd.notna(b2) and abs(b2) > 1e-12:
        out["d_star"] = float(-b1 / (2.0 * b2))
    out["d2_significant_p05"] = bool(pd.notna(out["p_d2"]) and out["p_d2"] < 0.05)
    return out


def main() -> int:
    rows: list[dict] = []
    for path in METRICS_FILES:
        if not path.exists():
            rows.append(
                {
                    "metrics_file": path.name,
                    "generator_model": "",
                    "label": path.stem,
                    "metric": METRIC,
                    "n": 0,
                    "fit_type": "",
                    "beta_d": float("nan"),
                    "beta_d2": float("nan"),
                    "p_d": float("nan"),
                    "p_d2": float("nan"),
                    "d_star": float("nan"),
                    "d2_significant_p05": False,
                    "error": "file_not_found",
                }
            )
            continue

        base = load_flat(path)
        base["metrics_file"] = path.name
        sub = base.loc[base["condition"] == "T3"].copy()
        if sub.empty:
            rows.append(
                {
                    "metrics_file": path.name,
                    "generator_model": "",
                    "label": path.stem,
                    "metric": METRIC,
                    "n": 0,
                    "fit_type": "",
                    "beta_d": float("nan"),
                    "beta_d2": float("nan"),
                    "p_d": float("nan"),
                    "p_d2": float("nan"),
                    "d_star": float("nan"),
                    "d2_significant_p05": False,
                    "error": "no_T3_rows",
                }
            )
            continue

        gen_models = sub["model"].dropna().unique().tolist()
        gen_model = str(gen_models[0]) if len(gen_models) == 1 else "|".join(sorted(map(str, gen_models)))

        r = _fit_one(sub, label=path.stem)
        r["metrics_file"] = path.name
        r["generator_model"] = gen_model
        rows.append(r)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    # Stable column order
    cols = [
        "metrics_file",
        "generator_model",
        "label",
        "metric",
        "n",
        "fit_type",
        "beta_d",
        "beta_d2",
        "p_d",
        "p_d2",
        "d_star",
        "d2_significant_p05",
        "error",
    ]
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[cols]
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[qwen_scale_regression] wrote {OUT_CSV}")
    print(out_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
