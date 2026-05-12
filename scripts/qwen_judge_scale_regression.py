"""T3-only scale regressions on judge outcomes (and per-dimension means), Qwen + main.

For each metrics JSONL (Qwen3 4B/8B/14B and ``main_metrics.jsonl``), T3 rows only:

  Y ~ d_H + d_H^2 + C(genre)  with random intercept on ``source_id``

(same mixed-effects spirit as ``scripts/qwen_scale_regression.py``; full OLS with both
``C(genre)`` and ``C(source_id)`` is rank-deficient when each source has a single genre.)

Outcomes:

  * ``creativity_judge``, ``novelty_judge`` (merged ``judge`` block on metrics rows)
  * each rubric dimension from ``judge.per_dim_mean``, scaled to $[0,1]$ as mean$/5$

Writes:

  * ``results/tables/qwen_judge_scale_regression.csv``
  * ``results/tables/qwen_judge_auto_creativity_compare.csv`` — per model,
    ``creativity_auto`` vs ``creativity_judge``: $\\beta_{d^2}$, $p$, significance.

Requires judge fields on metrics rows (run ``merge_judge_into_metrics`` for main).

Run from repo root::

    python scripts/qwen_judge_scale_regression.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import PROJECT_ROOT  # noqa: E402
from src.io_utils import read_jsonl  # noqa: E402
from src.judge import ABSOLUTE_RUBRIC_DIMS  # noqa: E402
from src.metrics.value import fuse_fidelity_judge_nli  # noqa: E402

METRICS_SPECS: list[tuple[str, Path]] = [
    ("Qwen3-4B", PROJECT_ROOT / "data" / "generated" / "main_qwen3_4b_metrics.jsonl"),
    ("Qwen3-8B", PROJECT_ROOT / "data" / "generated" / "main_qwen3_8b_metrics.jsonl"),
    ("Qwen3-14B", PROJECT_ROOT / "data" / "generated" / "main_qwen3_14b_metrics.jsonl"),
    ("GPT-4o", PROJECT_ROOT / "data" / "generated" / "main_metrics.jsonl"),
]

OUT_DETAIL = PROJECT_ROOT / "results" / "tables" / "qwen_judge_scale_regression.csv"
OUT_COMPARE = PROJECT_ROOT / "results" / "tables" / "qwen_judge_auto_creativity_compare.csv"


def _row_from_json(r: dict[str, Any]) -> dict[str, Any] | None:
    if r.get("condition") != "T3":
        return None
    m = r.get("metrics") or {}
    if not m.get("ok"):
        return None
    flat: dict[str, Any] = {
        "source_id": str(r.get("source_id", "")),
        "genre": str(r.get("genre", "unknown")),
        "model": r.get("model"),
        "d_H": r.get("d_H"),
        "creativity_auto": m.get("creativity_auto"),
    }
    j = r.get("judge") or {}
    if j.get("ok"):
        flat["novelty_judge"] = j.get("novelty_judge")
        nj, cj, fj = j.get("novelty_judge"), j.get("coherence_judge"), j.get("fidelity_judge")
        if nj is not None and cj is not None and fj is not None:
            ff = fuse_fidelity_judge_nli(fj, m.get("nli_entailment"), w_judge=0.5)
            value_j = float((max(0.0, ff) + max(0.0, float(cj))) / 2.0)
            flat["creativity_judge"] = float(nj) * value_j
        else:
            flat["creativity_judge"] = j.get("creativity_judge")
        pdim = j.get("per_dim_mean") or {}
        for dim in ABSOLUTE_RUBRIC_DIMS:
            v = pdim.get(dim)
            key = f"judge_dim__{dim}"
            flat[key] = float(v) / 5.0 if v is not None else float("nan")
    else:
        flat["creativity_judge"] = float("nan")
        flat["novelty_judge"] = float("nan")
        for dim in ABSOLUTE_RUBRIC_DIMS:
            flat[f"judge_dim__{dim}"] = float("nan")
    return flat


def load_t3_frame(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in read_jsonl(path):
        row = _row_from_json(r)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_scale(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[metric, "d_H", "genre", "source_id"]).copy()
    out: dict[str, Any] = {
        "metric": metric,
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
    if float(d[metric].std(ddof=0)) < 1e-12:
        out["error"] = "near_zero_variance"
        return out

    d["d"] = d["d_H"].astype(float)
    d["d2"] = d["d"] ** 2
    d["genre"] = d["genre"].astype("category")
    d["source_id"] = d["source_id"].astype(str)

    formula = f"{metric} ~ d + d2 + C(genre)"
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


def _outcome_columns() -> list[str]:
    cols = ["creativity_judge", "novelty_judge"]
    cols += [f"judge_dim__{d}" for d in ABSOLUTE_RUBRIC_DIMS]
    return cols


def _pretty_outcome(name: str) -> str:
    if name.startswith("judge_dim__"):
        return "dim_" + name.replace("judge_dim__", "", 1)
    return name


def main() -> int:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="statsmodels.regression.mixed_linear_model",
    )
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning as _SMConv
    except ImportError:
        _SMConv = None  # type: ignore[misc, assignment]
    if _SMConv is not None:
        warnings.filterwarnings("ignore", category=_SMConv)

    detail_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []

    for label, path in METRICS_SPECS:
        if not path.exists():
            for oc in _outcome_columns():
                detail_rows.append(
                    {
                        "generator_label": label,
                        "metrics_file": path.name,
                        "generator_model": "",
                        "outcome": _pretty_outcome(oc),
                        "outcome_column": oc,
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
            compare_rows.append(
                {
                    "generator_label": label,
                    "metrics_file": path.name,
                    "n_T3": 0,
                    "beta_d2_creativity_auto": float("nan"),
                    "p_d2_creativity_auto": float("nan"),
                    "sig_p05_creativity_auto": False,
                    "beta_d2_creativity_judge": float("nan"),
                    "p_d2_creativity_judge": float("nan"),
                    "sig_p05_creativity_judge": False,
                    "same_sign_beta_d2": "",
                    "both_sig_p05_d2": "",
                    "error": "file_not_found",
                }
            )
            continue

        df = load_t3_frame(path)
        if df.empty:
            err = "no_T3_rows"
        else:
            err = ""

        gen_models = df["model"].dropna().unique().tolist() if not df.empty else []
        gen_model = str(gen_models[0]) if len(gen_models) == 1 else "|".join(sorted(map(str, gen_models)))

        for oc in _outcome_columns():
            rfit = _fit_scale(df, oc)
            detail_rows.append(
                {
                    "generator_label": label,
                    "metrics_file": path.name,
                    "generator_model": gen_model,
                    "outcome": _pretty_outcome(oc),
                    "outcome_column": oc,
                    "n": rfit["n"],
                    "fit_type": rfit["fit_type"],
                    "beta_d": rfit["beta_d"],
                    "beta_d2": rfit["beta_d2"],
                    "p_d": rfit["p_d"],
                    "p_d2": rfit["p_d2"],
                    "d_star": rfit["d_star"],
                    "d2_significant_p05": rfit["d2_significant_p05"],
                    "error": rfit["error"] or err,
                }
            )

        r_auto = _fit_scale(df, "creativity_auto")
        r_cj = _fit_scale(df, "creativity_judge")
        b2a, b2j = r_auto["beta_d2"], r_cj["beta_d2"]
        same = ""
        if pd.notna(b2a) and pd.notna(b2j) and abs(b2a) > 1e-15 and abs(b2j) > 1e-15:
            same = "yes" if (b2a > 0) == (b2j > 0) else "no"
        elif pd.notna(b2a) and pd.notna(b2j):
            same = "yes" if abs(b2a) < 1e-15 and abs(b2j) < 1e-15 else "mixed"
        both_sig = (
            bool(r_auto["d2_significant_p05"] and r_cj["d2_significant_p05"])
            if r_cj["n"] > 0
            else False
        )
        compare_rows.append(
            {
                "generator_label": label,
                "metrics_file": path.name,
                "generator_model": gen_model,
                "n_T3": int(len(df)),
                "n_fit_creativity_auto": r_auto["n"],
                "n_fit_creativity_judge": r_cj["n"],
                "beta_d2_creativity_auto": r_auto["beta_d2"],
                "p_d2_creativity_auto": r_auto["p_d2"],
                "sig_p05_creativity_auto": r_auto["d2_significant_p05"],
                "beta_d2_creativity_judge": r_cj["beta_d2"],
                "p_d2_creativity_judge": r_cj["p_d2"],
                "sig_p05_creativity_judge": r_cj["d2_significant_p05"],
                "same_sign_beta_d2": same,
                "both_sig_p05_d2": both_sig,
                "error": err or r_cj["error"] or r_auto["error"],
            }
        )

    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(OUT_DETAIL, index=False, encoding="utf-8")
    pd.DataFrame(compare_rows).to_csv(OUT_COMPARE, index=False, encoding="utf-8")
    print(f"[qwen_judge_scale_regression] wrote {OUT_DETAIL}")
    print(f"[qwen_judge_scale_regression] wrote {OUT_COMPARE}")
    print(pd.DataFrame(compare_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
