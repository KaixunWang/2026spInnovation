"""Compare an external evaluator against automatic creativity on T3 rows.

This script is the analysis-side companion to
``scripts/merge_external_eval_into_metrics.py``. It keeps the existing judge
channel untouched and reads a parallel external block (default: ``creval``)
from metrics rows.

Outputs:

  * results/tables/<field>_scale_regression.csv
  * results/tables/<field>_auto_compare.csv
  * results/figures/scale_inverted_u_<field>.png

Default inputs mirror the current paper's scale-comparison slice:

  * data/generated/main_qwen3_4b_metrics.jsonl
  * data/generated/main_qwen3_8b_metrics.jsonl
  * data/generated/main_qwen3_14b_metrics.jsonl
  * data/generated/main_metrics.jsonl

Run from repo root::

    python scripts/external_eval_scale_regression.py --field-name creval --score-key score
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import PROJECT_ROOT  # noqa: E402
from src.io_utils import read_jsonl  # noqa: E402

DEFAULT_SPECS: list[tuple[str, Path]] = [
    ("Qwen3-4B", PROJECT_ROOT / "data" / "generated" / "main_qwen3_4b_metrics.jsonl"),
    ("Qwen3-8B", PROJECT_ROOT / "data" / "generated" / "main_qwen3_8b_metrics.jsonl"),
    ("Qwen3-14B", PROJECT_ROOT / "data" / "generated" / "main_qwen3_14b_metrics.jsonl"),
    ("GPT-4o", PROJECT_ROOT / "data" / "generated" / "main_metrics.jsonl"),
]


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "external_eval"


def _metric_col(field_name: str) -> str:
    return f"{_slug(field_name)}_score"


def _extract_external_score(row: dict[str, Any], field_name: str, score_key: str) -> float | None:
    block = row.get(field_name) or {}
    if not isinstance(block, dict):
        return None
    if "ok" in block and not block.get("ok"):
        return None
    score = block.get(score_key)
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _row_from_json(r: dict[str, Any], field_name: str, score_key: str) -> dict[str, Any] | None:
    if r.get("condition") != "T3":
        return None
    m = r.get("metrics") or {}
    if not m.get("ok"):
        return None
    ext_score = _extract_external_score(r, field_name, score_key)
    return {
        "source_id": str(r.get("source_id", "")),
        "genre": str(r.get("genre", "unknown")),
        "model": r.get("model"),
        "d_H": r.get("d_H"),
        "creativity_auto": m.get("creativity_auto"),
        _metric_col(field_name): ext_score,
    }


def load_t3_frame(path: Path, field_name: str, score_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in read_jsonl(path):
        row = _row_from_json(r, field_name, score_key)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_scale(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {
            "metric": metric,
            "n": int(len(df.dropna(subset=[metric], how="any"))) if metric in df.columns else 0,
            "fit_type": "",
            "beta_d": float("nan"),
            "beta_d2": float("nan"),
            "p_d": float("nan"),
            "p_d2": float("nan"),
            "d_star": float("nan"),
            "d2_significant_p05": False,
            "error": "missing_statsmodels",
        }

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


def _plot_curves(specs: list[tuple[str, Path]], field_name: str, score_key: str, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        import statsmodels.formula.api as smf
    except ImportError as e:
        print(f"[external_eval_scale_regression] skip plot for {field_name}: {e}")
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    d_grid = np.linspace(0.26, 0.84, 200)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
    metric_col = _metric_col(field_name)
    plotted = 0
    for idx, (label, path) in enumerate(specs):
        if not path.exists():
            continue
        df = load_t3_frame(path, field_name, score_key)
        df = df.dropna(subset=[metric_col, "d_H", "genre"])
        if len(df) < 30:
            continue
        df = df.rename(columns={metric_col: "y", "d_H": "d"})
        fit = smf.ols("y ~ d + I(d ** 2) + C(genre)", data=df).fit()
        genres = df["genre"].astype("category").cat.categories.tolist()
        preds = []
        for dv in d_grid:
            pv = []
            for g in genres:
                row = pd.DataFrame({"d": [dv], "genre": pd.Categorical([g], categories=genres)})
                pv.append(float(fit.predict(row).iloc[0]))
            preds.append(float(np.mean(pv)))
        color = palette[idx % len(palette)]
        ax.plot(d_grid, preds, color=color, lw=2.0, label=label)
        plotted += 1

    if plotted == 0:
        print(f"[external_eval_scale_regression] skip plot for {field_name}: no usable rows")
        plt.close(fig)
        return False

    ax.set_xlabel(r"$d_H$ (T3, discrete main)")
    ax.set_ylabel(f"mean predicted {_slug(field_name)} score")
    ax.set_title(f"Scale dependence ({field_name}): external score vs. $d_H$")
    ax.set_xlim(0.26, 0.84)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _resolve_specs(paths: list[Path] | None) -> list[tuple[str, Path]]:
    if not paths:
        return DEFAULT_SPECS
    return [(p.stem, p) for p in paths]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field-name", default="creval")
    p.add_argument("--score-key", default="score")
    p.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=None,
        help="optional metrics jsonl list; default uses Qwen3 4B/8B/14B + main",
    )
    args = p.parse_args()

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

    field_slug = _slug(args.field_name)
    metric_col = _metric_col(args.field_name)
    out_detail = PROJECT_ROOT / "results" / "tables" / f"{field_slug}_scale_regression.csv"
    out_compare = PROJECT_ROOT / "results" / "tables" / f"{field_slug}_auto_compare.csv"
    out_plot = PROJECT_ROOT / "results" / "figures" / f"scale_inverted_u_{field_slug}.png"

    specs = _resolve_specs(args.inputs)
    detail_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []

    for label, path in specs:
        if not path.exists():
            detail_rows.append(
                {
                    "generator_label": label,
                    "metrics_file": path.name,
                    "generator_model": "",
                    "outcome": metric_col,
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
                    "generator_model": "",
                    "n_T3": 0,
                    "mean_creativity_auto_T3": float("nan"),
                    f"mean_{metric_col}_T3": float("nan"),
                    "beta_d2_creativity_auto": float("nan"),
                    "p_d2_creativity_auto": float("nan"),
                    "sig_p05_creativity_auto": False,
                    f"beta_d2_{metric_col}": float("nan"),
                    f"p_d2_{metric_col}": float("nan"),
                    f"sig_p05_{metric_col}": False,
                    "same_sign_beta_d2": "",
                    "both_sig_p05_d2": False,
                    "error": "file_not_found",
                }
            )
            continue

        df = load_t3_frame(path, args.field_name, args.score_key)
        ext_fit = _fit_scale(df, metric_col)
        auto_fit = _fit_scale(df, "creativity_auto")

        gen_models = df["model"].dropna().unique().tolist() if not df.empty else []
        gen_model = str(gen_models[0]) if len(gen_models) == 1 else "|".join(sorted(map(str, gen_models)))
        detail_rows.append(
            {
                "generator_label": label,
                "metrics_file": path.name,
                "generator_model": gen_model,
                "outcome": metric_col,
                "n": ext_fit["n"],
                "fit_type": ext_fit["fit_type"],
                "beta_d": ext_fit["beta_d"],
                "beta_d2": ext_fit["beta_d2"],
                "p_d": ext_fit["p_d"],
                "p_d2": ext_fit["p_d2"],
                "d_star": ext_fit["d_star"],
                "d2_significant_p05": ext_fit["d2_significant_p05"],
                "error": ext_fit["error"],
            }
        )

        b2a, b2e = auto_fit["beta_d2"], ext_fit["beta_d2"]
        same = ""
        if pd.notna(b2a) and pd.notna(b2e) and abs(b2a) > 1e-15 and abs(b2e) > 1e-15:
            same = "yes" if (b2a > 0) == (b2e > 0) else "no"
        elif pd.notna(b2a) and pd.notna(b2e):
            same = "yes" if abs(b2a) < 1e-15 and abs(b2e) < 1e-15 else "mixed"

        compare_rows.append(
            {
                "generator_label": label,
                "metrics_file": path.name,
                "generator_model": gen_model,
                "n_T3": int(len(df)),
                "mean_creativity_auto_T3": float(df["creativity_auto"].dropna().mean()) if "creativity_auto" in df else float("nan"),
                f"mean_{metric_col}_T3": float(df[metric_col].dropna().mean()) if metric_col in df else float("nan"),
                "beta_d2_creativity_auto": auto_fit["beta_d2"],
                "p_d2_creativity_auto": auto_fit["p_d2"],
                "sig_p05_creativity_auto": auto_fit["d2_significant_p05"],
                f"beta_d2_{metric_col}": ext_fit["beta_d2"],
                f"p_d2_{metric_col}": ext_fit["p_d2"],
                f"sig_p05_{metric_col}": ext_fit["d2_significant_p05"],
                "same_sign_beta_d2": same,
                "both_sig_p05_d2": bool(auto_fit["d2_significant_p05"] and ext_fit["d2_significant_p05"]),
                "error": ext_fit["error"] or auto_fit["error"],
            }
        )

    out_detail.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(out_detail, index=False, encoding="utf-8")
    pd.DataFrame(compare_rows).to_csv(out_compare, index=False, encoding="utf-8")
    wrote_plot = _plot_curves(specs, args.field_name, args.score_key, out_plot)
    print(f"[external_eval_scale_regression] wrote {out_detail}")
    print(f"[external_eval_scale_regression] wrote {out_compare}")
    if wrote_plot:
        print(f"[external_eval_scale_regression] wrote {out_plot}")
    print(pd.DataFrame(compare_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())