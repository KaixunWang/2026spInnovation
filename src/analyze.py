"""Statistical analyses and figures.

Loads the JSONL outputs of `run_experiment metrics` (and optionally
`judge`), assembles a flat pandas DataFrame, and runs:

  - Main regression (H1, H4, H7): metric ~ d + d² + genre + ... + (1|source)
  - Directional regression (H3):  metric ~ ΔS + ΔR + ΔS² + ΔR² + ΔS·ΔR + |Δ|
  - Causal contrasts:              Δ_creativity(d), placebo gap
  - Mechanism contrast (H6):       M1 vs {M2, M3} in the high-d subset
  - Multi-hop drift / attractor analysis

Also produces paper-ready figures under results/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_loader import PROJECT_ROOT, load_experiment_config
from .io_utils import GENERATED_DIR, RESULTS_DIR, read_jsonl
from .progress_util import iter_progress
from .metrics.value import fuse_fidelity_judge_nli


FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


# ---------------------------------------------------------------------------
# Load & flatten
# ---------------------------------------------------------------------------


def load_flat(path: Path) -> pd.DataFrame:
    """Flatten a `<name>_metrics.jsonl` (or `_judged.jsonl`) into a DataFrame."""
    rows = list(read_jsonl(path))
    records: list[dict[str, Any]] = []
    for r in rows:
        flat = {
            "source_id": r.get("source_id"),
            "genre": r.get("genre"),
            "condition": r.get("condition", r.get("mode")),
            "target_persona": r.get("target_persona"),
            "model": r.get("model"),
            "mode": r.get("mode"),
            "prompt_variant": r.get("prompt_variant"),
            "repeat_idx": r.get("repeat_idx"),
            "d_H": r.get("d_H"),
            "delta_H": r.get("delta_H"),
            "text_len": len((r.get("text") or "").split()),
            "hop_index": r.get("hop_index"),
            "d_total": r.get("d_total"),
            "d_step": r.get("d_step"),
        }
        if flat["delta_H"] and isinstance(flat["delta_H"], list):
            flat["dS"] = flat["delta_H"][0]
            flat["dR"] = flat["delta_H"][1] if len(flat["delta_H"]) > 1 else 0.0
        else:
            flat["dS"] = 0.0
            flat["dR"] = 0.0
        m = r.get("metrics") or {}
        if m.get("ok"):
            flat["nli_entailment"] = m.get("nli_entailment")
            flat["creativity_auto"] = m.get("creativity_auto")
            flat["creativity_auto_geom"] = m.get("creativity_auto_geom")
            flat["value_auto"] = m.get("value_auto")
            flat["value_auto_geom"] = m.get("value_auto_geom")
            nov = m.get("novelty") or {}
            flat["novelty_auto_combined"] = nov.get("auto_combined")
            flat["distinct2_rel"] = nov.get("distinct2_rel")
            flat["novel_ngram_rel"] = nov.get("novel_ngram_rel")
            flat["embedding_distance_rel"] = nov.get("embedding_distance_rel")
            flat["coherence_auto"] = m.get("coherence_auto")
            flat["perplexity"] = m.get("perplexity")
            flat["sentiment_shift"] = m.get("sentiment_shift")
            flat["kendall_tau"] = m.get("sentence_kendall_tau")
            flat["levenshtein"] = m.get("normalised_levenshtein")
        j = r.get("judge") or {}
        if j.get("ok"):
            flat["novelty_judge"] = j.get("novelty_judge")
            flat["coherence_judge"] = j.get("coherence_judge")
            flat["fidelity_judge"] = j.get("fidelity_judge")
            # judge-based composites
            nj = j.get("novelty_judge")
            cj = j.get("coherence_judge")
            fj = j.get("fidelity_judge")
            if nj is not None and cj is not None and fj is not None:
                fidelity_fused = fuse_fidelity_judge_nli(fj, flat.get("nli_entailment"), w_judge=0.5)
                value_j = float((max(0.0, fidelity_fused) + max(0.0, cj)) / 2.0)
                value_j_geom = float(np.sqrt(max(0.0, fidelity_fused) * max(0.0, cj)))
                flat["value_judge"] = value_j
                flat["value_judge_geom"] = value_j_geom
                flat["creativity_judge"] = nj * value_j
                flat["creativity_judge_geom"] = nj * value_j_geom
                flat["fidelity_fused"] = fidelity_fused
        records.append(flat)
    return pd.DataFrame(records)


def load_all(root: Path = GENERATED_DIR) -> pd.DataFrame:
    # Only *_metrics.jsonl: raw *_judged.jsonl duplicates the same generations without
    # automatic metrics. After judge, run ``python -m scripts.merge_judge_into_metrics``
    # to copy ``judge`` into the metrics rows (see script docstring).
    paths = sorted(root.glob("*_metrics.jsonl"))
    if not paths:
        return pd.DataFrame()
    frames = [load_flat(p).assign(source_file=p.stem) for p in paths]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def main_regression(df: pd.DataFrame, metric: str, *, label: str) -> dict[str, Any]:
    """Mixed-effects (if statsmodels available) else OLS on metric ~ d + d² + genre + condition.

    Returns dict of fit diagnostics.
    """
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[metric, "d_H"]).copy()
    if d.empty:
        return {"label": label, "error": "no data"}
    if float(d[metric].std(ddof=0)) < 1e-12:
        return {
            "label": label,
            "metric": metric,
            "n": int(len(d)),
            "skipped_reason": "near-zero variance metric",
        }
    d["d"] = d["d_H"].astype(float)
    d["d2"] = d["d"] ** 2
    d["genre"] = d["genre"].astype("category")
    d["condition"] = d["condition"].astype("category")
    formula = f"{metric} ~ d + d2 + C(genre) + C(condition)"
    fit_type = "ols"
    coefs: dict[str, float] = {}
    pvals: dict[str, float] = {}
    model = smf.mixedlm(formula, data=d, groups=d["source_id"])
    mixed_ok = False
    for meth in ("lbfgs", "powell", "bfgs"):
        for reml in (True, False):
            try:
                fit = model.fit(method=meth, disp=False, reml=reml)
                fit_type = f"mixedlm_{meth}_reml{int(reml)}"
                coefs = fit.params.to_dict()
                pvals = fit.pvalues.to_dict()
                mixed_ok = True
                break
            except Exception:
                continue
        if mixed_ok:
            break
    if not mixed_ok:
        fit = smf.ols(formula, data=d).fit()
        coefs = fit.params.to_dict()
        pvals = fit.pvalues.to_dict()
    return {
        "label": label,
        "metric": metric,
        "fit_type": fit_type,
        "n": int(len(d)),
        "coefs": {k: float(v) for k, v in coefs.items()},
        "pvals": {k: float(v) for k, v in pvals.items()},
        "d2_coef": float(coefs.get("d2", float("nan"))),
        "d2_p": float(pvals.get("d2", float("nan"))),
    }


def directional_regression(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    """metric ~ dS + dR + dS^2 + dR^2 + dS·dR + |Δ| + C(genre) + C(condition)."""
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[metric, "dS", "dR"]).copy()
    if d.empty:
        return {"error": "no data"}
    if float(d[metric].std(ddof=0)) < 1e-12:
        return {
            "metric": metric,
            "n": int(len(d)),
            "skipped_reason": "near-zero variance metric",
        }
    d["dS2"] = d["dS"] ** 2
    d["dR2"] = d["dR"] ** 2
    d["dSdR"] = d["dS"] * d["dR"]
    d["absdelta"] = np.sqrt(d["dS"] ** 2 + d["dR"] ** 2)
    d["genre"] = d["genre"].astype("category")
    d["condition"] = d["condition"].astype("category")
    formula = f"{metric} ~ dS + dR + dS2 + dR2 + dSdR + absdelta + C(genre) + C(condition)"
    fit_type = "ols"
    model = smf.mixedlm(formula, data=d, groups=d["source_id"])
    fit = None
    for meth in ("lbfgs", "powell", "bfgs"):
        for reml in (True, False):
            try:
                fit = model.fit(method=meth, disp=False, reml=reml)
                fit_type = f"mixedlm_{meth}_reml{int(reml)}"
                break
            except Exception:
                continue
        if fit is not None:
            break
    if fit is None:
        fit = smf.ols(formula, data=d).fit()
        fit_type = "ols"
    return {
        "metric": metric,
        "fit_type": fit_type,
        "n": int(len(d)),
        "coefs": {k: float(v) for k, v in fit.params.to_dict().items()},
        "pvals": {k: float(v) for k, v in fit.pvalues.to_dict().items()},
    }


def causal_contrasts(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Bootstrapped Δ_creativity(d_bucket) = T3 - T1 and placebo gap = T3 - T2."""
    d = df.dropna(subset=[metric, "condition", "d_H"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["bucket"] = pd.qcut(d["d_H"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    out = []
    for b, grp in d.groupby("bucket", observed=True):
        mT3 = grp.loc[grp["condition"] == "T3", metric].mean()
        mT1 = grp.loc[grp["condition"] == "T1", metric].mean()
        mT2 = grp.loc[grp["condition"] == "T2", metric].mean()
        out.append(
            {
                "bucket": str(b),
                "n_T1": int((grp["condition"] == "T1").sum()),
                "n_T2": int((grp["condition"] == "T2").sum()),
                "n_T3": int((grp["condition"] == "T3").sum()),
                "delta_vs_T1": (mT3 - mT1) if pd.notna(mT3) and pd.notna(mT1) else float("nan"),
                "placebo_gap_vs_T2": (mT3 - mT2) if pd.notna(mT3) and pd.notna(mT2) else float("nan"),
                "mean_T1": float(mT1) if pd.notna(mT1) else None,
                "mean_T2": float(mT2) if pd.notna(mT2) else None,
                "mean_T3": float(mT3) if pd.notna(mT3) else None,
            }
        )
    return pd.DataFrame(out)


def mechanism_contrast(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """In the high-d subset, compare M1/M2/M3."""
    d = df.dropna(subset=[metric, "mode", "d_H"]).copy()
    if d.empty:
        return pd.DataFrame()
    threshold = d["d_H"].quantile(0.67)
    hi = d[d["d_H"] >= threshold]
    grp = hi.groupby("mode", observed=True)[metric].agg(["mean", "std", "count"]).reset_index()
    grp["threshold_d"] = float(threshold)
    return grp


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_inverted_u(df: pd.DataFrame, metric: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    d = df.dropna(subset=[metric, "d_H"])
    if d.empty:
        return
    d = d.copy()
    # visualization-only jitter; original d_H remains untouched for analysis/fit
    rng = np.random.default_rng(42)
    x = d["d_H"].values.astype(float)
    x_span = max(float(x.max() - x.min()), 1e-6)
    jitter_sigma = 0.015 * x_span
    d["d_H_jitter"] = x + rng.normal(0.0, jitter_sigma, size=len(d))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for genre, grp in d.groupby("genre"):
        ax.scatter(grp["d_H_jitter"], grp[metric], alpha=0.35, label=genre, s=18)
    # quadratic fit overall, using original (non-jittered) d_H
    y = d[metric].values.astype(float)
    if len(x) >= 4:
        coef = np.polyfit(x, y, 2)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            xs,
            np.polyval(coef, xs),
            "k--",
            lw=2,
            label=f"quad fit on raw d (d2={coef[0]:.2f})",
        )
    # binned mean + 95% CI (overall)
    binned = compute_binned_summary(d, metric, n_bins=8, by_condition=False)
    binned_plot = binned.dropna(subset=["d_bin_center", "mean"])
    if not binned_plot.empty:
        ax.plot(
            binned_plot["d_bin_center"],
            binned_plot["mean"],
            color="#d62728",
            lw=2,
            marker="o",
            ms=4,
            label="binned mean",
            zorder=3,
        )
        has_ci = binned_plot["ci95"].notna()
        if has_ci.any():
            x_ci = binned_plot.loc[has_ci, "d_bin_center"].to_numpy(dtype=float)
            m_ci = binned_plot.loc[has_ci, "mean"].to_numpy(dtype=float)
            ci = binned_plot.loc[has_ci, "ci95"].to_numpy(dtype=float)
            ax.fill_between(x_ci, m_ci - ci, m_ci + ci, color="#d62728", alpha=0.16, label="95% CI")
    ax.set_xlabel("Conflict intensity d (Space-H)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs d — inverted-U test")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pair_heatmap(df: pd.DataFrame, metric: str, path: Path) -> None:
    """4x4 persona-pair heatmap: mean metric value across (src-closest-persona, target_persona)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df.dropna(subset=[metric, "target_persona"])
    # For the x-axis: use `target_persona`; for the y-axis: we need the
    # source's nearest persona in Space-H. Approximate using sign of delta.
    def _src_bucket(row: pd.Series) -> str:
        s = float(row.get("dS") or 0.0)
        r = float(row.get("dR") or 0.0)
        tp = row.get("target_persona") or "_"
        tgt_sign_S = +1 if "rational" in str(tp) else -1
        tgt_sign_R = +1 if "adventurous" in str(tp) else -1
        src_S = tgt_sign_S - (1 if s >= 0 else -1)
        src_R = tgt_sign_R - (1 if r >= 0 else -1)
        return (
            ("rational" if src_S >= 0 else "emotional")
            + "_"
            + ("adventurous" if src_R >= 0 else "conservative")
        )

    if d.empty:
        return
    d = d.assign(src_bucket=d.apply(_src_bucket, axis=1))
    pivot = d.pivot_table(index="src_bucket", columns="target_persona", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(f"Mean {metric} per (source-bucket, target_persona)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_directional_field(df: pd.DataFrame, metric: str, path: Path) -> None:
    """Scatter Δ vectors (dS, dR) coloured by metric value."""
    import matplotlib.pyplot as plt

    d = df.dropna(subset=[metric, "dS", "dR"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(d["dS"], d["dR"], c=d[metric], cmap="viridis", s=18, alpha=0.85)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("ΔS (tgt - src, cognitive axis)")
    ax.set_ylabel("ΔR (tgt - src, register axis)")
    ax.set_title(f"Directional creativity field — {metric}")
    fig.colorbar(sc, ax=ax, label=metric)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_multihop_trajectories(path_in: Path, path_out: Path) -> None:
    import matplotlib.pyplot as plt

    if not path_in.exists():
        return
    rows = list(read_jsonl(path_in))
    # Build DataFrame with one row per hop
    recs = []
    for r in rows:
        v = r.get("cur_vec_H")
        if v is None or len(v) < 2:
            continue
        recs.append(
            {
                "source_id": r.get("source_id"),
                "genre": r.get("genre"),
                "path_id": r.get("path_id"),
                "hop_index": r.get("hop_index"),
                "x": v[0],
                "y": v[1],
            }
        )
    if not recs:
        return
    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(6, 6))
    for (sid, pid), grp in df.groupby(["source_id", "path_id"]):
        grp = grp.sort_values("hop_index")
        ax.plot(grp["x"], grp["y"], marker="o", alpha=0.6, lw=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("S")
    ax.set_ylabel("R")
    ax.set_title("Multi-hop trajectories in Space-H")
    fig.tight_layout()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def winsorize_series(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() < 5:
        return x
    q_lo, q_hi = x.quantile([lo, hi]).tolist()
    return x.clip(lower=q_lo, upper=q_hi)


def novelty_collinearity(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["distinct2_rel", "novel_ngram_rel", "embedding_distance_rel"]
    d = df[cols].dropna()
    if d.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r", "abs_r", "flag_gt_0_8"])
    corr = d.corr()
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = float(corr.loc[a, b])
            out.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "pearson_r": r,
                    "abs_r": abs(r),
                    "flag_gt_0_8": abs(r) > 0.8,
                }
            )
    return pd.DataFrame(out)


def multihop_prediction_check(df: pd.DataFrame) -> pd.DataFrame:
    """Test falsifiable multihop bridge prediction when both datasets exist."""
    if "source_file" not in df.columns or "creativity_auto" not in df.columns:
        return pd.DataFrame()
    mhop = df[df["source_file"].astype(str).str.contains("multihop_metrics", na=False)].copy()
    main_t3 = df[
        df["source_file"].astype(str).str.contains("main", na=False)
        & (df["condition"] == "T3")
        & df["creativity_auto"].notna()
    ].copy()
    if mhop.empty or main_t3.empty:
        return pd.DataFrame()
    # keep only final-hop snapshots when available
    if "hop_index" in mhop.columns:
        max_h = mhop.groupby("source_id")["hop_index"].transform("max")
        mhop = mhop[mhop["hop_index"] == max_h]
    # fallback on d_H if d_total absent
    dcol = "d_total" if "d_total" in mhop.columns and mhop["d_total"].notna().any() else "d_H"
    if dcol not in mhop.columns:
        return pd.DataFrame()
    mhop = mhop.dropna(subset=[dcol, "creativity_auto", "genre"])
    main_t3 = main_t3.dropna(subset=["d_H", "creativity_auto", "genre"])
    if mhop.empty or main_t3.empty:
        return pd.DataFrame()
    mhop["d_bin"] = pd.qcut(mhop[dcol], q=3, labels=["low", "mid", "high"], duplicates="drop")
    main_t3["d_bin"] = pd.qcut(main_t3["d_H"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    left = mhop.groupby(["genre", "d_bin"], observed=True)["creativity_auto"].mean().reset_index(name="mean_multihop")
    right = main_t3.groupby(["genre", "d_bin"], observed=True)["creativity_auto"].mean().reset_index(name="mean_direct_t3")
    out = left.merge(right, on=["genre", "d_bin"], how="inner")
    if out.empty:
        return out
    out["delta_multihop_minus_direct"] = out["mean_multihop"] - out["mean_direct_t3"]
    out["prediction_supported"] = out["delta_multihop_minus_direct"] > 0
    return out


def compute_binned_summary(
    df: pd.DataFrame, metric: str, *, n_bins: int = 8, by_condition: bool = False
) -> pd.DataFrame:
    d = df.dropna(subset=[metric, "d_H"]).copy()
    if d.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "bin_idx",
                "d_bin_left",
                "d_bin_right",
                "d_bin_center",
                "n",
                "mean",
                "std",
                "sem",
                "ci95",
            ]
        )
    labels = list(range(n_bins))
    d["bin_idx"] = pd.cut(d["d_H"], bins=n_bins, labels=labels, include_lowest=True, duplicates="drop")
    group_cols = ["condition", "bin_idx"] if by_condition else ["bin_idx"]
    grouped = (
        d.groupby(group_cols, observed=True)
        .agg(
            n=(metric, "count"),
            mean=(metric, "mean"),
            std=(metric, "std"),
            d_bin_left=("d_H", "min"),
            d_bin_right=("d_H", "max"),
        )
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["ci95"] = 1.96 * grouped["sem"]
    grouped.loc[grouped["n"] <= 1, "ci95"] = np.nan
    grouped["d_bin_center"] = 0.5 * (grouped["d_bin_left"] + grouped["d_bin_right"])
    if "condition" not in grouped.columns:
        grouped["condition"] = "all"
    return grouped[
        [
            "condition",
            "bin_idx",
            "d_bin_left",
            "d_bin_right",
            "d_bin_center",
            "n",
            "mean",
            "std",
            "sem",
            "ci95",
        ]
    ]


def _t3_main_subset(df: pd.DataFrame) -> pd.DataFrame:
    """T3 rows from discrete ``main_metrics`` arm (pre-registered H1 subset)."""
    m = (df.get("source_file") == "main_metrics") & (df.get("condition") == "T3")
    return df.loc[m].copy()


def plot_t3_d_h_preregister(df: pd.DataFrame, *, cfg: dict) -> None:
    """Histogram of d_H under T3 (main arm) with pre-registered quantile markers."""
    import matplotlib.pyplot as plt

    sub = _t3_main_subset(df).dropna(subset=["d_H"])
    if sub.empty or len(sub) < 5:
        print("[analyze] skip T3 d_H histogram (no data)")
        return
    pr = (cfg.get("main") or {}).get("h1_prereg") or {}
    band = pr.get("vertex_d_quantile_band") or [0.55, 0.85]
    q_lo, q_hi = float(band[0]), float(band[1])
    qs = sub["d_H"].quantile([0.03, q_lo, q_hi, 0.97])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(sub["d_H"].astype(float), bins=24, color="steelblue", alpha=0.75, edgecolor="white")
    for q, sty, lab in [
        (qs.loc[0.03], ":", "emp. q03"),
        (qs.loc[q_lo], "--", f"prereg band q{q_lo:.2f}"),
        (qs.loc[q_hi], "--", f"prereg band q{q_hi:.2f}"),
        (qs.loc[0.97], ":", "emp. q97"),
    ]:
        ax.axvline(float(q), color="crimson", ls=sty, lw=1.2, label=lab)
    ax.set_xlabel("d_H (T3, main_metrics)")
    ax.set_ylabel("count")
    ax.set_title("T3 conflict magnitude: empirical d_H coverage (pre-registration markers)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    path = FIGURES_DIR / "t3_d_H_distribution_main.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote {path}")


def write_judge_auto_validity(df: pd.DataFrame) -> None:
    """Pearson correlations: LLM judge sub-scores vs automatic metrics (appendix evidence)."""
    judge_cols = ["novelty_judge", "coherence_judge", "fidelity_judge"]
    auto_cols = [
        "novelty_auto_combined",
        "nli_entailment",
        "coherence_auto",
        "value_auto",
        "creativity_auto",
    ]
    cols = [c for c in judge_cols + auto_cols if c in df.columns]
    if len(cols) < 2:
        print("[analyze] skip judge–auto validity (missing merged judge columns?)")
        return
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method="pearson")
    out = TABLES_DIR / "judge_auto_validity_correlation.csv"
    corr.to_csv(out)
    print(f"[analyze] wrote {out}")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title("Judge vs automatic metrics (Pearson)")
        fig.tight_layout()
        pfig = FIGURES_DIR / "judge_auto_validity_matrix.png"
        fig.savefig(pfig, dpi=150)
        plt.close(fig)
        print(f"[analyze] wrote {pfig}")
    except Exception as e:
        print(f"[analyze] judge validity heatmap skipped: {e}")


def write_genre_t3_stratified(df: pd.DataFrame, metric: str) -> None:
    """T3 main arm: d_H and value_auto by genre (residual confound diagnostics)."""
    sub = _t3_main_subset(df).dropna(subset=["d_H", "genre"])
    if sub.empty:
        return
    rows = []
    for g, grp in sub.groupby("genre"):
        rows.append(
            {
                "genre": g,
                "n": int(len(grp)),
                "d_H_mean": float(grp["d_H"].mean()),
                "d_H_std": float(grp["d_H"].std(ddof=0)),
                "d_H_min": float(grp["d_H"].min()),
                "d_H_max": float(grp["d_H"].max()),
            }
        )
        if "value_auto" in grp.columns:
            rows[-1]["value_auto_mean"] = float(pd.to_numeric(grp["value_auto"], errors="coerce").mean())
            rows[-1]["value_auto_std"] = float(pd.to_numeric(grp["value_auto"], errors="coerce").std(ddof=0))
        if metric in grp.columns:
            rows[-1][f"{metric}_mean"] = float(pd.to_numeric(grp[metric], errors="coerce").mean())
            rows[-1][f"{metric}_std"] = float(pd.to_numeric(grp[metric], errors="coerce").std(ddof=0))
    write_table(pd.DataFrame(rows), TABLES_DIR / "genre_t3_d_value_stratified.csv")


def write_h1_preregistration_check(df: pd.DataFrame, metric: str, diag: dict[str, Any]) -> None:
    """One-row table: T3-only quadratic fit vs pre-registered falsification rules."""
    import statsmodels.formula.api as smf

    cfg = load_experiment_config()
    pr = (cfg.get("main") or {}).get("h1_prereg") or {}
    fals = (pr.get("falsify_if") or {})
    sub = _t3_main_subset(df).dropna(subset=[metric, "d_H", "genre"])
    row: dict[str, Any] = {"metric": metric, "n_T3_main": int(len(sub))}
    if len(sub) < 12:
        row["error"] = "insufficient_T3_rows"
        write_table(pd.DataFrame([row]), TABLES_DIR / "h1_preregistration_check.csv")
        return
    d = sub.copy()
    d["d"] = d["d_H"].astype(float)
    d["d2"] = d["d"] ** 2
    d["genre"] = d["genre"].astype("category")
    fit = smf.ols(f"{metric} ~ d + d2 + C(genre)", data=d).fit()
    b1 = float(fit.params.get("d", float("nan")))
    b2 = float(fit.params.get("d2", float("nan")))
    p2 = float(fit.pvalues.get("d2", float("nan")))
    row["d_coef"] = b1
    row["d2_coef"] = b2
    row["d2_pvalue"] = p2
    if np.isfinite(b2) and abs(b2) > 1e-12:
        d_vertex = float(-b1 / (2.0 * b2))
    else:
        d_vertex = float("nan")
    row["fitted_vertex_d"] = d_vertex
    q_band = pr.get("vertex_d_quantile_band") or [0.55, 0.85]
    q_out = fals.get("fitted_vertex_d_outside_sample_quantiles") or [0.03, 0.97]
    qs = d["d"].quantile([float(q_out[0]), float(q_band[0]), float(q_band[1]), float(q_out[1])])
    row["empirical_d_q03"] = float(qs.iloc[0])
    row["empirical_d_q_lo_band"] = float(qs.iloc[1])
    row["empirical_d_q_hi_band"] = float(qs.iloc[2])
    row["empirical_d_q97"] = float(qs.iloc[3])
    if np.isfinite(d_vertex):
        row["vertex_inside_prereg_band"] = bool(
            float(qs.iloc[1]) <= d_vertex <= float(qs.iloc[2])
        )
        row["vertex_inside_sample_mass"] = bool(
            float(qs.iloc[0]) <= d_vertex <= float(qs.iloc[3])
        )
    else:
        row["vertex_inside_prereg_band"] = False
        row["vertex_inside_sample_mass"] = False
    p_thr = float(fals.get("quadratic_term_p_gt", 0.05))
    row["falsify_quadratic_p_gt_threshold"] = bool(np.isfinite(p2) and p2 > p_thr)
    row["falsify_vertex_outside_03_97"] = bool(
        np.isfinite(d_vertex)
        and (d_vertex < float(qs.iloc[0]) or d_vertex > float(qs.iloc[3]))
    )
    # snapshot from pooled main regression (all conditions) if present
    main_key = f"main_{metric}"
    if main_key in diag and isinstance(diag[main_key], dict):
        row["pooled_main_d2_p"] = diag[main_key].get("d2_p")
    write_table(pd.DataFrame([row]), TABLES_DIR / "h1_preregistration_check.csv")


def run_all(metric: str = "creativity_auto") -> None:
    df = load_all()
    if df.empty:
        print("[analyze] no _metrics.jsonl files in data/generated/. Run metrics first.")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config()
    plot_t3_d_h_preregister(df, cfg=cfg)
    write_judge_auto_validity(df)
    write_genre_t3_stratified(df, metric)

    df.to_csv(RESULTS_DIR / "flat.csv", index=False)
    # Winsorized novelty for robustness diagnostics (5th/95th).
    if "novelty_auto_combined" in df.columns:
        df["novelty_auto_combined_winsor"] = winsorize_series(df["novelty_auto_combined"], 0.05, 0.95)
    if "value_auto" in df.columns and "novelty_auto_combined_winsor" in df.columns:
        df["creativity_auto_winsor"] = df["value_auto"] * df["novelty_auto_combined_winsor"]

    # regressions
    import json

    diag: dict[str, Any] = {}
    metric_list = [
        metric,
        "creativity_auto_geom",
        "creativity_auto_winsor",
        "value_auto",
        "value_auto_geom",
        "novelty_auto_combined",
        "novelty_auto_combined_winsor",
        "coherence_auto",
    ]
    to_run = [m for m in metric_list if m in df.columns]
    for m in iter_progress(to_run, total=len(to_run), desc="[analyze] regressions", unit="metric"):
        diag[f"main_{m}"] = main_regression(df, m, label="main")
        diag[f"dir_{m}"] = directional_regression(df, m)
    write_h1_preregistration_check(df, metric, diag)
    with (RESULTS_DIR / "regression_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2)

    # causal
    write_table(causal_contrasts(df, metric), TABLES_DIR / f"causal_{metric}.csv")
    write_table(mechanism_contrast(df, metric), TABLES_DIR / f"mechanism_{metric}.csv")
    write_table(compute_binned_summary(df, metric, n_bins=8, by_condition=False), TABLES_DIR / "binned_creativity_vs_d.csv")
    write_table(novelty_collinearity(df), TABLES_DIR / "novelty_collinearity.csv")
    write_table(multihop_prediction_check(df), TABLES_DIR / "multihop_prediction_check.csv")

    # plots
    plot_inverted_u(df, metric, FIGURES_DIR / f"inverted_u_{metric}.png")
    plot_inverted_u(df, metric, FIGURES_DIR / f"inverted_u_{metric}_enhanced.png")
    plot_directional_field(df, metric, FIGURES_DIR / f"directional_field_{metric}.png")
    try:
        plot_pair_heatmap(df, metric, FIGURES_DIR / f"pair_heatmap_{metric}.png")
    except Exception as e:
        print(f"[analyze] heatmap skipped: {e}")
    plot_multihop_trajectories(GENERATED_DIR / "multihop.jsonl", FIGURES_DIR / "multihop_trajectories.png")

    print(f"[analyze] wrote outputs to {RESULTS_DIR}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="analyze")
    p.add_argument("--metric", default="creativity_auto")
    args = p.parse_args(argv)
    run_all(metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
