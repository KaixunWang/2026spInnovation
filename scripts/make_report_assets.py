"""Create tables and figures for the cross-personality rewriting report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
GENERATED = ROOT / "data" / "generated"
FLAT = RESULTS / "flat.csv"
DIAG = RESULTS / "regression_diagnostics.json"


METRICS = [
    "nli_entailment",
    "novelty_auto_combined",
    "coherence_auto",
    "value_auto",
    "creativity_auto",
    "sentiment_shift",
    "kendall_tau",
    "levenshtein",
]


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def p_fmt(x: float) -> str:
    if pd.isna(x):
        return "--"
    x = float(x)
    if x < 1e-4:
        return "$<10^{-4}$"
    return f"{x:.4f}"


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, align: str | None = None) -> None:
    if align is None:
        align = "l" + "c" * (len(df.columns) - 1)
    body = df.to_latex(
        index=False,
        escape=False,
        column_format=align,
    )
    body = body.replace("\\toprule", "\\toprule")
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            body.strip(),
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def table_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition", sort=True):
        rows.append(
            {
                "Condition": cond,
                "n": len(g),
                "NLI": fmt(g["nli_entailment"].mean()),
                "Novelty": fmt(g["novelty_auto_combined"].mean()),
                "Coherence": fmt(g["coherence_auto"].mean()),
                "Value": fmt(g["value_auto"].mean()),
                "Creativity": fmt(g["creativity_auto"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "condition_summary.csv", index=False)
    write_latex_table(
        out,
        TABLES / "condition_summary.tex",
        "Mean automatic metrics by experimental arm. Creativity is the product of genre-normalised novelty and value; value combines NLI entailment and perplexity-derived coherence.",
        "tab:condition-summary",
    )
    return out


def table_genre_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for genre, g in df.groupby("genre", sort=True):
        rows.append(
            {
                "Genre": genre,
                "n": len(g),
                "NLI": fmt(g["nli_entailment"].mean()),
                "Novelty": fmt(g["novelty_auto_combined"].mean()),
                "Value": fmt(g["value_auto"].mean()),
                "Creativity": fmt(g["creativity_auto"].mean()),
                "PPL": fmt(g["perplexity"].mean(), 1),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "genre_summary.csv", index=False)
    write_latex_table(
        out,
        TABLES / "genre_summary.tex",
        "Mean metrics by genre, pooling all arms and generators. Genre-normalised novelty reduces but does not remove genre-level differences.",
        "tab:genre-summary",
    )
    return out


def table_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["genre", "condition"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"genre": "Genre"})
    )
    counts.to_csv(TABLES / "experiment_counts.csv", index=False)
    write_latex_table(
        counts,
        TABLES / "experiment_counts.tex",
        "Record counts by genre and experimental arm in the final analysed table.",
        "tab:experiment-counts",
    )
    return counts


def table_causal() -> pd.DataFrame | None:
    path = TABLES / "causal_creativity_auto.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "Bucket": raw["bucket"],
            "$n_{T1}$": raw["n_T1"].fillna(0).astype(int),
            "$n_{T2}$": raw["n_T2"].fillna(0).astype(int),
            "$n_{T3}$": raw["n_T3"].fillna(0).astype(int),
            "Mean $T1$": [fmt(x) for x in raw["mean_T1"]],
            "Mean $T2$": [fmt(x) for x in raw["mean_T2"]],
            "Mean $T3$": [fmt(x) for x in raw["mean_T3"]],
            "$T3-T1$": [fmt(x, 4) for x in raw["delta_vs_T1"]],
            "$T3-T2$": [fmt(x, 4) for x in raw["placebo_gap_vs_T2"]],
        }
    )
    write_latex_table(
        out,
        TABLES / "causal_creativity_auto.tex",
        "Causal contrasts for automatic creativity by conflict bucket. $T3-T1$ estimates directed-conflict lift over same-personality rewriting; $T3-T2$ is the placebo-adjusted gap.",
        "tab:causal-creativity",
        align="lcccccccc",
    )
    return out


def table_regression(diag: dict) -> pd.DataFrame:
    rows = []
    names = [
        ("main_creativity_auto", "Creativity"),
        ("main_value_auto", "Value"),
        ("main_novelty_auto_combined", "Novelty"),
    ]
    for key, label in names:
        item = diag[key]
        rows.append(
            {
                "Metric": label,
                "Fit": item.get("fit_type", "--"),
                "$n$": item.get("n", "--"),
                "$\\beta_d$": fmt(item["coefs"].get("d"), 4),
                "$p_d$": p_fmt(item["pvals"].get("d")),
                "$\\beta_{d^2}$": fmt(item["coefs"].get("d2"), 4),
                "$p_{d^2}$": p_fmt(item["pvals"].get("d2")),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "regression_key_terms.csv", index=False)
    write_latex_table(
        out,
        TABLES / "regression_key_terms.tex",
        "Key scalar-conflict regression terms. A negative $d^2$ coefficient supports an inverted-U pattern; a positive coefficient indicates monotonic or accelerating novelty.",
        "tab:regression-key",
        align="lcccccc",
    )
    return out


def table_direction(diag: dict) -> pd.DataFrame:
    item = diag["dir_creativity_auto"]
    rows = []
    for term, desc in [
        ("dS", "Cognitive shift $\\Delta S$"),
        ("dR", "Risk/register shift $\\Delta R$"),
        ("dS2", "Quadratic $\\Delta S^2$"),
        ("dR2", "Quadratic $\\Delta R^2$"),
        ("dSdR", "Interaction $\\Delta S\\Delta R$"),
        ("absdelta", "Magnitude $|\\Delta|$"),
    ]:
        rows.append(
            {
                "Term": desc,
                "Coef.": fmt(item["coefs"].get(term), 4),
                "$p$": p_fmt(item["pvals"].get(term)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "directional_creativity_terms.csv", index=False)
    write_latex_table(
        out,
        TABLES / "directional_creativity_terms.tex",
        "Directional regression terms for creativity. Positive linear terms indicate directions in persona space associated with higher creativity after controlling genre and arm.",
        "tab:direction-terms",
        align="lcc",
    )
    return out


def plot_condition_bar(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="condition", y="creativity_auto", hue="genre", errorbar=("ci", 95))
    plt.ylabel("Creativity (Novelty x Value)")
    plt.xlabel("Experimental arm")
    plt.title("Creativity by arm and genre")
    plt.legend(title="Genre", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "creativity_by_condition_genre.png", dpi=220)
    plt.close()


def plot_heatmap(df: pd.DataFrame) -> None:
    mat = df.pivot_table(index="genre", columns="condition", values="creativity_auto", aggfunc="mean")
    plt.figure(figsize=(6, 3.8))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Mean creativity by genre and arm")
    plt.tight_layout()
    plt.savefig(FIGURES / "creativity_heatmap_genre_condition.png", dpi=220)
    plt.close()


def plot_conflict_curves(df: pd.DataFrame) -> None:
    sub = df[df["condition"].isin(["T2", "T3"])].copy()
    sub = sub[np.isfinite(sub["d_H"])]
    sub["d_bin"] = pd.cut(sub["d_H"], bins=np.linspace(0, 1.01, 8), include_lowest=True)
    agg = (
        sub.groupby(["condition", "d_bin"], observed=True)
        .agg(
            d=("d_H", "mean"),
            creativity=("creativity_auto", "mean"),
            novelty=("novelty_auto_combined", "mean"),
            value=("value_auto", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["creativity", "novelty", "value"],
        ["Creativity", "Novelty", "Value"],
    ):
        sns.lineplot(data=agg, x="d", y=metric, hue="condition", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Conflict intensity $d_H$")
        ax.set_ylabel(title)
    plt.tight_layout()
    plt.savefig(FIGURES / "conflict_curves_t2_t3.png", dpi=220)
    plt.close()


def plot_direction(df: pd.DataFrame) -> None:
    sub = df[df["condition"] == "T3"].copy()
    sub = sub[np.isfinite(sub["dS"]) & np.isfinite(sub["dR"])]
    if sub.empty:
        return
    mat = sub.pivot_table(index="dR", columns="dS", values="creativity_auto", aggfunc="mean")
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(mat.sort_index(ascending=False), annot=True, fmt=".3f", cmap="mako")
    plt.xlabel("$\\Delta S$ (toward rational/S2)")
    plt.ylabel("$\\Delta R$ (toward adventurous register)")
    plt.title("T3 creativity by conflict direction")
    plt.tight_layout()
    plt.savefig(FIGURES / "directional_creativity_heatmap.png", dpi=220)
    plt.close()


def plot_metric_corr(df: pd.DataFrame) -> None:
    corr = df[METRICS].corr(numeric_only=True)
    plt.figure(figsize=(7, 5.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Metric correlation matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "metric_correlation_heatmap.png", dpi=220)
    plt.close()


def _t3_d_h_from_main_jsonl(path: Path) -> np.ndarray:
    """T3 rows with finite d_H from a main*.jsonl generation file."""
    if not path.exists():
        return np.array([], dtype=float)
    vals: list[float] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("condition") != "T3":
                continue
            d = r.get("d_H")
            if d is None:
                continue
            vals.append(float(d))
    return np.asarray(vals, dtype=float)


def plot_d_distribution_discrete_vs_continuous() -> None:
    """Match ``run_experiment._write_d_distribution_comparison`` histogram style."""
    old_d = _t3_d_h_from_main_jsonl(GENERATED / "main.jsonl")
    new_d = _t3_d_h_from_main_jsonl(GENERATED / "main_continuous.jsonl")
    if len(old_d) == 0 or len(new_d) == 0:
        print("[plot_d_distribution_discrete_vs_continuous] skip: empty T3 d_H in main or main_continuous")
        return
    bins = min(20, max(8, int(np.sqrt(len(old_d) + len(new_d)))))
    plt.figure(figsize=(6.4, 4.4))
    plt.hist(old_d, bins=bins, alpha=0.45, density=True, label="T3 discrete (persona corners)")
    plt.hist(new_d, bins=bins, alpha=0.45, density=True, label="T3 continuous (unit disk)")
    plt.xlabel("$d_H$")
    plt.ylabel("density")
    plt.title("T3 $d_H$ distribution: discrete vs continuous targets")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out = FIGURES / "d_distribution_discrete_vs_continuous.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot_d_distribution_discrete_vs_continuous] wrote {out}")


def plot_novelty_value(df: pd.DataFrame) -> None:
    sample = df.sample(min(len(df), 2500), random_state=42)
    plt.figure(figsize=(6, 4.6))
    sns.scatterplot(
        data=sample,
        x="novelty_auto_combined",
        y="value_auto",
        hue="condition",
        alpha=0.45,
        s=18,
        linewidth=0,
    )
    plt.xlabel("Genre-normalised novelty")
    plt.ylabel("Value")
    plt.title("Novelty--value trade-off")
    plt.legend(title="Arm", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "novelty_value_scatter.png", dpi=220)
    plt.close()


def build_all() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", font_scale=0.9)
    df = pd.read_csv(FLAT)
    with DIAG.open("r", encoding="utf-8") as fh:
        diag = json.load(fh)

    table_counts(df)
    table_condition_summary(df)
    table_genre_summary(df)
    table_causal()
    table_regression(diag)
    table_direction(diag)

    plot_condition_bar(df)
    plot_heatmap(df)
    plot_d_distribution_discrete_vs_continuous()
    plot_conflict_curves(df)
    plot_direction(df)
    plot_metric_corr(df)
    plot_novelty_value(df)

    integrity = {
        "n_rows": int(len(df)),
        "n_sources": int(df["source_id"].nunique()),
        "genres": sorted(df["genre"].unique().tolist()),
        "conditions": sorted(df["condition"].unique().tolist()),
        "models": sorted(df["model"].unique().tolist()),
        "nli_null": int(df["nli_entailment"].isna().sum()),
        "creativity_null": int(df["creativity_auto"].isna().sum()),
    }
    (TABLES / "data_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        choices=["d-distribution"],
        help="Build a single asset and exit (otherwise run full build_all).",
    )
    ns = ap.parse_args()
    if ns.only == "d-distribution":
        ensure_dirs()
        sns.set_theme(style="whitegrid", font_scale=0.9)
        plot_d_distribution_discrete_vs_continuous()
    else:
        build_all()
