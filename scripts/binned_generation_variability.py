"""
Exploratory: bin sentiment_shift and kendall_tau by conflict d_H (equal-width bins,
matching src.analyze.compute_binned_summary defaults). Writes CSV + PNG for the report.

Usage (from repo root):
  python scripts/binned_generation_variability.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FLAT = ROOT / "results" / "flat.csv"
OUT_CSV = ROOT / "results" / "tables" / "binned_generation_variability.csv"
OUT_PNG = ROOT / "results" / "figures" / "generation_variability_sentiment_kendall.png"


def _binned(df: pd.DataFrame, metric: str, *, n_bins: int = 8) -> pd.DataFrame:
    d = df.dropna(subset=["d_H", metric]).copy()
    if d.empty:
        return pd.DataFrame()
    labels = list(range(n_bins))
    d["bin_idx"] = pd.cut(d["d_H"], bins=n_bins, labels=labels, include_lowest=True, duplicates="drop")
    g = (
        d.groupby("bin_idx", observed=True)
        .agg(
            n=(metric, "count"),
            mean=(metric, "mean"),
            std=(metric, "std"),
            d_bin_left=("d_H", "min"),
            d_bin_right=("d_H", "max"),
        )
        .reset_index()
    )
    g["d_bin_center"] = 0.5 * (g["d_bin_left"] + g["d_bin_right"])
    g["metric"] = metric
    return g[
        ["metric", "bin_idx", "d_bin_left", "d_bin_right", "d_bin_center", "n", "mean", "std"]
    ]


def main() -> None:
    df = pd.read_csv(FLAT)
    for col in ("sentiment_shift", "kendall_tau"):
        if col not in df.columns:
            raise SystemExit(f"missing column {col} in {FLAT}")

    parts = []
    for col in ("sentiment_shift", "kendall_tau"):
        sub = _binned(df, col, n_bins=8)
        if not sub.empty:
            parts.append(sub)
    if not parts:
        raise SystemExit("no rows with finite d_H and variability metrics")

    out = pd.concat(parts, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    for ax, metric, title in zip(
        axes,
        ("sentiment_shift", "kendall_tau"),
        ("Mean sentiment delta [0,2] (absolute)", "Mean Kendall tau (word-order similarity)"),
    ):
        sub = out.loc[out["metric"] == metric]
        x = sub["d_bin_center"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        ax.plot(x, y, "o-", color="steelblue", lw=2, ms=5)
        ax.set_ylabel("Bin mean")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(r"Conflict $d_H$ (bin centre)")
    fig.suptitle("Generation variability vs conflict intensity (pooled flat.csv, 8 equal-width $d_H$ bins)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
