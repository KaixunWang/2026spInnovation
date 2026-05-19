"""Verify key report statistics against current metrics files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analyze import load_flat

ROOT = Path(__file__).resolve().parents[1]


def corr_table(df: pd.DataFrame, label: str) -> None:
    cols = [
        "creativity_auto",
        "creativity_judge",
        "novelty_judge",
        "novelty_auto_combined",
        "value_auto",
        "coherence_judge",
        "fidelity_judge",
    ]
    avail = [c for c in cols if c in df.columns]
    d = df[avail].apply(pd.to_numeric, errors="coerce")
    print(f"\n=== {label} (n={len(df)}) ===")
    if "creativity_auto" in avail and "creativity_judge" in avail:
        j = d.dropna(subset=["creativity_auto", "creativity_judge"])
        print(f"  r(C_auto, C_judge) = {j['creativity_auto'].corr(j['creativity_judge']):.4f} (n={len(j)})")
    if "creativity_auto" in avail and "novelty_judge" in avail:
        j = d.dropna(subset=["creativity_auto", "novelty_judge"])
        print(f"  r(C_auto, novelty_judge) = {j['creativity_auto'].corr(j['novelty_judge']):.4f}")
    if "novelty_auto_combined" in avail and "novelty_judge" in avail:
        j = d.dropna(subset=["novelty_auto_combined", "novelty_judge"])
        print(f"  r(novelty_auto, novelty_judge) = {j['novelty_auto_combined'].corr(j['novelty_judge']):.4f}")
    t3 = df[df["condition"] == "T3"] if "condition" in df.columns else df
    if len(t3) and "creativity_auto" in avail and "creativity_judge" in avail:
        j = t3.dropna(subset=["creativity_auto", "creativity_judge"])
        print(f"  T3 r(C_auto, C_judge) = {j['creativity_auto'].corr(j['creativity_judge']):.4f} (n={len(j)})")


def main() -> None:
    main_m = load_flat(ROOT / "data/generated/main_metrics.jsonl")
    corr_table(main_m, "GPT main_metrics (all conditions)")

    frames = [
        load_flat(ROOT / f"data/generated/main_qwen3_{t}_metrics.jsonl")
        for t in ("4b", "8b", "14b")
    ]
    qwen = pd.concat(frames, ignore_index=True)
    corr_table(qwen, "Qwen3 pooled (all conditions)")
    for t, df in zip(("4b", "8b", "14b"), frames):
        corr_table(df, f"Qwen3-{t.upper()}")


if __name__ == "__main__":
    main()
