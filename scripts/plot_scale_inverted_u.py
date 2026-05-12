"""Plot T3-only quadratic fits of creativity_auto vs d_H for Qwen3 4B/8B/14B.

Writes ``results/figures/scale_inverted_u.png``: three predicted curves (genre-averaged
OLS: $C \\sim d + d^2 + C(\\mathrm{genre})$), vertical markers at fitted vertices for 4B/8B,
and shaded ``hybrid band'' intervals on the $d_H$ axis (paper constants).

Run from repo root::

    python scripts/plot_scale_inverted_u.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "figures" / "scale_inverted_u.png"
FILES = [
    ("Qwen3-4B", ROOT / "data" / "generated" / "main_qwen3_4b_metrics.jsonl", "gen_qwen3_4b"),
    ("Qwen3-8B", ROOT / "data" / "generated" / "main_qwen3_8b_metrics.jsonl", "gen_qwen3_8b"),
    ("Qwen3-14B", ROOT / "data" / "generated" / "main_qwen3_14b_metrics.jsonl", "gen_qwen3_14b"),
]


def _load_t3(path: Path, model: str) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("condition") != "T3" or r.get("model") != model:
                continue
            m = r.get("metrics") or {}
            if not m.get("ok") or m.get("creativity_auto") is None:
                continue
            rows.append(
                {
                    "y": float(m["creativity_auto"]),
                    "d": float(r["d_H"]),
                    "genre": str(r.get("genre", "unknown")),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    d_grid = np.linspace(0.27, 0.82, 200)
    styles = [("-", "#1f77b4"), ("-", "#ff7f0e"), ("--", "#2ca02c")]
    vertices: dict[str, float] = {}
    for (label, path, model), (ls, color) in zip(FILES, styles):
        if not path.exists():
            print(f"[plot_scale_inverted_u] skip missing {path}", file=sys.stderr)
            continue
        df = _load_t3(path, model)
        if len(df) < 30:
            continue
        fit = smf.ols("y ~ d + I(d ** 2) + C(genre)", data=df).fit()
        b1 = float(fit.params["d"])
        b2 = float(fit.params["I(d ** 2)"])
        vertices[label] = float(-b1 / (2.0 * b2)) if abs(b2) > 1e-12 else float("nan")
        genres = df["genre"].astype("category").cat.categories.tolist()
        preds = []
        for dv in d_grid:
            pv = []
            for g in genres:
                row = pd.DataFrame({"d": [dv], "genre": pd.Categorical([g], categories=genres)})
                pv.append(float(fit.predict(row).iloc[0]))
            preds.append(float(np.mean(pv)))
        ax.plot(d_grid, preds, ls=ls, color=color, lw=2.0, label=label)

    # Shaded hybrid bands (paper constants; 4B / 8B only)
    ax.axvspan(0.30, 0.63, color="#1f77b4", alpha=0.12, label="Qwen3-4B band (approx.)")
    ax.axvspan(0.32, 0.74, color="#ff7f0e", alpha=0.12, label="Qwen3-8B band (approx.)")

    for lab, x in vertices.items():
        if not np.isfinite(x):
            continue
        if lab.startswith("Qwen3-4B"):
            ax.axvline(x, color="#1f77b4", ls=":", lw=1.5, alpha=0.9)
        if lab.startswith("Qwen3-8B"):
            ax.axvline(x, color="#ff7f0e", ls=":", lw=1.5, alpha=0.9)

    ax.set_xlabel(r"$d_H$ (T3, discrete main)")
    ax.set_ylabel(r"mean predicted $C_{\mathrm{auto}}$ (genre-averaged OLS)")
    ax.set_title("Scale dependence: open-source Qwen3 arms (T3 only)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0.26, 0.84)
    ax.set_ylim(0.20, 0.36)
    ax.grid(True, alpha=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"[plot_scale_inverted_u] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
