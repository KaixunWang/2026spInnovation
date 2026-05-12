"""T3-only quadratic fits of creativity_judge vs d_H for Qwen3 4B/8B/14B (mirror of ``plot_scale_inverted_u.py``).

Uses the same genre-averaged OLS specification as the automatic plot:
``y ~ d + I(d**2) + C(genre)`` on T3 rows only.  ``creativity_judge`` is assembled from
merged ``judge`` fields (novelty/coherence/fidelity + NLI fuse), matching
``scripts/qwen_judge_scale_regression.py``.

Writes ``results/figures/scale_inverted_u_judge.png``.

Run from repo root::

    python scripts/plot_scale_inverted_u_judge.py
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

from src.metrics.value import fuse_fidelity_judge_nli  # noqa: E402

OUT = ROOT / "results" / "figures" / "scale_inverted_u_judge.png"
FILES = [
    ("Qwen3-4B", ROOT / "data" / "generated" / "main_qwen3_4b_metrics.jsonl", "gen_qwen3_4b"),
    ("Qwen3-8B", ROOT / "data" / "generated" / "main_qwen3_8b_metrics.jsonl", "gen_qwen3_8b"),
    ("Qwen3-14B", ROOT / "data" / "generated" / "main_qwen3_14b_metrics.jsonl", "gen_qwen3_14b"),
]


def _creativity_judge_row(r: dict) -> float | None:
    m = r.get("metrics") or {}
    j = r.get("judge") or {}
    if not m.get("ok") or not j.get("ok"):
        return None
    nj, cj, fj = j.get("novelty_judge"), j.get("coherence_judge"), j.get("fidelity_judge")
    if nj is None or cj is None or fj is None:
        return None
    ff = fuse_fidelity_judge_nli(fj, m.get("nli_entailment"), w_judge=0.5)
    value_j = float((max(0.0, ff) + max(0.0, float(cj))) / 2.0)
    return float(nj) * value_j


def _load_t3_judge(path: Path, model: str) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("condition") != "T3" or r.get("model") != model:
                continue
            y = _creativity_judge_row(r)
            if y is None:
                continue
            rows.append(
                {
                    "y": y,
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
    all_preds: list[np.ndarray] = []
    for (label, path, model), (ls, color) in zip(FILES, styles):
        if not path.exists():
            print(f"[plot_scale_inverted_u_judge] skip missing {path}", file=sys.stderr)
            continue
        df = _load_t3_judge(path, model)
        if len(df) < 30:
            print(f"[plot_scale_inverted_u_judge] skip {label}: n={len(df)}", file=sys.stderr)
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
        parr = np.array(preds, dtype=float)
        all_preds.append(parr)
        ax.plot(d_grid, parr, ls=ls, color=color, lw=2.0, label=label)

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
    ax.set_ylabel(r"mean predicted $C_{\mathrm{judge}}$ (genre-averaged OLS)")
    ax.set_title("Scale dependence (judge): open-source Qwen3 arms (T3 only)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0.26, 0.84)
    if all_preds:
        lo = float(np.min([p.min() for p in all_preds]))
        hi = float(np.max([p.max() for p in all_preds]))
        pad = max(0.02, (hi - lo) * 0.08)
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    else:
        ax.set_ylim(0.15, 0.45)
    ax.grid(True, alpha=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"[plot_scale_inverted_u_judge] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
