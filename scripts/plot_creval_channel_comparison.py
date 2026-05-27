from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated"
CREVAL_WINRATE_PATH = ROOT / "creval_server_package" / "data" / "generated" / "creval_winrates.jsonl"
OUT_PATH = ROOT / "results" / "figures" / "creval_channel_comparison.png"


def _fuse_fidelity_judge_nli(fidelity_judge: float | None, nli_entailment: float | None, w_judge: float = 0.5) -> float:
    fj = None if fidelity_judge is None else max(0.0, min(1.0, float(fidelity_judge)))
    nli = None if nli_entailment is None else max(0.0, min(1.0, float(nli_entailment)))
    if fj is None and nli is None:
        return float("nan")
    if fj is None:
        return nli
    if nli is None:
        return fj
    w = max(0.0, min(1.0, float(w_judge)))
    return w * fj + (1.0 - w) * nli


def _creativity_judge_from_row(row: dict) -> float:
    judge = row.get("judge") or {}
    metrics = row.get("metrics") or {}
    if not judge.get("ok"):
        return float("nan")
    novelty_judge = judge.get("novelty_judge")
    coherence_judge = judge.get("coherence_judge")
    fidelity_judge = judge.get("fidelity_judge")
    if novelty_judge is None or coherence_judge is None or fidelity_judge is None:
        return float("nan")
    fidelity_fused = _fuse_fidelity_judge_nli(fidelity_judge, metrics.get("nli_entailment"), w_judge=0.5)
    value_judge = (max(0.0, fidelity_fused) + max(0.0, float(coherence_judge))) / 2.0
    return float(novelty_judge) * value_judge


def _load_creval_map() -> dict[tuple, dict]:
    out = {}
    with CREVAL_WINRATE_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row["source_id"],
                row["model"],
                row["condition"],
                row.get("target_persona", ""),
                row.get("repeat_idx", 0),
            )
            out[key] = row
    return out


def _build_dataframe() -> pd.DataFrame:
    creval_map = _load_creval_map()
    rows = []
    metric_files = [
        ("main_qwen3_4b_metrics.jsonl", None),
        ("main_qwen3_8b_metrics.jsonl", None),
        ("main_qwen3_14b_metrics.jsonl", None),
        ("main_metrics.jsonl", "openai"),
    ]
    for fname, model_filter in metric_files:
        with (DATA_DIR / fname).open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                model = row.get("model", "")
                if model_filter and model_filter not in model.lower():
                    continue
                key = (
                    row["source_id"],
                    model,
                    row["condition"],
                    row.get("target_persona", ""),
                    row.get("repeat_idx", 0),
                )
                if key not in creval_map:
                    continue
                metrics = row.get("metrics") or {}
                rows.append(
                    {
                        "condition": row.get("condition", ""),
                        "model": model,
                        "creval_winrate": creval_map[key].get("creval_winrate"),
                        "creativity_judge": _creativity_judge_from_row(row),
                        "creativity_auto": metrics.get("creativity_auto"),
                    }
                )
    return pd.DataFrame(rows)


def _plot() -> None:
    df = _build_dataframe()
    condition_order = ["T0", "T1", "T2", "T3"]
    model_order = ["gen_openai_4o", "gen_qwen3_14b", "gen_qwen3_8b", "gen_qwen3_4b"]
    model_label = {
        "gen_openai_4o": "GPT-4o",
        "gen_qwen3_14b": "Qwen3-14B",
        "gen_qwen3_8b": "Qwen3-8B",
        "gen_qwen3_4b": "Qwen3-4B",
    }
    channel_cols = ["creval_winrate", "creativity_judge", "creativity_auto"]
    channel_label = {
        "creval_winrate": r"$C_{\mathrm{creval}}$",
        "creativity_judge": r"$C_{\mathrm{judge}}$",
        "creativity_auto": r"$C_{\mathrm{auto}}$",
    }
    channel_colors = {
        "creval_winrate": "#1f77b4",
        "creativity_judge": "#ff7f0e",
        "creativity_auto": "#2ca02c",
    }

    cond_mean = df.groupby("condition", dropna=False)[channel_cols].mean().reindex(condition_order)
    model_mean = df.groupby("model", dropna=False)[channel_cols].mean().reindex(model_order)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    width = 0.24
    x = np.arange(len(condition_order))
    for i, col in enumerate(channel_cols):
        axes[0].bar(x + (i - 1) * width, cond_mean[col].to_numpy(), width=width, color=channel_colors[col], label=channel_label[col])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(condition_order)
    axes[0].set_ylim(0.0, 0.75)
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("(a) Condition means")
    axes[0].grid(axis="y", alpha=0.22)

    x2 = np.arange(len(model_order))
    for i, col in enumerate(channel_cols):
        axes[1].bar(x2 + (i - 1) * width, model_mean[col].to_numpy(), width=width, color=channel_colors[col], label=channel_label[col])
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([model_label[m] for m in model_order], rotation=12, ha="right")
    axes[1].set_ylim(0.0, 0.75)
    axes[1].set_title("(b) Model means")
    axes[1].grid(axis="y", alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    _plot()