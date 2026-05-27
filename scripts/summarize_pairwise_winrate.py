"""Summarize pairwise evaluator win rates from merged *_metrics.jsonl files.

This script reads external evaluator blocks (e.g. ``creval``, ``llm_pairwise``)
that were merged into metrics rows by scripts/merge_external_eval_into_metrics.py,
and reports win/tie/loss style summaries.

Score convention (same as existing CrEval integration):

- 1.0: candidate beats reference
- 0.5: tie
- 0.0: candidate loses to reference

So mean score equals adjusted win rate: (wins + 0.5 * ties) / n_valid.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "external_eval"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_score(row: dict[str, Any], field: str) -> float | None:
    block = row.get(field)
    if not isinstance(block, dict):
        return None
    if "ok" in block and not block.get("ok"):
        return None
    score = block.get("score")
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if math.isnan(s):
        return None
    return s


def _to_frame(path: Path, field: str, condition: str) -> pd.DataFrame:
    rows = _read_jsonl(path)
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("condition") != condition:
            continue
        out.append(
            {
                "metrics_file": path.name,
                "generator_model": r.get("model"),
                "source_id": r.get("source_id"),
                "genre": r.get("genre"),
                "condition": r.get("condition"),
                "d_H": r.get("d_H"),
                "score": _extract_score(r, field),
            }
        )
    return pd.DataFrame(out)


def _summarize(df: pd.DataFrame, *, field: str, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        cols = [*group_cols, "field", "n_rows", "n_valid", "win_rate", "tie_rate", "loss_rate", "mean_score"]
        return pd.DataFrame(columns=cols)

    work = df.copy()
    work["valid"] = work["score"].notna()
    work["is_win"] = work["score"] == 1.0
    work["is_tie"] = work["score"] == 0.5
    work["is_loss"] = work["score"] == 0.0

    def _agg(g: pd.DataFrame) -> pd.Series:
        n_rows = int(len(g))
        valid = g[g["valid"]]
        n_valid = int(len(valid))
        if n_valid == 0:
            return pd.Series(
                {
                    "field": field,
                    "n_rows": n_rows,
                    "n_valid": 0,
                    "win_rate": float("nan"),
                    "tie_rate": float("nan"),
                    "loss_rate": float("nan"),
                    "mean_score": float("nan"),
                }
            )

        win = float(valid["is_win"].mean())
        tie = float(valid["is_tie"].mean())
        loss = float(valid["is_loss"].mean())
        mean_score = float(valid["score"].mean())
        return pd.Series(
            {
                "field": field,
                "n_rows": n_rows,
                "n_valid": n_valid,
                "win_rate": win,
                "tie_rate": tie,
                "loss_rate": loss,
                "mean_score": mean_score,
            }
        )

    out = work.groupby(group_cols, dropna=False, observed=False).apply(_agg).reset_index()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[
            Path("data/generated/main_qwen3_4b_metrics.jsonl"),
            Path("data/generated/main_qwen3_8b_metrics.jsonl"),
            Path("data/generated/main_qwen3_14b_metrics.jsonl"),
        ],
        help="merged metrics jsonl files to summarize",
    )
    p.add_argument("--fields", nargs="+", default=["creval", "llm_pairwise"])
    p.add_argument("--condition", default="T3")
    p.add_argument("--output-summary", type=Path, default=Path("results/tables/pairwise_winrate_summary.csv"))
    p.add_argument("--output-by-genre", type=Path, default=Path("results/tables/pairwise_winrate_by_genre.csv"))
    p.add_argument(
        "--output-compare",
        type=Path,
        default=Path("results/tables/creval_vs_llm_pairwise_winrate_compare.csv"),
        help="wide compare table for the first two fields in --fields",
    )
    args = p.parse_args()

    frames: list[pd.DataFrame] = []
    for input_path in args.inputs:
        if not input_path.exists():
            print(f"[summarize_pairwise_winrate] skip missing {input_path}")
            continue
        for field in args.fields:
            df = _to_frame(input_path, field, args.condition)
            if df.empty:
                continue
            df["field"] = field
            frames.append(df)

    if not frames:
        print("[summarize_pairwise_winrate] no rows loaded")
        return 0

    long_df = pd.concat(frames, ignore_index=True)

    summary_parts: list[pd.DataFrame] = []
    by_genre_parts: list[pd.DataFrame] = []
    for field in args.fields:
        sub = long_df[long_df["field"] == field].copy()
        summary_parts.append(
            _summarize(sub, field=field, group_cols=["metrics_file", "generator_model"])
        )
        by_genre_parts.append(
            _summarize(sub, field=field, group_cols=["metrics_file", "generator_model", "genre"])
        )

    summary = pd.concat(summary_parts, ignore_index=True)
    by_genre = pd.concat(by_genre_parts, ignore_index=True)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False, encoding="utf-8")
    by_genre.to_csv(args.output_by_genre, index=False, encoding="utf-8")

    print(f"[summarize_pairwise_winrate] wrote {args.output_summary}")
    print(f"[summarize_pairwise_winrate] wrote {args.output_by_genre}")

    if len(args.fields) >= 2:
        left = args.fields[0]
        right = args.fields[1]
        s = summary.copy()
        key_cols = ["metrics_file", "generator_model"]

        l = s[s["field"] == left].rename(
            columns={
                "n_valid": f"n_valid_{_slug(left)}",
                "win_rate": f"win_rate_{_slug(left)}",
                "tie_rate": f"tie_rate_{_slug(left)}",
                "loss_rate": f"loss_rate_{_slug(left)}",
                "mean_score": f"mean_score_{_slug(left)}",
            }
        )
        r = s[s["field"] == right].rename(
            columns={
                "n_valid": f"n_valid_{_slug(right)}",
                "win_rate": f"win_rate_{_slug(right)}",
                "tie_rate": f"tie_rate_{_slug(right)}",
                "loss_rate": f"loss_rate_{_slug(right)}",
                "mean_score": f"mean_score_{_slug(right)}",
            }
        )

        drop_cols = ["field", "n_rows"]
        l = l.drop(columns=[c for c in drop_cols if c in l.columns])
        r = r.drop(columns=[c for c in drop_cols if c in r.columns])

        cmp_df = l.merge(r, on=key_cols, how="outer")
        l_mean = f"mean_score_{_slug(left)}"
        r_mean = f"mean_score_{_slug(right)}"
        if l_mean in cmp_df.columns and r_mean in cmp_df.columns:
            cmp_df[f"delta_{_slug(left)}_minus_{_slug(right)}"] = cmp_df[l_mean] - cmp_df[r_mean]

        args.output_compare.parent.mkdir(parents=True, exist_ok=True)
        cmp_df.to_csv(args.output_compare, index=False, encoding="utf-8")
        print(f"[summarize_pairwise_winrate] wrote {args.output_compare}")

    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
