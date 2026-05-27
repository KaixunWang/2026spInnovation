"""Run CrEval vs plain-LLM external-evaluation pipeline on existing metrics.

This script is additive and non-destructive to the existing judge channel:

1) score with CrEval (scripts/score_existing_results_with_creval.py)
2) score with plain pairwise LLM baseline (scripts/score_existing_results_with_pairwise_llm.py)
3) merge both external channels into *_metrics.jsonl
4) run scale regressions for both channels
5) write a side-by-side comparison table

Run from repo root::

    python scripts/run_creval_vs_untrained_pipeline.py --overwrite
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

DEFAULT_INPUTS = [
    ROOT / "data" / "generated" / "main_qwen3_4b_metrics.jsonl",
    ROOT / "data" / "generated" / "main_qwen3_8b_metrics.jsonl",
    ROOT / "data" / "generated" / "main_qwen3_14b_metrics.jsonl",
]


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "external_eval"


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _ext_path_for(metrics_path: Path, suffix: str) -> Path:
    stem = metrics_path.stem.replace("_metrics", "")
    clean = suffix.strip().strip("_")
    return metrics_path.with_name(f"{stem}_{clean}.jsonl")


def _merge_external(
    metrics_path: Path,
    external_path: Path,
    *,
    field_name: str,
    allow_missing: bool,
    allow_extra: bool,
) -> None:
    if not metrics_path.exists():
        print(f"[pipeline] skip merge: missing metrics {metrics_path}")
        return
    if not external_path.exists():
        print(f"[pipeline] skip merge: missing external {external_path}")
        return

    cmd = [
        PY,
        str(ROOT / "scripts" / "merge_external_eval_into_metrics.py"),
        str(metrics_path),
        str(external_path),
        "--field-name",
        field_name,
        "-o",
        str(metrics_path),
    ]
    if allow_missing:
        cmd.append("--allow-missing")
    if allow_extra:
        cmd.append("--allow-extra")
    _run(cmd)


def _build_dual_compare_table(
    *,
    creval_field: str,
    baseline_field: str,
) -> Path | None:
    tables_dir = ROOT / "results" / "tables"
    creval_slug = _slug(creval_field)
    baseline_slug = _slug(baseline_field)

    creval_compare = tables_dir / f"{creval_slug}_auto_compare.csv"
    baseline_compare = tables_dir / f"{baseline_slug}_auto_compare.csv"

    if not creval_compare.exists() or not baseline_compare.exists():
        print(
            "[pipeline] skip dual compare table: missing "
            f"{creval_compare.name if not creval_compare.exists() else ''} "
            f"{baseline_compare.name if not baseline_compare.exists() else ''}".strip()
        )
        return None

    df_c = pd.read_csv(creval_compare)
    df_b = pd.read_csv(baseline_compare)

    key_cols = ["generator_label", "metrics_file", "generator_model", "n_T3"]
    c_cols = {
        f"mean_{creval_slug}_score_T3": "mean_creval_score_T3",
        f"beta_d2_{creval_slug}_score": "beta_d2_creval_score",
        f"p_d2_{creval_slug}_score": "p_d2_creval_score",
        f"sig_p05_{creval_slug}_score": "sig_p05_creval_score",
    }
    b_cols = {
        f"mean_{baseline_slug}_score_T3": f"mean_{baseline_slug}_score_T3",
        f"beta_d2_{baseline_slug}_score": f"beta_d2_{baseline_slug}_score",
        f"p_d2_{baseline_slug}_score": f"p_d2_{baseline_slug}_score",
        f"sig_p05_{baseline_slug}_score": f"sig_p05_{baseline_slug}_score",
    }

    keep_c = key_cols + [c for c in c_cols if c in df_c.columns] + [
        "mean_creativity_auto_T3",
        "beta_d2_creativity_auto",
        "p_d2_creativity_auto",
        "sig_p05_creativity_auto",
    ]
    keep_b = key_cols + [c for c in b_cols if c in df_b.columns]

    c2 = df_c[keep_c].rename(columns=c_cols)
    b2 = df_b[keep_b].rename(columns=b_cols)

    merged = c2.merge(b2, on=key_cols, how="outer")

    baseline_mean_col = f"mean_{baseline_slug}_score_T3"
    if "mean_creval_score_T3" in merged.columns and baseline_mean_col in merged.columns:
        merged["delta_mean_creval_minus_baseline"] = (
            merged["mean_creval_score_T3"] - merged[baseline_mean_col]
        )

    b2_creval = "beta_d2_creval_score"
    b2_baseline = f"beta_d2_{baseline_slug}_score"
    if b2_creval in merged.columns and b2_baseline in merged.columns:
        def _same_sign(a: Any, b: Any) -> str:
            try:
                fa = float(a)
                fb = float(b)
            except (TypeError, ValueError):
                return ""
            if abs(fa) < 1e-15 and abs(fb) < 1e-15:
                return "yes"
            if abs(fa) < 1e-15 or abs(fb) < 1e-15:
                return "mixed"
            return "yes" if (fa > 0) == (fb > 0) else "no"

        merged["same_sign_beta_d2_creval_vs_baseline"] = [
            _same_sign(a, b) for a, b in zip(merged[b2_creval], merged[b2_baseline], strict=False)
        ]

    out = tables_dir / f"{creval_slug}_vs_{baseline_slug}_compare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"[pipeline] wrote {out}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="*", type=Path, default=None)
    p.add_argument("--reference-model", default="gen_openai_4o")
    p.add_argument("--condition", default="T3")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-creval", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--allow-extra", action="store_true")
    p.add_argument("--creval-field-name", default="creval")
    p.add_argument("--creval-retries", type=int, default=5)
    p.add_argument("--creval-sleep-seconds", type=float, default=1.5)
    p.add_argument("--creval-max-tokens", type=int, default=32)
    p.add_argument("--creval-progress-every", type=int, default=5)
    p.add_argument("--baseline-field-name", default="llm_pairwise")
    p.add_argument("--baseline-output-suffix", default="llm_pairwise")
    p.add_argument("--baseline-retries", type=int, default=5)
    p.add_argument("--baseline-sleep-seconds", type=float, default=1.5)
    p.add_argument("--baseline-max-tokens", type=int, default=128)
    p.add_argument("--baseline-progress-every", type=int, default=5)
    p.add_argument("--baseline-fallback-models", nargs="*", default=None)
    args = p.parse_args()

    inputs = args.inputs or DEFAULT_INPUTS

    if not args.skip_creval:
        cmd = [
            PY,
            str(ROOT / "scripts" / "score_existing_results_with_creval.py"),
            "--reference-model",
            args.reference_model,
            "--condition",
            args.condition,
            "--retries",
            str(args.creval_retries),
            "--sleep-seconds",
            str(args.creval_sleep_seconds),
            "--max-tokens",
            str(args.creval_max_tokens),
            "--progress-every",
            str(args.creval_progress_every),
            "--inputs",
            *[str(p) for p in inputs],
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.dry_run:
            cmd.append("--dry-run")
        _run(cmd)

    if not args.skip_baseline:
        cmd = [
            PY,
            str(ROOT / "scripts" / "score_existing_results_with_pairwise_llm.py"),
            "--reference-model",
            args.reference_model,
            "--condition",
            args.condition,
            "--field-name",
            args.baseline_field_name,
            "--output-suffix",
            args.baseline_output_suffix,
            "--retries",
            str(args.baseline_retries),
            "--sleep-seconds",
            str(args.baseline_sleep_seconds),
            "--max-tokens",
            str(args.baseline_max_tokens),
            "--progress-every",
            str(args.baseline_progress_every),
            "--inputs",
            *[str(p) for p in inputs],
        ]
        if args.baseline_fallback_models:
            cmd.extend(["--fallback-models", *args.baseline_fallback_models])
        if args.overwrite:
            cmd.append("--overwrite")
        if args.dry_run:
            cmd.append("--dry-run")
        _run(cmd)

    if args.dry_run:
        print("[pipeline] dry-run enabled: skip merge + analysis to keep metrics files unchanged")
    else:
        for metrics_path in inputs:
            creval_ext = _ext_path_for(metrics_path, "creval")
            baseline_ext = _ext_path_for(metrics_path, args.baseline_output_suffix)

            _merge_external(
                metrics_path,
                creval_ext,
                field_name=args.creval_field_name,
                allow_missing=args.allow_missing,
                allow_extra=args.allow_extra,
            )
            _merge_external(
                metrics_path,
                baseline_ext,
                field_name=args.baseline_field_name,
                allow_missing=args.allow_missing,
                allow_extra=args.allow_extra,
            )

    if not args.skip_analysis and not args.dry_run:
        _run(
            [
                PY,
                str(ROOT / "scripts" / "external_eval_scale_regression.py"),
                "--field-name",
                args.creval_field_name,
                "--score-key",
                "score",
                "--inputs",
                *[str(p) for p in inputs],
            ]
        )
        _run(
            [
                PY,
                str(ROOT / "scripts" / "external_eval_scale_regression.py"),
                "--field-name",
                args.baseline_field_name,
                "--score-key",
                "score",
                "--inputs",
                *[str(p) for p in inputs],
            ]
        )
        _build_dual_compare_table(
            creval_field=args.creval_field_name,
            baseline_field=args.baseline_field_name,
        )
        _run(
            [
                PY,
                str(ROOT / "scripts" / "summarize_pairwise_winrate.py"),
                "--fields",
                args.creval_field_name,
                args.baseline_field_name,
                "--inputs",
                *[str(p) for p in inputs],
            ]
        )

    print("[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
