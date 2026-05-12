"""Run optional experiment arms + main_continuous + metrics + judge + merge + analyze.

Usage (from repo root, with .env configured):

    python scripts/run_optional_pipeline.py

Steps: mechanism, multihop, main_continuous, build_space_l, metrics (all arms),
judge (all arms), merge judge into metrics, analyze.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    run([PY, "-m", "src.run_experiment", "mechanism", "--overwrite"])
    run([PY, "-m", "src.run_experiment", "multihop", "--overwrite"])
    run(
        [
            PY,
            "-m",
            "src.run_experiment",
            "main",
            "--target-sampling",
            "continuous",
            "--output-name",
            "main_continuous.jsonl",
            "--overwrite",
        ]
    )
    run([PY, "-m", "src.run_experiment", "build_space_l"])

    inputs = [
        "data/generated/main.jsonl",
        "data/generated/main_continuous.jsonl",
        "data/generated/mechanism.jsonl",
        "data/generated/multihop.jsonl",
    ]
    run(
        [PY, "-m", "src.run_experiment", "metrics", "--with-embedding", "--inputs", *inputs]
    )
    run([PY, "-m", "src.run_experiment", "judge", "--inputs", *inputs])

    pairs = [
        ("data/generated/main_metrics.jsonl", "data/generated/main_judged.jsonl"),
        ("data/generated/main_continuous_metrics.jsonl", "data/generated/main_continuous_judged.jsonl"),
        ("data/generated/mechanism_metrics.jsonl", "data/generated/mechanism_judged.jsonl"),
        ("data/generated/multihop_metrics.jsonl", "data/generated/multihop_judged.jsonl"),
    ]
    for mpath, jpath in pairs:
        mp = ROOT / mpath
        jp = ROOT / jpath
        if not mp.exists():
            print(f"[skip merge] missing {mp}", flush=True)
            continue
        if not jp.exists():
            print(f"[skip merge] missing {jp}", flush=True)
            continue
        run([PY, str(ROOT / "scripts" / "merge_judge_into_metrics.py"), str(mp), str(jp), "-o", str(mp)])

    run([PY, "-m", "src.analyze", "--metric", "creativity_auto"])
    print("\n[run_optional_pipeline] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
