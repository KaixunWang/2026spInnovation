"""Validate data/source_coords.jsonl: genre-level S/R means vs expected directions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import PROJECT_ROOT


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_coords", type=Path, nargs="?", default=PROJECT_ROOT / "data" / "source_coords.jsonl")
    p.add_argument("--essay-epsilon", type=float, default=0.2, help="|mean| must be below this for essay")
    args = p.parse_args()

    if not args.source_coords.exists():
        print(f"Missing {args.source_coords}", file=sys.stderr)
        return 1

    rows = _read_rows(args.source_coords)
    df = pd.DataFrame(rows)
    if df.empty or "genre" not in df.columns:
        print("No data", file=sys.stderr)
        return 1

    print("=== Per-genre S/R (mean, std, n) ===")
    gstats = df.groupby("genre")[["S", "R"]].agg(["mean", "std", "count"])
    print(gstats.to_string())
    print()

    checks: list[tuple[str, bool, str]] = []

    def mean_s(g: str) -> float:
        sub = df.loc[df["genre"] == g, "S"]
        return float(sub.mean()) if len(sub) else float("nan")

    def mean_r(g: str) -> float:
        sub = df.loc[df["genre"] == g, "R"]
        return float(sub.mean()) if len(sub) else float("nan")

    ms = mean_s("academic")
    mr = mean_r("academic")
    ok_a = ms > 0 and mr < 0
    checks.append(("academic: S>0 and R<0", ok_a, f"S_mean={ms:.4f}, R_mean={mr:.4f}"))

    ms = mean_s("poetry")
    mr = mean_r("poetry")
    ok_p = ms < 0 and mr > 0
    checks.append(("poetry: S<0 and R>0", ok_p, f"S_mean={ms:.4f}, R_mean={mr:.4f}"))

    ms = mean_s("narrative")
    ok_n = ms < 0
    checks.append(("narrative: S<0", ok_n, f"S_mean={ms:.4f}"))

    ms = mean_s("essay")
    mr = mean_r("essay")
    ok_e = abs(ms) < args.essay_epsilon and abs(mr) < args.essay_epsilon
    checks.append(
        (
            f"essay: |S|,|R| < {args.essay_epsilon}",
            ok_e,
            f"S_mean={ms:.4f}, R_mean={mr:.4f}",
        )
    )

    print("=== Acceptance checks ===")
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}: {detail}")

    if all_ok:
        print("\nAll checks passed.")
        return 0
    print("\nSome checks failed — tune anchors or models before locking coordinates.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
