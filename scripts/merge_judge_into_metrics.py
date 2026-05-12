"""Merge LLM-judge outputs into *_metrics.jsonl so analyze sees one row per generation.

Judge reads raw *_jsonl and writes *_judged.jsonl (same row order). Metrics reads the
same raw file and writes *_metrics.jsonl (same row order). This script zips the two
outputs line-wise and copies ``judge`` onto the metrics row.

Use ``--partial`` when *_judged.jsonl has fewer lines than *_metrics.jsonl (e.g. judge
stopped mid-run): only the prefix rows get ``judge`` copied; trailing metrics rows are
written unchanged so you can preview analyze on the judged prefix (tail rows keep prior
``judge`` if any, otherwise no field).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge(metrics_path: Path, judged_path: Path, out_path: Path, *, partial: bool = False) -> int:
    m_rows = _read_rows(metrics_path)
    j_rows = _read_rows(judged_path)
    if len(j_rows) > len(m_rows):
        raise SystemExit(
            f"judged has more rows than metrics: {judged_path.name}={len(j_rows)} > "
            f"{metrics_path.name}={len(m_rows)}"
        )
    if len(m_rows) != len(j_rows):
        if not partial:
            raise SystemExit(
                f"row count mismatch: {metrics_path.name}={len(m_rows)} vs {judged_path.name}={len(j_rows)} "
                f"(use --partial to merge only the first {len(j_rows)} judged rows)"
            )
        print(
            f"[merge_judge] partial: copying judge for rows 0..{len(j_rows) - 1}, "
            f"passing through {len(m_rows) - len(j_rows)} trailing metrics rows unchanged",
            file=sys.stderr,
        )

    out: list[dict] = []
    disable = os.environ.get("NO_TQDM", "").strip().lower() in ("1", "true", "yes")
    n_j = len(j_rows)
    prefix = list(zip(m_rows[:n_j], j_rows))
    z = prefix
    if not disable:
        z = tqdm(prefix, total=n_j, desc="[merge_judge] rows", file=sys.stderr, unit="row")
    for i, (m, j) in enumerate(z):
        if m.get("source_id") != j.get("source_id") or m.get("condition") != j.get("condition"):
            raise SystemExit(f"row {i}: source_id/condition mismatch between metrics and judged")
        m2 = dict(m)
        m2["judge"] = j.get("judge", {})
        out.append(m2)
    for m in m_rows[n_j:]:
        out.append(dict(m))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics_jsonl", type=Path)
    p.add_argument("judged_jsonl", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None, help="default: overwrite metrics path")
    p.add_argument(
        "--partial",
        action="store_true",
        help="allow fewer judged rows than metrics; merge judge only for the aligned prefix",
    )
    args = p.parse_args()
    out = args.output or args.metrics_jsonl
    n = merge(args.metrics_jsonl, args.judged_jsonl, out, partial=args.partial)
    print(f"[merge_judge] wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
