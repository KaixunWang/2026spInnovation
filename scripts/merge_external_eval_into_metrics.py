"""Merge an external evaluator's outputs into a *_metrics.jsonl file.

This is the non-destructive path for adding a parallel scoring channel such as
``creval`` without overwriting the existing ``judge`` block.

Supported external-row shapes:

1. Minimal row with top-level score fields::

       {"source_id": "essay_001", "condition": "T3", "repeat_idx": 0,
        "model": "gen_qwen3_4b", "score": 0.81, "details": {...}}

2. Nested payload under the target field name::

       {"source_id": "essay_001", "condition": "T3", "repeat_idx": 0,
        "model": "gen_qwen3_4b", "creval": {"ok": true, "score": 0.81}}

Matching is key-based by default, not line-wise, so the external file may be
generated separately as long as the identifying fields are preserved.

Example:

    python scripts/merge_external_eval_into_metrics.py \
        data/generated/main_qwen3_4b_metrics.jsonl \
        data/generated/main_qwen3_4b_creval.jsonl \
        --field-name creval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALIGN_KEYS = (
    "source_id",
    "condition",
    "mode",
    "repeat_idx",
    "model",
    "hop_index",
    "path_id",
    "target_persona",
    "prompt_variant",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _condition_value(row: dict[str, Any]) -> Any:
    return row.get("condition", row.get("mode"))


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_id"),
        _condition_value(row),
        row.get("repeat_idx"),
        row.get("model"),
        row.get("hop_index"),
        row.get("path_id"),
        row.get("target_persona"),
        row.get("prompt_variant"),
    )


def _payload_from_external(row: dict[str, Any], field_name: str, score_key: str) -> dict[str, Any]:
    payload = row.get(field_name)
    if isinstance(payload, dict):
        out = dict(payload)
    else:
        out = {k: v for k, v in row.items() if k not in ALIGN_KEYS}
    if not out:
        raise SystemExit(
            f"external row for source_id={row.get('source_id')!r} has no mergeable payload; "
            f"either add a {field_name!r} object or top-level score fields"
        )
    if "ok" not in out:
        score = out.get(score_key)
        out["ok"] = isinstance(score, (int, float))
    return out


def merge(
    metrics_path: Path,
    external_path: Path,
    out_path: Path,
    *,
    field_name: str,
    score_key: str,
    allow_missing: bool,
    allow_extra: bool,
) -> int:
    metrics_rows = _read_rows(metrics_path)
    external_rows = _read_rows(external_path)

    external_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in external_rows:
        key = _row_key(row)
        if key in external_by_key:
            raise SystemExit(f"duplicate external key: {key}")
        external_by_key[key] = _payload_from_external(row, field_name, score_key)

    matched_keys: set[tuple[Any, ...]] = set()
    out_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(metrics_rows):
        key = _row_key(row)
        payload = external_by_key.get(key)
        if payload is None:
            if not allow_missing:
                raise SystemExit(
                    f"metrics row {idx} missing external match for key={key}. "
                    f"Use --allow-missing to pass unmatched rows through unchanged."
                )
            out_rows.append(dict(row))
            continue
        matched_keys.add(key)
        merged = dict(row)
        merged[field_name] = dict(payload)
        out_rows.append(merged)

    if not allow_extra:
        extras = [k for k in external_by_key if k not in matched_keys]
        if extras:
            sample = extras[:3]
            raise SystemExit(
                f"external file has {len(extras)} unmatched rows, sample={sample}. "
                f"Use --allow-extra to ignore them."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(out_rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics_jsonl", type=Path)
    p.add_argument("external_jsonl", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None, help="default: overwrite metrics path")
    p.add_argument("--field-name", default="creval", help="target field name on metrics rows")
    p.add_argument("--score-key", default="score", help="numeric score key inside the external payload")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="pass metrics rows through unchanged when no external match exists",
    )
    p.add_argument(
        "--allow-extra",
        action="store_true",
        help="ignore external rows that do not match any metrics row",
    )
    args = p.parse_args()

    out = args.output or args.metrics_jsonl
    n = merge(
        args.metrics_jsonl,
        args.external_jsonl,
        out,
        field_name=args.field_name,
        score_key=args.score_key,
        allow_missing=args.allow_missing,
        allow_extra=args.allow_extra,
    )
    print(f"[merge_external_eval] wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())