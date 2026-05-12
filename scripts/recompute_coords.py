"""Compute deterministic (S,R) for every corpus source; write data/source_coords.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import PROJECT_ROOT
from src.corpus import load_sources
from src.text_style_coords import TextStyleCoords


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "source_coords.jsonl")
    p.add_argument("--limit-per-genre", type=int, default=None)
    args = p.parse_args()

    try:
        tsc = TextStyleCoords()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    sources = load_sources(limit_per_genre=args.limit_per_genre)
    if not sources:
        print("No sources found", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in tqdm(sources, desc="[recompute_coords]", unit="src"):
        s, r = tsc.get_Ps(src.text)
        rel_path = str(src.path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        rows.append(
            {
                "id": src.id,
                "genre": src.genre,
                "S": s,
                "R": r,
                "path": rel_path,
            }
        )

    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[recompute_coords] wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
