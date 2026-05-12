"""Copy data/generated/*_metrics.jsonl and *_judged.jsonl into results/backups/<timestamp>/.

Run before ``metrics --overwrite``-style passes so you can restore or use
``--preserve-judge-from`` with the backup paths.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "generated"
OUT = ROOT / "results" / "backups"


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = OUT / ts
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in ("*_metrics.jsonl", "*_judged.jsonl"):
        for p in sorted(GEN.glob(pattern)):
            shutil.copy2(p, dest / p.name)
            copied += 1
    print(f"[backup_generated_metrics] copied {copied} files -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
