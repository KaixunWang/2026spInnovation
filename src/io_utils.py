"""JSONL helpers and project-relative path conventions."""

from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .config_loader import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def write_jsonl(path: Path | str, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if is_dataclass(row):
                row = asdict(row)
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
    return n


def append_jsonl(path: Path | str, row: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(row):
        row = asdict(row)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path | str) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
