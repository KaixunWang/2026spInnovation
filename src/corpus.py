"""Load source texts from data/source_texts/.

Each text is a Markdown file with YAML front-matter (see data/source_texts/README.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config_loader import PROJECT_ROOT


SOURCE_DIR = PROJECT_ROOT / "data" / "source_texts"
GENRES = ("academic", "narrative", "essay", "poetry")
NON_SOURCE_MD = {"readme.md", "manual_collection.md"}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass(frozen=True)
class SourceText:
    id: str
    genre: str
    source: str
    license: str
    length: int
    note: str
    text: str
    path: Path

    def word_count(self) -> int:
        return len(self.text.split())


def _parse_markdown(path: Path) -> SourceText:
    raw = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError(f"{path} is missing YAML front-matter")
    fm_text, body = m.group(1), m.group(2).strip()
    fm = yaml.safe_load(fm_text)
    required = {"id", "genre"}
    missing = required - set(fm.keys())
    if missing:
        raise ValueError(f"{path} front-matter missing keys: {missing}")
    return SourceText(
        id=str(fm["id"]),
        genre=str(fm["genre"]),
        source=str(fm.get("source", "")),
        license=str(fm.get("license", "")),
        length=int(fm.get("length", len(body.split()))),
        note=str(fm.get("note", "")),
        text=body,
        path=path,
    )


def load_sources(
    root: Path | str = SOURCE_DIR,
    *,
    genres: tuple[str, ...] = GENRES,
    limit_per_genre: int | None = None,
) -> list[SourceText]:
    root = Path(root)
    out: list[SourceText] = []
    for g in genres:
        gdir = root / g
        if not gdir.exists():
            continue
        files = sorted(p for p in gdir.glob("*.md") if p.name.lower() not in NON_SOURCE_MD)
        if limit_per_genre is not None:
            files = files[:limit_per_genre]
        for p in files:
            out.append(_parse_markdown(p))
    return out


def validate_corpus(root: Path | str = SOURCE_DIR, *, expected_per_genre: int = 15) -> tuple[bool, list[str]]:
    """Lightweight check on counts, lengths, and uniqueness."""
    errors: list[str] = []
    sources = load_sources(root)
    ids = [s.id for s in sources]
    if len(ids) != len(set(ids)):
        errors.append(f"duplicate ids in corpus: {[i for i in ids if ids.count(i) > 1][:5]}")
    for g in GENRES:
        n = sum(1 for s in sources if s.genre == g)
        if n < expected_per_genre:
            errors.append(f"genre {g!r}: {n} file(s) found, expected {expected_per_genre}")
    for s in sources:
        wc = s.word_count()
        if wc < 80 or wc > 380:
            errors.append(f"{s.id}: word_count={wc} outside soft range [80, 380]")
    return (not errors), errors
