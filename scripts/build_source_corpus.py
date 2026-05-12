"""Build source_texts from recommended public datasets.

This script auto-builds three genres:
  - academic  <- gfissore/arxiv-abstracts-2021
  - narrative <- sanps/GutenbergFiction
  - poetry    <- merve/poetry

Essay is intentionally manual (Wikisource/public-domain essays), because
HF essay corpora are mixed-format and less clean for this study.

Usage:
  python -m scripts.build_source_corpus --target-per-genre 15

Requirements:
  pip install "datasets>=2.19.0"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm


def _scan_pbar(ds: Any, *, desc: str) -> Any:
    disable = os.environ.get("NO_TQDM", "").strip().lower() in ("1", "true", "yes")
    return tqdm(ds, desc=desc, file=sys.stderr, unit="row", disable=disable)


WORD_RE = re.compile(r"[A-Za-z']+")


def wc(text: str) -> int:
    return len(WORD_RE.findall(text))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    return text


def canonical_fingerprint(text: str, n: int = 160) -> str:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    return " ".join(toks[:n])


def write_source_file(
    *,
    out_path: Path,
    item_id: str,
    genre: str,
    source: str,
    license_name: str,
    note: str,
    text: str,
) -> None:
    text = clean_text(text)
    body_wc = wc(text)
    fm = {
        "id": item_id,
        "genre": genre,
        "source": source,
        "license": license_name,
        "length": body_wc,
        "note": note,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fm_text = yaml.safe_dump(fm, sort_keys=False, allow_unicode=False).strip()
    out_path.write_text(
        "---\n"
        + fm_text
        + "\n---\n"
        + text
        + "\n",
        encoding="utf-8",
    )


def maybe_get_first_str(row: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            parts = [str(x).strip() for x in v if str(x).strip()]
            if parts:
                return " ".join(parts)
    return None


def pull_academic(
    *,
    n: int,
    out_dir: Path,
    seed: int,
    min_words: int = 150,
    max_words: int = 300,
    overwrite: bool = False,
) -> int:
    from datasets import load_dataset

    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
    ds = ds.shuffle(seed=seed)
    kept = 0
    seen_fp: set[str] = set()

    # Basic field guesses across variants.
    abstract_keys = ["abstract", "summary", "text", "content"]
    cat_keys = ["categories", "category", "primary_category", "arxiv_primary_category"]
    id_keys = ["id", "arxiv_id", "paper_id"]

    by_top_cat: defaultdict[str, int] = defaultdict(int)
    max_per_top_cat = max(3, n // 4 + 1)

    for row in _scan_pbar(ds, desc="[corpus] academic"):
        if kept >= n:
            break
        abstract = maybe_get_first_str(row, abstract_keys)
        if not abstract:
            continue
        abstract = clean_text(abstract)
        n_words = wc(abstract)
        if n_words < min_words or n_words > max_words:
            continue
        fp = canonical_fingerprint(abstract)
        if fp in seen_fp:
            continue

        cats = maybe_get_first_str(row, cat_keys) or "unknown"
        top_cat = cats.split(".")[0].split(" ")[0]
        if top_cat != "unknown" and by_top_cat[top_cat] >= max_per_top_cat:
            continue

        raw_id = maybe_get_first_str(row, id_keys) or f"academic_auto_{kept+1:02d}"
        item_id = f"academic_{kept+1:02d}"
        out_path = out_dir / f"{item_id}.md"
        if out_path.exists() and not overwrite:
            kept += 1
            continue

        write_source_file(
            out_path=out_path,
            item_id=item_id,
            genre="academic",
            source=f"gfissore/arxiv-abstracts-2021::{raw_id}",
            license_name="research-use",
            note=f"auto-selected abstract; top_category={top_cat}",
            text=abstract,
        )
        seen_fp.add(fp)
        by_top_cat[top_cat] += 1
        kept += 1

    return kept


def pull_narrative(
    *,
    n: int,
    out_dir: Path,
    seed: int,
    min_words: int = 150,
    max_words: int = 300,
    overwrite: bool = False,
) -> int:
    from datasets import load_dataset

    ds = load_dataset("sanps/GutenbergFiction", split="train")
    ds = ds.shuffle(seed=seed)
    kept = 0
    seen_fp: set[str] = set()

    text_keys = ["text", "content", "paragraph", "passage"]
    source_keys = ["title", "book", "source", "author", "id"]

    for row in _scan_pbar(ds, desc="[corpus] narrative"):
        if kept >= n:
            break
        text = maybe_get_first_str(row, text_keys)
        if not text:
            continue
        text = clean_text(text)
        n_words = wc(text)
        if n_words < min_words or n_words > max_words:
            continue
        # Filter obvious non-prose blocks
        upper = text.upper()
        roman_toc_hits = len(re.findall(r"\b[IVXLCDM]+\.\s+[A-Z][A-Za-z' -]+ \d+\b", text))
        if "CHAPTER PAGE" in upper or roman_toc_hits > 5 or text.count("|") > 10:
            continue
        fp = canonical_fingerprint(text)
        if fp in seen_fp:
            continue

        src = maybe_get_first_str(row, source_keys) or "unknown"
        item_id = f"narrative_{kept+1:02d}"
        out_path = out_dir / f"{item_id}.md"
        if out_path.exists() and not overwrite:
            kept += 1
            continue

        write_source_file(
            out_path=out_path,
            item_id=item_id,
            genre="narrative",
            source=f"sanps/GutenbergFiction::{src}",
            license_name="public-domain",
            note="auto-selected paragraph window",
            text=text,
        )
        seen_fp.add(fp)
        kept += 1

    return kept


def split_poem_to_window(text: str, min_words: int, max_words: int) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    # For short poems under min_words, we keep full poem if >=80 words.
    full = " ".join(lines)
    full_wc = wc(full)
    if min_words <= full_wc <= max_words:
        return full
    if 80 <= full_wc < min_words:
        return full
    # For long poems, sliding line windows.
    for i in range(len(lines)):
        chunk: list[str] = []
        for j in range(i, len(lines)):
            chunk.append(lines[j])
            txt = " ".join(chunk)
            w = wc(txt)
            if min_words <= w <= max_words:
                return txt
            if w > max_words:
                break
    return None


def pull_poetry(
    *,
    n: int,
    out_dir: Path,
    seed: int,
    min_words: int = 150,
    max_words: int = 300,
    overwrite: bool = False,
) -> int:
    from datasets import load_dataset

    ds = load_dataset("merve/poetry", split="train")
    ds = ds.shuffle(seed=seed)
    kept = 0
    seen_fp: set[str] = set()

    text_keys = ["content", "poem", "text", "body"]
    title_keys = ["title", "poem_title", "id"]
    theme_keys = ["topic", "theme", "category"]

    for row in _scan_pbar(ds, desc="[corpus] poetry"):
        if kept >= n:
            break
        poem = maybe_get_first_str(row, text_keys)
        if not poem:
            continue
        window = split_poem_to_window(poem, min_words=min_words, max_words=max_words)
        if not window:
            continue
        fp = canonical_fingerprint(window)
        if fp in seen_fp:
            continue

        title = maybe_get_first_str(row, title_keys) or "unknown_title"
        theme = maybe_get_first_str(row, theme_keys) or "unknown_theme"
        item_id = f"poetry_{kept+1:02d}"
        out_path = out_dir / f"{item_id}.md"
        if out_path.exists() and not overwrite:
            kept += 1
            continue

        write_source_file(
            out_path=out_path,
            item_id=item_id,
            genre="poetry",
            source=f"merve/poetry::{title}",
            license_name="research-use",
            note=f"auto-selected poem window; theme={theme}",
            text=window,
        )
        seen_fp.add(fp)
        kept += 1

    return kept


def write_essay_manifest(out_root: Path, target_n: int) -> None:
    p = out_root / "essay" / "MANUAL_COLLECTION.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        textwrap.dedent(
            f"""\
            # Essay Manual Collection Guide

            HuggingFace currently has no clean, essay-only English dataset suitable for this study.
            Please manually collect {target_n} public-domain essays (recommended: Wikisource),
            each 150-300 words, and save as:

            - essay_01.md ... essay_{target_n:02d}.md

            Required front-matter fields:
            - id, genre=essay, source, license, length, note

            Suggested authors (public domain): Emerson, Bacon, Thoreau, Montaigne (English translation),
            Hazlitt, Stevenson.
            """
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-build source corpus from recommended datasets.")
    parser.add_argument("--target-per-genre", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=Path, default=Path("data/source_texts"))
    parser.add_argument("--min-words", type=int, default=150)
    parser.add_argument("--max-words", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Missing dependency: datasets. Install with:")
        print("  pip install 'datasets>=2.19.0'")
        return 1

    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    for genre in ("academic", "narrative", "poetry", "essay"):
        (out / genre).mkdir(parents=True, exist_ok=True)

    print("[build_source_corpus] Collecting academic ...")
    n_academic = pull_academic(
        n=args.target_per_genre,
        out_dir=out / "academic",
        seed=args.seed,
        min_words=args.min_words,
        max_words=args.max_words,
        overwrite=args.overwrite,
    )
    print(f"  academic: wrote {n_academic}/{args.target_per_genre}")

    print("[build_source_corpus] Collecting narrative ...")
    n_narrative = pull_narrative(
        n=args.target_per_genre,
        out_dir=out / "narrative",
        seed=args.seed + 1,
        min_words=args.min_words,
        max_words=args.max_words,
        overwrite=args.overwrite,
    )
    print(f"  narrative: wrote {n_narrative}/{args.target_per_genre}")

    print("[build_source_corpus] Collecting poetry ...")
    n_poetry = pull_poetry(
        n=args.target_per_genre,
        out_dir=out / "poetry",
        seed=args.seed + 2,
        min_words=args.min_words,
        max_words=args.max_words,
        overwrite=args.overwrite,
    )
    print(f"  poetry: wrote {n_poetry}/{args.target_per_genre}")

    write_essay_manifest(out, target_n=args.target_per_genre)
    print(
        "[build_source_corpus] essay is manual by design. See "
        f"{(out / 'essay' / 'MANUAL_COLLECTION.md').as_posix()}"
    )

    print("\nDone. Next steps:")
    print(f"  1) Manually fill essay_01..essay_{args.target_per_genre:02d} in data/source_texts/essay/")
    print("  2) Run: python -m src.run_experiment validate_corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
