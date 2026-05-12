"""Fetch essays from English Wikisource Category:Essays.

This script fills `data/source_texts/essay/essay_XX.md` with 150-300-word
excerpts and YAML front-matter compatible with this project.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import httpx
import yaml


API = "https://en.wikisource.org/w/api.php"
WORD_RE = re.compile(r"[A-Za-z']+")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def strip_html(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\\s*/?>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\\[[0-9]+\\]", " ", text)  # citation marks
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def truncate_window(text: str, min_words: int, max_words: int) -> str | None:
    words = text.split()
    if len(words) < min_words:
        return None
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def category_members(client: httpx.Client, category: str) -> list[str]:
    out: list[str] = []
    cmcontinue: str | None = None
    while True:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "page",
            "cmlimit": "500",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        r = client.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out.extend(
            [m["title"] for m in data.get("query", {}).get("categorymembers", []) if m.get("ns") == 0]
        )
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return out


def fetch_page_text(client: httpx.Client, title: str) -> str | None:
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "disabletoc": "1",
    }
    r = client.get(API, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    html_blob = data.get("parse", {}).get("text", {}).get("*")
    if not isinstance(html_blob, str):
        return None
    return strip_html(html_blob)


def write_essay(path: Path, idx: int, title: str, text: str) -> None:
    essay_id = f"essay_{idx:02d}"
    fm = {
        "id": essay_id,
        "genre": "essay",
        "source": f"https://en.wikisource.org/wiki/{title.replace(' ', '_')}",
        "license": "public-domain",
        "length": word_count(text),
        "note": f"auto-extracted from Wikisource Category:Essays; title={title}",
    }
    body = (
        "---\n"
        + yaml.safe_dump(fm, sort_keys=False, allow_unicode=False).strip()
        + "\n---\n"
        + text
        + "\n"
    )
    path.write_text(body, encoding="utf-8")


def existing_indices(essay_dir: Path) -> set[int]:
    out: set[int] = set()
    for p in essay_dir.glob("essay_*.md"):
        m = re.match(r"essay_(\d+)\.md$", p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch essays from Wikisource Category:Essays")
    parser.add_argument("--target-per-genre", type=int, default=25)
    parser.add_argument("--min-words", type=int, default=150)
    parser.add_argument("--max-words", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--essay-dir", type=Path, default=Path("data/source_texts/essay"))
    args = parser.parse_args(argv)

    args.essay_dir.mkdir(parents=True, exist_ok=True)
    present = existing_indices(args.essay_dir)
    needed = [i for i in range(1, args.target_per_genre + 1) if args.overwrite or i not in present]
    if not needed:
        print(f"essay already complete: {len(present)}/{args.target_per_genre}")
        return 0

    with httpx.Client(follow_redirects=True, headers={"User-Agent": "InnovationResearchBot/0.1"}) as client:
        titles = category_members(client, "Essays")
        print(f"fetched category members: {len(titles)}")
        used_fp: set[str] = set()
        wrote = 0
        ni = 0
        for title in titles:
            if wrote >= len(needed):
                break
            text = fetch_page_text(client, title)
            if not text:
                continue
            excerpt = truncate_window(text, args.min_words, args.max_words)
            if not excerpt:
                continue
            fp = " ".join(WORD_RE.findall(excerpt.lower())[:180])
            if fp in used_fp:
                continue
            target_idx = needed[ni]
            path = args.essay_dir / f"essay_{target_idx:02d}.md"
            write_essay(path, target_idx, title, excerpt)
            used_fp.add(fp)
            wrote += 1
            ni += 1
            print(f"wrote {path.name} <- {title}")

    if wrote < len(needed):
        print(f"warning: only wrote {wrote}/{len(needed)} needed essays")
        return 1
    print(f"done: wrote {wrote} essay files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
