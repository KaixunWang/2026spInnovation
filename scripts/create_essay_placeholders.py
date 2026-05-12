"""Create essay_01..essay_25 markdown placeholders for manual paste."""

from __future__ import annotations

from pathlib import Path

import yaml


def main() -> int:
    essay_dir = Path("data/source_texts/essay")
    essay_dir.mkdir(parents=True, exist_ok=True)

    manual = essay_dir / "MANUAL_COLLECTION.md"
    manual.write_text(
        "# Essay Manual Collection Guide\n\n"
        "请把你选好的英文 essay 正文粘贴到 essay_01.md ... essay_25.md 的 front-matter 下方。\n"
        "每篇建议 150-300 词（可在 80-380 词软区间内）。\n",
        encoding="utf-8",
    )

    for i in range(1, 26):
        item_id = f"essay_{i:02d}"
        out = essay_dir / f"{item_id}.md"
        fm = {
            "id": item_id,
            "genre": "essay",
            "source": "Wikisource/manual",
            "license": "public-domain",
            "length": 0,
            "note": "paste essay text below and update length",
        }
        out.write_text(
            "---\n"
            + yaml.safe_dump(fm, sort_keys=False, allow_unicode=False).strip()
            + "\n---\n"
            + "[PASTE ESSAY TEXT HERE]\n",
            encoding="utf-8",
        )
    print("Created essay placeholders: essay_01.md ... essay_25.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
