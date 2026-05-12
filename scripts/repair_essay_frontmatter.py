from pathlib import Path
import re
import yaml

base = Path("data/source_texts/essay")
fixed = []

for p in base.glob("essay_*.md"):
    t = p.read_text(encoding="utf-8")
    if re.match(r"^---\n.*?\n---\n", t, flags=re.S):
        continue
    body = t.strip() or "[PASTE ESSAY TEXT HERE]"
    m = re.search(r"(\d+)", p.stem)
    n = int(m.group(1)) if m else 0
    fm = {
        "id": f"essay_{n:02d}",
        "genre": "essay",
        "source": "Wikisource/manual",
        "license": "public-domain",
        "length": len(re.findall(r"[A-Za-z']+", body)),
        "note": "auto-recovered front-matter",
    }
    p.write_text(
        "---\n"
        + yaml.safe_dump(fm, sort_keys=False, allow_unicode=False).strip()
        + "\n---\n"
        + body
        + "\n",
        encoding="utf-8",
    )
    fixed.append(str(p))

print("fixed", len(fixed))
for x in fixed:
    print(x)
