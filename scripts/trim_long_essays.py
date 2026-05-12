import re
from pathlib import Path

import yaml

WORD_RE = re.compile(r"[A-Za-z']+")
files = ["essay_08.md", "essay_09.md", "essay_11.md", "essay_17.md", "essay_22.md"]
base = Path("data/source_texts/essay")

for fn in files:
    p = base / fn
    if not p.exists():
        print("missing", fn)
        continue
    t = p.read_text(encoding="utf-8")
    parts = t.split("---\n")
    if len(parts) < 3:
        print("skip malformed", fn)
        continue
    fm_txt = parts[1]
    body = ("---\n".join(parts[2:])).strip()
    fm = yaml.safe_load(fm_txt) or {}

    words = WORD_RE.findall(body)
    if len(words) <= 300:
        print("no_trim_needed", fn, len(words))
        continue

    # Keep first 300 lexical words while preserving rough punctuation spacing.
    kept_words = 0
    out_tokens: list[str] = []
    for tok in re.findall(r"[A-Za-z']+|[^\sA-Za-z']+", body):
        if WORD_RE.fullmatch(tok):
            kept_words += 1
            if kept_words > 300:
                break
        out_tokens.append(tok)
    new_body = " ".join(out_tokens).strip()
    fm["length"] = len(WORD_RE.findall(new_body))
    p.write_text(
        "---\n"
        + yaml.safe_dump(fm, sort_keys=False, allow_unicode=False).strip()
        + "\n---\n"
        + new_body
        + "\n",
        encoding="utf-8",
    )
    print("trimmed", fn, "to", fm["length"])
