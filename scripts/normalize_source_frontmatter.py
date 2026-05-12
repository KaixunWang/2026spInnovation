from pathlib import Path
import re
import yaml

roots = [
    Path("data/source_texts/academic"),
    Path("data/source_texts/narrative"),
    Path("data/source_texts/poetry"),
    Path("data/source_texts/essay"),
]
skip = {"readme.md", "manual_collection.md"}
keys = ["id", "genre", "source", "license", "length", "note"]
key_re = {k: re.compile(r"^" + k + r":\s*(.*)\s*$") for k in keys}

fixed = []
for root in roots:
    for p in root.glob("*.md"):
        if p.name.lower() in skip:
            continue
        t = p.read_text(encoding="utf-8").replace("\r\n", "\n")
        lines = t.split("\n")
        meta = {}
        idxs = []
        for i, ln in enumerate(lines[:40]):
            s = ln.strip()
            if s.startswith("## "):
                s = s[3:]
            if s == "---":
                continue
            for k, r in key_re.items():
                m = r.match(s)
                if m:
                    v = m.group(1).strip().strip('"')
                    if k == "length":
                        try:
                            v = int(float(v))
                        except Exception:
                            v = 0
                    meta[k] = v
                    idxs.append(i)
                    break
        if "id" not in meta or "genre" not in meta:
            continue
        start = max(idxs) + 1 if idxs else 0
        while start < len(lines) and lines[start].strip() != "":
            start += 1
        while start < len(lines) and lines[start].strip() == "":
            start += 1
        body = "\n".join(lines[start:]).strip()
        if not body:
            m = re.match(r"^---\n.*?\n---\n([\s\S]*)$", t)
            if m:
                body = m.group(1).strip()
        if (not meta.get("length")) and body:
            meta["length"] = len(re.findall(r"[A-Za-z']+", body))
        out_meta = {k: meta.get(k, "") for k in keys if k in meta}
        new = (
            "---\n"
            + yaml.safe_dump(out_meta, sort_keys=False, allow_unicode=False).strip()
            + "\n---\n"
            + body
            + "\n"
        )
        if new != t:
            p.write_text(new, encoding="utf-8")
            fixed.append(str(p))

print("NORMALIZED", len(fixed))
for x in fixed:
    print(x)
