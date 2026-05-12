"""Generate four poles of anchor passages via LLM; write data/anchors.json.

Each pole: 8 English passages, ~100 words, matching stylistic instructions.
Requires .env API keys and a model in configs/models.yaml.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import PROJECT_ROOT, get_role_models
from src.generator import generate

KEYS = ("S_neg", "S_pos", "R_neg", "R_pos")

PROMPTS = {
    "S_neg": """Generate exactly 8 short English passages for a style anchor corpus.
Each passage must be ~100 words (80-120). Output STRICT JSON only:
{"passages": ["...", ...]}
Style (S-, emotional / sensory pole): first person, overt emotion, rich sensory detail,
fragmented syntax, no logical connectives like therefore/thus, heavy adjectives and exclamations.
Topics can vary; no titles or numbering inside strings.""",
    "S_pos": """Generate exactly 8 short English passages for a style anchor corpus.
Each passage must be ~100 words (80-120). Output STRICT JSON only:
{"passages": ["...", ...]}
Style (S+, rational pole): fully depersonalised, no first person, passive voice preferred,
only logical connectives (therefore, thus, consequently), verifiable claims only, zero emotional language.""",
    "R_neg": """Generate exactly 8 short English passages for a style anchor corpus.
Each passage must be ~100 words (80-120). Output STRICT JSON only:
{"passages": ["...", ...]}
Style (R-, conservative pole): only high-frequency words, simple repeated sentence patterns,
no metaphors, no experimental phrasing, predictable structure like textbook examples.""",
    "R_pos": """Generate exactly 8 short English passages for a style anchor corpus.
Each passage must be ~100 words (80-120). Output STRICT JSON only:
{"passages": ["...", ...]}
Style (R+, adventurous pole): mix fragments and long sentences, occasional rare or coined words,
dense metaphor and image leaps, demands active interpretation, resists transparent exposition.""",
}


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _validate_passages(passages: list, key: str) -> list[str]:
    if not isinstance(passages, list) or len(passages) != 8:
        raise ValueError(f"{key}: expected list of 8 strings, got {passages!r}")
    out: list[str] = []
    for i, p in enumerate(passages):
        if not isinstance(p, str) or not p.strip():
            raise ValueError(f"{key}[{i}] invalid")
        wc = len(p.split())
        if wc < 50 or wc > 200:
            print(f"[warn] {key}[{i}] word count {wc} outside 50-200", file=sys.stderr)
        out.append(p.strip())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "anchors.json",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from models.yaml (default: first generators role entry)",
    )
    args = p.parse_args()
    model = args.model or get_role_models("generators")[0]

    out_obj: dict[str, list[str]] = {}
    for key in KEYS:
        print(f"[anchors] generating {key} with {model} ...", flush=True)
        res = generate(
            model,
            system="You output only valid JSON objects. No markdown fences.",
            user=PROMPTS[key],
            temperature=0.7,
            max_tokens=4000,
        )
        parsed = _extract_json_object(res.text)
        if parsed is None:
            print(f"[anchors] failed to parse JSON for {key}:\n{res.text[:800]}", file=sys.stderr)
            return 1
        passages = parsed.get("passages")
        try:
            out_obj[key] = _validate_passages(passages, key)
        except ValueError as e:
            print(f"[anchors] {e}", file=sys.stderr)
            return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[anchors] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
