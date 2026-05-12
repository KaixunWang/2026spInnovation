"""Build data/axis_vectors.npz from data/anchors.json using StyleDistance embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import PROJECT_ROOT
from src.text_style_coords import DEFAULT_STYLEDIST_MODEL, _normalize


def _mean_embed(model, texts: list[str]) -> np.ndarray:
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.mean(np.asarray(emb, dtype=float), axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchors", type=Path, default=PROJECT_ROOT / "data" / "anchors.json")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "axis_vectors.npz")
    p.add_argument("--model", type=str, default=DEFAULT_STYLEDIST_MODEL)
    args = p.parse_args()

    if not args.anchors.exists():
        print(f"Missing {args.anchors}; run scripts/generate_anchors.py first", file=sys.stderr)
        return 1

    data = json.loads(args.anchors.read_text(encoding="utf-8"))
    for k in ("R_neg", "R_pos", "S_neg", "S_pos"):
        if k not in data or not isinstance(data[k], list) or len(data[k]) != 8:
            print(f"anchors.json must have {k} as list of 8 strings", file=sys.stderr)
            return 1

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(args.model)

    r_neg_m = _mean_embed(st, data["R_neg"])
    r_pos_m = _mean_embed(st, data["R_pos"])
    r_axis = _normalize(r_pos_m - r_neg_m)

    s_neg_m = _mean_embed(st, data["S_neg"])
    s_pos_m = _mean_embed(st, data["S_pos"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        R_axis=r_axis,
        R_neg_mean=r_neg_m,
        R_pos_mean=r_pos_m,
        S_neg_mean=s_neg_m,
        S_pos_mean=s_pos_m,
    )
    print(f"[build_axis_vectors] wrote {args.output} (R_axis dim={len(r_axis)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
