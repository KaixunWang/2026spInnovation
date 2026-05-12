"""Offline smoke test: exercises every non-API part of the framework.

Run:
    python -m scripts.smoke_offline

This does NOT call any LLM. It verifies:
  * config loaders
  * persona / corpus loading
  * conflict math
  * novelty / structural metrics
  * info-theory divergences
  * JSONL IO

Online smoke (with API keys in .env):
    python -m src.run_experiment smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    print("=== 1. Config loaders ===")
    from src.config_loader import load_experiment_config, load_models_config

    exp = load_experiment_config()
    mdl = load_models_config()
    print(f"  experiment seed={exp['seed']} main.n_repeat={exp['main']['n_repeat']}")
    print(f"  models: {[m['name'] for m in mdl['models']]}")

    print("\n=== 2. Personas ===")
    from src.personalities import load_personas

    personas = load_personas()
    print(f"  {len(personas)} personas: {personas.names}")
    print(f"  vectors:\n{personas.vectors}")
    src_vec = np.array([0.5, -0.2])
    near = personas.closest_to(src_vec)
    far = personas.farthest_from(src_vec)
    print(f"  nearest to (0.5, -0.2): {near.name}; farthest: {far.name}")

    print("\n=== 3. Corpus ===")
    from src.corpus import load_sources, validate_corpus

    srcs = load_sources()
    print(f"  loaded {len(srcs)} source(s): {[s.id for s in srcs]}")
    ok, errs = validate_corpus(expected_per_genre=1)
    print(f"  validate_corpus: ok={ok} (errors: {errs or 'none'})")

    print("\n=== 4. Conflict math ===")
    from src.conflict import compute_conflict, bucketize

    c = compute_conflict(src_vec, near.vector, space="H")
    print(f"  nearest d={c.d:.3f} Δ={c.delta.tolist()}")
    c2 = compute_conflict(src_vec, far.vector, space="H")
    print(f"  farthest d={c2.d:.3f} Δ={c2.delta.tolist()}")
    ds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"  bucketize({ds}) = {bucketize(ds)}")

    print("\n=== 5. Novelty metrics (no SBERT) ===")
    from src.metrics.novelty import (
        compute_genre_baselines,
        distinct_n,
        normalise_novelty,
        novel_ngram_ratio,
    )

    by_genre: dict[str, list[str]] = {}
    for s in srcs:
        by_genre.setdefault(s.genre, []).append(s.text)
    base = compute_genre_baselines(by_genre, sbert_model=None)
    for g, b in base.items():
        print(f"  {g}: distinct2={b.distinct2_mean:.3f} (n={b.n})")

    src = srcs[0]
    pseudo_gen = src.text[::-1][:300]  # reversed text as a nonsense 'rewrite'
    nov = normalise_novelty(src.text, pseudo_gen, base[src.genre], sbert_model=None)
    print(f"  src={src.id} nonsense-novelty: d2_rel={nov.distinct2_rel:.2f}, "
          f"novel_ngram_rel={nov.novel_ngram_rel:.2f}, auto={nov.auto_combined:.2f}")

    print("\n=== 6. Structural metrics ===")
    from src.metrics.structural import sentence_kendall_tau, normalised_levenshtein

    a = "The cat sat on the mat. The dog ran. Birds sang."
    b = "Birds sang. The dog ran. The cat sat on the mat."
    print(f"  kendall_tau(a, shuffled) = {sentence_kendall_tau(a, b):.3f}")
    print(f"  lev(a, a) = {normalised_levenshtein(a, a):.3f}")
    print(f"  lev(a, 'completely different prose here') = "
          f"{normalised_levenshtein(a, 'completely different prose here'):.3f}")

    print("\n=== 7. Info-theory divergences ===")
    from src.info_theory import build_style_distribution, jsd, kl

    p1, _ = build_style_distribution([srcs[0].text])
    p2, _ = build_style_distribution([srcs[-1].text])
    print(f"  jsd(src[0], src[-1]) = {jsd(p1, p2):.4f}")
    print(f"  kl (src[0], src[-1]) = {kl(p1, p2):.4f}")

    print("\n=== 8. Baselines (T0 identity only, no API) ===")
    from src.baselines import arm_T0

    r = arm_T0(srcs[0], src_vec)
    print(f"  T0 record: condition={r.condition} d_H={r.d_H} text_len={len(r.text.split())}")

    print("\n=== 9. JSONL IO round-trip ===")
    from src.io_utils import read_jsonl, write_jsonl

    tmp = Path(".cache/_smoke_tmp.jsonl")
    n = write_jsonl(tmp, [{"a": 1, "b": [1, 2, 3]}, {"a": 2, "b": []}])
    back = list(read_jsonl(tmp))
    assert n == 2 and back[0]["a"] == 1, back
    tmp.unlink(missing_ok=True)
    print(f"  wrote/read {n} rows; ok")

    print("\n=== 10. CLI wiring ===")
    from src.run_experiment import build_parser

    p = build_parser()
    cmds = list(p._subparsers._group_actions[0].choices.keys())  # type: ignore[attr-defined]
    print(f"  subcommands: {cmds}")

    print("\n*** Offline smoke OK. All non-API components functional. ***")
    print("Next step: fill .env with API keys and run:")
    print("    python -m src.run_experiment smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
