# AI Cross-Personality Rewriting

An experimental framework for studying how conflict between **stylistic-cognitive personas** affects **creativity** and **coherence** in text rewriting, grounded in Dual Process Theory, Yerkes-Dodson Law, and Shannon channel capacity.

See [docs/proposal.md](docs/proposal.md) for the full research proposal, **[docs/rubrics.md](docs/rubrics.md)** for the measurement model (auto metrics vs judge), and **[docs/experiment_workflow.md](docs/experiment_workflow.md)** for the step-by-step experiment pipeline (aligned with the v2 plan).

## Scope

- **Language**: English only (persona prompts, source texts, judge rubrics)
- **Personas**: prompt-level, 4 poles spanning `System1–System2` × `risk` axes
- **Texts**: 60 short literary pieces (150–300 words), 4 genres × 15

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Configure API keys
cp .env.example .env
# edit .env, fill in at least 2 API families (e.g. OPENAI_API_KEY + ANTHROPIC_API_KEY)

# 3. Build source texts (recommended semi-automatic path)
# auto-collect academic/narrative/poetry from HF datasets
python -m scripts.build_source_corpus --target-per-genre 15
# then manually add essays up to 15 (see data/source_texts/essay/MANUAL_COLLECTION.md)
# and validate:
python -m src.run_experiment validate_corpus --expected-per-genre 15

# 4a. Offline smoke (no API calls): verify all local components
python -m scripts.smoke_offline

# 4b. Online smoke (needs .env API keys; 1 source, 2 personas, 1 generator, no judge)
# If coord_scoring.backend is hf, complete step 5 first (axis_vectors.npz) or temporarily set backend: llm.
python -m src.run_experiment smoke

# 5. Deterministic Space-H coords (default backend: hf; run once after corpus is ready)
python scripts/generate_anchors.py
python scripts/build_axis_vectors.py
python scripts/recompute_coords.py
python scripts/validate_coords.py   # optional genre-direction checks

# 6. Full pipeline (after smoke test passes; see docs/experiment_workflow.md for merge/analyze order)
python -m src.run_experiment coord_reliability   # genre separation diagnostics (before main)
python -m src.run_experiment coord               # write coord_scores.jsonl
python -m src.run_experiment main --overwrite   # generate T0-T3
python -m src.run_experiment mechanism --overwrite
python -m src.run_experiment multihop --overwrite
python -m src.run_experiment metrics --with-embedding --inputs data/generated/main.jsonl data/generated/mechanism.jsonl data/generated/multihop.jsonl
python -m src.run_experiment judge --inputs data/generated/main.jsonl data/generated/mechanism.jsonl data/generated/multihop.jsonl
# merge judge into *_metrics.jsonl (see scripts/merge_judge_into_metrics.py), then:
python -m src.analyze --metric creativity_auto
```

## Project layout

```
.
├── configs/           # personalities, models, experiment params
├── data/
│   ├── source_texts/  # user-provided, 4 genres x 15 pieces
│   ├── seed_topics.yaml  # 50 neutral topics for Space-L
│   └── generated/     # model outputs, jsonl, auto-cached
├── docs/              # proposal + rubrics
├── results/           # analysis outputs
└── src/
    ├── generator.py          # provider-agnostic API wrapper
    ├── judge.py              # dual-judge scoring
    ├── personalities.py      # persona loader
    ├── embedding_space.py    # Space-L (learned embedding space)
    ├── conflict.py           # scalar d + directional Δ
    ├── baselines.py          # T0 identity / T1 same / T2 random
    ├── mechanism.py          # M1 joint / M2 sequential / M3 constrained
    ├── info_theory.py        # JSD / KL for style distributions
    ├── metrics/              # content, novelty, value, coherence, sentiment, structural
    ├── run_experiment.py     # CLI: smoke / main / mechanism / multihop / judge
    └── analyze.py            # regressions, diff, heatmaps, gradient field
```

## Hypotheses tested


| ID  | Hypothesis                                      | Test                              |
| --- | ----------------------------------------------- | --------------------------------- |
| H1  | Inverted-U: Creativity ~ d + d²                 | `metric ~ d + d²` mixed-effects   |
| H2  | Collapse threshold τ in Value                   | piecewise regression              |
| H3  | Directional gradient: Δ direction matters       | `metric ~ ΔS + ΔR + interactions` |
| H4  | Genre tolerance differs                         | `d × genre` interaction           |
| H5  | JSD(style) follows inverted-U too               | `metric ~ JSD + JSD²`             |
| H6  | Two-stage generation mitigates high-d collapse  | M1 vs M2/M3 at high d             |
| H7  | Inverted-U invariant across Space-H and Space-L | Procrustes + sign agreement       |


## Reproducibility

- All API calls are hashed and disk-cached; reruns do not incur cost.
- All intermediate outputs are JSONL with full metadata.
- Random seeds are set per-experiment; see `configs/experiment.yaml`.

## Caveats

This is research-grade code: CPU-OK (GPU optional for local NLI/embeddings), single-machine, focused on correctness and reproducibility over throughput.