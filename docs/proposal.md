# AI Cross-Personality Rewriting: A Study of Conflict-Driven Creativity

## Abstract

We investigate how **controlled conflict between stylistic-cognitive personas** affects the creativity and coherence of AI-rewritten text. We introduce *Cross-Personality Rewriting*: given a source text, we deliberately instruct a large language model to rewrite it under a persona whose stylistic-cognitive profile deliberately clashes with that of the source. We ground our study in Dual Process Theory, stylistic register theory, and the Novelty × Value definition of creativity, and we provide mechanistic explanations of the central inverted-U effect via Yerkes-Dodson Law and Shannon channel capacity. Through a quasi-causal design with four control arms (identity, same-persona, random-persona, cross-persona), replicated across a theory-driven 2D persona space and a data-driven embedding space, we characterise a creativity–collapse curve, locate an optimal conflict intensity, and identify a directional gradient indicating which *kinds* of conflict are most creatively productive. Within the scope of prompt-level personas over short English literary texts, our framework aims to surface a principled operational strategy for high-creativity AI writing.

## 1. Introduction

Text style transfer rewrites a text while preserving content (Mukherjee et al. 2023). Typical work operates *within* a homogeneous register — rewriting a formal text in a (different) formal style, or a sentimental passage in a (different) sentimental style. We ask the opposite: what if the rewriter's persona is deliberately *incongruent* with the source? Does such conflict suppress creativity (by breaking coherence), amplify it (by forcing unusual combinations), or both, depending on the conflict intensity?

Three observations motivate the study:

1. **Psychometric style literature** (Pennebaker 2011) shows that individual style is tightly coupled to personality; interposing a different persona is thus a principled perturbation.
2. **Creativity research on LLMs** (Wang et al. 2025 *Nature Human Behaviour*) finds that model outputs are more uniform than human writing, and that increasing temperature raises novelty but eventually causes semantic breakdown — a non-trivial curve.
3. **Persona-prompting studies** (Serapio-García et al. 2023) confirm that LLMs reliably adopt persona-consistent styles when prompted, giving us a controllable knob.

We therefore hypothesise an *inverted-U* between conflict intensity and creativity. This paper is the first to test that hypothesis systematically across four genres with four control arms and two alternative persona spaces.

## 2. Theoretical framework

Our design is anchored in five lines of prior work.

### 2.1 Dual Process Theory (cognitive anchor)

System 1 is fast, associative, affect-driven; System 2 is slow, analytic, rule-governed (Kahneman 2011; Evans & Stanovich 2013). We map the *rational ↔ emotional* axis of our persona space onto *S2 ↔ S1*, reframing the study as "System 1 – System 2 conflict in language generation". We use this as an **operational analogy** at the functional level; we do not claim neural equivalence.

### 2.2 Register / Style Shift (linguistic anchor)

Biber & Conrad (2009) characterise register variation along dimensions such as lexical density, syntactic complexity, and figurative risk. We instantiate an *adventurous ↔ conservative* axis as a high-risk vs low-risk stylistic register — operationally, the tightness of latent stylistic constraints.

### 2.3 Novelty × Value (creativity anchor)

Boden (2004) and Runco & Jaeger (2012) define creativity as the joint presence of **novelty** and **value**. We operationalise this as a multiplicative composite: a merely *different* text is not creative unless it also remains coherent and content-preserving. This directly answers the critique "you just made the text change, not better".

### 2.4 Yerkes-Dodson Law (mechanistic explanation A)

Arousal–performance curves are classically inverted-U, with the optimum shifting with task complexity (Yerkes & Dodson 1908; Diamond 2005). We map arousal onto conflict intensity *d* and task complexity onto genre. This yields a falsifiable corollary: **optimal d should vary across genres** (the basis for H4).

### 2.5 Channel capacity (mechanistic explanation B)

Treat rewriting as a channel p(Y | X, persona). Shannon's mutual information
`I(X; Y) = H(Y) − H(Y|X)` is bounded by channel capacity. When the persona distribution has full support over the source content, I is high; when it diverges too far, Y's support moves off X and I collapses. The rate-distortion curve is classically inverted-U with a unique optimum — predicting H5 from first principles.

**Triangulated prediction**: the inverted-U relationship emerges from three independent theoretical lenses (cognitive, psychological, information-theoretic). Finding it empirically would be converging evidence, not post-hoc rationalisation.

## 3. Hypotheses


| ID  | Statement                                                                              | Test                                             |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------ |
| H1  | Creativity is inverted-U in d                                                          | `β₂ < 0, p < 0.05` in `metric ~ β₁d + β₂d² + ε`  |
| H2  | A threshold τ exists beyond which Value drops sharply                                  | piecewise or changepoint fit                     |
| H3  | Creativity depends on direction Δ, not only magnitude d                                | main effects of ΔS, ΔR in directional regression |
| H4  | Genre tolerance: poetry > narrative > essay > academic                                 | `d × genre` interaction                          |
| H5  | JSD(P_source, P_target) also yields inverted-U                                         | `metric ~ JSD + JSD²`                            |
| H6  | Two-stage generation (M2/M3) reduces Value collapse at high d while preserving Novelty | contrast M1 vs M2/M3 in high-d subset            |
| H7  | H1 holds in both Space-H (theory-driven) and Space-L (learned)                         | sign agreement of β₂ across spaces               |


## 4. Design

### 4.1 Persona space

**Space-H (theoretically-driven)**. Two orthogonal axes:

- `S2 ↔ S1`: rational ↔ emotional
- `risk`: adventurous ↔ conservative

Four poles at the corners of the unit square. Each pole has a structured English prompt containing role, positive/negative stylistic markers, and forbidden elements (see `[configs/personalities.yaml](../configs/personalities.yaml)`).

**Space-L (data-driven)**. For each persona P, we generate N_ref ≥ 200 short passages on 50 neutral English seed topics; compute `v_P = mean(SBERT(texts_under_P))`; apply PCA to obtain latent axes; align with Space-H via Procrustes. This probes whether Space-H axes are *discoverable* from data (not merely stipulated).

### 4.2 Conflict

- **Scalar**: `d = ‖src_vec − tgt_vec‖₂ / d_max`, computed in both spaces (d_H, d_L)
- **Directional**: `Δ = tgt_vec − src_vec`, preserving sign; direction-sensitive analyses answer "which kind of conflict?"

### 4.3 Four-arm quasi-causal design

Each source text is rewritten under four conditions within-subject:


| Arm | Description                                        | Purpose                  |
| --- | -------------------------------------------------- | ------------------------ |
| T0  | Identity (verbatim copy)                           | Measurement noise floor  |
| T1  | Same-personality rewrite (target persona ≈ source) | Pure-rewrite effect      |
| T2  | Random-personality rewrite                         | Placebo for "any change" |
| T3  | Cross-personality rewrite (chosen by d bucket)     | Treatment                |


Causal contrasts:

- `Δ_creativity(d) = Creativity(T3 | d) − Creativity(T1)` — effect of *directed* conflict
- `placebo gap = Creativity(T3 | d) − Creativity(T2 | d)` — rules out "any change produces novelty"

### 4.4 Mechanism arm (H6)

At high d only, we compare three stage decompositions:

- **M1 — Joint**: a single prompt demanding both content preservation and stylistic projection
- **M2 — Sequential**: step 1 elicits a faithful paraphrase; step 2 imposes target persona on the paraphrase
- **M3 — Constrained**: M1 plus an explicit propositional checklist (extracted from step 1) in the prompt

### 4.5 Multi-hop

Evolutionary rewrite `source → P_A → P_B → ... → P_K` up to K = 5 hops, studying drift, attractors, and path dependence.

## 5. Data

- n = 60 English texts, 150–300 words each
- 4 genres × 15: `academic` (arXiv abstracts), `narrative` (Project Gutenberg excerpts), `essay` (public-domain essays), `poetry` (public-domain verse)
- Front-matter per file: `id, genre, source, license, length, note`

Power calculation: with n = 60 and the within-subject four-arm design, a mixed-effects test of the d² coefficient has power ≥ 0.8 at α = 0.05 for moderate effect sizes (|β₂| ≈ 0.2σ). Genre-level comparisons (H4) retain > 0.7 power per contrast.

## 6. Evaluation

All metrics normalised to [0, 1].

**Novelty (genre-baseline-normalised)**. The key methodological correction: surface novelty (distinct-n, embedding distance) is confounded by genre-intrinsic diversity. We therefore compute per-genre baselines
`baseline_g = mean_{source in g} indicator(source)` and report each generated item's *relative* novelty
`distinct2_rel = distinct2(gen) / baseline_g.distinct2_source`
`embdist_rel = emb_dist(src, gen) / baseline_g.avg_intra_genre_emb_dist`
or equivalently as within-genre z-scores.

- Automatic: `distinct_rel`, `novel_ngram_rel`, `embdist_rel`
- Judge: TTCW-inspired rubric (Novelty, Surprise, Imagery), each 1–5

**Value**. Must co-occur with novelty to count as creativity.

- Entailment: NLI(src → gen), using `microsoft/deberta-v3-large-mnli`
- Coherence: judge rubric (fluency, consistency, logical completeness), 1–5 each
- `Value = geometric_mean(Entailment, Coherence)`

**Composite**.

- Primary: `Creativity = Novelty_norm × Value`
- Alternative: `Utility = Novelty − λ(1 − Value)`, λ ∈ {0.5, 1, 2} as sensitivity

**Auxiliary**: sentiment shift (cardiffnlp/twitter-roberta-base-sentiment-latest), structural remix (Kendall τ on sentence alignment + normalised Levenshtein), perplexity (gpt2 locally).

**Information-theoretic proxy**. `JSD_style ≈ JSD(unigram(P_source_ref), unigram(P_target_ref))` from Space-L reference pools.

**Judge anti-bias**: generator ≠ judge family; dual judges → ICC; both pairwise and absolute scoring; options shuffled; provenance masked; genre explicitly disclosed so raters calibrate within-genre.

## 7. Analysis

1. **Main regression (H1, H4, H7)**.
  `metric ~ d + d² + genre + target_persona + condition + (1|source_id) + (1|model)`, run separately in Space-H and Space-L.
2. **Directional regression (H3)**.
  `metric ~ ΔS + ΔR + ΔS² + ΔR² + ΔS·ΔR + |Δ| + controls` and a 2D gradient-field plot.
3. **Causal contrasts**. Δ_creativity(d) and placebo gap; bootstrap CIs.
4. **Information-theoretic check (H5)**. `metric ~ JSD + JSD²`.
5. **Mechanism (H6)**. In the high-d subset: `metric ~ condition_M + controls`.
6. **Multi-hop**. Trajectory in Space-H/L; drift slopes; attractor detection via repeated final-position coordinates.

Primary figures: (a) inverted-U curve in d (paired H and L panels); (b) 2D gradient field of creativity over Δ; (c) 4×4 persona-pair heatmap with significance; (d) multi-hop trajectories; (e) per-genre overlay.

## 8. Scope and limitations

Our findings, if they obtain, apply *within* the following scope:

- **Prompt-level** personas (not fine-tuned / RLHF personas — weight-level effects may differ)
- **Short English literary texts** (not long-form, non-literary, or non-English text)
- **Stylistic-cognitive proxy axes** (not psychometric models like Big Five; Space-H captures a 2D slice, not the full personality manifold)
- **Operational analogy** of Dual Process (no neural-level claim)

External-validity safeguards: three paraphrased prompts per persona (prompt-paraphrase robustness), ≥ 2 model families (model robustness), genre × persona sensitivity analyses, PCA dimensionality sweep (k ∈ {2, 3, 4}), and λ sweep for the Utility composite.

## 9. Timeline (4 weeks)

- **W1**: scaffolding, personas, models wiring, smoke test on 5–10 texts
- **W2**: full 60-text corpus; Space-L construction; main matrix (T0–T3) × 2 generators
- **W3**: metrics (content / novelty / value / coherence / sentiment / structural); mechanism + multi-hop; all regressions + figures
- **W4**: proposal/paper draft; sensitivity and robustness sweeps; reproducibility package

## 10. Deliverables

1. Open-source Python framework: `src/`
2. Pre-computed JSONL of all generations + metrics: `data/generated/` + `results/`
3. Reproducibility bundle: configs, seeds, cached API responses (redacted)
4. Paper-ready figures and tables

## References

- Biber, D. & Conrad, S. (2009). *Register, Genre, and Style.* CUP.
- Boden, M. A. (2004). *The Creative Mind.* Routledge.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory.* Wiley.
- Diamond, D. M. et al. (2005). The temporal dynamics model of emotional memory processing. *Neural Plasticity*.
- Evans, J. St. B. T. & Stanovich, K. E. (2013). Dual-process theories of higher cognition. *Perspectives on Psychological Science*.
- Fu, Z. et al. (2018). Style transfer in text: exploration and evaluation. *AAAI*.
- Kahneman, D. (2011). *Thinking, Fast and Slow.* FSG.
- Li, K. et al. (2025). LLM-based creative writing evaluation with reference scoring. *arXiv*.
- Mukherjee, S. et al. (2023). Text style transfer: survey. *TACL*.
- Pennebaker, J. W. (2011). *The Secret Life of Pronouns.* Bloomsbury.
- Runco, M. A. & Jaeger, G. J. (2012). The standard definition of creativity. *Creativity Research Journal*.
- Serapio-García, G. et al. (2023). Personality traits in large language models. *arXiv*.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*.
- Wang, Y. et al. (2025). Divergent creativity in humans and LLMs. *Nature Human Behaviour*.
- Yerkes, R. M. & Dodson, J. D. (1908). The relation of strength of stimulus to rapidity of habit-formation. *J. Comp. Neurol. Psychol.*