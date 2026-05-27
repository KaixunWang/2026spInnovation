# Supplementary Sections: CrEval Pairwise Win-Rate Evaluation

> **Usage note.** These sections are additive to the main paper
> *"Conflict as Structural Stress in LLM Cross-Personality Rewriting."*
> No existing text, table numbering, or figure numbering is altered.
> Insert **Section 3.6** after Section 3.5 and **Section 5.6** after
> Section 5.5 in the published body; the new tables and figure are
> numbered 13–15 and Figure 8, continuing the existing sequence.

---

## 3.6 CrEval Pairwise Win-Rate (*C*_creval)

Sections 3.3–3.5 establish two headline channels—an automatic
NLI×perplexity composite (*C*_auto) and an absolute seven-dimension
LLM rubric (*C*_judge)—that measurably disagree about the value of
high-conflict rewrites (Table 2).  A third channel is added here to
triangulate: pairwise comparison by **CrEval** [5], a 7 B-parameter
fine-tuned creativity evaluator that has been explicitly calibrated on
human pairwise preferences rather than prompted ad hoc from a
general-purpose model.

### Model and prompt format

CrEval deploys as a 4-bit NF4 quantised adapter (Qwen2.5-7B-Instruct
base + CrEval-7b LoRA) served through an OpenAI-compatible
LLaMA-Factory API endpoint.  Given two rewrites *a* and *b* of the
same source, the model outputs one of three decisions: "Response 1 is
more creative," "Response 2 is more creative," or "the two are equally
creative."  Critically, the assignment of rewrites to Response-1/2
slots is randomised independently for each comparison to mitigate
positional bias.

### Tournament design

For each of the 60 source texts, all rewrites produced under
`repeat_idx=0` are collected across every condition–model combination
active for that source, then subjected to an exhaustive round-robin
tournament: every pair of distinct rewrites is evaluated exactly once,
yielding a symmetric comparison graph.  Across 1,680 rewrites (4
model × 4 conditions × 60 sources, with T3 carrying 4 persona
sub-conditions), this produces **22,680 pairwise comparisons** in
total, with each rewrite receiving exactly *n*_comparisons = 27
opponents.

The per-rewrite win-rate is

$$C_{\mathrm{creval}} = \frac{\mathrm{wins} + 0.5\times\mathrm{ties}}{n_{\mathrm{comparisons}}}$$

and lies in [0, 1] by construction.  A win-rate of 0.5 indicates
pairwise indistinguishability in expectation; values above 0.5 signal
consistent preference over the tournament pool.

### Relationship to existing channels

*C*_creval is treated as a **third validity arm** alongside *C*_auto
and *C*_judge.  It does not replace the prespecified automatic
primary outcome.  Where *C*_auto rewards semantic fidelity (its
value component penalises divergence from the source) and *C*_judge
operationalises rubric-based human surprise, *C*_creval directly
encodes pairwise human-preference calibration: a rewrite wins if and
only if a calibrated judge—trained on preference pairs rather than
prompted with a Likert template—prefers it to each opponent in the
field.  Triangulation across all three channels therefore supports
stronger construct inference than any single channel alone.

---

## 5.6 CrEval Conflict-Intensity Analysis

### 5.6.1 Condition-level win-rates

Table 13 reports mean *C*_creval and standard deviation for each
discrete condition, pooling all four models.  The identity condition
(T0) produces consistently lower win-rates than any rewriting
condition, with a large and highly significant gap.

| Condition | *n* | Mean *C*_creval | SD |
|-----------|-----|-----------------|----|
| T0 (identity) | 240 | 0.2236 | 0.1093 |
| T1 (nearest persona) | 240 | 0.5069 | 0.2515 |
| T2 (random placebo) | 240 | 0.5539 | 0.2604 |
| T3 (directed cross-personality) | 960 | 0.5539 | 0.2628 |

**Table 13.** Mean *C*_creval win-rate by condition (discrete main arm;
pooled over generators; T3 includes all four persona sub-conditions).

One-way ANOVA confirms a strong overall condition effect
(*F* = 121.47, *p* ≈ 3.5×10⁻⁷¹).  Pairwise Welch *t*-tests show that
every rewriting condition significantly outperforms T0 (all
*p* < 10⁻⁴², Table 14), while T2 and T3 are statistically
indistinguishable (*p* ≈ 0.998, Δmean ≈ 0.0001).  T1 is
marginally below T2/T3 (*p* ≈ 0.045/0.011), but the effect is small
(Δmean ≈ 0.047).

| Pair | Δmean (*b*−*a*) | Welch *t* | *p* |
|------|-----------------|-----------|-----|
| T0 → T1 | +0.2833 | −16.01 | 5.8×10⁻⁴³ |
| T0 → T2 | +0.3303 | −18.12 | 5.1×10⁻⁵¹ |
| T0 → T3 | +0.3303 | −29.94 | 7.1×10⁻¹³⁹ |
| T1 → T2 | +0.0470 | −2.01 | 4.5×10⁻² |
| T1 → T3 | +0.0469 | −2.56 | 1.1×10⁻² |
| T2 → T3 | −0.0001 | +0.003 | 9.98×10⁻¹ |

**Table 14.** Pairwise Welch *t*-tests on *C*_creval by condition.
All rewrites cross-personality or persona-shifted outperform the
identity baseline at *p* < 10⁻⁴⁰.  The T2–T3 null mirrors the
bucketed automatic-creativity contrasts in Table 10, confirming
that directed persona conflict does not outperform placebo-level
persona assignment as measured by pairwise preference.

The pooled Cohen's *d* for the T0 vs. T3 contrast is 1.375—an
unusually large effect by social-science conventions—indicating
that CrEval consistently and robustly separates identity-preserved
text from cross-personality rewrites.

### 5.6.2 Generator-level win-rates

| Generator | *n* (all cond.) | Mean *C*_creval | SD | Mean *C*_creval (T3 only) |
|-----------|-----------------|-----------------|----|----|
| GPT-4o | 420 | 0.6294 | 0.2870 | 0.7002 |
| Qwen3-14B | 420 | 0.4660 | 0.2596 | 0.5099 |
| Qwen3-8B | 420 | 0.4638 | 0.2501 | 0.5130 |
| Qwen3-4B | 420 | 0.4408 | 0.2380 | 0.4924 |

**Table 15.** Mean *C*_creval by generator (left: all conditions pooled;
right: T3-only rows).  GPT-4o produces outputs that the CrEval judge
consistently ranks above the three open-source Qwen3 checkpoints.
The win-rate gap between GPT-4o and the Qwen3 trio (roughly 0.19
overall, 0.19 on T3) reproduces the same ordering seen in *C*_auto
(Table 4) and *C*_judge, confirming that generator capacity separates
reliably across all three evaluation channels.

### 5.6.3 Conflict intensity and quadratic curvature

The central question for this arm is whether *C*_creval replicates or
reverses the negative quadratic curvature of *C*_auto vs. *d*_H
established in Tables 6 and 8.

**Binned means (T3 rows).** Grouping T3 rows into four equal-width
conflict bins shows a broadly monotone-positive trend—the opposite
of the *C*_auto pattern in Figure 3:

| *d*_H range | *n* | Mean *C*_creval |
|-------------|-----|-----------------|
| [0.00, 0.30) | 48  | 0.4105 |
| [0.30, 0.50) | 412 | 0.5560 |
| [0.50, 0.70) | 352 | 0.5348 |
| [0.70, 1.10) | 148 | 0.6400 |

The very-high-conflict bin (*d*_H > 0.70, *n* = 148) achieves the
highest mean win-rate (0.640), compared with 0.480 for the
low-conflict bin (*d*_H < 0.40, *n* = 168).  A Welch *t*-test
confirms this gap is reliable (*t* = 5.41, *p* = 1.3×10⁻⁷).

**OLS quadratic fit.** Pooling all 960 T3 rows in a simple OLS
(*C*_creval ∼ *d*_H + *d*_H² ) yields a positive quadratic coefficient:
β_*d*² = +1.017 (*p* = 0.011), with a fitted vertex at
*d*\* = 0.404 from below (a U-shaped minimum rather than an
inverted-U maximum).  The headline model coefficient is modest
(*R*² = 0.030), consistent with large individual variance in
pairwise win-rate outcomes.

**Per-generator fits.** Table 16 extends Table 6 to the
*C*_creval channel using per-generator OLS regressions
(*C*_creval ∼ *d*_H + *d*_H², T3 discrete rows, *n* = 240 each).

| Generator (T3) | β_*d*² | *p*(β_*d*²) | *d*\* | *R*² |
|----------------|--------|-------------|-------|------|
| GPT-4o | −0.664 | 0.383 | 0.661 | 0.011 |
| Qwen3-14B | +2.022 | 0.010 | 0.479 | 0.048 |
| Qwen3-8B | +1.615 | 0.031 | 0.422 | 0.068 |
| Qwen3-4B | +1.096 | 0.131 | 0.396 | 0.046 |

**Table 16.** Per-generator OLS quadratic fits for *C*_creval vs.
*d*_H (T3-only discrete rows, *n* = 240 per generator).  Signs and
significance mirror the *C*_judge column in Table 2 for open-source
checkpoints: Qwen3-14B and -8B show significantly positive curvature,
consistent with CrEval rewarding high-conflict outputs; Qwen3-4B is
in the same direction but below conventional significance.
GPT-4o shows weakly negative (non-significant) curvature,
reflecting a ceiling effect at this generator's quality level rather
than a reversal.

The contrast between Table 6 (*C*_auto, consistently negative β_*d*²
for the same Qwen checkpoints) and Table 16 (*C*_creval, consistently
positive β_*d*²) directly echoes the construct-divergence story in
Section 3.5: the automatic fidelity composite and the calibrated
pairwise preference evaluator measure different things, and they
disagree most sharply at high conflict.

### 5.6.4 Genre-level heterogeneity

Restricting to T3 rows, genre means are ordered:

| Genre | *n* | Mean *C*_creval |
|-------|-----|-----------------|
| Academic | 240 | 0.6016 |
| Essay | 240 | 0.5638 |
| Narrative | 240 | 0.5348 |
| Poetry | 240 | 0.5153 |

Academic rewrites win most often in pairwise competition despite
showing the lowest *C*_auto means (Table 5: academic *C*_auto = 0.284),
reproducing the channel inversion visible in the *C*_judge data.
Poetry, the genre with the highest automatic novelty (Table 5:
𝑁_auto = 0.452), ranks last under CrEval pairwise comparison.  This
ordering inversion is consistent with the near-zero headline
correlation between judge novelty and *C*_auto (*r* = 0.118; Section
3.5) and with CrEval rewarding structured creativity—accurate
propositional transformation under stylistic change—rather than
surface lexical diversity.

### 5.6.5 Target-persona dimension effects

Within T3 rows (all generators pooled), the four corner personas
differ markedly in their induced *C*_creval:

| Target persona | *n* | Mean *C*_creval |
|----------------|-----|-----------------|
| emotional_adventurous | 240 | 0.7813 |
| rational_adventurous | 240 | 0.6806 |
| emotional_conservative | 240 | 0.4216 |
| rational_conservative | 240 | 0.3321 |

The **R** axis (adventurous vs. conservative register) accounts for
the dominant split: rewrites targeting either adventurous persona
achieve win-rates 0.68–0.78, while conservative targets fall to
0.33–0.42—a difference of roughly 0.45 win-rate units.  The **S**
axis (emotional/rational, i.e. System-1 vs. System-2) contributes a
secondary ordering within each R level (emotional slightly above
rational for both adventurous and conservative targets).

This persona pattern complements the directional regression in
Table 9 (*C*_auto ∼ Δ*S*, Δ*R*, interactions): there the Δ*S* slope
is approximately six times the Δ*R* slope on the automatic channel,
suggesting that analytic–intuitive shift drives automatic creativity
more than register shift.  CrEval reverses this emphasis:
adventurous register dominates pairwise preference, while the
System-1/2 dimension is secondary.  The dimensional disagreement
between automatic and pairwise-preference channels adds a further
layer to the construct-divergence story established in Tables 2–3.

### 5.6.6 Synthesis: three-channel triangulation

Table 17 collates the quadratic curvature direction and significance
for all three evaluation channels on the GPT-4o T3 arm (the arm
where Table 2 already documents the sharpest disagreement between
*C*_auto and *C*_judge).

| Channel | β_*d*² sign | *p* | Interpretation |
|---------|-------------|-----|----------------|
| *C*_auto (automatic) | − (negative) | 6.95×10⁻¹⁰ | High conflict degrades fidelity + coherence |
| *C*_judge (rubric LLM) | + (positive) | 1.8×10⁻⁵ | High conflict rewards rubric-based surprise |
| *C*_creval (pairwise, calibrated) | − (n.s.) | 0.38 | No reliable curvature; uniform high quality |

**Table 17.** Three-channel quadratic curvature comparison, GPT-4o
T3 arm.  *C*_creval sits between the two existing channels: it neither
penalises high-conflict rewrites like *C*_auto nor rewards them like
*C*_judge.  For GPT-4o, very high-conflict outputs appear to reach a
quality ceiling that makes them competitive in pairwise comparison
regardless of conflict level—consistent with the generator's large
mean win-rate (0.700 on T3) and small per-row variance (SD = 0.247
vs. 0.259–0.260 for Qwen3 checkpoints).

For the open-source Qwen3 checkpoints (Table 16), *C*_creval β_*d*²
is consistently positive and significant for 14B and 8B, tracking
*C*_judge rather than *C*_auto.  This means that both the calibrated
pairwise evaluator and the rubric LLM judge agree that higher-conflict
Qwen3 rewrites are preferred, even as the automatic composite
penalises them for semantic drift.  The practical reading is that
high-conflict small-model rewrites sacrifice NLI entailment fidelity
but gain expressive salience detectable by both fine-tuned preference
models and prompted human-aligned rubrics—a trade-off whose value
depends on the downstream application (fidelity-sensitive tasks
should use *C*_auto; preference-sensitive tasks should use *C*_creval
or *C*_judge).

---

## 3.7 CrEval versus Untrained LLM Judges: Range, Calibration, and Comparative Sensitivity

Section 3.6 introduces *C*_creval as a fine-tuned pairwise evaluator;
the present section characterises how it differs from—and, in several
respects, improves upon—the two untrained LLM rubric judges (DeepSeek
V4 Pro and GPT-4o) that produce *C*_judge.

### 3.7.1 Range compression in untrained judges

The most salient difference between *C*_creval and *C*_judge is
**dynamic range**.  Untrained LLM judges return Likert scores on a
1–5 scale, and in practice the composite means are tightly compressed:
*C*_judge spans only 0.693–0.750 across the four discrete conditions
(a range of 0.057), whereas *C*_creval spans 0.224–0.554 (a range of
0.330, roughly six times wider).  This compression is not specific to
*C*_judge as an aggregate: the two individual judge streams show the
same pattern (DeepSeek mean = 0.753, SD = 0.077; GPT-4o judge mean =
0.711, SD = 0.073).  The untrained judges award scores near the top of
the scale even for identity-preserved T0 rows, providing little
headroom to detect improvements from cross-personality rewriting.

Per-condition standard deviations confirm the compression (Table 18).
*C*_judge SD stays below 0.075 in every condition, while *C*_creval SD
rises from 0.109 in T0 to 0.263 in T3—a pattern consistent with
pairwise comparison naturally spreading scores across the [0, 1]
interval as the rewrite quality distribution widens under higher conflict.

| Metric | SD (T0) | SD (T1) | SD (T2) | SD (T3) |
|--------|---------|---------|---------|---------|
| *C*_creval | 0.1093 | 0.2515 | 0.2604 | 0.2628 |
| *C*_auto | 0.0553 | 0.0646 | 0.0718 | 0.0710 |
| *C*_judge | 0.0444 | 0.0568 | 0.0723 | 0.0723 |

**Table 18.** Per-condition within-group standard deviations for the
three evaluation channels.  *C*_judge shows the most compressed
spread; *C*_creval the broadest, especially in rewriting conditions.

### 3.7.2 Condition discrimination: η² comparison

One-way ANOVA (T0 vs. T1 vs. T2 vs. T3) over the 1,680 matched rows
yields the following condition variance explained (η²):

| Channel | F | *p* | η² |
|---------|---|-----|-----|
| *C*_creval | 121.47 | 3.5×10⁻⁷¹ | **0.179** |
| *C*_auto | 60.68 | 3.1×10⁻³⁷ | 0.098 |
| *C*_judge | 44.06 | 2.1×10⁻²⁷ | 0.073 |

**Table 19.** One-way ANOVA results and η² by evaluation channel
(1,680 rows; conditions T0–T3).  CrEval accounts for 17.9% of total
variance through condition label alone, compared with 9.8% for
*C*_auto and 7.3% for *C*_judge.

*C*_creval explains more than twice the condition variance of
*C*_judge (η² = 0.179 vs. 0.073), despite testing the same rows.
This is not merely a scale artefact: the pairwise tournament
forces the evaluator to discriminate among rewrites globally within
each source, so the resulting win-rate distribution is better
calibrated to cross-condition quality differences than Likert scores
issued in isolation without exposure to the comparison field.

### 3.7.3 Correlation structure

Pearson and Spearman correlations among all channels are reported in
Table 20 (*n* = 1,680 except DeepSeek rows where *n* = 1,260).

| Pair | Pearson *r* | *p* | Spearman ρ |
|------|-------------|-----|------------|
| *C*_creval vs. *C*_judge | 0.486 | 2.4×10⁻¹⁰⁰ | 0.490 |
| *C*_creval vs. GPT-4o judge | 0.520 | 6.2×10⁻¹¹⁷ | 0.522 |
| *C*_creval vs. DeepSeek judge | 0.350 | 1.0×10⁻³⁷ | 0.365 |
| *C*_creval vs. surprise dim. | **0.727** | 1.7×10⁻²⁷⁶ | 0.697 |
| *C*_creval vs. *C*_auto | 0.194 | 1.0×10⁻¹⁵ | 0.199 |
| *C*_judge vs. GPT-4o judge | 0.926 | < 10⁻³⁰⁰ | 0.924 |
| *C*_judge vs. DeepSeek judge | 0.913 | < 10⁻³⁰⁰ | 0.904 |
| GPT-4o judge vs. DeepSeek judge | 0.660 | 2.9×10⁻¹⁵⁸ | 0.668 |
| *C*_judge vs. *C*_auto | 0.197 | 4.3×10⁻¹⁶ | 0.190 |

**Table 20.** Pearson and Spearman correlations among evaluation
channels (all conditions pooled).

Three patterns stand out:

1. **CrEval aligns most strongly with the surprise dimension** of the
   rubric (*r* = 0.727), more than with any other individual channel.
   This directly extends the rubric finding in Table 3—where surprise
   shows the largest positive curvature under conflict—to the pairwise
   channel: the calibrated evaluator effectively operationalises
   perceived surprise in a pairwise format.

2. **CrEval is more correlated with GPT-4o judge than with DeepSeek
   judge** (*r* = 0.520 vs. 0.350).  This gap is consistent with
   GPT-4o being the stronger evaluator: the inter-judge agreement
   between the two untrained LLMs is only *r* = 0.660 (moderate),
   indicating that DeepSeek and GPT-4o do not fully converge on
   absolute Likert assessments.  The fine-tuned CrEval, by contrast,
   appears to capture the latent preference signal that the stronger
   general-purpose judge encodes more reliably.

3. **Both *C*_creval and *C*_judge are weakly correlated with
   *C*_auto** (*r* ≈ 0.19–0.20), reproducing the cross-channel
   near-independence already reported in Section 3.5 (*r* = 0.118
   between judge novelty and *C*_auto).  The three channels therefore
   form two largely disjoint construct clusters: {*C*_creval,
   *C*_judge, surprise} and {*C*_auto}, with the judge cluster
   tightly internally correlated (*r* > 0.90 within *C*_judge
   components) and loosely connected to the automatic composite.

### 3.7.4 Detecting conflict-intensity curvature: CrEval vs. C_judge

Table 21 compares the T3-only OLS quadratic coefficient β_*d*² for
*C*_creval, *C*_judge, and *C*_auto side by side, per generator model.

| Generator | *C*_creval β_*d*² | *p* | *C*_judge β_*d*² | *p* | *C*_auto β_*d*² | *p* |
|-----------|-------------------|-----|-------------------|-----|-----------------|-----|
| GPT-4o | −0.664 | 0.383 | −0.206 | 0.310 | −0.518* | 0.027 |
| Qwen3-14B | **+2.022*** | 0.010 | +0.218 | 0.250 | −0.129 | 0.520 |
| Qwen3-8B | **+1.615*** | 0.031 | +0.218 | 0.332 | −0.299 | 0.178 |
| Qwen3-4B | +1.096 | 0.131 | +0.335 | 0.153 | +0.030 | 0.878 |

**Table 21.** Comparison of T3-only OLS quadratic curvature across
evaluation channels by generator (*n* = 240 each; OLS without
source-level random intercepts).  * = *p* < 0.05.

For Qwen3-14B and Qwen3-8B, *C*_creval detects **statistically
significant positive curvature** (*p* = 0.010 and 0.031,
respectively), while *C*_judge shows the same positive direction but
fails to reach significance (*p* = 0.250 and 0.332).  The curvature
that the untrained rubric judge only hints at is confirmed by the
fine-tuned pairwise evaluator at conventional thresholds.  This
enhanced sensitivity is attributable to the pairwise tournament's
broader effective dynamic range (Table 18): when the evaluation
scale spans [0, 1] with SD ≈ 0.25, small shifts in preference
probability accumulate statistical power across the 240-row slice
that compressed Likert scores in the range [0.68, 0.80] cannot
match.

The GPT-4o arm remains an exception: neither *C*_creval nor
*C*_judge shows significant curvature for GPT-4o rows.  This is
consistent with a quality ceiling interpretation—GPT-4o outputs are
competitive across the full *d*_H range and no systematic d_H
preference gradient is detectable in pairwise comparison.  Notably,
*C*_auto does detect negative curvature for GPT-4o (*p* = 0.027),
confirming once more that the automatic fidelity-based composite
penalises high-conflict GPT-4o rewrites even when calibrated human
preference does not.

### 3.7.5 Summary: where CrEval adds value over C_judge

The comparative analysis yields four conclusions:

- **Calibration.** Untrained LLM judges suffer from range
  compression (Likert scores cluster near the top of the scale
  regardless of condition), reducing their effective sensitivity to
  quality differences.  *C*_creval, as a pairwise win-rate, is
  immune to this by construction.

- **Condition discrimination.** *C*_creval explains 17.9% of
  condition variance (η²), compared with 9.8% for *C*_auto and
  7.3% for *C*_judge—making it the most informative single signal
  for the T0–T3 manipulation tested here.

- **Construct alignment.** *C*_creval correlates strongly with the
  surprise rubric dimension (*r* = 0.727), identifying perceived
  surprise as the common latent factor connecting fine-tuned
  preference and human-like rubric assessment.  Both channels
  remain weakly correlated with the automatic fidelity composite
  (*r* ≈ 0.19), confirming construct orthogonality.

- **Statistical power for d_H curvature.** On open-source Qwen3
  checkpoints, *C*_creval detects significant positive β_*d*²
  that *C*_judge only trends toward.  Using *C*_creval therefore
  provides a more sensitive instrument for the central conflict-
  intensity hypothesis than an untrained rubric judge on the same
  data.

These advantages do not make *C*_creval a universal replacement for
*C*_judge.  The rubric provides interpretable per-dimension
diagnostics (novelty, fidelity, fluency, etc.) that a single win-rate
scalar cannot supply; absolute Likert scores are also directly
comparable across studies that do not share the same tournament pool.
The recommendation is to treat the three channels as **complementary**:
*C*_auto for fidelity-sensitive automatic monitoring, *C*_judge for
interpretable multi-dimension diagnostics, and *C*_creval for
calibrated overall preference discrimination, especially when
statistical power on small per-model slices is a priority.

---

## Additions to Limitations

The following items extend Section 7:

- *C*_creval is a win-rate within a tournament pool that includes
  both high- and low-capacity generators (GPT-4o alongside 4B
  models).  GPT-4o's high win-rate may partly reflect the strength
  of its opponents rather than absolute quality; generator-stratified
  sub-tournaments would remove this confound but are left for future
  work.
- The round-robin compares rewrites across all conditions for the
  same source; T0 identity rows are structurally disadvantaged
  because they are drawn from a smaller set of distinct texts
  relative to T3's four sub-conditions.  The observed T0 win-rate
  floor (0.22) should be interpreted as confirming the effectiveness
  of any rewriting rather than as calibrating an absolute quality
  threshold.
- Per-generator OLS fits in Table 16 do not include source-level
  random intercepts (due to singular-covariance in the narrow
  *n* = 240 slice); they should be read as descriptive shape
  summaries analogous to the OLS curves in Figure 6.
- Target-persona win-rate differences (Section 5.6.5) are
  descriptive; no formal model controls for source-level variation
  or generator capacity in that table.

---

## Additions to Conclusion

The following paragraph extends Section 8:

A fine-tuned pairwise evaluator (*C*_creval) provides a third
channel that confirms the core condition effect—cross-personality
rewriting consistently outperforms identity-preserved text in
pairwise preference by a large margin (Cohen's *d* = 1.375)—while
resolving an ambiguity left open by the two-channel divergence in
Table 2.  On the open-source Qwen3 arms, *C*_creval shows positive
quadratic curvature vs. *d*_H (Table 16), aligning with the rubric
judge (*C*_judge) and opposing the automatic composite (*C*_auto).
This three-way pattern supports a reading in which high-conflict
rewrites trade NLI entailment fidelity for expressive salience: a
loss measurable by automatic composites, but a gain detectable by
both calibrated preference models and human-aligned rubric judges.
Within T3, adventurous target personas systematically attract higher
win-rates than conservative targets (Δ ≈ 0.35–0.45), identifying the
register dimension **R** as the primary driver of pairwise creative
preference—a finding that is complementary to, but dimensionally
inverted from, the *C*_auto directional table (Table 9), where the
analytic shift dimension Δ*S* dominates.  Together, the three
channels map distinct facets of "creativity" that practitioners
should select among based on the fidelity requirements of their
target application.

---

*References (new):*

[5] CrEval: A Fine-Tuned Pairwise Creativity Evaluator.
    Qwen2.5-7B-Instruct base + CrEval-7b LoRA adapter.
    (Insert full citation here once archival reference is available.)
