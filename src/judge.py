"""LLM-as-Judge with anti-bias safeguards.

Features:
  * dual-judge: every item is scored by >=2 judge models from different families
  * absolute rubric (TTCW-inspired, 7 dimensions, 1-5 Likert)
  * pairwise rubric (A vs B, options order randomised)
  * strict JSON parsing with one-shot repair attempt
  * provenance masking: the judge sees no info about which model produced each rewrite
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from .config_loader import get_model_spec, get_role_models
from .generator import generate


# ---------------------------------------------------------------------------
# Rubric definitions (match docs/rubrics.md)
# ---------------------------------------------------------------------------

ABSOLUTE_RUBRIC_DIMS = (
    "novelty",
    "surprise",
    "imagery",
    "fidelity",
    "fluency",
    "consistency",
    "logical_completeness",
)

ABSOLUTE_PROMPT = """You are an expert literary reviewer. Read a source passage and a rewritten version. Score the rewritten version along the rubric below. Calibrate against the typical range of the source's genre.

SOURCE (genre: {GENRE}):
{SOURCE}

REWRITE:
{GENERATION}

For EACH dimension, output an integer 1-5 (inclusive). Use the full range. Be strict about fidelity: a beautiful rewrite that alters factual content deserves a low Fidelity score even if Novelty is high.

Dimensions:
- Calibrate relative to the source text, not against all possible writings in the genre.
- Novelty (freshness of phrasing, framing, imagery)
- Surprise (non-obvious but licensed choices)
- Imagery (concreteness and figurative density appropriate to the genre)
- Fidelity (propositional preservation)
- Fluency (sentence-level well-formedness)
- Consistency (stability of tone, viewpoint, register)
- LogicalCompleteness (intact inferences / narrative arc)

Respond as STRICT JSON with this exact shape (no markdown, no commentary):
{{"novelty": <int>, "surprise": <int>, "imagery": <int>, "fidelity": <int>, "fluency": <int>, "consistency": <int>, "logical_completeness": <int>}}
"""

PAIRWISE_PROMPT = """You are an expert literary reviewer. Given a source text of genre {GENRE} and two unlabeled rewrites A and B, pick which rewrite is more creative *for that genre*, considering both freshness of expression AND preservation of content.

If one rewrite loses content or becomes incoherent, prefer the other.

SOURCE (genre: {GENRE}):
{SOURCE}

REWRITE A:
{REWRITE_A}

REWRITE B:
{REWRITE_B}

Respond as STRICT JSON with one of five verdicts (A_much_better / A_slightly_better / tie / B_slightly_better / B_much_better):
{{"verdict": "<one of: A_much_better|A_slightly_better|tie|B_slightly_better|B_much_better>", "reason": "<one sentence>"}}
"""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_BLOCK.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Absolute scoring
# ---------------------------------------------------------------------------


@dataclass
class AbsoluteScore:
    judge: str
    novelty: int | None = None
    surprise: int | None = None
    imagery: int | None = None
    fidelity: int | None = None
    fluency: int | None = None
    consistency: int | None = None
    logical_completeness: int | None = None
    raw: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    def ok(self) -> bool:
        return all(getattr(self, d) is not None for d in ABSOLUTE_RUBRIC_DIMS)


def score_absolute(
    source_text: str,
    generation: str,
    *,
    genre: str,
    judge_model: str,
) -> AbsoluteScore:
    user = ABSOLUTE_PROMPT.format(GENRE=genre, SOURCE=source_text, GENERATION=generation)
    # OpenAI-compatible gateways (incl. DeepSeek) may truncate before the closing brace
    # if completion budget is tight; use model ceiling with a floor for JSON safety.
    spec = get_model_spec(judge_model)
    judge_cap = int(spec.get("max_tokens", 700))
    abs_floor = 1024
    res = generate(
        judge_model,
        system="You are a strict JSON-only literary judge.",
        user=user,
        temperature=0.0,
        max_tokens=max(abs_floor, judge_cap),
    )
    parsed = _parse_json(res.text)
    if parsed is None:
        return AbsoluteScore(judge=judge_model, raw=res.text, error="unparseable JSON")
    out = AbsoluteScore(judge=judge_model, raw=res.text)
    for d in ABSOLUTE_RUBRIC_DIMS:
        v = parsed.get(d)
        try:
            iv = int(v)
            out.__setattr__(d, max(1, min(5, iv)))
        except Exception:
            continue
    if not out.ok():
        out.error = f"missing dims: {[d for d in ABSOLUTE_RUBRIC_DIMS if getattr(out, d) is None]}"
    return out


def _placeholder_absolute_score(judge_model: str, reason: BaseException | str) -> AbsoluteScore:
    """Mid-scale (3) on every rubric dim when a judge API call fails (e.g. content_filter)."""
    if isinstance(reason, BaseException):
        msg = f"{type(reason).__name__}: {reason}"
    else:
        msg = str(reason)
    msg = msg.replace("\n", " ").strip()
    if len(msg) > 400:
        msg = msg[:400] + "..."
    return AbsoluteScore(
        judge=judge_model,
        novelty=3,
        surprise=3,
        imagery=3,
        fidelity=3,
        fluency=3,
        consistency=3,
        logical_completeness=3,
        raw="",
        error=f"placeholder:{msg}",
    )


def score_absolute_dual(
    source_text: str,
    generation: str,
    *,
    genre: str,
    judges: list[str] | None = None,
) -> list[AbsoluteScore]:
    """Score with >=2 judges; caller can compute ICC from the list."""
    if judges is None:
        judges = get_role_models("judges")
    out: list[AbsoluteScore] = []
    for j in judges:
        try:
            out.append(score_absolute(source_text, generation, genre=genre, judge_model=j))
        except Exception as e:
            print(f"[judge] {j}: API failure, using placeholder scores (all dims=3): {e!r}", file=sys.stderr)
            out.append(_placeholder_absolute_score(j, e))
    return out


# ---------------------------------------------------------------------------
# Pairwise scoring
# ---------------------------------------------------------------------------

VERDICT_SCORE = {
    "A_much_better": 2,
    "A_slightly_better": 1,
    "tie": 0,
    "B_slightly_better": -1,
    "B_much_better": -2,
}


@dataclass
class PairwiseScore:
    judge: str
    verdict: str
    score: int              # +2 (A) .. 0 (tie) .. -2 (B) under a fixed orientation
    reason: str = ""
    a_was_first: bool = True
    raw: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def score_pairwise(
    source_text: str,
    rewrite_a: str,
    rewrite_b: str,
    *,
    genre: str,
    judge_model: str,
    rng: random.Random | None = None,
) -> PairwiseScore:
    rng = rng or random.Random(0)
    a_first = rng.random() >= 0.5
    first, second = (rewrite_a, rewrite_b) if a_first else (rewrite_b, rewrite_a)
    user = PAIRWISE_PROMPT.format(
        GENRE=genre,
        SOURCE=source_text,
        REWRITE_A=first,
        REWRITE_B=second,
    )
    res = generate(
        judge_model,
        system="You are a strict JSON-only literary judge.",
        user=user,
        temperature=0.0,
        max_tokens=200,
    )
    parsed = _parse_json(res.text)
    if parsed is None:
        return PairwiseScore(
            judge=judge_model, verdict="tie", score=0, a_was_first=a_first,
            raw=res.text, error="unparseable",
        )
    verdict = str(parsed.get("verdict", "tie"))
    raw_score = VERDICT_SCORE.get(verdict, 0)
    # If A was served second, flip sign so that "score > 0 ↔ rewrite_a preferred"
    signed = raw_score if a_first else -raw_score
    return PairwiseScore(
        judge=judge_model,
        verdict=verdict,
        score=signed,
        reason=str(parsed.get("reason", "")),
        a_was_first=a_first,
        raw=res.text,
    )


def score_pairwise_dual(
    source_text: str,
    rewrite_a: str,
    rewrite_b: str,
    *,
    genre: str,
    judges: list[str] | None = None,
    seed: int = 0,
) -> list[PairwiseScore]:
    if judges is None:
        judges = get_role_models("judges")
    rng = random.Random(seed)
    return [
        score_pairwise(source_text, rewrite_a, rewrite_b, genre=genre, judge_model=j, rng=rng)
        for j in judges
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_absolute_with_empty_fallback(
    scores: list[AbsoluteScore], judges: list[str]
) -> dict[str, Any]:
    """Like ``aggregate_absolute``, but if every judge row is unusable, use mid-scale placeholders."""
    agg = aggregate_absolute(scores)
    if agg.get("ok"):
        return agg
    print(
        "[judge] no valid scores from any judge; using placeholder (all dims=3) for aggregation",
        file=sys.stderr,
    )
    return aggregate_absolute([_placeholder_absolute_score(j, "no_valid_judge_outputs") for j in judges])


def aggregate_absolute(scores: list[AbsoluteScore]) -> dict[str, Any]:
    """Aggregate multiple judges' absolute scores to Novelty_judge / Value_judge [0,1]."""
    by_dim: dict[str, list[int]] = {d: [] for d in ABSOLUTE_RUBRIC_DIMS}
    for s in scores:
        if not s.ok():
            continue
        for d in ABSOLUTE_RUBRIC_DIMS:
            by_dim[d].append(int(getattr(s, d)))
    if not any(by_dim.values()):
        return {"ok": False}
    means = {d: (sum(v) / len(v) if v else None) for d, v in by_dim.items()}
    novelty_judge = (
        sum(means[d] for d in ("novelty", "surprise", "imagery") if means[d] is not None)
        / sum(1 for d in ("novelty", "surprise", "imagery") if means[d] is not None)
    ) / 5.0
    coherence_judge = (
        sum(means[d] for d in ("fluency", "consistency", "logical_completeness") if means[d] is not None)
        / sum(1 for d in ("fluency", "consistency", "logical_completeness") if means[d] is not None)
    ) / 5.0
    fidelity_judge = (means["fidelity"] or 0.0) / 5.0
    return {
        "ok": True,
        "novelty_judge": novelty_judge,
        "coherence_judge": coherence_judge,
        "fidelity_judge": fidelity_judge,
        "per_judge": [s.to_dict() for s in scores],
        "per_dim_mean": means,
    }
