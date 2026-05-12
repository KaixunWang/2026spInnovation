"""Unified rewrite generator used by baselines and the mechanism arm.

A "rewrite job" is fully described by:
    source, target_persona, prompt_variant, mode, model, repeat_idx

where mode ∈ {"joint", "sequential", "constrained"} (the M1/M2/M3 stage design)
and condition/persona choice is decided by the caller (baselines.py / main).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from .corpus import SourceText
from .generator import GenerationResult, generate
from .personalities import Persona


REWRITE_USER_TEMPLATE = (
    "SOURCE TEXT (genre: {GENRE}):\n"
    "---\n"
    "{SOURCE}\n"
    "---\n\n"
    "Rewrite the source above according to the instructions you were given. "
    "Preserve every propositional/content commitment. Output only the rewritten "
    "text, without commentary or headers."
)


def _fill_source_template(template: str, source: SourceText) -> str:
    """Inject genre/source without str.format (source text may contain ``{`` / ``}``)."""
    return template.replace("{GENRE}", str(source.genre)).replace("{SOURCE}", source.text)


def _prompt_token_budget(source: SourceText) -> int:
    """Approximate prompt-token budget for mechanism control.

    Uses a conservative word-based proxy to keep M0/M1/M2/M3 prompts comparable.
    """
    wc = max(80, source.word_count())
    return int(max(120, min(1200, wc * 2.2)))


def _trim_words(text: str, max_words: int) -> str:
    ws = (text or "").split()
    if len(ws) <= max_words:
        return text
    return " ".join(ws[:max_words])


# -- M2: step-1 faithful paraphrase prompt -----------------------------------

PARAPHRASE_SYSTEM = (
    "You are a faithful paraphraser. You restate the source's propositional "
    "content in neutral, unadorned English prose. Do not add, remove, or alter "
    "any commitment of the source. Keep the length within 90-110% of the source."
)

PARAPHRASE_USER = (
    "SOURCE TEXT (genre: {GENRE}):\n"
    "---\n"
    "{SOURCE}\n"
    "---\n\n"
    "Provide a neutral, faithful paraphrase. Output only the paraphrase."
)


# -- M3: step-1 propositional checklist extractor ----------------------------

CHECKLIST_SYSTEM = (
    "You are a careful reading-comprehension assistant. "
    "From the passage below, extract a numbered checklist of its propositional "
    "commitments (factual claims, causal links, and concrete imagery) that any "
    "faithful rewrite must preserve. Be concise."
)

CHECKLIST_USER = (
    "SOURCE TEXT (genre: {GENRE}):\n"
    "---\n"
    "{SOURCE}\n"
    "---\n\n"
    "Output only the numbered checklist."
)


@dataclass
class RewriteResult:
    text: str
    mode: str
    intermediate: dict[str, str] = field(default_factory=dict)
    cached: bool = False
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode,
            "intermediate": self.intermediate,
            "cached": self.cached,
            "usage": self.usage,
        }


# ---------------------------------------------------------------------------
# Core rewrite with persona
# ---------------------------------------------------------------------------


def _length_constraint(source: SourceText) -> str:
    wc = max(80, source.word_count())
    lo, hi = int(wc * 0.8), int(wc * 1.2)
    return f"Target length: {lo}-{hi} words."


def rewrite_joint(
    source: SourceText,
    persona: Persona,
    *,
    model: str,
    prompt_variant: int = 0,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    extra_system: str = "",
    use_cache: bool = True,
) -> RewriteResult:
    """M1 baseline: one-shot rewrite under the persona's prompt."""
    system = persona.prompt(prompt_variant) + "\n\n" + _length_constraint(source)
    if extra_system:
        system += "\n\n" + extra_system
    user = _fill_source_template(REWRITE_USER_TEMPLATE, source)
    res = generate(
        model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens else max(200, int(source.word_count() * 2.5)),
        use_cache=use_cache,
    )
    return RewriteResult(
        text=(res.text or "").strip(),
        mode="joint",
        cached=res.cached,
        usage=res.usage,
    )


def rewrite_sequential(
    source: SourceText,
    persona: Persona,
    *,
    model: str,
    prompt_variant: int = 0,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> RewriteResult:
    """M2: first produce a neutral paraphrase, then apply the persona to it."""
    para_user = _fill_source_template(PARAPHRASE_USER, source)
    para_res = generate(
        model,
        system=PARAPHRASE_SYSTEM,
        user=para_user,
        temperature=0.3,
        max_tokens=max(200, int(source.word_count() * 2.5)),
        use_cache=use_cache,
    )
    paraphrase = (para_res.text or "").strip()
    if os.environ.get("INNOVATION_LOG_REWRITE_MESSAGES", "").strip().lower() in ("1", "true", "yes"):
        print(
            "[rewrite_sequential] stage1 messages (system + user) -> stderr\n"
            f"--- system ({len(PARAPHRASE_SYSTEM)} chars) ---\n{PARAPHRASE_SYSTEM}\n"
            f"--- user ({len(para_user)} chars) ---\n{para_user}",
            file=sys.stderr,
        )
    if not paraphrase:
        print(
            "[rewrite_sequential] stage1 returned empty paraphrase; messages sent were:\n"
            f"--- system ---\n{PARAPHRASE_SYSTEM}\n--- user (trunc 8000) ---\n{para_user[:8000]}",
            file=sys.stderr,
        )
    sys_prompt = persona.prompt(prompt_variant) + "\n\n" + _length_constraint(source)
    budget = _prompt_token_budget(source)
    paraphrase = _trim_words(paraphrase, max(40, budget // 2))
    stage2_user = (
        "You will receive a neutral paraphrase of a source passage. "
        "Rewrite it according to your persona. Preserve every propositional "
        "commitment.\n\n"
        f"NEUTRAL PARAPHRASE:\n---\n{paraphrase}\n---\n\n"
        "Output only the rewritten text."
    )
    stage2_res = generate(
        model,
        system=sys_prompt,
        user=stage2_user,
        temperature=temperature,
        max_tokens=max(200, int(source.word_count() * 2.5)),
        use_cache=use_cache,
    )
    return RewriteResult(
        text=(stage2_res.text or "").strip(),
        mode="sequential",
        intermediate={"paraphrase": paraphrase},
        cached=para_res.cached and stage2_res.cached,
        usage={
            "paraphrase": para_res.usage,
            "stylistic": stage2_res.usage,
        },
    )


def rewrite_constrained(
    source: SourceText,
    persona: Persona,
    *,
    model: str,
    prompt_variant: int = 0,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> RewriteResult:
    """M3: extract a propositional checklist, include it in the rewrite prompt."""
    chk_user = _fill_source_template(CHECKLIST_USER, source)
    chk_res = generate(
        model,
        system=CHECKLIST_SYSTEM,
        user=chk_user,
        temperature=0.0,
        max_tokens=400,
        use_cache=use_cache,
    )
    budget = _prompt_token_budget(source)
    checklist = _trim_words((chk_res.text or "").strip(), max(40, budget // 2))
    extra = (
        "The following numbered items are REQUIRED to survive the rewrite "
        "(do not remove, contradict, or alter them):\n"
        f"{checklist}"
    )
    j = rewrite_joint(
        source,
        persona,
        model=model,
        prompt_variant=prompt_variant,
        temperature=temperature,
        extra_system=extra,
        use_cache=use_cache,
    )
    j.mode = "constrained"
    j.intermediate = {"checklist": checklist}
    return j


def rewrite_constrained_control(
    source: SourceText,
    persona: Persona,
    *,
    model: str,
    prompt_variant: int = 0,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> RewriteResult:
    """M0 control: same two-stage structure as M3, but checklist content is neutral.

    Keeps approximately the same checklist token count while removing source-specific
    propositional constraints, isolating token/structure effects. The model still
    receives a real checklist extraction in ``checklist_extracted`` for auditing;
    the joint prompt uses ``dummy`` placeholders only (by design, not a failure path).
    """
    chk_user = _fill_source_template(CHECKLIST_USER, source)
    chk_res = generate(
        model,
        system=CHECKLIST_SYSTEM,
        user=chk_user,
        temperature=0.0,
        max_tokens=400,
        use_cache=use_cache,
    )
    checklist = (chk_res.text or "").strip()
    budget = _prompt_token_budget(source)
    checklist = _trim_words(checklist, max(40, budget // 2))
    n_words = max(20, len(checklist.split()))
    dummy = " ".join(["placeholder"] * n_words)
    extra = (
        "The following numbered list is a formatting control only. "
        "Keep output coherent but do not treat these placeholders as content constraints:\n"
        f"{dummy}"
    )
    j = rewrite_joint(
        source,
        persona,
        model=model,
        prompt_variant=prompt_variant,
        temperature=temperature,
        extra_system=extra,
        use_cache=use_cache,
    )
    j.mode = "constrained_control"
    j.intermediate = {"checklist_extracted": checklist, "checklist_control": dummy}
    return j


MODE_DISPATCH = {
    "joint": rewrite_joint,
    "sequential": rewrite_sequential,
    "constrained": rewrite_constrained,
    "constrained_control": rewrite_constrained_control,
    # convenience aliases
    "M1": rewrite_joint,
    "M2": rewrite_sequential,
    "M3": rewrite_constrained,
    "M0": rewrite_constrained_control,
}


def rewrite(
    source: SourceText,
    persona: Persona,
    *,
    mode: str = "joint",
    model: str,
    prompt_variant: int = 0,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> RewriteResult:
    fn = MODE_DISPATCH[mode]
    return fn(
        source,
        persona,
        model=model,
        prompt_variant=prompt_variant,
        temperature=temperature,
        use_cache=use_cache,
    )
