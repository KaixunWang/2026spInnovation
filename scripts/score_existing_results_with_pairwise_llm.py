"""Batch-score existing aligned rewrite results with a plain (untrained) pairwise LLM evaluator.

This script mirrors the alignment logic of scripts/score_existing_results_with_creval.py,
but calls a general OpenAI-compatible chat model as a baseline evaluator.

It writes compact ``*_<suffix>.jsonl`` files that can be merged back into
``*_metrics.jsonl`` via scripts/merge_external_eval_into_metrics.py.

Default use case in this repository:

  * candidate: main_qwen3_4b_metrics.jsonl / main_qwen3_8b_metrics.jsonl /
    main_qwen3_14b_metrics.jsonl
  * reference: main_metrics.jsonl with model == gen_openai_4o
  * condition: T3 only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from src.corpus import load_sources  # noqa: E402
from src.personalities import load_personas  # noqa: E402


SYSTEM_PROMPT = (
    "你是一个创意评估器，请比较两个回复在同一任务下的创意水平。"
    "只输出一个JSON对象，不要输出任何额外文本。"
)

DEFAULT_INPUTS = [
    ROOT / "data" / "generated" / "main_qwen3_4b_metrics.jsonl",
    ROOT / "data" / "generated" / "main_qwen3_8b_metrics.jsonl",
    ROOT / "data" / "generated" / "main_qwen3_14b_metrics.jsonl",
]

DEFAULT_REFERENCE = ROOT / "data" / "generated" / "main_metrics.jsonl"


def _env_first(*keys: str, default: str) -> str:
    for key in keys:
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _normalize_suffix(suffix: str) -> str:
    out = suffix.strip()
    if out.endswith(".jsonl"):
        out = out[: -len(".jsonl")]
    out = out.strip("_").strip()
    return out or "llm_pairwise"


def _split_models(text: str | None) -> list[str]:
    if not text:
        return []
    vals = [x.strip() for x in re.split(r"[;,\n]", text) if x.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _default_fallback_models(base_url: str) -> list[str]:
    u = (base_url or "").lower()
    if "openrouter" in u:
        return ["deepseek/deepseek-chat-v3-0324", "anthropic/claude-3.5-haiku", "openai/gpt-4o-mini"]
    if "deepseek.com" in u:
        return ["deepseek-chat", "deepseek-reasoner"]
    return []


def _is_region_block_error(err: Exception) -> bool:
    msg = str(err).lower()
    return ("403" in msg) and ("region" in msg or "not available" in msg or "permission" in msg)


def _probe_api(base_url: str, timeout_seconds: float) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/models"
    timeout = httpx.Timeout(connect=timeout_seconds, read=timeout_seconds, write=timeout_seconds, pool=timeout_seconds)
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as c:
            r = c.get(url)
        if r.status_code >= 500:
            return False, f"HTTP {r.status_code} from {url}"
        return True, f"HTTP {r.status_code} from {url}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_user_prompt(query: str, resp1: str, resp2: str) -> str:
    return (
        "请比较下面两个回复在创意上的优劣，输出JSON对象，格式必须是:\n"
        '{"winner":"response_1|response_2|tie","confidence":0~1,"reason":"一句话"}\n\n'
        "[[DATA FIELD START]]\n"
        "### Query:\n"
        f"{query}\n"
        "### Response 1:\n"
        f"{resp1}\n"
        "### Response 2:\n"
        f"{resp2}\n"
        "[[DATA FIELD END]]"
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    snippet = m.group(0)
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _winner_to_score(winner: str) -> tuple[str, float | None]:
    w = winner.strip().lower().replace("-", "_")
    if w in {"response_1", "resp1", "r1", "1", "option_1", "option1", "a"}:
        return "response_1", 1.0
    if w in {"response_2", "resp2", "r2", "2", "option_2", "option2", "b"}:
        return "response_2", 0.0
    if w in {"tie", "equal", "draw", "same"}:
        return "tie", 0.5
    return "unknown", None


def parse_pairwise_output(output: str) -> tuple[str, float | None]:
    raw = (output or "").strip()
    obj = _extract_json_object(raw)
    if obj is not None:
        verdict, score = _winner_to_score(str(obj.get("winner", "")))
        if score is not None:
            return verdict, score

    normalized = re.sub(r"\s+", " ", raw)
    if "创意程度相当" in normalized or re.search(r"\btie\b|\bequal\b|\bdraw\b", normalized, flags=re.IGNORECASE):
        return "tie", 0.5
    if re.search(r"response\s*1\b", normalized, flags=re.IGNORECASE):
        return "response_1", 1.0
    if re.search(r"response\s*2\b", normalized, flags=re.IGNORECASE):
        return "response_2", 0.0
    return "unknown", None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_id"),
        row.get("condition", row.get("mode")),
        row.get("target_persona"),
        row.get("repeat_idx"),
        row.get("prompt_variant"),
    )


def build_query(row: dict[str, Any], source_text: str, persona_prompt: str | None) -> str:
    target_persona = row.get("target_persona") or "none"
    mode = row.get("mode") or "joint"
    genre = row.get("genre") or "unknown"
    return (
        "You are evaluating two candidate rewrites for the same creativity task.\n\n"
        f"Genre: {genre}\n"
        f"Rewrite mode: {mode}\n"
        f"Target persona: {target_persona}\n\n"
        "Task instruction:\n"
        "Rewrite the source passage creatively while preserving every propositional and content commitment.\n"
        "The rewrite should follow the target persona/style specification below.\n\n"
        "Target persona specification:\n"
        f"{(persona_prompt or target_persona).strip()}\n\n"
        "Source passage:\n"
        f"{source_text}"
    )


def infer_once(
    client: OpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_with_retry(
    client: OpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    max_tokens: int,
    retries: int,
    sleep_seconds: float,
) -> str:
    last_error: Exception | None = None
    attempts = max(1, retries)
    for i in range(attempts):
        try:
            return infer_once(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=model_name,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if _is_region_block_error(e):
                raise
            last_error = e
            if i + 1 < attempts:
                time.sleep(max(0.0, sleep_seconds))
    if last_error is None:
        raise RuntimeError("unknown retry failure")
    raise last_error


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="*", type=Path, default=None)
    p.add_argument("--reference-metrics", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--reference-model", default="gen_openai_4o")
    p.add_argument("--condition", default="T3")
    p.add_argument("--field-name", default="llm_pairwise")
    p.add_argument("--output-suffix", default="llm_pairwise")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="print progress every N written rows (0 disables periodic progress logs)",
    )
    p.add_argument("--probe-timeout", type=float, default=3.0)
    p.add_argument("--fallback-models", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build aligned pairs and write sample metadata without calling API",
    )
    args = p.parse_args()

    api_key = _env_first("PAIRWISE_EVAL_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", default="0")
    base_url = _env_first(
        "PAIRWISE_EVAL_BASE_URL",
        "OPENAI_BASE_URL",
        "DEEPSEEK_BASE_URL",
        default="http://127.0.0.1:8000/v1",
    )
    model_name = _env_first("PAIRWISE_EVAL_MODEL", default="gpt-4o-mini")
    timeout_seconds = float(_env_first("PAIRWISE_EVAL_TIMEOUT", default="300"))

    env_fallbacks = _split_models(os.environ.get("PAIRWISE_EVAL_FALLBACK_MODELS"))
    cli_fallbacks = args.fallback_models or []
    model_candidates = [model_name, *cli_fallbacks, *env_fallbacks, *_default_fallback_models(base_url)]
    deduped: list[str] = []
    seen: set[str] = set()
    for m in model_candidates:
        mm = (m or "").strip()
        if not mm or mm in seen:
            continue
        seen.add(mm)
        deduped.append(mm)
    model_candidates = deduped or [model_name]

    if not args.dry_run and not args.skip_probe:
        ok, detail = _probe_api(base_url, max(1.0, args.probe_timeout))
        if not ok:
            raise SystemExit(
                "[pairwise_llm] API probe failed: "
                f"{detail}.\n"
                "Check PAIRWISE_EVAL_BASE_URL / OPENAI_BASE_URL / DEEPSEEK_BASE_URL connectivity first."
            )
        print(f"[pairwise_llm] API probe ok: {detail}")

    io_timeout = httpx.Timeout(
        connect=min(10.0, max(1.0, timeout_seconds)),
        read=max(1.0, timeout_seconds),
        write=60.0,
        pool=60.0,
    )
    http_client = httpx.Client(timeout=io_timeout, trust_env=False)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=io_timeout, http_client=http_client)

    suffix = _normalize_suffix(args.output_suffix)
    field_name = args.field_name.strip() or "llm_pairwise"

    sources = {s.id: s for s in load_sources()}
    personas = {p.name: p for p in load_personas()}

    active_model_idx = 0
    try:
        ref_rows = read_jsonl(args.reference_metrics)
        ref_by_key = {
            row_key(r): r
            for r in ref_rows
            if r.get("model") == args.reference_model and r.get("condition") == args.condition and r.get("text")
        }

        inputs = args.inputs or DEFAULT_INPUTS
        for input_path in inputs:
            if not input_path.exists():
                print(f"[pairwise_llm] skip missing {input_path}")
                continue

            rows = read_jsonl(input_path)
            base_stem = input_path.stem.replace("_metrics", "")
            out_path = input_path.with_name(f"{base_stem}_{suffix}.jsonl")
            if out_path.exists() and not args.overwrite:
                print(f"[pairwise_llm] {out_path.name} exists; use --overwrite to rebuild")
                continue
            if args.overwrite:
                out_path.unlink(missing_ok=True)

            aligned_rows: list[tuple[dict[str, Any], dict[str, Any], Any, Any]] = []
            for row in rows:
                if row.get("condition") != args.condition:
                    continue
                key = row_key(row)
                ref = ref_by_key.get(key)
                if ref is None:
                    continue
                sid = row.get("source_id")
                src = sources.get(sid)
                if src is None or not row.get("text") or not ref.get("text"):
                    continue

                persona = personas.get(row.get("target_persona"))
                aligned_rows.append((row, ref, src, persona))

            if args.limit is not None:
                aligned_rows = aligned_rows[: max(0, args.limit)]

            total_rows = len(aligned_rows)
            if total_rows == 0:
                print(f"[pairwise_llm] no aligned rows -> {out_path.name}")

            written = 0
            request_errors = 0
            parse_unknown = 0
            switched_models = 0
            sample_printed = False
            started = time.perf_counter()
            with out_path.open("w", encoding="utf-8") as fh:
                for row, ref, src, persona in aligned_rows:
                    key = row_key(row)
                    query = build_query(row, src.text, persona.system_prompt if persona else None)
                    user_prompt = build_user_prompt(query, str(row.get("text", "")), str(ref.get("text", "")))

                    if not sample_printed:
                        print(f"[pairwise_llm] sample key={key}")
                        print(
                            "[pairwise_llm] sample query chars="
                            f"{len(query)} response1 chars={len(str(row.get('text', '')))} "
                            f"response2 chars={len(str(ref.get('text', '')))}"
                        )
                        print(f"[pairwise_llm] model candidates: {model_candidates}")
                        sample_printed = True

                    selected_model = model_candidates[active_model_idx]
                    if args.dry_run:
                        payload = {
                            "ok": False,
                            "reason": "dry_run",
                            "reference_model": args.reference_model,
                            "reference_metrics": args.reference_metrics.name,
                            "evaluator_model_name": selected_model,
                        }
                    else:
                        while True:
                            selected_model = model_candidates[active_model_idx]
                            try:
                                raw = _call_with_retry(
                                    client,
                                    system_prompt=SYSTEM_PROMPT,
                                    user_prompt=user_prompt,
                                    model_name=selected_model,
                                    max_tokens=args.max_tokens,
                                    retries=args.retries,
                                    sleep_seconds=args.sleep_seconds,
                                )
                                verdict, score = parse_pairwise_output(raw)
                                if score is None:
                                    parse_unknown += 1
                                payload = {
                                    "ok": score is not None,
                                    "reason": "parse_unknown" if score is None else "ok",
                                    "score": score,
                                    "verdict": verdict,
                                    "raw": raw,
                                    "reference_model": args.reference_model,
                                    "reference_metrics": args.reference_metrics.name,
                                    "evaluator_model_name": selected_model,
                                }
                                break
                            except Exception as e:
                                if _is_region_block_error(e) and active_model_idx + 1 < len(model_candidates):
                                    prev = selected_model
                                    active_model_idx += 1
                                    switched_models += 1
                                    print(
                                        f"[pairwise_llm] model switch due to region block: {prev} -> "
                                        f"{model_candidates[active_model_idx]}"
                                    )
                                    continue
                                if args.fail_fast:
                                    raise
                                request_errors += 1
                                payload = {
                                    "ok": False,
                                    "reason": "request_error",
                                    "error": f"{type(e).__name__}: {e}",
                                    "reference_model": args.reference_model,
                                    "reference_metrics": args.reference_metrics.name,
                                    "evaluator_model_name": selected_model,
                                }
                                break

                    out_row = {
                        "source_id": row.get("source_id"),
                        "genre": row.get("genre"),
                        "condition": row.get("condition"),
                        "target_persona": row.get("target_persona"),
                        "model": row.get("model"),
                        "repeat_idx": row.get("repeat_idx"),
                        "prompt_variant": row.get("prompt_variant"),
                        "mode": row.get("mode"),
                        "d_H": row.get("d_H"),
                        field_name: payload,
                    }
                    fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    written += 1
                    if args.progress_every > 0 and (written % args.progress_every == 0 or written == total_rows):
                        elapsed = time.perf_counter() - started
                        rate = written / elapsed if elapsed > 0 else 0.0
                        remaining = max(0, total_rows - written)
                        eta_seconds = (remaining / rate) if rate > 0 else 0.0
                        print(
                            f"[pairwise_llm] progress {out_path.name}: {written}/{total_rows} "
                            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)}"
                        )

            print(
                f"[pairwise_llm] wrote {written} rows -> {out_path} "
                f"(request_errors={request_errors}, parse_unknown={parse_unknown}, switched_models={switched_models})"
            )
    finally:
        http_client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
