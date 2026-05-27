"""Batch-score existing aligned rewrite results with CrEval.

CrEval is a pairwise evaluator, so this script compares a candidate metrics file
against a reference metrics file on aligned rewrite tasks, then writes a compact
``*_creval.jsonl`` file that can be merged back into ``*_metrics.jsonl`` via
``scripts/merge_external_eval_into_metrics.py``.

Default use case in this repository:

  * candidate: ``main_qwen3_4b_metrics.jsonl`` / ``main_qwen3_8b_metrics.jsonl`` /
    ``main_qwen3_14b_metrics.jsonl``
  * reference: ``main_metrics.jsonl`` with ``model == gen_openai_4o``
  * condition: ``T3`` only, because those rows align exactly across generators

The script mirrors the official CrEval prompt format from
``Aman-4-Real/CrEval/inference.py``.
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
    "你是一个语言创意评估专家。我会给你一条指令和对应的两个回复，"
    "请对两个回复的创意程度进行评估。下面是具体的数据："
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


def _probe_creval_api(base_url: str, probe_timeout: float) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/models"
    timeout = httpx.Timeout(connect=probe_timeout, read=probe_timeout, write=probe_timeout, pool=probe_timeout)
    try:
        # trust_env=False avoids routing localhost through system HTTP proxy.
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
        "[[DATA FIELD START]]\n"
        "### Query:\n"
        f"{query}\n"
        "### Response 1:\n"
        f"{resp1}\n"
        "### Response 2:\n"
        f"{resp2}\n"
        "[[DATA FIELD END]]\n"
        "请注意：1.挖掘创意的核心内涵，即对指令有用且新颖的回复；"
        "2.仔细比较和评估上述两个回复的创意程度，并以“更有创意的回复是：Response ”或"
        "“二者的创意程度相当。”的形式作为结尾给出你的评估决定。"
    )


def parse_creval_output(output: str) -> tuple[str, float | None]:
    text = (output or "").strip()
    normalized = re.sub(r"\s+", " ", text)
    if "二者的创意程度相当" in normalized or "创意程度相当" in normalized:
        return "tie", 0.5
    if re.search(r"更有创意的回复是[:：]?\s*Response\s*1\b", normalized, flags=re.IGNORECASE):
        return "response_1", 1.0
    if re.search(r"更有创意的回复是[:：]?\s*Response\s*2\b", normalized, flags=re.IGNORECASE):
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
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--sleep-seconds", type=float, default=1.5)
    p.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="print progress every N written rows (0 disables periodic progress logs)",
    )
    p.add_argument("--probe-timeout", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--skip-probe",
        action="store_true",
        help="skip API reachability probe before scoring",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="abort immediately when a request still fails after retries",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build aligned pairs and write sample metadata without calling the CrEval API",
    )
    args = p.parse_args()

    api_key = _env_first("CREVAL_API_KEY", default="0")
    base_url = _env_first("CREVAL_BASE_URL", "CREVAL_API_BASE_URL", default="http://127.0.0.1:8000/v1")
    model_name = _env_first("CREVAL_MODEL_NAME", "CREVAL_API_MODEL", default="meta-llama/Meta-Llama-3-8B-Instruct")
    timeout_seconds = float(_env_first("CREVAL_API_TIMEOUT", default="1200"))
    if not args.dry_run and not args.skip_probe:
        ok, detail = _probe_creval_api(base_url, max(1.0, args.probe_timeout))
        if not ok:
            raise SystemExit(
                "[creval] API probe failed: "
                f"{detail}.\n"
                "Check that CrEval API server is running (not interactive inference client), "
                f"and that CREVAL_BASE_URL={base_url} is reachable from this Windows process."
            )
        print(f"[creval] API probe ok: {detail}")

    io_timeout = httpx.Timeout(
        connect=min(10.0, max(1.0, timeout_seconds)),
        read=max(1.0, timeout_seconds),
        write=60.0,
        pool=60.0,
    )
    http_client = httpx.Client(timeout=io_timeout, trust_env=False)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=io_timeout, http_client=http_client)

    sources = {s.id: s for s in load_sources()}
    personas = {p.name: p for p in load_personas()}

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
                print(f"[creval] skip missing {input_path}")
                continue
            rows = read_jsonl(input_path)
            out_path = input_path.with_name(input_path.stem.replace("_metrics", "") + "_creval.jsonl")
            if out_path.exists() and not args.overwrite:
                print(f"[creval] {out_path.name} exists; use --overwrite to rebuild")
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
                print(f"[creval] no aligned rows -> {out_path.name}")

            written = 0
            request_errors = 0
            parse_unknown = 0
            sample_printed = False
            started = time.perf_counter()
            with out_path.open("w", encoding="utf-8") as fh:
                for row, ref, src, persona in aligned_rows:
                    key = row_key(row)
                    query = build_query(row, src.text, persona.system_prompt if persona else None)
                    user_prompt = build_user_prompt(query, str(row.get("text", "")), str(ref.get("text", "")))

                    if not sample_printed:
                        print(f"[creval] sample key={key}")
                        print(f"[creval] sample query chars={len(query)} response1 chars={len(str(row.get('text', '')))} response2 chars={len(str(ref.get('text', '')))}")
                        sample_printed = True

                    if args.dry_run:
                        payload = {
                            "ok": False,
                            "reason": "dry_run",
                            "reference_model": args.reference_model,
                            "reference_metrics": args.reference_metrics.name,
                        }
                    else:
                        try:
                            raw = _call_with_retry(
                                client,
                                system_prompt=SYSTEM_PROMPT,
                                user_prompt=user_prompt,
                                model_name=model_name,
                                max_tokens=args.max_tokens,
                                retries=args.retries,
                                sleep_seconds=args.sleep_seconds,
                            )
                            verdict, score = parse_creval_output(raw)
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
                                "creval_model_name": model_name,
                            }
                        except Exception as e:
                            if args.fail_fast:
                                raise
                            request_errors += 1
                            payload = {
                                "ok": False,
                                "reason": "request_error",
                                "error": f"{type(e).__name__}: {e}",
                                "reference_model": args.reference_model,
                                "reference_metrics": args.reference_metrics.name,
                                "creval_model_name": model_name,
                            }

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
                        "creval": payload,
                    }
                    fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    written += 1
                    if args.progress_every > 0 and (written % args.progress_every == 0 or written == total_rows):
                        elapsed = time.perf_counter() - started
                        rate = written / elapsed if elapsed > 0 else 0.0
                        remaining = max(0, total_rows - written)
                        eta_seconds = (remaining / rate) if rate > 0 else 0.0
                        print(
                            f"[creval] progress {out_path.name}: {written}/{total_rows} "
                            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)}"
                        )
            print(
                f"[creval] wrote {written} rows -> {out_path} "
                f"(request_errors={request_errors}, parse_unknown={parse_unknown})"
            )
    finally:
        http_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())