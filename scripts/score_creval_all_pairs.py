"""Score all rewriting results with CrEval via exhaustive pairwise comparison.

Replaces the untrained-LLM absolute-scoring judge (C_judge) with CrEval, a
fine-tuned pairwise creativity evaluator.  For each source text, all rewrites
(across models GPT-4o / Qwen3-4B / 8B / 14B and conditions T0–T3) are
compared in a round-robin tournament.  Each rewrite's win rate serves as its
final creativity score.

Default scope (repeat_idx=0 only): ~35 rewrites/source, ~595 pairs/source,
~35 700 total comparisons across 60 sources.

CrEval prompt format mirrors the official ``Aman-4-Real/CrEval/inference.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ---------------------------------------------------------------------------
# Reused from score_existing_results_with_creval.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一个语言创意评估专家。我会给你一条指令和对应的两个回复，"
    "请对两个回复的创意程度进行评估。下面是具体的数据："
)


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
    if re.search(
        r"更有创意的回复是[:：]?\s*Response\s*1\b", normalized, flags=re.IGNORECASE
    ):
        return "response_1", 1.0
    if re.search(
        r"更有创意的回复是[:：]?\s*Response\s*2\b", normalized, flags=re.IGNORECASE
    ):
        return "response_2", 0.0
    return "unknown", None


def _env_first(*keys: str, default: str) -> str:
    for key in keys:
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _probe_creval_api(base_url: str, probe_timeout: float) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/models"
    timeout = httpx.Timeout(
        connect=probe_timeout, read=probe_timeout, write=probe_timeout, pool=probe_timeout
    )
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as c:
            r = c.get(url)
        if r.status_code >= 500:
            return False, f"HTTP {r.status_code} from {url}"
        return True, f"HTTP {r.status_code} from {url}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _resolve_model_name(client: OpenAI, configured: str) -> str:
    """Auto-detect model name from the CrEval API if the configured name isn't found."""
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if configured in available:
            return configured
        if available:
            print(
                f"[creval_all_pairs] model '{configured}' not found; "
                f"using '{available[0]}' (available: {available})"
            )
            return available[0]
    except Exception as e:
        print(f"[creval_all_pairs] WARNING: could not list models: {e}")
    return configured


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _infer_once(
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
            return _infer_once(
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


# ---------------------------------------------------------------------------
# Phase 1: data loading
# ---------------------------------------------------------------------------

DEFAULT_MODEL_FILES: dict[str, Path] = {}


def _init_default_model_files() -> dict[str, Path]:
    gen = ROOT / "data" / "generated"
    return {
        "gen_openai_4o": gen / "main.jsonl",
        "gen_qwen3_4b": gen / "main_qwen3_4b.jsonl",
        "gen_qwen3_8b": gen / "main_qwen3_8b.jsonl",
        "gen_qwen3_14b": gen / "main_qwen3_14b.jsonl",
    }


DEFAULT_MODEL_FILES = _init_default_model_files()


def _safe_get(row: dict[str, Any], key: str) -> Any:
    return row.get(key)


def load_all_rewrites(
    file_paths: dict[str, Path],
    *,
    max_repeat_idx: int,
    conditions: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Read all model JSONL files and return a flat list of rewrite records.

    Each record has keys: rewrite_id, source_id, model, condition, repeat_idx,
    prompt_variant, target_persona, genre, d_H, mode, text, source_text.
    """
    sources = {s.id: s for s in load_sources()}
    rewrites: list[dict[str, Any]] = []

    for model_name, file_path in file_paths.items():
        if not file_path.exists():
            print(f"[creval_all_pairs] WARNING: skip missing {file_path}")
            continue
        with file_path.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if _safe_get(row, "model") != model_name:
                    continue
                condition = _safe_get(row, "condition")
                if condition not in conditions:
                    continue
                repeat_idx = _safe_get(row, "repeat_idx")
                if repeat_idx is None or (max_repeat_idx >= 0 and repeat_idx > max_repeat_idx):
                    continue
                text = _safe_get(row, "text")
                if not isinstance(text, str) or not text.strip():
                    continue
                sid = _safe_get(row, "source_id")
                src = sources.get(sid)
                if src is None:
                    continue

                pv = _safe_get(row, "prompt_variant")
                tp = _safe_get(row, "target_persona")
                rewrite_id = (
                    f"{sid}|{model_name}|{condition}|{repeat_idx}|{pv}|{tp or 'none'}"
                )
                rewrites.append(
                    {
                        "rewrite_id": rewrite_id,
                        "source_id": sid,
                        "model": model_name,
                        "condition": condition,
                        "repeat_idx": repeat_idx,
                        "prompt_variant": pv,
                        "target_persona": tp,
                        "genre": _safe_get(row, "genre"),
                        "d_H": _safe_get(row, "d_H"),
                        "mode": _safe_get(row, "mode"),
                        "text": text,
                        "source_text": src.text,
                    }
                )

    # Report
    by_model_source: dict[str, set[str]] = defaultdict(set)
    for r in rewrites:
        by_model_source[r["model"]].add(r["source_id"])
    for model_name in sorted(by_model_source):
        src_count = len(by_model_source[model_name])
        row_count = sum(1 for r in rewrites if r["model"] == model_name)
        print(
            f"[creval_all_pairs] loaded {model_name}: {row_count} rewrites "
            f"across {src_count} sources"
        )
    print(f"[creval_all_pairs] total rewrites: {len(rewrites)}")
    return rewrites


# ---------------------------------------------------------------------------
# Phase 2: pair generation
# ---------------------------------------------------------------------------


def _pair_id(rid_a: str, rid_b: str, swapped: bool) -> str:
    raw = f"{rid_a}|{rid_b}|{int(swapped)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def generate_pairs(
    rewrites: list[dict[str, Any]],
    *,
    max_pairs_per_source: int | None,
    sample_pairs: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate all pairwise comparisons within each source_id group.

    Randomises Response-1-vs-2 ordering to prevent positional bias.
    """
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rewrites:
        by_source[r["source_id"]].append(r)

    pairs: list[dict[str, Any]] = []
    for sid, group in sorted(by_source.items()):
        combs = list(itertools.combinations(group, 2))
        if max_pairs_per_source is not None and len(combs) > max_pairs_per_source:
            if sample_pairs:
                combs = rng.sample(combs, max_pairs_per_source)
            else:
                combs = combs[:max_pairs_per_source]

        for ra, rb in combs:
            swapped = rng.random() < 0.5
            if swapped:
                text_a, text_b = rb["text"], ra["text"]
            else:
                text_a, text_b = ra["text"], rb["text"]
            pairs.append(
                {
                    "pair_id": _pair_id(ra["rewrite_id"], rb["rewrite_id"], swapped),
                    "source_id": sid,
                    "rewrite_a": {
                        "rewrite_id": ra["rewrite_id"],
                        "model": ra["model"],
                        "condition": ra["condition"],
                        "repeat_idx": ra["repeat_idx"],
                        "prompt_variant": ra["prompt_variant"],
                    },
                    "rewrite_b": {
                        "rewrite_id": rb["rewrite_id"],
                        "model": rb["model"],
                        "condition": rb["condition"],
                        "repeat_idx": rb["repeat_idx"],
                        "prompt_variant": rb["prompt_variant"],
                    },
                    "text_a": text_a,
                    "text_b": text_b,
                    "order_swapped": swapped,
                }
            )

    per_source = defaultdict(int)
    for p in pairs:
        per_source[p["source_id"]] += 1
    counts = sorted(per_source.values())
    print(
        f"[creval_all_pairs] generated {len(pairs)} pairs across "
        f"{len(per_source)} sources "
        f"(min={counts[0] if counts else 0}, "
        f"median={counts[len(counts)//2] if counts else 0}, "
        f"max={counts[-1] if counts else 0} per source)"
    )
    return pairs


# ---------------------------------------------------------------------------
# Phase 3: CrEval scoring
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["pair_id"])
                except (KeyError, json.JSONDecodeError):
                    continue
    except Exception:
        return set()
    return ids


def _score_single(
    pair: dict[str, Any],
    client: OpenAI,
    sources: dict[str, Any],
    model_name: str,
    max_tokens: int,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    """Score one pair.  Called from worker threads."""
    src = sources.get(pair["source_id"])
    if src is None:
        return {
            "pair_id": pair["pair_id"],
            "ok": False,
            "reason": "missing_source",
            "score": None,
            "verdict": None,
            "raw": "",
            "error": f"source {pair['source_id']} not found",
        }

    query = src.text
    try:
        raw = _call_with_retry(
            client,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(query, pair["text_a"], pair["text_b"]),
            model_name=model_name,
            max_tokens=max_tokens,
            retries=retries,
            sleep_seconds=sleep_seconds,
        )
        verdict, parsed = parse_creval_output(raw)

        # De-swap: the API answer is about Response 1 vs Response 2.
        # text_a / text_b were already swapped during pair generation; undo
        # that here so the score always reflects rewrite_a vs rewrite_b.
        if pair.get("order_swapped") and parsed is not None and parsed != 0.5:
            parsed = 1.0 - parsed
            if verdict == "response_1":
                verdict = "response_2"
            elif verdict == "response_2":
                verdict = "response_1"

        return {
            "pair_id": pair["pair_id"],
            "ok": parsed is not None,
            "reason": "parse_unknown" if parsed is None else "ok",
            "score": parsed,
            "verdict": verdict,
            "raw": raw,
        }
    except Exception as e:
        return {
            "pair_id": pair["pair_id"],
            "ok": False,
            "reason": "request_error",
            "score": None,
            "verdict": None,
            "raw": "",
            "error": f"{type(e).__name__}: {e}",
        }


def run_creval_scoring(
    pairs: list[dict[str, Any]],
    sources: dict[str, Any],
    output_path: Path,
    client: OpenAI,
    model_name: str,
    *,
    max_tokens: int,
    retries: int,
    sleep_seconds: float,
    concurrency: int,
    progress_every: int,
    fail_fast: bool,
) -> tuple[int, int, int]:
    """Call CrEval for every pair.  Resumable; writes incrementally.

    Returns (written, request_errors, parse_unknown) counts.
    """
    completed_ids = _load_completed_ids(output_path)
    pending = [p for p in pairs if p["pair_id"] not in completed_ids]
    total_pairs = len(pairs)
    already_done = total_pairs - len(pending)

    if already_done:
        print(
            f"[creval_all_pairs] resuming: {already_done}/{total_pairs} already scored"
        )

    if not pending:
        print("[creval_all_pairs] all pairs already scored")
        # count errors in existing file
        req_err = 0
        parse_unk = 0
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        cr = row.get("creval", {})
                        if cr.get("reason") == "request_error":
                            req_err += 1
                        elif cr.get("reason") == "parse_unknown":
                            parse_unk += 1
                    except json.JSONDecodeError:
                        continue
        return total_pairs, req_err, parse_unk

    total_pending = len(pending)
    written = 0
    request_errors = 0
    parse_unknown = 0
    started = time.perf_counter()

    if concurrency <= 1:
        # Sequential path — simpler, no lock needed
        for pair in pending:
            result = _score_single(
                pair, client, sources, model_name, max_tokens, retries, sleep_seconds
            )
            _write_result(output_path, pair, result)
            written += 1
            if not result["ok"]:
                if result.get("reason") == "request_error":
                    request_errors += 1
                    if fail_fast:
                        raise RuntimeError(
                            f"request failed: {result.get('error', 'unknown')}"
                        )
                elif result.get("reason") == "parse_unknown":
                    parse_unknown += 1
            _report_progress(
                written + already_done,
                total_pairs,
                started,
                output_path.name,
                progress_every,
            )
    else:
        # Concurrent path
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_pair = {
                executor.submit(
                    _score_single,
                    pair,
                    client,
                    sources,
                    model_name,
                    max_tokens,
                    retries,
                    sleep_seconds,
                ): pair
                for pair in pending
            }
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "pair_id": pair["pair_id"],
                        "ok": False,
                        "reason": "request_error",
                        "score": None,
                        "verdict": None,
                        "raw": "",
                        "error": f"{type(e).__name__}: {e}",
                    }
                _write_result(output_path, pair, result)
                written += 1
                if not result["ok"]:
                    if result.get("reason") == "request_error":
                        request_errors += 1
                        if fail_fast:
                            raise RuntimeError(
                                f"request failed: {result.get('error', 'unknown')}"
                            )
                    elif result.get("reason") == "parse_unknown":
                        parse_unknown += 1
                _report_progress(
                    written + already_done,
                    total_pairs,
                    started,
                    output_path.name,
                    progress_every,
                )

    return written, request_errors, parse_unknown


def _write_result(
    output_path: Path, pair: dict[str, Any], result: dict[str, Any]
) -> None:
    row = {
        "pair_id": result["pair_id"],
        "source_id": pair["source_id"],
        "rewrite_a": pair["rewrite_a"],
        "rewrite_b": pair["rewrite_b"],
        "order_swapped": pair["order_swapped"],
        "creval": {
            "ok": result["ok"],
            "reason": result.get("reason", "unknown"),
            "score": result["score"],
            "verdict": result["verdict"],
            "raw": result.get("raw", ""),
        },
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"
    if concurrency_active():
        with _write_lock:
            _append_line(output_path, line)
    else:
        _append_line(output_path, line)


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


_concurrency_mode = False


def concurrency_active() -> bool:
    return _concurrency_mode


def _report_progress(
    done: int, total: int, started: float, name: str, every: int
) -> None:
    if every <= 0:
        return
    if done % every != 0 and done != total:
        return
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - done)
    eta = (remaining / rate) if rate > 0 else 0.0
    print(
        f"[creval_all_pairs] {name}: {done}/{total} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
        f"({rate * 60:.1f} pairs/min)"
    )


# ---------------------------------------------------------------------------
# Phase 4: win-rate computation
# ---------------------------------------------------------------------------


def compute_winrates(
    results_path: Path,
    winrates_path: Path,
    rewrite_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate pairwise results into per-rewrite win rates.

    win_rate = (wins + 0.5 * ties) / n_comparisons
    """
    if not results_path.exists():
        print("[creval_all_pairs] no results file; skipping winrate computation")
        return []

    # Per-rewrite accumulators
    acc: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"wins": 0, "ties": 0, "losses": 0, "n": 0}
    )

    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cr = row.get("creval", {})
            if not cr.get("ok"):
                continue
            score = cr.get("score")
            if score is None:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue

            rid_a = row["rewrite_a"]["rewrite_id"]
            rid_b = row["rewrite_b"]["rewrite_id"]

            # score = 1.0 means rewrite_a wins (Response 1 was judged more creative)
            # score = 0.0 means rewrite_b wins
            # score = 0.5 means tie
            if score == 1.0:
                acc[rid_a]["wins"] = int(acc[rid_a]["wins"]) + 1
                acc[rid_b]["losses"] = int(acc[rid_b]["losses"]) + 1
            elif score == 0.0:
                acc[rid_a]["losses"] = int(acc[rid_a]["losses"]) + 1
                acc[rid_b]["wins"] = int(acc[rid_b]["wins"]) + 1
            else:
                acc[rid_a]["ties"] = int(acc[rid_a]["ties"]) + 1
                acc[rid_b]["ties"] = int(acc[rid_b]["ties"]) + 1
            acc[rid_a]["n"] = int(acc[rid_a]["n"]) + 1
            acc[rid_b]["n"] = int(acc[rid_b]["n"]) + 1

    out_rows: list[dict[str, Any]] = []
    for rid, counts in sorted(acc.items()):
        n = int(counts["n"])
        wins = int(counts["wins"])
        ties = int(counts["ties"])
        losses = int(counts["losses"])
        win_rate = (wins + 0.5 * ties) / n if n > 0 else float("nan")

        rw = rewrite_lookup.get(rid)
        out_rows.append(
            {
                "rewrite_id": rid,
                "source_id": rw["source_id"] if rw else rid.split("|")[0],
                "model": rw["model"] if rw else rid.split("|")[1],
                "condition": rw["condition"] if rw else rid.split("|")[2],
                "repeat_idx": rw["repeat_idx"] if rw else None,
                "prompt_variant": rw["prompt_variant"] if rw else None,
                "target_persona": rw["target_persona"] if rw else None,
                "genre": rw["genre"] if rw else None,
                "d_H": rw["d_H"] if rw else None,
                "n_comparisons": n,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "creval_winrate": win_rate,
            }
        )

    # Write winrates file
    winrates_path.parent.mkdir(parents=True, exist_ok=True)
    with winrates_path.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary stats
    if out_rows:
        rates = [r["creval_winrate"] for r in out_rows if r["creval_winrate"] == r["creval_winrate"]]
        comps = [r["n_comparisons"] for r in out_rows]
        print(
            f"[creval_all_pairs] winrates: {len(out_rows)} rewrites, "
            f"{len(rates)} with valid scores, "
            f"mean_winrate={sum(rates)/len(rates):.4f}, "
            f"mean_comparisons={sum(comps)/len(comps):.1f}"
        )

    return out_rows


# ---------------------------------------------------------------------------
# Phase 5: merge into metrics
# ---------------------------------------------------------------------------


def merge_winrates_into_metrics(
    winrates: list[dict[str, Any]],
    metrics_dir: Path,
    *,
    field_name: str,
    dry_run: bool,
) -> None:
    """Add creval_winrate to each matching row in the *_metrics.jsonl files.

    Writes split per-model winrate files then merges via
    merge_external_eval_into_metrics.py.
    """
    # Split winrates by model
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metrics_file_map = {
        "gen_openai_4o": "main_metrics.jsonl",
        "gen_qwen3_4b": "main_qwen3_4b_metrics.jsonl",
        "gen_qwen3_8b": "main_qwen3_8b_metrics.jsonl",
        "gen_qwen3_14b": "main_qwen3_14b_metrics.jsonl",
    }
    for wr in winrates:
        model = wr.get("model", "")
        fname = metrics_file_map.get(model)
        if fname is None:
            print(f"[creval_all_pairs] WARNING: unknown model {model}, skipping merge")
            continue
        by_file[fname].append(wr)

    from scripts.merge_external_eval_into_metrics import merge

    for fname, rows in sorted(by_file.items()):
        metrics_path = metrics_dir / fname
        if not metrics_path.exists():
            print(f"[creval_all_pairs] WARNING: {metrics_path} not found, skipping")
            continue

        # Write temporary external file
        ext_path = metrics_dir / fname.replace(".jsonl", "_creval_winrate_ext.jsonl")
        ext_rows = []
        for wr in rows:
            ext_rows.append(
                {
                    "source_id": wr["source_id"],
                    "condition": wr["condition"],
                    "mode": None,  # will be matched by merge
                    "repeat_idx": wr["repeat_idx"],
                    "model": wr["model"],
                    "hop_index": None,
                    "path_id": None,
                    "target_persona": wr["target_persona"],
                    "prompt_variant": wr["prompt_variant"],
                    field_name: {
                        "ok": True,
                        "score": wr["creval_winrate"],
                        "n_comparisons": wr["n_comparisons"],
                        "wins": wr["wins"],
                        "ties": wr["ties"],
                        "losses": wr["losses"],
                    },
                }
            )

        ext_path.parent.mkdir(parents=True, exist_ok=True)
        with ext_path.open("w", encoding="utf-8") as fh:
            for row in ext_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if dry_run:
            print(
                f"[creval_all_pairs] DRY-RUN: would merge {len(ext_rows)} winrates "
                f"into {metrics_path} as '{field_name}'"
            )
            ext_path.unlink(missing_ok=True)
            continue

        # Merge into metrics (overwrites in-place)
        n = merge(
            metrics_path,
            ext_path,
            metrics_path,  # overwrite
            field_name=field_name,
            score_key="score",
            allow_missing=True,
            allow_extra=True,
        )
        print(f"[creval_all_pairs] merged {n} rows into {metrics_path}")
        ext_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Phase control
    p.add_argument(
        "--skip-scoring",
        action="store_true",
        help="skip CrEval API calls; only compute winrates",
    )
    p.add_argument(
        "--skip-winrates",
        action="store_true",
        help="skip winrate computation; only do scoring",
    )
    p.add_argument(
        "--skip-merge",
        action="store_true",
        help="skip merging winrates into metrics files",
    )

    # Scoping
    p.add_argument(
        "--max-repeat-idx",
        type=int,
        default=0,
        help="max repeat_idx (0 = fastest, -1 = all)",
    )
    p.add_argument(
        "--conditions",
        nargs="*",
        default=["T0", "T1", "T2", "T3"],
        help="conditions to include",
    )
    p.add_argument(
        "--max-pairs-per-source",
        type=int,
        default=None,
        help="cap on pairs per source",
    )
    p.add_argument(
        "--sample-pairs",
        action="store_true",
        help="random sample when max-pairs-per-source active",
    )

    # Model selection
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="models to include (default: all)",
    )

    # CrEval API
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--sleep-seconds", type=float, default=1.5)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--probe-timeout", type=float, default=3.0)
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument("--fail-fast", action="store_true")

    # I/O
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "generated",
    )
    p.add_argument("--results-file", type=Path, default=None)
    p.add_argument("--winrates-file", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--field-name",
        default="creval_winrate",
        help="target field name on merged metrics rows",
    )
    args = p.parse_args()

    # Resolve paths
    results_path = args.results_file or (
        args.output_dir / "creval_all_pairs_results.jsonl"
    )
    winrates_path = args.winrates_file or (
        args.output_dir / "creval_winrates.jsonl"
    )

    # Filter model files
    file_paths = dict(DEFAULT_MODEL_FILES)
    if args.models:
        file_paths = {m: p for m, p in file_paths.items() if m in args.models}

    show_conditions = args.conditions
    if not show_conditions:
        show_conditions = ["T0", "T1", "T2", "T3"]
    print(
        f"[creval_all_pairs] models={list(file_paths.keys())} "
        f"conditions={show_conditions} "
        f"max_repeat_idx={args.max_repeat_idx}"
    )

    # Phase 1: load
    rewrites = load_all_rewrites(
        file_paths,
        max_repeat_idx=args.max_repeat_idx,
        conditions=tuple(args.conditions) if args.conditions else ("T0", "T1", "T2", "T3"),
    )
    if not rewrites:
        print("[creval_all_pairs] no rewrites loaded; aborting")
        return 1

    # Phase 2: generate pairs
    pairs = generate_pairs(
        rewrites,
        max_pairs_per_source=args.max_pairs_per_source,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
    )
    if not pairs:
        print("[creval_all_pairs] no pairs generated; aborting")
        return 1

    if args.dry_run:
        # Print per-source breakdown
        by_src = defaultdict(int)
        for p in pairs:
            by_src[p["source_id"]] += 1
        print("[creval_all_pairs] DRY-RUN: per-source pair counts:")
        for sid in sorted(by_src):
            print(f"  {sid}: {by_src[sid]}")
        print(f"[creval_all_pairs] DRY-RUN total: {len(pairs)} pairs")
        return 0

    # Phase 3: CrEval scoring
    if not args.skip_scoring:
        if args.overwrite and results_path.exists():
            results_path.unlink()

        api_key = _env_first("CREVAL_API_KEY", default="0")
        base_url = _env_first(
            "CREVAL_BASE_URL", "CREVAL_API_BASE_URL", default="http://127.0.0.1:8000/v1"
        )
        model_name = _env_first(
            "CREVAL_MODEL_NAME", "CREVAL_API_MODEL", default="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        timeout_seconds = float(_env_first("CREVAL_API_TIMEOUT", default="1200"))

        if not args.skip_probe:
            ok, detail = _probe_creval_api(base_url, max(1.0, args.probe_timeout))
            if not ok:
                raise SystemExit(
                    "[creval_all_pairs] API probe failed: "
                    f"{detail}.\n"
                    "Check that the CrEval API server is running in WSL, "
                    f"and that CREVAL_BASE_URL={base_url} is reachable."
                )
            print(f"[creval_all_pairs] API probe ok: {detail}")

        io_timeout = httpx.Timeout(
            connect=min(10.0, max(1.0, timeout_seconds)),
            read=max(1.0, timeout_seconds),
            write=60.0,
            pool=60.0,
        )
        http_client = httpx.Client(timeout=io_timeout, trust_env=False)
        client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=io_timeout, http_client=http_client
        )

        # Pre-load sources for query building
        sources = {s.id: s for s in load_sources()}

        # Auto-resolve model name
        resolved_model = _resolve_model_name(client, model_name)

        # Set concurrency flag used by _writeResult
        global _concurrency_mode
        _concurrency_mode = args.concurrency > 1

        try:
            written, req_err, parse_unk = run_creval_scoring(
                pairs,
                sources,
                results_path,
                client,
                resolved_model,
                max_tokens=args.max_tokens,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                concurrency=args.concurrency,
                progress_every=args.progress_every,
                fail_fast=args.fail_fast,
            )
            print(
                f"[creval_all_pairs] scoring done: {written} new rows "
                f"(request_errors={req_err}, parse_unknown={parse_unk})"
            )
        finally:
            http_client.close()

    # Phase 4: compute winrates
    rewrite_lookup = {r["rewrite_id"]: r for r in rewrites}
    wr_rows: list[dict[str, Any]] = []
    if not args.skip_winrates:
        wr_rows = compute_winrates(results_path, winrates_path, rewrite_lookup)
        if not wr_rows:
            print("[creval_all_pairs] no winrates computed")

    # Phase 5: merge into metrics
    if not args.skip_merge:
        # Reuse Phase 4 results; re-read from file if phase 4 was skipped
        if not wr_rows and winrates_path.exists():
            wr_rows = [
                json.loads(line)
                for line in winrates_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if wr_rows:
            merge_winrates_into_metrics(
                wr_rows,
                args.output_dir,
                field_name=args.field_name,
                dry_run=args.dry_run,
            )
        else:
            print("[creval_all_pairs] no winrates to merge")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
