"""Main experiment CLI.

Subcommands
-----------
smoke             tiny end-to-end verification (1-2 sources, skip expensive steps)
validate_corpus   check data/source_texts/ layout
coord             score all sources in Space-H (coordinate scorer)
coord_reliability genre separation / diagnostics for deterministic coords (run before main)
build_space_l     build and cache Space-L (needs API + SBERT)
main              generate T0/T1/T2/T3 records for all sources
mechanism         generate M1/M2/M3 records on high-d subset
multihop          generate evolutionary rewrite trajectories
judge             score all existing generations (absolute + pairwise)
metrics           run automatic metrics over generated records
all               run main + mechanism + multihop + judge + metrics

Usage:
    python -m src.run_experiment smoke
    python -m src.run_experiment main

Progress bars use stderr (tqdm). Set ``NO_TQDM=1`` to disable them (e.g. CI logs).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .baselines import BaselineRecord, arm_T0, arm_T1, arm_T2, arm_T3
from .config_loader import (
    PROJECT_ROOT,
    get_model_spec,
    get_role_models,
    load_experiment_config,
)
from .conflict import CoordinateScore, score_source_space_h_deterministic, score_sources_batch
from .corpus import SOURCE_DIR, SourceText, load_sources, validate_corpus
from .io_utils import GENERATED_DIR, RESULTS_DIR, append_jsonl, ensure_dirs, read_jsonl, write_jsonl
from .mechanism import MECHANISM_MODES, mechanism_cells
from .multi_hop import run_multi_hop
from .personalities import Persona, load_personas
from .progress_util import iter_progress, progress_counter


def _allow_same_family_coord() -> bool:
    """When only one API vendor works, set ``ALLOW_SAME_FAMILY_COORD=1`` to skip the coord check."""
    return os.environ.get("ALLOW_SAME_FAMILY_COORD", "").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _load_coord_scores(path: Path) -> dict[str, CoordinateScore]:
    out: dict[str, CoordinateScore] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        out[row["source_id"]] = CoordinateScore(
            source_id=row["source_id"],
            S_mean=row["S_mean"],
            S_sigma=row["S_sigma"],
            R_mean=row["R_mean"],
            R_sigma=row["R_sigma"],
        )
    return out


def _coord_scoring_backend() -> str:
    return str((load_experiment_config().get("coord_scoring") or {}).get("backend", "llm")).lower()


def _source_coords_precalc_path() -> Path:
    rel = (load_experiment_config().get("coord_scoring") or {}).get("source_coords_path") or "data/source_coords.jsonl"
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_precalc_source_coords() -> dict[str, CoordinateScore]:
    path = _source_coords_precalc_path()
    if not path.exists():
        return {}
    out: dict[str, CoordinateScore] = {}
    for row in read_jsonl(path):
        sid = row.get("id") or row.get("source_id")
        if sid is None:
            continue
        out[str(sid)] = CoordinateScore(
            source_id=str(sid),
            S_mean=float(row["S"]),
            S_sigma=float(row.get("S_sigma", 0.0)),
            R_mean=float(row["R"]),
            R_sigma=float(row.get("R_sigma", 0.0)),
        )
    return out


def _coord_scores(sources: list[SourceText], recompute: bool = False) -> dict[str, CoordinateScore]:
    path = GENERATED_DIR / "coord_scores.jsonl"
    backend = _coord_scoring_backend()
    existing = {} if recompute else _load_coord_scores(path)
    precalc = _load_precalc_source_coords() if backend == "hf" else {}
    missing = [s for s in sources if s.id not in existing]
    if missing:
        print(f"[coord] scoring {len(missing)} source(s) in Space-H ({backend}) ...")
        if backend == "hf":
            scored: list[CoordinateScore] = []
            for s in missing:
                if s.id in precalc:
                    scored.append(precalc[s.id])
                else:
                    scored.append(score_source_space_h_deterministic(s))
            for cs in scored:
                existing[cs.source_id] = cs
        else:
            batch = score_sources_batch(missing, cache_file=path.with_suffix(".tmp.jsonl"))
            for cs in batch:
                existing[cs.source_id] = cs
        write_jsonl(path, [
            {
                "source_id": cs.source_id,
                "S_mean": cs.S_mean,
                "S_sigma": cs.S_sigma,
                "R_mean": cs.R_mean,
                "R_sigma": cs.R_sigma,
            }
            for cs in existing.values()
        ])
    return existing


def _sample_continuous_target(
    rng: random.Random,
    *,
    radius_min: float = 0.05,
    radius_max: float = 1.0,
) -> np.ndarray:
    """Sample one continuous target vector in Space-H unit disk."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    r = np.sqrt(rng.uniform(radius_min * radius_min, radius_max * radius_max))
    return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=float)


def _continuous_persona_from_vec(vec: np.ndarray) -> Persona:
    s, r = float(vec[0]), float(vec[1])
    s_desc = "rational" if s >= 0 else "emotional"
    r_desc = "adventurous" if r >= 0 else "conservative"
    sys_prompt = (
        "Adopt this continuous style target in Space-H: "
        f"S={s:+.3f}, R={r:+.3f}. "
        f"Lean toward {s_desc} and {r_desc} expression proportionally to magnitudes."
    )
    return Persona(
        name=f"continuous_S{s:+.3f}_R{r:+.3f}",
        vector=vec,
        theoretical_anchor="continuous_space_h",
        system_prompt=sys_prompt,
        paraphrase_prompts=(),
        positive_markers=(),
        negative_markers=(),
    )


def _write_d_distribution_comparison(
    old_path: Path,
    new_path: Path,
    *,
    table_path: Path,
    fig_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    def _load_d(p: Path) -> np.ndarray:
        rows = list(read_jsonl(p))
        ds = [float(r.get("d_H")) for r in rows if r.get("condition") == "T3" and r.get("d_H") is not None]
        return np.array(ds, dtype=float)

    old_d = _load_d(old_path)
    new_d = _load_d(new_path)
    if len(old_d) == 0 or len(new_d) == 0:
        print("[main] skip d-distribution comparison (empty T3 d values)")
        return

    stats = pd.DataFrame(
        [
            {
                "run": "old_discrete",
                "n": int(len(old_d)),
                "mean_d": float(np.mean(old_d)),
                "std_d": float(np.std(old_d, ddof=0)),
                "unique_d_rounded_3dp": int(len(np.unique(np.round(old_d, 3)))),
                "q10": float(np.quantile(old_d, 0.10)),
                "q50": float(np.quantile(old_d, 0.50)),
                "q90": float(np.quantile(old_d, 0.90)),
            },
            {
                "run": "new_continuous",
                "n": int(len(new_d)),
                "mean_d": float(np.mean(new_d)),
                "std_d": float(np.std(new_d, ddof=0)),
                "unique_d_rounded_3dp": int(len(np.unique(np.round(new_d, 3)))),
                "q10": float(np.quantile(new_d, 0.10)),
                "q50": float(np.quantile(new_d, 0.50)),
                "q90": float(np.quantile(new_d, 0.90)),
            },
        ]
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(table_path, index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bins = min(20, max(8, int(np.sqrt(len(old_d) + len(new_d)))))
    ax.hist(old_d, bins=bins, alpha=0.45, density=True, label="old discrete")
    ax.hist(new_d, bins=bins, alpha=0.45, density=True, label="new continuous")
    ax.set_xlabel("d_H")
    ax.set_ylabel("density")
    ax.set_title("T3 d distribution: discrete vs continuous")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[main] wrote d comparison table: {table_path}")
    print(f"[main] wrote d comparison figure: {fig_path}")


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_validate_corpus(args: argparse.Namespace) -> int:
    ok, errors = validate_corpus(SOURCE_DIR, expected_per_genre=args.expected_per_genre)
    if ok:
        print(f"[validate_corpus] ok; {args.expected_per_genre}/genre present")
        return 0
    print("[validate_corpus] issues:")
    for e in errors:
        print(" -", e)
    return 1


def cmd_coord(args: argparse.Namespace) -> int:
    ensure_dirs()
    if _coord_scoring_backend() != "hf":
        generators = get_role_models("generators")
        coord_model = get_role_models("coordinate_scorer")[0]
        # Compare with PRIMARY generator family; with mixed-family generator lists
        # it is impossible for coordinate_scorer to differ from all families.
        primary_gen_family = get_model_spec(generators[0])["provider"]
        coord_family = get_model_spec(coord_model)["provider"]
        if coord_family == primary_gen_family and not _allow_same_family_coord():
            raise RuntimeError(
                "coordinate_scorer and PRIMARY generator must be from different model families. "
                f"coord={coord_model}({coord_family}), primary_generator={generators[0]}({primary_gen_family})"
            )
    sources = load_sources()
    scores = _coord_scores(sources, recompute=args.recompute)
    recs: list[dict[str, Any]] = []
    for s in iter_progress(sources, total=len(sources), desc="[coord] summary", unit="src"):
        cs = scores.get(s.id)
        if cs:
            print(f"  {s.id:20s} S={cs.S_mean:+.2f} R={cs.R_mean:+.2f}")
            recs.append({"source_id": s.id, "genre": s.genre, "S": cs.S_mean, "R": cs.R_mean})
    if recs:
        import pandas as pd

        df = pd.DataFrame(recs)
        out_csv = RESULTS_DIR / "tables" / "coord_space_h_distribution.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.5, 5.0))
            for g, grp in df.groupby("genre"):
                ax.scatter(grp["S"], grp["R"], label=g, alpha=0.75, s=28)
            ax.axhline(0, color="gray", lw=0.6)
            ax.axvline(0, color="gray", lw=0.6)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            # Short labels (full semantics in configs/personalities.yaml): +S = S2 analytic,
            # -S = S1 affective; +R = adventurous, -R = conservative.
            ax.set_xlabel("S: −1 S1 affective → +1 S2 analytic")
            ax.set_ylabel("R: −1 conservative → +1 adventurous")
            ax.set_title("Source vector distribution in Space-H")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            out_fig = RESULTS_DIR / "figures" / "coord_space_h_distribution.png"
            out_fig.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_fig, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[coord] plotting skipped: {e}")
    return 0


def cmd_coord_reliability(args: argparse.Namespace) -> int:
    """Genre separation diagnostics on deterministic coords; writes coord_reliability.csv."""
    ensure_dirs()
    from .coord_reliability import run_coord_reliability

    sources = load_sources(limit_per_genre=args.limit_per_genre)
    run_coord_reliability(sources=sources)
    return 0


def cmd_build_space_l(args: argparse.Namespace) -> int:
    from .embedding_space import build_space_l

    space = build_space_l(force_rebuild=args.force)
    print("[build_space_l] done")
    print("  explained variance:", space.pca.explained_variance_ratio_.tolist())
    print("  cross-generator Procrustes disparity:", space.cross_generator_procrustes())
    return 0


def cmd_main(args: argparse.Namespace) -> int:
    """Generate T0/T1/T2/T3 rewrites for every (source, target_persona, model, repeat) cell."""
    ensure_dirs()
    cfg = load_experiment_config()
    rng = random.Random(cfg["seed"])

    sources = load_sources(limit_per_genre=args.limit_per_genre)
    if args.genres:
        allowed = {g.strip() for g in args.genres if g and g.strip()}
        sources = [s for s in sources if s.genre in allowed]
        print(f"[main] genre filter: {sorted(allowed)} -> {len(sources)} sources")
    personas = load_personas()
    scores = _coord_scores(sources)

    generators = get_role_models("generators")
    if _coord_scoring_backend() != "hf":
        coord_model = get_role_models("coordinate_scorer")[0]
        # Compare with PRIMARY generator family; with mixed-family generator lists
        # it is impossible for coordinate_scorer to differ from all families.
        primary_gen_family = get_model_spec(generators[0])["provider"]
        coord_family = get_model_spec(coord_model)["provider"]
        if coord_family == primary_gen_family and not _allow_same_family_coord():
            raise RuntimeError(
                "coordinate_scorer and PRIMARY generator must be from different model families. "
                f"coord={coord_model}({coord_family}), primary_generator={generators[0]}({primary_gen_family})"
            )
    if args.max_generators is not None:
        generators = generators[: max(1, int(args.max_generators))]
    n_repeat = int(cfg["main"]["n_repeat"]) if not args.n_repeat else args.n_repeat

    out_name = "main.jsonl" if not args.output_name else args.output_name
    out_path = GENERATED_DIR / out_name
    if out_path.exists() and not args.overwrite and not args.append_output:
        print(f"[main] {out_path} exists; use --overwrite to rebuild")
        return 0
    if args.overwrite and not args.append_output:
        out_path.unlink(missing_ok=True)

    total_cells = 0
    continuous_rng = random.Random(cfg["seed"] + 7919)
    t3_mode = getattr(args, "target_sampling", "discrete")
    t3_cont_n = int(getattr(args, "continuous_targets_per_source", 4) or 4)
    for src in iter_progress(sources, total=len(sources), desc="[main] sources", unit="src"):
        cs = scores.get(src.id)
        if cs is None:
            print(f"  [main] skipping {src.id} (no coord score)")
            continue
        src_vec = cs.vec()
        for model in iter_progress(
            generators,
            total=len(generators),
            desc=f"[main] {src.id[:20]}",
            unit="model",
            leave=False,
        ):
            # T0 once per source per model (cheap; output is source verbatim)
            r = arm_T0(src, src_vec)
            r.model = model
            append_jsonl(out_path, r)
            total_cells += 1
            # T1 n_repeat times
            for rep in range(n_repeat):
                r = arm_T1(src, src_vec, personas, model=model, prompt_variant=0, repeat_idx=rep)
                append_jsonl(out_path, r)
                total_cells += 1
            # T2 n_repeat times (fresh rng each source for reproducibility)
            for rep in range(n_repeat):
                r = arm_T2(
                    src, src_vec, personas, model=model, prompt_variant=0,
                    repeat_idx=rep, rng=random.Random(cfg["seed"] + rep + hash(src.id) % 1000),
                )
                append_jsonl(out_path, r)
                total_cells += 1
            # T3 treatment arm: discrete personas (legacy) or continuous sampled targets.
            if t3_mode == "continuous":
                for rep in range(n_repeat):
                    for _ in range(t3_cont_n):
                        tgt_vec = _sample_continuous_target(continuous_rng)
                        target = _continuous_persona_from_vec(tgt_vec)
                        r = arm_T3(
                            src,
                            src_vec,
                            target,
                            model=model,
                            prompt_variant=0,
                            repeat_idx=rep,
                            mode="joint",
                            target_sampling="continuous",
                        )
                        append_jsonl(out_path, r)
                        total_cells += 1
            else:
                for target in personas:
                    for rep in range(n_repeat):
                        r = arm_T3(
                            src,
                            src_vec,
                            target,
                            model=model,
                            prompt_variant=0,
                            repeat_idx=rep,
                            mode="joint",
                            target_sampling="discrete",
                        )
                        append_jsonl(out_path, r)
                        total_cells += 1
    print(f"[main] wrote {total_cells} records to {out_path}")
    if args.compare_d_with:
        _write_d_distribution_comparison(
            old_path=Path(args.compare_d_with),
            new_path=out_path,
            table_path=RESULTS_DIR / "tables" / "d_distribution_discrete_vs_continuous.csv",
            fig_path=RESULTS_DIR / "figures" / "d_distribution_discrete_vs_continuous.png",
        )
    return 0


def _cmd_mechanism_patch_modes(args: argparse.Namespace, patch_modes: tuple[str, ...]) -> int:
    """Regenerate selected mechanism modes in-place in ``mechanism.jsonl`` (line order preserved)."""
    ensure_dirs()
    invalid = [m for m in patch_modes if m not in MECHANISM_MODES]
    if invalid:
        print(f"[mechanism] --patch-modes: unknown mode(s) {invalid}; allowed={MECHANISM_MODES}", file=sys.stderr)
        return 1
    if getattr(args, "overwrite", False):
        print("[mechanism] do not combine --overwrite with --patch-modes", file=sys.stderr)
        return 1

    out_path = GENERATED_DIR / "mechanism.jsonl"
    if not out_path.exists():
        print(f"[mechanism] patch: missing {out_path}", file=sys.stderr)
        return 1

    sources = {s.id: s for s in load_sources(limit_per_genre=args.limit_per_genre)}
    personas = {p.name: p for p in load_personas()}
    use_cache = not bool(getattr(args, "bypass_generation_cache", False))

    rows = list(read_jsonl(out_path))
    patch_indices = [i for i, r in enumerate(rows) if r.get("mode") in patch_modes]
    if not patch_indices:
        print("[mechanism] patch: no rows match patch_modes", file=sys.stderr)
        return 1

    print(
        f"[mechanism] patch: rewriting {len(patch_indices)} rows for modes={patch_modes} "
        f"(use_cache={use_cache}); saving after each row for resume safety"
    )
    from tqdm import tqdm

    disable = os.environ.get("NO_TQDM", "").strip().lower() in ("1", "true", "yes")
    force = bool(getattr(args, "patch_force", False))
    for i in tqdm(patch_indices, desc="[mechanism] patch", unit="row", disable=disable):
        row = rows[i]
        if row.get("mode") not in patch_modes:
            continue
        # By default skip rows already filled so a second run resumes after timeouts.
        if not force and (row.get("text") or "").strip():
            continue
        sid = row.get("source_id")
        pname = row.get("target_persona")
        src = sources.get(sid)
        pers = personas.get(pname) if pname else None
        if src is None or pers is None:
            print(f"[mechanism] patch: skip row {i} missing source/persona ({sid!r}, {pname!r})", file=sys.stderr)
            continue
        vec = np.array(row["src_vec_H"], dtype=float)
        rec = next(
            mechanism_cells(
                src,
                vec,
                pers,
                model=row["model"],
                prompt_variant=int(row.get("prompt_variant", 0)),
                repeat_idx=int(row.get("repeat_idx", 0)),
                modes=(row["mode"],),
                use_cache=use_cache,
            )
        )
        rows[i] = rec.to_dict()
        write_jsonl(out_path, rows)

    print(f"[mechanism] patch: wrote {len(rows)} rows -> {out_path}")
    return 0


def cmd_mechanism(args: argparse.Namespace) -> int:
    """M1/M2/M3 on the high-d subset."""
    patch_modes = getattr(args, "patch_modes", None)
    if patch_modes:
        return _cmd_mechanism_patch_modes(args, tuple(patch_modes))

    ensure_dirs()
    cfg = load_experiment_config()
    sources = load_sources(limit_per_genre=args.limit_per_genre)
    personas = load_personas()
    scores = _coord_scores(sources)

    generators = get_role_models("generators")
    mech_override = os.environ.get("MECHANISM_GENERATOR", "").strip()
    if mech_override:
        get_model_spec(mech_override)  # validate name
        generators = [mech_override]
        print(f"[mechanism] MECHANISM_GENERATOR={mech_override!r} -> single generator")
    elif cfg["mechanism"].get("generator_model"):
        gm = str(cfg["mechanism"]["generator_model"]).strip()
        get_model_spec(gm)
        generators = [gm]
        print(f"[mechanism] mechanism.generator_model={gm!r} -> single generator")
    elif os.environ.get("MECHANISM_GENERATORS_ONLY_MINI", "").strip().lower() in ("1", "true", "yes"):
        mini_only = [g for g in generators if "mini" in g.lower()]
        if mini_only:
            generators = mini_only
            print(f"[mechanism] MECHANISM_GENERATORS_ONLY_MINI=1 -> generators={generators}")
    n_repeat = int(cfg["mechanism"]["n_repeat"])
    modes = tuple(cfg["mechanism"].get("modes", ["M0", "M1", "M2", "M3"]))
    q = float(cfg["mechanism"]["high_d_quantile"])

    # Select high-d (source, target) pairs using Space-H
    from .conflict import compute_conflict

    pairs: list[tuple[SourceText, Any, float]] = []
    for src in sources:
        cs = scores.get(src.id)
        if cs is None:
            continue
        for p in personas:
            d = compute_conflict(cs.vec(), p.vector, space="H").d
            pairs.append((src, p, d))
    if not pairs:
        print("[mechanism] no pairs")
        return 0
    ds = sorted(d for _, _, d in pairs)
    threshold = ds[int(len(ds) * q)] if len(ds) >= 3 else ds[-1]
    high_with_d = [(s, p, d) for s, p, d in pairs if d >= threshold]
    high_with_d.sort(key=lambda t: t[2], reverse=True)
    high_with_d = high_with_d[:15]
    high = [(s, p) for s, p, d in high_with_d]
    print(f"[mechanism] high-d threshold={threshold:.3f}; pairs={len(high)} (top-15 by d_H)")

    out_path = GENERATED_DIR / "mechanism.jsonl"
    if out_path.exists() and not args.overwrite:
        print(f"[mechanism] {out_path} exists; use --overwrite to rebuild")
        return 0
    out_path.unlink(missing_ok=True)

    n = 0
    cells: list[tuple[SourceText, Persona, str, int]] = []
    for src, target in high:
        for model in generators:
            for rep in range(n_repeat):
                cells.append((src, target, model, rep))
    n_cells = len(cells)
    use_cache = not bool(getattr(args, "bypass_generation_cache", False))
    if not use_cache:
        print("[mechanism] --bypass-generation-cache: generate(use_cache=False) for all cells")

    write_lock = threading.Lock()

    def _mechanism_run_cell(cell: tuple[SourceText, Persona, str, int]) -> list[dict]:
        src, target, model, rep = cell
        cs = scores[src.id]
        return [
            rec.to_dict()
            for rec in mechanism_cells(
                src,
                cs.vec(),
                target,
                model=model,
                prompt_variant=0,
                repeat_idx=rep,
                modes=modes,
                use_cache=use_cache,
            )
        ]

    with progress_counter(total=n_cells, desc="[mechanism] (src,target)×model×rep", unit="cell") as bar:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_mechanism_run_cell, c): c for c in cells}
            for fut in as_completed(futures):
                cell = futures[fut]
                src, target, model, rep = cell
                try:
                    rows = fut.result()
                except Exception as e:
                    print(f"[mechanism] cell failed ({src.id!r}, {getattr(target, 'name', target)!r}, {model!r}, rep={rep}): {e}", file=sys.stderr)
                    with write_lock:
                        bar.update(1)
                    continue
                with write_lock:
                    for row in rows:
                        append_jsonl(out_path, row)
                        n += 1
                    bar.update(1)
                    bar.set_postfix_str(f"{src.id[:12]} {model}", refresh=False)
    print(f"[mechanism] wrote {n} records to {out_path}")
    return 0


def cmd_multihop(args: argparse.Namespace) -> int:
    ensure_dirs()
    cfg = load_experiment_config()
    paths_cfg = cfg["multihop"]["paths"]
    n_sample = int(cfg["multihop"]["n_source_sample"])

    sources = load_sources()
    personas = load_personas()
    scores = _coord_scores(sources)

    # sample n/4 per genre to keep cost bounded
    per_genre = max(1, n_sample // 4)
    rng = random.Random(cfg["seed"])
    sampled: list[SourceText] = []
    by_genre: dict[str, list[SourceText]] = {}
    for s in sources:
        by_genre.setdefault(s.genre, []).append(s)
    for g, items in by_genre.items():
        rng.shuffle(items)
        sampled.extend(items[:per_genre])

    generator = get_role_models("generators")[0]
    coord_scorer = get_role_models("coordinate_scorer")[0]

    out_path = GENERATED_DIR / "multihop.jsonl"
    if out_path.exists() and not args.overwrite:
        print(f"[multihop] {out_path} exists; use --overwrite to rebuild")
        return 0
    out_path.unlink(missing_ok=True)

    n = 0
    for src in iter_progress(sampled, total=len(sampled), desc="[multihop] sources", unit="src"):
        cs = scores.get(src.id)
        if cs is None:
            continue
        for pid, path in enumerate(paths_cfg):
            records = run_multi_hop(
                src,
                path=path,
                personas=personas,
                model=generator,
                path_id=pid,
                src_vec=cs.vec(),
                score_intermediates=not args.fast,
                coord_scorer_model=coord_scorer,
            )
            for r in records:
                append_jsonl(out_path, r)
                n += 1
    print(f"[multihop] wrote {n} records to {out_path}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    """Run absolute LLM-judge scoring on all generations in the given file(s)."""
    from .judge import aggregate_absolute_with_empty_fallback, score_absolute_dual

    in_paths = [Path(p) for p in (args.inputs or [])]
    if not in_paths:
        in_paths = [p for p in (
            GENERATED_DIR / "main.jsonl",
            GENERATED_DIR / "mechanism.jsonl",
            GENERATED_DIR / "multihop.jsonl",
        ) if p.exists()]
    if not in_paths:
        print("[judge] no input files")
        return 1

    sources = {s.id: s for s in load_sources()}
    judges = get_role_models("judges")

    resume = bool(getattr(args, "resume", False))
    judge_api_rows = getattr(args, "judge_api_rows", None)
    if judge_api_rows is not None and judge_api_rows < 0:
        print("[judge] --judge-api-rows must be >= 0", file=sys.stderr)
        return 1

    for path in in_paths:
        rows = list(read_jsonl(path))
        out_path = path.with_name(path.stem + "_judged.jsonl")
        start_idx = 0
        if resume and out_path.exists():
            existing = list(read_jsonl(out_path))
            if len(existing) > len(rows):
                print(
                    f"[judge] resume: {out_path.name} has {len(existing)} lines > "
                    f"input {len(rows)}; refusing to resume",
                    file=sys.stderr,
                )
                return 1
            if len(existing) == len(rows):
                print(f"[judge] resume: {out_path.name} already complete ({len(existing)} rows)")
                continue
            for i, (e_row, in_row) in enumerate(zip(existing, rows[: len(existing)])):
                if e_row.get("source_id") != in_row.get("source_id") or e_row.get("condition") != in_row.get(
                    "condition"
                ):
                    print(
                        f"[judge] resume: row {i} source_id/condition mismatch vs input; "
                        f"delete {out_path.name} or run without --resume",
                        file=sys.stderr,
                    )
                    return 1
                for align_k in ("repeat_idx", "model", "hop_index", "path_id"):
                    if align_k in in_row and e_row.get(align_k) != in_row.get(align_k):
                        print(
                            f"[judge] resume: row {i} field {align_k!r} mismatch; "
                            f"delete {out_path.name} or run without --resume",
                            file=sys.stderr,
                        )
                        return 1
            start_idx = len(existing)
            print(f"[judge] resume: appending from row {start_idx}/{len(rows)} -> {out_path.name}")
        else:
            out_path.unlink(missing_ok=True)

        tail = rows[start_idx:]
        if judge_api_rows is not None:
            print(
                f"[judge] API budget: real judge calls for row indices < {judge_api_rows} "
                f"(remaining rows get placeholder judge for merge)",
                file=sys.stderr,
            )
        print(f"[judge] {path.name}: {len(rows)} records -> {out_path.name}")
        for offset, row in enumerate(
            iter_progress(
                tail,
                total=len(rows),
                initial=start_idx,
                desc=f"[judge] {path.name}",
                unit="row",
            )
        ):
            idx = start_idx + offset
            sid = row.get("source_id")
            src = sources.get(sid)
            if judge_api_rows is not None and idx >= judge_api_rows:
                append_jsonl(
                    out_path,
                    {
                        **row,
                        "judge": {"ok": False, "reason": "judge_budget_skip"},
                    },
                )
                continue
            if src is None or not row.get("text"):
                append_jsonl(out_path, {**row, "judge": {"ok": False, "reason": "no src or empty text"}})
                continue
            scores = score_absolute_dual(
                src.text, row["text"], genre=src.genre, judges=judges
            )
            agg = aggregate_absolute_with_empty_fallback(scores, judges)
            append_jsonl(out_path, {**row, "judge": agg})
    return 0


def _metrics_rows_align_for_judge(prev: dict, row: dict) -> bool:
    """Line-up check when copying ``judge`` from a prior *_metrics.jsonl (see --preserve-judge-from)."""
    if prev.get("source_id") != row.get("source_id"):
        return False
    if prev.get("condition") != row.get("condition"):
        return False
    for k in ("hop_index", "path_id"):
        if k in prev and k in row and prev.get(k) != row.get(k):
            return False
    return True


def _metrics_hf_batch_size(cli_val: int | None) -> int:
    """HF pipeline batch size for NLI/sentiment (CLI overrides env ``INNOVATION_METRICS_BATCH_SIZE``)."""
    if cli_val is not None:
        return max(1, int(cli_val))
    env = os.environ.get("INNOVATION_METRICS_BATCH_SIZE", "").strip()
    if env.isdigit():
        return max(1, int(env))
    return 32


def cmd_metrics(args: argparse.Namespace) -> int:
    """Run automatic metrics (content / novelty / coherence / sentiment / structural).

    Writes `<name>_metrics.jsonl` next to each input.
    """
    from .metrics.content import nli_entailment_batch
    from .metrics.novelty import compute_genre_baselines, normalise_novelty
    from .metrics.coherence import perplexity, perplexity_to_unit
    from .metrics.sentiment import sentiment_shift_batch
    from .metrics.structural import sentence_kendall_tau, normalised_levenshtein
    from .metrics.value import combine_creativity, combine_value_arith, combine_value_geom

    batch_size = _metrics_hf_batch_size(getattr(args, "metrics_batch_size", None))
    cfg = load_experiment_config()
    sbert = get_role_models("embedder")[0] if args.with_embedding else None
    sources = {s.id: s for s in load_sources()}

    # per-genre baselines
    by_genre: dict[str, list[str]] = {}
    for s in sources.values():
        by_genre.setdefault(s.genre, []).append(s.text)
    baselines = compute_genre_baselines(by_genre, sbert_model=sbert)
    print(f"[metrics] genre baselines: {list(baselines.keys())}")

    in_paths = [Path(p) for p in (args.inputs or [])]
    if not in_paths:
        in_paths = [p for p in (
            GENERATED_DIR / "main.jsonl",
            GENERATED_DIR / "mechanism.jsonl",
        ) if p.exists()]
    preserve_per_path: list[list[dict] | None] = []
    preserve_arg = getattr(args, "preserve_judge_from", None) or []
    if preserve_arg:
        if len(preserve_arg) != len(in_paths):
            print(
                "[metrics] --preserve-judge-from must have the same number of paths as --inputs "
                f"(got {len(preserve_arg)} vs {len(in_paths)})",
                file=sys.stderr,
            )
            return 1
        for pj in preserve_arg:
            preserve_per_path.append(list(read_jsonl(Path(pj))))
    else:
        preserve_per_path = [None] * len(in_paths)

    for path_i, path in enumerate(in_paths):
        rows = list(read_jsonl(path))
        preserve_rows = preserve_per_path[path_i]
        if preserve_rows is not None and len(preserve_rows) != len(rows):
            print(
                f"[metrics] {path.name}: row count mismatch vs preserve file "
                f"({len(rows)} vs {len(preserve_rows)}) — aborting",
                file=sys.stderr,
            )
            return 1
        out_path = path.with_name(path.stem + "_metrics.jsonl")
        out_path.unlink(missing_ok=True)
        print(f"[metrics] {path.name}: {len(rows)} records -> {out_path.name} (HF batch_size={batch_size})")
        pending: list[tuple[dict, Any, str, Any]] = []

        for row in rows:
            sid = row.get("source_id")
            src = sources.get(sid)
            gen = row.get("text", "")
            if not src or not gen:
                continue
            base = baselines.get(src.genre)
            if base is None:
                continue
            pending.append((row, src, gen, base))

        nli_by_i: list[float | None] = [None] * len(pending)
        sent_by_i: list[float] = [0.0] * len(pending)
        if pending and (not args.skip_nli or not args.skip_sent):
            batch_ranges = list(range(0, len(pending), batch_size))
            for start in iter_progress(
                batch_ranges,
                total=len(batch_ranges),
                desc=f"[metrics] NLI/sent batches · {path.name}",
                unit="batch",
            ):
                chunk = pending[start : start + batch_size]
                pairs = [(src.text, gen) for _, src, gen, _ in chunk]
                if not args.skip_nli:
                    try:
                        chunk_nli = nli_entailment_batch(pairs)
                    except Exception as e:
                        sid0 = chunk[0][1].id if chunk else "?"
                        print(f"  [metrics] NLI failed on batch starting {sid0}: {e}")
                        chunk_nli = [None] * len(pairs)
                    for j, v in enumerate(chunk_nli):
                        nli_by_i[start + j] = v
                if not args.skip_sent:
                    try:
                        chunk_sent = sentiment_shift_batch(pairs)
                    except Exception as e:
                        sid0 = chunk[0][1].id if chunk else "?"
                        print(f"  [metrics] sentiment failed on batch starting {sid0}: {e}")
                        chunk_sent = [0.0] * len(pairs)
                    for j, v in enumerate(chunk_sent):
                        sent_by_i[start + j] = v

        def _attach_preserved_judge(i: int, row: dict, out: dict) -> dict:
            if preserve_rows is None or i >= len(preserve_rows):
                return out
            pr = preserve_rows[i]
            if "judge" not in pr:
                return out
            if _metrics_rows_align_for_judge(pr, row):
                out = dict(out)
                out["judge"] = pr["judge"]
            else:
                print(
                    f"  [metrics] preserve-judge line {i}: row keys do not match, skip copying judge",
                    file=sys.stderr,
                )
            return out

        pend_idx = 0
        for i, row in enumerate(
            iter_progress(rows, total=len(rows), desc=f"[metrics] per-row · {path.name}", unit="row")
        ):
            sid = row.get("source_id")
            src = sources.get(sid)
            gen = row.get("text", "")
            if not src or not gen:
                append_jsonl(
                    out_path,
                    _attach_preserved_judge(i, row, {**row, "metrics": {"ok": False}}),
                )
                continue
            base = baselines.get(src.genre)
            if base is None:
                append_jsonl(
                    out_path,
                    _attach_preserved_judge(
                        i, row, {**row, "metrics": {"ok": False, "reason": "no baseline"}}
                    ),
                )
                continue
            _, src, gen, base = pending[pend_idx]
            nli = nli_by_i[pend_idx] if not args.skip_nli else None
            sent_shift = sent_by_i[pend_idx] if not args.skip_sent else 0.0
            pend_idx += 1
            nov = normalise_novelty(src.text, gen, base, sbert_model=sbert)
            try:
                ppl = perplexity(gen) if not args.skip_ppl else float("nan")
                coh_auto = perplexity_to_unit(ppl)
            except Exception as e:
                ppl = float("nan")
                coh_auto = 0.0
                print(f"  [metrics] PPL failed on {sid}: {e}")
            ktau = sentence_kendall_tau(src.text, gen)
            lev = normalised_levenshtein(src.text, gen)
            fidelity_proxy = nli if nli is not None else coh_auto
            val = combine_value_arith(fidelity_proxy, coh_auto)
            val_geom = combine_value_geom(fidelity_proxy, coh_auto)
            crea = combine_creativity(nov.auto_combined, val)
            crea_geom = combine_creativity(nov.auto_combined, val_geom)
            out_row: dict[str, Any] = {
                **row,
                "metrics": {
                    "ok": True,
                    "nli_entailment": nli,
                    "novelty": nov.to_dict(),
                    "perplexity": ppl,
                    "coherence_auto": coh_auto,
                    "sentiment_shift": sent_shift,
                    "sentence_kendall_tau": ktau,
                    "normalised_levenshtein": lev,
                    "value_auto": val,
                    "value_auto_geom": val_geom,
                    "creativity_auto": crea,
                    "creativity_auto_geom": crea_geom,
                    "genre_baseline": base.to_dict(),
                },
            }
            append_jsonl(out_path, _attach_preserved_judge(i, row, out_row))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """End-to-end smoke: 2 sources x 2 conditions x 1 generator, no Space-L, no judge."""
    ensure_dirs()
    cfg = load_experiment_config()
    smoke_cfg = cfg["smoke"]
    personas = load_personas()
    sources = load_sources(limit_per_genre=1)[: smoke_cfg["n_sources"]]
    if not sources:
        print("[smoke] no sources found. Add files to data/source_texts/<genre>/")
        return 1

    generator = get_role_models("generators")[smoke_cfg["generator_role_index"]]
    print(f"[smoke] sources={[s.id for s in sources]} generator={generator}")

    # Coordinate scoring
    scores = _coord_scores(sources)
    for s in sources:
        cs = scores.get(s.id)
        if cs:
            print(f"  coord {s.id}: S={cs.S_mean:+.2f} R={cs.R_mean:+.2f}")

    # Generate T1 and T3 for each source; skip T0/T2/judge/Space-L
    out_path = GENERATED_DIR / "smoke.jsonl"
    out_path.unlink(missing_ok=True)
    n = 0
    for s in sources:
        cs = scores.get(s.id)
        if cs is None:
            continue
        src_vec = cs.vec()
        # T1
        r = arm_T1(s, src_vec, personas, model=generator, prompt_variant=0, repeat_idx=0)
        append_jsonl(out_path, r)
        n += 1
        # T3 farthest target (max d)
        far = personas.farthest_from(src_vec)
        r = arm_T3(s, src_vec, far, model=generator, prompt_variant=0, repeat_idx=0, mode="joint")
        append_jsonl(out_path, r)
        n += 1
        print(f"  T1+T3 for {s.id} done")

    # Run a minimal metrics pass (skip NLI/PPL/Sent to avoid downloading big models)
    from .metrics.novelty import compute_genre_baselines, normalise_novelty

    by_genre: dict[str, list[str]] = {}
    for s in load_sources():
        by_genre.setdefault(s.genre, []).append(s.text)
    baselines = compute_genre_baselines(by_genre, sbert_model=None)

    for row in list(read_jsonl(out_path)):
        src = next(x for x in sources if x.id == row["source_id"])
        base = baselines.get(src.genre)
        nov = normalise_novelty(src.text, row["text"], base, sbert_model=None)
        print(
            f"    {src.id:14s} cond={row['condition']} "
            f"target={row.get('target_persona')} "
            f"d={row['d_H']:.3f} distinct2_rel={nov.distinct2_rel:.2f} "
            f"novel_ngram_rel={nov.novel_ngram_rel:.2f}"
        )

    print(f"[smoke] wrote {n} records to {out_path}")
    print("[smoke] end-to-end OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_experiment")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("validate_corpus")
    sp.add_argument("--expected-per-genre", type=int, default=15)
    sp.set_defaults(func=cmd_validate_corpus)

    sp = sub.add_parser("coord")
    sp.add_argument("--recompute", action="store_true")
    sp.set_defaults(func=cmd_coord)

    sp = sub.add_parser("coord_reliability")
    sp.add_argument("--limit-per-genre", type=int, default=None)
    sp.set_defaults(func=cmd_coord_reliability)

    sp = sub.add_parser("build_space_l")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_build_space_l)

    sp = sub.add_parser("main")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--limit-per-genre", type=int, default=None)
    sp.add_argument("--n-repeat", type=int, default=None)
    sp.add_argument("--max-generators", type=int, default=None)
    sp.add_argument("--output-name", type=str, default=None)
    sp.add_argument("--append-output", action="store_true", help="append to output jsonl instead of replacing")
    sp.add_argument("--genres", nargs="*", default=None, help="run only specified genres (e.g. academic narrative)")
    sp.add_argument("--target-sampling", choices=["discrete", "continuous"], default="discrete")
    sp.add_argument("--continuous-targets-per-source", type=int, default=4)
    sp.add_argument("--compare-d-with", type=str, default=None)
    sp.set_defaults(func=cmd_main)

    sp = sub.add_parser("mechanism")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--limit-per-genre", type=int, default=None)
    sp.add_argument(
        "--patch-modes",
        nargs="+",
        metavar="MODE",
        default=None,
        help="regenerate only these modes in-place in mechanism.jsonl (e.g. M1); keep other lines unchanged",
    )
    sp.add_argument(
        "--bypass-generation-cache",
        action="store_true",
        help="with --patch-modes, call generate(use_cache=False) so disk cache cannot serve stale empty text",
    )
    sp.add_argument(
        "--patch-force",
        action="store_true",
        help="with --patch-modes, regenerate even when text is already non-empty (default: only fill empty rows)",
    )
    sp.set_defaults(func=cmd_mechanism)

    sp = sub.add_parser("multihop")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--fast", action="store_true", help="skip coord scoring of intermediates")
    sp.set_defaults(func=cmd_multihop)

    sp = sub.add_parser("judge")
    sp.add_argument("--inputs", nargs="*", default=None)
    sp.add_argument(
        "--resume",
        action="store_true",
        help="continue into existing *_judged.jsonl (same line order as input); skip prefix rows already written",
    )
    sp.add_argument(
        "--judge-api-rows",
        type=int,
        default=None,
        metavar="N",
        help=(
            "only call judge APIs for row indices 0..N-1 (file order); "
            "write placeholder judge for the rest so merge still matches full metrics rows (saves cost)"
        ),
    )
    sp.set_defaults(func=cmd_judge)

    sp = sub.add_parser("metrics")
    sp.add_argument("--inputs", nargs="*", default=None)
    sp.add_argument("--with-embedding", action="store_true", help="use SBERT for emb-distance novelty")
    sp.add_argument(
        "--preserve-judge-from",
        nargs="*",
        default=None,
        metavar="PATH",
        help="one prior *_metrics.jsonl per --inputs (same order); copy line-wise judge after recomputing metrics",
    )
    sp.add_argument("--skip-nli", action="store_true")
    sp.add_argument("--skip-ppl", action="store_true")
    sp.add_argument("--skip-sent", action="store_true")
    sp.add_argument(
        "--metrics-batch-size",
        type=int,
        default=None,
        help="HF NLI/sentiment pipeline batch size (default: env INNOVATION_METRICS_BATCH_SIZE or 32)",
    )
    sp.set_defaults(func=cmd_metrics)

    sp = sub.add_parser("smoke")
    sp.set_defaults(func=cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
