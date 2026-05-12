"""Qwen 4B / 8B hypothesis checks H1, H3–H7 on main_qwen3_*_metrics.jsonl.

Writes tidy rows to results/tables/qwen_h_full_analysis.csv

Run from repo root::

    python scripts/qwen_h_full_analysis.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze import load_flat  # noqa: E402
from src.config_loader import PROJECT_ROOT, cache_dir  # noqa: E402
from src.corpus import load_sources  # noqa: E402
from src.io_utils import read_jsonl  # noqa: E402
from src.metrics.jsd import word_jsd_normalized  # noqa: E402

METRIC = "creativity_auto"
MODELS = ("gen_qwen3_4b", "gen_qwen3_8b")
FILES = {
    "gen_qwen3_4b": PROJECT_ROOT / "data" / "generated" / "main_qwen3_4b_metrics.jsonl",
    "gen_qwen3_8b": PROJECT_ROOT / "data" / "generated" / "main_qwen3_8b_metrics.jsonl",
}
OUT_CSV = PROJECT_ROOT / "results" / "tables" / "qwen_h_full_analysis.csv"


def _row(
    hypothesis: str,
    generator_model: str,
    statistic: str,
    value: float | str | bool | None,
    *,
    notes: str = "",
) -> dict:
    return {
        "hypothesis": hypothesis,
        "generator_model": generator_model,
        "statistic": statistic,
        "value": value if isinstance(value, str) else (float(value) if pd.notna(value) else ""),
        "notes": notes,
    }


def _fit_mixedlm_robust(formula: str, data: pd.DataFrame, groups_col: str) -> tuple[str, dict, dict]:
    """Try MixedLM with multiple optimisers; fallback OLS same formula."""
    import statsmodels.formula.api as smf

    d = data.copy()
    model = smf.mixedlm(formula, data=d, groups=d[groups_col])
    for meth in ("lbfgs", "powell", "bfgs"):
        for reml in (True, False):
            try:
                fit = model.fit(method=meth, disp=False, reml=reml)
                coefs = {k: float(v) for k, v in fit.params.items()}
                pvals = {k: float(v) for k, v in fit.pvalues.items()}
                return f"mixedlm_{meth}_reml{int(reml)}", coefs, pvals
            except Exception:
                continue
    fit = smf.ols(formula, data=d).fit()
    coefs = {k: float(v) for k, v in fit.params.items()}
    pvals = {k: float(v) for k, v in fit.pvalues.items()}
    return "ols", coefs, pvals


def _safe_ols(formula: str, data: pd.DataFrame) -> tuple[str, dict, dict]:
    import statsmodels.formula.api as smf

    try:
        fit = smf.ols(formula, data=data).fit()
        return (
            "ols",
            {k: float(v) for k, v in fit.params.items()},
            {k: float(v) for k, v in fit.pvalues.items()},
        )
    except Exception as e:
        return "failed", {}, {"error": str(e)}


def load_arm_df(model_key: str) -> pd.DataFrame:
    path = FILES[model_key]
    base = load_flat(path)
    base["generator_key"] = model_key
    return base


def run_h1(rows: list[dict]) -> None:
    """Vertex vs empirical q03/q97 on T3."""
    for mk in MODELS:
        df = load_arm_df(mk)
        sub = df[(df["condition"] == "T3") & (df["model"] == mk)].dropna(
            subset=[METRIC, "d_H", "genre"]
        )
        if sub.empty:
            rows.append(_row("H1", mk, "error", "no_T3_rows"))
            continue
        dcol = sub["d_H"].astype(float)
        q03 = float(dcol.quantile(0.03))
        q97 = float(dcol.quantile(0.97))
        dd = sub.copy()
        dd["d"] = dd["d_H"].astype(float)
        dd["d2"] = dd["d"] ** 2
        dd["genre"] = dd["genre"].astype("category")
        import statsmodels.formula.api as smf

        fit = smf.ols(f"{METRIC} ~ d + d2 + C(genre)", data=dd).fit()
        b1 = float(fit.params.get("d", float("nan")))
        b2 = float(fit.params.get("d2", float("nan")))
        p2 = float(fit.pvalues.get("d2", float("nan")))
        vertex = float(-b1 / (2.0 * b2)) if np.isfinite(b2) and abs(b2) > 1e-12 else float("nan")
        inside = bool(np.isfinite(vertex) and q03 <= vertex <= q97)
        rows.extend(
            [
                _row("H1", mk, "empirical_d_q03", q03),
                _row("H1", mk, "empirical_d_q97", q97),
                _row("H1", mk, "fitted_vertex_d_H", vertex),
                _row("H1", mk, "beta_d_OLS_T3", b1, notes="genre FE"),
                _row("H1", mk, "beta_d2_OLS_T3", b2),
                _row("H1", mk, "p_d2_OLS_T3", p2),
                _row("H1", mk, "vertex_inside_q03_q97", inside),
                _row("H1", mk, "n_T3", int(len(sub))),
            ]
        )


def run_h3(rows: list[dict]) -> None:
    """Directional regression with source FE; fallback MixedLM without source in FE."""
    for mk in MODELS:
        df = load_arm_df(mk)
        sub = df[(df["condition"] == "T3") & (df["model"] == mk)].copy()
        sub = sub.dropna(subset=[METRIC, "dS", "dR", "genre", "source_id"])
        if len(sub) < 30:
            rows.append(_row("H3", mk, "error", "too_few_rows"))
            continue
        sub["dS2"] = sub["dS"].astype(float) ** 2
        sub["dR2"] = sub["dR"].astype(float) ** 2
        sub["dSdR"] = sub["dS"].astype(float) * sub["dR"].astype(float)
        sub["absdelta"] = np.sqrt(sub["dS"].astype(float) ** 2 + sub["dR"].astype(float) ** 2)
        sub["genre"] = sub["genre"].astype("category")
        sub["source_id"] = sub["source_id"].astype(str)

        form_fe = (
            f"{METRIC} ~ dS + dR + dS2 + dR2 + dSdR + absdelta + C(genre) + C(source_id)"
        )
        form_re = f"{METRIC} ~ dS + dR + dS2 + dR2 + dSdR + absdelta + C(genre)"

        import statsmodels.formula.api as smf

        fit_type = ""
        try:
            fit = smf.ols(form_fe, data=sub).fit()
            fit_type = "ols_source_fe"
            coefs = {k: float(v) for k, v in fit.params.items()}
            pvals = {k: float(v) for k, v in fit.pvalues.items()}
        except Exception:
            ft, coefs, pvals = _fit_mixedlm_robust(form_re, sub, "source_id")
            fit_type = ft + "_random_intercept_instead_of_source_fe"

        rows.append(_row("H3", mk, "fit_type", fit_type, notes=""))
        rows.append(_row("H3", mk, "n", int(len(sub))))
        for key in ("dS", "dR", "dS2", "dR2", "dSdR", "absdelta"):
            rows.append(_row("H3", mk, f"beta_{key}", coefs.get(key, float("nan"))))
            rows.append(_row("H3", mk, f"p_{key}", pvals.get(key, float("nan"))))


def run_h4(rows: list[dict]) -> None:
    """Per-genre beta_d2 from creativity_auto ~ d + d^2 + C(source_id) on T3."""
    for mk in MODELS:
        df = load_arm_df(mk)
        sub = df[(df["condition"] == "T3") & (df["model"] == mk)].copy()
        sub = sub.dropna(subset=[METRIC, "d_H", "genre", "source_id"])
        sub["d"] = sub["d_H"].astype(float)
        sub["source_id"] = sub["source_id"].astype(str)
        for g in sorted(sub["genre"].unique()):
            gs = sub[sub["genre"] == g]
            if len(gs) < 12:
                rows.append(_row("H4", mk, f"beta_d2_genre_{g}", "", notes="too_few_rows"))
                continue
            try:
                import statsmodels.formula.api as smf

                fit = smf.ols(f"{METRIC} ~ d + I(d ** 2) + C(source_id)", data=gs).fit()
                b2 = float(fit.params.get("I(d ** 2)", float("nan")))
                p2 = float(fit.pvalues.get("I(d ** 2)", float("nan")))
            except Exception:
                ft, coefs, pvals = _fit_mixedlm_robust(
                    f"{METRIC} ~ d + I(d ** 2)", gs, "source_id"
                )
                b2 = float(coefs.get("I(d ** 2)", float("nan")))
                p2 = float(pvals.get("I(d ** 2)", float("nan")))
                rows.append(_row("H4", mk, f"fit_type_genre_{g}", ft))
            rows.append(_row("H4", mk, f"beta_d2_genre_{g}", b2))
            rows.append(_row("H4", mk, f"p_d2_genre_{g}", p2))
            rows.append(_row("H4", mk, f"n_genre_{g}", int(len(gs))))


def run_h5(rows: list[dict]) -> None:
    """JSD ~ quadratic + genre + source FE."""
    sources = {s.id: s for s in load_sources()}
    for mk in MODELS:
        path = FILES[mk]
        recs: list[dict] = []
        for r in read_jsonl(path):
            if r.get("condition") != "T3" or r.get("model") != mk:
                continue
            m = r.get("metrics") or {}
            if not m.get("ok") or m.get("creativity_auto") is None:
                continue
            sid = r.get("source_id")
            src = sources.get(sid)
            gen_txt = (r.get("text") or "").strip()
            if src is None:
                continue
            recs.append(
                {
                    METRIC: m.get("creativity_auto"),
                    "genre": r.get("genre"),
                    "source_id": str(sid),
                    "jsd": word_jsd_normalized(src.text, gen_txt),
                }
            )
        sub = pd.DataFrame(recs)
        if sub.empty:
            rows.append(_row("H5", mk, "error", "no_rows_after_filter"))
            continue
        sub["jsd2"] = sub["jsd"].astype(float) ** 2
        sub["genre"] = sub["genre"].astype("category")
        sub["source_id"] = sub["source_id"].astype(str)

        form_fe = f"{METRIC} ~ jsd + jsd2 + C(genre) + C(source_id)"
        form_re = f"{METRIC} ~ jsd + jsd2 + C(genre)"
        import statsmodels.formula.api as smf

        try:
            fit = smf.ols(form_fe, data=sub).fit()
            ft = "ols_source_fe"
            coefs = {k: float(v) for k, v in fit.params.items()}
            pvals = {k: float(v) for k, v in fit.pvalues.items()}
        except Exception:
            ft, coefs, pvals = _fit_mixedlm_robust(form_re, sub, "source_id")
            ft = ft + "_random_intercept"

        rows.append(_row("H5", mk, "fit_type", ft))
        rows.append(_row("H5", mk, "n", int(len(sub))))
        rows.append(_row("H5", mk, "beta_jsd", coefs.get("jsd", float("nan"))))
        rows.append(_row("H5", mk, "p_jsd", pvals.get("jsd", float("nan"))))
        rows.append(_row("H5", mk, "beta_jsd2", coefs.get("jsd2", float("nan"))))
        rows.append(_row("H5", mk, "p_jsd2", pvals.get("jsd2", float("nan"))))


def run_h6(rows: list[dict]) -> None:
    """T1/T2/T3 means by global d_H tertiles."""
    for mk in MODELS:
        df = load_arm_df(mk)
        sub = df[(df["condition"].isin(["T1", "T2", "T3"])) & (df["model"] == mk)].dropna(
            subset=[METRIC, "d_H"]
        )
        if len(sub) < 30:
            rows.append(_row("H6", mk, "error", "too_few_rows"))
            continue
        try:
            sub = sub.copy()
            sub["d_bin"] = pd.qcut(sub["d_H"].astype(float), q=3, labels=["low", "mid", "high"])
        except Exception as e:
            rows.append(_row("H6", mk, "error", str(e)))
            continue
        for b in ["low", "mid", "high"]:
            for cond in ["T1", "T2", "T3"]:
                cell = sub[(sub["d_bin"] == b) & (sub["condition"] == cond)]
                m = float(cell[METRIC].mean()) if len(cell) else float("nan")
                rows.append(
                    _row(
                        "H6",
                        mk,
                        f"mean_{METRIC}_{cond}_d_bin_{b}",
                        m,
                        notes=f"n={len(cell)}",
                    )
                )
        mid_t3 = sub[(sub["d_bin"] == "mid") & (sub["condition"] == "T3")][METRIC].mean()
        mid_t2 = sub[(sub["d_bin"] == "mid") & (sub["condition"] == "T2")][METRIC].mean()
        rows.append(_row("H6", mk, "mid_bin_mean_T3_minus_T2", float(mid_t3 - mid_t2)))


def _space_l_path() -> Path:
    return cache_dir() / "space_l" / "space_l.pkl"


def run_h7(rows: list[dict]) -> None:
    """Space-L d_L same quadratic spec as H1 on T3 if cache exists."""
    path = _space_l_path()
    if not path.exists():
        rows.append(
            _row(
                "H7",
                "both",
                "status",
                "skipped_no_space_l",
                notes=f"missing {path} — run: python -m src.run_experiment build_space_l",
            )
        )
        return

    with path.open("rb") as fh:
        space = pickle.load(fh)

    persona_vecs = space.persona_vectors_L
    from itertools import combinations

    idx_range = range(persona_vecs.shape[0])
    dmx = 0.0
    for i, j in combinations(idx_range, 2):
        dmx = max(dmx, float(np.linalg.norm(persona_vecs[i] - persona_vecs[j])))
    if dmx < 1e-9:
        dmx = 1.0

    sources = {s.id: s for s in load_sources()}
    # One batched SBERT+PCA pass per unique source text (avoids thousands of HF calls).
    src_ids = sorted(sources.keys())
    src_texts = [sources[sid].text for sid in src_ids]
    try:
        proj_mat = space.project(src_texts)
    except Exception as e:
        rows.append(_row("H7", "both", "error", str(e), notes="space.project batch failed"))
        return
    proj_by_id = {sid: proj_mat[i] for i, sid in enumerate(src_ids)}

    for mk in MODELS:
        df = load_arm_df(mk)
        sub = df[(df["condition"] == "T3") & (df["model"] == mk)].copy()
        d_ls: list[float] = []
        for _, r in sub.iterrows():
            sid = r.get("source_id")
            pname = r.get("target_persona")
            if sid not in proj_by_id or not pname:
                d_ls.append(float("nan"))
                continue
            try:
                sv = np.asarray(proj_by_id[sid], dtype=float)
                tv = np.asarray(space.persona_vector(str(pname)), dtype=float)
                raw = float(np.linalg.norm(sv - tv))
                d_ls.append(float(np.clip(raw / dmx, 0.0, 1.0)))
            except Exception:
                d_ls.append(float("nan"))
        sub["d_L"] = d_ls
        sub = sub.dropna(subset=["d_L", METRIC, "genre", "source_id"])
        sub["d"] = sub["d_L"].astype(float)
        sub["d2"] = sub["d"] ** 2
        sub["genre"] = sub["genre"].astype("category")
        sub["source_id"] = sub["source_id"].astype(str)

        ft, coefs, pvals = _fit_mixedlm_robust(f"{METRIC} ~ d + d2 + C(genre)", sub, "source_id")
        rows.append(_row("H7", mk, "fit_type", ft))
        rows.append(_row("H7", mk, "n", int(len(sub))))
        rows.append(_row("H7", mk, "beta_d", coefs.get("d", float("nan"))))
        rows.append(_row("H7", mk, "beta_d2", coefs.get("d2", float("nan"))))
        rows.append(_row("H7", mk, "p_d2", pvals.get("d2", float("nan"))))


def main() -> int:
    rows: list[dict] = []

    print("=== H1 vertex / q03-q97 ===")
    run_h1(rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(pd.DataFrame([r for r in rows if r["hypothesis"] == "H1"]).to_string(index=False))

    print("\n=== H3 directional ===")
    n0 = len(rows)
    run_h3(rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(pd.DataFrame(rows[n0:]).to_string(index=False))

    print("\n=== H4 genre-specific d2 ===")
    n0 = len(rows)
    run_h4(rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(pd.DataFrame(rows[n0:]).head(40).to_string(index=False))
    if len(rows) - n0 > 40:
        print(f"... ({len(rows) - n0} H4 rows total)")

    print("\n=== H5 JSD quadratic ===")
    n0 = len(rows)
    run_h5(rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(pd.DataFrame(rows[n0:]).to_string(index=False))

    print("\n=== H6 tertile mechanism table ===")
    n0 = len(rows)
    run_h6(rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(pd.DataFrame(rows[n0:]).to_string(index=False))

    print("\n=== H7 Space-L ===")
    n0 = len(rows)
    run_h7(rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(pd.DataFrame(rows[n0:]).to_string(index=False))

    print(f"\n[qwen_h_full_analysis] wrote {OUT_CSV} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
