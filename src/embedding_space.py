"""Space-L: the data-driven embedding space.

Pipeline
--------
1. Generate N_ref short texts per persona, over N_topics neutral seeds.
   (per generator family, so we can audit cross-model consistency.)
2. Embed all texts with SBERT (BAAI/bge-large-en-v1.5 default).
3. Aggregate: v_P = mean(embeddings for P) over all topics.
4. Reduce: PCA over {v_P : P in personas, generator in generators}.
5. Align with Space-H via Procrustes.

The module exposes:
  build_space_l()                      -> SpaceL  (cached on disk)
  SpaceL.project(text: str)            -> np.ndarray
  SpaceL.persona_vector(name)          -> np.ndarray
  SpaceL.procrustes_to_h(PersonaSet)   -> dict(stats)

All intermediate tensors are cached under .cache/space_l/.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

from .config_loader import (
    PROJECT_ROOT,
    cache_dir,
    env,
    get_model_spec,
    get_role_models,
    load_experiment_config,
)
from .generator import generate
from .personalities import PersonaSet, load_personas

from .hf_device import torch_device_str
from .progress_util import iter_progress, progress_counter, tqdm_disabled


# ---------------------------------------------------------------------------
# SBERT wrapper (lazy; keeps import-time light)
# ---------------------------------------------------------------------------

_SBERT = None


def _openai_like_embed(
    api_key: str,
    base_url: str | None,
    model_id: str,
    texts: list[str],
) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    out: list[np.ndarray] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model_id, input=chunk)
        vecs = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
        out.extend(vecs)
    arr = np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)
    if arr.size:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-8, None)
    return arr


def _maybe_embed_via_api(model_name: str, texts: list[str]) -> np.ndarray | None:
    try:
        spec = get_model_spec(model_name)
    except Exception:
        return None

    provider = spec.get("provider")
    model_id = spec.get("model_id")
    if provider != "openai" or not model_id:
        return None

    api_key = env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI embedding models")
    base_url = env("OPENAI_BASE_URL")
    return _openai_like_embed(api_key, base_url, model_id, texts)


def _sbert_model(name: str):
    global _SBERT
    if _SBERT is None or getattr(_SBERT, "_model_name", None) != name:
        from sentence_transformers import SentenceTransformer

        _SBERT = SentenceTransformer(name)
        _SBERT._model_name = name
        # Hard cap for cross-encoder-style BERT stacks (BGE default is often 512).
        try:
            _SBERT.max_seq_length = min(int(getattr(_SBERT, "max_seq_length", 512)), 512)
        except Exception:
            pass
        _SBERT.to(torch_device_str())
    return _SBERT


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """Encode a batch of strings to a (n, d) float32 array (L2-normalised)."""
    clipped = [(t or "").strip() or " " for t in texts]
    clipped = [t if len(t) <= 50000 else t[:50000] for t in clipped]
    api_vecs = _maybe_embed_via_api(model_name, clipped)
    if api_vecs is not None:
        return api_vecs

    model = _sbert_model(model_name)
    # Avoid pathological lengths that can destabilise GPU kernels after other HF models.
    arr = model.encode(
        clipped,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=not tqdm_disabled(),
    )
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Reference corpus generation
# ---------------------------------------------------------------------------


def _load_seed_topics() -> list[str]:
    cfg = load_experiment_config()
    path = PROJECT_ROOT / cfg["space_l"]["seed_topics_file"]
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return list(data["topics"])


def generate_reference_corpus(
    personas: PersonaSet,
    generator_model: str,
    *,
    n_per_persona: int,
    topics: list[str] | None = None,
    max_tokens: int = 180,
) -> dict[str, list[str]]:
    """For each persona, generate `n_per_persona` short passages covering
    the neutral seed topics. Returns {persona_name: [text, ...]}.

    If len(topics) < n_per_persona we loop over topics with distinct seed
    temperatures so cached entries remain unique.
    """
    if topics is None:
        topics = _load_seed_topics()
    out: dict[str, list[str]] = {p.name: [] for p in personas}
    base_temps = [0.5, 0.7, 0.9]
    total_target = len(personas) * n_per_persona
    with progress_counter(total=total_target, desc="[space_l] reference LLM texts", unit="txt") as bar:
        for persona in personas:
            needed = n_per_persona
            k = 0
            while needed > 0:
                topic = topics[k % len(topics)]
                temp = base_temps[(k // len(topics)) % len(base_temps)]
                user = f"Write a 80-120 word passage on: {topic}"
                res = generate(
                    generator_model,
                    system=persona.system_prompt,
                    user=user,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                text = (res.text or "").strip()
                if text:
                    out[persona.name].append(text)
                    needed -= 1
                    bar.update(1)
                    bar.set_postfix_str(persona.name[:20], refresh=False)
                k += 1
                # safety bound
                if k > n_per_persona * 4:
                    break
    return out


# ---------------------------------------------------------------------------
# SpaceL dataclass
# ---------------------------------------------------------------------------


@dataclass
class SpaceL:
    persona_names: list[str]
    generator_models: list[str]
    # v_P_per_gen[gen_idx][persona_idx] = (d_sbert,)
    persona_embeddings: np.ndarray  # (n_gen, n_persona, d_sbert)
    pca: PCA
    k: int
    sbert_model_name: str
    # cached mean persona vectors in L-space (averaged across generators)
    persona_vectors_L: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        mean_sbert = self.persona_embeddings.mean(axis=0)  # (n_persona, d_sbert)
        self.persona_vectors_L = self.pca.transform(mean_sbert)

    def project(self, texts: list[str] | str) -> np.ndarray:
        """Embed texts with SBERT and project into Space-L."""
        if isinstance(texts, str):
            texts = [texts]
        sbert = embed(texts, self.sbert_model_name)
        return self.pca.transform(sbert)

    def persona_vector(self, name: str) -> np.ndarray:
        idx = self.persona_names.index(name)
        return self.persona_vectors_L[idx]

    # ---- cross-generator consistency: Procrustes between generators ----
    def cross_generator_procrustes(self) -> float:
        """Mean pairwise Procrustes disparity across generators (lower = more consistent)."""
        n_gen = self.persona_embeddings.shape[0]
        if n_gen < 2:
            return float("nan")
        L_per_gen = [self.pca.transform(self.persona_embeddings[i]) for i in range(n_gen)]
        ds = []
        for i in range(n_gen):
            for j in range(i + 1, n_gen):
                _, _, d = procrustes(L_per_gen[i], L_per_gen[j])
                ds.append(d)
        return float(np.mean(ds))

    # ---- alignment with Space-H ----
    def procrustes_to_h(self, personas: PersonaSet) -> dict[str, Any]:
        """Orthogonal Procrustes alignment between Space-L persona vectors
        and Space-H persona vectors. Reports:
            - disparity (lower = better)
            - correlation of L-axis-i with H-axis-S and H-axis-R
        """
        H = personas.vectors  # (n_persona, 2)
        L = self.persona_vectors_L[:, : H.shape[1]] if self.k >= H.shape[1] else self.persona_vectors_L
        # pad H if k > 2 so dims match
        if self.k > H.shape[1]:
            H_padded = np.zeros((H.shape[0], self.k))
            H_padded[:, : H.shape[1]] = H
            H_use = H_padded
            L_use = self.persona_vectors_L
        else:
            H_use = H
            L_use = L
        mtx1, mtx2, disparity = procrustes(H_use, L_use)
        # Correlation of each PC axis with each H axis
        corrs: dict[str, float] = {}
        for i in range(self.k):
            for j, hname in enumerate(["S", "R"]):
                if j >= H.shape[1]:
                    continue
                c = np.corrcoef(self.persona_vectors_L[:, i], H[:, j])[0, 1]
                corrs[f"PC{i+1}_vs_{hname}"] = float(c)
        return {
            "disparity": float(disparity),
            "axis_correlations": corrs,
            "aligned_H": mtx1.tolist(),
            "aligned_L": mtx2.tolist(),
        }


# ---------------------------------------------------------------------------
# Build / save / load
# ---------------------------------------------------------------------------


def _cache_path() -> Path:
    d = cache_dir() / "space_l"
    d.mkdir(parents=True, exist_ok=True)
    return d / "space_l.pkl"


def build_space_l(
    *,
    personas: PersonaSet | None = None,
    generator_models: list[str] | None = None,
    n_per_persona: int | None = None,
    k: int | None = None,
    force_rebuild: bool = False,
) -> SpaceL:
    cache_file = _cache_path()
    if cache_file.exists() and not force_rebuild:
        with cache_file.open("rb") as fh:
            return pickle.load(fh)

    cfg = load_experiment_config()
    if personas is None:
        personas = load_personas()
    if generator_models is None:
        generator_models = get_role_models("generators")
    if n_per_persona is None:
        n_per_persona = int(cfg["space_l"]["n_ref_per_persona"])
    if k is None:
        k = int(cfg["space_l"]["default_pca_dim"])
    embedder_name = get_role_models("embedder")[0]

    topics = _load_seed_topics()
    all_embeds: list[np.ndarray] = []  # per generator
    for gen_name in iter_progress(
        list(generator_models),
        total=len(generator_models),
        desc="[space_l] generators→PCA",
        unit="gen",
    ):
        refs = generate_reference_corpus(
            personas,
            gen_name,
            n_per_persona=n_per_persona,
            topics=topics,
        )
        gen_mat = []
        for p in personas:
            vecs = embed(refs[p.name], embedder_name) if refs[p.name] else np.zeros((1, 1024), dtype=np.float32)
            gen_mat.append(vecs.mean(axis=0))
        all_embeds.append(np.stack(gen_mat))  # (n_persona, d_sbert)

    persona_embeddings = np.stack(all_embeds)  # (n_gen, n_persona, d_sbert)
    flat = persona_embeddings.reshape(-1, persona_embeddings.shape[-1])
    pca = PCA(n_components=k, random_state=cfg["seed"])
    pca.fit(flat)

    space = SpaceL(
        persona_names=[p.name for p in personas],
        generator_models=list(generator_models),
        persona_embeddings=persona_embeddings,
        pca=pca,
        k=k,
        sbert_model_name=embedder_name,
    )
    with cache_file.open("wb") as fh:
        pickle.dump(space, fh)

    # Also write a JSON summary for human inspection
    summary = {
        "persona_names": space.persona_names,
        "generators": space.generator_models,
        "k": k,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "persona_vectors_L": space.persona_vectors_L.tolist(),
        "cross_generator_procrustes": space.cross_generator_procrustes(),
        "procrustes_to_h": space.procrustes_to_h(personas),
    }
    with (cache_file.parent / "space_l_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return space
