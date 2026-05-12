"""Deterministic Space-H style coordinates from HuggingFace models (no LLM).

S axis: formality probability minus non-neutral emotion mass, clipped to [-1, 1].
R axis: dot product of StyleDistance embedding with a fixed R_axis from axis_vectors.npz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config_loader import PROJECT_ROOT, load_experiment_config

DEFAULT_AXIS_NPZ = PROJECT_ROOT / "data" / "axis_vectors.npz"
DEFAULT_FORMALITY_MODEL = "s-nlp/roberta-base-formality-ranker"
DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_STYLEDIST_MODEL = "StyleDistance/styledistance"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _truncate_words(text: str, max_words: int = 380) -> str:
    """RoBERTa-based classifiers use 512 subword tokens; ~380 words stays safe."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _formality_score(pipe: Any, text: str) -> float:
    """Probability mass on the 'formal' / rational pole (label varies by model)."""
    out = pipe(
        _truncate_words(text, max_words=256),
        top_k=None,
        truncation=True,
        max_length=510,
    )
    if not out:
        return 0.5
    scores = out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
    if isinstance(scores, dict):
        scores = [scores]
    formal_p = None
    informal_p = None
    for d in scores:
        lab = str(d.get("label", "")).lower()
        p = float(d.get("score", 0.0))
        if "informal" in lab:
            informal_p = p
        elif "formal" in lab:
            formal_p = p
    if formal_p is not None:
        return formal_p
    if informal_p is not None:
        return 1.0 - informal_p
    # LABEL_0 / unknown: take max-prob class as informal vs formal heuristically
    if len(scores) == 2:
        a, b = scores[0], scores[1]
        if float(a["score"]) >= float(b["score"]):
            return float(b["score"])
        return float(a["score"])
    return float(scores[0].get("score", 0.5)) if scores else 0.5


def _emotion_intensity(pipe: Any, text: str) -> float:
    """Sum of class probabilities except neutral."""
    out = pipe(
        _truncate_words(text, max_words=256),
        top_k=None,
        truncation=True,
        max_length=510,
    )
    scores = out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
    if isinstance(scores, dict):
        scores = [scores]
    total = 0.0
    for d in scores:
        lab = str(d.get("label", "")).lower()
        if lab == "neutral":
            continue
        total += float(d.get("score", 0.0))
    return float(np.clip(total, 0.0, 1.0))


class TextStyleCoords:
    """Lazy-loaded HF pipelines + StyleDistance encoder; reads R_axis from npz."""

    def __init__(
        self,
        *,
        axis_npz: Path | str | None = None,
        formality_model: str | None = None,
        emotion_model: str | None = None,
        styledist_model: str | None = None,
    ) -> None:
        cfg = load_experiment_config()
        hf_cfg = (cfg.get("coord_scoring") or {}).get("hf") or {}
        self._axis_path = Path(axis_npz or hf_cfg.get("axis_vectors_npz") or DEFAULT_AXIS_NPZ)
        self._formality_model = formality_model or hf_cfg.get("formality_model") or DEFAULT_FORMALITY_MODEL
        self._emotion_model = emotion_model or hf_cfg.get("emotion_model") or DEFAULT_EMOTION_MODEL
        self._styledist_model = styledist_model or hf_cfg.get("styledist_model") or DEFAULT_STYLEDIST_MODEL

        self._formality_pipe: Any = None
        self._emotion_pipe: Any = None
        self._st_model: Any = None
        self._r_axis: np.ndarray | None = None

    def _load_npz(self) -> None:
        if self._r_axis is not None:
            return
        if not self._axis_path.is_absolute():
            self._axis_path = PROJECT_ROOT / self._axis_path
        if not self._axis_path.exists():
            raise FileNotFoundError(
                f"axis vectors not found: {self._axis_path}. Run scripts/build_axis_vectors.py first."
            )
        data = np.load(self._axis_path, allow_pickle=False)
        if "R_axis" not in data:
            raise KeyError(f"{self._axis_path} must contain array 'R_axis'")
        self._r_axis = _normalize(np.asarray(data["R_axis"], dtype=float).ravel())

    def _ensure_formality(self) -> None:
        if self._formality_pipe is None:
            from transformers import pipeline

            self._formality_pipe = pipeline(
                "text-classification",
                model=self._formality_model,
                top_k=None,
            )

    def _ensure_emotion(self) -> None:
        if self._emotion_pipe is None:
            from transformers import pipeline

            self._emotion_pipe = pipeline(
                "text-classification",
                model=self._emotion_model,
                top_k=None,
            )

    def _ensure_st(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self._styledist_model)

    def encode_r_direction(self, text: str) -> np.ndarray:
        """Unit-normalized StyleDistance embedding (for debugging)."""
        self._ensure_st()
        emb = self._st_model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb[0], dtype=float)

    def get_Ps(self, text: str) -> tuple[float, float]:
        """Return (S, R) in [-1, 1]."""
        self._load_npz()
        self._ensure_formality()
        self._ensure_emotion()
        self._ensure_st()

        f = _formality_score(self._formality_pipe, text)
        e = _emotion_intensity(self._emotion_pipe, text)
        s = float(np.clip(f - e, -1.0, 1.0))

        emb = self._st_model.encode(
            [_truncate_words(text, max_words=256)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        v = np.asarray(emb[0], dtype=float).ravel()
        r_raw = float(np.dot(v, self._r_axis))  # both unit -> in [-1, 1]
        r = float(np.clip(r_raw, -1.0, 1.0))
        return s, r


_GLOBAL: TextStyleCoords | None = None


def get_text_style_coords() -> TextStyleCoords:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = TextStyleCoords()
    return _GLOBAL


def get_Ps(text: str) -> tuple[float, float]:
    return get_text_style_coords().get_Ps(text)
