"""Load and represent the four stylistic-cognitive personas of Space-H."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class Persona:
    name: str
    vector: np.ndarray  # shape (2,) in [-1, 1]^2
    theoretical_anchor: str
    system_prompt: str
    paraphrase_prompts: tuple[str, ...]
    positive_markers: tuple[str, ...]
    negative_markers: tuple[str, ...]

    def prompt(self, variant: int = 0) -> str:
        """Return the primary prompt (variant=0) or a paraphrase (variant>=1)."""
        if variant == 0:
            return self.system_prompt
        idx = variant - 1
        if idx >= len(self.paraphrase_prompts):
            raise IndexError(
                f"Persona {self.name} has {len(self.paraphrase_prompts)} paraphrases; "
                f"requested variant {variant}."
            )
        return self.paraphrase_prompts[idx]


@dataclass(frozen=True)
class PersonaSet:
    personas: tuple[Persona, ...]
    scorer_prompt: str

    def __iter__(self) -> Iterable[Persona]:
        return iter(self.personas)

    def __len__(self) -> int:
        return len(self.personas)

    def __getitem__(self, key) -> Persona:
        if isinstance(key, int):
            return self.personas[key]
        for p in self.personas:
            if p.name == key:
                return p
        raise KeyError(f"Persona {key!r} not found. Available: {[p.name for p in self.personas]}")

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.personas]

    @property
    def vectors(self) -> np.ndarray:
        """Matrix of shape (n_personas, 2)."""
        return np.stack([p.vector for p in self.personas])

    def closest_to(self, vec: np.ndarray) -> Persona:
        """Persona whose vector is nearest to `vec`."""
        dists = np.linalg.norm(self.vectors - vec, axis=1)
        return self.personas[int(np.argmin(dists))]

    def farthest_from(self, vec: np.ndarray) -> Persona:
        dists = np.linalg.norm(self.vectors - vec, axis=1)
        return self.personas[int(np.argmax(dists))]


def load_personas(path: str | Path = "configs/personalities.yaml") -> PersonaSet:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    personas: list[Persona] = []
    for entry in data["personas"]:
        personas.append(
            Persona(
                name=entry["name"],
                vector=np.array(entry["vector"], dtype=float),
                theoretical_anchor=entry["theoretical_anchor"],
                system_prompt=entry["system_prompt"].strip(),
                paraphrase_prompts=tuple(p.strip() for p in entry.get("paraphrase_prompts", [])),
                positive_markers=tuple(entry.get("positive_markers", [])),
                negative_markers=tuple(entry.get("negative_markers", [])),
            )
        )

    return PersonaSet(personas=tuple(personas), scorer_prompt=data["scorer_prompt"].strip())


# Axis label conventions (used by plots and conflict.py):
AXIS_NAMES = ("S", "R")  # S: System2+ / System1-  ;  R: adventurous+ / conservative-
D_MAX_SPACE_H = float(np.linalg.norm(np.array([2.0, 2.0])))  # ‖(1,1)-(-1,-1)‖ = 2√2
