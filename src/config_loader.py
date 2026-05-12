"""Load the three YAML configs (personalities / models / experiment) once."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env as early as possible so that API keys are visible to everything.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_models_config() -> dict[str, Any]:
    return _read_yaml(PROJECT_ROOT / "configs" / "models.yaml")


@lru_cache(maxsize=1)
def load_experiment_config() -> dict[str, Any]:
    return _read_yaml(PROJECT_ROOT / "configs" / "experiment.yaml")


# Old JSONL / env may still reference this id; it maps to the same API model.
_MODEL_ALIASES: dict[str, str] = {"gen_deepseek_chat": "gen_deepseek_v4_pro"}


def resolve_model_name(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)


def get_model_spec(name: str) -> dict[str, Any]:
    """Return the `models:` entry for a given name."""
    name = resolve_model_name(name)
    cfg = load_models_config()
    for entry in cfg["models"]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"Model {name!r} not found in models.yaml")


def get_role_models(role: str) -> list[str]:
    """Return a list of model-names for a given role from the `roles:` block."""
    cfg = load_models_config()
    value = cfg["roles"].get(role)
    if value is None:
        raise KeyError(f"Role {role!r} not found in models.yaml under roles:")
    return list(value) if isinstance(value, list) else [value]


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def cache_dir() -> Path:
    d = Path(env("CACHE_DIR") or ".cache")
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d
