"""Provider-agnostic generation API with disk caching and retries.

Every provider's API is abstracted behind a single ``generate()`` function
that accepts a model_name (defined in configs/models.yaml), a system prompt,
and a user prompt, and returns the generated text plus metadata.

Responses are cached by hash of (provider, model_id, system, user, params).
Re-running the pipeline with the same inputs does not cost any API calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import diskcache

from .config_loader import cache_dir, env, get_model_spec


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    model_id: str
    usage: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


_cache: diskcache.Cache | None = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(str(cache_dir() / "generations"))
    return _cache


def _cache_key(provider: str, model_id: str, system: str, user: str, params: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model_id": model_id,
            "system": system,
            "user": user,
            "params": {k: params[k] for k in sorted(params)},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    pass


def _openai_like_chat(
    api_key: str,
    base_url: str | None,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Handles any provider whose HTTP schema is OpenAI-compatible
    (OpenAI, DeepSeek, DashScope (/compatible-mode), Zhipu, Moonshot)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **(extra or {}),
    )
    ch0 = resp.choices[0]
    text = ch0.message.content or ""
    if not (text or "").strip():
        fr = getattr(ch0, "finish_reason", None)
        print(
            f"[generate] empty assistant content model_id={model_id!r} finish_reason={fr!r}",
            file=sys.stderr,
        )
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
        "total_tokens": getattr(resp.usage, "total_tokens", None),
    }
    return text, usage


def _anthropic_chat(
    api_key: str,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> tuple[str, dict[str, Any]]:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model_id,
        system=system if system else None,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    text = "".join(part.text for part in msg.content if part.type == "text")
    usage = {
        "prompt_tokens": getattr(msg.usage, "input_tokens", None),
        "completion_tokens": getattr(msg.usage, "output_tokens", None),
        "total_tokens": (getattr(msg.usage, "input_tokens", 0) or 0)
        + (getattr(msg.usage, "output_tokens", 0) or 0),
    }
    return text, usage


# DashScope OpenAI-compatible API (qwen-*); never fall back to OPENAI_BASE_URL.
_DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Map provider -> (env-var-for-key, env-var-for-base-url-or-None)
_PROVIDER_ENV: dict[str, tuple[str, str | None]] = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
    "dashscope": ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"),
    "zhipu": ("ZHIPU_API_KEY", "ZHIPU_BASE_URL"),
    "moonshot": ("MOONSHOT_API_KEY", "MOONSHOT_BASE_URL"),
}


def _dispatch(
    provider: str,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    openai_create_extras: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if provider == "anthropic":
        key = env("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY not set in .env")
        return _anthropic_chat(key, model_id, system, user, temperature, max_tokens, top_p)

    if provider not in _PROVIDER_ENV:
        raise ProviderError(f"Unknown provider {provider!r}")

    key_env, base_env = _PROVIDER_ENV[provider]
    key = env(key_env)
    if not key:
        raise ProviderError(f"{key_env} not set in .env (required for provider={provider})")
    base_url_raw = env(base_env) if base_env else None
    base_url = (base_url_raw or "").strip() or None
    if provider == "dashscope":
        base_url = base_url or _DASHSCOPE_DEFAULT_BASE_URL
    elif not base_url and provider in {"deepseek", "zhipu", "moonshot"}:
        # Single OpenAI-compatible proxy (e.g. GlobalAI) for these providers only.
        openai_base = env("OPENAI_BASE_URL")
        base_url = (openai_base or "").strip() or None
    return _openai_like_chat(
        key,
        base_url,
        model_id,
        system,
        user,
        temperature,
        max_tokens,
        top_p,
        extra=openai_create_extras,
    )


# Backoff before each retry (1st retry .. 5th) after InternalServerError / APITimeoutError.
_GATEWAY_BACKOFF_S: tuple[int, ...] = (2, 4, 8, 16, 32)
_MAX_GATEWAY_RETRIES = 5


def _generate_with_retry(
    provider: str,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    openai_create_extras: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call ``_dispatch`` with exponential backoff on transient gateway/timeouts."""
    from openai import APITimeoutError, InternalServerError

    for attempt in range(1 + _MAX_GATEWAY_RETRIES):
        try:
            return _dispatch(
                provider,
                model_id,
                system,
                user,
                temperature,
                max_tokens,
                top_p,
                openai_create_extras=openai_create_extras,
            )
        except (InternalServerError, APITimeoutError) as e:
            if attempt >= _MAX_GATEWAY_RETRIES:
                raise
            wait = _GATEWAY_BACKOFF_S[attempt]
            print(
                f"[generate] retry {attempt + 1}/5 after {wait}s: {e}",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError("_generate_with_retry: unreachable")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(
    model_name: str,
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float = 0.95,
    use_cache: bool = True,
) -> GenerationResult:
    """Generate text from a model registered in configs/models.yaml.

    Parameters
    ----------
    model_name : str
        Key in `models:` from `configs/models.yaml`.
    system, user : str
        System and user prompts. Providers that lack a system role concatenate.
    temperature, max_tokens : optional
        Override defaults from the model spec.
    top_p : float
        Sampling nucleus (not all providers honour this; passed when possible).
    use_cache : bool
        Skip the disk cache when False (used for invariance sanity checks).
    """
    spec = get_model_spec(model_name)
    provider = spec["provider"]
    model_id = spec["model_id"]
    temperature = spec.get("temperature", 0.7) if temperature is None else temperature
    max_tokens = spec.get("max_tokens", 600) if max_tokens is None else max_tokens
    raw_extras = spec.get("openai_create_extras")
    if raw_extras is not None and not isinstance(raw_extras, dict):
        raise ProviderError(f"Model {model_name!r}: openai_create_extras must be a mapping if set")
    openai_create_extras: dict[str, Any] | None = dict(raw_extras) if raw_extras else None

    params = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
    if openai_create_extras:
        params["openai_create_extras"] = json.dumps(openai_create_extras, ensure_ascii=False, sort_keys=True)
    key = _cache_key(provider, model_id, system, user, params)

    cache = _get_cache()
    if use_cache and key in cache:
        cached_payload = cache[key]
        return GenerationResult(
            text=cached_payload["text"],
            model=model_name,
            provider=provider,
            model_id=model_id,
            usage=cached_payload.get("usage", {}),
            cached=True,
            latency_s=0.0,
        )

    t0 = time.time()
    text, usage = _generate_with_retry(
        provider,
        model_id,
        system,
        user,
        temperature,
        max_tokens,
        top_p,
        openai_create_extras=openai_create_extras,
    )
    dt = time.time() - t0

    # Do not persist empty completions: they poison diskcache and make retries futile.
    if use_cache and (text or "").strip():
        cache[key] = {"text": text, "usage": usage}

    return GenerationResult(
        text=text,
        model=model_name,
        provider=provider,
        model_id=model_id,
        usage=usage,
        cached=False,
        latency_s=dt,
    )


# ---------------------------------------------------------------------------
# Smoke helper
# ---------------------------------------------------------------------------


def ping(model_name: str) -> bool:
    """Return True if the model answers a trivial prompt; else False."""
    try:
        # Some OpenAI-compatible gateways (incl. DeepSeek routes) return empty
        # assistant content with finish_reason=length when max_tokens is tiny
        # but a system message is present; keep headroom so ping is not a false negative.
        res = generate(
            model_name,
            system="You are a terse assistant.",
            user="Reply with the single word: ready",
            temperature=0.0,
            max_tokens=48,
            use_cache=False,
        )
        return "ready" in res.text.lower()
    except Exception as e:  # noqa: BLE001
        print(f"[ping] {model_name} failed: {e}")
        return False
