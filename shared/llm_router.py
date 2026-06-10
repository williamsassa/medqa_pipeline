"""
LLM router with provider rotation and fallback.

Supports any combination of:
  - helix-v2          : OUR trained model served via any OpenAI-compatible endpoint
                        (HF Inference Endpoint with TGI, vLLM serve, Together.ai,
                        Replicate, Ollama, etc.). Configured via HELIX_V2_ENDPOINT_URL
                        and (optional) HELIX_V2_API_KEY. Silently skipped if URL unset.
  - OpenRouter free models (OpenAI-compatible API)
  - Groq / Cerebras / Gemini / Mistral / Moonshot / Zhipu (OpenAI-compatible free tiers)
  - Ollama local (HTTP API on localhost:11434)
  - DeepSeek / OpenAI / Anthropic (if a paid key is available)

On 429 (rate-limit), automatically rotates to the next provider in the chain.
helix-v2 is tried first when configured; free providers act as fallback if it fails
or is not configured.

The active provider chain is configured by the BMC_PROVIDER_CHAIN env var, e.g.:
    BMC_PROVIDER_CHAIN="helix-v2:BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2,groq:llama-3.3-70b-versatile,ollama:qwen2.5:3b-instruct"

Each entry is "{provider}:{model}".
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# ─── Provider chain ───────────────────────────────────────────────────────

DEFAULT_PROVIDER_CHAIN = [
    # Interleaved across providers so we spread load and don't hammer one rate-limit.
    # Each entry needs its provider's key in env (see PROVIDER_REGISTRY); a provider with
    # no key is skipped silently. Fastest/most-reliable free tiers first.
    #
    # OUR MODEL — tried first when HELIX_V2_ENDPOINT_URL is set. Skipped otherwise.
    "helix-v2:BrainHealthAI/MedQA-Llama3.1-8B-HELIX-v2",
    "cerebras:llama3.3-70b",                                   # Cerebras free — very fast (slug has NO hyphen after llama)
    "groq:llama-3.3-70b-versatile",                            # Groq free — fast
    "zhipu:glm-4.7-flash",                                     # GLM free flash
    "openrouter:meta-llama/llama-3.3-70b-instruct:free",
    "gemini:gemini-2.0-flash",                                 # Google AI Studio free
    "moonshot:kimi-k2.5",                                      # Kimi (Moonshot) free tier / trial
    "mistral:mistral-small-latest",                            # Mistral free tier
    "openrouter:openai/gpt-oss-20b:free",
    "groq:llama-3.1-8b-instant",
    "zhipu:glm-4.5-flash",
    "cerebras:llama3.1-8b",
    "openrouter:deepseek/deepseek-r1-distill-llama-70b:free",
    "gemini:gemini-2.5-flash",                                 # second Gemini free model (1.5-flash is retired)
    "openrouter:nvidia/llama-3.1-nemotron-70b-instruct:free",
    # Local Ollama — slower but no rate limit. Final reliable fallback (no key needed).
    "ollama:qwen2.5:3b-instruct",
]


# Registry of OpenAI-compatible chat-completions endpoints. All of these speak the same
# request/response shape, so one generic caller handles them. `key_envs` is tried in order;
# the first env var that's set wins. `json_mode` toggles the response_format hint (some
# providers error on an unknown response_format, so we only send it where it's supported).
PROVIDER_REGISTRY: dict[str, dict] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_envs": ["OPENROUTER_API_KEY"],
        "json_mode": True,
        "extra_headers": {
            "HTTP-Referer": "https://huggingface.co/BrainHealthAI",
            "X-Title": "BrainMedCoT dataset build",
        },
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "key_envs": ["GROQ_API_KEY"],
        "json_mode": True,
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1/chat/completions",
        "key_envs": ["CEREBRAS_API_KEY"],
        "json_mode": True,
    },
    "gemini": {
        # Google AI Studio exposes an OpenAI-compatible surface.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_envs": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "json_mode": False,  # the OpenAI-compat layer is finicky about response_format
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "key_envs": ["MISTRAL_API_KEY"],
        "json_mode": True,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1/chat/completions",
        "key_envs": ["TOGETHER_API_KEY"],
        "json_mode": True,
    },
    "zhipu": {
        # Zhipu / Z.ai GLM — OpenAI-compatible. Free models: glm-4.7-flash, glm-4.5-flash.
        "base_url": "https://api.z.ai/api/paas/v4/chat/completions",
        "key_envs": ["ZHIPU_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY"],
        "json_mode": True,
    },
    "moonshot": {
        # Moonshot Kimi — OpenAI-compatible. Trial credits / free tier on platform.moonshot.ai.
        "base_url": "https://api.moonshot.ai/v1/chat/completions",
        "key_envs": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "json_mode": True,
    },
}


def get_provider_chain() -> list[tuple[str, str]]:
    """Returns a list of (provider, model) tuples."""
    raw = os.getenv("BMC_PROVIDER_CHAIN", "").strip()
    entries = raw.split(",") if raw else DEFAULT_PROVIDER_CHAIN
    out: list[tuple[str, str]] = []
    for e in entries:
        e = e.strip()
        if not e:
            continue
        if ":" not in e:
            logger.warning("invalid provider chain entry: %r", e)
            continue
        provider, model = e.split(":", 1)
        out.append((provider.strip(), model.strip()))
    return out


# ─── Per-provider status (in-memory across one process run) ───────────────

@dataclass
class ProviderState:
    name: str
    model: str
    consecutive_failures: int = 0
    cooldown_until: float = 0.0  # unix ts; we won't try before this
    total_calls: int = 0
    total_successes: int = 0


_states: dict[tuple[str, str], ProviderState] = {}
_states_lock = asyncio.Lock()


def _get_state(provider: str, model: str) -> ProviderState:
    key = (provider, model)
    st = _states.get(key)
    if st is None:
        st = ProviderState(name=provider, model=model)
        _states[key] = st
    return st


async def _mark_failure(state: ProviderState, cooldown_seconds: float = 10.0):
    async with _states_lock:
        state.consecutive_failures += 1
        # Mild exponential backoff capped at 90s — Ollama as final fallback never rate-limits
        backoff = min(90.0, cooldown_seconds * (1.3 ** (state.consecutive_failures - 1)))
        state.cooldown_until = time.time() + backoff


async def _mark_success(state: ProviderState):
    async with _states_lock:
        state.consecutive_failures = 0
        state.cooldown_until = 0.0
        state.total_successes += 1


# ─── Per-provider call implementations ────────────────────────────────────

def provider_api_key(provider: str) -> str | None:
    """Return the first configured key for an OpenAI-compatible provider, or None."""
    spec = PROVIDER_REGISTRY.get(provider)
    if not spec:
        return None
    for env in spec["key_envs"]:
        v = os.getenv(env, "").strip()
        if v:
            return v
    return None


async def _call_oai_compatible(
    client: httpx.AsyncClient, provider: str, api_key: str, model: str,
    system: str, user: str, max_tokens: int, temperature: float,
) -> tuple[str | None, str]:
    """Generic caller for any OpenAI-compatible /chat/completions endpoint."""
    spec = PROVIDER_REGISTRY[provider]
    headers = {"Authorization": f"Bearer {api_key}"}
    headers.update(spec.get("extra_headers", {}))
    payload: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if spec.get("json_mode"):
        payload["response_format"] = {"type": "json_object"}
    try:
        r = await client.post(spec["base_url"], json=payload, headers=headers, timeout=120.0)
    except Exception as e:
        return None, f"network_error:{type(e).__name__}:{e}"

    if r.status_code == 429:
        return None, f"rate_limit:429:{r.text[:100]}"
    if r.status_code in (401, 403):
        return None, f"auth_error:{r.status_code}:{r.text[:100]}"
    if r.status_code != 200:
        return None, f"http_{r.status_code}:{r.text[:150]}"
    try:
        body = r.json()
        content = body["choices"][0]["message"]["content"] or ""
        return content, ""
    except Exception as e:
        return None, f"parse_error:{type(e).__name__}:{e}; body={r.text[:200]}"


async def _call_helix_v2(
    client: httpx.AsyncClient, endpoint_url: str, api_key: str, model: str,
    system: str, user: str, max_tokens: int, temperature: float,
) -> tuple[str | None, str]:
    """Caller for HELIX-v2 served via any OpenAI-compatible /v1/chat/completions endpoint.

    Supports — same code path for all:
      - HF Inference Endpoint (TGI container exposes /v1/chat/completions)
      - vLLM serve --port 8000 (with `--enable-lora` + HELIX-v2 adapter loaded)
      - Ollama via OpenAI-compatible proxy
      - Together.ai (https://api.together.xyz/v1/chat/completions)
      - Replicate (OpenAI-compatible deployments)
      - Any custom inference server exposing /v1/chat/completions

    Config (env, both required for activation):
      HELIX_V2_ENDPOINT_URL — full URL ending in /v1/chat/completions
      HELIX_V2_API_KEY      — optional; falls back to HF_TOKEN if unset

    Returns (content, "") on success, (None, "<reason>") on failure.
    Cold start on HF Inference Endpoint (scale-to-zero) returns 503 — we mark that
    as a soft failure so the next call retries after a short cooldown.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # Request strict JSON output when the endpoint supports it (TGI, vLLM, Together do).
    payload["response_format"] = {"type": "json_object"}

    try:
        # Longer timeout to absorb HF Inference Endpoint cold starts (up to 5 min).
        r = await client.post(endpoint_url, json=payload, headers=headers, timeout=300.0)
    except Exception as e:
        return None, f"network_error:{type(e).__name__}:{e}"

    if r.status_code == 429:
        return None, f"rate_limit:429:{r.text[:120]}"
    if r.status_code in (401, 403):
        return None, f"auth_error:{r.status_code}:{r.text[:120]}"
    if r.status_code == 503:
        # HF Inference Endpoint scale-to-zero cold start, or model still loading.
        return None, f"cold_start:503:{r.text[:120]}"
    if r.status_code != 200:
        return None, f"http_{r.status_code}:{r.text[:150]}"

    try:
        body = r.json()
        content = body["choices"][0]["message"]["content"] or ""
        return content, ""
    except Exception as e:
        return None, f"parse_error:{type(e).__name__}:{e}; body={r.text[:200]}"


async def _call_ollama(
    client: httpx.AsyncClient, model: str,
    system: str, user: str, max_tokens: int, temperature: float,
) -> tuple[str | None, str]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "format": "json",
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    try:
        r = await client.post("http://localhost:11434/api/chat",
                              json=payload, timeout=600.0)
    except Exception as e:
        return None, f"network_error:{type(e).__name__}:{e}"

    if r.status_code != 200:
        return None, f"http_{r.status_code}:{r.text[:150]}"
    try:
        body = r.json()
        content = body["message"]["content"] or ""
        return content, ""
    except Exception as e:
        return None, f"parse_error:{type(e).__name__}:{e}"


# ─── Public entry point ───────────────────────────────────────────────────

def _extract_json_obj(text: str) -> Optional[dict]:
    """Robust JSON-object extraction (fences, trailing commas, truncation, single quotes)."""
    try:
        from .robust_json import loads_robust
    except ImportError:
        from robust_json import loads_robust  # type: ignore
    return loads_robust(text)


def _extract_think(text: str) -> tuple[Optional[str], str]:
    """Pull the <think> string out of an LLM response, tolerating alternate keys + broken JSON."""
    try:
        from .robust_json import extract_think
    except ImportError:
        from robust_json import extract_think  # type: ignore
    return extract_think(text)


async def generate_think(
    *,
    system_prompt: str,
    user_msg: str,
    max_tokens: int = 2400,
    temperature: float = 0.0,
    http_client: httpx.AsyncClient,
    chain: Optional[list[tuple[str, str]]] = None,
    max_chain_passes: int = 2,
    extract: bool = True,
) -> tuple[str | None, str]:
    """
    Try each provider in `chain` until one succeeds. Returns (text, failure_reason).

    If `extract` is True (default), the model is expected to wrap its output in a
    {"think": "..."} envelope and we return the extracted "think" field. If `extract`
    is False, we return the RAW model content unchanged — use this when the caller's
    system prompt already asks for a direct JSON object (e.g. the SOAP agent), avoiding
    a fragile double-JSON-encoding step.

    A provider is skipped if its cooldown has not expired. On 429/5xx, the provider enters
    cooldown and we try the next one. After one full pass through the chain, we wait for the
    soonest cooldown to expire and try again, up to `max_chain_passes` total passes.
    """
    if chain is None:
        chain = get_provider_chain()
    if not chain:
        return None, "empty_provider_chain"

    last_err = "no_provider_attempted"

    for pass_idx in range(max_chain_passes):
        attempted_any = False
        now = time.time()

        for provider, model in chain:
            state = _get_state(provider, model)
            if state.cooldown_until > now:
                continue

            if provider == "helix-v2":
                # Our trained model. Only attempted when HELIX_V2_ENDPOINT_URL is set;
                # otherwise silently skipped so the chain continues to free providers.
                url = os.getenv("HELIX_V2_ENDPOINT_URL", "").strip()
                if not url:
                    last_err = "helix_v2_endpoint_not_configured"
                    continue
                key = (os.getenv("HELIX_V2_API_KEY") or os.getenv("HF_TOKEN") or "").strip()
                attempted_any = True
                state.total_calls += 1
                content, err = await _call_helix_v2(
                    http_client, url, key, model,
                    system_prompt, user_msg, max_tokens, temperature,
                )
            elif provider == "ollama":
                attempted_any = True
                state.total_calls += 1
                content, err = await _call_ollama(
                    http_client, model, system_prompt, user_msg, max_tokens, temperature,
                )
            elif provider in PROVIDER_REGISTRY:
                key = provider_api_key(provider)
                if not key:
                    # No key configured for this provider — skip without burning a pass slot.
                    last_err = f"{provider}_no_api_key"
                    continue
                attempted_any = True
                state.total_calls += 1
                content, err = await _call_oai_compatible(
                    http_client, provider, key, model, system_prompt, user_msg, max_tokens, temperature,
                )
            else:
                last_err = f"unsupported_provider:{provider}"
                continue

            if content is None:
                last_err = f"{provider}:{model}::{err}"
                if "cold_start" in err:
                    # HF Inference Endpoint scale-to-zero typically warms up in 30-60s.
                    # Skip this provider for ~45s so we don't hammer it while it spins up.
                    await _mark_failure(state, cooldown_seconds=45.0)
                elif "rate_limit" in err or "http_5" in err or "network_error" in err:
                    await _mark_failure(state, cooldown_seconds=15.0)
                elif "auth_error" in err:
                    await _mark_failure(state, cooldown_seconds=3600.0)
                elif "http_404" in err:
                    # Model doesn't exist on this provider — long cooldown, effectively skipped
                    await _mark_failure(state, cooldown_seconds=3600.0)
                else:
                    await _mark_failure(state, cooldown_seconds=5.0)
                continue

            if not extract:
                # Raw mode: return the model content directly (caller parses it).
                if content and content.strip():
                    await _mark_success(state)
                    return content, ""
                last_err = f"{provider}:{model}::empty_content"
                await _mark_failure(state, cooldown_seconds=2.0)
                continue

            think, parse_reason = _extract_think(content)
            if think is None:
                last_err = f"{provider}:{model}::{parse_reason}"
                # json_parse_failed → short cooldown (likely transient); empty_think → model
                # genuinely returned nothing useful, mark success so we don't punish the provider.
                if parse_reason == "empty_think":
                    await _mark_success(state)
                else:
                    await _mark_failure(state, cooldown_seconds=2.0)
                continue

            await _mark_success(state)
            return think, ""

        # End of one pass — no provider succeeded
        if not attempted_any:
            # Everyone in cooldown. Wait for the soonest to expire, then try again.
            now = time.time()
            soonest = min((s.cooldown_until for s in _states.values() if s.cooldown_until > now),
                          default=now + 5.0)
            wait = max(0.5, min(20.0, soonest - now + 0.5))
            await asyncio.sleep(wait)

    return None, f"all_providers_exhausted_after_{max_chain_passes}_passes: last={last_err}"


def get_router_stats() -> dict[str, dict[str, int]]:
    """For logging/metadata: per-provider call counts."""
    return {
        f"{s.name}:{s.model}": {
            "calls": s.total_calls,
            "successes": s.total_successes,
            "failures": s.total_calls - s.total_successes,
        }
        for s in _states.values()
    }
