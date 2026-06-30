"""
Unified LLM client for ApplyAssist.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for any provider.
"""

import logging
import os
import time
from datetime import datetime, timedelta

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed errors so callers can fail safe (checkpoint + stop) instead of
# silently recording garbage results.
# ---------------------------------------------------------------------------

class LLMHaltError(Exception):
    """Base class for errors that should HALT a batch, not poison results.

    When one of these is raised, the caller should stop the run, keep whatever
    progress is already committed, and tell the user when/how to resume —
    rather than treating the failure as a low score.
    """

    def __init__(self, message: str, *, resume_hint: str = "") -> None:
        super().__init__(message)
        self.resume_hint = resume_hint


class LLMRateLimitError(LLMHaltError):
    """The provider rejected us with HTTP 429 even after backoff/retries.

    For the Gemini free tier this almost always means the daily quota is
    exhausted (a per-minute spike would clear within ~60s).
    """


class LLMConnectionError(LLMHaltError):
    """Could not reach the provider at all (DNS failure, no network, timeout)."""


def next_quota_reset_hint() -> str:
    """Human-readable hint for when to retry, in the machine's local time.

    Gemini's free-tier daily quota resets at midnight US Pacific. We surface
    that moment converted to the system's local timezone so the user knows when
    to re-run. Best-effort: falls back to a generic message if tz data is absent.
    """
    try:
        from zoneinfo import ZoneInfo

        pacific = ZoneInfo("America/Los_Angeles")
        now_pt = datetime.now(pacific)
        next_midnight_pt = (now_pt + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        local = next_midnight_pt.astimezone()  # system local timezone
        tz = local.strftime("%Z") or "local time"
        return (
            f"the free-tier daily quota resets at midnight US Pacific — "
            f"about {local.strftime('%I:%M %p').lstrip('0')} {tz} on "
            f"{local.strftime('%a %d %b')}. Re-run after then and it will "
            f"continue from where it stopped"
        )
    except Exception:
        return (
            "the free-tier daily quota resets at midnight US Pacific. Re-run "
            "after then and it will continue from where it stopped"
        )

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) based on environment variables.

    Reads env at call time (not module import time) so that load_env() called
    in _bootstrap() is always visible here.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    model_override = os.environ.get("LLM_MODEL", "")

    if gemini_key and not local_url:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_override or "gemini-2.0-flash",
            gemini_key,
        )

    if openai_key and not local_url:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            openai_key,
        )

    if local_url:
        return (
            local_url.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

# Base wait on first 429/503 (doubles each retry, caps at 60s).
# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10

# If a provider's Retry-After exceeds this, treat it as a daily-quota halt and
# stop (checkpointing) rather than block-sleeping the process for hours.
_MAX_INLINE_WAIT = 300


_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        # True once we've confirmed the native Gemini API works for this model
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the native Gemini generateContent API.

        Used automatically when the OpenAI-compat endpoint returns 403,
        which happens for preview/experimental models not exposed via compat.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Reasoning models (e.g. gpt-oss) spend the token budget on a hidden
        # reasoning trace before answering; keep it minimal so the actual answer
        # isn't truncated away and the budget isn't wasted.
        if "gpt-oss" in self.model.lower():
            payload["reasoning_effort"] = "low"

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        # Raise a specific sentinel so chat() can switch to native API.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp)

    @staticmethod
    def _handle_compat_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        # Reasoning models may return null/missing content (e.g. when the answer
        # got truncated by max_tokens). Return "" rather than crashing — the
        # caller treats empty as a failed attempt and retries.
        content = msg.get("content")
        return content if content is not None else ""

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                # Route to native Gemini if we've already confirmed it's needed
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens)

                return self._chat_compat(messages, temperature, max_tokens)

            except _GeminiCompatForbidden as exc:
                # Model not available on OpenAI-compat layer — switch to native.
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API. "
                    "(Preview/experimental models are often compat-only on native.)",
                    self.model,
                )
                self._use_native_gemini = True
                # Retry immediately with native — don't count as a rate-limit wait
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503):
                    # Honor Retry-After if the provider sends one.
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    server_wait = None
                    if retry_after:
                        try:
                            server_wait = float(retry_after)
                        except (ValueError, TypeError):
                            server_wait = None

                    # A long Retry-After (e.g. 86400 = 24h) means a DAILY quota is
                    # exhausted — do NOT block the process for hours. Halt now so
                    # progress is checkpointed and the user/--auto-resume handles it.
                    if server_wait is not None and server_wait > _MAX_INLINE_WAIT:
                        hours = server_wait / 3600.0
                        raise LLMRateLimitError(
                            f"Provider asked to wait {int(server_wait)}s (~{hours:.1f}h) — "
                            f"daily quota exhausted.",
                            resume_hint=(
                                f"the provider's quota resets in about {hours:.1f}h; "
                                f"re-run after that (or switch --provider) — it continues "
                                f"from where it stopped"
                            ),
                        ) from exc

                    if attempt < _MAX_RETRIES - 1:
                        wait = (server_wait if server_wait is not None
                                else min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60))
                        log.warning(
                            "LLM rate limited (HTTP %s). Waiting %ds before retry %d/%d. "
                            "(Free tiers cap requests/min and tokens/day; slow down, "
                            "switch --provider, or upgrade.)",
                            resp.status_code, int(wait), attempt + 1, _MAX_RETRIES,
                        )
                        time.sleep(wait)
                        continue

                    # Retries exhausted — halt so the batch checkpoints.
                    hint = (next_quota_reset_hint() if self._is_gemini else
                            "this is a per-minute rate limit — wait ~60s and re-run; "
                            "it continues from where it stopped")
                    raise LLMRateLimitError(
                        f"Provider returned HTTP {resp.status_code} after "
                        f"{_MAX_RETRIES} attempts (quota/rate limit).",
                        resume_hint=hint,
                    ) from exc
                raise

            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # DNS failure / no network. Retry a few times (blips happen),
                # then halt — never silently degrade to a bad score.
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 30)
                    log.warning(
                        "Cannot reach LLM provider (%s). Retrying in %ds (attempt %d/%d).",
                        exc, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise LLMConnectionError(
                    f"Could not reach the LLM provider after {_MAX_RETRIES} attempts: {exc}",
                    resume_hint="check your internet connection, then re-run — "
                                "it will continue from where it stopped",
                ) from exc

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise LLMConnectionError(
                    f"LLM requests kept timing out after {_MAX_RETRIES} attempts.",
                    resume_hint="check your connection, then re-run to continue",
                )

        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None
_default_provider_name: str | None = None


def set_default_provider(name: str | None) -> None:
    """Select the default named provider for this run (a `--provider` flag).

    Overrides the LLM_PROVIDER env default and the bare LLM_URL/GEMINI/OPENAI
    config. Resets the cached client so the next get_client() rebuilds.
    """
    global _default_provider_name, _instance
    _default_provider_name = (name or "").strip() or None
    _instance = None


def get_client() -> LLMClient:
    """Return (or create) the default LLM client.

    Resolution order: an explicit --provider (set_default_provider) → the
    LLM_PROVIDER env default (a named LLM_*_<NAME> block) → the bare
    LLM_URL/GEMINI_API_KEY/OPENAI_API_KEY config.
    """
    global _instance
    if _instance is None:
        name = _default_provider_name or os.environ.get("LLM_PROVIDER", "").strip()
        if name:
            _instance = build_client_for_provider(name)
        else:
            base_url, model, api_key = _detect_provider()
            log.info("LLM provider: %s  model: %s", base_url, model)
            _instance = LLMClient(base_url, model, api_key)
    return _instance


def build_client_for_model(model: str) -> LLMClient:
    """Build a fresh client for an explicit model, inferring its provider.

    Lets one stage use a different model/provider than the default (e.g. cheap
    fast scoring on Gemini while tailoring/cover use the configured LLM_URL
    model). A "gemini-*" model routes to Google (needs GEMINI_API_KEY) even when
    LLM_URL is set; anything else uses the configured OpenAI-compatible endpoint
    (LLM_URL) or OpenAI.
    """
    m = (model or "").strip()
    gem = os.environ.get("GEMINI_API_KEY", "")
    url = os.environ.get("LLM_URL", "")
    openai = os.environ.get("OPENAI_API_KEY", "")
    if m.lower().startswith("gemini"):
        if not gem:
            raise RuntimeError(
                f"Score model '{m}' is a Gemini model but GEMINI_API_KEY is not set "
                f"in ~/.applyassist/.env."
            )
        client = LLMClient(_GEMINI_COMPAT_BASE, m, gem)
    elif url:
        client = LLMClient(url.rstrip("/"), m, os.environ.get("LLM_API_KEY", ""))
    elif openai:
        client = LLMClient("https://api.openai.com/v1", m, openai)
    elif gem:
        client = LLMClient(_GEMINI_COMPAT_BASE, m, gem)
    else:
        raise RuntimeError(f"No provider available to run model '{m}'.")
    log.info("Stage client: %s  model: %s", client.base_url, client.model)
    return client


class RotatingClient:
    """Chain of providers that auto-rotates on rate-limit/quota exhaustion.

    Calls the current provider; if it returns a rate-limit/quota halt, advances
    to the next provider and retries. Only raises LLMRateLimitError when EVERY
    provider in the chain is exhausted — so several free tiers' daily quotas add
    up into one larger effective budget.
    """

    def __init__(self, clients: list[LLMClient], names: list[str]) -> None:
        self._clients = clients
        self._names = names
        self._i = 0
        self.base_url = "rotating"
        self.model = "+".join(names)

    def chat(self, messages: list[dict], **kwargs) -> str:
        n = len(self._clients)
        last_exc: Exception | None = None
        for _ in range(n):
            name = self._names[self._i]
            try:
                return self._clients[self._i].chat(messages, **kwargs)
            except Exception as e:  # rate-limit, dead key, connection blip, etc.
                last_exc = e
                nxt = (self._i + 1) % n
                reason = type(e).__name__
                log.warning("Provider '%s' failed (%s) — rotating to '%s'.",
                            name, reason, self._names[nxt])
                self._i = nxt
        # Every provider failed this pass. Raise a halt (not a generic error) so
        # the batch checkpoints cleanly instead of recording poisoned results.
        raise LLMRateLimitError(
            f"All configured providers failed/exhausted: {', '.join(self._names)} "
            f"(last error: {type(last_exc).__name__}).",
            resume_hint=("every provider in the chain hit a limit or error; check the "
                         "keys, or wait for a daily reset and re-run — it continues "
                         "from where it stopped"),
        ) from last_exc

    def ask(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        for c in self._clients:
            try:
                c.close()
            except Exception:
                pass


def build_client_for_provider(name: str):
    """Build a client for a named provider, or a RotatingClient for a chain.

    Reads LLM_URL_<NAME>, LLM_API_KEY_<NAME>, LLM_MODEL_<NAME> from the env
    (NAME upper-cased), e.g. provider "cerebras" reads LLM_URL_CEREBRAS etc. A
    comma-separated value ("cerebras,gemini,groq") builds a RotatingClient that
    falls through providers as each hits its quota.
    """
    names = [n.strip() for n in (name or "").split(",") if n.strip()]
    if len(names) > 1:
        return RotatingClient([build_client_for_provider(n) for n in names], names)

    u = (names[0] if names else "").upper()
    url = os.environ.get(f"LLM_URL_{u}")
    model = os.environ.get(f"LLM_MODEL_{u}")
    key = os.environ.get(f"LLM_API_KEY_{u}", "")
    if not url or not model:
        raise RuntimeError(
            f"Provider '{name}' is not configured. Set LLM_URL_{u}, "
            f"LLM_API_KEY_{u}, and LLM_MODEL_{u} in ~/.applyassist/.env."
        )
    client = LLMClient(url.rstrip("/"), model, key)
    log.info("Stage provider '%s': %s  model: %s", name, client.base_url, client.model)
    return client
