"""OpenAI-compatible LLM client for the Nemotron-Personas-Vietnam pipeline.

Single concurrent client supporting the four NVIDIA-build models the
project standardised on (other OpenAI-compatible endpoints work too —
just override ``base_url``):

* ``nvidia/nemotron-3-super-120b-a12b`` (default)
* ``qwen/qwen3.5-122b-a10b``
* ``qwen/qwen3.5-397b-a17b``
* ``openai/gpt-oss-120b``

Design priorities:

1. **Resumable** — every successful response is appended to a JSONL
   cache keyed by ``(uuid, stage, lang, model_fingerprint)``. A killed
   build resumes by reading the cache and skipping completed rows.
2. **Concurrent** — a ``ThreadPoolExecutor`` keeps ``concurrency``
   requests in flight at once. Configurable per call site.
3. **Robust** — JSON parsing tolerates markdown code fences, retries
   transient HTTP errors with exponential backoff, falls back to
   ``None`` per row on permanent failure (caller decides what to do
   with the gap).

We deliberately do **not** use any vendor-specific SDK — every
provider here speaks the OpenAI ``/v1/chat/completions`` schema, so
plain ``httpx`` keeps the code small and replaceable.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from packages.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model menu — the four pages the user pinned at build.nvidia.com
# ---------------------------------------------------------------------------
KNOWN_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3-super-120b-a12b",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-397b-a17b",
    "openai/gpt-oss-120b",
)
DEFAULT_MODEL: str = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_BASE_URL: str = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Knobs for :class:`LLMClient`."""

    model:        str   = DEFAULT_MODEL
    base_url:     str   = DEFAULT_BASE_URL
    api_key_env:  str   = "PERSONAS_VN_LLM_API_KEY"
    # Generation
    temperature:  float = 0.7
    top_p:        float = 0.9
    max_tokens:   int   = 1500
    # Reliability
    timeout_s:    float = 90.0
    retries:      int   = 3
    retry_backoff_s: float = 2.0
    # Concurrency
    concurrency:  int   = 8
    # Optional sleep between requests inside one worker (rate-limit cushion).
    request_delay_s: float = 0.0


# ---------------------------------------------------------------------------
# Cache layer (JSONL, append-only)
# ---------------------------------------------------------------------------
@dataclass
class JSONLCache:
    """Append-only line-delimited JSON cache.

    Each line is a complete JSON object with a ``key`` string (the cache
    key) and a ``value`` field (the cached LLM response, parsed). On
    load we read the whole file once and build an in-memory dict.

    The append-only design means a Ctrl-C in the middle of writing one
    line at worst loses that one line — never the whole cache.
    """

    path: Path
    _store: dict[str, Any] = field(default_factory=dict)
    _lock:  threading.Lock = field(default_factory=threading.Lock)
    _file:  Any            = None  # open file handle

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rec = json.loads(ln)
                        self._store[rec["key"]] = rec["value"]
                    except (json.JSONDecodeError, KeyError):
                        continue
            log.info("loaded LLM cache: %d entries from %s",
                     len(self._store), self.path)
        # Open append-mode file for streaming writes
        self._file = self.path.open("a", encoding="utf-8")

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._file.write(
                json.dumps({"key": key, "value": value},
                           ensure_ascii=False, sort_keys=True) + "\n"
            )
            self._file.flush()

    def __len__(self) -> int:
        return len(self._store)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
def _model_fingerprint(cfg: LLMConfig) -> str:
    """Short hash that disambiguates cache entries by model + temperature.

    Two runs with the same seed but different models should NOT collide
    in the cache — flipping the model expects fresh outputs.
    """
    sig = f"{cfg.model}|t={cfg.temperature}|p={cfg.top_p}|max={cfg.max_tokens}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]


@dataclass
class LLMClient:
    """OpenAI-compatible chat-completions client with cache + concurrency.

    Use :meth:`call_many` for the typical "generate one structured JSON
    response per persona" case. Returns a list of parsed dict (or
    None per row that permanently failed).
    """

    cfg: LLMConfig
    cache: JSONLCache
    _api_key: str = field(init=False)
    _fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        self._api_key = os.environ.get(self.cfg.api_key_env, "")
        self._fingerprint = _model_fingerprint(self.cfg)
        if not self._api_key:
            log.warning(
                "no API key in env $%s — calls will fail unless cache covers them",
                self.cfg.api_key_env,
            )

    # ------------------------------------------------------------------
    # Single-call entrypoint, used inside the thread pool
    # ------------------------------------------------------------------
    def call_one(
        self,
        *,
        cache_key: str,
        system_prompt: str,
        user_prompt: str,
        json_only: bool = True,
    ) -> dict[str, Any] | None:
        """One chat-completion round-trip with cache + retries.

        Returns the parsed JSON dict on success, or ``None`` if every
        retry failed and the cache also doesn't have it. Successful
        responses are written to the cache.
        """
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        body: dict[str, Any] = {
            "model":       self.cfg.model,
            "temperature": self.cfg.temperature,
            "top_p":       self.cfg.top_p,
            "max_tokens":  self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }
        if json_only:
            body["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.retries + 2):
            try:
                with httpx.Client(
                    base_url=self.cfg.base_url,
                    headers=headers,
                    timeout=self.cfg.timeout_s,
                ) as client:
                    r = client.post("/chat/completions", json=body)
                    r.raise_for_status()
                    payload = r.json()
                content = payload["choices"][0]["message"]["content"]
                obj = _parse_json_lenient(content)
                self.cache.put(cache_key, obj)
                if self.cfg.request_delay_s:
                    time.sleep(self.cfg.request_delay_s)
                return obj
            except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
                last_exc = exc
                if attempt > self.cfg.retries:
                    break
                sleep_for = self.cfg.retry_backoff_s * attempt
                log.debug(
                    "LLM attempt %d for key %s failed (%s); retrying in %.1fs",
                    attempt, cache_key, exc, sleep_for,
                )
                time.sleep(sleep_for)

        log.warning("LLM permanently failed for key %s after %d tries: %s",
                    cache_key, self.cfg.retries + 1, last_exc)
        return None

    # ------------------------------------------------------------------
    # Concurrent batch entry point
    # ------------------------------------------------------------------
    def call_many(
        self,
        items: list[dict[str, Any]],
        *,
        progress_label: str = "llm",
    ) -> list[dict[str, Any] | None]:
        """Submit many calls concurrently and return results in input order.

        ``items`` is a list of dicts with keys
        ``{"cache_key", "system_prompt", "user_prompt"}``. The order of
        the returned list matches the input order regardless of which
        future completes first.
        """
        n = len(items)
        if n == 0:
            return []

        # Pre-pull cached items with no thread overhead.
        results: list[dict[str, Any] | None] = [None] * n
        pending: list[int] = []
        for i, it in enumerate(items):
            cached = self.cache.get(it["cache_key"])
            if cached is not None:
                results[i] = cached
            else:
                pending.append(i)

        if not pending:
            log.info("%s: %d/%d items served from cache (no calls needed)",
                     progress_label, n, n)
            return results

        log.info(
            "%s: %d/%d items in cache, %d need network "
            "(model=%s, concurrency=%d)",
            progress_label, n - len(pending), n, len(pending),
            self.cfg.model, self.cfg.concurrency,
        )

        completed = 0
        successful = 0
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as ex:
            futures = {
                ex.submit(self.call_one,
                          cache_key=items[i]["cache_key"],
                          system_prompt=items[i]["system_prompt"],
                          user_prompt=items[i]["user_prompt"]): i
                for i in pending
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    log.error("%s row %d crashed: %s", progress_label, i, exc)
                    res = None
                results[i] = res
                completed += 1
                if res is not None:
                    successful += 1
                if completed % 25 == 0 or completed == len(pending):
                    elapsed = time.perf_counter() - start
                    rate = completed / max(elapsed, 0.001)
                    eta = (len(pending) - completed) / max(rate, 0.001)
                    log.info(
                        "  %s: %d/%d (%.0f%%, %d ok, %.1f/s, ETA %.0fs)",
                        progress_label, completed, len(pending),
                        100 * completed / len(pending),
                        successful, rate, eta,
                    )

        return results

    def close(self) -> None:
        self.cache.close()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_json_lenient(content: str) -> dict[str, Any]:
    """Parse a JSON object out of an LLM reply.

    Some models wrap the JSON in a markdown code fence. We strip both
    forms (`````` / ``` ```json) before decoding.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        # Strip trailing fence if present
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3].rstrip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object, got {type(obj).__name__}")
    return obj


# ---------------------------------------------------------------------------
# Test-only seam
# ---------------------------------------------------------------------------
class _MockClient(LLMClient):  # pragma: no cover - constructed by tests only
    """LLMClient subclass that returns canned responses instead of making
    HTTP calls. Tests use this to exercise the cache and pipeline
    without network. Production code path uses LLMClient directly."""

    def __init__(
        self,
        cache: JSONLCache,
        responder: Callable[[str, str, str], dict[str, Any]],
    ) -> None:
        self.cfg = LLMConfig(model="mock://test")
        self.cache = cache
        self._api_key = "mock"
        self._fingerprint = "mock0001"
        self._responder = responder

    def call_one(self, *, cache_key, system_prompt, user_prompt, json_only=True):
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        out = self._responder(cache_key, system_prompt, user_prompt)
        self.cache.put(cache_key, out)
        return out
