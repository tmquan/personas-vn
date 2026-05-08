"""HTTP client with retries, on-disk caching, and graceful error wrapping.

Why we don't just use ``requests``:

* The NSO API returns ``X-WP-TotalPages`` etc. in headers — we want a tiny
  wrapper that returns both the parsed JSON and the headers in one call.
* Re-running the scraper should be cheap; we cache every successful GET on
  disk keyed by the canonical URL so re-runs hit local files.
* We need exponential backoff on transient 5xx / network errors without
  pulling a heavy dep just for that.

The only third-party deps are ``httpx`` and ``tenacity`` — both already in
the project requirements.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from packages.common.logging import get_logger
from packages.common.paths import ensure_dir

log = get_logger(__name__)


class HttpError(RuntimeError):
    """Raised when an HTTP request ultimately fails (after retries)."""


@dataclass(frozen=True)
class HttpResponse:
    """Minimal response container — we only need json + headers."""

    status_code: int
    headers: dict[str, str]
    json: Any
    from_cache: bool


class HttpClient:
    """Thin wrapper around :class:`httpx.Client` with caching + retries."""

    def __init__(
        self,
        *,
        base_url: str = "",
        user_agent: str = "personas-vn/0.1",
        verify_ssl: bool = True,
        timeout_s: float = 20.0,
        retries: int = 3,
        retry_backoff_s: float = 1.5,
        delay_between_requests_s: float = 0.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            verify=verify_ssl,
            timeout=timeout_s,
            follow_redirects=True,
        )
        self._retries = max(1, retries)
        self._retry_backoff_s = retry_backoff_s
        self._delay = delay_between_requests_s
        self._cache_dir = Path(ensure_dir(cache_dir)) if cache_dir else None

    # ----- public API -------------------------------------------------------
    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        use_cache: bool = True,
    ) -> HttpResponse:
        """GET ``path`` and return parsed JSON + headers, with retries + cache."""
        params = params or {}
        cache_path = self._cache_path(path, params) if (use_cache and self._cache_dir) else None

        if cache_path and cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                log.debug("cache hit %s", cache_path.name)
                return HttpResponse(
                    status_code=cached["status_code"],
                    headers=cached["headers"],
                    json=cached["json"],
                    from_cache=True,
                )
            except Exception as exc:
                log.warning("cache read failed (%s); refetching", exc)

        try:
            response = self._fetch(path, params)
        except RetryError as exc:
            raise HttpError(f"GET {path} failed after {self._retries} attempts: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HttpError(f"GET {path} failed: {exc}") from exc
        except (JSONDecodeError, ValueError) as exc:
            # NSO occasionally serves an HTML challenge / "Website Filtered"
            # page even with a 200; this catches that and the more general
            # "got HTML where JSON was expected" failure mode.
            raise HttpError(f"GET {path} returned non-JSON body: {exc}") from exc

        if cache_path is not None:
            payload = {
                "status_code": response.status_code,
                "headers": response.headers,
                "json": response.json,
            }
            try:
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            except OSError as exc:  # pragma: no cover - cache write is best-effort
                log.warning("cache write failed: %s", exc)

        if self._delay:
            time.sleep(self._delay)

        return response

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- internals --------------------------------------------------------
    def _fetch(self, path: str, params: dict[str, Any]) -> HttpResponse:
        # We use a closure so tenacity's retry parameters can read the
        # instance attributes set on __init__.
        @retry(
            stop=stop_after_attempt(self._retries),
            wait=wait_exponential(multiplier=self._retry_backoff_s, min=1, max=30),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        )
        def _do() -> HttpResponse:
            log.debug("GET %s params=%s", path, params)
            r = self._client.get(path, params=params)
            # Treat 5xx as retriable, 4xx as terminal.
            if 500 <= r.status_code < 600:
                raise httpx.HTTPStatusError(
                    f"{r.status_code} {r.reason_phrase}", request=r.request, response=r
                )
            r.raise_for_status()
            # Detect non-JSON bodies *before* trying to decode. This is what
            # NSO + an upstream "Website Filtered" page hit us with.
            content_type = (r.headers.get("content-type") or "").lower()
            if "json" not in content_type:
                snippet = r.text[:160].strip().replace("\n", " ")
                raise HttpError(
                    f"expected JSON, got content-type={content_type!r}; body starts: {snippet!r}"
                )
            return HttpResponse(
                status_code=r.status_code,
                headers=dict(r.headers),
                json=r.json(),
                from_cache=False,
            )

        return _do()

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        assert self._cache_dir is not None
        # Canonicalise: sort params, urlencode, hash. This keeps filenames
        # short and stable regardless of key insertion order.
        qs = urlencode(sorted(params.items()), doseq=True)
        key = f"{path}?{qs}".encode()
        digest = hashlib.sha1(key).hexdigest()[:16]
        # A human-readable prefix helps debugging.
        slug = path.strip("/").replace("/", "_") or "root"
        return self._cache_dir / f"{slug}__{digest}.json"
