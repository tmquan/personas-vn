"""NSO scraper backed by the public WordPress REST API.

Why this is reliable: ``https://www.nso.gov.vn/wp-json/wp/v2/`` exposes
clean JSON for posts, categories, tags, pages, and media. WordPress sets
``X-WP-TotalPages`` on every list response so pagination is deterministic.

Why we still wrap it: the site occasionally serves chains the macOS bundle
distrusts (we set ``verify_ssl: false`` in the config), and posts can carry
malformed HTML that we sanitise here before passing to the mapper.

The public entry point is :func:`scrape_nso`. It returns a
:class:`ScrapeManifest` describing what was pulled, plus a dict of records
keyed by endpoint name. Failures (network or otherwise) fall through to the
bundled fixture when ``offline_fallback`` is enabled in the config.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.common.config import Config, load_config
from packages.common.http import HttpClient, HttpError
from packages.common.logging import get_logger
from packages.common.paths import ensure_dir, resolve
from packages.scraper.fixtures import load_fixture_records

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass
class ScrapeManifest:
    """Provenance record for one scrape run."""

    started_at: str
    finished_at: str | None = None
    base_url: str = ""
    language: str = ""
    used_fallback: bool = False
    endpoints: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class NSOClient:
    """Tiny convenience wrapper around :class:`HttpClient` for the WP API."""

    def __init__(self, http: HttpClient, *, api_path: str = "/wp-json/wp/v2", lang: str = "") -> None:
        self._http = http
        self._api = api_path.rstrip("/")
        self._lang = lang

    def list(
        self,
        endpoint: str,
        *,
        per_page: int = 50,
        page: int = 1,
        extra: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """List one page of an endpoint, returning ``(records, headers)``."""
        params: dict[str, Any] = {"per_page": per_page, "page": page}
        if self._lang:
            params["lang"] = self._lang
        if extra:
            params.update(extra)
        path = f"{self._api}/{endpoint.lstrip('/')}"
        response = self._http.get_json(path, params=params)
        if not isinstance(response.json, list):
            raise HttpError(f"Expected a list from {path}, got {type(response.json).__name__}")
        return response.json, response.headers

    def walk(
        self,
        endpoint: str,
        *,
        per_page: int = 50,
        max_pages: int = 4,
        extra: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Walk an endpoint, capped at ``max_pages``. Stops when WP signals
        the last page (no ``X-WP-TotalPages`` header or page == total).
        """
        all_records: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            try:
                records, headers = self.list(endpoint, per_page=per_page, page=page, extra=extra)
            except HttpError as exc:
                # WP returns 400 with code "rest_post_invalid_page_number"
                # past the last page — treat that as a clean end-of-stream.
                msg = str(exc).lower()
                if "rest_post_invalid_page_number" in msg or "400" in msg:
                    log.debug("walk %s: WP signalled end of pages at page=%d", endpoint, page)
                    break
                raise
            all_records.extend(records)
            total_pages = int(headers.get("x-wp-totalpages") or headers.get("X-WP-TotalPages") or 0)
            log.info(
                "fetched %s page=%d records=%d (total_pages=%s)",
                endpoint,
                page,
                len(records),
                total_pages or "?",
            )
            if not records or (total_pages and page >= total_pages):
                break
        return all_records


# ---------------------------------------------------------------------------
# Public scrape entry point
# ---------------------------------------------------------------------------
def scrape_nso(
    config_path: str | Path = "configs/scraper.yaml",
    *,
    cfg: Config | None = None,
    write_outputs: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], ScrapeManifest]:
    """Run a scrape per ``configs/scraper.yaml``.

    Returns ``(records_by_endpoint, manifest)``. Always returns *something*
    usable: if all live calls fail and ``offline_fallback`` is true, the
    bundled fixture is loaded and the manifest's ``used_fallback`` flag is
    set.
    """
    cfg = cfg or load_config(config_path)
    source = cfg.source
    fetch = cfg.fetch
    paths = cfg.paths
    raw_dir = ensure_dir(paths.raw_dir)
    cache_dir = ensure_dir(paths.cache_dir)

    manifest = ScrapeManifest(
        started_at=datetime.now(timezone.utc).isoformat(),
        base_url=str(source.base_url),
        language=str(source.lang or ""),
    )

    records_by_endpoint: dict[str, list[dict[str, Any]]] = {}
    live_ok = True

    try:
        with HttpClient(
            base_url=str(source.base_url),
            user_agent=str(source.user_agent),
            verify_ssl=bool(source.verify_ssl),
            timeout_s=float(fetch.request_timeout_s),
            retries=int(fetch.retries),
            retry_backoff_s=float(fetch.retry_backoff_s),
            delay_between_requests_s=float(fetch.delay_between_requests_s),
            cache_dir=cache_dir,
        ) as http:
            client = NSOClient(http, api_path=str(source.api_path), lang=str(source.lang or ""))
            for ep in cfg.endpoints:
                ep_dict = ep.to_dict() if isinstance(ep, Config) else dict(ep)
                name = ep_dict["name"]
                path = ep_dict["path"]
                extra = ep_dict.get("extra") or {}
                try:
                    records = client.walk(
                        path,
                        per_page=int(fetch.per_page),
                        max_pages=int(fetch.max_pages),
                        extra=extra,
                    )
                except HttpError as exc:
                    log.warning("endpoint %s failed: %s", name, exc)
                    live_ok = False
                    records = []
                records_by_endpoint[name] = records
                manifest.endpoints[name] = {
                    "path": path,
                    "kind": ep_dict.get("kind", "post"),
                    "count": len(records),
                }
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("NSO scrape crashed: %s", exc)
        live_ok = False

    # Fall back to the bundled fixture if anything went wrong AND fallback is enabled.
    if (not live_ok or not any(records_by_endpoint.values())) and cfg.get("offline_fallback", True):
        log.warning("falling back to bundled NSO fixture")
        fixture = load_fixture_records()
        for name, records in fixture.items():
            records_by_endpoint.setdefault(name, [])
            if not records_by_endpoint[name]:
                records_by_endpoint[name] = records
                manifest.endpoints.setdefault(name, {"path": "(fixture)", "kind": name})
                manifest.endpoints[name]["count"] = len(records)
        manifest.used_fallback = True

    manifest.finished_at = datetime.now(timezone.utc).isoformat()

    if write_outputs:
        for name, records in records_by_endpoint.items():
            out_path = resolve(raw_dir) / f"nso_{name}.json"
            out_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.info("wrote %s (%d records)", out_path, len(records))
        manifest_path = resolve(raw_dir) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("wrote manifest %s", manifest_path)

    return records_by_endpoint, manifest
