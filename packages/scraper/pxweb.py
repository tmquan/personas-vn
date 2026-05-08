"""PX-Web v2 REST client.

The Vietnamese General Statistics Office runs a standard PX-Web
installation at https://pxweb.nso.gov.vn/. Its REST API follows the
PC-Axis API spec used by most national statistics offices (Sweden,
Norway, Finland, etc.):

    GET  /api/v1/{lang}/                            -> list databases
    GET  /api/v1/{lang}/{db}/                       -> list folders + tables
    GET  /api/v1/{lang}/{db}/.../{table}.px         -> table metadata
    POST /api/v1/{lang}/{db}/.../{table}.px         -> table data query

Catalog entries have ``"type": "l"`` for folders and ``"type": "t"`` for
tables. Data queries POST a JSON body of the form::

    {
      "query":    [{"code": "VAR", "selection": {"filter": "item", "values": ["0", "1"]}}],
      "response": {"format": "json"}
    }

An empty ``query: []`` returns the full table.

We expose three things:

* :class:`PxWebClient` — thin wrapper around :class:`HttpClient`.
* :class:`PxWebTable`  — typed metadata + data for one table.
* :func:`walk_catalog` — recursive walk of a database, returning the
  list of every reachable table path.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packages.common.http import HttpClient, HttpError
from packages.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Catalog entry / table
# ---------------------------------------------------------------------------
@dataclass
class CatalogEntry:
    """One row in a catalog listing — either a folder or a table."""

    id: str
    text: str
    type: str  # "l" = folder, "t" = table
    parent_path: tuple[str, ...]
    updated: str | None = None

    @property
    def is_table(self) -> bool:
        return self.type == "t"

    @property
    def full_path(self) -> tuple[str, ...]:
        return (*self.parent_path, self.id)


@dataclass
class PxWebTable:
    """Metadata + data for one PX-Web table."""

    path: tuple[str, ...]                 # (db, [folders...], "Vxx.yy.px")
    title: str
    variables: list[dict[str, Any]] = field(default_factory=list)
    data_columns: list[dict[str, str]] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)

    @property
    def table_id(self) -> str:
        # Strip the trailing ".px" so the ID matches the on-screen matrix code.
        return self.path[-1].removesuffix(".px")

    @property
    def database(self) -> str:
        return self.path[0]

    @property
    def n_cells(self) -> int:
        return len(self.data)

    def variable_codes(self) -> list[str]:
        return [v.get("code", "") for v in self.variables]

    def value_text(self, var_code: str, value: str) -> str:
        """Map an opaque value id (``"0"``, ``"1"``, ...) to its label."""
        for v in self.variables:
            if v.get("code") == var_code:
                values = v.get("values") or []
                texts = v.get("valueTexts") or []
                try:
                    idx = values.index(value)
                    return texts[idx] if idx < len(texts) else value
                except ValueError:
                    return value
        return value

    def to_records(self) -> list[dict[str, Any]]:
        """Flatten ``data`` into a list of dicts ``{var: label, ..., value: float}``.

        This is the form the persona-generator distributions consume.
        Rows whose value is ``..`` (PX-Web's NA marker) are skipped.
        """
        var_codes = self.variable_codes()
        out: list[dict[str, Any]] = []
        for row in self.data:
            keys = row.get("key") or []
            values = row.get("values") or []
            if not values or values[0] in ("", "..", ".", "-"):
                continue
            try:
                val = float(values[0].replace(",", "."))
            except (TypeError, ValueError):
                continue
            rec: dict[str, Any] = {}
            for var_code, key in zip(var_codes, keys, strict=False):
                rec[var_code] = self.value_text(var_code, key)
            rec["value"] = val
            out.append(rec)
        return out


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class PxWebClient:
    """Thin convenience wrapper for the PX-Web REST API."""

    def __init__(self, http: HttpClient, *, lang: str = "vi", api_path: str = "/api/v1") -> None:
        self._http = http
        self._lang = lang
        self._api = api_path.rstrip("/")
        self._delay_s = 0.0  # set by the runner

    @property
    def lang(self) -> str:
        return self._lang

    def list_databases(self) -> list[CatalogEntry]:
        path = f"{self._api}/{self._lang}/"
        resp = self._http.get_json(path)
        return [
            CatalogEntry(
                id=item.get("dbid") or item.get("id") or "",
                text=item.get("text", ""),
                type=item.get("type", "l"),
                parent_path=(),
            )
            for item in (resp.json or [])
        ]

    def list_folder(self, parts: tuple[str, ...]) -> list[CatalogEntry]:
        path = f"{self._api}/{self._lang}/" + "/".join(parts)
        resp = self._http.get_json(path)
        return [
            CatalogEntry(
                id=item.get("id", ""),
                text=item.get("text", ""),
                type=item.get("type", "l"),
                parent_path=parts,
                updated=item.get("updated"),
            )
            for item in (resp.json or [])
        ]

    def get_metadata(self, parts: tuple[str, ...]) -> dict[str, Any]:
        """Fetch a table's metadata (variables, value labels, etc.)."""
        path = f"{self._api}/{self._lang}/" + "/".join(parts)
        resp = self._http.get_json(path)
        return resp.json or {}

    def get_data(self, parts: tuple[str, ...]) -> dict[str, Any]:
        """Fetch the full data array for a table (POST with empty query).

        We POST manually because the project's :class:`HttpClient` only
        wraps GET. The HTTPX client itself is reused via the underscored
        attribute. Errors are wrapped in :class:`HttpError`.
        """
        path = f"{self._api}/{self._lang}/" + "/".join(parts)
        body = {"query": [], "response": {"format": "json"}}
        try:
            r = self._http._client.post(path, json=body)
            r.raise_for_status()
            content_type = (r.headers.get("content-type") or "").lower()
            if "json" not in content_type:
                snippet = r.text[:120].replace("\n", " ")
                raise HttpError(
                    f"POST {path} returned content-type={content_type!r}; body starts: {snippet!r}"
                )
            return r.json()
        except HttpError:
            raise
        except Exception as exc:  # network / JSON / etc.
            raise HttpError(f"POST {path} failed: {exc}") from exc

    def fetch_table(self, parts: tuple[str, ...]) -> PxWebTable:
        """Fetch metadata + data and return a :class:`PxWebTable`."""
        meta = self.get_metadata(parts)
        if self._delay_s:
            time.sleep(self._delay_s)
        data = self.get_data(parts)
        if self._delay_s:
            time.sleep(self._delay_s)
        return PxWebTable(
            path=parts,
            title=str(meta.get("title", parts[-1])),
            variables=list(meta.get("variables") or []),
            data_columns=list(data.get("columns") or []),
            data=list(data.get("data") or []),
        )


# ---------------------------------------------------------------------------
# Catalog walker
# ---------------------------------------------------------------------------
def walk_catalog(
    client: PxWebClient,
    *,
    skip_dbs: tuple[str, ...] = (),
    only_dbs: tuple[str, ...] = (),
    delay_s: float = 0.0,
) -> Iterator[CatalogEntry]:
    """Yield every table reachable from the API root.

    ``skip_dbs`` / ``only_dbs`` filter at the database level by exact id
    match. ``delay_s`` adds a small sleep between catalog requests to be
    polite to the upstream IIS server.
    """
    dbs = client.list_databases()
    for db in dbs:
        if only_dbs and db.id not in only_dbs:
            continue
        if db.id in skip_dbs:
            continue
        log.info("walking database: %s", db.id)
        try:
            yield from _walk_path(client, (db.id,), delay_s=delay_s)
        except HttpError as exc:
            log.warning("skip db %s: %s", db.id, exc)


def _walk_path(
    client: PxWebClient,
    parts: tuple[str, ...],
    *,
    delay_s: float,
) -> Iterator[CatalogEntry]:
    try:
        items = client.list_folder(parts)
    except HttpError as exc:
        log.warning("skip path /%s: %s", "/".join(parts), exc)
        return
    if delay_s:
        time.sleep(delay_s)
    for item in items:
        if item.is_table:
            yield item
        else:
            yield from _walk_path(client, item.full_path, delay_s=delay_s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_filename(parts: tuple[str, ...]) -> Path:
    """Build a filesystem-safe filename for a table path.

    Replaces spaces and Vietnamese diacritics with their slug equivalents so
    the file works on macOS APFS, Linux ext4, and Windows NTFS.
    """
    import re
    import unicodedata

    flat = "__".join(parts)
    nfkd = unicodedata.normalize("NFKD", flat)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_marks = no_marks.replace("đ", "d").replace("Đ", "d")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", no_marks).strip("-")
    return Path(slug or "table")
