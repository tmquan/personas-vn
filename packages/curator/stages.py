"""Curation pipeline stages.

The five stages map onto NeMo Curator's ``ProcessingStage`` lifecycle:

* ``setup()`` — one-time initialisation (e.g. load an embedding model).
* ``process(record_or_batch) -> record_or_batch`` — pure transformation.
* ``teardown()`` — release resources.

When ``nemo_curator`` is importable, every stage class can ALSO be wrapped
into a real NeMo Curator ``ProcessingStage`` via :func:`as_nc_stage` — so
the same code runs locally OR inside a Ray-backed Curator pipeline without
modification. When NeMo Curator is missing, the in-house executor in
:mod:`packages.curator.pipeline` runs the stage callables directly.

Every stage is deliberately tiny: it reads JSONL from the previous stage's
directory and writes JSONL/Parquet to its own. That makes each stage
re-runnable in isolation (``--from extract`` etc.), and means a partial
crash never destroys earlier work.

The download stage supports two backends, picked by ``download.source``
in ``configs/curator.yaml``:

* ``pxweb`` (default) — pulls structured PC-Axis tables from
  ``pxweb.nso.gov.vn``. This is the real statistical database and the
  source the rest of the pipeline (ontology, persona generator) is
  grounded in.
* ``wordpress`` (legacy) — the older WordPress REST scraper kept around
  for the curator's text-embedding demo (it has actual prose to embed,
  unlike the numeric tables).
"""

from __future__ import annotations

import html
import json
import re
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.config import Config
from packages.common.http import HttpClient, HttpError
from packages.common.logging import get_logger
from packages.common.paths import ensure_dir
from packages.scraper.pxweb import (
    PxWebClient,
    PxWebTable,
    safe_filename,
    walk_catalog,
)
from packages.scraper.pxweb_html import (
    PxWebHtmlClient,
    discover_html_databases,
    list_html_tables,
    xlsx_to_records,
)

log = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_VN_DIACRITIC_RE = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]")


def _strip_html(value: str | None) -> str:
    if not value:
        return ""
    no_tags = _TAG_RE.sub(" ", value)
    return _WS_RE.sub(" ", html.unescape(no_tags)).strip()


def _detect_language(text: str) -> str:
    """Cheap language detection: presence of Vietnamese diacritics + a few
    high-signal common words. Good enough for triage; we do not need a real
    fastText model for this dataset.
    """
    if not text:
        return "unknown"
    has_vn = bool(_VN_DIACRITIC_RE.search(text))
    text_lower = text.lower()
    en_hits = sum(1 for w in (" the ", " of ", " and ", " in ", " for ") if w in text_lower)
    if has_vn and en_hits < 5:
        return "vi"
    if en_hits >= 2 and not has_vn:
        return "en"
    if has_vn:
        return "vi"
    return "en" if en_hits else "unknown"


# ---------------------------------------------------------------------------
# Stage 1 — download (full NSO crawl)
# ---------------------------------------------------------------------------
@dataclass
class DownloadStage:
    """Download stage with two backends:

    * ``pxweb`` — walks the PX-Web v2 catalog and dumps each table as
      ``{out_dir}/pxweb/{lang}/{slug}.parquet`` (long-format records:
      one row per cell with categorical labels + value), plus a
      ``{slug}.metadata.json`` capturing the variables/labels.
    * ``wordpress`` — legacy WordPress REST scraper kept around so the
      curator's embedding demo still has free-text to chew on.
    """

    config: Config
    out_dir: Path

    name: str = "download"

    def setup(self) -> None:
        self._cache_dir = ensure_dir(self.out_dir / "_cache")

    def teardown(self) -> None:
        pass

    def run(self) -> dict[str, Any]:
        source = str(self.config.get("source", "pxweb")).lower()
        if source == "pxweb":
            return self._run_pxweb()
        if source == "wordpress":
            return self._run_wordpress()
        raise ValueError(f"unknown download.source: {source!r}; expected 'pxweb' or 'wordpress'")

    # ----- PX-Web backend --------------------------------------------------
    def _run_pxweb(self) -> dict[str, Any]:
        cfg = self.config
        out_root = ensure_dir(self.out_dir / "pxweb")
        langs = [lang for lang in (cfg.get("langs") or ["vi"]) if lang]
        only_dbs = tuple(cfg.get("only_dbs") or ())
        skip_dbs = tuple(cfg.get("skip_dbs") or ())
        # When `include_html_dbs: true` (default), brute-force scrape every
        # database visible on the website's HTML nav whose tables aren't
        # listed by the JSON API root — the V01.* / V03.* / V06.* /
        # V08.* / V12.* / V14.* / V15.* hidden series.
        include_html = bool(cfg.get("include_html_dbs", True))

        per_lang_summary: dict[str, dict[str, Any]] = {}
        total_tables = 0
        total_cells = 0

        with HttpClient(
            base_url=str(cfg.base_url),
            user_agent=str(cfg.user_agent),
            verify_ssl=bool(cfg.verify_ssl),
            timeout_s=float(cfg.request_timeout_s),
            retries=int(cfg.retries),
            retry_backoff_s=float(cfg.retry_backoff_s),
            delay_between_requests_s=float(cfg.delay_between_requests_s),
            cache_dir=self._cache_dir,
        ) as http:
            for lang in langs:
                lang_dir = ensure_dir(out_root / lang)
                catalog: list[dict[str, Any]] = []
                tables_in_lang = 0
                cells_in_lang = 0

                # ----- 1. JSON API: 5 well-behaved databases -----
                api_client = PxWebClient(http, lang=lang, api_path=str(cfg.api_path))
                api_client._delay_s = float(cfg.delay_between_requests_s)
                api_db_ids: set[str] = set()
                for entry in walk_catalog(
                    api_client,
                    only_dbs=only_dbs,
                    skip_dbs=skip_dbs,
                    delay_s=float(cfg.delay_between_requests_s),
                ):
                    parts = entry.full_path
                    if parts:
                        api_db_ids.add(parts[0])
                    try:
                        table = api_client.fetch_table(parts)
                    except HttpError as exc:
                        log.warning("skip table %s: %s", "/".join(parts), exc)
                        continue
                    written = self._write_pxweb_table(table, lang_dir)
                    catalog.append({
                        "id": table.table_id,
                        "title": table.title,
                        "source": "api",
                        "path": list(parts),
                        "n_cells": table.n_cells,
                        "variables": [
                            {"code": v.get("code"), "n_values": len(v.get("values") or [])}
                            for v in table.variables
                        ],
                        "files": written,
                        "updated": entry.updated,
                    })
                    tables_in_lang += 1
                    cells_in_lang += table.n_cells
                    log.info("[api ] downloaded %-12s n_cells=%-6d (%s)",
                             table.table_id, table.n_cells, table.title[:50])

                # ----- 2. HTML nav: 7 hidden databases -----
                html_tables_count = 0
                html_cells_count = 0
                if include_html and lang == "vi":
                    html_tables_count, html_cells_count = self._download_html_dbs(
                        http_client=http,
                        lang=lang,
                        out_dir=lang_dir,
                        catalog=catalog,
                        already_have=api_db_ids,
                        only_dbs=only_dbs,
                        skip_dbs=skip_dbs,
                    )
                    tables_in_lang += html_tables_count
                    cells_in_lang += html_cells_count

                catalog_path = lang_dir / "_catalog.json"
                catalog_path.write_text(
                    json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                per_lang_summary[lang] = {
                    "n_tables":       tables_in_lang,
                    "n_cells":        cells_in_lang,
                    "n_tables_api":   tables_in_lang - html_tables_count,
                    "n_tables_html":  html_tables_count,
                    "catalog":        str(catalog_path),
                }
                total_tables += tables_in_lang
                total_cells += cells_in_lang

        return {
            "source": "pxweb",
            "langs": per_lang_summary,
            "total_tables": total_tables,
            "total_cells": total_cells,
        }

    # ----- HTML-form backend (for DBs the JSON API hides) ------------------
    def _download_html_dbs(
        self, *,
        http_client: HttpClient,
        lang: str,
        out_dir: Path,
        catalog: list[dict[str, Any]],
        already_have: set[str],
        only_dbs: tuple[str, ...],
        skip_dbs: tuple[str, ...],
    ) -> tuple[int, int]:
        """Discover + scrape every PX-Web database the JSON API doesn't list."""
        cfg = self.config
        # 1. Discover databases via the website nav.
        try:
            html_dbs = discover_html_databases(http_client._client, lang=lang)
        except Exception as exc:
            log.warning("HTML db discovery failed: %s", exc)
            return (0, 0)
        new_dbs = [
            db for db in html_dbs
            if db not in already_have
            and (not only_dbs or db in only_dbs)
            and db not in skip_dbs
        ]
        log.info("[html] discovered %d hidden databases (%d total via website)",
                  len(new_dbs), len(html_dbs))

        n_tables = 0
        n_cells = 0
        delay_s = float(cfg.delay_between_requests_s)
        for db in new_dbs:
            try:
                tables = list_html_tables(http_client._client, db, lang=lang)
            except Exception as exc:
                log.warning("[html] skip db %s: %s", db, exc)
                continue
            log.info("[html] %s: %d tables", db, len(tables))
            time.sleep(delay_s)
            for entry in tables:
                try:
                    n_added = self._download_html_table(
                        db=db,
                        table_id=entry.table_id,
                        title=entry.title,
                        lang=lang,
                        out_dir=out_dir,
                        catalog=catalog,
                    )
                except Exception as exc:
                    log.warning("[html] %s/%s failed: %s", db, entry.table_id, exc)
                    continue
                if n_added:
                    n_tables += 1
                    n_cells += n_added
                time.sleep(delay_s)
        return (n_tables, n_cells)

    def _download_html_table(
        self, *,
        db: str,
        table_id: str,
        title: str,
        lang: str,
        out_dir: Path,
        catalog: list[dict[str, Any]],
    ) -> int:
        """Drive the form workflow for one table; persist parquet + metadata.

        Returns the number of long-format cells written, or 0 if the table
        ended up empty (which can happen for malformed PC-Axis matrices).
        """
        # Use a fresh client per table — PX-Web's session is sticky and
        # occasionally returns the wrong selection if we reuse one across
        # tables.
        with PxWebHtmlClient(
            base_url=str(self.config.base_url),
            user_agent=str(self.config.user_agent),
            verify_ssl=bool(self.config.verify_ssl),
            timeout_s=float(self.config.request_timeout_s),
        ) as cl:
            xlsx = cl.fetch_table_xlsx(db, table_id, lang=lang)

        title_from_xlsx, records, dim_names = xlsx_to_records(xlsx)
        if not records:
            log.warning("[html] %s/%s: no parsable cells (xlsx=%d B)",
                         db, table_id, len(xlsx))
            return 0

        # Persist matching the API-fetched layout: parquet + metadata.json.
        path = (db, db, table_id)
        slug = safe_filename(path)
        parquet_path = out_dir / f"{slug}.parquet"
        meta_path = out_dir / f"{slug}.metadata.json"

        df = pd.DataFrame(records)
        df.to_parquet(parquet_path, index=False)

        meta = {
            "table_id": table_id.removesuffix(".px"),
            "database": db,
            "title": title_from_xlsx or title,
            "path": list(path),
            "source": "html_form",
            "variables": [
                {"code": dim_names[0], "n_values": df[dim_names[0]].nunique()
                  if dim_names[0] in df.columns else 0},
                {"code": dim_names[1], "n_values": df[dim_names[1]].nunique()
                  if dim_names[1] in df.columns else 0},
            ],
            "n_cells": len(df),
            "n_kept":  len(df),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                              encoding="utf-8")

        catalog.append({
            "id":     table_id.removesuffix(".px"),
            "title":  meta["title"],
            "source": "html_form",
            "path":   list(path),
            "n_cells": len(df),
            "variables": meta["variables"],
            "files":  {"data": str(parquet_path), "metadata": str(meta_path)},
            "updated": None,
        })
        log.info("[html] downloaded %-12s n_cells=%-6d (%s)",
                  table_id, len(df), title[:50])
        return len(df)

    @staticmethod
    def _write_pxweb_table(table: PxWebTable, lang_dir: Path) -> dict[str, str]:
        slug = safe_filename(table.path)
        records = table.to_records()
        df = pd.DataFrame(records)
        parquet_path = lang_dir / f"{slug}.parquet"
        meta_path = lang_dir / f"{slug}.metadata.json"
        df.to_parquet(parquet_path, index=False)
        meta = {
            "table_id": table.table_id,
            "database": table.database,
            "title": table.title,
            "path": list(table.path),
            "variables": table.variables,
            "data_columns": table.data_columns,
            "n_cells": table.n_cells,
            "n_kept": len(df),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"data": str(parquet_path), "metadata": str(meta_path)}

    # ----- legacy WordPress backend ----------------------------------------
    def _run_wordpress(self) -> dict[str, Any]:
        cfg = self.config
        ensure_dir(self.out_dir)
        ensure_dir(self._cache_dir)

        langs = list(cfg.get("langs", [])) or [""]
        endpoints_meta: dict[str, dict[str, Any]] = {}

        with HttpClient(
            base_url=str(cfg.base_url),
            user_agent=str(cfg.user_agent),
            verify_ssl=bool(cfg.verify_ssl),
            timeout_s=float(cfg.request_timeout_s),
            retries=int(cfg.retries),
            retry_backoff_s=float(cfg.retry_backoff_s),
            delay_between_requests_s=float(cfg.delay_between_requests_s),
            cache_dir=self._cache_dir,
        ) as http:
            for ep in cfg.endpoints:
                ep_d = ep.to_dict() if isinstance(ep, Config) else dict(ep)
                ep_name = ep_d["name"]
                ep_path = f"{cfg.api_path.rstrip('/')}/{ep_d['path'].lstrip('/')}"
                ep_per_page = int(ep_d.get("per_page", cfg.per_page))
                ep_extra = ep_d.get("extra") or {}
                total_for_endpoint = 0
                lang_counts: dict[str, int] = {}

                for lang in langs:
                    out_path = self.out_dir / f"{ep_name}__{(lang or 'all')}.jsonl"
                    n = 0
                    with out_path.open("w", encoding="utf-8") as fh:
                        for record in self._walk_wp(
                            http, ep_path, ep_per_page,
                            int(cfg.max_pages_per_endpoint),
                            lang, ep_extra,
                        ):
                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                            n += 1
                    lang_counts[lang or "all"] = n
                    total_for_endpoint += n
                    log.info("download %-12s lang=%-3s -> %s (%d records)",
                             ep_name, lang or "all", out_path.name, n)

                endpoints_meta[ep_name] = {
                    "path": ep_path,
                    "total": total_for_endpoint,
                    "by_lang": lang_counts,
                }

        return {"source": "wordpress", "endpoints": endpoints_meta}

    @staticmethod
    def _walk_wp(
        http: HttpClient, path: str, per_page: int, max_pages: int,
        lang: str, extra: dict[str, Any],
    ) -> Iterable[dict[str, Any]]:
        for page in range(1, max_pages + 1):
            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if lang:
                params["lang"] = lang
            params.update(extra)
            try:
                resp = http.get_json(path, params=params)
            except HttpError as exc:
                msg = str(exc).lower()
                if "rest_post_invalid_page_number" in msg or "400" in msg or "404" in msg:
                    return
                log.warning("download stopped at %s page=%d (lang=%s): %s",
                            path, page, lang or "all", exc)
                return
            records = resp.json if isinstance(resp.json, list) else []
            if not records:
                return
            yield from records
            total_pages = int(resp.headers.get("x-wp-totalpages") or 0)
            if total_pages and page >= total_pages:
                return


# ---------------------------------------------------------------------------
# Stage 2 — parse
# ---------------------------------------------------------------------------
@dataclass
class ParseStage:
    """Read every ``raw/*.jsonl``, strip HTML from title/excerpt/content,
    detect language, and emit a unified ``parsed/parsed.jsonl`` with one
    record per WP post / page.
    """

    config: Config
    in_dir: Path
    out_dir: Path
    name: str = "parse"

    def setup(self) -> None:
        ensure_dir(self.out_dir)
        self._min_chars = int(self.config.get("min_text_chars", 60))

    def teardown(self) -> None:
        pass

    def run(self) -> dict[str, Any]:
        # Auto-setup if the caller forgot; keeps stages usable as plain
        # callables in tests / scripts.
        if not hasattr(self, "_min_chars"):
            self.setup()
        # Dispatch on what the download stage actually produced. The
        # PX-Web backend writes ``raw/pxweb/<lang>/*.parquet`` + sidecar
        # ``*.metadata.json``; the legacy WordPress backend writes
        # ``raw/{posts,pages,categories,tags}__<lang>.jsonl`` at the top
        # of ``raw/``. We honour either.
        pxweb_dir = self.in_dir / "pxweb"
        if pxweb_dir.exists() and any(pxweb_dir.glob("*/*.parquet")):
            return self._run_pxweb(pxweb_dir)
        return self._run_wordpress()

    # ----- PX-Web ----------------------------------------------------------
    def _run_pxweb(self, pxweb_dir: Path) -> dict[str, Any]:
        """Emit one ``parsed.jsonl`` record per PX-Web *table* (not per cell).

        The text body for each record is a synthetic descriptor combining
        the table title, its variable labels, and the year range covered.
        That lets the downstream embed stage map every table to a single
        2-D point so the visualiser can show the whole 502-table catalog
        at once instead of 326k cells.
        """
        import pandas as pd

        out_path = self.out_dir / "parsed.jsonl"
        kept = 0
        dropped = 0
        with out_path.open("w", encoding="utf-8") as out:
            for lang_dir in sorted(p for p in pxweb_dir.iterdir() if p.is_dir()):
                lang = lang_dir.name
                for parquet_path in sorted(lang_dir.glob("*.parquet")):
                    meta_path = parquet_path.with_suffix(".metadata.json")
                    if not meta_path.exists():
                        dropped += 1
                        continue
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        df = pd.read_parquet(parquet_path)
                    except Exception as exc:                          # noqa: BLE001
                        log.warning("parse: failed to read %s (%s)", parquet_path.name, exc)
                        dropped += 1
                        continue
                    parsed = self._parse_pxweb_table(meta, df, lang=lang)
                    if parsed is None:
                        dropped += 1
                        continue
                    out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    kept += 1
        log.info("parse: pxweb mode  kept=%d dropped=%d -> %s",
                  kept, dropped, out_path.name)
        return {"kept": kept, "dropped": dropped, "mode": "pxweb",
                 "output": str(out_path)}

    def _parse_pxweb_table(
        self, meta: dict[str, Any], df: "pd.DataFrame", *, lang: str,
    ) -> dict[str, Any] | None:
        import re as _re

        table_id = (meta.get("table_id") or meta.get("matrix") or "").strip()
        title = (meta.get("title") or "").strip()
        if not table_id:
            return None
        database = (meta.get("database") or "").strip()

        # Variable labels — every PX-Web matrix has a list of variables in
        # its metadata.json. Use ``valueTexts`` when available (human-
        # readable) but fall back to ``text`` / ``code``.
        variable_labels: list[str] = []
        for v in meta.get("variables") or []:
            label = (v.get("text") or v.get("code") or "").strip()
            if label:
                variable_labels.append(label)

        # Try to extract the year range from any column whose name looks
        # like ``Năm`` (Vietnamese for "year") — works for ~85% of tables.
        year_min: int | None = None
        year_max: int | None = None
        for col in df.columns:
            cl = str(col).lower()
            if cl == "năm" or "năm" in cl or "year" in cl:
                ys = [int(m.group(0))
                      for v in df[col].astype(str)
                      if (m := _re.search(r"\d{4}", v))]
                if ys:
                    year_min, year_max = min(ys), max(ys)
                break

        # Synthesised text body: the embedder + TF-IDF read this. We keep
        # the title first (most signal), then variable labels (the
        # statistical "schema"), then year range + database for context.
        parts = [title] if title else []
        if variable_labels:
            parts.append("Biến: " + " · ".join(variable_labels))
        if year_min and year_max:
            parts.append(f"Năm: {year_min}–{year_max}")
        if database:
            parts.append(f"Cơ sở dữ liệu: {database}")
        text = "\n".join(parts)
        if len(text) < self._min_chars:
            return None

        return {
            "id":        f"nso:pxweb:{lang}:{table_id}",
            "kind":      "pxweb",
            "table_id":  table_id,
            "lang":      lang,
            "title":     title,
            "slug":      meta.get("slug") or "",
            "database":  database,
            "path":      meta.get("path") or [],
            "variables": variable_labels,
            "n_cells":   int(len(df)),
            "year_min":  year_min,
            "year_max":  year_max,
            "text":      text,
        }

    # ----- WordPress (legacy) ---------------------------------------------
    def _run_wordpress(self) -> dict[str, Any]:
        out_path = self.out_dir / "parsed.jsonl"
        kept = 0
        dropped = 0
        with out_path.open("w", encoding="utf-8") as out:
            for raw_file in sorted(self.in_dir.glob("*.jsonl")):
                # We only parse posts + pages; categories/tags become a side-car.
                stem = raw_file.stem
                kind = stem.split("__", 1)[0]
                if kind not in ("posts", "pages"):
                    continue
                lang_hint = stem.split("__", 1)[1] if "__" in stem else "all"
                with raw_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            dropped += 1
                            continue
                        parsed = self._parse_record(record, kind=kind, lang_hint=lang_hint)
                        if not parsed:
                            dropped += 1
                            continue
                        out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        kept += 1
        log.info("parse: wordpress mode  kept=%d dropped=%d -> %s",
                  kept, dropped, out_path.name)
        return {"kept": kept, "dropped": dropped, "mode": "wordpress",
                 "output": str(out_path)}

    def _parse_record(
        self, record: dict[str, Any], *, kind: str, lang_hint: str,
    ) -> dict[str, Any] | None:
        title_obj = record.get("title") or {}
        excerpt_obj = record.get("excerpt") or {}
        content_obj = record.get("content") or {}
        title = _strip_html(title_obj.get("rendered") if isinstance(title_obj, dict) else title_obj)
        excerpt = _strip_html(excerpt_obj.get("rendered") if isinstance(excerpt_obj, dict) else excerpt_obj)
        content = _strip_html(content_obj.get("rendered") if isinstance(content_obj, dict) else content_obj)
        text_parts = [p for p in (title, excerpt, content) if p]
        text = " ".join(text_parts)
        if len(text) < self._min_chars:
            return None
        lang = record.get("lang") or _detect_language(text)
        if lang_hint and lang_hint != "all" and not record.get("lang"):
            lang = lang_hint
        return {
            "id": f"nso:{kind}:{record.get('id')}",
            "kind": kind,
            "wp_id": record.get("id"),
            "lang": lang,
            "url": record.get("link"),
            "slug": record.get("slug"),
            "date": record.get("date"),
            "modified": record.get("modified"),
            "title": title,
            "excerpt": excerpt,
            "content": content,
            "text": text,
            "categories": record.get("categories") or [],
            "tags": record.get("tags") or [],
        }


# ---------------------------------------------------------------------------
# Stage 3 — extract
# ---------------------------------------------------------------------------
@dataclass
class ExtractStage:
    """Map each parsed record onto its ontology domain, extract top-N TF-IDF
    keywords, and emit a flat ``extracted.jsonl`` ready for embedding.
    """

    config: Config
    in_dir: Path
    out_dir: Path
    ontology_config: str = "configs/ontology.yaml"
    name: str = "extract"

    def setup(self) -> None:
        from packages.ontology.registry import load_registry

        ensure_dir(self.out_dir)
        # WordPress backend needs the raw category dump to map post →
        # domain. PX-Web tables don't ship one (the database name is
        # already the category) so the loop below is a no-op there.
        cat_index: dict[int, dict[str, Any]] = {}
        for raw_file in self.in_dir.parent.joinpath("raw").glob("categories__*.jsonl"):
            with raw_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        cat = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = int(cat.get("id") or 0)
                    if cid:
                        cat_index[cid] = cat
        self._cat_index = cat_index
        self._registry = load_registry(self.ontology_config)
        self._top_n = int(self.config.get("top_keywords", 8))

    def teardown(self) -> None:
        pass

    def run(self) -> dict[str, Any]:
        if not hasattr(self, "_top_n"):
            self.setup()
        in_path = self.in_dir / "parsed.jsonl"
        out_path = self.out_dir / "extracted.jsonl"
        records: list[dict[str, Any]] = []
        with in_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # PX-Web records carry a ``database`` string (e.g.
                # ``"Dân số và lao động"``); WordPress records carry a
                # list of integer ``categories`` ids that need joining
                # against the cat_index. Pick whichever resolver fits.
                if rec.get("kind") == "pxweb":
                    rec["domain_id"] = self._resolve_domain_pxweb(rec.get("database"))
                else:
                    rec["domain_id"] = self._resolve_domain(rec.get("categories") or [])
                records.append(rec)

        keywords_list = self._compute_keywords([r["text"] for r in records])
        for rec, kws in zip(records, keywords_list):
            rec["keywords"] = kws

        with out_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Domain histogram is cheap to compute now and useful in the manifest.
        hist = Counter(r["domain_id"] for r in records)
        log.info("extract: %d records, %d domains", len(records), len(hist))
        return {
            "n": len(records),
            "domain_histogram": dict(hist),
            "output": str(out_path),
        }

    def _resolve_domain(self, category_ids: list[int]) -> str:
        for cid in category_ids:
            cat = self._cat_index.get(int(cid))
            if not cat:
                continue
            domain = self._registry.map_nso_category(cat.get("name"), cat.get("slug"))
            if domain != "other":
                return domain
        return "other"

    def _resolve_domain_pxweb(self, database: str | None) -> str:
        """Map a PX-Web database name → ontology domain id.

        ``OntologyRegistry.map_nso_category`` already does substring
        matching against the curated alias index (Vietnamese + English
        forms), so passing the database label directly works for the
        12 NSO databases — ``"Dân số và lao động"`` → ``population``,
        ``"Doanh nghiệp"`` → ``enterprises``, etc. Falls back to
        ``"other"`` if the database is empty / unrecognised.
        """
        if not database:
            return "other"
        return self._registry.map_nso_category(database, slug=None)

    def _compute_keywords(self, texts: list[str]) -> list[list[str]]:
        if not texts:
            return []
        vectorizer_kind = str(self.config.get("vectorizer", "tfidf"))
        if vectorizer_kind != "tfidf":
            return [[] for _ in texts]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            log.warning("scikit-learn missing; skipping keyword extraction")
            return [[] for _ in texts]
        # Multilingual-friendly: use char-agnostic token pattern + bigrams.
        ngram_range = tuple(self.config.get("ngram_range", [1, 2]))
        try:
            vec = TfidfVectorizer(
                max_df=float(self.config.get("max_df", 0.85)),
                min_df=int(self.config.get("min_df", 2)),
                ngram_range=ngram_range,  # type: ignore[arg-type]
                token_pattern=r"(?u)\b[\wÀ-ỹ]{3,}\b",
            )
            matrix = vec.fit_transform(texts)
        except ValueError as exc:
            log.warning("TF-IDF skipped: %s", exc)
            return [[] for _ in texts]
        vocab = vec.get_feature_names_out()
        out: list[list[str]] = []
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i).toarray()[0]
            if not row.any():
                out.append([])
                continue
            top = sorted(enumerate(row), key=lambda x: -x[1])[: self._top_n]
            out.append([str(vocab[j]) for j, w in top if w > 0])
        return out


# ---------------------------------------------------------------------------
# Stage 4 — embed
# ---------------------------------------------------------------------------
@dataclass
class EmbedStage:
    """Run a text-embedding model over every record's ``text`` field.

    Backend is auto-selected from the model name: anything starting with
    ``nvidia/`` goes through NVIDIA NIM hosted embeddings (set
    ``$PERSONAS_VN_LLM_API_KEY`` or ``$NVIDIA_API_KEY``); everything else
    is loaded as a local ``sentence-transformers`` model. Set
    ``embed.backend`` in ``configs/curator.yaml`` to ``"local"`` /
    ``"nim"`` to override. Output is a parquet of ``(id, vector)``.
    """

    config: Config
    in_dir: Path
    out_dir: Path
    name: str = "embed"

    def setup(self) -> None:
        ensure_dir(self.out_dir)
        from packages.personas.embed.backends import (
            NimEmbeddingsConfig,
            make_embedding_backend,
        )

        nim_cfg = NimEmbeddingsConfig(
            base_url=str(self.config.get("base_url") or NimEmbeddingsConfig().base_url),
            api_key_env=str(self.config.get("api_key_env", "PERSONAS_VN_LLM_API_KEY")),
            truncate=str(self.config.get("truncate", "END")),
            request_delay_s=float(self.config.get("request_delay_s", 0.0)),
        )
        self._backend = make_embedding_backend(
            model=str(self.config.model),
            backend=str(self.config.get("backend", "auto")),
            device=str(self.config.get("device", "cpu")),
            nim_config=nim_cfg,
        )
        self._batch_size = int(self.config.batch_size)
        self._normalize = bool(self.config.normalize)
        self._text_field = str(self.config.get("text_field", "text"))
        self._max_records = self.config.get("max_records")

    def teardown(self) -> None:
        # Drop the model reference so the GC can reclaim it; avoids holding
        # ~500MB resident across pipeline reruns inside the visualizer.
        backend = getattr(self, "_backend", None)
        if backend is not None:
            try:
                backend.close()
            finally:
                self._backend = None  # type: ignore[assignment]

    def run(self) -> dict[str, Any]:
        import pandas as pd

        in_path = self.in_dir / "extracted.jsonl"
        out_path = self.out_dir / "embedded.parquet"

        ids: list[str] = []
        texts: list[str] = []
        meta: list[dict[str, Any]] = []
        cap = int(self._max_records) if self._max_records else None
        with in_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = rec.get(self._text_field) or ""
                if not txt:
                    continue
                ids.append(rec["id"])
                texts.append(txt[:2000])  # truncate for fairness across short/long records
                meta.append({
                    "id": rec["id"],
                    "title": rec.get("title", ""),
                    "url": rec.get("url"),
                    "domain_id": rec.get("domain_id", "other"),
                    "lang": rec.get("lang", "unknown"),
                    "kind": rec.get("kind", "posts"),
                    "date": rec.get("date"),
                })
                if cap and len(ids) >= cap:
                    break

        log.info("embedding %d texts (backend=%s, model=%s, bs=%d)",
                 len(texts), self._backend.name, self._backend.model, self._batch_size)
        vectors = self._backend.encode(
            texts,
            batch_size=self._batch_size,
            normalize=self._normalize,
            input_type="passage",
            show_progress=True,
        )

        df = pd.DataFrame(meta)
        df["vector"] = list(vectors)
        df.to_parquet(out_path, index=False)
        log.info("embed: wrote %s (n=%d, dim=%d)", out_path.name, len(df), vectors.shape[1])
        return {
            "n": len(df),
            "dim": int(vectors.shape[1]),
            "model": self._backend.model,
            "backend": self._backend.name,
            "output": str(out_path),
        }


# ---------------------------------------------------------------------------
# Stage 5 — reduce
# ---------------------------------------------------------------------------
@dataclass
class ReduceStage:
    """Project the embedding vectors down to 2-D for plotting and
    optionally run a density-based cluster pass.

    Clustering is **opt-in**: set ``reduce.cluster: true`` in
    ``configs/curator.yaml`` to additionally compute HDBSCAN cluster
    labels and emit them as the ``cluster`` column of
    ``reduced.parquet``. The default is ``false`` because most
    downstream consumers (the visualiser's ontology / database /
    domain colourings, the analysis notebook scatter plots) read raw
    data columns, not cluster ids — and because computing HDBSCAN on
    a 30K-row corpus adds ~1-2 minutes to the pipeline for output
    most users don't read.
    """

    config: Config
    in_dir: Path
    out_dir: Path
    name: str = "reduce"

    def setup(self) -> None:
        ensure_dir(self.out_dir)

    def teardown(self) -> None:
        pass

    def run(self) -> dict[str, Any]:
        import numpy as np
        import pandas as pd

        in_path = self.in_dir / "embedded.parquet"
        out_path = self.out_dir / "reduced.parquet"
        df = pd.read_parquet(in_path)
        do_cluster = bool(self.config.get("cluster", False))
        if df.empty:
            log.warning("reduce: no rows in %s", in_path)
            empty = df.assign(x=[], y=[])
            if do_cluster:
                empty = empty.assign(cluster=[])
            empty.to_parquet(out_path, index=False)
            return {"n": 0, "output": str(out_path)}
        X = np.stack(df["vector"].to_list())
        algo = str(self.config.get("algorithm", "umap")).lower()
        coords = self._project(X, algo)
        out = df.drop(columns=["vector"]).copy()
        out["x"] = coords[:, 0]
        out["y"] = coords[:, 1]
        if do_cluster:
            clusters = self._cluster(X)
            out["cluster"] = clusters
            log.info("reduce: wrote %s (n=%d, algo=%s, clusters=%d)",
                     out_path.name, len(out), algo, len(set(clusters)))
        else:
            log.info("reduce: wrote %s (n=%d, algo=%s, clustering=off)",
                     out_path.name, len(out), algo)
        out.to_parquet(out_path, index=False)
        return {
            "n": len(out),
            "algorithm": algo,
            "clustered": do_cluster,
            "output": str(out_path),
        }

    def _project(self, X, algo: str):
        n = X.shape[0]
        if algo == "umap" and n >= 4:
            try:
                import umap

                reducer = umap.UMAP(
                    n_components=int(self.config.get("n_components", 2)),
                    n_neighbors=min(int(self.config.get("n_neighbors", 15)), max(2, n - 1)),
                    min_dist=float(self.config.get("min_dist", 0.1)),
                    metric=str(self.config.get("metric", "cosine")),
                    random_state=int(self.config.get("random_state", 42)),
                )
                return reducer.fit_transform(X)
            except Exception as exc:
                log.warning("UMAP failed (%s); falling back to PCA", exc)
        if algo == "tsne" and n >= 5:
            from sklearn.manifold import TSNE

            return TSNE(
                n_components=int(self.config.get("n_components", 2)),
                random_state=int(self.config.get("random_state", 42)),
                perplexity=min(30, max(5, n // 4)),
                init="pca",
            ).fit_transform(X)
        from sklearn.decomposition import PCA

        return PCA(n_components=int(self.config.get("n_components", 2))).fit_transform(X)

    @staticmethod
    def _cluster(X) -> list[int]:
        """Density-based clustering via HDBSCAN.

        We pick HDBSCAN over KMeans because it (1) finds cluster *count*
        from the data instead of demanding a fixed ``k``, (2) tolerates
        non-spherical cluster shapes which UMAP-friendly embeddings tend
        to produce, and (3) explicitly labels low-density points as
        noise (``-1``) rather than forcing every point into a cluster.

        Embeddings are L2-normalised upstream (see
        :class:`packages.curator.stages.EmbedStage` and the persona
        embed backends), so squared-Euclidean distance ranks identically
        to cosine — sklearn's HDBSCAN doesn't ship a cosine metric, but
        on normalised vectors euclidean is equivalent.

        ``min_cluster_size`` scales with the corpus so a 502-table
        catalogue gets ~10-15 clusters and a 30K-persona subset gets
        ~30-50, comparable to the previous KMeans cap.
        """
        n = X.shape[0]
        if n < 4:
            return [0] * n
        try:
            from sklearn.cluster import HDBSCAN

            min_cluster_size = max(5, n // 80)
            min_samples = max(3, min_cluster_size // 4)
            return HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            ).fit_predict(X).tolist()
        except Exception:
            return [0] * n
