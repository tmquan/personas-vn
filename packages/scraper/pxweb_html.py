"""HTML / form-based PX-Web scraper for databases the JSON API hides.

Background:

  ``https://pxweb.nso.gov.vn/api/v1/vi/`` lists 8 databases, of which 5
  expose tables (Công nghiệp, Doanh nghiệp, Dân số và lao động, Giáo
  dục, Đầu tư). The other **7 databases visible on the website**

      Đơn vị hành chính, đất đai và khí hậu      → V01.* (18 tables)
      Tài khoản quốc gia                         → V03.* (24 tables)
      Nông, lâm nghiệp và thủy sản               → V06.* (70 tables)
      Thương mại, giá cả                         → V08.* (72 tables)
      Vận tải và bưu điện                        → V12.* (24 tables)
      Y tế, văn hóa và đời sống                  → V14.* (97 tables)
      Thống kê nước ngoài                        → V15.* (10 tables)

  return 404 from the JSON API. Their data is reachable only through the
  legacy ASP.NET PX-Web web UI (``https://pxweb.nso.gov.vn/pxweb/vi/``),
  which uses a form-driven flow:

      GET  /pxweb/vi/{db}/{db}/{table}.px/?rxid=...
        → variable-selection form with __VIEWSTATE
      POST same URL with all variable values selected + ButtonViewTable
        → server stores the selection in session, returns the data view
      GET  /pxweb/vi/{db}/{db}/{table}.px/table/tableViewLayout1/
              ?rxid=...&downloadfile=FileTypeExcelX
        → an .xlsx download (Vietnamese-encoding-clean)

This module exposes two clean entry points that hide that workflow:

* :func:`discover_html_databases` — scrape ``/pxweb/vi/`` for every
  database link the website exposes.
* :func:`list_html_tables` — scrape the ``?tablelist=true`` page for
  one database.
* :class:`PxWebHtmlClient.fetch_table_xlsx` — drive the form POST and
  return parsed long-format records ready for parquet write.

The on-disk output matches what :class:`PxWebTable.to_records` produces
from the JSON API, so downstream code (:func:`packages.scraper.pxweb.safe_filename`,
the curator's writer) doesn't care which path the data took.
"""

from __future__ import annotations

import io
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import unescape as html_unescape

import httpx
import pandas as pd

from packages.common.logging import get_logger

log = get_logger(__name__)

# A stable rxid the website uses for the public pxweb instance. PX-Web's
# rxid is a session id stored in cookies; using a constant default works
# because we always do GET → POST → download in a single connection (so
# the actual session cookie supersedes whatever rxid was in the URL).
DEFAULT_RXID = "041bcbb5-1b89-48bb-9e7d-d8e55969465a"

# PX-Web's "Save as" dropdown exposes seven formats. We pick xlsx because:
#   * pandas reads it natively via openpyxl,
#   * binary, so encoding ambiguity is gone (NSO's CSV export is a known
#     CP1258 round-trip that loses some Vietnamese characters).
DOWNLOAD_FORMAT = "FileTypeExcelX"


@dataclass
class HtmlTableEntry:
    """One row in the website's per-database table list."""

    table_id: str           # e.g. "V03.01.px"
    title: str              # the human-readable matrix description
    href: str               # /pxweb/vi/{db}/{db}/V03.01.px

    @property
    def code(self) -> str:
        return self.table_id.removesuffix(".px")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_html_databases(
    client: httpx.Client,
    *,
    lang: str = "vi",
    rxid: str = DEFAULT_RXID,
) -> list[str]:
    """Scrape ``/pxweb/{lang}/`` for every database name in the navigation."""
    r = client.get(f"/pxweb/{lang}/", params={"rxid": rxid})
    r.raise_for_status()
    # The nav uses URL-encoded hrefs of shape /pxweb/{lang}/{db}/?rxid=...
    raw_hrefs = re.findall(rf'href="/pxweb/{lang}/([^/?"]+)/', r.text)
    seen: list[str] = []
    for h in raw_hrefs:
        decoded = urllib.parse.unquote(h)
        # Skip anchor-only / fragment-only entries the regex sometimes catches.
        if decoded and "?" not in decoded and decoded not in seen:
            seen.append(decoded)
    return seen


def list_html_tables(
    client: httpx.Client,
    db: str,
    *,
    lang: str = "vi",
    rxid: str = DEFAULT_RXID,
) -> list[HtmlTableEntry]:
    """Scrape ``{db}/{db}/?tablelist=true`` for every matrix in one DB.

    Returns an empty list if the database doesn't exist or has no tables.
    """
    db_path = f"/pxweb/{lang}/{urllib.parse.quote(db)}/{urllib.parse.quote(db)}/"
    r = client.get(db_path, params={"tablelist": "true", "rxid": rxid})
    if r.status_code != 200:
        return []
    # Tables look like: href="...{table}.px"  with the title as the link text.
    entries: list[HtmlTableEntry] = []
    for m in re.finditer(
        r'<a[^>]+href="(/pxweb/[^"]+/(V\d+\.[\d-]+\.px))/?[^"]*"[^>]*>([^<]+)</a>',
        r.text,
    ):
        href, table_id, title = m.group(1), m.group(2), m.group(3)
        entries.append(HtmlTableEntry(
            table_id=table_id,
            title=html_unescape(title).strip(),
            href=urllib.parse.unquote(href),
        ))
    # De-duplicate by table id while preserving order.
    seen: dict[str, HtmlTableEntry] = {}
    for e in entries:
        seen.setdefault(e.table_id, e)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Form-driven download
# ---------------------------------------------------------------------------
class PxWebHtmlClient:
    """Wraps the ASP.NET form-POST workflow for one PX-Web table.

    A *fresh* client is recommended per table — PX-Web's session
    propagates the last selection across requests, and reusing a session
    across tables can occasionally return the wrong table's data.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://pxweb.nso.gov.vn",
        user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
        verify_ssl: bool = False,
        timeout_s: float = 60.0,
        rxid: str = DEFAULT_RXID,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": user_agent, "Accept-Language": "vi,en;q=0.9"},
            verify=verify_ssl,
            timeout=timeout_s,
            follow_redirects=True,
        )
        self._rxid = rxid

    @property
    def http(self) -> httpx.Client:
        return self._client

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PxWebHtmlClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- the form workflow ----------------------------------------------
    def fetch_table_xlsx(
        self,
        db: str,
        table_id: str,
        *,
        lang: str = "vi",
    ) -> bytes:
        """Drive the form workflow and return the raw .xlsx bytes."""
        encoded_db = urllib.parse.quote(db)
        form_url = f"/pxweb/{lang}/{encoded_db}/{encoded_db}/{urllib.parse.quote(table_id)}/"
        view_url = (
            f"/pxweb/{lang}/{encoded_db}/{encoded_db}/{urllib.parse.quote(table_id)}/"
            f"table/tableViewLayout1/"
        )
        # Step 1 — GET the variable-selection form (sets up the IIS session
        # cookie and gives us the __VIEWSTATE we need to POST).
        r = self._client.get(form_url, params={"rxid": self._rxid})
        r.raise_for_status()
        html = r.text

        def grab(name: str) -> str:
            m = re.search(rf'<input[^>]+name="{re.escape(name)}"[^>]+value="([^"]*)"', html)
            return m.group(1) if m else ""

        # Step 2 — collect every multi-select <select> + its full <option> list.
        var_selects = []
        for sec in re.finditer(
            r'<select[^>]+name="([^"]+)"[^>]+multiple.*?</select>', html, re.S
        ):
            var_selects.append((
                sec.group(1),
                re.findall(r'<option value="([^"]+)"', sec.group(0)),
            ))

        # Step 3 — build the form body.
        payload: list[tuple[str, str]] = [
            (n, grab(n)) for n in (
                "__EVENTTARGET", "__EVENTARGUMENT", "__LASTFOCUS",
                "__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION",
            )
        ]
        # Add every other hidden input so the postback validates.
        seen = {k for k, _ in payload}
        for m in re.finditer(
            r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html,
        ):
            if m.group(1) not in seen:
                payload.append((m.group(1), m.group(2)))
                seen.add(m.group(1))
        # Add every multi-select's full option list + the matching count field.
        for name, opts in var_selects:
            payload.extend((name, o) for o in opts)
            payload.append((
                name.replace("ValuesListBox", "NumberValuesSelected"),
                str(len(opts)),
            ))
        # Click the "Continue" / "View Table" button.
        payload.append((
            "ctl00$ContentPlaceHolderMain$VariableSelector1$VariableSelector1$ButtonViewTable",
            "Tiếp tục",
        ))

        body = urllib.parse.urlencode(payload, encoding="utf-8").encode("utf-8")
        r2 = self._client.post(
            form_url,
            params={"rxid": self._rxid},
            content=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        r2.raise_for_status()

        # Step 4 — fetch the rendered table view in xlsx format.
        r3 = self._client.get(
            view_url,
            params={"rxid": self._rxid, "downloadfile": DOWNLOAD_FORMAT},
        )
        r3.raise_for_status()
        ct = r3.headers.get("content-type", "")
        if "spreadsheet" not in ct and "excel" not in ct and "octet-stream" not in ct:
            raise RuntimeError(
                f"unexpected content-type for {table_id}: {ct!r} "
                f"(expected an xlsx download)"
            )
        return r3.content


# ---------------------------------------------------------------------------
# xlsx → long-format records
# ---------------------------------------------------------------------------
def xlsx_to_records(xlsx_bytes: bytes) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Convert a PX-Web xlsx export into long-format records.

    Returns ``(title, records, dimension_names)``.

    Three layouts are handled:

    * **2-D wide:** col 0 = first dim labels, header row = second dim
      labels (typically years), inner cells = numeric values. The vast
      majority of NSO matrices use this shape.
    * **3-D wide:** col 0 = section header (sparse, repeats), col 1 =
      sub-label, header row = third dim labels. Section header rows
      themselves carry no data; we propagate the most recent non-NaN
      section header as the first dimension.
    * **Single-column:** col 1 holds a single value per row, no header
      row — used for "as of <year>" snapshots and percentages. We emit
      one record per row, with the second dim collapsed to a constant
      ``"value"`` label.
    """
    df = pd.read_excel(io.BytesIO(xlsx_bytes), header=None)
    if df.empty:
        return ("", [], [])

    # Row 0 is the matrix title (e.g. "Một số chỉ tiêu chủ yếu...").
    title = str(df.iloc[0, 0]) if df.shape[0] > 0 else ""

    def _to_float(v: Any) -> float | None:
        if v is None or v != v:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s or s in ("..", ".", "-"):
                return None
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _clean(v: Any) -> str | None:
        if v is None or v != v:
            return None
        s = str(v).strip()
        return s or None

    # ----- find the header row (the first row with NaN in col 0 + values
    # in col >=1). For 3-D tables col 0 may also be NaN here while col 1
    # carries a non-numeric value -- handled below.
    header_idx = None
    for i in range(1, min(8, df.shape[0])):
        row = df.iloc[i]
        if row.iloc[0] != row.iloc[0] and row.iloc[1:].notna().any():
            header_idx = i
            break
    # Detect single-column layout (no header row, every row is "label, value").
    if header_idx is None and df.shape[1] >= 2:
        records: list[dict[str, Any]] = []
        dim_name = "Phân tổ"
        for i in range(1, df.shape[0]):
            row = df.iloc[i]
            label = _clean(row.iloc[0])
            value = _to_float(row.iloc[1])
            if label is None or value is None:
                continue
            records.append({dim_name: label, "value": value})
        return (title, records, [dim_name])

    if header_idx is None:
        return (title, [], [])

    # ----- decide between 2-D and 3-D ------
    # If col 0 of the header row is NaN AND col 1 of the header row is also
    # NaN AND the *next* row (header_idx+1) has a string in col 0 with NaN
    # values in cols 2+, this is a 3-D table.
    is_3d = False
    if df.shape[1] >= 3 and header_idx + 1 < df.shape[0]:
        h = df.iloc[header_idx]
        next_row = df.iloc[header_idx + 1]
        if (h.iloc[1] != h.iloc[1]) and h.iloc[2:].notna().any():
            # header row has nan in cols 0+1, values from col 2 onward
            if (
                next_row.iloc[0] == next_row.iloc[0]  # col 0 is a string
                and (next_row.iloc[1] == next_row.iloc[1])  # col 1 is a string
                and next_row.iloc[2:].apply(_to_float).isna().all()
            ):
                is_3d = True

    # Title parsing for nice dimension names: "X chia theo D1 và D2" /
    # "X chia theo D1, D2 và D3".
    title_dims: list[str] = []
    m3 = re.search(r"chia theo\s+(.+?),\s*(.+?)\s+và\s+(.+?)\s*$", title)
    m2 = re.search(r"chia theo\s+(.+?)\s+và\s+(.+?)\s*$", title) if not m3 else None
    if m3:
        title_dims = [m3.group(1).strip(), m3.group(2).strip(), m3.group(3).strip()]
    elif m2:
        title_dims = [m2.group(1).strip(), m2.group(2).strip()]

    records = []
    if is_3d:
        # 3-D layout: col 0 = section, col 1 = sub-label, header row[2:] = third dim.
        last_dim_labels = [_clean(v) for v in df.iloc[header_idx, 2:].tolist()]
        last_dim_labels = [v for v in last_dim_labels if v is not None]
        dim_names = (
            (title_dims[0] if len(title_dims) >= 1 else "D0"),
            (title_dims[1] if len(title_dims) >= 2 else "D1"),
            (title_dims[2] if len(title_dims) >= 3 else "D2"),
        )
        section: str | None = None
        for i in range(header_idx + 1, df.shape[0]):
            row = df.iloc[i]
            c0 = _clean(row.iloc[0])
            c1 = _clean(row.iloc[1])
            # Section-header rows carry no data; propagate.
            if c0 is not None and c1 is not None and (
                row.iloc[2:].apply(_to_float).isna().all()
            ):
                section = c0
                continue
            if c1 is None:
                continue
            sec = section if c0 is None else c0
            if sec is None:
                continue
            for col_idx, last_label in enumerate(last_dim_labels, start=2):
                v = row.iloc[col_idx] if col_idx < df.shape[1] else None
                value = _to_float(v)
                if value is None:
                    continue
                records.append({
                    dim_names[0]: sec,
                    dim_names[1]: c1,
                    dim_names[2]: last_label,
                    "value": value,
                })
        return (title, records, list(dim_names))

    # ----- 2-D layout (the common case) -----
    last_dim_labels = [
        _clean(v) for v in df.iloc[header_idx, 1:].tolist()
    ]
    last_dim_labels = [v for v in last_dim_labels if v is not None]
    dim_names = (
        (title_dims[0] if len(title_dims) >= 1 else "D0"),
        (title_dims[1] if len(title_dims) >= 2 else "D1"),
    )
    for i in range(header_idx + 1, df.shape[0]):
        row = df.iloc[i]
        first = _clean(row.iloc[0])
        if first is None:
            continue
        for col_idx, last_label in enumerate(last_dim_labels, start=1):
            v = row.iloc[col_idx] if col_idx < df.shape[1] else None
            value = _to_float(v)
            if value is None:
                continue
            records.append({
                dim_names[0]: first,
                dim_names[1]: last_label,
                "value": value,
            })

    return (title, records, list(dim_names))
