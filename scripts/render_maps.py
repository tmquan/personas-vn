"""Render every map from DATAVISUALIZATION.ipynb to PNG + HTML.

We have the figure source duplicated in
:mod:`scripts._build_datavisualization_nb` (which builds the .ipynb).
Running ``jupyter nbconvert --execute`` on that notebook fails on
Choroplethmapbox traces — kaleido v1's headless-Chrome instance
appears to lose its Mapbox-GL worker when nbconvert spawns the
kernel, returning *Error 525: Mapbox error*. The standalone Python
process used by this script doesn't have that problem (verified by
``write_image`` smoke-tests).

So we keep two paths:

* The **notebook** (DATAVISUALIZATION.ipynb) is the interactive
  artefact users open in JupyterLab. Cells render fine in the live
  kernel — the kaleido issue only affects nbconvert + write_image.
* This script reproduces the figure logic and writes static PNGs +
  interactive HTMLs to ``docs/figures/maps/`` for embedding in
  ``DATAVISUALIZATION.md``.

The two paths share helpers via :mod:`packages.viz.vietnam_geo` and
the inline figure builders below — keep them in sync if you change
either side.

Usage::

    python -m scripts.render_maps        # writes 11 PNGs + 11 HTMLs
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.common.logging import get_logger
from packages.personas.datasets.provinces import PROVINCE_TO_REGION_VI
from packages.viz.vietnam_geo import (
    HOANG_SA,
    NOTABLE_CITIES,
    NOTABLE_ISLANDS,
    TRUONG_SA,
    load_vietnam_geojson,
    normalise_province_name,
)
from scripts._nvidia_style import (
    NV_BLACK,
    NV_DARK,
    NV_DISCRETE,
    NV_FONT_FAMILY,
    NV_GREEN,
    NV_GREEN_DARK,
    NV_GREEN_SOFT,
    NV_LIGHT_GREY,
    NV_SEQUENTIAL,
    save_figure,
)

log = get_logger(__name__)

PXWEB_DIR = REPO_ROOT / "data" / "nso-gov-vn" / "raw" / "pxweb" / "vi"
OUT_DIR   = REPO_ROOT / "docs" / "figures" / "maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure the NVIDIA template + load the geojson once.
pio.templates["nvidia"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=13),
        colorway=NV_DISCRETE,
    )
)
pio.templates.default = "nvidia"

GEO = load_vietnam_geojson()


# ---------------------------------------------------------------------------
# Bilingual labelling helpers — same contract as DATAANALYSIS.ipynb
# ---------------------------------------------------------------------------
_MAX_INLINE = 48

_VI_EN_DICT: dict[str, str] = {
    "Đồng bằng sông Hồng":                  "Red River Delta",
    "Trung du và miền núi phía Bắc":        "Northern midlands & mountains",
    "Bắc Trung Bộ và Duyên hải miền Trung": "North Central & Central Coast",
    "Tây Nguyên":                           "Central Highlands",
    "Đông Nam Bộ":                          "South East",
    "Đồng bằng sông Cửu Long":              "Mekong Delta",
}


def vi_en(vi: str, en: str) -> str:
    """Compose VI / EN label, wrap to two lines if too long."""
    one = f"{vi} / {en}"
    return one if len(one) <= _MAX_INLINE else f"{vi}<br>{en}"


def tr(s) -> str:
    if s is None:
        return ""
    s = str(s)
    en = _VI_EN_DICT.get(s)
    return vi_en(s, en) if en else s


# ---------------------------------------------------------------------------
# Province-level NSO loaders
# ---------------------------------------------------------------------------
def _year_int(value):
    m = re.search(r"\d{4}", str(value))
    return int(m.group(0)) if m else None


def _province_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = c.lower()
        if "phương" in cl or "tỉnh" in cl:
            return c
    raise KeyError(f"no province column in {list(df.columns)}")


# Aggregate rows the choropleth must skip: macro-region totals (in both
# Title-case and lowercase forms NSO uses inconsistently across tables),
# national totals, and the deprecated ``Hà Tây`` province (merged into
# Hà Nội in 2008 — V14.47 still ships its pre-merger rows which would
# otherwise be plotted with stale data on the modern boundary set).
_NON_PROVINCE = {
    "Cả nước", "CẢ NƯỚC", "Tổng số", "TỔNG SỐ",
    "Đồng bằng sông Hồng", "Trung du và miền núi phía Bắc",
    "Bắc Trung Bộ và Duyên hải miền Trung",
    "Bắc Trung Bộ và duyên hải miền Trung",  # NSO casing variant (V05.04 / V14.36)
    "Tây Nguyên", "Đông Nam Bộ", "Đồng bằng sông Cửu Long",
    "Hà Tây",                                # merged into Hà Nội in 2008
}


def load_province_table(table_id: str, *,
                          indicator_filter: dict[str, str] | None = None) -> pd.DataFrame:
    """Long-format province × year × value DataFrame, NSO macro-regions stripped."""
    matches = list(PXWEB_DIR.glob(f"*{table_id}.px.parquet"))
    if not matches:
        raise FileNotFoundError(f"PX-Web matrix {table_id} not on disk")
    df = pd.read_parquet(matches[0])
    if "Năm" in df.columns:
        df = df.copy()
        df["year"] = df["Năm"].map(_year_int)
    if indicator_filter:
        for col, value in indicator_filter.items():
            if col in df.columns:
                df = df[df[col] == value]
    pcol = _province_column(df)
    df = df.rename(columns={pcol: "province"})
    df["province"] = df["province"].map(normalise_province_name)
    df = df[~df["province"].isin(_NON_PROVINCE)]
    if "year" in df.columns:
        df = df.dropna(subset=["year"])
    return df.dropna(subset=["value"])


def latest_year_per_province(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce to the most recent year per province."""
    if "year" not in df.columns:
        return df
    idx = df.groupby("province")["year"].idxmax()
    return df.loc[idx, ["province", "year", "value"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mapbox-canvas + overlay traces (no Mapbox token needed for white-bg)
# ---------------------------------------------------------------------------
# Pixel-perfect mapbox area, expressed in paper fraction. Combined with
# ``margin = 0`` this carves out a fixed rectangle on the canvas that no
# auto-margin pass can shrink or shift. Tuned for the 1100 × 900 export:
#
#   x: [0.01, 0.84]  →  pixels  11 .. 924  (913 px wide)
#   y: [0.01, 0.90]  →  pixels  90 .. 891  (801 px tall)  (PNG y, top-down)
#
# Leaves ~90 px on top for the (1- or 2-line) bilingual title, and a
# ~176 px right strip for the colorbar bar + tick labels + rotated title
# without ever encroaching on the mapbox area.
_MAPBOX_DOMAIN_X = (0.01, 0.84)
_MAPBOX_DOMAIN_Y = (0.01, 0.90)

# Fixed colorbar geometry, anchored to the *container* (the full canvas)
# rather than the plot fraction, so its position is independent of any
# margin auto-adjustment. Tick-label width may still vary between figures
# (e.g. ``20`` vs ``10k``) but only the labels' right edge moves — the
# bar itself, and crucially the mapbox to its left, stay put.
_COLORBAR_KW = dict(
    xref="container", x=0.87, xanchor="left",
    yref="container", y=0.50, yanchor="middle",
    len=0.62, lenmode="fraction",
    thickness=18,
)


def vietnam_map_layout(*, title: str) -> dict:
    """Common layout for all 11 maps.

    Vietnam must sit at IDENTICAL pixel coordinates on every figure so a
    reader scrolling through ``docs/figures/maps/`` never sees the
    country "jump" between exports. Three layout primitives, used
    together, make this strict rather than approximate:

    * ``margin = 0`` on every side — eliminates the auto-margin
      mechanism's ability to shift the plot area when a colorbar's
      tick labels or title happen to be wider than the reserved strip.
    * ``mapbox.domain`` — explicitly carves a fixed rectangle on the
      canvas in paper-fraction coordinates (see ``_MAPBOX_DOMAIN_*``).
      Plotly draws the mapbox tile + choropleth + overlays *inside*
      this rectangle; nothing else (title, colorbar) can push it.
    * ``mapbox.center`` and ``zoom`` — constant across all figures.
      Combined with the fixed pixel rectangle from ``domain``, the
      visible lon/lat extent is bit-for-bit identical between figures.

    Title is positioned with explicit ``y`` + ``yanchor='top'`` and
    ``automargin=False`` so a 1-line vs 2-line title doesn't shift any
    other element. Two-line titles still fit in the ~90 px top strip
    above the mapbox rectangle.
    """
    return dict(
        title=dict(text=title,
                    font=dict(family=NV_FONT_FAMILY, color=NV_DARK, size=18),
                    x=0.02, xanchor="left",
                    y=0.97, yanchor="top",
                    automargin=False, pad=dict(l=0, r=0, t=0, b=0)),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        mapbox=dict(style="white-bg",
                     center=dict(lon=109.5, lat=14.8),
                     zoom=4.4,
                     domain=dict(x=list(_MAPBOX_DOMAIN_X),
                                  y=list(_MAPBOX_DOMAIN_Y))),
    )


def _archipelago_outline_trace(meta: dict) -> go.Scattermapbox:
    """Dashed-look polygon for Hoàng Sa / Trường Sa.

    Scattermapbox doesn't support `dash`, so we approximate by emitting
    every other segment — visually identical at country scale.
    """
    poly = meta["polygon"]
    lons, lats = [], []
    n_segs = 32
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]; x1, y1 = poly[i + 1]
        for k in range(n_segs):
            if k % 2 == 0:
                t0, t1 = k / n_segs, (k + 1) / n_segs
                lons.extend([x0 + (x1 - x0) * t0, x0 + (x1 - x0) * t1, None])
                lats.extend([y0 + (y1 - y0) * t0, y0 + (y1 - y0) * t1, None])
    return go.Scattermapbox(
        lon=lons, lat=lats, mode="lines",
        line=dict(color=NV_GREEN_DARK, width=2),
        hoverinfo="skip", showlegend=False,
    )


def _archipelago_label_trace(meta: dict) -> go.Scattermapbox:
    """Bilingual label anchored just BELOW the archipelago bounding box.

    Three Mapbox-GL constraints handled here:

    * ``mode='text'`` (alone) crashes with ``Mapbox error 525`` —
      Mapbox-GL's text engine refuses to render text without a
      symbol-layer host. Workaround: ``mode='markers+text'`` with an
      invisible 0-size marker.
    * ``textfont.family`` also crashes the same way because
      ``mapbox.style='white-bg'`` ships *no* glyphs URL — any custom
      font triggers a Mapbox-GL glyph fetch, which fails. We omit
      ``family`` so Mapbox-GL's default Open Sans is used; size and
      colour still work. Vietnamese diacritics render correctly.
    * Mapbox-GL's text engine does **not** parse HTML / pseudo-HTML —
      ``<b>``, ``<i>``, ``<br>`` etc. show up literally. We emit plain
      text and use ``textfont.size`` for emphasis instead.

    The label sits ~0.4° below the box's bottom edge with
    ``textposition='bottom center'`` so the text reads cleanly
    *under* the dashed bounding box.
    """
    return go.Scattermapbox(
        lon=[meta["centre"][0]],
        # 0.50° below the box — sits just under the dashed bottom edge
        # with a small gap, no longer taking up extra vertical space.
        lat=[meta["lat_min"] - 0.15],
        mode="markers+text",
        # Vietnamese-only label (English name lives in hover tooltips
        # only, to keep the static map uncluttered for screenshot use).
        text=[meta["name_vi"]],
        textfont=dict(color=NV_BLACK, size=12),
        textposition="bottom center",
        marker=dict(size=1, color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False,
    )


def _archipelago_islands_trace(islands: list[dict]) -> go.Scattermapbox:
    return go.Scattermapbox(
        lon=[i["lon"] for i in islands],
        lat=[i["lat"] for i in islands],
        mode="markers",
        marker=dict(size=8, color=NV_GREEN_DARK),
        text=[f"{i['name_vi']} / {i['name_en']}" for i in islands],
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    )


def _leader_line_traces(
    items: list[dict],
    *,
    marker_size: int = 11,
    marker_color: str = NV_GREEN,
    text_size: int = 11,
    text_color: str = NV_DARK,
    line_color: str = "rgba(0,0,0,0.45)",
    line_width: float = 0.7,
    hover_extras: list[str] | None = None,
) -> list[go.Scattermapbox]:
    """Render each item as: marker at (lon, lat) + thin leader line +
    label at (label_lon, label_lat). Standard cartographic offset-label
    technique used to avoid label-on-polygon clutter.

    Three Scattermapbox traces emitted per item (3 × N), packed into a
    list. The leader line is drawn first (under everything), then
    markers (clickable), then labels (top-most). Items are expected
    to carry ``lon``, ``lat``, ``name_vi`` plus optional ``label_lon``,
    ``label_lat`` (defaults to the marker's lon/lat — i.e. no leader
    line / label-on-marker rendering).

    ``hover_extras`` is a parallel list of pre-formatted HTML strings,
    one per item, to be rendered as the marker's hover tooltip. If
    omitted, the marker shows just the bilingual name.
    """
    n = len(items)
    if not n:
        return []

    # Each leader is rendered as a **CAD-style elbow callout**: from
    # the marker, a strictly *vertical* segment to the elbow corner,
    # then a strictly *horizontal* segment to the label position.
    # Both segments are axis-aligned, the corner is exactly 90°, and
    # the label hangs off the END of the horizontal segment (never on
    # top of it). This matches GD&T / cartographic dimension-callout
    # convention and reads naturally off a flat baseline.
    #
    # Three special cases are handled cleanly:
    #
    # * ``label == marker`` (offset < ~5 km): no leader, label drawn
    #   centred on the marker.
    # * ``label_lon == marker_lon`` (label directly above/below): a
    #   single vertical segment, no elbow.
    # * ``label_lat == marker_lat`` (label directly to the side):
    #   a single horizontal segment, no elbow.
    #
    # In every non-trivial case the line ENDS with a horizontal
    # approach, so labels are split into two text traces by
    # direction (right-of-bend → ``textposition='middle right'``,
    # left-of-bend → ``'middle left'``), keeping the text always
    # hanging off the line's endpoint rather than overlapping it.
    line_lons:   list[float | None] = []
    line_lats:   list[float | None] = []
    marker_lons: list[float] = []
    marker_lats: list[float] = []
    text_right_lons: list[float] = []; text_right_lats: list[float] = []
    text_right_str:  list[str]   = []
    text_left_lons:  list[float] = []; text_left_lats:  list[float] = []
    text_left_str:   list[str]   = []
    text_center_lons: list[float] = []; text_center_lats: list[float] = []
    text_center_str:  list[str]   = []
    hover_lines: list[str] = []

    for idx, it in enumerate(items):
        mlon, mlat = it["lon"], it["lat"]
        llon = it.get("label_lon", mlon)
        llat = it.get("label_lat", mlat)
        marker_lons.append(mlon); marker_lats.append(mlat)
        hover_lines.append((hover_extras[idx] if hover_extras
                              else f"<b>{it['name_vi']}</b>")
                            + "<extra></extra>")

        dlon = llon - mlon
        dlat = llat - mlat

        if abs(dlon) < 0.05 and abs(dlat) < 0.05:
            # No offset — label sits on top of marker.
            text_center_lons.append(llon)
            text_center_lats.append(llat)
            text_center_str.append(it["name_vi"])
            continue

        if abs(dlon) < 0.05:
            # Pure vertical: marker → label (no elbow). Label centred
            # vertically, no left/right preference.
            line_lons.extend([mlon, llon, None])
            line_lats.extend([mlat, llat, None])
            text_center_lons.append(llon)
            text_center_lats.append(llat)
            text_center_str.append(it["name_vi"])
        elif abs(dlat) < 0.05:
            # Pure horizontal: marker → label (no elbow).
            line_lons.extend([mlon, llon, None])
            line_lats.extend([mlat, llat, None])
            if dlon > 0:
                text_right_lons.append(llon); text_right_lats.append(llat)
                text_right_str.append(it["name_vi"])
            else:
                text_left_lons.append(llon); text_left_lats.append(llat)
                text_left_str.append(it["name_vi"])
        else:
            # Full elbow: vertical from marker to (mlon, llat),
            # then horizontal to (llon, llat). Both segments
            # strictly axis-aligned; corner at 90°.
            line_lons.extend([mlon, mlon, llon, None])
            line_lats.extend([mlat, llat, llat, None])
            if dlon > 0:
                text_right_lons.append(llon); text_right_lats.append(llat)
                text_right_str.append(it["name_vi"])
            else:
                text_left_lons.append(llon); text_left_lats.append(llat)
                text_left_str.append(it["name_vi"])

    def _text_trace(lons, lats, txts, position):
        return go.Scattermapbox(
            lon=lons, lat=lats, mode="markers+text",
            text=txts,
            textfont=dict(color=text_color, size=text_size),
            textposition=position,
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        )

    out: list[go.Scattermapbox] = []
    if line_lons:
        out.append(go.Scattermapbox(
            lon=line_lons, lat=line_lats, mode="lines",
            line=dict(color=line_color, width=line_width),
            hoverinfo="skip", showlegend=False,
        ))
    out.append(go.Scattermapbox(
        lon=marker_lons, lat=marker_lats, mode="markers",
        marker=dict(size=marker_size, color=marker_color),
        hovertemplate=hover_lines, showlegend=False,
    ))
    if text_right_str:
        out.append(_text_trace(text_right_lons, text_right_lats,
                                text_right_str, "middle right"))
    if text_left_str:
        out.append(_text_trace(text_left_lons, text_left_lats,
                                text_left_str, "middle left"))
    if text_center_str:
        out.append(_text_trace(text_center_lons, text_center_lats,
                                text_center_str, "middle center"))
    return out


def _notable_islands_traces() -> list[go.Scattermapbox]:
    """Markers + bilingual labels for the 7 major named Vietnamese islands,
    each with a leader line into open sea.
    """
    hover = [
        f"<b>{i['name_vi']} / {i['name_en']}</b><br>"
        f"{i['note']}<br>"
        f"({i['lon']:.2f}°E, {i['lat']:.2f}°N)"
        for i in NOTABLE_ISLANDS
    ]
    return _leader_line_traces(NOTABLE_ISLANDS,
                                  marker_color=NV_GREEN,
                                  text_color=NV_DARK,
                                  hover_extras=hover)


def _notable_islands_trace() -> go.Scattermapbox:
    """Backwards-compatible single-trace getter (returns first only).

    Prefer :func:`_notable_islands_traces` which returns the full
    multi-trace list.
    """
    traces = _notable_islands_traces()
    return traces[0] if traces else go.Scattermapbox()


# ---------------------------------------------------------------------------
# Cities + capital
# ---------------------------------------------------------------------------
# Per-city textposition tuned so labels never collide with each other,
# the coastline, or the notable-island labels. Eyeballed at zoom 4.4.
def _cities_traces() -> list[go.Scattermapbox]:
    """Markers + bilingual labels for the national capital + 10 major cities,
    each with a leader line into open sea / off-Vietnam region.

    The **capital** (Hà Nội) gets a star-prefixed bigger label and an
    extra-large marker so it visually pops; the 10 secondary cities
    share the standard leader-line treatment. Each marker's hover
    tooltip carries the English name, the city's "role" descriptor, and
    its precise lat/lon.
    """
    out: list[go.Scattermapbox] = []

    capitals = [c for c in NOTABLE_CITIES if c.get("capital")]
    others   = [c for c in NOTABLE_CITIES if not c.get("capital")]

    # Capital — one entry, prefixed with a star, larger marker.
    if capitals:
        cap_items = [
            {**c, "name_vi": f"★ {c['name_vi']}"}    # star prefix
            for c in capitals
        ]
        cap_hover = [
            f"<b>{c['name_vi']} / {c['name_en']}</b><br>"
            f"{c['role']}<br>"
            f"({c['lon']:.2f}°E, {c['lat']:.2f}°N)"
            for c in capitals
        ]
        out.extend(_leader_line_traces(
            cap_items,
            marker_size=15, marker_color=NV_BLACK,
            text_size=13,   text_color=NV_BLACK,
            hover_extras=cap_hover,
        ))

    # Other major cities — same leader-line treatment, smaller markers.
    if others:
        other_hover = [
            f"<b>{c['name_vi']} / {c['name_en']}</b><br>"
            f"{c['role']}<br>"
            f"({c['lon']:.2f}°E, {c['lat']:.2f}°N)"
            for c in others
        ]
        out.extend(_leader_line_traces(
            others,
            marker_size=10, marker_color=NV_BLACK,
            text_size=11,   text_color=NV_BLACK,
            hover_extras=other_hover,
        ))

    return out


def add_overlays(fig: go.Figure) -> go.Figure:
    """Append the cartographic overlay stack in the right order.

    Order matters because Mapbox-GL applies symbol-collision pruning
    in trace-order (later traces win): we want **archipelago labels**
    to never be hidden, so they're added *last*.

    Stack (bottom → top):

        1. Archipelago dashed-outline polygons + their principal-island
           markers (Phú Lâm, Trường Sa Lớn, etc.)
        2. The 7 named Vietnamese islands (Phú Quốc, Cát Bà, ...) with
           leader lines + bilingual labels.
        3. The capital + 10 major cities, with leader lines + labels.
        4. Archipelago bilingual labels (last so Mapbox-GL won't prune
           them when they happen to overlap a city's offset label).
    """
    for meta in (HOANG_SA, TRUONG_SA):
        fig.add_trace(_archipelago_outline_trace(meta))
        fig.add_trace(_archipelago_islands_trace(meta["islands"]))
    for t in _notable_islands_traces():
        fig.add_trace(t)
    for t in _cities_traces():
        fig.add_trace(t)
    # Archipelago labels go LAST in the trace order so Mapbox-GL's
    # symbol-collision pruning never hides them in favour of an
    # earlier city label.
    for meta in (HOANG_SA, TRUONG_SA):
        fig.add_trace(_archipelago_label_trace(meta))
    return fig


# All 64 admin-1 polygon names (63 provinces + Côn Đảo) — canonical
# spellings as they appear in ``GEO["properties"].shapeName``. Used as
# the locations list for the always-on base outline trace below.
_ALL_PROVINCE_NAMES: list[str] = [
    f["properties"]["shapeName"] for f in GEO["features"]
    if not f["properties"].get("is_archipelago")
]


def _base_outline_trace(*, line_color: str = NV_BLACK,
                          line_width: float = 0.9) -> go.Choroplethmapbox:
    """Transparent-fill choropleth covering every admin-1 polygon.

    Drawn UNDER the data trace inside :func:`build_choropleth` so that
    every province carries a visible border even when the upstream NSO
    table has no row for it (e.g. Côn Đảo is missing from every
    table; tables V03.12 and V03.22 use Latin-Eth ``Ð`` so their
    province names fail to normalise and get silently dropped). Using
    a stable canonical name list from the GeoJSON ensures the base
    trace is always 64-wide regardless of upstream data quality.
    """
    return go.Choroplethmapbox(
        geojson=GEO,
        locations=_ALL_PROVINCE_NAMES,
        z=[0] * len(_ALL_PROVINCE_NAMES),
        featureidkey="properties.shapeName",
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        marker=dict(line=dict(color=line_color, width=line_width),
                     opacity=1.0),
        showscale=False,
        hoverinfo="skip",
    )


def build_choropleth(values: pd.DataFrame, *,
                       title: str,
                       colorbar_title: str,
                       hover_unit: str = "",
                       colorscale=NV_SEQUENTIAL) -> go.Figure:
    # Approach (A) — pre-emptive base outline. We add a transparent-fill
    # base choropleth covering all 64 admin-1 polygons FIRST, then the
    # data trace on top. This guarantees every province has a visible
    # border regardless of whether the data trace has a row for it.
    # Approach (B) — reindexing ``values`` to all 64 names with NaN
    # for missing — was rejected because Plotly's Choroplethmapbox
    # silently skips polygons whose ``z`` is NaN: no fill AND no
    # border, defeating the purpose. The data trace below keeps its
    # own border (same colour & width as the base) so that for
    # provinces *with* data the border draws on top of the 0.85-opacity
    # fill at full crispness; the two stacked borders share the exact
    # same path so they read as a single line, not double-thickness.
    fig = go.Figure()
    fig.add_trace(_base_outline_trace())
    fig.add_trace(go.Choroplethmapbox(
        geojson=GEO,
        locations=values["province"],
        z=values["value"],
        featureidkey="properties.shapeName",
        colorscale=colorscale,
        marker=dict(line=dict(color=NV_BLACK, width=0.9), opacity=0.85),
        colorbar=dict(
            title=dict(text=colorbar_title,
                        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12),
                        side="right"),
            tickfont=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=11),
            outlinecolor=NV_LIGHT_GREY, outlinewidth=0.5,
            **_COLORBAR_KW,
        ),
        hovertemplate=("<b>%{location}</b><br>%{z:,.2f}"
                        + (" " + hover_unit if hover_unit else "")
                        + "<extra></extra>"),
    ))
    add_overlays(fig)
    fig.update_layout(**vietnam_map_layout(title=title))
    return fig


def _province_centroid(name: str) -> tuple[float, float] | None:
    for f in GEO["features"]:
        if f["properties"]["shapeName"] != name:
            continue
        g = f["geometry"]
        if g["type"] == "MultiPolygon":
            ring = max(g["coordinates"], key=lambda p: len(p[0]))[0]
        else:
            ring = g["coordinates"][0]
        lons = [c[0] for c in ring]; lats = [c[1] for c in ring]
        return (sum(lons) / len(lons), sum(lats) / len(lats))
    return None


# ---------------------------------------------------------------------------
# Per-figure render functions — mirror the notebook cells 1:1
# ---------------------------------------------------------------------------
# `scale=2` for the analysis charts gives crisp small text but pushes
# Mapbox-GL beyond a comfortable render budget for choropleth maps with
# ~70 features + multiple Scattermapbox overlays. We export maps at
# scale=1 (the canvas is already 1100×900, plenty for a markdown embed).
_MAP_SCALE = 1


def _save(fig, name: str, **kw) -> None:
    save_figure(fig, name, scale=_MAP_SCALE, out_dir=OUT_DIR, **kw)


def fig01_country_reference() -> None:
    log.info("M01 — country reference map")
    prov_names = [f["properties"]["shapeName"] for f in GEO["features"]
                  if not f["properties"].get("is_archipelago")]
    fig = go.Figure(go.Choroplethmapbox(
        geojson=GEO,
        locations=prov_names, z=[1] * len(prov_names),
        featureidkey="properties.shapeName",
        colorscale=[[0, NV_GREEN_SOFT], [1, NV_GREEN_SOFT]],
        # Bigger 1.0 px border on the reference map because the fill is
        # uniform — without crisp internal borders the country reads
        # as a single shape rather than 63 admin-1 units.
        marker=dict(line=dict(color=NV_BLACK, width=1.0), opacity=0.85),
        showscale=False,
        hovertemplate="<b>%{location}</b><extra></extra>",
    ))
    add_overlays(fig)
    fig.update_layout(**vietnam_map_layout(
        title=vi_en("Việt Nam — bản đồ tham chiếu",
                     "Vietnam — reference map"),
    ))
    _save(fig, "01_country_reference", width=1100, height=700)


def fig02_population() -> None:
    log.info("M02 — population by province")
    df = load_province_table("V02.01",
                                indicator_filter={"Chỉ tiêu":
                                    "Dân số trung bình (Nghìn người)"})
    latest = latest_year_per_province(df)
    fig = build_choropleth(
        latest,
        title=vi_en(
            f"Dân số bình quân theo tỉnh — {int(latest['year'].max())}",
            f"Average population by province — {int(latest['year'].max())}") +
            " (NSO V02.01)",
        colorbar_title=vi_en("Nghìn người", "Thousand persons"),
        hover_unit="nghìn người",
    )
    _save(fig, "02_population_by_province", width=1100, height=700)


def fig03_grdp() -> None:
    log.info("M03 — GRDP per capita")
    df = load_province_table("V03.12")
    latest = latest_year_per_province(df)
    fig = build_choropleth(
        latest,
        title=vi_en(
            f"GRDP bình quân đầu người theo tỉnh — {int(latest['year'].max())}",
            f"GRDP per capita by province — {int(latest['year'].max())}") +
            " (NSO V03.12)",
        colorbar_title=vi_en("Triệu VND/người", "Million VND/person"),
        hover_unit="triệu VND/người",
    )
    _save(fig, "03_grdp_per_capita", width=1100, height=700)


def fig04_iip() -> None:
    log.info("M04 — IIP by province")
    df = load_province_table("V07.02")
    latest = latest_year_per_province(df)
    fig = build_choropleth(
        latest,
        title=vi_en(
            f"Chỉ số sản xuất công nghiệp theo tỉnh — {int(latest['year'].max())}",
            f"Industrial Production Index by province — {int(latest['year'].max())}") +
            " (NSO V07.02, " + vi_en("năm trước = 100", "prev year = 100") + ")",
        colorbar_title="IIP", hover_unit="IIP",
    )
    _save(fig, "04_iip_by_province", width=1100, height=700)


def fig05_tourism() -> None:
    log.info("M05 — tourism revenue by province")
    df = load_province_table("V10.03")
    latest = latest_year_per_province(df)
    fig = build_choropleth(
        latest,
        title=vi_en(
            f"Doanh thu du lịch lữ hành theo tỉnh — {int(latest['year'].max())}",
            f"Tourism (organised-travel) revenue by province — {int(latest['year'].max())}") +
            " (NSO V10.03)",
        colorbar_title=vi_en("Tỷ VND", "Billion VND"),
        hover_unit="tỷ VND",
    )
    _save(fig, "05_tourism_revenue", width=1100, height=700)


def fig06_income() -> None:
    log.info("M06 — household income per capita")
    df = load_province_table("V14.36")
    latest = latest_year_per_province(df)
    fig = build_choropleth(
        latest,
        title=vi_en(
            f"Thu nhập bình quân đầu người tháng — {int(latest['year'].max())}",
            f"Monthly income per capita by province — {int(latest['year'].max())}") +
            " (NSO V14.36)",
        colorbar_title=vi_en("Nghìn VND/tháng", "Thousand VND/month"),
        hover_unit="nghìn VND/tháng",
    )
    _save(fig, "06_income_per_capita", width=1100, height=700)


def fig07_poverty() -> None:
    log.info("M07 — poverty rate by province")
    df = load_province_table("V14.47")
    latest = latest_year_per_province(df)
    fig = build_choropleth(
        latest,
        title=vi_en(
            f"Tỷ lệ hộ nghèo theo tỉnh — {int(latest['year'].max())}",
            f"Multi-dimensional poverty rate by province — {int(latest['year'].max())}") +
            " (NSO V14.47, %)",
        colorbar_title=vi_en("Tỷ lệ hộ nghèo (%)", "Poverty rate (%)"),
        hover_unit="%",
    )
    _save(fig, "07_poverty_rate", width=1100, height=700)


def fig08_enterprises() -> None:
    log.info("M08 — enterprises by province")
    df = load_province_table("V05.04")
    latest = latest_year_per_province(df)
    fig = build_choropleth(
        latest,
        title=vi_en(
            f"Số doanh nghiệp đang hoạt động theo tỉnh — {int(latest['year'].max())}",
            f"Active enterprises by province — {int(latest['year'].max())}") +
            " (NSO V05.04)",
        colorbar_title=vi_en("Doanh nghiệp", "Enterprises"),
        hover_unit="doanh nghiệp",
    )
    _save(fig, "08_enterprises_by_province", width=1100, height=700)


def fig09_bhxh() -> None:
    log.info("M09 — social insurance participation")
    df = load_province_table("V03.22")
    latest = latest_year_per_province(df)
    fig = build_choropleth(
        latest,
        title=vi_en(
            f"Tỷ lệ tham gia BHXH theo tỉnh — {int(latest['year'].max())}",
            f"Social insurance participation by province — {int(latest['year'].max())}") +
            " (NSO V03.22, %)",
        colorbar_title=vi_en("Tỷ lệ tham gia (%)", "Participation rate (%)"),
        hover_unit="%",
    )
    _save(fig, "09_bhxh_participation", width=1100, height=700)


def fig10_bubble() -> None:
    log.info("M10 — enterprises × IIP bubble map")
    ents = latest_year_per_province(load_province_table("V05.04"))
    iip  = latest_year_per_province(load_province_table("V07.02"))
    merged = (ents[["province", "value"]].rename(columns={"value": "enterprises"})
                .merge(iip[["province", "value"]].rename(columns={"value": "iip"}),
                        on="province"))
    merged[["lon", "lat"]] = merged["province"].apply(
        lambda p: pd.Series(_province_centroid(p) or (np.nan, np.nan)))
    merged = merged.dropna(subset=["lon", "lat"])

    prov_names = [f["properties"]["shapeName"] for f in GEO["features"]
                  if not f["properties"].get("is_archipelago")]
    fig = go.Figure()
    fig.add_trace(go.Choroplethmapbox(
        geojson=GEO, locations=prov_names, z=[1] * len(prov_names),
        featureidkey="properties.shapeName",
        colorscale=[[0, "#F0F4ED"], [1, "#F0F4ED"]], showscale=False,
        # Pale grey 0.7 px borders — visible enough to delineate the 63
        # admin-1 units behind the bubbles without competing with them.
        marker=dict(line=dict(color=NV_LIGHT_GREY, width=0.7), opacity=0.55),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scattermapbox(
        lon=merged["lon"], lat=merged["lat"], mode="markers",
        marker=dict(
            size=np.sqrt(merged["enterprises"].clip(lower=1)) * 0.7,
            sizemode="diameter", sizemin=5,
            color=merged["iip"], colorscale=NV_SEQUENTIAL,
            cmin=merged["iip"].quantile(0.05),
            cmax=merged["iip"].quantile(0.95),
            showscale=True,
            colorbar=dict(
                title=dict(text=vi_en("Chỉ số IIP", "IIP"),
                            font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12),
                            side="right"),
                tickfont=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=11),
                outlinecolor=NV_LIGHT_GREY, outlinewidth=0.5,
                **_COLORBAR_KW,
            ),
        ),
        hovertemplate=("<b>%{customdata[0]}</b><br>" +
                        vi_en("Doanh nghiệp", "Enterprises") +
                        ": %{customdata[1]:,.0f}<br>" +
                        vi_en("IIP", "IIP") + ": %{customdata[2]:.1f}<extra></extra>"),
        customdata=merged[["province", "enterprises", "iip"]].values,
        showlegend=False,
    ))
    add_overlays(fig)
    fig.update_layout(**vietnam_map_layout(
        title=vi_en("Doanh nghiệp × Tăng trưởng IIP theo tỉnh",
                     "Enterprises × IIP growth by province") +
                " (size = enterprises, NSO V05.04 + V07.02)",
    ))
    _save(fig, "10_enterprises_iip_bubble", width=1100, height=700)


def fig11_macro_regions() -> None:
    log.info("M11 — macro-region overlay")
    REGION_PALETTE = {
        "Đồng bằng sông Hồng":                  "#76B900",
        "Trung du và miền núi phía Bắc":        "#AED581",
        "Bắc Trung Bộ và Duyên hải miền Trung": "#5C9300",
        "Tây Nguyên":                           "#94BD51",
        "Đông Nam Bộ":                          "#3F6F00",
        "Đồng bằng sông Cửu Long":              "#C5DDA0",
    }
    region_idx = {r: i for i, r in enumerate(REGION_PALETTE)}
    provs, zs = [], []
    for f in GEO["features"]:
        if f["properties"].get("is_archipelago"):
            continue
        name = f["properties"]["shapeName"]
        region = PROVINCE_TO_REGION_VI.get(name)
        provs.append(name)
        zs.append(region_idx[region] if region is not None else -1)

    n = len(REGION_PALETTE)
    cs = []
    for i, color in enumerate(REGION_PALETTE.values()):
        cs.append([i / n,       color])
        cs.append([(i + 1) / n, color])

    fig = go.Figure(go.Choroplethmapbox(
        geojson=GEO, locations=provs, z=zs,
        featureidkey="properties.shapeName",
        colorscale=cs, zmin=-1, zmax=n - 1,
        # 1.0 px border so adjacent same-region provinces stay visually
        # distinguishable inside their colour band.
        marker=dict(line=dict(color=NV_BLACK, width=1.0), opacity=0.85),
        showscale=False,
        hovertemplate="<b>%{location}</b><extra></extra>",
    ))
    for region in REGION_PALETTE:
        region_provs = [p for p, r in PROVINCE_TO_REGION_VI.items() if r == region]
        centres = [_province_centroid(p) for p in region_provs]
        centres = [c for c in centres if c is not None]
        if not centres:
            continue
        cx = sum(c[0] for c in centres) / len(centres)
        cy = sum(c[1] for c in centres) / len(centres)
        fig.add_trace(go.Scattermapbox(
            lon=[cx], lat=[cy], mode="markers+text",
            text=[tr(region)],
            textfont=dict(color=NV_BLACK, size=12),
            textposition="middle center",
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))
    add_overlays(fig)
    fig.update_layout(**vietnam_map_layout(
        title=vi_en("6 vùng kinh tế của Việt Nam",
                     "Six macro-economic regions of Vietnam"),
    ))
    _save(fig, "11_macro_regions", width=1100, height=700)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
ALL_FIGS = (
    fig01_country_reference,
    fig02_population,
    fig03_grdp,
    fig04_iip,
    fig05_tourism,
    fig06_income,
    fig07_poverty,
    fig08_enterprises,
    fig09_bhxh,
    fig10_bubble,
    fig11_macro_regions,
)


def main() -> None:
    for fn in ALL_FIGS:
        try:
            fn()
        except Exception as e:
            log.error("  failed: %s — %s: %s", fn.__name__, type(e).__name__, e)
    n = len(list(OUT_DIR.glob("*.png")))
    log.info("done — %d PNGs (+ HTML siblings) in %s/", n,
              OUT_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
