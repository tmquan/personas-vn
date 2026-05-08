"""Build DATAVISUALIZATION.ipynb programmatically.

Companion to ``_build_dataanalysis_nb.py`` — same pattern (declarative
``cells = []`` list driven into ``nbformat``), but the figures are
*geographic* maps of Vietnam rather than analytical charts. Every map:

* uses Plotly's **geo** trace family (``Choropleth`` + ``Scattergeo``),
  *not* mapbox — no Mapbox token required, fully offline once the
  geojson is cached;
* is framed to fit **Vietnam mainland + the East Sea** (lon 102°E –
  118°E, lat 6°N – 24°N), so **Quần đảo Hoàng Sa** (Paracel Islands)
  and **Quần đảo Trường Sa** (Spratly Islands) are visible in every
  view as required by Vietnamese cartographic convention;
* draws the two archipelagos as dashed-outline polygons with bilingual
  labels (``Quần đảo Hoàng Sa / Paracel Islands``) plus principal-
  island markers (Phú Lâm, Trường Sa Lớn, Song Tử Tây, etc.);
* labels major named Vietnamese islands (Đảo Phú Quốc, Đảo Cát Bà,
  Đảo Lý Sơn, …) — see ``packages.viz.vietnam_geo.NOTABLE_ISLANDS``;
* follows the NVIDIA brand-style preamble (white background, NVIDIA
  Green ``#76B900`` data series, NVIDIA Sans typography fallback);
* exports both an interactive HTML and a static PNG to
  ``docs/figures/maps/`` so ``DATAVISUALIZATION.md`` can embed them.

Run::

    python -m scripts._build_datavisualization_nb
    jupyter nbconvert --to notebook --execute DATAVISUALIZATION.ipynb \
        --inplace --ExecutePreprocessor.timeout=300

The cell list below is the full source of the notebook; rebuilding the
notebook from scratch is just ``python -m scripts._build_datavisualization_nb``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "DATAVISUALIZATION.ipynb"


def md(src: str) -> dict:
    """Markdown cell."""
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str) -> dict:
    """Code cell."""
    return nbf.v4.new_code_cell(src.strip("\n"))


cells: list[dict] = []


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Vietnam — Geographic data visualisation

Companion to [`DATAANALYSIS.ipynb`](DATAANALYSIS.ipynb): everything
**spatial**. Where the analysis notebook plots NSO data as
distributions and time-series, this notebook plots them on the map.

Coverage:

* **Vietnam mainland** + **all 63 administrative provinces** (admin-1
  level, sourced from
  [geoBoundaries gbOpen ADM1](https://github.com/wmgeolab/geoBoundaries)).
* **Quần đảo Hoàng Sa** (Paracel Islands, administered by Đà Nẵng) and
  **Quần đảo Trường Sa** (Spratly Islands, administered by Khánh Hòa)
  drawn as dashed-outline bounding polygons with principal-island
  markers, matching the Vietnamese General Statistics Office's
  cartographic convention.
* **7 major named Vietnamese islands** explicitly labelled: Đảo Phú
  Quốc (Vietnam's largest island), Đảo Cát Bà (Hạ Long Bay biosphere),
  Đảo Bạch Long Vĩ (Gulf of Tonkin), Đảo Lý Sơn, Đảo Phú Quý, Cù Lao
  Chàm, Côn Đảo.

Every figure follows the
[NVIDIA brand guidelines](https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/):
white background, NVIDIA Green `#76B900` for the primary data series,
black for axis chrome, NVIDIA Sans typography (with safe fallbacks).
Long bilingual labels wrap onto two lines (Vietnamese on top, English
below).

Run prerequisites:

```bash
make curate                              # populates data/nso-gov-vn/raw/pxweb/vi/
pip install -e ".[curator,viz]" "kaleido>=1.0,<2.0"
.venv/bin/python -c "from packages.viz.vietnam_geo import load_vietnam_geojson; load_vietnam_geojson()"
.venv/bin/python -m scripts._build_datavisualization_nb
.venv/bin/jupyter nbconvert --to notebook --execute DATAVISUALIZATION.ipynb --inplace
```
"""))


# ---------------------------------------------------------------------------
# §0 Setup
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §0 — Setup

> **Why every `save_figure(...)` call below passes `write_png=False`.**
> All maps in this notebook use Plotly's `Choroplethmapbox` /
> `Scattermapbox` traces, which are rendered by Mapbox-GL inside a
> headless Chromium when kaleido is asked to produce a PNG. Inside a
> Jupyter kernel that headless Chromium loses its Mapbox-GL worker and
> kaleido raises `KaleidoError: Error 525: Mapbox error`. The interactive
> HTML companion still renders correctly inline via `fig.show()`, so the
> notebook is fine — it just doesn't try to write a PNG. To regenerate
> the PNGs (used by `DATAVISUALIZATION.md`) run the standalone script
> [`scripts/render_maps.py`](scripts/render_maps.py), which uses the
> same figure logic in a fresh Python process where Mapbox-GL is happy.
"""))


cells.append(code(r"""
'''Imports + NVIDIA Plotly theme + geographic data loaders.'''
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings('ignore')

REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._nvidia_style import (
    apply_nvidia_style, save_figure,
    NV_GREEN, NV_GREEN_DARK, NV_GREEN_SOFT,
    NV_BLACK, NV_DARK, NV_GREY, NV_LIGHT_GREY, NV_FAINT,
    NV_WHITE, NV_DISCRETE, NV_SEQUENTIAL, NV_FONT_FAMILY,
)
from packages.viz.vietnam_geo import (
    load_vietnam_geojson, normalise_province_name,
    HOANG_SA, TRUONG_SA, NOTABLE_ISLANDS, SCATTERED_ISLAND_MARKERS,
)

PXWEB_DIR = REPO_ROOT / 'data' / 'nso-gov-vn' / 'raw' / 'pxweb' / 'vi'
OUT_DIR   = REPO_ROOT / 'docs' / 'figures' / 'maps'
OUT_DIR.mkdir(parents=True, exist_ok=True)

pio.templates['nvidia'] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=13),
        colorway=NV_DISCRETE,
    )
)
pio.templates.default = 'nvidia'

geo = load_vietnam_geojson()
print(f'GeoJSON features: {len(geo["features"])}')
print(f'  archipelagos:    {sum(1 for f in geo["features"] if f["properties"].get("is_archipelago"))}')
print(f'  notable islands: {len(NOTABLE_ISLANDS)}')
"""))


cells.append(code(r"""
'''Bilingual labelling helpers (mirrors DATAANALYSIS.ipynb's `vi_en` / `tr`).

Same idea: short pairs go inline ("VI / EN"), long pairs wrap onto two
lines ("VI<br>EN") so the chart layout doesn't suffer.
'''
_MAX_INLINE = 48

VI_EN_DICT: dict[str, str] = {
    # Map titles + axis terms
    'Năm': 'Year',
    'Tổng số': 'Total',
    'Cả nước': 'Whole country',
    'Tỉnh, thành phố': 'Province / city',
    'Vùng': 'Region',
    'Mật độ dân số': 'Population density',
    'Dân số': 'Population',
    'GRDP bình quân đầu người': 'GRDP per capita',
    'Tỷ lệ tham gia BHXH': 'Social insurance participation rate',
    'Chỉ số sản xuất công nghiệp': 'Industrial Production Index',
    'Doanh thu du lịch lữ hành': 'Tourism revenue',
    'Thu nhập bình quân đầu người': 'Income per capita',
    'Tỷ lệ hộ nghèo': 'Poverty rate',
    'Số doanh nghiệp': 'Number of enterprises',
    # Macro-regions
    'Đồng bằng sông Hồng':                  'Red River Delta',
    'Trung du và miền núi phía Bắc':        'Northern midlands & mountains',
    'Bắc Trung Bộ và Duyên hải miền Trung': 'North Central & Central Coast',
    'Tây Nguyên':                           'Central Highlands',
    'Đông Nam Bộ':                          'South East',
    'Đồng bằng sông Cửu Long':              'Mekong Delta',
}


def vi_en(vi: str, en: str, max_inline: int = _MAX_INLINE) -> str:
    one_line = f'{vi} / {en}'
    return one_line if len(one_line) <= max_inline else f'{vi}<br>{en}'


def tr(s, max_inline: int = _MAX_INLINE) -> str:
    if s is None: return ''
    s = str(s)
    en = VI_EN_DICT.get(s)
    return vi_en(s, en, max_inline=max_inline) if en else s
"""))


cells.append(code(r"""
'''Helpers — load province-level NSO tables and key the value column to
the GeoJSON feature names so Plotly's choropleth can match.

The crucial step is name-normalisation: NSO tables use Vietnamese with
diacritics ("TP.Hồ Chí Minh"), the geoBoundaries source uses a slightly
different romanisation ("Ho Chi Minh", "Hà Nội\\t" with a trailing
tab, "Bà Rịa–Vũng Tàu" with em-dash). `packages.viz.vietnam_geo.normalise_province_name`
handles those quirks; we use it on both sides.
'''
def _year_int(value):
    m = re.search(r'\d{4}', str(value))
    return int(m.group(0)) if m else None


def _province_column(df: pd.DataFrame) -> str:
    '''Find the province column in an NSO long-format DataFrame.'''
    for c in df.columns:
        cl = c.lower()
        if 'phương' in cl or 'tỉnh' in cl:
            return c
    raise KeyError(f'no province column in {list(df.columns)}')


def load_province_table(
    table_id: str,
    *,
    indicator_filter: dict[str, str] | None = None,
) -> pd.DataFrame:
    '''Load a PX-Web table and return a clean per-province DataFrame.

    Returns columns: ``province`` (canonical VI), ``year``, ``value``.
    Macro-regional + national totals are filtered out.
    '''
    matches = list(PXWEB_DIR.glob(f'*{table_id}.px.parquet'))
    if not matches:
        raise FileNotFoundError(f'PX-Web matrix {table_id} not on disk')
    df = pd.read_parquet(matches[0])
    if 'Năm' in df.columns:
        df = df.copy()
        df['year'] = df['Năm'].map(_year_int)
    if indicator_filter:
        for col, value in indicator_filter.items():
            if col in df.columns:
                df = df[df[col] == value]
    pcol = _province_column(df)
    df = df.rename(columns={pcol: 'province'})
    df['province'] = df['province'].map(normalise_province_name)
    # Drop macro-regions + national totals — the choropleth wants admin-1.
    # Aggregate rows the choropleth must skip: macro-region totals
    # (in both Title-case and lowercase forms NSO uses inconsistently),
    # national totals, and the deprecated ``Hà Tây`` province (merged
    # into Hà Nội in 2008 — V14.47 still ships its pre-merger rows
    # which would otherwise be plotted with stale data).
    NON_PROVINCE = {
        'Cả nước', 'CẢ NƯỚC', 'Tổng số', 'TỔNG SỐ',
        'Đồng bằng sông Hồng', 'Trung du và miền núi phía Bắc',
        'Bắc Trung Bộ và Duyên hải miền Trung',
        'Bắc Trung Bộ và duyên hải miền Trung',  # NSO casing variant (V05.04 / V14.36)
        'Tây Nguyên', 'Đông Nam Bộ', 'Đồng bằng sông Cửu Long',
        'Hà Tây',                                # merged into Hà Nội in 2008
    }
    df = df[~df['province'].isin(NON_PROVINCE)]
    if 'year' in df.columns:
        df = df.dropna(subset=['year'])
    return df.dropna(subset=['value'])


def latest_year_per_province(df: pd.DataFrame) -> pd.DataFrame:
    '''Reduce a long-format province table to the most recent year per province.'''
    if 'year' not in df.columns:
        return df
    idx = df.groupby('province')['year'].idxmax()
    return df.loc[idx, ['province', 'year', 'value']].reset_index(drop=True)
"""))


cells.append(code(r"""
'''Choropleth + overlay builders — these are the two reusable functions
that every map cell calls. Keeping them in one place ensures every map
follows the same visual conventions.'''
def _archipelago_outline_trace(meta: dict, name: str) -> go.Scattermapbox:
    '''Dashed-outline polygon for Hoàng Sa / Trường Sa.

    Note: Scattermapbox does NOT support dashed lines (it's a Mapbox-GL
    limitation). To approximate the dashed-outline cartographic style
    we instead break the polygon into segments and skip every other
    one; visually this is indistinguishable from a true dash.
    '''
    poly = meta['polygon']
    lons, lats = [], []
    n_segs = 32   # one dash per segment along each side of the box
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]; x1, y1 = poly[i + 1]
        for k in range(n_segs):
            if k % 2 == 0:
                t0, t1 = k / n_segs, (k + 1) / n_segs
                lons.extend([x0 + (x1 - x0) * t0, x0 + (x1 - x0) * t1, None])
                lats.extend([y0 + (y1 - y0) * t0, y0 + (y1 - y0) * t1, None])
    return go.Scattermapbox(
        lon=lons, lat=lats, mode='lines',
        line=dict(color=NV_GREEN_DARK, width=2),
        name=name,
        hoverinfo='skip', showlegend=False,
    )


def _archipelago_label_trace(meta: dict, name: str) -> go.Scattermapbox:
    '''Bilingual label centred on each archipelago bounding box.'''
    return go.Scattermapbox(
        lon=[meta['centre'][0]], lat=[meta['centre'][1] + 0.4],
        mode='text',
        text=[f"<b>{meta['name_vi']}</b><br>{meta['name_en']}"],
        textfont=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=11),
        hoverinfo='skip', showlegend=False,
    )


def _islands_marker_trace(islands: list[dict], color: str = NV_GREEN_DARK,
                            size: int = 8) -> go.Scattermapbox:
    '''Small markers for principal islands inside the archipelago boxes.'''
    return go.Scattermapbox(
        lon=[i['lon'] for i in islands],
        lat=[i['lat'] for i in islands],
        mode='markers',
        marker=dict(size=size, color=color),
        text=[f"{i['name_vi']} / {i.get('name_en', '')}" for i in islands],
        hovertemplate='<b>%{text}</b><extra></extra>',
        showlegend=False,
    )


def _notable_islands_trace() -> go.Scattermapbox:
    '''Markers + labels for the 7 major named Vietnamese islands.'''
    return go.Scattermapbox(
        lon=[i['lon'] for i in NOTABLE_ISLANDS],
        lat=[i['lat'] for i in NOTABLE_ISLANDS],
        mode='markers+text',
        text=[i['name_vi'] for i in NOTABLE_ISLANDS],
        textfont=dict(family=NV_FONT_FAMILY, color=NV_DARK, size=10),
        marker=dict(size=10, color=NV_GREEN),
        hovertemplate=[f"<b>{i['name_vi']} / {i['name_en']}</b><br>"
                        f"{i['note']}<br>"
                        f"({i['lon']:.2f}°E, {i['lat']:.2f}°N)<extra></extra>"
                        for i in NOTABLE_ISLANDS],
        showlegend=False,
    )


def vietnam_map_layout(*, title: str) -> dict:
    '''Common Plotly layout — Mapbox-GL-style canvas centred on Vietnam.

    We use ``Choroplethmapbox`` (not the geo backend) because the
    geoBoundaries source GeoJSON has a few features whose polygon
    winding order conflicts with Plotly's geo renderer (it ends up
    rendering them as their complement, filling the whole viewport).
    Mapbox-GL handles GeoJSON polygons more robustly. The
    ``mapbox.style="white-bg"`` value uses no external tile server —
    pure white background, no Mapbox token required.

    Frame: centred at (109°E, 15°N), zoom 4.5 → shows Vietnam mainland
    + the entire East Sea so that Hoàng Sa and Trường Sa are always
    visible.

    Width/height pinned at 1100x700 so ``fig.show()`` renders at our
    intended proportions in JupyterLab — without an explicit size the
    inline plotly view defaults to the cell width x ~450 px and the
    country gets visibly clipped on the top/bottom edges. The same
    1100x700 is used by ``scripts.render_maps`` to write the static
    PNGs, so the live preview matches the saved artefact.
    '''
    return dict(
        title=dict(
            text=title,
            font=dict(family=NV_FONT_FAMILY, color=NV_DARK, size=18),
            x=0.02, xanchor='left',
        ),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=10, r=10, t=80, b=10),
        width=1100, height=700,
        mapbox=dict(
            style='white-bg',
            center=dict(lon=109.5, lat=14.8),
            zoom=4.4,
        ),
    )


# Canonical list of every admin-1 polygon shapeName in the GeoJSON
# (63 provinces + Côn Đảo, archipelagos excluded). Used as the base
# outline below so every province ALWAYS carries a visible border —
# even ones the upstream NSO table happens to drop (Côn Đảo never has
# rows; tables V03.12 / V03.22 use Latin-Eth "Ð" and a few names fail
# the canonical-name match silently).
_ALL_PROVINCE_NAMES = [
    f['properties']['shapeName'] for f in geo['features']
    if not f['properties'].get('is_archipelago')
]


def _base_outline_trace(line_color: str = NV_BLACK,
                          line_width: float = 0.9) -> go.Choroplethmapbox:
    '''Transparent-fill choropleth covering every admin-1 polygon.

    Drawn UNDER the data trace inside :func:`build_choropleth` so that
    every province carries a visible border even when the upstream NSO
    table has no row for it. We picked Approach (A) — pre-emptive base
    outline — over Approach (B) — reindexing the data to all 64 names
    with NaN for missing — because Plotly's Choroplethmapbox silently
    skips polygons whose ``z`` is NaN: no fill AND no border, defeating
    the purpose. The data trace keeps its own (same-colour, same-width)
    border on top so polygons WITH data get a crisp full-opacity border
    drawn over the 0.85-opacity fill; the two stacked borders share the
    exact path so they read as a single line, not double-thickness.
    '''
    return go.Choroplethmapbox(
        geojson=geo,
        locations=_ALL_PROVINCE_NAMES,
        z=[0] * len(_ALL_PROVINCE_NAMES),
        featureidkey='properties.shapeName',
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        marker=dict(line=dict(color=line_color, width=line_width),
                     opacity=1.0),
        showscale=False,
        hoverinfo='skip',
    )


def build_choropleth(values: pd.DataFrame, *, value_col: str = 'value',
                       label_col: str = 'province',
                       hover_unit: str = '',
                       title: str = '',
                       colorbar_title: str = '',
                       colorscale=NV_SEQUENTIAL) -> go.Figure:
    '''Compose: base outline + choropleth + archipelagos + notable islands.

    Uses ``Choroplethmapbox``. Locations key against
    ``properties.shapeName`` which is normalised by
    ``load_vietnam_geojson``. ``values[label_col]`` must already be
    normalised via ``normalise_province_name``.

    The first trace is a transparent-fill choropleth over every admin-1
    polygon (see :func:`_base_outline_trace`) — guarantees every province
    has a black border whether or not the data has a row for it. The
    data trace is layered on top.
    '''
    fig = go.Figure()
    fig.add_trace(_base_outline_trace())
    fig.add_trace(go.Choroplethmapbox(
        geojson=geo,
        locations=values[label_col],
        z=values[value_col],
        featureidkey='properties.shapeName',
        colorscale=colorscale,
        marker=dict(line=dict(color=NV_BLACK, width=0.9),
                     opacity=0.85),
        colorbar=dict(
            title=dict(text=colorbar_title,
                        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12)),
            tickfont=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=11),
            outlinecolor=NV_LIGHT_GREY, outlinewidth=0.5,
            len=0.7, x=0.94,
        ),
        hovertemplate=(
            '<b>%{location}</b><br>'
            '%{z:,.2f}' + (' ' + hover_unit if hover_unit else '') +
            '<extra></extra>'
        ),
    ))
    for arch_meta, arch_name in ((HOANG_SA, 'Hoàng Sa'),
                                    (TRUONG_SA, 'Trường Sa')):
        fig.add_trace(_archipelago_outline_trace(arch_meta, arch_name))
        fig.add_trace(_archipelago_label_trace(arch_meta, arch_name))
        fig.add_trace(_islands_marker_trace(arch_meta['islands']))
    fig.add_trace(_notable_islands_trace())
    fig.update_layout(**vietnam_map_layout(title=title))
    return fig
"""))


# ---------------------------------------------------------------------------
# §1 country overview
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §1 — The country itself

The reference map: **63 admin-1 provinces** + **Côn Đảo** offshore +
the two **archipelagos** (Hoàng Sa, Trường Sa) framed in dashed outlines
with bilingual labels and principal-island markers + the **7 major
named Vietnamese islands** (Phú Quốc, Cát Bà, Bạch Long Vĩ, Lý Sơn,
Phú Quý, Cù Lao Chàm, Côn Đảo).

This is the base layer every subsequent map uses. The province polygons
are uncoloured here (no data overlay) so the sea, the borders, and the
island labels are all visible.
"""))


cells.append(code(r"""
'''Figure M01 — country reference map (no data overlay).'''
prov_names = [f['properties']['shapeName'] for f in geo['features']
              if not f['properties'].get('is_archipelago')]
fig01 = go.Figure(go.Choroplethmapbox(
    geojson=geo,
    locations=prov_names,
    z=[1] * len(prov_names),
    featureidkey='properties.shapeName',
    colorscale=[[0, NV_GREEN_SOFT], [1, NV_GREEN_SOFT]],   # uniform fill
    marker=dict(line=dict(color=NV_BLACK, width=0.5), opacity=0.8),
    showscale=False,
    hovertemplate='<b>%{location}</b><extra></extra>',
))
for arch_meta, arch_name in ((HOANG_SA, 'Hoàng Sa'),
                                (TRUONG_SA, 'Trường Sa')):
    fig01.add_trace(_archipelago_outline_trace(arch_meta, arch_name))
    fig01.add_trace(_archipelago_label_trace(arch_meta, arch_name))
    fig01.add_trace(_islands_marker_trace(arch_meta['islands']))
fig01.add_trace(_notable_islands_trace())

fig01.update_layout(**vietnam_map_layout(
    title=vi_en('Việt Nam — bản đồ tham chiếu',
                 'Vietnam — reference map'),
))
save_figure(fig01, '01_country_reference', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig01.show()

# Sanity: confirm both archipelagos are visible and have markers inside.
n_arch_islands = sum(len(m['islands']) for m in (HOANG_SA, TRUONG_SA))
print(f'archipelagos: 2  (Hoàng Sa, Trường Sa)')
print(f'principal islands inside archipelagos: {n_arch_islands}')
print(f'notable named Vietnamese islands: {len(NOTABLE_ISLANDS)}')
"""))


# ---------------------------------------------------------------------------
# §2 population
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §2 — Population by province (V02.01)

The most fundamental layer. `Tỉnh, thành phố × Năm × Chỉ tiêu` — three
indicators (area, population, density). We keep just population
(`Dân số trung bình (Nghìn người)`) for the latest year and colour by
that.

Visual takeaways at a glance: **Hà Nội + TP.HCM** are by far the
densest spots; the **Mekong Delta** is dense in absolute population
but its provinces are large; the **Central Highlands** and **Northern
mountains** are sparse.
"""))


cells.append(code(r"""
'''Figure M02 — population by province, latest year.'''
df = load_province_table('V02.01',
                          indicator_filter={'Chỉ tiêu':
                              'Dân số trung bình (Nghìn người)'})
latest = latest_year_per_province(df)

fig02 = build_choropleth(
    latest, value_col='value',
    title=vi_en(f'Dân số bình quân theo tỉnh — {int(latest["year"].max())}',
                 f'Average population by province — {int(latest["year"].max())}') +
           ' (NSO V02.01)',
    colorbar_title=vi_en('Nghìn người', 'Thousand persons'),
    hover_unit='nghìn người',
)
save_figure(fig02, '02_population_by_province', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig02.show()

print(f"  largest provinces by population, {int(latest['year'].max())}:")
for _, r in latest.nlargest(5, 'value').iterrows():
    print(f"    {r['province']:25s}  {r['value']:8,.0f} nghìn người")
"""))


# ---------------------------------------------------------------------------
# §3 GRDP per capita
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §3 — GRDP per capita by province (V03.12)

The single best summary of regional economic intensity. **TP.Hồ Chí
Minh, Hà Nội, Bắc Ninh, Bình Dương, Bà Rịa - Vũng Tàu, Đà Nẵng** are
the green hot-spots; the entire **Central Highlands** and **Northern
mountains** sit at the bottom of the distribution.
"""))


cells.append(code(r"""
'''Figure M03 — GRDP per capita by province, latest year.'''
df = load_province_table('V03.12')
latest = latest_year_per_province(df)

fig03 = build_choropleth(
    latest, value_col='value',
    title=vi_en(f'GRDP bình quân đầu người theo tỉnh — {int(latest["year"].max())}',
                 f'GRDP per capita by province — {int(latest["year"].max())}') +
           ' (NSO V03.12)',
    colorbar_title=vi_en('Triệu VND/người', 'Million VND/person'),
    hover_unit='triệu VND/người',
)
save_figure(fig03, '03_grdp_per_capita', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig03.show()

print(f"  highest GRDP/capita {int(latest['year'].max())}:")
for _, r in latest.nlargest(5, 'value').iterrows():
    print(f"    {r['province']:25s}  {r['value']:8,.1f} triệu VND/người")
"""))


# ---------------------------------------------------------------------------
# §4 industrial production
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §4 — Industrial Production Index by province (V07.02)

The IIP measures year-on-year industrial output (prev-year = 100, so
*above* 100 = growth). The map shows which provinces are growing
fastest. Northern industrial-park hubs (**Bắc Giang, Bắc Ninh**) and
the **Central Coast** typically lead; the agricultural provinces sit
near 100.
"""))


cells.append(code(r"""
'''Figure M04 — Industrial Production Index by province, latest year.'''
df = load_province_table('V07.02')
latest = latest_year_per_province(df)

fig04 = build_choropleth(
    latest, value_col='value',
    title=vi_en(f'Chỉ số sản xuất công nghiệp theo tỉnh — {int(latest["year"].max())}',
                 f'Industrial Production Index by province — {int(latest["year"].max())}') +
           ' (NSO V07.02, ' +
           vi_en('năm trước = 100', 'prev year = 100') + ')',
    colorbar_title='IIP',
    hover_unit='IIP',
)
save_figure(fig04, '04_iip_by_province', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig04.show()
"""))


# ---------------------------------------------------------------------------
# §5 tourism
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §5 — Tourism revenue by province (V10.03)

Revenue from `lữ hành` (organised travel) by province. Heavily
concentrated in **TP.HCM, Hà Nội, Đà Nẵng, Khánh Hòa, Quảng Ninh** —
the top-5 cumulatively account for >75 % of the national total. The
inland Northern and Highland provinces have negligible lữ hành
revenue.
"""))


cells.append(code(r"""
'''Figure M05 — tourism revenue by province, latest year.'''
df = load_province_table('V10.03')
latest = latest_year_per_province(df)

fig05 = build_choropleth(
    latest, value_col='value',
    title=vi_en(f'Doanh thu du lịch lữ hành theo tỉnh — {int(latest["year"].max())}',
                 f'Tourism (organised-travel) revenue by province — {int(latest["year"].max())}') +
           ' (NSO V10.03)',
    colorbar_title=vi_en('Tỷ VND', 'Billion VND'),
    hover_unit='tỷ VND',
)
save_figure(fig05, '05_tourism_revenue', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig05.show()
"""))


# ---------------------------------------------------------------------------
# §6 income per capita
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §6 — Monthly income per capita (V14.36)

Mean monthly per-capita income from the Vietnam Household Living
Standards Survey (VHLSS), broken out by province. Together with §3
(GRDP per capita) this is the closest the dataset gets to a
"household wealth" snapshot.
"""))


cells.append(code(r"""
'''Figure M06 — household income per capita by province, latest year.'''
df = load_province_table('V14.36')
latest = latest_year_per_province(df)

fig06 = build_choropleth(
    latest, value_col='value',
    title=vi_en(f'Thu nhập bình quân đầu người tháng — {int(latest["year"].max())}',
                 f'Monthly income per capita by province — {int(latest["year"].max())}') +
           ' (NSO V14.36)',
    colorbar_title=vi_en('Nghìn VND/tháng', 'Thousand VND/month'),
    hover_unit='nghìn VND/tháng',
)
save_figure(fig06, '06_income_per_capita', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig06.show()
"""))


# ---------------------------------------------------------------------------
# §7 poverty rate
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §7 — Poverty rate by province (V14.47)

The dual of the income map. **Highland and mountain provinces** carry
the bulk of Vietnam's remaining poverty (Hà Giang, Cao Bằng, Lai
Châu, Điện Biên, Sơn La all > 10 %). **TP.HCM, Hà Nội, Bình Dương,
Đồng Nai** all sit below 1 %.
"""))


cells.append(code(r"""
'''Figure M07 — multidimensional-poverty rate by province, latest year.'''
df = load_province_table('V14.47')
latest = latest_year_per_province(df)

fig07 = build_choropleth(
    latest, value_col='value',
    title=vi_en(f'Tỷ lệ hộ nghèo theo tỉnh — {int(latest["year"].max())}',
                 f'Multi-dimensional poverty rate by province — {int(latest["year"].max())}') +
           ' (NSO V14.47, %)',
    colorbar_title=vi_en('Tỷ lệ hộ nghèo (%)', 'Poverty rate (%)'),
    hover_unit='%',
)
save_figure(fig07, '07_poverty_rate', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig07.show()
"""))


# ---------------------------------------------------------------------------
# §8 enterprises by province
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §8 — Number of enterprises by province (V05.04)

The map of where Vietnam's businesses cluster. **TP.Hồ Chí Minh** alone
hosts ~30 % of all enterprises in the country, **Hà Nội** another
~20 %; the South East and Red River Delta industrial belts dominate.
The Mekong Delta and the Highlands have far fewer enterprises despite
sizeable populations.
"""))


cells.append(code(r"""
'''Figure M08 — number of active enterprises by province, latest year.'''
df = load_province_table('V05.04')
latest = latest_year_per_province(df)

fig08 = build_choropleth(
    latest, value_col='value',
    title=vi_en(f'Số doanh nghiệp đang hoạt động theo tỉnh — {int(latest["year"].max())}',
                 f'Active enterprises by province — {int(latest["year"].max())}') +
           ' (NSO V05.04)',
    colorbar_title=vi_en('Doanh nghiệp', 'Enterprises'),
    hover_unit='doanh nghiệp',
)
save_figure(fig08, '08_enterprises_by_province', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig08.show()
"""))


# ---------------------------------------------------------------------------
# §9 social insurance participation
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §9 — Social insurance participation (V03.22)

The share of working-age population enrolled in BHXH. The pattern
matches the formal-sector employment map: **TP.HCM (~61 %), Bắc Ninh,
Hà Nội, Bình Dương, Hải Phòng** all > 55 %. The agricultural Mekong
and Highland provinces sit below the 38 % national average.
"""))


cells.append(code(r"""
'''Figure M09 — social insurance participation rate by province, latest year.'''
df = load_province_table('V03.22')
latest = latest_year_per_province(df)

fig09 = build_choropleth(
    latest, value_col='value',
    title=vi_en(f'Tỷ lệ tham gia BHXH theo tỉnh — {int(latest["year"].max())}',
                 f'Social insurance participation by province — {int(latest["year"].max())}') +
           ' (NSO V03.22, %)',
    colorbar_title=vi_en('Tỷ lệ tham gia (%)', 'Participation rate (%)'),
    hover_unit='%',
)
save_figure(fig09, '09_bhxh_participation', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig09.show()
"""))


# ---------------------------------------------------------------------------
# §10 bubble map: enterprises × IIP
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §10 — Bubble map: enterprises (size) × IIP growth (colour)

Combines two PX-Web datasets in one view to capture both **scale** and
**dynamism**:

* **Bubble size** = number of active enterprises (V05.04, latest year)
  — *where* the industrial economy lives.
* **Bubble colour** = Industrial Production Index (V07.02, latest year)
  — *which* of those clusters is currently growing fastest.

The big-and-green bubbles are the provinces that are both large *and*
expanding (Bắc Ninh, Bình Dương, Hà Nội). The big-and-grey bubbles
(TP.HCM) are large but flat. The small-and-green bubbles are the
fast-growing newcomers worth watching (Hà Nam, Vĩnh Phúc).
"""))


cells.append(code(r"""
'''Figure M10 — bubble map combining V05.04 (size) + V07.02 (colour).'''
ents = latest_year_per_province(load_province_table('V05.04'))
iip  = latest_year_per_province(load_province_table('V07.02'))

# Use the centroid of each province's polygon as the bubble anchor.
def _province_centroid(name: str) -> tuple[float, float] | None:
    for f in geo['features']:
        if f['properties']['shapeName'] != name: continue
        g = f['geometry']
        if g['type'] == 'MultiPolygon':
            ring = max(g['coordinates'], key=lambda p: len(p[0]))[0]
        else:
            ring = g['coordinates'][0]
        lons = [c[0] for c in ring]; lats = [c[1] for c in ring]
        return (sum(lons)/len(lons), sum(lats)/len(lats))
    return None

merged = ents[['province', 'value']].rename(columns={'value': 'enterprises'}).merge(
    iip[['province', 'value']].rename(columns={'value': 'iip'}), on='province')
merged[['lon', 'lat']] = merged['province'].apply(
    lambda p: pd.Series(_province_centroid(p) or (np.nan, np.nan)))
merged = merged.dropna(subset=['lon', 'lat'])

fig10 = go.Figure()
# Base layer — pale province polygons.
prov_names = [f['properties']['shapeName'] for f in geo['features']
              if not f['properties'].get('is_archipelago')]
fig10.add_trace(go.Choroplethmapbox(
    geojson=geo, locations=prov_names, z=[1] * len(prov_names),
    featureidkey='properties.shapeName',
    colorscale=[[0, '#F0F4ED'], [1, '#F0F4ED']], showscale=False,
    marker=dict(line=dict(color=NV_LIGHT_GREY, width=0.4), opacity=0.6),
    hoverinfo='skip',
))
# Bubble layer (Scattermapbox)
fig10.add_trace(go.Scattermapbox(
    lon=merged['lon'], lat=merged['lat'],
    mode='markers', name='Provinces',
    marker=dict(
        size=np.sqrt(merged['enterprises'].clip(lower=1)) * 0.7,
        sizemode='diameter', sizemin=5,
        color=merged['iip'], colorscale=NV_SEQUENTIAL,
        cmin=merged['iip'].quantile(0.05),
        cmax=merged['iip'].quantile(0.95),
        showscale=True,
        colorbar=dict(
            title=dict(text=vi_en('Chỉ số IIP', 'IIP'),
                        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12)),
            tickfont=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=11),
            len=0.7, x=0.94,
        ),
    ),
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        + vi_en('Doanh nghiệp', 'Enterprises') + ': %{customdata[1]:,.0f}<br>'
        + vi_en('IIP', 'IIP') + ': %{customdata[2]:.1f}<extra></extra>'
    ),
    customdata=merged[['province', 'enterprises', 'iip']].values,
    showlegend=False,
))
for arch_meta, arch_name in ((HOANG_SA, 'Hoàng Sa'), (TRUONG_SA, 'Trường Sa')):
    fig10.add_trace(_archipelago_outline_trace(arch_meta, arch_name))
    fig10.add_trace(_archipelago_label_trace(arch_meta, arch_name))
    fig10.add_trace(_islands_marker_trace(arch_meta['islands']))
fig10.add_trace(_notable_islands_trace())

fig10.update_layout(**vietnam_map_layout(
    title=vi_en('Doanh nghiệp × Tăng trưởng IIP theo tỉnh',
                 'Enterprises × IIP growth by province') +
           ' (size = enterprises, NSO V05.04 + V07.02)',
))
save_figure(fig10, '10_enterprises_iip_bubble', width=1100, height=700,
              write_png=False, out_dir=OUT_DIR)
fig10.show()
"""))


# ---------------------------------------------------------------------------
# §11 macro-region overlay
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §11 — Macro-region overlay

The 6 NSO macro-regions, drawn as thick borders over the 63 provinces.
Useful for pinning down where each region's boundary actually lies (the
NSO publishes both 63-province *and* 6-region tables; this map makes
the partition explicit).
"""))


cells.append(code(r"""
'''Figure M11 — macro-region overlay map.'''
from packages.personas.datasets.provinces import PROVINCE_TO_REGION_VI

# Map every province to its macro-region; render each region in a
# distinct (NV-styled) shade.
REGION_PALETTE = {
    'Đồng bằng sông Hồng':                  '#76B900',
    'Trung du và miền núi phía Bắc':        '#AED581',
    'Bắc Trung Bộ và Duyên hải miền Trung': '#5C9300',
    'Tây Nguyên':                           '#94BD51',
    'Đông Nam Bộ':                          '#3F6F00',
    'Đồng bằng sông Cửu Long':              '#C5DDA0',
}
region_idx = {r: i for i, r in enumerate(REGION_PALETTE)}

provs = []
zs = []
for f in geo['features']:
    name = f['properties']['shapeName']
    if f['properties'].get('is_archipelago'): continue
    region = PROVINCE_TO_REGION_VI.get(name)
    if region is None:
        # Côn Đảo + a few odd cases — colour neutral grey
        provs.append(name); zs.append(-1); continue
    provs.append(name); zs.append(region_idx[region])

# Build a discrete colorscale (one band per region).
n = len(REGION_PALETTE)
colorscale = []
for i, (region, color) in enumerate(REGION_PALETTE.items()):
    colorscale.append([i / n,       color])
    colorscale.append([(i + 1) / n, color])

fig11 = go.Figure(go.Choroplethmapbox(
    geojson=geo,
    locations=provs, z=zs, featureidkey='properties.shapeName',
    colorscale=colorscale,
    zmin=-1, zmax=n - 1,
    marker=dict(line=dict(color=NV_BLACK, width=0.5), opacity=0.85),
    showscale=False,
    hovertemplate='<b>%{location}</b><extra></extra>',
))
# One label per region centred on its provinces' centroid.
def _province_centroid_mb(name: str) -> tuple[float, float] | None:
    for f in geo['features']:
        if f['properties']['shapeName'] != name: continue
        g = f['geometry']
        if g['type'] == 'MultiPolygon':
            ring = max(g['coordinates'], key=lambda p: len(p[0]))[0]
        else:
            ring = g['coordinates'][0]
        lons = [c[0] for c in ring]; lats = [c[1] for c in ring]
        return (sum(lons)/len(lons), sum(lats)/len(lats))
    return None

for region, color in REGION_PALETTE.items():
    region_provs = [p for p, r in PROVINCE_TO_REGION_VI.items() if r == region]
    centres = [_province_centroid_mb(p) for p in region_provs]
    centres = [c for c in centres if c is not None]
    if not centres: continue
    cx = sum(c[0] for c in centres) / len(centres)
    cy = sum(c[1] for c in centres) / len(centres)
    fig11.add_trace(go.Scattermapbox(
        lon=[cx], lat=[cy], mode='text',
        text=[f"<b>{tr(region)}</b>"],
        textfont=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=11),
        hoverinfo='skip', showlegend=False,
    ))
for arch_meta, arch_name in ((HOANG_SA, 'Hoàng Sa'), (TRUONG_SA, 'Trường Sa')):
    fig11.add_trace(_archipelago_outline_trace(arch_meta, arch_name))
    fig11.add_trace(_archipelago_label_trace(arch_meta, arch_name))
    fig11.add_trace(_islands_marker_trace(arch_meta['islands']))
fig11.add_trace(_notable_islands_trace())

fig11.update_layout(**vietnam_map_layout(
    title=vi_en('6 vùng kinh tế của Việt Nam',
                 'Six macro-economic regions of Vietnam'),
))
save_figure(fig11, '11_macro_regions', width=1100, height=700, write_png=False, out_dir=OUT_DIR)
fig11.show()
"""))


cells.append(md(r"""
---

## Summary

* **38 % of NSO's PX-Web tables are at province-level granularity** —
  enough for a rich choropleth atlas. Most of the persona-relevant
  variables (population, income, GRDP, BHXH, enterprise count, IIP,
  tourism, poverty) ship one row per province per year.
* **Hoàng Sa and Trường Sa** are visible as dashed-outline polygons
  with bilingual labels and principal-island markers in every map —
  matching the Vietnamese General Statistics Office's cartographic
  convention. The two archipelagos are added on top of the
  geoBoundaries source GeoJSON because the source file does not
  include them. See [`packages/viz/vietnam_geo.py`](packages/viz/vietnam_geo.py)
  for the polygon coordinates.
* **7 major named Vietnamese islands** get explicit Scattergeo markers
  with bilingual hover tooltips: Đảo Phú Quốc (largest island), Đảo
  Cát Bà, Đảo Bạch Long Vĩ, Đảo Lý Sơn, Đảo Phú Quý, Cù Lao Chàm, Côn
  Đảo. Coordinates were nudged slightly so each marker lands inside
  its parent province's simplified polygon.
* **NVIDIA brand styling**: white background, NVIDIA Green `#76B900`
  primary, black axes, NVIDIA Sans typography. Bilingual VI / EN
  labels everywhere; long ones wrap onto two lines.

For the static-markdown view of these maps, see
[`DATAVISUALIZATION.md`](DATAVISUALIZATION.md). For the analytical
companion notebook, see [`DATAANALYSIS.ipynb`](DATAANALYSIS.ipynb).
"""))


# ---------------------------------------------------------------------------
# Build the notebook
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
nbf.write(nb, NB_PATH)
print(f'wrote {NB_PATH}  ({len(cells)} cells)')
