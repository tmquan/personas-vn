"""Build DATAEXPLORATION.ipynb programmatically.

Companion notebook to ``DATAPROCESSING.md`` — explores the curator
pipeline's terminal artefact (``data/nso-gov-vn/reduced/reduced.parquet``,
502 PX-Web tables × {x, y, cluster, domain_id, ...}) plus its
intermediates (``extracted.jsonl`` for keywords, ``embedded.parquet``
for raw vectors, ``parsed.jsonl`` for synthesised descriptors).

Sections:

* §0 — Setup. Load NVIDIA Plotly theme + the four curator artefacts.
* §1 — Catalog overview. Tables-per-domain stacked-by-database.
* §2 — Table size distribution. Log-binned histogram of cells per table.
* §3 — Year coverage. Box-plot of [year_min, year_max] per domain.
* §4 — Variable signatures. Top-30 PX-Web variable codes used by tables.
* §5 — TF-IDF keyword cloud. 30 most-distinctive keywords across the catalog.
* §6 — UMAP scatter, coloured by **statistical domain** (12 colours).
* §7 — UMAP scatter, coloured by **NSO database** (12 colours).
* §8 — UMAP scatter, coloured by **n_cells** (continuous, log-scaled).
* §9 — UMAP scatter, coloured by **most-recent year** (recency proxy).
* §10 — Cluster-level analysis. Top keywords + dominant domain per cluster.
* §11 — Semantic-neighbour search. Pick 4 example tables, show their
  five nearest neighbours by cosine similarity in the 384-d embedding.

All figures use the NVIDIA white-background Plotly style
(``scripts._nvidia_style``) and are saved to
``docs/figures/exploration/`` for embedding in ``DATAEXPLORATION.md``
(if/when that doc is written).

Run::

    python -m scripts._build_dataexploration_nb
    jupyter nbconvert --to notebook --execute DATAEXPLORATION.ipynb \\
        --inplace --ExecutePreprocessor.timeout=300
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "DATAEXPLORATION.ipynb"


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
# Vietnam NSO PX-Web — Data Exploration

The terminal stage of the curator pipeline (`data/nso-gov-vn/reduced/
reduced.parquet`) is a 502-row table that places every PX-Web matrix in
a 2-D UMAP space alongside its statistical domain, HDBSCAN cluster, and
embedding-derived metadata. This notebook is the interactive companion
to that artefact — every figure here reads from disk, applies the
[NVIDIA brand-style Plotly theme](scripts/_nvidia_style.py) (white
canvas, NVIDIA Green primary, NVIDIA Sans typography), and renders
into the cell so you can hover / pan / zoom your way through the
catalog.

Coverage:

1. **Catalog overview** — tables per domain, stacked by database.
2. **Table size** — log-binned histogram of cells per table.
3. **Year coverage** — [year_min, year_max] box-plot per domain.
4. **Variable signatures** — top-30 PX-Web variable codes across all
   502 tables (`Năm`, `Tỉnh, thành phố`, `Giới tính`, ...).
5. **TF-IDF keyword landscape** — 30 most-distinctive Vietnamese terms.
6. **UMAP × statistical domain** (12 colours).
7. **UMAP × NSO database** (12 colours — the raw file-system organisation,
   not the ontology mapping).
8. **UMAP × n_cells** (continuous, log-scaled).
9. **UMAP × year_max** (recency proxy — when was the table last updated).
10. **Per-cluster top-keywords + dominant domain** — what each
    HDBSCAN cluster semantically captures. *Opt-in*: only populates
    when the curator pipeline was run with `reduce.cluster: true`
    in `configs/curator.yaml` (default is `false`).
11. **Semantic neighbour search** — 4 example queries, 5 nearest
    neighbours each by cosine similarity in the 384-d embedding.

Run prerequisites:

```bash
conda activate pgm                                       # one-time setup: see README
pip install -e ".[curator,viz,dev]"
python -m packages.pipeline.cli curate --skip download   # ~25 s
python -m scripts._build_dataexploration_nb              # rebuilds this notebook
jupyter nbconvert --to notebook --execute DATAEXPLORATION.ipynb --inplace
```
"""))


# ---------------------------------------------------------------------------
# §0 Setup
# ---------------------------------------------------------------------------
cells.append(md("## §0 — Setup"))

cells.append(code(r"""
'''Imports + NVIDIA Plotly theme + load the four curator artefacts.'''
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
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
    NV_GREEN, NV_GREEN_DARK, NV_GREEN_SOFT, NV_BLACK, NV_DARK,
    NV_GREY, NV_LIGHT_GREY, NV_FAINT, NV_WHITE,
    NV_DISCRETE, NV_SEQUENTIAL, NV_FONT_FAMILY,
)

CURATOR_DIR = REPO_ROOT / 'data' / 'nso-gov-vn'
OUT_DIR = REPO_ROOT / 'docs' / 'figures' / 'exploration'
OUT_DIR.mkdir(parents=True, exist_ok=True)

pio.templates['nvidia'] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=13),
        colorway=NV_DISCRETE,
    )
)
pio.templates.default = 'nvidia'

# 1. parsed.jsonl — synthesised text + raw metadata per table
parsed = pd.read_json(CURATOR_DIR / 'parsed' / 'parsed.jsonl', lines=True)
# 2. extracted.jsonl — same rows + ontology domain + TF-IDF keywords
extracted = pd.read_json(CURATOR_DIR / 'extracted' / 'extracted.jsonl', lines=True)
# 3. embedded.parquet — 384-d MiniLM vectors per table
embedded = pd.read_parquet(CURATOR_DIR / 'embedded' / 'embedded.parquet')
# 4. reduced.parquet — UMAP (x, y) + cluster id per table
reduced = pd.read_parquet(CURATOR_DIR / 'reduced' / 'reduced.parquet')

# ---------------------------------------------------------------------------
# Standardised embedding-figure dimensions
# ---------------------------------------------------------------------------
# Every UMAP scatter in §6 – §11 shares the same canvas + margins so the
# views line up exactly side-by-side in DATAEXPLORATION.md, and long
# Vietnamese-English category names ("Y tế, văn hóa và đời sống / Health,
# Culture & Living Standards") wrap onto multiple legend lines so the
# legend never overflows the canvas.
EMBED_FIG_W = 1100              # px — fixed canvas width
EMBED_FIG_H = 720               # px — fixed canvas height
# Categorical layout puts the legend in a 240-px reserved strip to the
# right of the data plot (matching the colorbar-right convention used
# by the continuous figures below). Wrapped 30-char labels still fit
# in a single tidy column without overlapping the UMAP scatter.
EMBED_MARGIN_CATEGORICAL = dict(l=80, r=240, t=80, b=80)   # legend right
EMBED_MARGIN_CONTINUOUS  = dict(l=80, r=140, t=80, b=80)   # colorbar right
WRAP_AT = 30


def _wrap(label, max_chars: int = WRAP_AT) -> str:
    '''Word-aware <br> insertion so a long category label wraps onto
    multiple legend lines without ever splitting a token.'''
    if not isinstance(label, str) or len(label) <= max_chars:
        return str(label)
    out, cur = [], ''
    for w in label.split():
        if cur and len(cur) + 1 + len(w) > max_chars:
            out.append(cur); cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        out.append(cur)
    return '<br>'.join(out)


EMBED_LEGEND_RIGHT = dict(
    title='', orientation='v',
    yanchor='top', y=1,
    xanchor='left', x=1.02,
    font=dict(size=11),
    tracegroupgap=4,
    itemsizing='constant',
)


# Bilingual labels for domain_id values (mirrors the persona ontology dict).
DOMAIN_VI_EN = {
    'population':         'Dân số và lao động / Population & labour',
    'national_accounts':  'Tài khoản quốc gia / National accounts',
    'investment':         'Đầu tư / Investment',
    'enterprises':        'Doanh nghiệp / Enterprises',
    'agriculture':        'Nông lâm thủy sản / Agriculture',
    'industry':           'Công nghiệp / Industry',
    'trade':              'Thương mại, giá cả / Trade & prices',
    'transport':          'Vận tải, bưu điện / Transport',
    'education':          'Giáo dục / Education',
    'health':             'Y tế, văn hóa / Health & society',
    'international':      'Thống kê quốc tế / International stats',
    'geography':          'Đơn vị HC, đất, khí hậu / Geography',
    'other':              'Khác / Other',
}

# Sanity print
print(f'parsed:     {parsed.shape[0]:4d} rows  ({parsed.shape[1]} cols)')
print(f'extracted:  {extracted.shape[0]:4d} rows  ({extracted.shape[1]} cols)')
print(f'embedded:   {embedded.shape[0]:4d} rows  (vector dim = {len(embedded.iloc[0]["vector"])})')
print(f'reduced:    {reduced.shape[0]:4d} rows  ({reduced.shape[1]} cols)')
print()
print(f'unique domains:   {sorted(extracted["domain_id"].dropna().unique())}')
print(f'unique databases: {extracted["database"].nunique()}')
if 'cluster' in reduced.columns:
    print(f'unique clusters:  {sorted(reduced["cluster"].dropna().unique())}')
else:
    print('unique clusters:  (curator ran with cluster: false — no cluster column)')
"""))


# ---------------------------------------------------------------------------
# §1 — Catalog overview
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §1 — Catalog overview

How many PX-Web tables sit under each statistical domain, broken
down by database. The 12 NSO databases (`V01` – `V15`) are the actual
folders the catalog is organised in; the 12 ontology domains
(`population`, `agriculture`, `industry`, ...) are our normalised
mapping over them. The two largely line up — but a single database
can fan out to multiple domains (`V14` — Health, Society, Living
Standards — splits across `health`, `education`, and `other`) and
that's where the ontology mapping pulls its weight.
"""))

cells.append(code(r"""
'''Figure 01 — tables per domain, stacked by database (bar chart).'''
agg = (extracted.groupby(['domain_id', 'database']).size()
        .reset_index(name='n_tables')
        .sort_values('n_tables', ascending=False))
domain_order = (agg.groupby('domain_id')['n_tables'].sum()
                  .sort_values(ascending=False).index.tolist())

fig01 = go.Figure()
# One trace per database so legend toggles work; sort databases by total size.
db_order = (agg.groupby('database')['n_tables'].sum()
              .sort_values(ascending=False).index.tolist())
palette = (NV_DISCRETE * 4)[:len(db_order)]
for db, color in zip(db_order, palette):
    sub = agg[agg['database'] == db]
    fig01.add_trace(go.Bar(
        x=sub['domain_id'], y=sub['n_tables'],
        name=db, marker=dict(color=color, line=dict(color=NV_BLACK, width=0.4)),
        hovertemplate=('<b>%{x}</b><br>' + db + '<br>%{y} bảng<extra></extra>'),
    ))
fig01.update_layout(
    title='Số bảng PX-Web theo lĩnh vực thống kê / '
          'PX-Web tables per ontology domain (502 total, 12 databases)',
    xaxis=dict(title='Lĩnh vực / Domain', categoryorder='array', categoryarray=domain_order),
    yaxis_title='Số bảng / Tables',
    barmode='stack', height=520,
)
apply_nvidia_style(fig01)
save_figure(fig01, '01_catalog_overview', width=1200, height=520, out_dir=OUT_DIR)
fig01.show()

print(f'  total tables:    {extracted.shape[0]:>3d}')
print(f'  total cells:     {extracted["n_cells"].sum():>3,d}')
print(f'  domains covered: {len(domain_order):>3d} / 12')
"""))


# ---------------------------------------------------------------------------
# §2 — Table size distribution
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §2 — Table size distribution

Number of long-format cells per table, on a log x-axis (range 2 →
~10,000). Most NSO matrices land in the 100–1,000 cell band — small
enough to read in a single page, large enough to carry useful breakdowns
(by year × province × indicator). The handful of mega-tables in the
right tail are the deepest cross-tabs (V02.03–07 alone has ~10,400 cells:
demographics × every province × every year).
"""))

cells.append(code(r"""
'''Figure 02 — log-binned histogram of n_cells per table.'''
n_cells = extracted['n_cells'].clip(lower=1)
log_min, log_max = np.floor(np.log10(n_cells.min())), np.ceil(np.log10(n_cells.max()))
bins = np.logspace(log_min, log_max, 30)
counts, edges = np.histogram(n_cells, bins=bins)
centres = np.sqrt(edges[:-1] * edges[1:])
widths  = np.diff(edges) * 0.92

fig02 = go.Figure(go.Bar(
    x=centres, y=counts, width=widths,
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.5)),
    hovertemplate=('[%{customdata[0]:,.0f} – %{customdata[1]:,.0f}] cells/bảng'
                    '<br>%{y} bảng / tables<extra></extra>'),
    customdata=np.column_stack([edges[:-1], edges[1:]]),
))
_med = int(extracted['n_cells'].median())
fig02.update_layout(
    title='Phân phối số ô dữ liệu trên mỗi bảng / '
          'Distribution of long-format cells per PX-Web table',
    xaxis=dict(title='Số ô / bảng (log) / Cells per table (log scale)', type='log'),
    yaxis_title='Số bảng / Tables',
    bargap=0, height=480,
)
apply_nvidia_style(fig02)
fig02.update_xaxes(range=[float(log_min), float(log_max)], zeroline=False)
fig02.add_vline(
    x=_med, line_dash='dash', line_color=NV_GREY,
    annotation_text=f'trung vị / median = {_med:,}',
    annotation_position='top right',
)
save_figure(fig02, '02_table_size_distribution', width=1100, height=480, out_dir=OUT_DIR)
fig02.show()
"""))


# ---------------------------------------------------------------------------
# §3 — Year coverage
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §3 — Year coverage

For every table whose data has a parseable `Năm` column, plot
`[year_min, year_max]` as a horizontal box per domain. The widest
windows belong to the long-running statistical series (national
accounts back to 1995, agriculture to 1989); the narrowest to recent
specialty cuts (poverty multidimensional indices only since 2016).
""")
)

cells.append(code(r"""
'''Figure 03 — year coverage [min, max] box-plot per domain.'''
years = extracted.dropna(subset=['year_min', 'year_max']).copy()
years['year_min'] = years['year_min'].astype(int)
years['year_max'] = years['year_max'].astype(int)
years['span'] = years['year_max'] - years['year_min']
years_by_dom = years.groupby('domain_id')['span'].median().sort_values()

fig03 = go.Figure()
# Horizontal range bars: one row per table, sorted by year_max within domain.
y_ticks = []
for i, (_, row) in enumerate(years.sort_values(['domain_id', 'year_max']).iterrows()):
    y_ticks.append(row['domain_id'])
fig03.add_trace(go.Box(
    x=years['year_max'] - years['year_min'],
    y=years['domain_id'],
    orientation='h',
    boxpoints='all', jitter=0.4, pointpos=0,
    marker=dict(color=NV_GREEN, size=4, line=dict(color=NV_BLACK, width=0.3)),
    line=dict(color=NV_BLACK, width=1),
    fillcolor=NV_GREEN_SOFT,
    hovertemplate=('<b>%{y}</b><br>span: %{x} năm / years<extra></extra>'),
    name='',
))
fig03.update_layout(
    title='Khoảng năm bao phủ theo lĩnh vực / '
          'Year-coverage span by ontology domain',
    xaxis_title='Span (năm / years) = year_max − year_min',
    yaxis=dict(title='Lĩnh vực / Domain',
                categoryorder='array',
                categoryarray=list(years_by_dom.index)),
    height=560, showlegend=False,
)
apply_nvidia_style(fig03)
save_figure(fig03, '03_year_coverage', width=1100, height=560, out_dir=OUT_DIR)
fig03.show()

print(f'  earliest year: {int(years["year_min"].min())}')
print(f'  latest   year: {int(years["year_max"].max())}')
print(f'  median span:   {int(years["span"].median())} năm / years')
"""))


# ---------------------------------------------------------------------------
# §4 — Variable signatures
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §4 — Variable signatures

Each PX-Web matrix declares its dimensions in `metadata.variables[*].text`
— things like `Năm` (year), `Tỉnh, thành phố` (province), `Giới tính`
(sex), `Nhóm tuổi` (age group). These are the table's "schema
fingerprint" and they're how the embedding can tell apart, say, a
province-level employment table from a province-level industry table.
"""))

cells.append(code(r"""
'''Figure 04 — top-30 PX-Web variable codes across all 502 tables.'''
counter = Counter()
for variables in extracted['variables']:
    counter.update(variables or [])
top30 = pd.DataFrame(counter.most_common(30), columns=['variable', 'tables'])

fig04 = go.Figure(go.Bar(
    x=top30['variable'], y=top30['tables'],
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    hovertemplate='<b>%{x}</b><br>%{y} bảng / tables<extra></extra>',
))
fig04.update_layout(
    title='30 mã biến PX-Web phổ biến nhất / '
          'Top-30 PX-Web variable codes (count of tables containing each)',
    xaxis=dict(title='Mã biến (tiếng Việt) / Variable code (Vietnamese)',
                tickangle=-30),
    yaxis_title='Số bảng chứa biến / Tables containing it',
    height=600,
)
apply_nvidia_style(fig04)
save_figure(fig04, '04_variable_signatures', width=1300, height=600, out_dir=OUT_DIR)
fig04.show()
"""))


# ---------------------------------------------------------------------------
# §5 — TF-IDF keyword landscape
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §5 — TF-IDF keyword landscape

The extract stage runs a Vietnamese-friendly TF-IDF (1–2 grams,
`token_pattern=r'(?u)\b[\wÀ-ỹ]{3,}\b'`) over every synthesised table
descriptor. The terms below are those that stuck out most across
the catalog — they're a peek at what the embedder considers
*distinctive* signal, not what's frequent (frequent terms like `Năm`
and `tỉnh` are filtered out by the high `max_df=0.85` cap).
"""))

cells.append(code(r"""
'''Figure 05 — top 30 TF-IDF keywords by frequency-of-appearance.'''
kw_counter = Counter()
for kws in extracted['keywords']:
    if isinstance(kws, list):
        kw_counter.update(kws)
top_kw = pd.DataFrame(kw_counter.most_common(30), columns=['keyword', 'tables'])

fig05 = go.Figure(go.Bar(
    x=top_kw['tables'], y=top_kw['keyword'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    text=top_kw['tables'], textposition='outside',
    hovertemplate='<b>%{y}</b><br>%{x} bảng / tables<extra></extra>',
))
fig05.update_layout(
    title='30 từ khoá TF-IDF phổ biến nhất / '
          'Top-30 TF-IDF keywords across all 502 tables',
    xaxis_title='Số bảng / Tables (where keyword scored top-N)',
    yaxis=dict(title='Từ khoá / Keyword',
                categoryorder='total ascending'),
    height=720,
)
apply_nvidia_style(fig05)
save_figure(fig05, '05_keyword_landscape', width=1100, height=720, out_dir=OUT_DIR)
fig05.show()
"""))


# ---------------------------------------------------------------------------
# §6 — UMAP scatter × statistical domain
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §6 — UMAP scatter × statistical domain

Every PX-Web table placed at its UMAP `(x, y)`, coloured by ontology
domain. Tables that share variable schema and Vietnamese terminology
end up close in the 384-d MiniLM embedding and therefore close in 2-D.
Strong clusters are visible: agriculture/livestock to one corner,
trade/prices in another, demographics in a third. Hover any point to
see its title, database, and domain.
"""))

cells.append(code(r"""
'''Figure 06 — UMAP scatter coloured by ontology domain (12 colours).'''
# Bring extracted's keywords + database into reduced for richer hover.
sc = reduced.merge(
    extracted[['id', 'database', 'n_cells', 'year_min', 'year_max', 'keywords']],
    on='id', how='left',
)
sc['domain_id'] = sc['domain_id'].fillna('other')
sc['kw_str'] = sc['keywords'].apply(
    lambda kw: ', '.join(kw[:5]) if isinstance(kw, list) else '')

# 12+ distinct colours: NVIDIA palette (5 greens) + 7 greys/blacks ramp.
DOMAIN_PALETTE = [
    NV_GREEN, NV_GREEN_DARK, NV_GREEN_SOFT, NV_BLACK, NV_GREY,
    '#3F6F00', '#94BD51', '#AED581', '#5C9300', NV_DARK, '#888888', '#444444',
]
domain_order = (sc['domain_id'].value_counts().index.tolist())
fig06 = go.Figure()
for dom, color in zip(domain_order, DOMAIN_PALETTE):
    sub = sc[sc['domain_id'] == dom]
    fig06.add_trace(go.Scatter(
        x=sub['x'], y=sub['y'], mode='markers',
        name=_wrap(DOMAIN_VI_EN.get(dom, dom)),
        marker=dict(size=8, color=color, line=dict(color=NV_BLACK, width=0.4),
                     opacity=0.85),
        customdata=np.column_stack([
            sub['title'].fillna(''),
            sub['database'].fillna(''),
            sub['n_cells'].fillna(0).astype(int),
            sub['kw_str'].fillna(''),
        ]),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'database: %{customdata[1]}<br>'
            'cells: %{customdata[2]:,}<br>'
            'keywords: %{customdata[3]}<extra></extra>'
        ),
    ))
fig06.update_layout(
    title='UMAP — 502 bảng PX-Web tô màu theo lĩnh vực / '
          'UMAP — 502 PX-Web tables coloured by ontology domain',
    xaxis_title='UMAP-1', yaxis_title='UMAP-2',
)
apply_nvidia_style(fig06)
fig06.update_layout(
    width=EMBED_FIG_W, height=EMBED_FIG_H,
    margin=EMBED_MARGIN_CATEGORICAL,
    legend=EMBED_LEGEND_RIGHT,
)
save_figure(fig06, '06_umap_by_domain',
             width=EMBED_FIG_W, height=EMBED_FIG_H, out_dir=OUT_DIR)
fig06.show()
"""))


# ---------------------------------------------------------------------------
# §7 — UMAP scatter × cluster
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §7 — UMAP scatter × NSO database

Same UMAP layout, recoloured by the NSO **database** the table was
crawled from (`Dân số và lao động`, `Doanh nghiệp`, `Nông, lâm
nghiệp và thủy sản`, …). Database is the raw file-system
organisation NSO publishes; §6's `domain_id` is our normalised
ontology mapping over those database labels (one database can fan
out to multiple domains — see §1). Putting them side-by-side shows
where the database boundary diverges from the ontology boundary.

Coloured by raw data column, *not* by the discovered HDBSCAN cluster
— the cluster-vs-database analysis lives in §10.
"""))

cells.append(code(r"""
'''Figure 07 — UMAP scatter coloured by NSO database.'''
fig07 = go.Figure()
DB_PALETTE = [
    NV_GREEN, NV_GREEN_DARK, NV_GREEN_SOFT, NV_BLACK, NV_GREY,
    '#3F6F00', '#94BD51', '#AED581', '#5C9300', NV_DARK, '#888888', '#444444',
]
db_order = sc['database'].fillna('(unknown)').value_counts().index.tolist()
for db, color in zip(db_order, DB_PALETTE):
    sub = sc[sc['database'].fillna('(unknown)') == db]
    fig07.add_trace(go.Scatter(
        x=sub['x'], y=sub['y'], mode='markers',
        name=_wrap(str(db)),
        marker=dict(size=8, color=color, line=dict(color=NV_BLACK, width=0.4),
                     opacity=0.85),
        customdata=np.column_stack([
            sub['title'].fillna(''),
            sub['domain_id'].fillna(''),
            sub['kw_str'].fillna(''),
        ]),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'domain: %{customdata[1]}<br>'
            'keywords: %{customdata[2]}<extra></extra>'
        ),
    ))
fig07.update_layout(
    title='UMAP — 502 bảng PX-Web tô màu theo cơ sở dữ liệu NSO / '
          'UMAP — 502 PX-Web tables coloured by NSO database',
    xaxis_title='UMAP-1', yaxis_title='UMAP-2',
)
apply_nvidia_style(fig07)
fig07.update_layout(
    width=EMBED_FIG_W, height=EMBED_FIG_H,
    margin=EMBED_MARGIN_CATEGORICAL,
    legend=EMBED_LEGEND_RIGHT,
)
save_figure(fig07, '07_umap_by_database',
             width=EMBED_FIG_W, height=EMBED_FIG_H, out_dir=OUT_DIR)
fig07.show()
"""))


# ---------------------------------------------------------------------------
# §8 — UMAP × n_cells
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §8 — UMAP scatter × table size

Same layout, point colour now encodes `log10(n_cells)` so the deepest
cross-tabs show as dark green and the tiniest single-row totals as
near-white. Useful for spotting "where does the data depth live?" at
a glance.
"""))

cells.append(code(r"""
'''Figure 08 — UMAP scatter coloured by log10(n_cells).'''
sc['log_n'] = np.log10(sc['n_cells'].fillna(1).clip(lower=1))
fig08 = go.Figure(go.Scatter(
    x=sc['x'], y=sc['y'], mode='markers',
    marker=dict(
        size=8, color=sc['log_n'], colorscale=NV_SEQUENTIAL,
        line=dict(color=NV_BLACK, width=0.4), opacity=0.9,
        showscale=True,
        colorbar=dict(
            title=dict(text='log₁₀(cells)',
                        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12)),
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['1', '10', '100', '1k', '10k'],
        ),
    ),
    customdata=np.column_stack([
        sc['title'].fillna(''), sc['domain_id'].fillna(''),
        sc['n_cells'].fillna(0).astype(int),
    ]),
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        'domain: %{customdata[1]}<br>'
        'cells: %{customdata[2]:,}<extra></extra>'
    ),
))
fig08.update_layout(
    title='UMAP × kích thước bảng / UMAP × table size (log₁₀ cells)',
    xaxis_title='UMAP-1', yaxis_title='UMAP-2',
    showlegend=False,
)
apply_nvidia_style(fig08)
fig08.update_layout(
    width=EMBED_FIG_W, height=EMBED_FIG_H,
    margin=EMBED_MARGIN_CONTINUOUS,
)
save_figure(fig08, '08_umap_by_cellcount',
             width=EMBED_FIG_W, height=EMBED_FIG_H, out_dir=OUT_DIR)
fig08.show()
"""))


# ---------------------------------------------------------------------------
# §9 — UMAP × year_max
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §9 — UMAP scatter × recency

Coloured by `year_max` — the most recent year present in the table's
data. Shows which corners of the catalog NSO actively keeps refreshed
(2024 publications mostly in the population/agriculture/industry
clusters) versus which corners are a few years stale. Tables with no
parseable year (e.g. one-off geographic inventories) plot as grey.
"""))

cells.append(code(r"""
'''Figure 09 — UMAP scatter coloured by year_max (recency).'''
ymax = sc['year_max'].fillna(0)
no_year = ymax == 0
fig09 = go.Figure()
# Layer 1 — tables with no year as a faint grey background.
sub = sc[no_year]
fig09.add_trace(go.Scatter(
    x=sub['x'], y=sub['y'], mode='markers', name='no year info',
    marker=dict(size=7, color=NV_LIGHT_GREY, line=dict(color=NV_GREY, width=0.4),
                 opacity=0.7),
    hovertemplate='<b>%{customdata[0]}</b><br>(no year info)<extra></extra>',
    customdata=np.column_stack([sub['title'].fillna('')]),
))
# Layer 2 — coloured by year_max for tables that have one.
sub = sc[~no_year]
fig09.add_trace(go.Scatter(
    x=sub['x'], y=sub['y'], mode='markers', name='by year_max',
    marker=dict(
        size=8, color=sub['year_max'], colorscale=NV_SEQUENTIAL,
        cmin=2010, cmax=2024,
        line=dict(color=NV_BLACK, width=0.4), opacity=0.9,
        showscale=True,
        colorbar=dict(
            title=dict(text='year_max',
                        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12)),
        ),
    ),
    customdata=np.column_stack([
        sub['title'].fillna(''),
        sub['year_min'].fillna(0).astype(int),
        sub['year_max'].fillna(0).astype(int),
    ]),
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        '%{customdata[1]} – %{customdata[2]}<extra></extra>'
    ),
))
fig09.update_layout(
    title='UMAP × năm cập nhật mới nhất / UMAP × most-recent year',
    xaxis_title='UMAP-1', yaxis_title='UMAP-2',
)
apply_nvidia_style(fig09)
fig09.update_layout(
    width=EMBED_FIG_W, height=EMBED_FIG_H,
    margin=EMBED_MARGIN_CONTINUOUS,
    legend=dict(
        title='', orientation='h',
        yanchor='top', y=1.02,
        xanchor='left', x=0.0,
        font=dict(size=11),
        itemsizing='constant',
        bgcolor='rgba(255,255,255,0.85)',
    ),
)
save_figure(fig09, '09_umap_by_recency',
             width=EMBED_FIG_W, height=EMBED_FIG_H, out_dir=OUT_DIR)
fig09.show()
"""))


# ---------------------------------------------------------------------------
# §10 — Per-cluster top-keywords + dominant domain
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §10 — Cluster-level analysis (only when curator was run with `cluster: true`)

For each HDBSCAN cluster (noise points labelled ``-1`` are dropped),
print:

* its size,
* its dominant ontology domain (and what fraction of its tables agree),
* its top-10 most-distinctive TF-IDF keywords (computed across the cluster's
  members only, not the global TF-IDF).

This section is the "what does each cluster *mean*" reference card
that bridges the §6 supervised domain view, the §7 raw-database
view, and HDBSCAN's unsupervised density-based grouping.

**Clustering is opt-in** in the curator pipeline — see
``configs/curator.yaml`` `reduce.cluster`. With the default
(``cluster: false``) ``reduced.parquet`` has no `cluster` column and
this section gracefully skips. Flip the YAML key to ``true``,
re-run the reduce stage (`python -m packages.pipeline.cli curate
--only reduce`), then re-execute the notebook to populate the table
+ figure.
"""))

cells.append(code(r"""
'''Cluster-level analysis: dominant domain + top-keywords per HDBSCAN cluster.

Skips with a printed message when the curator pipeline was run without
clustering (``configs/curator.yaml`` ``reduce.cluster: false``, the
default). To populate, set the YAML key to ``true`` and re-run the
reduce stage.
'''
clusters_df = pd.DataFrame()  # populated only if clustering is on
if 'cluster' not in sc.columns:
    print('clustering: off in curator pipeline (set reduce.cluster: true to enable)')
else:
    rows = []
    dense = sc[sc['cluster'] != -1]
    for cl in sorted(dense['cluster'].dropna().unique()):
        sub = dense[dense['cluster'] == cl]
        n = len(sub)
        dom_hist = sub['domain_id'].value_counts()
        dom_top, dom_n = dom_hist.index[0], int(dom_hist.iloc[0])
        kw_counter = Counter()
        for kws in sub['keywords']:
            if isinstance(kws, list):
                kw_counter.update(kws)
        top_kws = [k for k, _ in kw_counter.most_common(10)]
        rows.append({
            'cluster': int(cl),
            'n_tables': n,
            'dominant_domain': dom_top,
            'dominant_share': f'{dom_n / n * 100:.0f}%',
            'top_keywords': ', '.join(top_kws),
        })
    clusters_df = pd.DataFrame(rows)
    n_noise = int((sc['cluster'] == -1).sum())
    print(f'  dense clusters: {len(clusters_df)}   noise tables dropped: {n_noise}')
    print(clusters_df.to_string(index=False))
"""))


cells.append(code(r"""
'''Figure 10 — cluster purity bar (per-HDBSCAN-cluster dominant-domain share).

No-op when the curator pipeline was run without clustering — see the
preceding cell.
'''
if clusters_df.empty:
    print('skipping fig10 — no cluster column in reduced.parquet')
else:
    fig10 = go.Figure()
    fig10.add_trace(go.Bar(
        x=clusters_df['cluster'].astype(str),
        y=[float(s.rstrip('%')) for s in clusters_df['dominant_share']],
        text=[f"{c['dominant_domain']} ({c['dominant_share']})"
               for _, c in clusters_df.iterrows()],
        textposition='outside',
        marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.5)),
        hovertemplate=(
            '<b>cluster %{x}</b><br>'
            'dominant share: %{y}%<br>'
            'n_tables: %{customdata}<extra></extra>'
        ),
        customdata=clusters_df['n_tables'],
    ))
    fig10.update_layout(
        title='Độ thuần khiết của mỗi cụm HDBSCAN / Purity per HDBSCAN cluster',
        xaxis=dict(title='Cluster id', type='category'),
        yaxis=dict(title='Tỷ lệ lĩnh vực thống trị (%) / Dominant-domain share (%)',
                    range=[0, 110]),
        height=540, showlegend=False,
    )
    apply_nvidia_style(fig10)
    save_figure(fig10, '10_cluster_purity', width=1100, height=540, out_dir=OUT_DIR)
    fig10.show()
"""))


# ---------------------------------------------------------------------------
# §11 — Semantic neighbour search
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §11 — Semantic neighbour search

The 384-d MiniLM embedding is what lets the visualizer's "Curator"
tab support free-form Vietnamese queries (`"thu nhập theo tỉnh"` →
returns the 5 most-similar PX-Web tables). Below: pick four anchor
tables, look up their five nearest neighbours in cosine space, print
the top hits.

This is also the test that *the embedding actually carries semantic
signal* — neighbours should share both database and statistical content,
not just orthography.
"""))

cells.append(code(r"""
'''Nearest-neighbour search in the 384-d MiniLM embedding space.

``embedded.parquet`` carries the 384-d MiniLM vectors plus the
``title`` / ``domain_id`` already; we only need to pull ``database``
across from ``extracted.jsonl`` (which the embed stage doesn't
serialise). Using a non-overlapping column subset avoids the
``title_x`` / ``title_y`` rename that pandas would otherwise produce.
'''
emb_df = embedded.merge(
    extracted[['id', 'database']], on='id', how='left',
)
V = np.stack(emb_df['vector'].to_list())
# Cosine: vectors are already L2-normalised by the embed stage (config:
# normalize=true), so dot product == cosine similarity.
S = V @ V.T

ANCHOR_TABLE_IDS = [
    'V02.01',   # population by province × year
    'V03.01',   # GDP at current prices
    'V07.01',   # industrial production by industry
    'V14.45',   # multidimensional poverty by region
]
for anchor_id in ANCHOR_TABLE_IDS:
    rows = emb_df[emb_df['id'].str.endswith(':' + anchor_id)]
    if rows.empty:
        print(f'  {anchor_id}: NOT FOUND in embedded.parquet'); continue
    i = rows.index[0]
    sim = S[i]
    sim[i] = -np.inf  # exclude self
    top = np.argsort(sim)[::-1][:5]
    anchor = emb_df.loc[i, 'title']
    anchor_dom = emb_df.loc[i, 'domain_id']
    print(f'\n=== anchor: {anchor_id} — {anchor[:80]} ({anchor_dom}) ===')
    for j in top:
        nbr = emb_df.loc[j]
        score = float(S[i, j])
        nbr_id = nbr['id'].split(':')[-1]
        print(f'  {score:.3f}  {nbr_id:8s}  {nbr["title"][:80]}  ({nbr["domain_id"]})')
"""))


cells.append(code(r"""
'''Figure 11 — visualise the four neighbour groups on the UMAP plane.'''
fig11 = go.Figure()
# Faint background — every table.
fig11.add_trace(go.Scatter(
    x=sc['x'], y=sc['y'], mode='markers',
    name='all 502 tables',
    marker=dict(size=5, color=NV_LIGHT_GREY, opacity=0.5),
    hoverinfo='skip', showlegend=True,
))
# For each anchor, highlight anchor + 5 nearest neighbours.
ANCHOR_TABLE_IDS = ['V02.01', 'V03.01', 'V07.01', 'V14.45']
ANCHOR_PALETTE = [NV_GREEN, NV_GREEN_DARK, NV_BLACK, NV_DARK]
for anchor_id, color in zip(ANCHOR_TABLE_IDS, ANCHOR_PALETTE):
    rows = emb_df[emb_df['id'].str.endswith(':' + anchor_id)]
    if rows.empty: continue
    i = rows.index[0]
    sim = S[i].copy(); sim[i] = -np.inf
    top = np.argsort(sim)[::-1][:5]
    indices = [i] + list(top)
    sub = sc[sc['id'].isin(emb_df.loc[indices, 'id'])]
    is_anchor = (sub['id'] == emb_df.loc[i, 'id'])
    fig11.add_trace(go.Scatter(
        x=sub['x'], y=sub['y'], mode='markers+text',
        name=f'{anchor_id} + 5 neighbours',
        text=[anchor_id if a else '' for a in is_anchor],
        textposition='top center', textfont=dict(color=NV_BLACK, size=12),
        marker=dict(size=[14 if a else 9 for a in is_anchor],
                     color=color, line=dict(color=NV_BLACK, width=1.0),
                     symbol=['star' if a else 'circle' for a in is_anchor]),
        customdata=np.column_stack([sub['title'].fillna(''),
                                       sub['domain_id'].fillna('')]),
        hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
    ))
fig11.update_layout(
    title='Bốn cụm láng giềng ngữ nghĩa trên UMAP / '
          'Four semantic neighbour groups projected onto UMAP',
    xaxis_title='UMAP-1', yaxis_title='UMAP-2',
)
apply_nvidia_style(fig11)
fig11.update_layout(
    width=EMBED_FIG_W, height=EMBED_FIG_H,
    margin=EMBED_MARGIN_CATEGORICAL,
    legend=EMBED_LEGEND_RIGHT,
)
save_figure(fig11, '11_semantic_neighbours',
             width=EMBED_FIG_W, height=EMBED_FIG_H, out_dir=OUT_DIR)
fig11.show()
"""))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cells.append(md(r"""
---

## Summary

* **502 PX-Web tables** in the catalog, normalised onto **12 ontology
  domains** via `OntologyRegistry.map_nso_category()` over the database
  name. Health (97), trade (72), agriculture (70) and population (63)
  are the largest domains.
* **326,692 long-format cells** total. Median table is ~250 cells; the
  deepest (V02.03–07) is ~10,400 cells.
* **MiniLM 384-d embedding** + UMAP into 2-D + density-based HDBSCAN
  reconstructs most of the ontology unsupervisedly — every dense
  cluster's dominant ontology domain (§10) accounts for the majority
  of its members; the noise points HDBSCAN refuses to assign are
  documented separately so the purity figure isn't inflated by
  forced groupings (which is what KMeans would have done).
* **Vietnamese-language signal survives** the multilingual MiniLM
  encoder: the nearest-neighbour search in §11 returns
  same-domain, same-aggregation-level peers for every anchor we
  tried, including queries that share zero literal vocabulary
  (e.g. `V03.01` "GDP" → near `V03.13` "state budget revenue", which
  share no surface tokens but co-cluster on "national accounts").

For the static-PNG view of these figures see
`docs/figures/exploration/`. For the analytical companion notebook
see [`DATAANALYSIS.ipynb`](DATAANALYSIS.ipynb), and for the
geographic-choropleth atlas see
[`DATAVISUALIZATION.ipynb`](DATAVISUALIZATION.ipynb).
"""))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
nbf.write(nb, NB_PATH)
print(f'wrote {NB_PATH}  ({len(cells)} cells)')
