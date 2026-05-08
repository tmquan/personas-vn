"""Build DATASYNTHESIS.ipynb programmatically.

Companion to ``_build_dataanalysis_nb.py`` and ``_build_datavisualization_nb.py``
— same declarative ``cells = []`` pattern driven into ``nbformat`` — but
the focus is the **synthesis side** of the project: take a 10% subset of
the small Nemotron-Personas-Vietnam parquet (300K rows × 22 columns, see
``DATASETS.md``) and run it through the same four post-generation stages
the curator uses for NSO source documents:

    parse → extract → embed → reduce

Stage-by-stage:

* **parse**   — load the parquet, validate the 22-column schema, build a
  unified ``text`` field per persona by concatenating its narrative
  columns. Mirrors :class:`packages.curator.stages.ParseStage`.
* **extract** — TF-IDF top-N keywords per persona narrative + per macro-
  region. Mirrors :class:`packages.curator.stages.ExtractStage`.
* **embed**   — local multilingual SBERT (384-d) on each narrative.
  One flag (``backend="nim"`` + ``model="nvidia/...`"``) flips to NIM.
  Mirrors :func:`packages.personas.embed.pipeline.embed_personas`.
* **reduce**  — UMAP project to 2-D for plotting. Density-based
  HDBSCAN clustering is **optional** (off by default, mirroring the
  production pipeline `cluster=False` flag); the four scatter plots
  in §6 colour by raw demographic columns (region / occupation /
  education / area) regardless. Mirrors
  :class:`packages.curator.stages.ReduceStage`.

Every figure follows the NVIDIA brand-style preamble (white background,
NVIDIA Green ``#76B900`` for the primary data series, NVIDIA Sans
typography fallback) and is exported to ``docs/figures/synthesis/``
both as interactive HTML and as a static PNG so a markdown companion
can embed them inline.

Run::

    python -m scripts._build_datasynthesis_nb
    jupyter nbconvert --to notebook --execute DATASYNTHESIS.ipynb \\
        --inplace --ExecutePreprocessor.timeout=1800

The cell list below is the full source of the notebook; rebuilding the
notebook from scratch is just ``python -m scripts._build_datasynthesis_nb``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "DATASYNTHESIS.ipynb"


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
# Data synthesis — running parse / extract / embed / reduce on a 10% subset

Companion to [`DATASYNTHESIS.md`](DATASYNTHESIS.md) and
[`DATASETS.md`](DATASETS.md). Where those documents describe how the
synthesis side *works*, this notebook is the **end-to-end runnable
demo**: take 10% of the small Nemotron-Personas-Vietnam parquet (300K
rows × 22 columns) and push it through the same four post-generation
stages the curator uses on NSO source documents:

    parse → extract → embed → reduce

| Stage     | What it does                                             | Mirror in code                                              |
| --------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| parse     | load + schema-validate + build unified narrative text    | `packages.curator.stages.ParseStage`                        |
| extract   | TF-IDF top-N keywords per persona + per region           | `packages.curator.stages.ExtractStage`                      |
| embed     | sentence-transformers (384-d) over the narrative         | `packages.personas.embed.pipeline.embed_personas`           |
| reduce    | UMAP → 2-D (HDBSCAN clustering optional, off by default) | `packages.curator.stages.ReduceStage`                       |

Every figure follows the
[NVIDIA brand guidelines](https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/):

* **White** background
* **NVIDIA Green** `#76B900` for the primary data series
* **Black** axis chrome / **NVIDIA Sans** fallback typography

PNG snapshots land under `docs/figures/synthesis/`; the interactive
HTML lives next to each PNG so reviewers can pan / zoom / hover.

Run prerequisites:

```bash
# build the four parquets first (~5s smoke run, ~27 min full)
python -m packages.pipeline.cli build-nemotron --large-size 10000 --small-size 1000

# extras for embed + reduce + plotting
pip install -e ".[curator,viz]"

python -m scripts._build_datasynthesis_nb
jupyter nbconvert --to notebook --execute DATASYNTHESIS.ipynb \
    --inplace --ExecutePreprocessor.timeout=1800
```

Wall-clock on an M-series Mac (CPU-only, 30K-row default subset):

| Stage   | Time     |
| ------- | -------- |
| parse   | <1s      |
| extract | ~10s     |
| embed   | ~6 min   |
| reduce  | ~1-2 min |

If you want to iterate faster, dial `SAMPLE_FRAC` in §1 down (`0.01`
gives a 3K-row subset that finishes in <1 minute end to end).
"""))


# ---------------------------------------------------------------------------
# §0 — Setup
# ---------------------------------------------------------------------------
cells.append(md("## §0 — Setup"))


cells.append(code(r"""
'''Setup — imports, NVIDIA Plotly theme, paths to the small parquet datasets.'''
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings('ignore')

REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._nvidia_style import (
    apply_nvidia_style, save_figure,
    NV_GREEN, NV_GREEN_DARK, NV_GREEN_SOFT,
    NV_BLACK, NV_DARK, NV_GREY, NV_LIGHT_GREY,
    NV_WHITE, NV_DISCRETE, NV_SEQUENTIAL, NV_FONT_FAMILY,
)

DATA_DIR = REPO_ROOT / 'data' / 'Nemotron-Personas-Vietnam'
OUT_DIR  = REPO_ROOT / 'docs' / 'figures' / 'synthesis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage outputs (parsed / embedded / reduced parquets) land in a sub-
# folder so the demo never touches the canonical 300K small datasets.
SUBSET_DIR = DATA_DIR / '_subset_pipeline'
SUBSET_DIR.mkdir(parents=True, exist_ok=True)

pio.templates['nvidia'] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=13),
        colorway=NV_DISCRETE,
    )
)
pio.templates.default = 'nvidia'

print(f'data dir:   {DATA_DIR.relative_to(REPO_ROOT)}')
print(f'subset dir: {SUBSET_DIR.relative_to(REPO_ROOT)}')
print(f'figures:    {OUT_DIR.relative_to(REPO_ROOT)}')
"""))


# ---------------------------------------------------------------------------
# §1 — Load + 10% subset
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §1 — Load the small parquet & take a 10% subset

The small variant is 300K rows (10% of the published 3M ``-large-``).
Sampling 10% of *that* gives a 30K-row subset — small enough to
embed on CPU in a few minutes, large enough that the 2-D UMAP layout
captures the demographic + occupational structure.

We sample by `uuid` so the Vietnamese and English mirrors stay aligned
1-to-1 (every persona in the subset has both narrations available).
"""))


cells.append(code(r"""
'''Load both small parquets (vi + en) and take a uuid-aligned 10% sample.'''
SMALL_VI = DATA_DIR / 'Nemotron-Personas-Vietnam-small-vi.parquet'
SMALL_EN = DATA_DIR / 'Nemotron-Personas-Vietnam-small-en.parquet'

df_vi_full = pd.read_parquet(SMALL_VI)
df_en_full = pd.read_parquet(SMALL_EN)
print(f'vi parquet: {df_vi_full.shape}  ({SMALL_VI.stat().st_size / 1e6:.1f} MB)')
print(f'en parquet: {df_en_full.shape}  ({SMALL_EN.stat().st_size / 1e6:.1f} MB)')

SAMPLE_FRAC = 0.10           # 10% of small = 30K rows; lower for fast iteration
SAMPLE_SEED = 20260506

sample_uuids = (
    df_vi_full['uuid']
    .sample(frac=SAMPLE_FRAC, random_state=SAMPLE_SEED)
    .sort_values()
    .reset_index(drop=True)
)
df_vi = df_vi_full[df_vi_full['uuid'].isin(sample_uuids)].reset_index(drop=True)
df_en = df_en_full[df_en_full['uuid'].isin(sample_uuids)].reset_index(drop=True)
df_vi = df_vi.sort_values('uuid').reset_index(drop=True)
df_en = df_en.sort_values('uuid').reset_index(drop=True)
assert (df_vi['uuid'].values == df_en['uuid'].values).all(), 'uuid alignment broke'
print(f'\nsubset:     {SAMPLE_FRAC:.0%}  →  {len(df_vi):,} rows  (vi + en aligned)')

# Drop the full frames so the kernel reclaims ~600MB of resident RAM.
del df_vi_full, df_en_full
"""))


# ---------------------------------------------------------------------------
# §2 — parse
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §2 — Parse

The parquet's 22 columns split into three groups:

* **1 identifier** — `uuid`
* **12 narrative fields** — long natural-language strings
  (`persona`, `professional_persona`, `cultural_background`, …) plus
  two `_list` mirrors of the freeform skills / hobbies fields.
* **9 structured fields** — demographic, socio-economic, geographic.

`parse` validates the schema, summarises the column-level dtypes, and
concatenates the six freeform narrative columns into a single `text`
field — the same shape `packages.curator.stages.ParseStage` produces
for NSO source documents, so the downstream `extract` / `embed` stages
can be reused unchanged.
"""))


cells.append(code(r"""
'''Parse — schema-validate the 22 columns, summarise the row, build a unified
   `text` field for embedding.'''
from packages.personas.datasets.schema import (
    NEMOTRON_PERSONAS_VIETNAM_COLUMNS,
    NEMOTRON_PERSONAS_VIETNAM_FEATURES,
)

assert tuple(df_vi.columns) == NEMOTRON_PERSONAS_VIETNAM_COLUMNS, 'unexpected column layout'
print(f'columns ({len(df_vi.columns)}):')
for col in df_vi.columns:
    dtype = NEMOTRON_PERSONAS_VIETNAM_FEATURES[col]
    sample = df_vi[col].iloc[0]
    if isinstance(sample, str) and len(sample) > 60:
        sample = sample[:57] + '…'
    print(f'  {col:32s} {dtype:6s}  e.g. {sample!r}')

# Build the unified `text` field — the six long-form narrative columns.
# (The two `_list` mirrors are excluded; their information is already
# subsumed by `skills_and_expertise` / `hobbies_and_interests`.)
NARRATIVE_COLS = [
    'persona', 'professional_persona', 'cultural_background',
    'skills_and_expertise', 'hobbies_and_interests',
    'career_goals_and_ambitions',
]
def _build_text(row: pd.Series) -> str:
    return '\n'.join(str(row[c]) for c in NARRATIVE_COLS if isinstance(row[c], str) and row[c])

parsed = df_vi.copy()
parsed['text'] = parsed.apply(_build_text, axis=1)
text_lens = parsed['text'].str.len()
print(f'\nnarrative length: median={text_lens.median():.0f}  '
      f'p95={text_lens.quantile(0.95):.0f}  max={text_lens.max():.0f}  chars')

parsed_path = SUBSET_DIR / 'parsed.parquet'
parsed.to_parquet(parsed_path, index=False)
print(f'wrote {parsed_path.relative_to(REPO_ROOT)}  (n={len(parsed):,})')
"""))


cells.append(code(r"""
'''Quick demographic snapshot of the subset — sanity check that the 10%
   sample preserves the marginal shape of the parent 300K small dataset.'''
def _value_counts(col: str, top: int = 6) -> pd.DataFrame:
    s = parsed[col].value_counts(normalize=True).head(top)
    return s.mul(100).round(1).rename('%').reset_index()

snap = pd.concat({
    'sex':              _value_counts('sex'),
    'region':           _value_counts('region'),
    'education_level':  _value_counts('education_level'),
    'occupation':       _value_counts('occupation', top=5),
}, axis=1)
snap
"""))


# ---------------------------------------------------------------------------
# §3 — extract (TF-IDF)
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §3 — Extract — TF-IDF keywords

`extract` mirrors `packages.curator.stages.ExtractStage`: a
multilingual-friendly TF-IDF (with the same Unicode-aware token pattern
the curator uses, ``\b[\wÀ-ỹ]{3,}\b``) gives every narrative its
top-N keywords. Unlike the curator we don't need to map records to an
ontology domain — the parquet already carries `region` / `occupation`
/ `education_level` etc as first-class columns, so we group on those
directly.

Two views below:

1. **Per-persona top-N keywords** — a flat column we attach back to
   `parsed`.
2. **Per-region top keywords** — what tokens carry the most TF-IDF
   weight in each macro-region's narratives. The bar chart's facets
   show how the six regions diverge linguistically.
"""))


cells.append(code(r"""
'''Extract — TF-IDF top-8 keywords per persona narrative.'''
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(
    max_df=0.85, min_df=5,
    ngram_range=(1, 2),
    token_pattern=r'(?u)\b[\wÀ-ỹ]{3,}\b',  # match the curator
    max_features=20000,
)
matrix = vec.fit_transform(parsed['text'])
vocab = vec.get_feature_names_out()
print(f'tfidf matrix: {matrix.shape[0]:,} docs × {matrix.shape[1]:,} terms')

TOP_N = 8
def _top_keywords(i: int) -> list[str]:
    row = matrix.getrow(i).toarray()[0]
    if not row.any():
        return []
    top = sorted(enumerate(row), key=lambda kv: -kv[1])[:TOP_N]
    return [str(vocab[j]) for j, w in top if w > 0]

parsed['keywords'] = [_top_keywords(i) for i in range(matrix.shape[0])]
parsed[['uuid', 'occupation', 'region', 'keywords']].head(5)
"""))


cells.append(code(r"""
'''Top TF-IDF keywords per macro-region — what's distinctive in each
   region's persona narratives.'''
region_kw = (
    parsed[['region', 'keywords']]
    .explode('keywords').dropna(subset=['keywords'])
    .groupby(['region', 'keywords'], as_index=False).size()
    .rename(columns={'size': 'count'})
)
top_per_region = (
    region_kw.sort_values(['region', 'count'], ascending=[True, False])
             .groupby('region').head(8)
             .reset_index(drop=True)
)
fig = px.bar(
    top_per_region, x='count', y='keywords',
    facet_col='region', facet_col_wrap=3, orientation='h',
    title=f'Top TF-IDF keywords per region · 10% subset (n={len(parsed):,})',
    color_discrete_sequence=[NV_GREEN],
)
fig.update_yaxes(autorange='reversed', matches=None, showticklabels=True)
fig.update_xaxes(matches=None)
fig.for_each_annotation(lambda a: a.update(text=a.text.replace('region=', '')))
apply_nvidia_style(fig)
fig.update_layout(height=900, showlegend=False)
save_figure(fig, '01_tfidf_keywords_per_region', out_dir=OUT_DIR, height=900)
fig.show()
"""))


# ---------------------------------------------------------------------------
# §4 — embed
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §4 — Embed

`embed` runs a multilingual sentence-transformers model
(`paraphrase-multilingual-MiniLM-L12-v2`, 384-d, free + offline) over
each persona's narrative. Switching to a hosted NIM model is one line:

```python
from packages.personas.embed.backends import make_embedding_backend
backend = make_embedding_backend(
    model='nvidia/llama-3.2-nv-embedqa-1b-v2',  # 2048-d
    backend='nim',
    nim_config=NimEmbeddingsConfig(api_key_env='NVIDIA_API_KEY'),
)
```

The model id alone is enough — `backend="auto"` (the default) routes
anything starting with `nvidia/` to NIM and everything else to the
local SBERT path.

Wall-clock on an M-series Mac, CPU-only, 30K rows: **~6 minutes**.
"""))


cells.append(code(r"""
'''Embed — local SBERT (384-d) over the 30K narratives.'''
from packages.personas.embed.backends import (
    LOCAL_DEFAULT_MODEL,
    make_embedding_backend,
)

EMBED_MODEL   = LOCAL_DEFAULT_MODEL              # paraphrase-multilingual-MiniLM-L12-v2
EMBED_BACKEND = 'auto'                           # auto-routes (no `nvidia/...` → local)
backend = make_embedding_backend(model=EMBED_MODEL, backend=EMBED_BACKEND, device='cpu')
print(f'backend: {backend.name}   model: {backend.model}')

# SBERT truncates at ~128 tokens (~500 chars); cap at 2000 to be safe
# without slowing the encoder.
texts = parsed['text'].str.slice(0, 2000).tolist()
vectors = backend.encode(
    texts, batch_size=64, normalize=True,
    input_type='passage', show_progress=True,
)
backend.close()
print(f'\nembedded: {vectors.shape}   dtype={vectors.dtype}')

embedded = parsed[[
    'uuid', 'sex', 'age', 'marital_status', 'education_level',
    'occupation', 'region', 'area', 'province', 'country',
]].copy()
embedded['vector'] = list(vectors)
embedded_path = SUBSET_DIR / 'embedded.parquet'
embedded.to_parquet(embedded_path, index=False)
print(f'wrote {embedded_path.relative_to(REPO_ROOT)}')
"""))


# ---------------------------------------------------------------------------
# §5 — reduce
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §5 — Reduce — UMAP → 2-D (clustering optional)

`reduce` always projects the 384-d vectors to 2-D for plotting via
UMAP. **Density-based HDBSCAN clustering is optional** and matches
the production pipeline default (`packages.personas.embed.pipeline.PersonaEmbedConfig.cluster=False`,
`configs/curator.yaml` `reduce.cluster: false`, `embed-personas
--cluster` opt-in). The four scatter plots in §6 colour by raw
persona columns regardless, so clustering is only useful if you
want the §7 *modal-demographic-per-cluster* summary table.

Flip `CLUSTER = True` in the cell below to additionally compute
[HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
labels. We pick HDBSCAN over KMeans because it (1) discovers cluster
count from local density instead of demanding a fixed `k`, (2)
tolerates non-spherical cluster shapes, and (3) labels low-density
points as noise (`-1`) rather than forcing every persona into a
cluster.
"""))


cells.append(code(r"""
'''Reduce — UMAP project to 2-D, optionally HDBSCAN cluster the 384-d vectors.

``CLUSTER`` is the opt-in toggle that mirrors the production pipeline.
Default is ``False`` — only the 2-D UMAP coords get written. Flip to
``True`` to additionally compute HDBSCAN cluster labels and add a
``cluster`` column to ``reduced.parquet`` (see §7 below).
'''
import umap

CLUSTER = False     # pipeline default — flip to True for §7 cluster summary

X = np.stack(embedded['vector'].to_list())
print(f'input: {X.shape}')

reducer = umap.UMAP(
    n_components=2, n_neighbors=25, min_dist=0.10,
    metric='cosine', random_state=20260506,
)
coords = reducer.fit_transform(X)
print(f'UMAP coords: {coords.shape}')

reduced = embedded.drop(columns=['vector']).copy()
reduced['x'] = coords[:, 0]
reduced['y'] = coords[:, 1]

if CLUSTER:
    from sklearn.cluster import HDBSCAN

    # min_cluster_size scales with the corpus so a 30K-row subset gets
    # ~30-50 clusters of ~375 personas each.
    MIN_CLUSTER_SIZE = max(5, X.shape[0] // 80)
    MIN_SAMPLES      = max(3, MIN_CLUSTER_SIZE // 4)
    clusters = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom',
    ).fit_predict(X)
    n_dense = int(clusters.max()) + 1 if (clusters >= 0).any() else 0
    n_noise = int((clusters == -1).sum())
    print(f'HDBSCAN:    min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}')
    print(f'  found {n_dense} clusters + {n_noise} noise points '
          f'({n_noise / len(clusters):.1%} of input)')
    reduced['cluster'] = clusters
else:
    print('clustering: off (pipeline default — see §5 prose to opt in)')

reduced_path = SUBSET_DIR / 'reduced.parquet'
reduced.to_parquet(reduced_path, index=False)
print(f'wrote {reduced_path.relative_to(REPO_ROOT)}')
"""))


# ---------------------------------------------------------------------------
# §6 — visualise
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §6 — Visualise the 2-D semantic map

Four UMAP scatter plots, all on the same `(x, y)` layout and same
fixed canvas, coloured by **structured persona columns** (not by the
discovered HDBSCAN cluster) so each view lines up directly with one
axis of the 22-column schema:

* §6.1 — `region` — six NSO macro-regions
* §6.2 — `occupation` — 10-class ISCO breakdown
* §6.3 — `education_level` — five-step qualification ladder
* §6.4 — `area` — urban / rural

The discovered HDBSCAN clusters are surfaced separately in §7 as a
cluster-vs-data summary table — this section is exclusively about
how each *raw* persona attribute manifests in the embedding.

We sub-sample to 6K points for the HTML/PNG so the inline render
stays light; the underlying `reduced.parquet` keeps every row.
"""))


cells.append(code(r"""
'''Helper — UMAP scatter coloured by an arbitrary categorical column.

Every embedding figure in §6 shares the same canvas + margins so the
four views line up exactly side-by-side in DATASYNTHESIS.md, and long
Vietnamese-English category names (e.g. ``Bắc Trung Bộ và Duyên hải
miền Trung``, ``Lao động có kỹ năng trong nông nghiệp...``) wrap onto
multiple legend lines so the legend never overflows the canvas.
'''
EMBED_FIG_W   = 1100        # px — fixed canvas width
EMBED_FIG_H   = 720         # px — fixed canvas height
EMBED_MARGIN  = dict(l=80, r=40, t=80, b=260)   # bottom-heavy: legend lives there
WRAP_AT       = 30          # break long legend labels every ~30 chars
PLOT_SAMPLE   = min(6000, len(reduced))         # cap for inline rendering


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


def _scatter(color_col: str, title: str) -> go.Figure:
    df_plot = reduced.sample(PLOT_SAMPLE, random_state=20260506).copy()
    df_plot['_legend'] = df_plot[color_col].map(_wrap)
    legend_order = sorted(df_plot['_legend'].unique())
    fig = px.scatter(
        df_plot, x='x', y='y', color='_legend',
        title=title, opacity=0.7,
        color_discrete_sequence=NV_DISCRETE,
        category_orders={'_legend': legend_order},
        hover_data={
            'occupation': True, 'region': True, 'education_level': True,
            '_legend': False, 'x': False, 'y': False,
        },
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.update_xaxes(title='UMAP-x')
    fig.update_yaxes(title='UMAP-y')
    apply_nvidia_style(fig)
    fig.update_layout(
        width=EMBED_FIG_W, height=EMBED_FIG_H,
        margin=EMBED_MARGIN,
        legend=dict(
            title='', orientation='h',
            yanchor='top', y=-0.15,
            xanchor='center', x=0.5,
            font=dict(size=11),
            tracegroupgap=4,
            itemsizing='constant',
        ),
    )
    return fig
"""))


cells.append(code(r"""
fig = _scatter('region', 'UMAP — coloured by macro-region')
save_figure(fig, '02_umap_by_region', out_dir=OUT_DIR,
             width=EMBED_FIG_W, height=EMBED_FIG_H)
fig.show()
"""))


cells.append(code(r"""
fig = _scatter('occupation', 'UMAP — coloured by occupation (10-class ISCO)')
save_figure(fig, '03_umap_by_occupation', out_dir=OUT_DIR,
             width=EMBED_FIG_W, height=EMBED_FIG_H)
fig.show()
"""))


cells.append(code(r"""
fig = _scatter('education_level', 'UMAP — coloured by education level')
save_figure(fig, '04_umap_by_education', out_dir=OUT_DIR,
             width=EMBED_FIG_W, height=EMBED_FIG_H)
fig.show()
"""))


cells.append(code(r"""
fig = _scatter('area', 'UMAP — coloured by area (urban / rural)')
save_figure(fig, '05_umap_by_area', out_dir=OUT_DIR,
             width=EMBED_FIG_W, height=EMBED_FIG_H)
fig.show()
"""))


# ---------------------------------------------------------------------------
# §7 — what's in each cluster
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §7 — What's in each HDBSCAN cluster (only if §5 `CLUSTER=True`)

HDBSCAN is unsupervised — it doesn't know the structured persona
columns exist, and unlike KMeans it doesn't force every point into
a cluster. The most useful sanity check is to look at the **modal
demographics inside each cluster**: if the embedding learned
something real about the narrative, clusters should split cleanly on
high-information axes (occupation > region > education > area).

Cluster id ``-1`` collects the noise points HDBSCAN couldn't
confidently assign — we drop those from the summary so the table
only describes the dense regions of the embedding.

This cell is a no-op when §5 ran with the pipeline default
(`CLUSTER = False`); flip §5's toggle on, re-execute it, and re-run
this cell to populate the summary table.
"""))


cells.append(code(r"""
'''Modal demographic profile per HDBSCAN cluster (noise dropped).

Skips with a printed message if §5 ran with ``CLUSTER = False`` — i.e.
``reduced.parquet`` has no ``cluster`` column. Flip §5's toggle and
re-run §5 + §7 to populate the table.
'''
if 'cluster' not in reduced.columns:
    cluster_summary = None
    print('clustering: off (re-run §5 with CLUSTER=True to populate this summary)')
else:
    def _mode(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iat[0]) if not m.empty else ''

    dense = reduced[reduced['cluster'] != -1]
    cluster_summary = (
        dense.groupby('cluster')
             .agg(n=('uuid', 'size'),
                  top_region=('region', _mode),
                  top_occupation=('occupation', _mode),
                  top_education=('education_level', _mode),
                  top_area=('area', _mode))
             .reset_index()
             .sort_values('n', ascending=False)
             .reset_index(drop=True)
    )
    print(f'  dense clusters: {len(cluster_summary)}   '
          f'noise rows dropped: {(reduced["cluster"] == -1).sum():,} '
          f'({(reduced["cluster"] == -1).mean():.1%} of subset)')
cluster_summary
"""))


cells.append(md(r"""
---

## Where to look in code

| If you want to...                                              | Read                                                          |
| -------------------------------------------------------------- | ------------------------------------------------------------- |
| Change the parse → extract → embed → reduce stages on NSO docs | [`packages/curator/stages.py`](packages/curator/stages.py)    |
| Switch from local SBERT to NIM hosted embeddings               | [`packages/personas/embed/backends.py`](packages/personas/embed/backends.py) |
| Replay this notebook on the **full** 300K small parquet        | bump `SAMPLE_FRAC = 1.00` in §1                               |
| Run the same flow against the **3M large** parquet             | swap `SMALL_VI` / `SMALL_EN` for the `-large-` paths in §1    |
| Regenerate the four parquets from scratch                      | `python -m packages.pipeline.cli build-nemotron`              |

See [`DATASYNTHESIS.md`](DATASYNTHESIS.md) for the conceptual
walkthrough and [`DATASETS.md`](DATASETS.md) for the schema definition
and source-of-truth Python identifiers.
"""))


# ---------------------------------------------------------------------------
# Write out the notebook
# ---------------------------------------------------------------------------
def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {NB_PATH.relative_to(REPO)}  ({n_md} markdown + {n_code} code cells)")


if __name__ == "__main__":
    main()
