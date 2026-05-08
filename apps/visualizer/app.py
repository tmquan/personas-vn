"""Gradio visualizer.

We chose Gradio over a TypeScript/Next.js front-end because the entire data
pipeline is Python-native (pandas, Pydantic, numpy) — Gradio lets us plug
DataFrames straight into a Web UI with zero serialisation glue.

Tabs:

* **Overview** — pipeline status, top-level counts (PX-Web tables crawled,
  personas, enrichment).
* **Ontology** — interactive view of the statistical domains and the
  persona dimension graph (rendered with Plotly so the user can hover for
  details and the page stays responsive at 80+ nodes).
* **PX-Web tables** — searchable table of every NSO PX-Web matrix the
  curator has downloaded, with click-through into the actual data values.
* **Personas** — generate / inspect a persona batch, language-toggle
  (vi/en) on labels, side-by-side bilingual bios, enrichment status,
  and CSV export.
* **Persona embeddings** — 2-D scatter of persona bios (UMAP), coloured
  by any structured field.
* **Curator** — staged-pipeline status + 2-D scatter of NSO documents
  + semantic neighbour search.
* **Distributions** — preview the real NSO PX-Web count tables feeding
  the persona generator's PGM (regions, occupations, education, etc.).
"""

from __future__ import annotations

import inspect
import os
from io import StringIO
from typing import Any

import pandas as pd

from apps.visualizer.data_loader import (
    PxWebTableEntry,
    VisualizerState,
    get_distributions,
    load_pxweb_table,
    load_state,
)
from packages.common.config import load_config
from packages.common.logging import get_logger
from packages.common.paths import REPO_ROOT
from packages.ontology import PersonaBatch
from packages.personas.pgm import generate_personas

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy gradio import: keeps unit tests + headless CLI runs cheap and lets
# users skip the visualiser deps entirely if they don't need the UI.
# ---------------------------------------------------------------------------
def _import_gradio():
    try:
        import gradio as gr  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "Visualizer dependencies missing. Install with `pip install -e \".[viz]\"`."
        ) from exc
    return gr


def _import_plotly():
    try:
        import plotly.express as px  # noqa: WPS433
        import plotly.graph_objects as go  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "plotly is required for the visualizer. Install with `pip install plotly`."
        ) from exc
    return px, go


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
# White background + Computer-Modern-style serif font ("LaTeX look"). We list
# several fallbacks so the font resolves correctly regardless of which TeX /
# CM family the user has installed locally; the browser picks the first one
# that's actually available.
_PAPER_FONT_FAMILY = (
    "Computer Modern, CMU Serif, Latin Modern Roman, "
    "STIX Two Text, STIX, Times New Roman, serif"
)
_PAPER_AXIS_GRID = dict(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zerolinecolor="rgba(0,0,0,0.25)",
    linecolor="black",
    ticks="outside",
    tickcolor="black",
    showline=True,
    mirror=True,
)


def _apply_paper_style(fig, *, axes: bool = True):
    """Apply the white-background + Computer-Modern look to a Plotly figure.

    ``axes=False`` is for plots where axes carry no information (e.g. the
    persona-dimension graph) — we hide grid + ticks but still keep the
    serif font and white canvas.
    """
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=_PAPER_FONT_FAMILY, color="black", size=14),
        title_font=dict(family=_PAPER_FONT_FAMILY, color="black", size=18),
        legend=dict(
            font=dict(family=_PAPER_FONT_FAMILY, color="black"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    if axes:
        fig.update_xaxes(**_PAPER_AXIS_GRID)
        fig.update_yaxes(**_PAPER_AXIS_GRID)
    else:
        for ax in (fig.update_xaxes, fig.update_yaxes):
            ax(showgrid=False, zeroline=False, showticklabels=False,
               showline=False, ticks="")
    return fig


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------
def _overview_panel(gr, state: VisualizerState) -> tuple[Any, Any]:
    """Markdown summary + a bar chart of PX-Web tables per database."""
    px, _ = _import_plotly()
    n_tables = len(state.pxweb_tables)
    n_personas = len(state.personas)
    n_enriched = state.n_enriched
    n_cells = sum(t.n_cells for t in state.pxweb_tables)

    enrichment = (state.persona_batch.enrichment if state.persona_batch else {}) or {}
    if enrichment:
        enrich_line = (
            f"- **LLM-enriched personas:** `{n_enriched}` of `{n_personas}` "
            f"(model: `{enrichment.get('model', '—')}`)\n"
        )
    else:
        enrich_line = (
            f"- **LLM-enriched personas:** `{n_enriched}` of `{n_personas}` "
            "(run `make enrich-personas` to add bilingual bios)\n"
        )

    md_text = (
        "## Pipeline status\n\n"
        f"- **PX-Web tables crawled:** `{n_tables}` (`{n_cells:,}` data cells)\n"
        f"- **Synthetic personas loaded:** `{n_personas}`\n"
        f"{enrich_line}"
        f"- **Ontology version:** `{state.registry.version}`\n\n"
        "Use the tabs above to browse the real NSO PX-Web statistical "
        "matrices, inspect the persona-dimension Bayesian DAG, generate "
        "or filter synthetic personas (bilingual VN/EN), explore the "
        "embedding map, or peek at the curator pipeline outputs."
    )
    md = gr.Markdown(md_text)

    if n_tables:
        df = state.pxweb_tables_df
        agg = df.groupby("database").size().reset_index(name="tables")
        fig = px.bar(
            agg.sort_values("tables", ascending=False),
            x="database",
            y="tables",
            title="PX-Web tables per database",
        )
        fig.update_traces(marker_color="black", marker_line_color="black")
        chart = gr.Plot(_apply_paper_style(fig))
    else:
        chart = gr.Markdown(
            "_No PX-Web tables loaded yet._ "
            "Run `personas-vn curate --only download` to populate "
            "`data/nso-gov-vn/raw/pxweb/vi/`."
        )
    return md, chart


def _ontology_panel(gr, state: VisualizerState) -> tuple[Any, Any]:
    """Domain table + persona-dimension graph."""
    _, go = _import_plotly()  # only `go` is needed in this panel

    domains = state.registry.list_domains()
    domain_df = pd.DataFrame(
        [
            {
                "id": d.id,
                "label_en": d.label_en,
                "label_vi": d.label_vi,
                "aliases": ", ".join(d.aliases),
            }
            for d in domains
        ]
    )
    table = gr.Dataframe(
        value=domain_df,
        label="Statistical domains",
        wrap=True,
        interactive=False,
    )

    # Persona dimension DAG.
    dims = state.registry.list_dimensions()
    # We do a deterministic layered layout: x = topological order index,
    # y = jittered slot inside the layer, so the graph reads left-to-right.
    layer_of: dict[str, int] = {}
    for dim in dims:
        layer_of[dim.id] = (max([layer_of[p] for p in dim.parents], default=-1)) + 1

    by_layer: dict[int, list[str]] = {}
    for dim_id, layer in layer_of.items():
        by_layer.setdefault(layer, []).append(dim_id)

    pos: dict[str, tuple[float, float]] = {}
    for layer, ids in by_layer.items():
        for i, dim_id in enumerate(sorted(ids)):
            pos[dim_id] = (float(layer), -float(i) - 0.5 * (len(ids) - 1))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for dim in dims:
        for parent in dim.parents:
            x0, y0 = pos[parent]
            x1, y1 = pos[dim.id]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="black"),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=[pos[d.id][0] for d in dims],
        y=[pos[d.id][1] for d in dims],
        mode="markers+text",
        text=[d.id for d in dims],
        textposition="bottom center",
        textfont=dict(family=_PAPER_FONT_FAMILY, color="black", size=12),
        marker=dict(
            size=22,
            color="white",
            line=dict(width=1.4, color="black"),
        ),
        hovertext=[
            f"<b>{d.id}</b><br>domain: {d.domain}<br>parents: {', '.join(d.parents) or '—'}"
            for d in dims
        ],
        hoverinfo="text",
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Persona-dimension Bayesian DAG",
            showlegend=False,
            height=520,
        ),
    )
    graph = gr.Plot(_apply_paper_style(fig, axes=False))
    return table, graph


def _pxweb_tables_panel(gr, state: VisualizerState):
    """Browse the PX-Web table catalog + preview the data of any table."""
    catalog_df = state.pxweb_tables_df
    if catalog_df.empty:
        gr.Markdown(
            "## PX-Web tables\n\n"
            "_No PX-Web tables on disk._ Run `personas-vn curate --only download` "
            "first; that walks `https://pxweb.nso.gov.vn/` and writes one parquet "
            "per matrix into `data/nso-gov-vn/raw/pxweb/vi/`."
        )
        return

    db_choices = ["(all)"] + sorted({t.database for t in state.pxweb_tables})
    table_id_choices = sorted({t.id for t in state.pxweb_tables})
    by_id: dict[str, PxWebTableEntry] = {t.id: t for t in state.pxweb_tables}

    gr.Markdown(
        "## PX-Web tables\n"
        f"`{len(state.pxweb_tables)}` matrices crawled from "
        "`https://pxweb.nso.gov.vn/api/v1/vi/`. Pick a row's id and the "
        "preview table below shows the actual data values."
    )

    db_filter = gr.Dropdown(choices=db_choices, value="(all)", label="Filter by database")
    text_filter = gr.Textbox(
        label="Search (table id / title)",
        placeholder="V02.43, occupation, dân số, ...",
    )
    catalog_view = gr.Dataframe(
        value=catalog_df, wrap=True, interactive=False, max_height=420,
    )

    table_id_input = gr.Dropdown(
        choices=table_id_choices,
        value=table_id_choices[0] if table_id_choices else None,
        label="Inspect table",
    )
    preview = gr.Dataframe(wrap=True, interactive=False, max_height=420)
    preview_caption = gr.Markdown()

    def filter_catalog(db: str, query: str) -> pd.DataFrame:
        out = catalog_df
        if db and db != "(all)":
            out = out[out["database"] == db]
        if query:
            q = query.lower().strip()
            mask = (
                out["id"].str.lower().str.contains(q, na=False)
                | out["title"].str.lower().str.contains(q, na=False)
            )
            out = out[mask]
        return out

    def show_table(table_id: str) -> tuple[pd.DataFrame, str]:
        if not table_id:
            return pd.DataFrame(), ""
        entry = by_id.get(table_id)
        if entry is None:
            return pd.DataFrame(), f"Unknown table id: `{table_id}`"
        df = load_pxweb_table(entry)
        # Preview just the first 200 rows; the full parquet may be 10K+ cells.
        preview_df = df.head(200)
        caption = (
            f"### `{entry.id}` — {entry.title}\n"
            f"- Database: `{entry.database}`\n"
            f"- Variables: `{', '.join(entry.variables)}`\n"
            f"- Total cells: `{entry.n_cells:,}` (showing first 200)\n"
            f"- File: `{entry.parquet_path.name}`"
        )
        return preview_df, caption

    db_filter.change(filter_catalog, inputs=[db_filter, text_filter], outputs=catalog_view)
    text_filter.change(filter_catalog, inputs=[db_filter, text_filter], outputs=catalog_view)
    table_id_input.change(show_table, inputs=table_id_input, outputs=[preview, preview_caption])

    # Initialise the preview with the first table.
    if table_id_choices:
        first_df, first_caption = show_table(table_id_choices[0])
        preview.value = first_df
        preview_caption.value = first_caption


_PERSONA_DISPLAY_COLS_VI = [
    "persona_id", "name", "age", "sex", "region", "urbanicity",
    "ethnicity", "marital_status", "education_level",
    "employment_status", "occupation", "industry_sector",
    "income_quintile", "enriched",
]
_PERSONA_DISPLAY_COLS_EN = [
    "persona_id", "name", "age", "sex_en", "region_en", "urbanicity_en",
    "ethnicity_en", "marital_status_en", "education_level_en",
    "employment_status_en", "occupation_en", "industry_sector_en",
    "income_quintile_en", "enriched",
]


def _project_personas(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Pick a sensible column subset based on the chosen language."""
    cols = _PERSONA_DISPLAY_COLS_VI if lang == "vi" else _PERSONA_DISPLAY_COLS_EN
    keep = [c for c in cols if c in df.columns]
    return df[keep]


def _personas_panel(gr, state: VisualizerState):
    """Browse + filter + (re)generate + inspect personas (bilingual)."""
    px, _ = _import_plotly()
    initial_df = state.personas_df

    region_choices = ["(all)"] + sorted(initial_df["region"].dropna().unique()) if not initial_df.empty else ["(all)"]
    occ_choices = ["(all)"] + sorted(initial_df["occupation"].dropna().unique()) if not initial_df.empty else ["(all)"]

    enrichment = (state.persona_batch.enrichment if state.persona_batch else {}) or {}
    enrichment_md = (
        f"**Enrichment:** {state.n_enriched} of {len(state.personas)} personas have "
        f"LLM-written bilingual bios "
        f"(model: `{enrichment.get('model', '—')}`)"
        if enrichment
        else f"**Enrichment:** 0 of {len(state.personas)} personas enriched. "
             "Run `personas-vn enrich-personas` to add LLM-written bilingual bios."
    )
    gr.Markdown(enrichment_md)

    with gr.Row():
        n_input = gr.Number(value=len(state.personas) or 200,
                            label="Number of personas",
                            precision=0, minimum=1)
        seed_input = gr.Number(value=20260505, label="Seed", precision=0)
        regen_btn = gr.Button("Regenerate personas", variant="primary")

    with gr.Row():
        region_filter = gr.Dropdown(choices=region_choices, value="(all)", label="Region (Vietnamese label)")
        occ_filter = gr.Dropdown(choices=occ_choices, value="(all)", label="Occupation (Vietnamese label)")
        lang_choice = gr.Radio(choices=["vi", "en"], value="vi", label="Display language")
        only_enriched = gr.Checkbox(value=False, label="Only show LLM-enriched")

    table = gr.Dataframe(
        value=_project_personas(initial_df, "vi"),
        wrap=True, interactive=False, max_height=420,
    )

    gr.Markdown("### Inspect a persona — side-by-side bilingual bio")
    detail_id = gr.Dropdown(
        choices=initial_df["persona_id"].head(50).tolist() if not initial_df.empty else [],
        label="Persona id (first 50 of current filtered set)",
    )
    bio_vi_box = gr.Markdown()
    bio_en_box = gr.Markdown()

    plot_region = gr.Plot()
    plot_occ = gr.Plot()
    download = gr.File(label="Download CSV", visible=False)

    cache: dict[str, pd.DataFrame] = {"current": initial_df}

    def refresh_plots(df: pd.DataFrame):
        if df.empty:
            return None, None
        agg_r = df.groupby("region").size().reset_index(name="n")
        agg_o = df.groupby("occupation").size().reset_index(name="n").sort_values("n", ascending=False)
        fig_r = px.bar(agg_r, x="region", y="n", title="Personas per region (VN labels)")
        fig_o = px.bar(agg_o, x="occupation", y="n", title="Personas per occupation (VN labels)")
        for fig in (fig_r, fig_o):
            fig.update_traces(marker_color="black", marker_line_color="black")
            _apply_paper_style(fig)
        return fig_r, fig_o

    def apply_filters(region: str, occ: str, lang: str, only_enr: bool):
        df = cache["current"]
        if region and region != "(all)":
            df = df[df["region"] == region]
        if occ and occ != "(all)":
            df = df[df["occupation"] == occ]
        if only_enr and "enriched" in df.columns:
            df = df[df["enriched"]]
        p_region, p_occ = refresh_plots(df)
        new_id_choices = df["persona_id"].head(50).tolist()
        return (
            _project_personas(df, lang),
            p_region,
            p_occ,
            gr.update(choices=new_id_choices,
                      value=new_id_choices[0] if new_id_choices else None),
        )

    def show_persona_detail(pid: str):
        if not pid:
            return "", ""
        df = cache["current"]
        row = df[df["persona_id"] == pid]
        if row.empty:
            return f"_persona id not found: {pid}_", ""
        r = row.iloc[0]
        vi = (
            f"**🇻🇳 Tiếng Việt** — {'(LLM-enriched)' if r.get('enriched') else '(template)'}\n\n"
            f"> {r.get('bio_vi') or '_(no bio)_'}\n\n"
            f"- Tên: **{r.get('name', '')}**\n"
            f"- Khu vực: {r.get('region', '')} ({r.get('urbanicity', '')})\n"
            f"- Tuổi: {r.get('age', '')} ({r.get('age_group', '')})\n"
            f"- Giới tính: {r.get('sex', '')}\n"
            f"- Dân tộc: {r.get('ethnicity', '')}\n"
            f"- Trình độ: {r.get('education_level', '')}\n"
            f"- Vị thế việc làm: {r.get('employment_status', '')}\n"
            f"- Nghề nghiệp: {r.get('occupation', '')}\n"
            f"- Ngành: {r.get('industry_sector', '')}\n"
            f"- Nhóm thu nhập: {r.get('income_quintile', '')}\n"
        )
        en = (
            f"**🇬🇧 English**\n\n"
            f"> {r.get('bio_en') or '_(no bio)_'}\n\n"
            f"- Name: **{r.get('name', '')}**\n"
            f"- Region: {r.get('region_en', '')} ({r.get('urbanicity_en', '')})\n"
            f"- Age: {r.get('age', '')} ({r.get('age_group', '')})\n"
            f"- Sex: {r.get('sex_en', '')}\n"
            f"- Ethnicity: {r.get('ethnicity_en', r.get('ethnicity', ''))}\n"
            f"- Education: {r.get('education_level_en', '')}\n"
            f"- Employment status: {r.get('employment_status_en', '')}\n"
            f"- Occupation: {r.get('occupation_en', '')}\n"
            f"- Industry: {r.get('industry_sector_en', r.get('industry_sector', ''))}\n"
            f"- Income quintile: {r.get('income_quintile', '')}\n"
        )
        return vi, en

    def regenerate(n: float, seed: float, lang: str, only_enr: bool):
        n_int = max(1, int(n or 1))
        seed_int = int(seed) if seed is not None else None
        batch: PersonaBatch = generate_personas(
            n_int,
            registry=state.registry,
            seed=seed_int,
            scrape_manifest=state.scrape_manifest,
        )
        df = pd.DataFrame([p.model_dump() for p in batch.personas])
        cache["current"] = df

        new_regions = ["(all)"] + sorted(df["region"].dropna().unique())
        new_occs = ["(all)"] + sorted(df["occupation"].dropna().unique())

        # Write to a temp CSV so the gr.File component can serve it.
        buf = StringIO()
        df.to_csv(buf, index=False)
        csv_path = REPO_ROOT / "data" / "personas" / "personas_export.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(buf.getvalue(), encoding="utf-8")

        p_region, p_occ = refresh_plots(df)
        new_id_choices = df["persona_id"].head(50).tolist()
        return (
            _project_personas(df, lang),
            gr.update(choices=new_regions, value="(all)"),
            gr.update(choices=new_occs, value="(all)"),
            p_region,
            p_occ,
            gr.update(value=str(csv_path), visible=True),
            gr.update(choices=new_id_choices,
                      value=new_id_choices[0] if new_id_choices else None),
        )

    regen_btn.click(
        regenerate,
        inputs=[n_input, seed_input, lang_choice, only_enriched],
        outputs=[table, region_filter, occ_filter, plot_region, plot_occ, download, detail_id],
    )
    for trigger in (region_filter, occ_filter, lang_choice, only_enriched):
        trigger.change(
            apply_filters,
            inputs=[region_filter, occ_filter, lang_choice, only_enriched],
            outputs=[table, plot_region, plot_occ, detail_id],
        )
    detail_id.change(show_persona_detail, inputs=detail_id, outputs=[bio_vi_box, bio_en_box])

    # Initialise: first detail panel + plots.
    init_r, init_o = refresh_plots(initial_df)
    plot_region.value = init_r
    plot_occ.value = init_o
    if not initial_df.empty:
        first_id = initial_df["persona_id"].iloc[0]
        bio_vi_box.value, bio_en_box.value = show_persona_detail(first_id)
    return table, plot_region, plot_occ, download


def _persona_embedding_panel(gr, state: VisualizerState):
    """2-D scatter of persona embeddings, coloured by demographic axis."""
    _import_plotly()  # ensures plotly is installed; helpers below import as needed

    if state.personas_reduced is None or state.personas_reduced.empty:
        gr.Markdown(
            "## Persona embedding map\n\n"
            "_No persona embedding artefacts yet._ Run the persona-embedding "
            "pipeline first:\n\n"
            "```bash\n"
            "personas-vn generate --n 100000\n"
            "personas-vn embed-personas --max-records 10000\n"
            "```\n\n"
            "Outputs land in `data/personas/embedded.parquet` + "
            "`data/personas/reduced.parquet`."
        )
        return

    df = state.personas_reduced
    # Choose what to colour by; cluster + occupation + region are usually most informative.
    color_options = [c for c in (
        "occupation", "region", "urbanicity", "education_level",
        "industry_sector", "income_quintile", "age_group", "sex",
        "ethnicity", "marital_status", "employment_status", "cluster",
    ) if c in df.columns]
    color_choice = gr.Radio(
        choices=color_options,
        value="occupation" if "occupation" in color_options else color_options[0],
        label="Colour points by",
    )

    n_total = state.persona_batch.n if state.persona_batch else len(state.personas)
    gr.Markdown(
        f"### Persona embedding map\n"
        f"- **Total personas in batch:** `{n_total}`\n"
        f"- **Personas embedded (sample):** `{len(df)}`\n"
        f"- Each point is one synthetic persona; UMAP placement reflects "
        f"semantic similarity between bios."
    )

    plot = gr.Plot(value=_persona_scatter(df, color_options[0] if "occupation" not in color_options else "occupation"))
    color_choice.change(lambda c: _persona_scatter(df, c), inputs=color_choice, outputs=plot)


def _persona_scatter(df, color_column: str):
    px, _ = _import_plotly()
    if color_column not in df.columns:
        color_column = "cluster" if "cluster" in df.columns else df.columns[0]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=df[color_column].astype(str) if color_column == "cluster" else color_column,
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data={
            "persona_id": True,
            "name": True,
            "region": True,
            "occupation": True,
            "age": True,
            "sex": True,
            "x": False,
            "y": False,
        },
        title=f"Personas UMAP — {len(df)} embedded",
        height=620,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7, line=dict(width=0.3, color="black")))
    return _apply_paper_style(fig)


def _curator_panel(gr, state: VisualizerState):
    """Curator pipeline status + 2D semantic scatter + neighbour search."""
    _import_plotly()  # ensures plotly is installed; the helpers below use it

    if state.curator_reduced is None or state.curator_reduced.empty:
        gr.Markdown(
            "## Curator pipeline\n\n"
            "_No curator artefacts yet._ Run the staged pipeline first:\n\n"
            "```bash\n"
            "pip install -e \".[curator]\"\n"
            "python -m packages.pipeline.cli curate\n"
            "```\n\n"
            "Outputs land in `data/nso-gov-vn/{raw,parsed,extracted,embedded,reduced}/`."
        )
        return

    df = state.curator_reduced
    # ----- summary -----
    manifest = state.curator_manifest
    started = manifest.get("started_at", "—")
    finished = manifest.get("finished_at", "—")
    backend = manifest.get("backend", "local")
    stages = manifest.get("stages", {})
    rows = [{"stage": s, **(info or {})} for s, info in stages.items()]
    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame([{"stage": "(none)"}])

    gr.Markdown(
        f"## Curator pipeline\n\n"
        f"- **Dataset:** `{manifest.get('dataset', '—')}`\n"
        f"- **Backend:** `{backend}`\n"
        f"- **Started / finished:** `{started}` / `{finished}`\n"
        f"- **Records embedded + reduced:** `{len(df)}`\n"
    )
    gr.Dataframe(value=summary_df, label="Per-stage summary", interactive=False, wrap=True)

    # ----- scatter (color by domain or cluster) -----
    # ``cluster`` is opt-in in the curator pipeline (see
    # ``configs/curator.yaml`` ``reduce.cluster``) so the column may be
    # absent — only offer it when present.
    curator_color_options = [
        c for c in ("domain_id", "cluster", "lang", "kind") if c in df.columns
    ]
    color_choice = gr.Radio(
        choices=curator_color_options,
        value="domain_id" if "domain_id" in curator_color_options else curator_color_options[0],
        label="Colour points by",
    )
    plot = gr.Plot(value=_scatter(df, color_choice.value))

    color_choice.change(lambda c: _scatter(df, c), inputs=color_choice, outputs=plot)

    # ----- semantic search via cached embedded.parquet -----
    gr.Markdown("### Semantic neighbour search\nFinds the records whose embeddings are closest to your query.")
    query = gr.Textbox(label="Query", placeholder="e.g. 'industrial production index 2025'")
    top_k = gr.Slider(minimum=3, maximum=25, value=8, step=1, label="Top K")
    search_btn = gr.Button("Search", variant="primary")
    results_table = gr.Dataframe(
        value=pd.DataFrame(columns=["score", "domain_id", "lang", "title", "url"]),
        interactive=False,
        wrap=True,
    )

    def search(q: str, k: int):
        if not q or not q.strip():
            return pd.DataFrame(columns=["score", "domain_id", "lang", "title", "url"])
        try:
            return _semantic_search(state, q.strip(), int(k))
        except Exception as exc:
            log.warning("semantic search failed: %s", exc)
            return pd.DataFrame([{"score": 0.0, "title": f"(search failed: {exc})"}])

    search_btn.click(search, inputs=[query, top_k], outputs=results_table)


def _scatter(df, color_column: str):
    """Build the embedding scatter; defensive against the visualiser being
    asked for a column that doesn't exist on a partial run.
    """
    px, _ = _import_plotly()
    if color_column not in df.columns:
        color_column = "domain_id" if "domain_id" in df.columns else None
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_column,
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data={"id": True, "title": True, "url": True, "domain_id": True, "lang": True},
        title=f"Curator UMAP — {len(df)} NSO records",
        height=620,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85, line=dict(width=0.4, color="black")))
    return _apply_paper_style(fig)


def _semantic_search(state: VisualizerState, query: str, k: int):
    """Re-load the embedded vectors + the same encoder that produced them.

    The encoder is sourced from ``configs/curator.yaml`` so the query
    embedding lives in the same vector space as the corpus parquet —
    works equally for the local SBERT default and for any NIM model
    (``nvidia/llama-3.2-nv-embedqa-1b-v2`` etc.). The encoder is cached
    on the state object so we only pay the load cost once per process.
    """
    import numpy as np
    import yaml as _yaml

    from packages.personas.embed.backends import (
        NimEmbeddingsConfig,
        make_embedding_backend,
    )

    embedded_path = state.curator_root / "embedded" / "embedded.parquet" if state.curator_root else None
    if not embedded_path or not embedded_path.exists():
        raise RuntimeError("embedded.parquet not found; run `personas-vn curate` first")

    backend = getattr(state, "_embed_backend", None)
    if backend is None:
        with open("configs/curator.yaml", encoding="utf-8") as fh:
            cur_cfg = _yaml.safe_load(fh)
        embed_cfg = cur_cfg.get("embed") or {}
        nim_cfg = NimEmbeddingsConfig(
            base_url=embed_cfg.get("base_url") or NimEmbeddingsConfig().base_url,
            api_key_env=embed_cfg.get("api_key_env", "PERSONAS_VN_LLM_API_KEY"),
            truncate=embed_cfg.get("truncate", "END"),
        )
        backend = make_embedding_backend(
            model=embed_cfg.get("model"),
            backend=embed_cfg.get("backend", "auto"),
            device=embed_cfg.get("device", "cpu"),
            nim_config=nim_cfg,
        )
        state._embed_backend = backend  # type: ignore[attr-defined]

    df = pd.read_parquet(embedded_path)
    vectors = np.stack(df["vector"].to_list())
    q_vec = backend.encode_query(query, normalize=True).astype(vectors.dtype)
    scores = vectors @ q_vec  # cosine since both are normalised
    top_idx = np.argsort(-scores)[:k]
    out = df.iloc[top_idx][["id", "domain_id", "lang", "title", "url"]].copy()
    out.insert(0, "score", [float(scores[i]) for i in top_idx])
    return out


def _distributions_panel(gr, state: VisualizerState):
    """Preview the real PX-Web count tables feeding the persona PGM.

    Uses :class:`packages.personas.pgm.distributions.PxWebDistributions`
    — these are the same DataFrames the SDG-PGMs ``fill_na_and_gen_cpd``
    consumes when building the cascaded Bayesian network. Looking at
    them is the easiest way to sanity-check why a generated persona
    looks the way it does.
    """
    try:
        dist = get_distributions()
    except FileNotFoundError as exc:
        gr.Markdown(
            "## Persona-PGM distribution tables\n\n"
            f"_Couldn't load PX-Web distributions: {exc}._\n\n"
            "Run `personas-vn curate --only download` first to populate "
            "`data/nso-gov-vn/raw/pxweb/vi/`."
        )
        return

    summary_rows = [{"distribution": k, "rows": v} for k, v in dist.summary().items()]
    summary = pd.DataFrame(summary_rows)
    gr.Markdown(
        "## Persona-PGM distribution tables\n"
        "Real NSO PX-Web tables loaded from `data/nso-gov-vn/raw/pxweb/vi/`. "
        "These feed the SDG-PGMs `PGMGenerator.fill_na_and_gen_cpd` calls "
        "inside `VNPersonaGenerator.get_cpds`."
    )
    summary_table = gr.Dataframe(value=summary, label="Distribution catalogue",
                                  interactive=False, wrap=True, max_height=380)

    distribution_names = list(dist.summary().keys())
    default_name = "region_counts" if "region_counts" in distribution_names else distribution_names[0]
    selector = gr.Dropdown(
        choices=distribution_names,
        value=default_name,
        label="Inspect distribution",
    )
    body = gr.Dataframe(value=getattr(dist, default_name),
                         wrap=True, interactive=False, max_height=520)

    def show(name: str):
        try:
            return getattr(dist, name)
        except AttributeError:
            return pd.DataFrame([{"error": f"unknown distribution: {name!r}"}])

    selector.change(show, inputs=selector, outputs=body)
    return summary_table, selector, body


# ---------------------------------------------------------------------------
# Top-level builders
# ---------------------------------------------------------------------------
def build_app(visualizer_config: str = "configs/visualizer.yaml") -> Any:
    gr = _import_gradio()
    cfg = load_config(visualizer_config)
    state = load_state(visualizer_config=visualizer_config)

    title = cfg.ui.get("title", "Personas-VN")

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"# {title}\n"
            "Vietnamese National Statistics Office data → ontology → "
            "synthetic personas. Built with `httpx`, `pydantic`, `pandas`, "
            "and Gradio."
        )

        with gr.Tab("Overview"):
            _overview_panel(gr, state)
        with gr.Tab("Ontology"):
            _ontology_panel(gr, state)
        with gr.Tab("PX-Web tables"):
            _pxweb_tables_panel(gr, state)
        with gr.Tab("Personas"):
            _personas_panel(gr, state)
        with gr.Tab("Persona embeddings"):
            _persona_embedding_panel(gr, state)
        with gr.Tab("Curator"):
            _curator_panel(gr, state)
        with gr.Tab("Distributions"):
            _distributions_panel(gr, state)

        gr.Markdown(
            "_Personas are synthetic and grounded in NSO-published "
            "aggregate statistics; they are not real individuals._"
        )

    return demo


def _ensure_localhost_bypasses_proxy() -> None:
    """Make sure 127.0.0.1/localhost are in NO_PROXY.

    httpx (which Gradio uses internally to verify the server is reachable
    after launch) defaults to ``trust_env=True``. On corp networks where
    HTTP_PROXY/HTTPS_PROXY is exported, the loopback probe gets routed
    through the proxy and fails — Gradio then raises a misleading
    "localhost is not accessible" ValueError even though the server is up.
    """
    needed = ("127.0.0.1", "localhost", "::1")
    for var in ("NO_PROXY", "no_proxy"):
        existing = {
            p.strip() for p in os.environ.get(var, "").split(",") if p.strip()
        }
        os.environ[var] = ",".join(sorted(existing | set(needed)))


def _patch_gradio_url_ok() -> None:
    """Replace ``gradio.networking.url_ok`` with a more forgiving probe.

    Gradio 6.14's check only accepts ``{200, 401, 302, 303, 307}`` from a
    ``httpx.head`` call that does not follow redirects, and only catches
    ``ConnectionError`` / ``httpx.ConnectError`` / ``httpx.TimeoutException``.
    On some setups (macOS + IPv6, or HEAD on the SSR root returning 308),
    that yields False even though the server is fully serving requests —
    causing ``launch()`` to raise the misleading "set share=True" error.
    The patched probe is functionally equivalent on healthy setups but
    treats any non-5xx response as "up" and falls back to GET if HEAD
    misbehaves.
    """
    try:
        import time

        import httpx
        from gradio import networking as gradio_networking
    except Exception:
        return

    def _url_ok(url: str) -> bool:
        for _ in range(5):
            for method in ("HEAD", "GET"):
                try:
                    r = httpx.request(
                        method,
                        url,
                        timeout=3,
                        verify=False,
                        follow_redirects=True,
                    )
                except httpx.RequestError:
                    continue
                if r.status_code < 500:
                    return True
            time.sleep(0.5)
        return False

    gradio_networking.url_ok = _url_ok


def launch(visualizer_config: str = "configs/visualizer.yaml") -> None:
    _ensure_localhost_bypasses_proxy()
    _patch_gradio_url_ok()
    gr = _import_gradio()
    cfg = load_config(visualizer_config)
    server = cfg.get("server", {})
    theme_name = cfg.ui.get("theme", "soft")
    theme_map = {
        "default":    gr.themes.Default(),
        "soft":       gr.themes.Soft(),
        "monochrome": gr.themes.Monochrome(),
        # "glass" was removed/renamed in Gradio 6; fall back gracefully.
        "glass":      gr.themes.Soft(),
    }
    demo = build_app(visualizer_config)
    # Forward only the launch kwargs Gradio still accepts. We inspect the
    # signature up-front instead of try/except'ing around demo.launch(...) —
    # a failed launch still binds the port, and the retry then crashes with
    # a misleading "localhost not accessible" error.
    launch_kwargs: dict[str, Any] = {
        "server_name": str(server.get("host", "127.0.0.1")),
        "server_port": int(server.get("port", 7860)),
        "share":       bool(server.get("share", False)),
        "theme":       theme_map.get(theme_name, gr.themes.Soft()),
    }
    launch_params = inspect.signature(demo.launch).parameters
    if "show_api" in launch_params:
        launch_kwargs["show_api"] = bool(server.get("show_api", False))
    demo.launch(**launch_kwargs)
