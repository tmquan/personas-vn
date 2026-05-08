"""NVIDIA brand-aligned Plotly theme.

Strictly follows NVIDIA's logo & brand guidelines
(https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/):

* **Colour palette** — NVIDIA Green (`#76B900`) on a clean white background,
  with black for text and a small set of greys for axis chrome / muted
  fills. We never recolour the green wordmark; brand green is reserved
  for the *primary* data series in every chart.
* **Typography** — NVIDIA Sans is the proprietary brand font; we use a
  fallback chain (`NVIDIA Sans, NVIDIASans, NVIDIA, Inter, Helvetica
  Neue, Arial, sans-serif`) so the chart looks correct on machines with
  the licensed font installed and gracefully degrades on machines
  without it.
* **Logo** — only the NVIDIA wordmark may be used on NVIDIA artefacts;
  we don't draw it on the figures themselves to avoid implying
  endorsement. The helper is named after the brand convention but the
  produced charts are clearly third-party analyses of NSO data.

The module exposes:

* :func:`apply_nvidia_style(fig)` — in-place styling for any Plotly fig.
* :func:`save_figure(fig, name)` — writes both an interactive HTML and
  a PNG snapshot under ``docs/figures/analysis/`` for embedding in
  ``DATAANALYSIS.md``.
* Constants ``NV_GREEN``, ``NV_BLACK``, ``NV_WHITE``, etc. for ad-hoc
  use inside notebook cells.

Usage in a notebook cell::

    import plotly.express as px
    from scripts._nvidia_style import apply_nvidia_style, save_figure, NV_GREEN
    fig = px.bar(df, x="region", y="personas")
    fig.update_traces(marker_color=NV_GREEN)
    apply_nvidia_style(fig)
    save_figure(fig, "01_marginal_region")
    fig.show()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from packages.common.paths import REPO_ROOT, ensure_dir

# kaleido >= 1 + its choreographer (headless-Chrome) backend log a steady
# stream of INFO lines on every `fig.write_image` call ("Chromium init'ed",
# "Conforming 1 to file:///.../tmp.../index.html", "Getting tab from queue",
# "Got DF4E", "Putting tab DF4E back", "Closing browser.", ...). When the
# notebooks are executed via `jupyter nbconvert --execute` those rich-
# formatted lines get captured as display_data outputs and pollute the
# .ipynb (~100 noise lines per notebook). Mute them here so every consumer
# of `save_figure` gets a quiet kaleido by default.
for _lib in ("kaleido", "choreographer", "logistro"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Brand palette (NVIDIA logo & brand guidelines)
# ---------------------------------------------------------------------------
NV_GREEN = "#76B900"        # the NVIDIA wordmark green — reserved for the
                            # primary data series in every figure
NV_GREEN_DARK = "#5C9300"   # press-release "deep green" for hovers / focus
NV_GREEN_SOFT = "#B5D88A"   # 50% tint for fills / area under curve
NV_BLACK = "#000000"        # body text + axis tick labels
NV_DARK = "#1A1A1A"         # near-black for chart titles
NV_GREY = "#666666"         # secondary text / faint chrome
NV_LIGHT_GREY = "#CCCCCC"   # axis lines, table borders
NV_FAINT = "#F5F5F5"        # background fill for separators
NV_WHITE = "#FFFFFF"        # canvas

# Discrete sequence for multi-series plots: green leads, then a curated
# greyscale ramp + a single accent black. Avoids the rainbow defaults of
# Plotly Express (off-brand for NVIDIA).
NV_DISCRETE = [
    NV_GREEN,
    NV_BLACK,
    "#888888",
    NV_GREEN_SOFT,
    "#444444",
    NV_LIGHT_GREY,
]
# Continuous (sequential) scale for heatmaps / choropleths: faint
# green-tinted near-white → NVIDIA green. The bottom stop is **not**
# pure ``#FFFFFF`` because choropleth-fill against a white paper
# background would render low-value provinces as invisible (only the
# polygon border showing — easy to misread as "missing data"). A faint
# but visible ``#ECF5DC`` keeps the lowest-bin polygons distinguishable
# from the canvas while still leaving plenty of contrast headroom for
# the mid- and high-end stops. Looks much more on-brand than the
# default Viridis.
NV_SEQUENTIAL = [
    [0.00, "#ECF5DC"],
    [0.25, "#D4E8B0"],
    [0.50, NV_GREEN_SOFT],
    [0.75, NV_GREEN],
    [1.00, NV_GREEN_DARK],
]

# Font fallback chain — NVIDIA Sans is proprietary; we list it first plus
# the closest free fallbacks so a viewer who has it installed sees the
# real brand face.
NV_FONT_FAMILY = (
    "NVIDIA Sans, NVIDIASans, NVIDIA, Inter, "
    "Helvetica Neue, Helvetica, Arial, sans-serif"
)

# Where rendered PNG / HTML pairs land. Used by ``save_figure``.
FIGURE_DIR = ensure_dir(REPO_ROOT / "docs" / "figures" / "analysis")


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------
_AXIS_COMMON: dict[str, Any] = dict(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.06)",
    zeroline=True,
    zerolinecolor=NV_LIGHT_GREY,
    linecolor=NV_BLACK,
    linewidth=1,
    ticks="outside",
    tickcolor=NV_BLACK,
    tickfont=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12),
    title_font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=13),
    showline=True,
    mirror=False,
    # Crucial — Plotly auto-extends the layout margin so long Vietnamese
    # category labels ("Đồng bằng sông Hồng", "Bắc Trung Bộ và Duyên hải
    # miền Trung", ...) are never clipped. Without this every horizontal
    # bar chart needs a hand-tuned `margin=dict(l=…)` that's brittle.
    automargin=True,
)


def apply_nvidia_style(fig, *, axes: bool = True) -> Any:
    """Restyle ``fig`` in place so it follows NVIDIA's brand guidelines.

    Parameters
    ----------
    fig:
        A Plotly ``Figure`` (or anything ``update_layout`` accepts).
    axes:
        ``True`` shows axes with the NVIDIA grid + tick treatment;
        ``False`` hides them entirely (graphs / network plots).
    """
    fig.update_layout(
        paper_bgcolor=NV_WHITE,
        plot_bgcolor=NV_WHITE,
        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=13),
        title=dict(
            font=dict(family=NV_FONT_FAMILY, color=NV_DARK, size=18),
            # Small left offset so the first character isn't flush with the
            # paper edge — matches NVIDIA editorial layouts.
            x=0.02, xanchor="left", y=0.96,
        ),
        legend=dict(
            font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=NV_LIGHT_GREY,
            borderwidth=1,
        ),
        colorway=NV_DISCRETE,
        # Generous defaults that the per-axis automargin can grow further.
        # The right margin gets 80px so `textposition='outside'` labels on
        # horizontal-bar charts always have room to render.
        margin=dict(l=120, r=80, t=80, b=80),
        hoverlabel=dict(
            font=dict(family=NV_FONT_FAMILY, color=NV_WHITE, size=12),
            bgcolor=NV_BLACK,
            bordercolor=NV_GREEN,
        ),
    )
    if axes:
        fig.update_xaxes(**_AXIS_COMMON)
        fig.update_yaxes(**_AXIS_COMMON)
    else:
        for ax in (fig.update_xaxes, fig.update_yaxes):
            ax(showgrid=False, zeroline=False, showticklabels=False,
               showline=False, ticks="")
    # `textposition='outside'` on horizontal bars otherwise gets clipped at
    # the chart edge for the longest-value bar — disable axis-clipping for
    # bar traces so the value annotations always render fully.
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def _in_jupyter_kernel() -> bool:
    """Return ``True`` iff this code is executing inside a Jupyter kernel.

    Used by :func:`save_figure` to gate the Mapbox-GL auto-skip
    heuristic: kaleido + Mapbox-GL fails when the kernel is the parent
    of kaleido's headless Chromium subprocess (the well-known
    ``Error 525: Mapbox error``), but works fine when kaleido is invoked
    from a fresh standalone Python process — which is exactly what
    :mod:`scripts.render_maps` does. So we only auto-skip in the failing
    environment.

    Detection strategy: ``IPython.get_ipython().__class__.__name__``
    returns ``"ZMQInteractiveShell"`` inside a Jupyter kernel,
    ``"TerminalInteractiveShell"`` in plain ``ipython`` at the terminal,
    and the function returns ``None`` outside any IPython context. This
    is the canonical detection pattern used by the IPython project
    itself.
    """
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        if ipy is None:
            return False
        return type(ipy).__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def save_figure(
    fig,
    name: str,
    *,
    width: int = 1100,
    height: int = 600,
    scale: int = 2,
    write_png: bool = True,
    write_html: bool = True,
    out_dir: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Persist a styled Plotly figure to PNG (and HTML by default).

    The PNG is what ``DATAANALYSIS.md`` embeds; the HTML is what users
    open for an interactive version. Returns
    ``(png_path or None, html_path or None)``.

    ``name`` is the filename stem (no extension); the convention used by
    the analysis notebook is ``"NN_short_label"``.

    Parameters
    ----------
    write_png:
        Whether to render the static PNG via ``fig.write_image`` (kaleido).
        Set to ``False`` for figures that use ``Choroplethmapbox`` /
        ``Scattermapbox`` traces inside a Jupyter kernel — kaleido's
        headless Chromium loses its Mapbox-GL worker in that environment
        and raises ``KaleidoError: Error 525: Mapbox error``. The
        standalone :mod:`scripts.render_maps` process is the intended
        path for rendering PNGs of those figures.
    write_html:
        Whether to also write a self-contained interactive HTML alongside.

    Notes
    -----
    Two layers of protection against the kaleido + Mapbox-GL crash:

    1. *Auto-detection (Jupyter only)* — if any trace in ``fig.data`` is
       of a Mapbox / MapLibre family (``choroplethmapbox``,
       ``scattermapbox``, ``densitymapbox``, or the newer
       ``choroplethmap`` / ``scattermap`` / ``densitymap`` MapLibre
       variants) AND we're running inside a Jupyter kernel
       (``ZMQInteractiveShell``), PNG export is skipped automatically
       even when the caller forgot to pass ``write_png=False``. Saves the
       wasted kaleido / headless-Chrome boot and surfaces a clear
       warning instead of an opaque ``KaleidoError 525``. The Jupyter
       guard is essential — kaleido + Mapbox-GL works fine in a
       standalone Python process (that's how
       :mod:`scripts.render_maps` produces the canonical PNGs).
    2. *Try/except* — any other kaleido failure (e.g. ``DeprecationWarning``-
       turned-error from a future plotly version) downgrades to a warning
       and still writes the HTML companion. Cells keep executing.
    """
    from packages.common.logging import get_logger

    log = get_logger(__name__)

    target_dir = ensure_dir(Path(out_dir) if out_dir else FIGURE_DIR)

    # Layer 1 — auto-detect Mapbox / MapLibre traces and skip PNG export
    # *only when running inside a Jupyter kernel*. ``fig.data`` is a
    # tuple of plotly graph objects; each has a ``.type`` string like
    # ``"choroplethmapbox"``. We deliberately match a *prefix* set
    # rather than exact strings so future plotly variants (e.g. a
    # hypothetical ``choroplethmapbox3d``) are caught too.
    #
    # The Jupyter-only guard matters because kaleido + Mapbox-GL DOES
    # work in a fresh standalone Python process (that's exactly what
    # ``scripts/render_maps.py`` relies on); it ONLY fails when the
    # kernel is the parent of kaleido's headless Chromium subprocess.
    # Without this guard the standalone renderer would also auto-skip
    # PNGs and never produce the artefacts ``DATAVISUALIZATION.md``
    # embeds.
    _MAPBOX_PREFIXES = (
        "choroplethmapbox", "scattermapbox", "densitymapbox",  # plotly < 5.24 (Mapbox-GL)
        "choroplethmap",   "scattermap",   "densitymap",       # plotly >= 5.24 (MapLibre)
    )
    has_mapbox = any(
        (getattr(t, "type", "") or "").lower().startswith(_MAPBOX_PREFIXES)
        for t in (getattr(fig, "data", None) or ())
    )
    if has_mapbox and write_png and _in_jupyter_kernel():
        log.warning(
            "%s contains Mapbox/MapLibre trace(s) and we're running inside "
            "a Jupyter kernel — skipping PNG export (kaleido's headless "
            "Chromium can't render Mapbox-GL in this environment; run "
            "`python -m scripts.render_maps` for PNGs)",
            name,
        )
        write_png = False

    png_path: Path | None = None
    if write_png:
        png_path = target_dir / f"{name}.png"
        # Layer 2 — defensive try/except around the kaleido call. Any
        # failure (Mapbox 525 leaking past the auto-detect, kaleido
        # subprocess crash, ...) is logged and swallowed so the cell
        # still produces the interactive HTML via ``fig.show()``.
        try:
            fig.write_image(png_path, width=width, height=height, scale=scale)
        except Exception as exc:
            log.warning(
                "PNG export failed for %s (%s: %s) — keeping HTML only",
                name, type(exc).__name__, exc,
            )
            png_path = None
    html_path = target_dir / f"{name}.html" if write_html else None
    if html_path is not None:
        fig.write_html(html_path, include_plotlyjs="cdn")
    return png_path, html_path
