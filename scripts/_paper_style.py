"""Shared paper-style matplotlib helper for the analysis scripts.

We render the static PNG figures with matplotlib (rather than Plotly +
Kaleido) because Kaleido's image-export path is unreliable on macOS
arm64 — it either depends on a headless Chrome launched by
`choreographer` (kaleido >= 1) or segfaults out of the box (kaleido <
1). matplotlib has none of those issues and produces nicer
publication-style figures with native serif fonts.

Plotly is still used for *interactive* HTML output (e.g. the
geographic-choropleth).
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

_PAPER_FONT_FAMILY = [
    "Computer Modern Serif", "CMU Serif", "Latin Modern Roman",
    "STIX Two Text", "Times New Roman", "DejaVu Serif", "serif",
]


def use_paper_style() -> None:
    """Install global matplotlib rc params: white bg + serif font + faint grid."""
    mpl.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "savefig.edgecolor":  "white",
        "savefig.dpi":        160,
        "font.family":        "serif",
        "font.serif":         _PAPER_FONT_FAMILY,
        "font.size":          12,
        "axes.titlesize":     14,
        "axes.titleweight":   "normal",
        "axes.labelsize":     12,
        "axes.edgecolor":     "black",
        "axes.linewidth":     1.0,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.grid":          True,
        "grid.color":         "#000000",
        "grid.alpha":         0.18,
        "grid.linewidth":     0.6,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.color":        "black",
        "ytick.color":        "black",
        "legend.frameon":     True,
        "legend.edgecolor":   "black",
        "legend.fontsize":    11,
    })


@contextmanager
def paper_axes(figsize=(9.0, 5.0), title: str | None = None,
                xlabel: str | None = None, ylabel: str | None = None):
    """Context manager: ``with paper_axes(...) as ax:`` and draw."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    try:
        yield ax
    finally:
        fig.tight_layout()


def save(out_path, fig=None) -> None:
    fig = fig or plt.gcf()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
