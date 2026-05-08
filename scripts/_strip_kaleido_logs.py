"""Strip ``RichHandler`` INFO log spam from notebook cell outputs.

Why this script exists
----------------------
``packages/common/logging.py`` configures the root logger with a
``rich.logging.RichHandler`` at INFO level, so every ``logging.INFO``
record emitted while a notebook executes gets captured by Jupyter as a
``display_data`` output and persisted in the .ipynb. Three big offenders:

1. ``kaleido`` (>= 1.0, used by ``fig.write_image``) and its
   ``choreographer`` headless-Chrome backend fire ~10 INFO lines per
   figure ("Chromium init'ed", "Conforming 1 to file:///.../tmp.../
   index.html", "Getting tab from queue", "Got DF4E", "Putting tab DF4E
   back", "Closing browser.", ...).
2. ``httpx`` / ``urllib3`` log every HTTP request at INFO.
3. Any package-internal logger from ``packages.*`` that uses
   ``get_logger(__name__)``.

Across the analysis + viz + HuggingFace notebooks that's ~100+ INFO
outputs per file — which makes diffs unreadable, bloats the file, and
clutters the dataset-card preview on HuggingFace.

``packages/common/logging.py`` now mutes the worst third-party loggers
(``kaleido``, ``choreographer``, ``logistro``, ``httpx``, ``httpcore``,
``urllib3``, ``asyncio``) at WARNING level inside ``_configure_root``,
so fresh kernel runs produce clean notebooks. This script is the
*retrofit*: it scrubs INFO outputs already saved in the .ipynb files.

The strip predicate matches *any* ``RichHandler``-formatted INFO line
(``[timestamp] INFO    <message>``) inside a ``display_data`` output
and drops the entire output. ``stream`` outputs from ``print`` calls
and other non-INFO outputs (figures, tables, error tracebacks) are
preserved.

Usage::

    python -m scripts._strip_kaleido_logs                     # default 3 NBs
    python -m scripts._strip_kaleido_logs path/to/some.ipynb  # explicit list
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Matches the distinctive ``[timestamp] INFO     <message>`` rich-handler
# prefix that's used for every INFO log routed through
# ``rich.logging.RichHandler`` — including kaleido, choreographer, httpx,
# and any package-internal logger that goes through
# ``packages.common.logging.get_logger``. Dropping all of them is what
# the user wants ("strip INFO*"); they're never useful inside a saved
# notebook and they bloat the file.
#
# The format produced by ``RichHandler`` (with ``show_time=True``) is::
#
#     [DD/MM/YY HH:MM:SS] INFO    <four+ spaces><message>
#
# When the same record is rendered later in the notebook (no new
# timestamp) the leading ``[…]`` block is replaced by whitespace::
#
#                         INFO    <four+ spaces><message>
#
# So matching ``INFO`` followed by 2+ whitespace characters and at least
# one more printable character is enough to identify a rich INFO log
# line. We strip ANSI CSI escapes first because ``RichHandler`` wraps
# the level word and any embedded numbers in ``\x1b[34m…\x1b[0m`` /
# ``\x1b[1m…\x1b[0m``, which would otherwise break the literal match.
_INFO_LINE_RE = re.compile(r"\bINFO\s{2,}\S")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def _output_is_noise(output: dict) -> bool:
    """Return True iff ``output`` is a kaleido/choreographer log line.

    rich.RichHandler emits these as ``display_data`` outputs with both a
    pre-formatted ``text/html`` block and an ANSI ``text/plain`` block.
    The text/plain side is the simplest to match against — it preserves
    the literal log message verbatim with the rich box-drawing
    characters and ANSI codes intact.
    """
    if output.get("output_type") != "display_data":
        return False
    data = output.get("data", {}) or {}
    plain = data.get("text/plain")
    if isinstance(plain, list):
        plain = "".join(plain)
    if not isinstance(plain, str):
        return False
    return bool(_INFO_LINE_RE.search(_ANSI_RE.sub("", plain)))


def strip(nb_path: Path) -> tuple[int, int]:
    """Remove kaleido/choreographer log outputs from ``nb_path`` in place.

    Returns ``(n_outputs_dropped, n_cells_touched)``.
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    n_dropped = 0
    cells_touched = 0
    for cell in nb.get("cells", []):
        outs = cell.get("outputs")
        if not outs:
            continue
        kept = [o for o in outs if not _output_is_noise(o)]
        if len(kept) != len(outs):
            n_dropped += len(outs) - len(kept)
            cells_touched += 1
            cell["outputs"] = kept
    if n_dropped:
        # Preserve nbformat's idiomatic 1-space JSON + trailing newline so
        # the diff is purely the dropped-output lines, not a reflow.
        nb_path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return n_dropped, cells_touched


_DEFAULT_TARGETS = [
    REPO / "DATAANALYSIS.ipynb",
    REPO / "DATAVISUALIZATION.ipynb",
    REPO / "data" / "nso-gov-vn" / "_hf" / "notebook.ipynb",
]


def main(argv: list[str]) -> int:
    targets = [Path(p).resolve() for p in argv[1:]] or _DEFAULT_TARGETS
    total_dropped = 0
    for nb_path in targets:
        if not nb_path.exists():
            print(f"  skip   {nb_path.relative_to(REPO)}  (not found)")
            continue
        dropped, cells = strip(nb_path)
        total_dropped += dropped
        rel = nb_path.relative_to(REPO) if nb_path.is_relative_to(REPO) else nb_path
        if dropped:
            print(f"  clean  {rel}: dropped {dropped} log outputs across {cells} cells")
        else:
            print(f"  ok     {rel}: already clean")
    print(f"total dropped outputs: {total_dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
