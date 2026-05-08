"""Lightweight logging wrapper.

We use ``rich.logging.RichHandler`` when available so CLI output looks nice,
and fall back to ``logging.StreamHandler`` so tests / minimal installs still
work.
"""

from __future__ import annotations

import logging
import os
from functools import cache

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Third-party libraries whose INFO chatter is irrelevant to ``personas-vn``
# users and pollutes notebook outputs (kaleido + choreographer fire ~10
# INFO lines on every ``fig.write_image`` call; httpx logs every HTTP
# request at INFO; urllib3 narrates each retry; etc.). We pin them to
# WARNING here, in the *single* place every package log goes through, so
# any caller of :func:`get_logger` automatically gets a quiet stack
# without having to remember module-level ``setLevel`` boilerplate.
_NOISY_LIBS = (
    "kaleido",       # plotly image-export logs each tab/temp-dir/browser op
    "choreographer", # headless-Chrome backend used by kaleido
    "logistro",      # logistro is the wrapper kaleido/choreographer use
    "httpx",         # noisy at INFO ("HTTP Request: GET ...")
    "httpcore",
    "urllib3",
    "asyncio",       # "Using selector: KqueueSelector" on macOS
)


@cache
def _configure_root() -> None:
    level_name = os.getenv("PERSONAS_VN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        fmt = "%(message)s"
    except Exception:
        handler = logging.StreamHandler()
        fmt = _DEFAULT_FORMAT

    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # Mute noisy third-party libraries (see ``_NOISY_LIBS`` above). Setting
    # an explicit level on the named logger is enough: child loggers (e.g.
    # ``kaleido._page_generator``, ``choreographer.browser_sync``) inherit
    # this level via Python's effective-level traversal, so a single
    # ``WARNING`` on the package root mutes the whole tree.
    for _lib in _NOISY_LIBS:
        logging.getLogger(_lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    _configure_root()
    return logging.getLogger(name)
