"""Path helpers anchored at the repository root.

Anything that wants to read or write project-relative files goes through
``resolve()`` so the same code works whether you invoke `python -m
packages.pipeline.cli` from the repo root, from a sibling directory, or via
`pytest`.
"""

from __future__ import annotations

from pathlib import Path

# packages/common/paths.py -> packages/common -> packages -> <repo root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Return ``path`` resolved against the repository root.

    Absolute paths pass through untouched.
    """
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    """``mkdir -p`` and return the resolved path."""
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
