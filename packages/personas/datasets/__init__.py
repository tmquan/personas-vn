"""Nemotron-Personas-Vietnam parquet dataset packaging.

Builds the four published parquet datasets from the structured persona
generator (:mod:`packages.personas.pgm`) plus the OCEAN sampler
(:mod:`packages.personas.ocean`) plus the narrative renderer
(templated or LLM A+B from :mod:`packages.personas.llm`):

* ``Nemotron-Personas-Vietnam-large-vi.parquet`` — 3 M rows, Vietnamese
* ``Nemotron-Personas-Vietnam-large-en.parquet`` — 3 M rows, English
* ``Nemotron-Personas-Vietnam-small-vi.parquet`` — 300 k stratified subset
* ``Nemotron-Personas-Vietnam-small-en.parquet`` — 300 k same-uuid mirror

The 22-column schema mirrors the published ``nvidia/Nemotron-Personas-Japan``
with Vietnam-specific geographic columns (``region`` / ``area`` /
``province``).

Public entry point:

    from packages.personas.datasets import build_nemotron_personas_vietnam_datasets
    build_nemotron_personas_vietnam_datasets(out_dir="data/Nemotron-Personas-Vietnam",
                               large_size=3_000_000,
                               small_size=300_000,
                               seed=42)

Pre-unification this content lived in :mod:`packages.personas.datasets`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from packages.personas.datasets.schema import (
    NEMOTRON_PERSONAS_VIETNAM_COLUMNS,
    NEMOTRON_PERSONAS_VIETNAM_FEATURES,
    PYARROW_SCHEMA,
)

# ``builder`` pulls in pgmpy + torch (via SDG-PGMs); expose its public
# symbols lazily so ``from packages.personas.datasets import
# NEMOTRON_PERSONAS_VIETNAM_COLUMNS`` stays cheap and import-error-free even when the
# curator extras aren't installed.

if TYPE_CHECKING:  # pragma: no cover
    from packages.personas.datasets.builder import (
        BuilderConfig,
        build_nemotron_personas_vietnam_datasets,
    )

__all__ = [
    "BuilderConfig",
    "build_nemotron_personas_vietnam_datasets",
    "NEMOTRON_PERSONAS_VIETNAM_COLUMNS",
    "NEMOTRON_PERSONAS_VIETNAM_FEATURES",
    "PYARROW_SCHEMA",
]


def __getattr__(name: str) -> Any:
    if name in {"BuilderConfig", "build_nemotron_personas_vietnam_datasets"}:
        from packages.personas.datasets import builder

        return getattr(builder, name)
    raise AttributeError(
        f"module 'packages.personas.datasets' has no attribute {name!r}"
    )
