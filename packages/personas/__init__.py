"""Vietnamese personas — unified PGM + OCEAN + LLM + dataset packaging.

The ``personas`` package is the single home for synthetic-Vietnamese-persona
generation. It replaces the older split between ``personagen`` (PGM,
embedding, single-bio enrichment) and ``datagen`` (Nemotron parquet
packaging, OCEAN, LLM A/B). One layered package, four sub-packages plus
:mod:`packages.personas.ocean`:

* :mod:`packages.personas.pgm`      — SDG-PGMs Bayesian-network sampler
  for the structured demographic columns (region, age, sex, education,
  occupation, …) grounded in real NSO PX-Web statistics.
* :mod:`packages.personas.ocean`    — Big Five (OCEAN) personality
  sampler conditioned on age × sex, grounded in peer-reviewed
  psychology priors (Roberts 2006, Schmitt 2008).
* :mod:`packages.personas.embed`    — sentence-transformers / NIM
  embedding + UMAP projection for the typed ``PersonaBatch``.
* :mod:`packages.personas.llm`      — unified OpenAI-compatible LLM
  layer: HTTP client + JSONL cache + retry, plus the Data Designer
  LLM-A/LLM-B narrative pipeline and the single-bio
  ``enrich_personas`` flow (both routed through the same client).
* :mod:`packages.personas.datasets` — Nemotron-Personas-Vietnam parquet
  builder (4 outputs: large/small × vi/en, 22 columns each).

Layered relationship: ``datasets`` consumes ``pgm`` + ``ocean`` + ``llm``
to produce HuggingFace-style parquets; ``embed`` operates on the
``personagen``-flavour ``PersonaBatch`` (``data/personas/personas.json``).

Pre-unification (Dec 2025): ``packages/personagen/`` and
``packages/datagen/``. Both names are gone.
"""

from __future__ import annotations

# Re-export the most-used public surface so callers can write
# ``from packages.personas import VNPersonaGenerator, build_nemotron_personas_vietnam_datasets``
# without juggling sub-package paths.
from packages.personas.pgm import (
    Edge,
    PGMGenerator,
    PxWebDistributions,
    TarFileMixin,
    VNPersonaData,
    VNPersonaGenerator,
    generate_personas,
)

__all__ = [
    "Edge",
    "PGMGenerator",
    "PxWebDistributions",
    "TarFileMixin",
    "VNPersonaData",
    "VNPersonaGenerator",
    "generate_personas",
]
