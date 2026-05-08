"""Persona embedding + 2-D projection.

Embeds the bilingual persona bios (``bio_vi``/``bio_en``) using
sentence-transformers locally or NVIDIA NIM embedding models, then
runs UMAP + HDBSCAN to project to 2-D and discover density-based
clusters for the visualizer.

Public surface:

* :class:`PersonaEmbedConfig`, :func:`embed_personas`,
  :func:`embed_personas_from_disk` — the embedding + reduce pipeline.
* :class:`EmbeddingBackend`, :class:`SentenceTransformersBackend`,
  :class:`NimEmbeddingsBackend`, :class:`NimEmbeddingsConfig`,
  :func:`make_embedding_backend`, :data:`LOCAL_DEFAULT_MODEL`,
  :data:`NIM_EMBEDDING_MODELS` — configurable backend abstraction
  shared with the curator pipeline.

This sub-package was previously
:mod:`packages.personas.embed.pipeline` / :mod:`packages.personas.embed.backends`
(pre-unification).
"""

from __future__ import annotations

from packages.personas.embed.backends import (
    LOCAL_DEFAULT_MODEL,
    NIM_EMBEDDING_MODELS,
    EmbeddingBackend,
    NimEmbeddingsBackend,
    NimEmbeddingsConfig,
    SentenceTransformersBackend,
    make_embedding_backend,
)
from packages.personas.embed.pipeline import (
    PersonaEmbedConfig,
    embed_personas,
    embed_personas_from_disk,
)

__all__ = [
    "EmbeddingBackend",
    "LOCAL_DEFAULT_MODEL",
    "NIM_EMBEDDING_MODELS",
    "NimEmbeddingsBackend",
    "NimEmbeddingsConfig",
    "SentenceTransformersBackend",
    "make_embedding_backend",
    "PersonaEmbedConfig",
    "embed_personas",
    "embed_personas_from_disk",
]
