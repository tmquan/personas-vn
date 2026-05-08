"""Persona embedding + 2-D reduction pipeline.

Why this exists separately from ``packages.curator``:

* The curator pipeline operates on NSO *source documents* — its inputs are
  raw HTML, its outputs are document embeddings.
* This module operates on *generated personas* — a much larger dataset
  (100K+) with very different texts (templated bios) and very different
  downstream uses (cluster discovery in attribute space, semantic search
  for "find me personas like X").

Backends (see :mod:`packages.personas.embed.backends`):

* ``sentence-transformers`` (default) — local multilingual SBERT, free,
  offline, 384-dim.
* ``nim`` — NVIDIA NIM hosted embeddings. Verified models:
  ``nvidia/llama-3.2-nv-embedqa-1b-v2`` (2048-dim),
  ``nvidia/llama-nemotron-embed-1b-v2``,
  ``nvidia/llama-3.2-nemoretriever-300m-embed-v1``.

The backend is auto-selected from the model name (``nvidia/...`` → NIM,
else local) but can be pinned via :attr:`PersonaEmbedConfig.backend`.

Outputs (under ``output.personas_dir`` from ``configs/ontology.yaml``):

    embedded.parquet  — id + N-d float32 vector + key categorical fields
    reduced.parquet   — id + 2-D coordinates + cluster id + key fields
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.common.logging import get_logger
from packages.common.paths import ensure_dir, resolve
from packages.ontology import PersonaBatch
from packages.ontology.registry import OntologyRegistry, load_registry
from packages.personas.embed.backends import (
    LOCAL_DEFAULT_MODEL,
    NimEmbeddingsConfig,
    make_embedding_backend,
)

log = get_logger(__name__)


@dataclass
class PersonaEmbedConfig:
    """Knobs for :func:`embed_personas`.

    ``backend="auto"`` picks NIM for any ``nvidia/...`` model and the
    local sentence-transformers backend for everything else. Explicit
    values: ``"local"`` / ``"nim"``.
    """

    model: str = LOCAL_DEFAULT_MODEL
    backend: str = "auto"
    device: str = "cpu"  # only used by the local backend
    batch_size: int = 64
    normalize: bool = True
    # NIM-only knobs (ignored by the local backend).
    base_url: str | None = None
    api_key_env: str = "PERSONAS_VN_LLM_API_KEY"
    truncate: str = "END"
    request_delay_s: float = 0.0
    # If a batch is huge (e.g. 100K) embedding on CPU is slow; set this to
    # ~10000 to embed a representative sample. Use ``None`` for "all".
    max_records: int | None = 10000
    # UMAP knobs
    n_neighbors: int = 25
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 20260505
    # HDBSCAN clustering — opt-in. Off by default so the pipeline only
    # produces what every consumer needs (the 2-D UMAP coords); flip to
    # ``True`` to additionally compute density-based cluster labels and
    # write them to ``reduced.parquet``'s ``cluster`` column.
    cluster: bool = False
    # Density-based clustering knobs — only consulted when
    # ``cluster=True``. HDBSCAN picks cluster count from the data and
    # labels low-density points as noise (-1).
    min_cluster_size: int | None = None  # None = auto: max(5, n // 80)
    min_samples: int | None = None       # None = auto: min_cluster_size // 4


# ---------------------------------------------------------------------------
# Embedding + reduction
# ---------------------------------------------------------------------------
def embed_personas(
    batch: PersonaBatch,
    *,
    output_dir: str | Path,
    config: PersonaEmbedConfig | None = None,
) -> dict[str, Any]:
    """Embed each persona's bio, reduce to 2-D, optionally cluster, and persist.

    Clustering (HDBSCAN) is **off by default** — the pipeline only writes
    the 2-D UMAP coordinates every consumer needs. Pass
    ``PersonaEmbedConfig(cluster=True)`` to additionally compute density-
    based cluster labels and add a ``cluster`` column to
    ``reduced.parquet``.

    Returns a small summary dict suitable for stashing in a manifest.
    """
    cfg = config or PersonaEmbedConfig()
    out_dir = ensure_dir(output_dir)
    embedded_path = out_dir / "embedded.parquet"
    reduced_path = out_dir / "reduced.parquet"

    df = pd.DataFrame([p.model_dump() for p in batch.personas])
    if df.empty:
        log.warning("embed_personas: empty batch, nothing to do")
        empty = df.assign(x=[], y=[])
        if cfg.cluster:
            empty = empty.assign(cluster=[])
        empty.to_parquet(embedded_path, index=False)
        empty.to_parquet(reduced_path, index=False)
        return {"n_total": 0, "n_embedded": 0}

    # Optional sub-sampling for the embed pass — uniform random sample so
    # downstream UMAP still covers the demographic / occupational space.
    n_total = len(df)
    if cfg.max_records and n_total > cfg.max_records:
        rng = np.random.default_rng(cfg.random_state)
        sample_idx = rng.choice(n_total, size=int(cfg.max_records), replace=False)
        sample_idx.sort()
        df_to_embed = df.iloc[sample_idx].reset_index(drop=True)
        log.info("sub-sampling %d / %d personas for embedding", len(df_to_embed), n_total)
    else:
        df_to_embed = df

    # ----- embed -----
    nim_cfg = NimEmbeddingsConfig(
        base_url=(cfg.base_url or NimEmbeddingsConfig().base_url),
        api_key_env=cfg.api_key_env,
        truncate=cfg.truncate,
        request_delay_s=cfg.request_delay_s,
    )
    backend = make_embedding_backend(
        model=cfg.model, backend=cfg.backend, device=cfg.device, nim_config=nim_cfg,
    )

    # Embed the Vietnamese bio (canonical source-of-truth narrative); falls
    # back to the English mirror or, finally, an empty string if neither is
    # populated.
    if "bio_vi" in df_to_embed.columns:
        bio_series = df_to_embed["bio_vi"].fillna(df_to_embed.get("bio_en", ""))
    elif "bio_en" in df_to_embed.columns:
        bio_series = df_to_embed["bio_en"]
    else:
        bio_series = df_to_embed.get("bio", pd.Series([""] * len(df_to_embed)))
    texts = bio_series.fillna("").astype(str).to_list()
    log.info("encoding %d persona bios (backend=%s, model=%s, bs=%d)",
             len(texts), backend.name, backend.model, cfg.batch_size)
    try:
        vectors = backend.encode(
            texts,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
            input_type="passage",
            show_progress=True,
        )
    finally:
        backend.close()

    keep_cols = [
        "persona_id", "name", "bio_vi", "bio_en",
        "region", "region_en", "urbanicity", "urbanicity_en",
        "age_group", "age", "sex", "sex_en",
        "ethnicity", "ethnicity_en", "marital_status", "marital_status_en",
        "education_level", "education_level_en",
        "employment_status", "employment_status_en",
        "occupation", "occupation_en", "industry_sector",
        "income_quintile",
    ]
    keep_cols = [c for c in keep_cols if c in df_to_embed.columns]
    embedded = df_to_embed[keep_cols].copy()
    embedded["vector"] = list(vectors)
    embedded.to_parquet(embedded_path, index=False)
    log.info("wrote %s (n=%d, dim=%d)", embedded_path, len(embedded), vectors.shape[1])

    # ----- reduce -----
    coords = _reduce_2d(vectors, cfg)
    reduced = embedded.drop(columns=["vector"]).copy()
    reduced["x"] = coords[:, 0]
    reduced["y"] = coords[:, 1]
    if cfg.cluster:
        clusters = _cluster(vectors, cfg)
        reduced["cluster"] = clusters
        log.info("wrote %s (n=%d, clusters=%d)", reduced_path, len(reduced), len(set(clusters)))
    else:
        log.info("wrote %s (n=%d, clustering=off)", reduced_path, len(reduced))
    reduced.to_parquet(reduced_path, index=False)

    return {
        "n_total": n_total,
        "n_embedded": len(embedded),
        "dim": int(vectors.shape[1]),
        "embedded_path": str(embedded_path),
        "reduced_path": str(reduced_path),
        "model": cfg.model,
        "backend": backend.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _reduce_2d(X: np.ndarray, cfg: PersonaEmbedConfig) -> np.ndarray:
    n = X.shape[0]
    if n >= 4:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(cfg.n_neighbors, max(2, n - 1)),
                min_dist=cfg.min_dist,
                metric=cfg.metric,
                random_state=cfg.random_state,
            )
            return reducer.fit_transform(X)
        except Exception as exc:
            log.warning("UMAP failed (%s); falling back to PCA", exc)
    from sklearn.decomposition import PCA

    return PCA(n_components=2).fit_transform(X)


def _cluster(X: np.ndarray, cfg: PersonaEmbedConfig) -> list[int]:
    """Density-based HDBSCAN clustering.

    Picks ``min_cluster_size`` from the data (``max(5, n // 80)``) so a
    10K-persona embedding gets ~125-cell clusters and a 100K-persona
    embedding gets ~1250-cell clusters, both reasonable granularities
    for downstream segmentation. Override via
    :attr:`PersonaEmbedConfig.min_cluster_size`.

    Embeddings are L2-normalised upstream so squared-euclidean ranks
    identically to cosine; sklearn's HDBSCAN doesn't ship cosine but
    euclidean works fine on the unit sphere.

    HDBSCAN labels low-density points as ``-1`` (noise) rather than
    forcing every point into a cluster — pass-through to the caller.
    """
    n = X.shape[0]
    if n < 4:
        return [0] * n
    if cfg.min_cluster_size is not None:
        min_cluster_size = max(2, int(cfg.min_cluster_size))
    else:
        min_cluster_size = max(5, n // 80)
    if cfg.min_samples is not None:
        min_samples = max(1, int(cfg.min_samples))
    else:
        min_samples = max(3, min_cluster_size // 4)
    try:
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(X).tolist()
    except Exception as exc:
        log.warning("HDBSCAN failed (%s); assigning a single cluster", exc)
        return [0] * n


# ---------------------------------------------------------------------------
# Convenience: run from disk
# ---------------------------------------------------------------------------
def embed_personas_from_disk(
    *,
    personas_path: str | Path = "data/personas/personas.json",
    output_dir: str | Path | None = None,
    ontology_config: str | Path = "configs/ontology.yaml",
    max_records: int | None = 10000,
    cluster: bool = False,
    min_cluster_size: int | None = None,
    model: str | None = None,
    backend: str = "auto",
    base_url: str | None = None,
    api_key_env: str = "PERSONAS_VN_LLM_API_KEY",
) -> dict[str, Any]:
    """Load a persisted ``PersonaBatch`` and run :func:`embed_personas`.

    Picks the embedding backend automatically based on ``model`` (anything
    starting with ``nvidia/`` → NIM, else local sentence-transformers).

    Clustering is **off by default**; pass ``cluster=True`` to add a
    ``cluster`` column to ``reduced.parquet``.
    """
    p_path = resolve(personas_path)
    if not p_path.exists():
        raise FileNotFoundError(f"personas not found: {p_path}")
    batch = PersonaBatch.model_validate_json(p_path.read_text(encoding="utf-8"))

    if output_dir is None:
        registry: OntologyRegistry = load_registry(ontology_config)
        output_dir = registry.output.get("personas_dir", "data/personas")
    out_dir = ensure_dir(output_dir)

    cfg_kwargs: dict[str, Any] = {
        "max_records": max_records,
        "cluster": cluster,
        "min_cluster_size": min_cluster_size,
        "backend": backend,
        "api_key_env": api_key_env,
    }
    if model is not None:
        cfg_kwargs["model"] = model
    if base_url is not None:
        cfg_kwargs["base_url"] = base_url
    cfg = PersonaEmbedConfig(**cfg_kwargs)
    summary = embed_personas(batch, output_dir=out_dir, config=cfg)
    log.info("embed_personas_from_disk: %s", summary)
    return summary
