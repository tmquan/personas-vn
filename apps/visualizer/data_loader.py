"""Data-loading helpers for the visualizer.

The state object pulls together every artefact the UI needs:

* The ontology registry (for the Ontology tab's domain table + DAG).
* The PX-Web table catalog (the new Datasets tab — replaces the legacy
  WordPress posts dump).
* The latest persona batch (Personas + Persona-embeddings tabs).
* The curator pipeline outputs (Curator tab).
* The PX-Web-grounded distribution tables (Distributions tab).

Everything is lazy: if a particular artefact isn't on disk we just leave
it as ``None`` and the corresponding tab renders a placeholder with
instructions for how to populate it. The visualizer never starts up
empty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.config import load_config
from packages.common.logging import get_logger
from packages.common.paths import resolve
from packages.ontology import Persona, PersonaBatch
from packages.ontology.registry import OntologyRegistry, load_registry
from packages.personas.pgm import generate_personas
from packages.personas.pgm.distributions import PxWebDistributions

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class PxWebTableEntry:
    """One row in the PX-Web table catalog (drives the Datasets tab)."""

    id: str
    title: str
    database: str
    parquet_path: Path
    n_cells: int
    variables: list[str] = field(default_factory=list)


@dataclass
class VisualizerState:
    """Everything the UI needs in one place."""

    registry: OntologyRegistry
    pxweb_tables: list[PxWebTableEntry]
    personas: list[Persona]
    persona_batch: PersonaBatch | None
    scrape_manifest: dict[str, Any] = field(default_factory=dict)
    # Curator artefacts; populated lazily so users without the [curator]
    # extras can still launch the visualizer.
    curator_root: Path | None = None
    curator_manifest: dict[str, Any] = field(default_factory=dict)
    curator_reduced: Any = None  # pandas.DataFrame when loaded

    # Persona-embedding artefacts (from `personas-vn embed-personas`).
    personas_dir: Path | None = None
    personas_reduced: Any = None

    # Convenience: derived DataFrames cached on the dataclass.
    @property
    def pxweb_tables_df(self) -> pd.DataFrame:
        if not self.pxweb_tables:
            return pd.DataFrame(columns=["id", "title", "database", "n_cells", "variables"])
        return pd.DataFrame(
            [
                {
                    "id": t.id,
                    "title": t.title,
                    "database": t.database,
                    "n_cells": t.n_cells,
                    "variables": ", ".join(t.variables),
                }
                for t in self.pxweb_tables
            ]
        )

    @property
    def personas_df(self) -> pd.DataFrame:
        if not self.personas:
            return pd.DataFrame()
        return pd.DataFrame([p.model_dump() for p in self.personas])

    @property
    def n_enriched(self) -> int:
        return sum(1 for p in self.personas if getattr(p, "enriched", False))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_pxweb_catalog(pxweb_dir: Path) -> list[PxWebTableEntry]:
    """Read every PX-Web ``*.metadata.json`` next to the parquets.

    The curator's download stage drops a ``_catalog.json`` per language
    plus a ``<slug>.metadata.json`` per table. We prefer per-table files
    because they survive partial re-runs.
    """
    if not pxweb_dir.exists():
        return []
    entries: list[PxWebTableEntry] = []
    for meta_path in sorted(pxweb_dir.glob("*.metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("invalid metadata json: %s", meta_path)
            continue
        parquet = meta_path.with_name(meta_path.name.replace(".metadata.json", ".parquet"))
        if not parquet.exists():
            continue
        entries.append(
            PxWebTableEntry(
                id=meta.get("table_id", parquet.stem),
                title=meta.get("title", parquet.stem),
                database=meta.get("database", "?"),
                parquet_path=parquet,
                n_cells=int(meta.get("n_kept") or meta.get("n_cells") or 0),
                variables=[v.get("code", "") for v in meta.get("variables") or []],
            )
        )
    return entries


def _load_personas_from_disk(path: Path) -> PersonaBatch:
    return PersonaBatch.model_validate_json(path.read_text(encoding="utf-8"))


def load_state(
    *,
    ontology_config: str | Path = "configs/ontology.yaml",
    visualizer_config: str | Path = "configs/visualizer.yaml",
    n_personas_if_missing: int | None = None,
    force_regenerate: bool = False,
) -> VisualizerState:
    """Build a fresh :class:`VisualizerState`.

    If a previous pipeline run has produced
    ``data/personas/personas.json`` we re-use it. If a curator run has
    produced PX-Web parquets, the Datasets / Distributions tabs are
    populated from them. Anything missing falls back to an informative
    placeholder rather than a crash.
    """
    registry = load_registry(ontology_config)
    viz_cfg = load_config(visualizer_config)

    # ---- PX-Web table catalog (replaces the legacy WordPress dataset list)
    pxweb_dir = resolve("data/nso-gov-vn/raw/pxweb/vi")
    pxweb_tables = _load_pxweb_catalog(pxweb_dir)
    log.info("PX-Web catalog: %d tables", len(pxweb_tables))

    # ---- Personas
    personas_path = resolve(viz_cfg.paths.personas_data) / "personas.json"
    enriched_path = resolve(viz_cfg.paths.personas_data) / "personas_enriched.json"
    persona_batch: PersonaBatch | None = None
    # Prefer the enriched batch if it's newer than the raw one.
    candidates = sorted(
        [p for p in (personas_path, enriched_path) if p.exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        if force_regenerate:
            break
        try:
            persona_batch = _load_personas_from_disk(cand)
            log.info("loaded personas from %s (n=%d)", cand, persona_batch.n)
            break
        except Exception as exc:
            log.warning("failed to load %s (%s); trying next", cand, exc)

    if persona_batch is None:
        n = n_personas_if_missing or int(viz_cfg.ui.get("default_n_personas", 200))
        log.info("generating %d personas on the fly", n)
        persona_batch = generate_personas(n, registry=registry)

    # ---- Curator artefacts
    curator_root = resolve("data/nso-gov-vn")
    curator_manifest: dict[str, Any] = {}
    curator_reduced = None
    manifest_file = curator_root / "manifest.json"
    if manifest_file.exists():
        try:
            curator_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("curator manifest at %s is not valid JSON", manifest_file)
    reduced_file = curator_root / "reduced" / "reduced.parquet"
    if reduced_file.exists():
        try:
            curator_reduced = pd.read_parquet(reduced_file)
            log.info("loaded curator reduced.parquet (%d rows)", len(curator_reduced))
        except Exception as exc:
            log.warning("failed to load reduced.parquet (%s)", exc)

    # ---- Persona embeddings
    personas_reduced = None
    personas_dir = resolve(viz_cfg.paths.personas_data)
    p_reduced_file = personas_dir / "reduced.parquet"
    if p_reduced_file.exists():
        try:
            personas_reduced = pd.read_parquet(p_reduced_file)
            log.info("loaded persona reduced.parquet (%d rows)", len(personas_reduced))
        except Exception as exc:
            log.warning("failed to load persona reduced.parquet (%s)", exc)

    scrape_manifest = persona_batch.scrape_manifest if persona_batch else {}

    return VisualizerState(
        registry=registry,
        pxweb_tables=pxweb_tables,
        personas=list(persona_batch.personas) if persona_batch else [],
        persona_batch=persona_batch,
        scrape_manifest=scrape_manifest,
        curator_root=curator_root if curator_root.exists() else None,
        curator_manifest=curator_manifest,
        curator_reduced=curator_reduced,
        personas_dir=personas_dir,
        personas_reduced=personas_reduced,
    )


@lru_cache(maxsize=1)
def get_distributions() -> PxWebDistributions:
    """Cached singleton — distribution tables are read-only.

    Now uses :class:`packages.personas.pgm.distributions.PxWebDistributions`
    (real NSO PX-Web statistics) instead of the legacy hand-coded
    ``VietnameseDistributions``.
    """
    return PxWebDistributions()


def load_pxweb_table(entry: PxWebTableEntry) -> pd.DataFrame:
    """Load one PX-Web parquet into a DataFrame; small per-call cost."""
    return pd.read_parquet(entry.parquet_path)
