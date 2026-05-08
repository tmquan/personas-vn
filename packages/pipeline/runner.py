"""End-to-end pipeline runner.

Two responsibilities:

1. Orchestrate the scrape -> map -> persist flow into ``data/ontology/``.
2. Orchestrate the load-ontology -> generate -> persist flow into
   ``data/personas/``.

We deliberately keep this module thin so the CLI can compose the two halves
in any order and the visualizer can call them on-demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.common.config import Config, load_config
from packages.common.logging import get_logger
from packages.common.paths import ensure_dir, resolve
from packages.ontology import Dataset, PersonaBatch
from packages.ontology.registry import load_registry
from packages.personas.pgm import generate_personas
from packages.scraper import map_posts_to_datasets, scrape_nso

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
@dataclass
class PipelineArtefacts:
    """Paths produced by a successful pipeline run."""

    raw_dir: Path
    ontology_dir: Path
    personas_dir: Path
    datasets_path: Path | None = None
    manifest_path: Path | None = None
    personas_path: Path | None = None


def _write_datasets(datasets: list[Dataset], ontology_dir: Path, manifest: dict[str, Any]) -> Path:
    out_path = ontology_dir / "datasets.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scrape_manifest": manifest,
        "datasets": [d.model_dump(mode="json") for d in datasets],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote ontology datasets -> %s (%d records)", out_path, len(datasets))
    return out_path


def _write_personas(batch: PersonaBatch, personas_dir: Path) -> Path:
    out_path = personas_dir / "personas.json"
    out_path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    log.info("wrote personas -> %s (%d records)", out_path, batch.n)
    return out_path


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def run_scrape(
    scraper_config: str | Path = "configs/scraper.yaml",
    ontology_config: str | Path = "configs/ontology.yaml",
) -> PipelineArtefacts:
    """Scrape NSO, map to datasets, persist to ``data/ontology/``."""
    cfg = load_config(scraper_config)
    raw_dir = ensure_dir(cfg.paths.raw_dir)
    ontology_dir = ensure_dir(cfg.paths.ontology_dir)
    personas_dir = ensure_dir(load_config(ontology_config).output.get("personas_dir", "data/personas"))

    records, manifest = scrape_nso(cfg=cfg, write_outputs=True)
    registry = load_registry(ontology_config)
    datasets = map_posts_to_datasets(records, registry)
    datasets_path = _write_datasets(datasets, ontology_dir, manifest.to_dict())
    manifest_path = ontology_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return PipelineArtefacts(
        raw_dir=raw_dir,
        ontology_dir=ontology_dir,
        personas_dir=personas_dir,
        datasets_path=datasets_path,
        manifest_path=manifest_path,
    )


def run_generate(
    *,
    n: int | None = None,
    seed: int | None = None,
    ontology_config: str | Path = "configs/ontology.yaml",
) -> PipelineArtefacts:
    """Generate a persona batch and persist to ``data/personas/``."""
    cfg: Config = load_config(ontology_config)
    registry = load_registry(ontology_config)
    output = cfg.get("output", Config())
    n = n or int(output.get("default_n_personas", 200))
    personas_dir = ensure_dir(output.get("personas_dir", "data/personas"))

    # Load the most recent scrape manifest so personas record their
    # provenance back to the NSO snapshot they were grounded in.
    manifest_path = resolve("data/ontology/manifest.json")
    scrape_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            scrape_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("manifest at %s is not valid JSON; ignoring", manifest_path)

    batch = generate_personas(
        n,
        registry=registry,
        seed=seed,
        scrape_manifest=scrape_manifest,
    )
    personas_path = _write_personas(batch, personas_dir)

    return PipelineArtefacts(
        raw_dir=resolve("data/raw"),
        ontology_dir=resolve("data/ontology"),
        personas_dir=personas_dir,
        personas_path=personas_path,
    )


def run_all(
    *,
    n: int | None = None,
    seed: int | None = None,
    scraper_config: str | Path = "configs/scraper.yaml",
    ontology_config: str | Path = "configs/ontology.yaml",
) -> PipelineArtefacts:
    """Run scrape then generate. Returns the merged artefact paths."""
    scrape_art = run_scrape(scraper_config=scraper_config, ontology_config=ontology_config)
    gen_art = run_generate(n=n, seed=seed, ontology_config=ontology_config)
    return PipelineArtefacts(
        raw_dir=scrape_art.raw_dir,
        ontology_dir=scrape_art.ontology_dir,
        personas_dir=gen_art.personas_dir,
        datasets_path=scrape_art.datasets_path,
        manifest_path=scrape_art.manifest_path,
        personas_path=gen_art.personas_path,
    )
