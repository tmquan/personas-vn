"""Curator pipeline orchestrator.

Two execution backends:

1. **Local sequential** (always available, what we use by default).
2. **NeMo Curator** ``Pipeline`` + ``XennaExecutor`` when ``nemo_curator`` is
   importable. The local backend is what the unit tests + the visualizer
   exercise; the NeMo Curator backend is what you would flip to for a
   distributed Ray deployment.

In both cases each stage reads / writes the same on-disk shape under
``data/nso-gov-vn/``, so the visualizer doesn't need to know which backend
ran.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.common.config import Config
from packages.common.config import load_config as _load_yaml
from packages.common.logging import get_logger
from packages.common.paths import ensure_dir
from packages.curator.stages import (
    DownloadStage,
    EmbedStage,
    ExtractStage,
    ParseStage,
    ReduceStage,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass
class CurationConfig:
    """Resolved + path-anchored view of ``configs/curator.yaml``."""

    name: str
    root: Path
    raw_dir: Path
    parsed_dir: Path
    extracted_dir: Path
    embedded_dir: Path
    reduced_dir: Path
    download: Config
    parse: Config
    extract: Config
    embed: Config
    reduce: Config

    @classmethod
    def from_yaml(cls, path: str | Path) -> CurationConfig:
        cfg = _load_yaml(path)
        root = ensure_dir(cfg.dataset.root)
        return cls(
            name=str(cfg.dataset.name),
            root=root,
            raw_dir=ensure_dir(root / "raw"),
            parsed_dir=ensure_dir(root / "parsed"),
            extracted_dir=ensure_dir(root / "extracted"),
            embedded_dir=ensure_dir(root / "embedded"),
            reduced_dir=ensure_dir(root / "reduced"),
            download=cfg.download,
            parse=cfg.parse,
            extract=cfg.extract,
            embed=cfg.embed,
            reduce=cfg.reduce,
        )


@dataclass
class CurationArtefacts:
    """Paths produced by a curation run."""

    root: Path
    manifest_path: Path
    parsed_path: Path | None = None
    extracted_path: Path | None = None
    embedded_path: Path | None = None
    reduced_path: Path | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class CurationPipeline:
    """Sequential local executor, with optional NeMo Curator handoff."""

    STAGE_ORDER = ("download", "parse", "extract", "embed", "reduce")

    def __init__(self, config: CurationConfig) -> None:
        self.cfg = config

    # ----- stage factories -------------------------------------------------
    def _build_stages(self) -> dict[str, Any]:
        return {
            "download": DownloadStage(self.cfg.download, self.cfg.raw_dir),
            "parse":    ParseStage(self.cfg.parse, self.cfg.raw_dir, self.cfg.parsed_dir),
            "extract":  ExtractStage(self.cfg.extract, self.cfg.parsed_dir, self.cfg.extracted_dir),
            "embed":    EmbedStage(self.cfg.embed, self.cfg.extracted_dir, self.cfg.embedded_dir),
            "reduce":   ReduceStage(self.cfg.reduce, self.cfg.embedded_dir, self.cfg.reduced_dir),
        }

    # ----- runner ----------------------------------------------------------
    def run(
        self,
        *,
        only: list[str] | None = None,
        skip: list[str] | None = None,
        backend: str = "local",
    ) -> CurationArtefacts:
        if backend == "nemo_curator":
            return self._run_nemo_curator(only=only, skip=skip)
        return self._run_local(only=only, skip=skip)

    # ----- local backend ---------------------------------------------------
    def _run_local(
        self,
        *,
        only: list[str] | None,
        skip: list[str] | None,
    ) -> CurationArtefacts:
        stages = self._build_stages()
        manifest: dict[str, Any] = {
            "dataset": self.cfg.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "backend": "local",
            "stages": {},
        }

        for name in self.STAGE_ORDER:
            if only and name not in only:
                continue
            if skip and name in skip:
                continue
            stage = stages[name]
            log.info(">>> stage: %s", name)
            stage.setup()
            try:
                summary = stage.run()
            except Exception as exc:
                manifest["stages"][name] = {"status": "error", "error": str(exc)}
                self._write_manifest(manifest)
                raise
            finally:
                stage.teardown()
            manifest["stages"][name] = {"status": "ok", **(summary or {})}

        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path = self._write_manifest(manifest)

        return CurationArtefacts(
            root=self.cfg.root,
            manifest_path=manifest_path,
            parsed_path=self.cfg.parsed_dir / "parsed.jsonl",
            extracted_path=self.cfg.extracted_dir / "extracted.jsonl",
            embedded_path=self.cfg.embedded_dir / "embedded.parquet",
            reduced_path=self.cfg.reduced_dir / "reduced.parquet",
        )

    # ----- NeMo Curator backend (optional) ---------------------------------
    def _run_nemo_curator(
        self,
        *,
        only: list[str] | None,
        skip: list[str] | None,
    ) -> CurationArtefacts:
        """Wrap each local stage as a NeMo Curator ``ProcessingStage`` and
        hand them to a ``Pipeline`` with the ``XennaExecutor``.

        The wrappers below intentionally do NOT split work into per-document
        Ray actors — every stage in this dataset is small enough (≤ ~5k
        records, embedding ~10s on CPU) that the orchestration overhead of
        Ray actors would dwarf the work. We use NeMo Curator as the
        *driver* (so production users can swap in real ProcessingStage
        sub-classes for sharded execution later) but the stage payload is
        the same callable as the local backend.
        """
        try:
            from nemo_curator.core.pipeline import Pipeline as NCPipeline
            from nemo_curator.core.stage import ProcessingStage as NCProcessingStage
            from nemo_curator.tasks import DocumentBatch as NCDocumentBatch
        except ImportError as exc:
            log.warning("NeMo Curator not available (%s); falling back to local backend", exc)
            return self._run_local(only=only, skip=skip)

        stages = self._build_stages()
        manifest: dict[str, Any] = {
            "dataset": self.cfg.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "backend": "nemo_curator",
            "stages": {},
        }

        nc_pipeline = NCPipeline(name=f"{self.cfg.name}-curator")
        wrappers: list[tuple[str, Any]] = []
        for name in self.STAGE_ORDER:
            if only and name not in only:
                continue
            if skip and name in skip:
                continue
            local_stage = stages[name]
            wrapper = _wrap_for_nemo_curator(name, local_stage, NCProcessingStage, NCDocumentBatch)
            nc_pipeline.add_stage(wrapper)
            wrappers.append((name, wrapper))

        # Run the Curator pipeline. The wrappers below capture per-stage
        # summaries into the shared ``manifest`` dict.
        try:
            from nemo_curator.backends.experimental.in_process import InProcessExecutor

            executor = InProcessExecutor()
            nc_pipeline.run(executor=executor)
        except Exception as exc:
            log.warning("NeMo Curator pipeline crashed (%s); falling back to local", exc)
            return self._run_local(only=only, skip=skip)
        for name, wrapper in wrappers:
            manifest["stages"][name] = {"status": "ok", **(wrapper.summary or {})}

        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path = self._write_manifest(manifest)
        return CurationArtefacts(
            root=self.cfg.root,
            manifest_path=manifest_path,
            parsed_path=self.cfg.parsed_dir / "parsed.jsonl",
            extracted_path=self.cfg.extracted_dir / "extracted.jsonl",
            embedded_path=self.cfg.embedded_dir / "embedded.parquet",
            reduced_path=self.cfg.reduced_dir / "reduced.parquet",
        )

    # ----- common ----------------------------------------------------------
    def _write_manifest(self, manifest: dict[str, Any]) -> Path:
        path = self.cfg.root / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# NeMo Curator wrapper
# ---------------------------------------------------------------------------
def _wrap_for_nemo_curator(name: str, local_stage: Any, NCProcessingStage, NCDocumentBatch):
    """Build a ProcessingStage subclass that delegates to the local stage's
    ``run()`` method and stores the summary on the wrapper instance.
    """

    class _Wrapper(NCProcessingStage):  # type: ignore[misc, valid-type]
        _name = f"nso.{name}"

        def __init__(self) -> None:
            super().__init__()
            self.summary: dict[str, Any] = {}

        # NeMo Curator's setup/teardown signatures.
        def setup(self) -> None:  # type: ignore[override]
            local_stage.setup()

        def teardown(self) -> None:  # type: ignore[override]
            local_stage.teardown()

        def process(self, task):  # type: ignore[override]
            # Each stage in this pipeline is a "leaf" that reads from disk
            # and writes to disk; the ``task`` payload is irrelevant. We
            # simply invoke the local run and stash the summary.
            self.summary = local_stage.run() or {}
            # Return an empty doc-batch so downstream stages see something.
            try:
                return NCDocumentBatch(documents=[])
            except TypeError:
                # NeMo Curator's API has shifted on the DocumentBatch
                # constructor across releases; an empty list always works
                # as a positional arg.
                return NCDocumentBatch([])

    _Wrapper.__name__ = f"NSO_{name.capitalize()}Stage"
    return _Wrapper()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def load_config(path: str | Path = "configs/curator.yaml") -> CurationConfig:
    return CurationConfig.from_yaml(path)


def run_curation(
    config_path: str | Path = "configs/curator.yaml",
    *,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    backend: str = "local",
) -> CurationArtefacts:
    cfg = load_config(config_path)
    return CurationPipeline(cfg).run(only=only, skip=skip, backend=backend)
