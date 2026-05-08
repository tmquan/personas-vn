"""NSO curation pipeline.

Mirrors ViLA's ``packages/datasites/anle/`` five-stage layout — download →
parse → extract → embed → reduce — and is wire-compatible with NeMo
Curator's ``ProcessingStage`` / ``Pipeline`` API. When NeMo Curator is
installed (``pip install -e ".[curator]"``) the stages register themselves
as real ``ProcessingStage`` subclasses; otherwise we fall back to a thin
in-house executor that runs the same callables sequentially on the local
process. Either way the on-disk layout under ``data/nso-gov-vn/`` is
identical, so the visualizer and downstream code do not care which path was
taken.

Stage outputs (under ``dataset.root`` from ``configs/curator.yaml``):

    raw/        # one JSON-Lines file per endpoint (full crawl)
    parsed/     # parsed.jsonl  — HTML stripped, language detected
    extracted/  # extracted.jsonl + keyword_vocab.json
    embedded/   # embedded.parquet (id + 384-d float32 vector)
    reduced/    # reduced.parquet  (id + 2-D layout + cluster id)
    manifest.json
"""

from packages.curator.pipeline import (
    CurationArtefacts,
    CurationConfig,
    CurationPipeline,
    load_config,
    run_curation,
)
from packages.curator.stages import (
    DownloadStage,
    EmbedStage,
    ExtractStage,
    ParseStage,
    ReduceStage,
)

__all__ = [
    "CurationArtefacts",
    "CurationConfig",
    "CurationPipeline",
    "DownloadStage",
    "EmbedStage",
    "ExtractStage",
    "ParseStage",
    "ReduceStage",
    "load_config",
    "run_curation",
]
