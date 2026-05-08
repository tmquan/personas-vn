"""Pipeline orchestration.

Two public entry points:

* :func:`run_scrape` — scrape NSO + map raw posts onto ontology datasets.
* :func:`run_generate` — load the latest ontology snapshot + manifest, then
  generate a :class:`PersonaBatch`.

Both are wrapped by the CLI in :mod:`packages.pipeline.cli`.
"""

from packages.pipeline.runner import (
    PipelineArtefacts,
    run_all,
    run_generate,
    run_scrape,
)

__all__ = ["PipelineArtefacts", "run_all", "run_generate", "run_scrape"]
