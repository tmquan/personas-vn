"""Unified LLM layer — OpenAI-compatible client, prompts, and pipelines.

Two pipelines share one HTTP client:

* :mod:`packages.personas.llm.client`   — concurrent OpenAI-compatible
  client with JSONL cache + retries + ``_MockClient`` test seam.
* :mod:`packages.personas.llm.prompts`  — system/user prompts + JSON
  validators for the Data Designer LLM-A and LLM-B stages.
* :mod:`packages.personas.llm.pipeline` — Data Designer narrative
  pipeline (PGM + OCEAN → LLM A → LLM B) used by the dataset builder.
* :mod:`packages.personas.llm.enrich`   — single-bio enrichment of a
  saved ``PersonaBatch`` (the older flow that produces
  ``data/personas/personas_enriched.json``).

Both pipelines route through :class:`packages.personas.llm.client.LLMClient`,
which is the single source of truth for HTTP, retry, and JSON parsing.
This dedupe is the whole point of the unification — there used to be
two parallel httpx clients.

Pre-unification this content was split across
:mod:`packages.personas.llm.enrich` and :mod:`packages.personas.datasets.{llm_client,
llm_prompts,llm_pipeline}`.
"""

from __future__ import annotations

from packages.personas.llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    KNOWN_MODELS,
    JSONLCache,
    LLMClient,
    LLMConfig,
)
from packages.personas.llm.enrich import (
    LLMEnrichConfig,
    enrich_personas,
    enrich_personas_from_disk,
)
from packages.personas.llm.pipeline import (
    LLMPipelineConfig,
    open_clients,
    render_narratives_via_llm,
)
from packages.personas.llm.prompts import (
    LLM_A_REQUIRED_KEYS,
    LLM_B_REQUIRED_KEYS,
    SYSTEM_LLM_A_EN,
    SYSTEM_LLM_A_VI,
    SYSTEM_LLM_B_EN,
    SYSTEM_LLM_B_VI,
    build_user_prompt_a,
    build_user_prompt_b,
    validate_llm_a,
    validate_llm_b,
)

__all__ = [
    # client
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "JSONLCache",
    "KNOWN_MODELS",
    "LLMClient",
    "LLMConfig",
    # prompts
    "LLM_A_REQUIRED_KEYS",
    "LLM_B_REQUIRED_KEYS",
    "SYSTEM_LLM_A_EN",
    "SYSTEM_LLM_A_VI",
    "SYSTEM_LLM_B_EN",
    "SYSTEM_LLM_B_VI",
    "build_user_prompt_a",
    "build_user_prompt_b",
    "validate_llm_a",
    "validate_llm_b",
    # pipeline (LLM A + LLM B for the Data Designer dataset build)
    "LLMPipelineConfig",
    "open_clients",
    "render_narratives_via_llm",
    # single-bio enrichment of a PersonaBatch
    "LLMEnrichConfig",
    "enrich_personas",
    "enrich_personas_from_disk",
]
