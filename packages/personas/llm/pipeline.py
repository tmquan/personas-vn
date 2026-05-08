"""Two-stage LLM narrative pipeline (LLM A + LLM B).

Plugs into :mod:`packages.personas.datasets.builder` as an alternative
to the deterministic templating renderer in
:mod:`packages.personas.datasets.narrative_render`. The structured PGM /
OCEAN / name / province draws stay unchanged; only the rendering of
the eleven narrative columns swaps from templates to LLM-generated
text.

Architecture (matches the Data Designer reference diagram):

::

    PGM (structured) ─┐
                      ├─►  LLM A  ─►  cultural_background
    OCEAN (Big Five) ─┘             skills_and_expertise{,_list}
                                    hobbies_and_interests{,_list}
                                    career_goals_and_ambitions
                                              │
                                              ▼
        PGM + OCEAN + LLM A outputs ─► LLM B  ─► persona
                                                  professional_persona
                                                  arts_persona
                                                  sports_persona
                                                  travel_persona
                                                  culinary_persona

The pipeline keeps a per-row, per-stage, per-language **JSONL cache**
on disk so any failure (rate limit, timeout, OOM) is recoverable —
re-running picks up exactly where the previous run stopped.

Validation: every LLM JSON is checked for the required keys before
caching. Permanent failures fall back to a templated stub so the
final parquet always has every cell populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.logging import get_logger
from packages.personas.datasets.narrative_render import render_narratives
from packages.personas.llm.client import JSONLCache, LLMClient, LLMConfig
from packages.personas.llm.prompts import (
    SYSTEM_LLM_A_EN,
    SYSTEM_LLM_A_VI,
    SYSTEM_LLM_B_EN,
    SYSTEM_LLM_B_VI,
    build_user_prompt_a,
    build_user_prompt_b,
    validate_llm_a,
    validate_llm_b,
)

log = get_logger(__name__)


@dataclass
class LLMPipelineConfig:
    """Knobs for the LLM-driven narrative pipeline."""

    llm: LLMConfig
    cache_dir: Path | str = "data/Nemotron-Personas-Vietnam/_llm_cache"
    # When True (default), rows the LLM permanently failed on fall back to
    # the templated renderer so the final parquet still has 22 columns
    # populated. When False, those rows raise and abort the build.
    fallback_to_templates: bool = True


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------
def _cache_key(uuid: str, stage: str, lang: str, fingerprint: str) -> str:
    """Stable cache key for one (row, stage, language, model) result.

    The model fingerprint is folded in so a second run with a different
    model doesn't accidentally serve stale cache entries from the
    previous model.
    """
    return f"{stage}|{lang}|{fingerprint}|{uuid}"


# ---------------------------------------------------------------------------
# Public entry point — used by `nemotron_builder` as a renderer
# ---------------------------------------------------------------------------
def render_narratives_via_llm(
    structured_chunk: pd.DataFrame,
    *,
    lang: str,
    cfg: LLMPipelineConfig,
    client_a: LLMClient,
    client_b: LLMClient,
) -> pd.DataFrame:
    """Drop-in replacement for :func:`narrative_render.render_narratives`.

    Parameters
    ----------
    structured_chunk:
        Same DataFrame the templating renderer expects, plus OCEAN
        columns (``personality_summary_vi`` / ``personality_summary_en``).
    lang:
        ``"vi"`` or ``"en"``.
    cfg:
        :class:`LLMPipelineConfig` (carries cache_dir + fallback policy).
    client_a, client_b:
        Two pre-built :class:`LLMClient` instances (typically the same
        model and config; we keep them separate so callers can wire
        different models for the two stages if they want to).

    Returns
    -------
    DataFrame with the same 11 narrative columns the templating
    renderer would produce, indexed identically to ``structured_chunk``.
    """
    if structured_chunk.empty:
        # Match the empty-output shape of the templating renderer.
        return render_narratives(
            structured_chunk, lang,  # type: ignore[arg-type]
            picks_chunk=_empty_picks_for_chunk(),
        )

    # Reset index so positional arrays line up; restore at the end.
    df = structured_chunk.reset_index(drop=True).copy()
    n = len(df)

    # ------- LLM A — narrative attribute fields
    items_a = []
    sys_prompt_a = SYSTEM_LLM_A_VI if lang == "vi" else SYSTEM_LLM_A_EN
    for i in range(n):
        record = df.iloc[i].to_dict()
        items_a.append({
            "cache_key":    _cache_key(record["uuid"], "a", lang,
                                       client_a.fingerprint),
            "system_prompt": sys_prompt_a,
            "user_prompt":   build_user_prompt_a(record, lang),  # type: ignore[arg-type]
        })
    a_results = client_a.call_many(items_a, progress_label=f"llm-a/{lang}")

    # Validate + replace failures with a sentinel
    a_clean: list[dict[str, Any] | None] = []
    n_a_fail = 0
    for i, res in enumerate(a_results):
        if res is None:
            a_clean.append(None)
            n_a_fail += 1
            continue
        err = validate_llm_a(res)
        if err is not None:
            log.debug("llm-a validation failed at row %d: %s", i, err)
            a_clean.append(None)
            n_a_fail += 1
        else:
            a_clean.append(res)
    if n_a_fail:
        log.warning("llm-a/%s: %d/%d rows failed validation/network",
                    lang, n_a_fail, n)

    # ------- LLM B — persona narratives, conditioned on LLM-A outputs
    items_b = []
    sys_prompt_b = SYSTEM_LLM_B_VI if lang == "vi" else SYSTEM_LLM_B_EN
    for i in range(n):
        record = df.iloc[i].to_dict()
        a_obj = a_clean[i] or {}
        items_b.append({
            "cache_key":    _cache_key(record["uuid"], "b", lang,
                                       client_b.fingerprint),
            "system_prompt": sys_prompt_b,
            "user_prompt":   build_user_prompt_b(record, a_obj, lang),  # type: ignore[arg-type]
        })
    b_results = client_b.call_many(items_b, progress_label=f"llm-b/{lang}")

    b_clean: list[dict[str, Any] | None] = []
    n_b_fail = 0
    for i, res in enumerate(b_results):
        if res is None:
            b_clean.append(None)
            n_b_fail += 1
            continue
        err = validate_llm_b(res)
        if err is not None:
            log.debug("llm-b validation failed at row %d: %s", i, err)
            b_clean.append(None)
            n_b_fail += 1
        else:
            b_clean.append(res)
    if n_b_fail:
        log.warning("llm-b/%s: %d/%d rows failed validation/network",
                    lang, n_b_fail, n)

    # ------- Assemble the 11-column narrative DataFrame
    out = _assemble_narratives(df, a_clean, b_clean, lang, cfg)

    # Restore original index so the caller can concat with structured_chunk
    out.index = structured_chunk.index
    return out


# ---------------------------------------------------------------------------
# Assembly + fallback
# ---------------------------------------------------------------------------
def _assemble_narratives(
    df: pd.DataFrame,
    a_results: list[dict[str, Any] | None],
    b_results: list[dict[str, Any] | None],
    lang: str,
    cfg: LLMPipelineConfig,
) -> pd.DataFrame:
    """Build the 11-column narrative DataFrame from LLM responses.

    Rows where either LLM stage failed permanently are filled in from
    the existing templating renderer (when ``fallback_to_templates=True``)
    so the final parquet keeps row-count parity with the structured
    frame.
    """
    n = len(df)
    cols = (
        "professional_persona", "sports_persona", "arts_persona",
        "travel_persona", "culinary_persona", "persona",
        "cultural_background", "skills_and_expertise",
        "skills_and_expertise_list", "hobbies_and_interests",
        "hobbies_and_interests_list", "career_goals_and_ambitions",
    )

    data: dict[str, list[str]] = {c: [""] * n for c in cols}
    fallback_idx: list[int] = []

    for i in range(n):
        a, b = a_results[i], b_results[i]
        if a is None or b is None:
            fallback_idx.append(i)
            continue
        # LLM-A maps to the four contextual narrative fields (+ list mirrors)
        data["cultural_background"][i]         = a["cultural_background"]
        data["skills_and_expertise"][i]        = a["skills_and_expertise"]
        data["skills_and_expertise_list"][i]   = a["skills_and_expertise_list"]
        data["hobbies_and_interests"][i]       = a["hobbies_and_interests"]
        data["hobbies_and_interests_list"][i]  = a["hobbies_and_interests_list"]
        data["career_goals_and_ambitions"][i]  = a["career_goals_and_ambitions"]
        # LLM-B maps to the six persona narratives
        for k in ("persona", "professional_persona", "sports_persona",
                  "arts_persona", "travel_persona", "culinary_persona"):
            data[k][i] = b[k]

    out = pd.DataFrame(data, index=range(n))

    if fallback_idx:
        if not cfg.fallback_to_templates:
            raise RuntimeError(
                f"{len(fallback_idx)} rows had no LLM output and "
                "fallback_to_templates=False"
            )
        log.info(
            "filling %d rows with templated fallback (LLM permanent fail)",
            len(fallback_idx),
        )
        # Build a tiny picks_chunk on-the-fly for just the fallback rows.
        # The deterministic renderer accepts the same input shape.
        import numpy as np

        from packages.personas.datasets.narrative_render import precompute_narrative_picks

        rng = np.random.default_rng(seed=hash(f"fallback-{lang}") & 0xFFFFFFFF)
        picks_full = precompute_narrative_picks(n, rng)
        sub_df = df.iloc[fallback_idx].copy()
        sub_picks = {k: v[fallback_idx] for k, v in picks_full.items()}
        templated = render_narratives(sub_df, lang, sub_picks)  # type: ignore[arg-type]
        for col in cols:
            for src_idx, target_idx in enumerate(fallback_idx):
                out.at[target_idx, col] = templated.iloc[src_idx][col]

    return out


def _empty_picks_for_chunk() -> dict[str, Any]:
    """Used only when the chunk is empty — keeps the empty-frame branch
    of ``render_narratives`` happy without rebuilding the full picks."""
    import numpy as np

    from packages.personas.datasets.narrative_render import PERM_UNIVERSE
    return {k: np.empty((0, PERM_UNIVERSE), dtype=np.uint8) for k in (
        "professional", "skills", "goals",
        "sports", "arts", "travel", "culinary", "culture",
        "hobbies",
    )}


# ---------------------------------------------------------------------------
# Convenience: open the two LLM clients with cache files in one place
# ---------------------------------------------------------------------------
def open_clients(cfg: LLMPipelineConfig) -> tuple[LLMClient, LLMClient]:
    """Open the two clients (LLM A + LLM B) with separate cache files."""
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_a = JSONLCache(path=cache_dir / "llm_a.jsonl")
    cache_b = JSONLCache(path=cache_dir / "llm_b.jsonl")
    return (
        LLMClient(cfg=cfg.llm, cache=cache_a),
        LLMClient(cfg=cfg.llm, cache=cache_b),
    )
