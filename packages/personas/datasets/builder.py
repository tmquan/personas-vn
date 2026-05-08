"""Streaming builder for the Nemotron-Personas-Vietnam datasets.

End-to-end pipeline:

1. **Structured personas** — :class:`packages.personas.pgm.VNPersonaGenerator`
   produces ``large_size`` rows of structured demographic / socio-economic
   data sampled from NSO PX-Web distributions.
2. **Province** — for each row, pick a province *within* its sampled
   macro-region using V02.01 population weights.
3. **Names** — vectorised draw of Vietnamese full names from
   :mod:`packages.personas.datasets.names`.
4. **UUIDs** — UUID4 hex strings derived from the seeded RNG so each row
   has one stable global identifier shared across the vi/en variants.
5. **Narrative rendering** — :func:`packages.personas.datasets.narrative_render.render_narratives`
   produces the 11 narrative columns in both Vietnamese and English. Both
   renders draw from RNG copies seeded identically, so the *vi* and *en*
   strings describe the same persona row-for-row.
6. **Streaming write** — four ``pyarrow.parquet.ParquetWriter`` instances
   stay open across chunks, one per (language × size) variant. The small
   variants take a stratified-by-region uuid subset of the large rows.

Reproducibility contract
========================
A single integer ``seed`` in :class:`BuilderConfig` produces bit-identical
output (modulo parquet metadata) on every run. Internally we
:func:`numpy.random.SeedSequence.spawn` six independent streams so that
adding a new narrative variant does not perturb the structured draws,
and changing chunk size does not perturb the per-row narrative.

::

    seed seq
       ├─ spawn[0] → structured personas (pgmpy via VNPersonaGenerator)
       ├─ spawn[1] → uuids
       ├─ spawn[2] → province (region-conditional V02.01 weights)
       ├─ spawn[3] → vn-fullname-generator name pools
       ├─ spawn[4] → narrative variants (vi & en consume the same per-chunk seq)
       └─ spawn[5] → small-subset stratified sample
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from packages.common.logging import get_logger
from packages.personas.datasets.names import generate_names
from packages.personas.datasets.narrative_render import (
    precompute_narrative_picks,
    render_narratives,
)
from packages.personas.datasets.provinces import (
    PROVINCE_VI_TO_EN,
    load_province_table,
)
from packages.personas.datasets.schema import (
    AREA_VI_TO_EN,
    COUNTRY_EN,
    COUNTRY_VI,
    EDUCATION_VI_TO_EN,
    MARITAL_VI_TO_EN,
    NEMOTRON_PERSONAS_VIETNAM_COLUMNS,
    OCCUPATION_VI_TO_EN,
    PYARROW_SCHEMA,
    REGIONS_VI_TO_EN,
    SEX_VI_TO_EN,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuilderConfig:
    """All knobs the builder accepts.

    Parameters
    ----------
    out_dir:
        Directory the four parquet files are written to. Created if
        missing.
    large_size:
        Total number of rows in the *-large-* parquets. Default 3 000 000.
    small_size:
        Stratified subset used for the *-small-* parquets. Default
        300 000. Must satisfy ``small_size <= large_size``.
    chunk_size:
        Per-chunk persona count. Larger uses more RAM but fewer pyarrow
        flushes. Default 100 000.
    seed:
        Single integer seed for the whole reproducibility contract; see
        the module docstring.
    pxweb_root:
        Override path to the NSO PX-Web parquets. ``None`` uses the
        project default (``data/nso-gov-vn/raw/pxweb/vi``).
    parquet_compression:
        Compression codec for parquet output (``"snappy"`` is the
        Hugging Face default for the published Nemotron-Personas
        datasets).
    use_llm_pipeline:
        When True, swap the deterministic templated narrative renderer
        for the canonical Data Designer two-stage LLM pipeline
        (PGM + OCEAN → LLM A → LLM B). When False (default), the
        templated renderer is used and no LLM/OCEAN code is loaded.
    llm_pipeline_config:
        Required when ``use_llm_pipeline`` is True; an instance of
        :class:`packages.personas.llm.pipeline.LLMPipelineConfig`
        carrying the model id, base URL, concurrency, and cache dir.
    """

    out_dir: str | Path = field(default="data/Nemotron-Personas-Vietnam")
    large_size: int = 3_000_000
    small_size: int = 300_000
    chunk_size: int = 100_000
    seed: int = 42
    pxweb_root: str | None = None
    parquet_compression: str = "snappy"
    # Stratification axes for the small subset. Default mirrors what the
    # published Nemotron-Personas datasets emphasise (region/sex/education).
    stratify_by: tuple[str, ...] = ("region", "sex", "education_level")
    # LLM pipeline (optional, not used by default)
    use_llm_pipeline: bool = False
    llm_pipeline_config: Any = None  # type: LLMPipelineConfig | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _filename(language: str, size_label: str) -> str:
    return f"Nemotron-Personas-Vietnam-{size_label}-{language}.parquet"


def _generate_uuids(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` UUID4-style hex strings drawn from ``rng``.

    UUID4 spec: 128 random bits with byte 6's high nibble set to 0x4
    and byte 8's two high bits set to 0b10. We do the bit-fiddling
    ourselves so the bytes come from our seeded RNG rather than
    ``os.urandom`` (which is seedable only by patching the module).
    """
    raw = rng.bytes(16 * n)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, 16).copy()
    arr[:, 6] = (arr[:, 6] & 0x0F) | 0x40
    arr[:, 8] = (arr[:, 8] & 0x3F) | 0x80
    hex_arr = np.asarray(
        [b.tobytes().hex() for b in arr],
        dtype=object,
    )
    # Shape "8-4-4-4-12" canonical UUID format.
    return np.asarray(
        [
            f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
            for h in hex_arr
        ],
        dtype=object,
    )


def _draw_provinces(
    regions: pd.Series,
    rng: np.random.Generator,
    province_table: pd.DataFrame,
) -> np.ndarray:
    """Vectorised province draw, conditioned on each row's macro-region.

    Returns a numpy ``object`` array of Vietnamese province labels, in
    the same order as ``regions``. Within each region the marginal
    follows V02.01 population shares (``province_table.weight``).
    """
    out = np.empty(len(regions), dtype=object)
    for region, idx in regions.groupby(regions, observed=True).groups.items():
        sub = province_table[province_table["region"] == region]
        if sub.empty:
            log.warning("No provinces for region %r — falling back to label-as-province", region)
            out[np.asarray(idx)] = str(region)
            continue
        names = sub["province"].to_numpy()
        weights = sub["weight"].to_numpy()
        idx_arr = np.asarray(idx)
        choices = rng.choice(names, size=len(idx_arr), replace=True, p=weights)
        out[idx_arr] = choices
    return out


def _add_en_mirrors(structured: pd.DataFrame) -> pd.DataFrame:
    """Add the English mirror columns the renderer needs for the en pass."""
    out = structured.copy()
    out["region_en"] = out["region"].map(REGIONS_VI_TO_EN).fillna(out["region"])
    out["area_en"] = out["area"].map(AREA_VI_TO_EN).fillna(out["area"])
    out["sex_en"] = out["sex"].map(SEX_VI_TO_EN).fillna(out["sex"])
    out["marital_status_en"] = out["marital_status"].map(MARITAL_VI_TO_EN).fillna(
        out["marital_status"]
    )
    out["education_level_en"] = out["education_level"].map(EDUCATION_VI_TO_EN).fillna(
        out["education_level"]
    )
    out["occupation_en"] = out["occupation"].map(OCCUPATION_VI_TO_EN).fillna(
        out["occupation"]
    )
    out["province_en"] = out["province"].map(PROVINCE_VI_TO_EN).fillna(out["province"])
    return out


def _materialise_chunk(
    structured_chunk: pd.DataFrame,
    narratives: pd.DataFrame,
    *,
    language: str,
) -> pd.DataFrame:
    """Assemble the final 22-column DataFrame in canonical order.

    For ``language="vi"`` the structured columns hold their canonical
    Vietnamese labels; for ``"en"`` they're swapped for the English
    mirrors.
    """
    if language == "vi":
        cols = {
            "uuid":           structured_chunk["uuid"].to_numpy(),
            "sex":            structured_chunk["sex"].to_numpy(),
            "age":            structured_chunk["age"].astype("int64").to_numpy(),
            "marital_status": structured_chunk["marital_status"].to_numpy(),
            "education_level":structured_chunk["education_level"].to_numpy(),
            "occupation":     structured_chunk["occupation"].to_numpy(),
            "region":         structured_chunk["region"].to_numpy(),
            "area":           structured_chunk["area"].to_numpy(),
            "province":       structured_chunk["province"].to_numpy(),
            "country":        np.full(len(structured_chunk), COUNTRY_VI, dtype=object),
        }
    else:  # en
        cols = {
            "uuid":           structured_chunk["uuid"].to_numpy(),
            "sex":            structured_chunk["sex_en"].to_numpy(),
            "age":            structured_chunk["age"].astype("int64").to_numpy(),
            "marital_status": structured_chunk["marital_status_en"].to_numpy(),
            "education_level":structured_chunk["education_level_en"].to_numpy(),
            "occupation":     structured_chunk["occupation_en"].to_numpy(),
            "region":         structured_chunk["region_en"].to_numpy(),
            "area":           structured_chunk["area_en"].to_numpy(),
            "province":       structured_chunk["province_en"].to_numpy(),
            "country":        np.full(len(structured_chunk), COUNTRY_EN, dtype=object),
        }

    out = pd.DataFrame(cols, index=structured_chunk.index)
    out = pd.concat([out, narratives.reindex(out.index)], axis=1)
    return out[list(NEMOTRON_PERSONAS_VIETNAM_COLUMNS)]


def _stratified_uuid_subset(
    structured_full: pd.DataFrame,
    *,
    n_target: int,
    rng: np.random.Generator,
    stratify_by: Iterable[str],
) -> set[str]:
    """Return ``n_target`` uuids stratified by the given axes.

    Uses **proportional allocation**: each stratum's quota is its
    share of the parent population. Tiny strata always get at least
    one row, so the union may overshoot ``n_target`` by a handful —
    we trim back deterministically using the same RNG.
    """
    keys = structured_full[list(stratify_by)].astype(str).agg("|".join, axis=1)
    parent_n = len(structured_full)
    sizes = keys.value_counts()
    quotas = np.maximum(1, np.round(sizes * (n_target / parent_n)).astype(int))

    chosen: list[str] = []
    groups = structured_full.groupby(keys, observed=True).indices
    # Deterministic key ordering so output doesn't depend on Python's
    # dict-iteration-by-insertion (it does today, but lock it in anyway).
    for key in sorted(groups.keys()):
        idxs = groups[key]
        take = min(int(quotas.get(key, 0)), len(idxs))
        if take <= 0:
            continue
        sampled = rng.choice(idxs, size=take, replace=False)
        chosen.extend(structured_full.iloc[sampled]["uuid"].tolist())

    if len(chosen) > n_target:
        chosen.sort()
        idx = rng.choice(len(chosen), size=n_target, replace=False)
        chosen = [chosen[i] for i in idx]

    return set(chosen)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_nemotron_personas_vietnam_datasets(
    *,
    out_dir: str | Path | None = None,
    large_size: int = 3_000_000,
    small_size: int = 300_000,
    chunk_size: int = 100_000,
    seed: int = 42,
    pxweb_root: str | None = None,
    parquet_compression: str = "snappy",
    use_llm_pipeline: bool = False,
    llm_pipeline_config: Any = None,
) -> dict[str, Any]:
    """Build the four Nemotron-Personas-Vietnam parquet files.

    Returns a dict with paths and basic counts (handy for tests and
    progress reporting). See :class:`BuilderConfig` for the full
    parameter surface.

    Renderer choice:

    * ``use_llm_pipeline=False`` (default) — deterministic templated
      narrative renderer, no network or LLM dependency.
    * ``use_llm_pipeline=True``  — canonical Data Designer two-stage
      pipeline: PGM + OCEAN → LLM A → LLM B. Requires
      ``llm_pipeline_config`` (a
      :class:`packages.personas.llm.pipeline.LLMPipelineConfig`) with
      a model id, base URL, API key env var, and concurrency setting.
    """
    cfg = BuilderConfig(
        out_dir=Path(out_dir) if out_dir is not None else BuilderConfig.out_dir,
        large_size=large_size,
        small_size=small_size,
        chunk_size=chunk_size,
        seed=seed,
        pxweb_root=pxweb_root,
        parquet_compression=parquet_compression,
        use_llm_pipeline=use_llm_pipeline,
        llm_pipeline_config=llm_pipeline_config,
    )
    return _build(cfg)


def _build(cfg: BuilderConfig) -> dict[str, Any]:
    if cfg.small_size > cfg.large_size:
        raise ValueError(
            f"small_size={cfg.small_size} exceeds large_size={cfg.large_size}"
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------- Step 0: spawn seven independent RNG streams from one seed.
    # The 7th stream drives the OCEAN (Big Five) sampler, kept independent
    # so adding/removing personality doesn't perturb the structured PGM
    # or narrative-template draws — preserves seed reproducibility across
    # variants.
    seed_seq = np.random.SeedSequence(cfg.seed)
    (sq_struct, sq_uuid, sq_prov, sq_name, sq_narr_root, sq_subset,
     sq_ocean) = seed_seq.spawn(7)

    province_table = load_province_table(cfg.pxweb_root)

    # ------- Step 1: generate ALL structured personas first.
    # Required to compute the stratified small-subset uuid set before we
    # start streaming chunks, so each chunk knows which rows go to small.
    log.info(
        "step 1/4 — generate %s structured personas via VNPersonaGenerator (seed=%d)",
        f"{cfg.large_size:,}", int(sq_struct.entropy & 0x7FFFFFFF),
    )
    t0 = time.perf_counter()
    structured_full = _generate_structured(
        n=cfg.large_size, sq_struct=sq_struct,
    )
    log.info("  structured frame ready in %.1fs (%d rows × %d cols)",
             time.perf_counter() - t0,
             len(structured_full), len(structured_full.columns))

    # ------- Step 2: enrich with uuid / province / name (all rows at once;
    # the columns are tiny relative to the narrative columns we generate
    # later, so doing this up-front is cheaper than re-running per chunk).
    log.info("step 2/4 — uuids + provinces + names")
    rng_uuid = np.random.default_rng(sq_uuid)
    rng_prov = np.random.default_rng(sq_prov)
    rng_name = np.random.default_rng(sq_name)

    structured_full["uuid"] = _generate_uuids(cfg.large_size, rng_uuid)
    structured_full["province"] = _draw_provinces(
        structured_full["region"], rng_prov, province_table,
    )
    structured_full["name"] = generate_names(structured_full["sex"].to_numpy(), rng_name)
    structured_full = _add_en_mirrors(structured_full)

    # ------- Step 2b: Big Five (OCEAN) personality traits.
    # Sampled from peer-reviewed age × sex priors (Roberts 2006, Schmitt
    # 2008) — see packages.personas.ocean for citations and methodology.
    # Always computed, even on the templated path: the LLM pipeline
    # consumes the personality_summary_* columns, and the templated
    # path simply ignores them. Keeping OCEAN unconditional preserves
    # the seed contract regardless of which renderer the user picks.
    log.info("step 2b — OCEAN (Big Five) traits via Roberts/Schmitt priors")
    from packages.personas.ocean import sample_ocean
    rng_ocean = np.random.default_rng(sq_ocean)
    ocean_df = sample_ocean(
        age=structured_full["age"].to_numpy(),
        sex=structured_full["sex"].to_numpy(),
        rng=rng_ocean,
    )
    structured_full = pd.concat([structured_full, ocean_df], axis=1)

    # ------- Step 3: pick the stratified uuid subset for *-small-*.
    log.info(
        "step 3/4 — stratified subset (%s/%s ≈ %.1f%%) by %s",
        f"{cfg.small_size:,}", f"{cfg.large_size:,}",
        100 * cfg.small_size / cfg.large_size, list(cfg.stratify_by),
    )
    rng_subset = np.random.default_rng(sq_subset)
    small_uuids = _stratified_uuid_subset(
        structured_full,
        n_target=cfg.small_size,
        rng=rng_subset,
        stratify_by=cfg.stratify_by,
    )
    log.info("  small subset has %d uuids", len(small_uuids))

    # ------- Step 4: pre-compute narrative picks for the templated path
    # OR open LLM clients for the LLM-pipeline path. Either way, the
    # downstream chunk loop calls a single ``_render`` lambda that
    # produces the 11 narrative columns for the chunk.
    rng_narr = np.random.default_rng(sq_narr_root)
    picks = precompute_narrative_picks(cfg.large_size, rng_narr)

    llm_client_a = None
    llm_client_b = None
    if cfg.use_llm_pipeline:
        if cfg.llm_pipeline_config is None:
            raise ValueError(
                "use_llm_pipeline=True requires llm_pipeline_config "
                "(an LLMPipelineConfig instance)"
            )
        from packages.personas.llm.pipeline import (
            open_clients,
            render_narratives_via_llm,
        )
        log.info(
            "step 4/4 — LLM pipeline (PGM + OCEAN → LLM A → LLM B), "
            "model=%s, concurrency=%d",
            cfg.llm_pipeline_config.llm.model,
            cfg.llm_pipeline_config.llm.concurrency,
        )
        llm_client_a, llm_client_b = open_clients(cfg.llm_pipeline_config)

        def _render(chunk_df: pd.DataFrame, lang: str, start: int, end: int) -> pd.DataFrame:
            return render_narratives_via_llm(
                chunk_df, lang=lang, cfg=cfg.llm_pipeline_config,
                client_a=llm_client_a, client_b=llm_client_b,
            )
    else:
        log.info("step 4/4 — templated narrative renderer + stream-write 4 parquet files")

        def _render(chunk_df: pd.DataFrame, lang: str, start: int, end: int) -> pd.DataFrame:
            picks_chunk = {k: v[start:end] for k, v in picks.items()}
            return render_narratives(chunk_df, lang, picks_chunk)

    paths = {
        ("vi", "large"): out_dir / _filename("vi", "large"),
        ("en", "large"): out_dir / _filename("en", "large"),
        ("vi", "small"): out_dir / _filename("vi", "small"),
        ("en", "small"): out_dir / _filename("en", "small"),
    }
    writers = {
        key: pq.ParquetWriter(
            str(p),
            PYARROW_SCHEMA,
            compression=cfg.parquet_compression,
        )
        for key, p in paths.items()
    }
    rows_written = {key: 0 for key in writers}

    n_chunks = (cfg.large_size + cfg.chunk_size - 1) // cfg.chunk_size

    try:
        for ch_id in range(n_chunks):
            start = ch_id * cfg.chunk_size
            end = min(start + cfg.chunk_size, cfg.large_size)
            chunk = structured_full.iloc[start:end]
            in_small = chunk["uuid"].isin(small_uuids).to_numpy()

            narr_vi = _render(chunk, "vi", start, end)
            narr_en = _render(chunk, "en", start, end)

            df_vi = _materialise_chunk(chunk, narr_vi, language="vi")
            df_en = _materialise_chunk(chunk, narr_en, language="en")

            tbl_vi = pa.Table.from_pandas(df_vi, schema=PYARROW_SCHEMA, preserve_index=False)
            tbl_en = pa.Table.from_pandas(df_en, schema=PYARROW_SCHEMA, preserve_index=False)
            writers[("vi", "large")].write_table(tbl_vi)
            writers[("en", "large")].write_table(tbl_en)
            rows_written[("vi", "large")] += len(df_vi)
            rows_written[("en", "large")] += len(df_en)

            if in_small.any():
                df_vi_small = df_vi.loc[in_small]
                df_en_small = df_en.loc[in_small]
                tbl_vi_small = pa.Table.from_pandas(
                    df_vi_small, schema=PYARROW_SCHEMA, preserve_index=False
                )
                tbl_en_small = pa.Table.from_pandas(
                    df_en_small, schema=PYARROW_SCHEMA, preserve_index=False
                )
                writers[("vi", "small")].write_table(tbl_vi_small)
                writers[("en", "small")].write_table(tbl_en_small)
                rows_written[("vi", "small")] += len(df_vi_small)
                rows_written[("en", "small")] += len(df_en_small)

            if (ch_id + 1) % max(1, n_chunks // 20) == 0 or ch_id == n_chunks - 1:
                log.info(
                    "  chunk %d/%d (%.0f%%) — rows: large=%s small=%s",
                    ch_id + 1, n_chunks,
                    100 * (ch_id + 1) / n_chunks,
                    f"{rows_written[('vi', 'large')]:,}",
                    f"{rows_written[('vi', 'small')]:,}",
                )
    finally:
        for w in writers.values():
            w.close()
        if llm_client_a is not None:
            llm_client_a.close()
        if llm_client_b is not None:
            llm_client_b.close()

    summary = {
        "out_dir":   str(out_dir),
        "seed":      cfg.seed,
        "rows":      rows_written,
        "paths":     {f"{lang}-{size}": str(p) for (lang, size), p in paths.items()},
        "schema":    list(NEMOTRON_PERSONAS_VIETNAM_COLUMNS),
    }
    log.info("complete — %s", summary["paths"])
    return summary


def _generate_structured(
    *,
    n: int,
    sq_struct: np.random.SeedSequence,
) -> pd.DataFrame:
    """Drive ``VNPersonaGenerator`` with the structured-stream seed.

    Imported lazily so importing :mod:`packages.personas.datasets` does not pull
    pgmpy / numpy-tweaks unless we're actually building.
    """
    from packages.personas.pgm.generator import VNPersonaGenerator

    # SDG-PGMs takes a Python int; derive one deterministically from
    # the SeedSequence so the entire build remains driven by one seed.
    seed_int = int(np.random.default_rng(sq_struct).integers(0, 2**31 - 1))

    gen = VNPersonaGenerator(seed=seed_int)
    df = gen.generate_samples(size=n, seed=seed_int, disable_progress_bar=True)
    # The generator runs its post-processing chain (age derivation, name
    # injection, bilingual bio). We re-do name + bio downstream in our
    # own renderer, but the structured columns are exactly what we need.
    keep = ["region", "area" if "area" in df.columns else "urbanicity",
            "age_group", "age", "sex", "ethnicity",
            "marital_status", "education_level",
            "employment_status", "occupation", "industry_sector"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].reset_index(drop=True).copy()
    if "urbanicity" in out.columns and "area" not in out.columns:
        out = out.rename(columns={"urbanicity": "area"})
    return out
