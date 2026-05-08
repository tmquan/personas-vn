"""Single-bio LLM enrichment for a saved ``PersonaBatch``.

This is the older, narrower companion to :mod:`packages.personas.llm.pipeline`
(which runs the full Data Designer LLM-A + LLM-B for the Nemotron parquet
dataset). ``enrich.py`` is the lightweight pass that takes a
``PersonaBatch`` (e.g. ``data/personas/personas.json`` produced by
``personas-vn generate``) and overwrites each persona's ``bio_vi`` /
``bio_en`` with an LLM-written bilingual narrative.

It uses the shared :class:`packages.personas.llm.client.LLMClient` for
the HTTP layer (concurrent worker pool + retry + cache), so the
networking, JSON parsing, and rate-limit logic stay in one place.

Cost note: 100K personas at ~400 input + 300 output tokens × $0.50/1M
input + $1.50/1M output ≈ $65 of API spend. Defaults cap the run at
50 personas; pass ``--max-records 0`` to enrich everything.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.common.logging import get_logger
from packages.common.paths import resolve
from packages.ontology import PersonaBatch
from packages.personas.llm.client import JSONLCache, LLMClient, LLMConfig

log = get_logger(__name__)


@dataclass
class LLMEnrichConfig:
    """Knobs for :func:`enrich_personas`. Maps onto :class:`LLMConfig`
    plus the enrichment-specific subset selection."""

    # OpenAI-compatible provider config — all routed through LLMClient
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "PERSONAS_VN_LLM_API_KEY"
    model: str = "nvidia/nemotron-3-super-49b-instruct-v2"

    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 600
    timeout_s: float = 60.0
    retries: int = 2
    retry_backoff_s: float = 2.0
    request_delay_s: float = 0.0      # rate-limit cushion (sequential pacing)

    # How many personas to enrich (None or 0 = all)
    max_records: int | None = 50

    # Subset selection: random sample vs first-N
    random_sample: bool = True
    random_state: int = 20260505

    # Optional disk cache (resumable enrichments)
    cache_path: str | Path | None = None


# ---------------------------------------------------------------------------
# Prompt — Vietnamese instruction, JSON-only response
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "Bạn là một trợ lý chuyên viết tiểu sử nhân vật ảo (synthetic personas) "
    "dựa trên dữ liệu thống kê dân số chính thức của Việt Nam. "
    "Mỗi tiểu sử cần ngắn gọn (3-5 câu), mang đậm văn hoá Việt Nam, và "
    "phản ánh trung thực các thuộc tính nhân khẩu học - kinh tế xã hội đã cung cấp. "
    "Bạn LUÔN trả lời bằng JSON hợp lệ duy nhất với hai trường: "
    '{"bio_vi": "...", "bio_en": "..."}. '
    "Không thêm bất kỳ văn bản nào khác bên ngoài JSON."
)


def _build_user_prompt(record: dict[str, Any]) -> str:
    """Build the user prompt from a persona record."""
    return (
        "Hãy viết tiểu sử cho nhân vật ảo sau (3-5 câu) bằng tiếng Việt tự nhiên, "
        "rồi dịch sang tiếng Anh. Trả về JSON {\"bio_vi\": ..., \"bio_en\": ...}.\n\n"
        f"- Tên: {record.get('name')}\n"
        f"- Tuổi: {record.get('age')} (nhóm tuổi {record.get('age_group')})\n"
        f"- Giới tính: {record.get('sex')}\n"
        f"- Dân tộc: {record.get('ethnicity')}\n"
        f"- Vùng kinh tế: {record.get('region')}\n"
        f"- Khu vực sinh sống: {record.get('urbanicity')}\n"
        f"- Tình trạng hôn nhân: {record.get('marital_status')}\n"
        f"- Trình độ chuyên môn kỹ thuật: {record.get('education_level')}\n"
        f"- Vị thế việc làm: {record.get('employment_status')}\n"
        f"- Nghề nghiệp: {record.get('occupation')}\n"
        f"- Ngành kinh tế: {record.get('industry_sector')}\n"
        f"- Nhóm thu nhập (ngũ phân vị): {record.get('income_quintile')}\n"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def enrich_personas(
    batch: PersonaBatch,
    *,
    config: LLMEnrichConfig | None = None,
    out_path: str | Path | None = None,
) -> PersonaBatch:
    """Enrich a (subset of) ``PersonaBatch`` with LLM-written bilingual bios.

    Returns a new :class:`PersonaBatch` with ``enrichment`` populated and
    every successfully-enriched persona's ``bio_vi`` / ``bio_en`` /
    ``enriched`` updated. Personas the LLM failed on are left as-is.
    """
    cfg = config or LLMEnrichConfig()
    if not os.environ.get(cfg.api_key_env):
        raise RuntimeError(
            f"missing API key: set the {cfg.api_key_env} environment variable "
            f"to a key valid for {cfg.base_url}"
        )

    # Pick which personas to enrich
    n_total = len(batch.personas)
    cap = cfg.max_records if (cfg.max_records and cfg.max_records > 0) else n_total
    cap = min(cap, n_total)
    if cap == 0:
        log.warning("nothing to enrich (max_records=0 and batch is empty)")
        return batch

    if cfg.random_sample and cap < n_total:
        import numpy as np
        rng = np.random.default_rng(cfg.random_state)
        idx = sorted(rng.choice(n_total, size=cap, replace=False).tolist())
    else:
        idx = list(range(cap))

    # Spin up the unified client (HTTP + retry + optional cache).
    cache_path = Path(cfg.cache_path) if cfg.cache_path else \
                 Path("data/personas/_enrich_cache.jsonl")
    client = LLMClient(
        cfg=LLMConfig(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.timeout_s,
            retries=cfg.retries,
            retry_backoff_s=cfg.retry_backoff_s,
            concurrency=1,                 # single-stream pacing for now
            request_delay_s=cfg.request_delay_s,
        ),
        cache=JSONLCache(path=cache_path),
    )

    log.info("enriching %d / %d personas with %s @ %s",
             cap, n_total, cfg.model, cfg.base_url)
    started = datetime.now(timezone.utc)
    n_ok = 0
    n_fail = 0

    try:
        for i, row_idx in enumerate(idx):
            persona = batch.personas[row_idx]
            obj = client.call_one(
                cache_key=f"enrich|{cfg.model}|{persona.persona_id}",
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(persona.model_dump()),
            )
            if not obj or not obj.get("bio_vi") or not obj.get("bio_en"):
                n_fail += 1
                log.warning("enrich failed for %s", persona.persona_id)
                continue
            persona.bio_vi = str(obj["bio_vi"]).strip()
            persona.bio_en = str(obj["bio_en"]).strip()
            persona.enriched = True
            n_ok += 1
            if (i + 1) % 10 == 0:
                log.info("  enriched %d/%d (ok=%d, fail=%d)",
                         i + 1, cap, n_ok, n_fail)
            if cfg.request_delay_s:
                time.sleep(cfg.request_delay_s)
    finally:
        client.close()

    finished = datetime.now(timezone.utc)
    batch.enrichment = {
        "model":      cfg.model,
        "base_url":   cfg.base_url,
        "n_attempted": cap,
        "n_enriched":  n_ok,
        "n_failed":    n_fail,
        "started_at":  started.isoformat(),
        "finished_at": finished.isoformat(),
    }
    log.info("enrichment complete: ok=%d fail=%d in %.1fs",
             n_ok, n_fail, (finished - started).total_seconds())

    if out_path is not None:
        path = Path(resolve(out_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
        log.info("wrote enriched batch -> %s", path)

    return batch


# ---------------------------------------------------------------------------
# Convenience: enrich from a saved batch on disk
# ---------------------------------------------------------------------------
def enrich_personas_from_disk(
    *,
    personas_path: str | Path = "data/personas/personas.json",
    out_path: str | Path = "data/personas/personas_enriched.json",
    config: LLMEnrichConfig | None = None,
) -> dict[str, Any]:
    """Load a persisted ``PersonaBatch``, enrich, write output."""
    p = Path(resolve(personas_path))
    if not p.exists():
        raise FileNotFoundError(f"personas not found: {p}")
    batch = PersonaBatch.model_validate_json(p.read_text(encoding="utf-8"))
    enriched = enrich_personas(batch, config=config, out_path=out_path)
    return enriched.enrichment
