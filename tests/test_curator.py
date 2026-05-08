"""Curator pipeline tests.

We exercise:

* Stage wiring: every stage is constructible from the YAML config and
  ``setup()`` / ``teardown()`` are idempotent.
* Parse + Extract on a tiny synthetic JSONL so we don't need network or a
  heavy embedding model in CI.

The Embed and Reduce stages need ``sentence-transformers`` and
``umap-learn`` respectively; we skip those when the optional ``[curator]``
extras are absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from packages.common.config import Config
from packages.curator import (
    EmbedStage,
    ExtractStage,
    ParseStage,
    ReduceStage,
    load_config,
)


def _make_raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    posts_path = raw / "posts__en.jsonl"
    cats_path = raw / "categories__en.jsonl"
    posts = [
        {
            "id": 100,
            "lang": "en",
            "date": "2026-04-15T08:00:00",
            "slug": "labour-force-q1",
            "link": "https://www.nso.gov.vn/employment/2026/04/labour-force-q1/",
            "title": {"rendered": "Labour force survey Q1 2026"},
            "excerpt": {"rendered": "<p>Headline labour force participation 68.7%.</p>"},
            "content": {"rendered": "<p>Labour force participation continued to be stable in Q1.</p>"},
            "categories": [14],
            "tags": [],
        },
        {
            "id": 101,
            "lang": "en",
            "date": "2026-03-22T08:00:00",
            "slug": "industry-iip-2025",
            "link": "https://www.nso.gov.vn/cong-nghiep/2026/03/industry-iip-2025/",
            "title": {"rendered": "Industrial Production Index 2025"},
            "excerpt": {"rendered": "<p>IIP rose 8.3% in 2025 led by manufacturing.</p>"},
            "content": {"rendered": "<p>Manufacturing contributed 6.8 percentage points to the headline.</p>"},
            "categories": [28],
            "tags": [],
        },
        {
            "id": 102,
            "lang": "en",
            "date": "2026-02-10T08:00:00",
            "slug": "tiny",
            "link": "https://www.nso.gov.vn/x/2026/02/tiny/",
            "title": {"rendered": "x"},
            "excerpt": {"rendered": ""},
            "content": {"rendered": ""},
            "categories": [],
            "tags": [],
        },
    ]
    cats = [
        {"id": 14, "name": "Employment", "slug": "employment", "parent": 0},
        {"id": 28, "name": "Industry",   "slug": "industry",   "parent": 0},
    ]
    posts_path.write_text("\n".join(json.dumps(p) for p in posts), encoding="utf-8")
    cats_path.write_text("\n".join(json.dumps(c) for c in cats), encoding="utf-8")
    return raw


def test_load_config_paths(tmp_path):
    cfg = load_config()
    assert cfg.name == "nso-gov-vn"
    # Stage configs are nested Config objects.
    assert "model" in cfg.embed
    assert "algorithm" in cfg.reduce


def test_parse_and_extract_local(tmp_path):
    raw = _make_raw_dir(tmp_path)
    parsed = tmp_path / "parsed"
    extracted = tmp_path / "extracted"
    parsed.mkdir()
    extracted.mkdir()

    parse_stage = ParseStage(Config({"min_text_chars": 30}), raw, parsed)
    parse_stage.setup()
    parse_summary = parse_stage.run()
    parse_stage.teardown()
    assert parse_summary["kept"] == 2          # tiny record dropped
    assert parse_summary["dropped"] == 1
    parsed_records = [
        json.loads(line)
        for line in (parsed / "parsed.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all("text" in r and r["text"] for r in parsed_records)
    assert all(r["lang"] == "en" for r in parsed_records)

    extract_stage = ExtractStage(
        Config({"top_keywords": 5, "vectorizer": "tfidf",
                "ngram_range": [1, 1], "max_df": 1.0, "min_df": 1}),
        parsed,
        extracted,
    )
    extract_stage.setup()
    extract_summary = extract_stage.run()
    extract_stage.teardown()
    assert extract_summary["n"] == 2
    extracted_records = [
        json.loads(line)
        for line in (extracted / "extracted.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    domains = {r["domain_id"] for r in extracted_records}
    assert "employment" in domains
    assert "industry" in domains
    # Each non-tiny record gets keywords (TF-IDF over only 2 docs is degenerate
    # but still yields at least one feature).
    assert all(isinstance(r.get("keywords"), list) for r in extracted_records)


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (needs `pip install -e .[curator]`)",
)
def test_embed_stage_smoke(tmp_path):
    """End-to-end embed on 2 docs. Network access is required only the first
    time the model is downloaded; subsequent runs hit the HF cache.
    """
    raw = _make_raw_dir(tmp_path)
    parsed = tmp_path / "parsed"
    extracted = tmp_path / "extracted"
    embedded = tmp_path / "embedded"
    parsed.mkdir()
    extracted.mkdir()
    embedded.mkdir()

    ParseStage(Config({"min_text_chars": 30}), raw, parsed).run()
    ExtractStage(
        Config({"top_keywords": 5, "vectorizer": "tfidf",
                "ngram_range": [1, 1], "max_df": 1.0, "min_df": 1}),
        parsed,
        extracted,
    ).run()

    cfg = load_config()
    embed = EmbedStage(cfg.embed, extracted, embedded)
    embed.setup()
    summary = embed.run()
    embed.teardown()
    assert summary["n"] == 2
    assert summary["dim"] > 32


@pytest.mark.skipif(
    importlib.util.find_spec("umap") is None
    or importlib.util.find_spec("sentence_transformers") is None,
    reason="curator extras (umap-learn / sentence-transformers) not installed",
)
def test_reduce_stage_smoke(tmp_path):
    raw = _make_raw_dir(tmp_path)
    parsed = tmp_path / "parsed"
    extracted = tmp_path / "extracted"
    embedded = tmp_path / "embedded"
    reduced = tmp_path / "reduced"
    for d in (parsed, extracted, embedded, reduced):
        d.mkdir()

    ParseStage(Config({"min_text_chars": 30}), raw, parsed).run()
    ExtractStage(
        Config({"top_keywords": 5, "vectorizer": "tfidf",
                "ngram_range": [1, 1], "max_df": 1.0, "min_df": 1}),
        parsed,
        extracted,
    ).run()
    cfg = load_config()
    embed = EmbedStage(cfg.embed, extracted, embedded)
    embed.setup()
    embed.run()
    embed.teardown()

    # UMAP needs >= 4 points; PCA fallback covers fewer.
    # Clustering is opt-in (default ``False``) — flip it on here so the
    # ``cluster`` column path is exercised by the test.
    reduce = ReduceStage(
        Config({"algorithm": "pca", "n_components": 2, "cluster": True}),
        embedded,
        reduced,
    )
    reduce.setup()
    summary = reduce.run()
    reduce.teardown()
    assert summary["n"] == 2
    assert summary["clustered"] is True
    import pandas as pd
    df = pd.read_parquet(reduced / "reduced.parquet")
    assert {"x", "y", "cluster", "domain_id"}.issubset(df.columns)

    # Default config (no ``cluster`` key) must NOT produce a cluster column.
    reduced_default = tmp_path / "reduced_default"
    reduced_default.mkdir()
    reduce_default = ReduceStage(
        Config({"algorithm": "pca", "n_components": 2}),
        embedded,
        reduced_default,
    )
    reduce_default.setup()
    default_summary = reduce_default.run()
    reduce_default.teardown()
    assert default_summary["clustered"] is False
    df_default = pd.read_parquet(reduced_default / "reduced.parquet")
    assert {"x", "y"}.issubset(df_default.columns)
    assert "cluster" not in df_default.columns
