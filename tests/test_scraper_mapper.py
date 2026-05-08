"""Tests for the NSO scraper's pure functions (mapper + fixture loader).

We hit no network here — we feed the mapper handcrafted records that mirror
the real WordPress shape.
"""

from __future__ import annotations

from packages.ontology.registry import load_registry
from packages.scraper import load_fixture_records, map_posts_to_datasets


def test_fixture_loader_shape():
    fixture = load_fixture_records()
    assert set(fixture.keys()) == {"categories", "tags", "posts"}
    assert all("id" in c for c in fixture["categories"])
    assert all("id" in p for p in fixture["posts"])


def test_mapper_basic_mapping():
    reg = load_registry()
    fixture = load_fixture_records()
    datasets = map_posts_to_datasets(fixture, reg)
    assert datasets, "mapper produced no datasets"
    assert len(datasets) == len(fixture["posts"])

    # Every dataset id is namespaced.
    assert all(d.id.startswith("nso:") for d in datasets)
    # No dataset ends up with the catch-all domain when its category is one
    # of the well-known ones in our fixture.
    domain_by_title = {d.title: d.domain_id for d in datasets}
    assert "Labour force survey, Q1 2026" in domain_by_title
    assert domain_by_title["Labour force survey, Q1 2026"] == "employment"
    assert domain_by_title["Population projections, 2026–2050"] == "population"
    assert domain_by_title["GDP, Q4 2025, by economic region"] == "national_accounts"


def test_mapper_strips_html_in_summary():
    reg = load_registry()
    fixture = load_fixture_records()
    datasets = map_posts_to_datasets(fixture, reg)
    for d in datasets:
        if d.summary:
            assert "<p>" not in d.summary
            assert "</p>" not in d.summary


def test_mapper_skips_records_without_id():
    reg = load_registry()
    fixture = load_fixture_records()
    fixture["posts"].append({"date": "2026-01-01", "title": {"rendered": "ghost"}})
    datasets = map_posts_to_datasets(fixture, reg)
    assert len(datasets) == len([p for p in fixture["posts"] if p.get("id") is not None])
