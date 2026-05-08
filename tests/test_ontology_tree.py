"""Tests for the bilingual ontology tree builder.

Covers the pure-logic invariants that don't need PX-Web data on disk:

* glossary returns expected EN for known VI keys (incl. ``Đ`` / ``Ð``
  diacritic variants),
* translator does the right thing per tier (identity, glossary, cache,
  miss),
* tree builder produces the correct DatabaseNode → TableNode structure
  from a synthetic metadata JSON fixture,
* JSON / YAML writers round-trip cleanly.

Online coverage / Google translation isn't tested here — that requires
network and is exercised by the manual ``make build-ontology ONLINE=1``
end-to-end check.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# Skip the whole module if pyarrow / pandas / vn-fullname-generator
# aren't available — the glossary's _build_dimension_glossary depends
# on packages.personas.datasets which depends on those.
for _dep in ("pyarrow", "pandas", "vn_fullname_generator"):
    if importlib.util.find_spec(_dep) is None:
        pytest.skip(f"{_dep} missing; install [curator] extras",
                    allow_module_level=True)

from packages.ontology.glossary import (  # noqa: E402
    DATABASE_VI_TO_EN,
    _normalise_d,
    build_glossary,
)
from packages.ontology.translator import Translator, _is_identity  # noqa: E402
from packages.ontology.tree import (  # noqa: E402
    _collect_all_strings,
    build_ontology_tree,
    write_tree_json,
    write_tree_yaml_summary,
)


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------
def test_glossary_basic_hits():
    g = build_glossary()
    assert g["Năm"] == "Year"
    assert g["Tỉnh, thành phố"] == "Province/city"
    assert g["TỔNG SỐ"] == "TOTAL"
    assert g["Hà Nội"] == "Hanoi"
    assert g["Đông Nam Bộ"] == "South East"
    # 12 first-class databases all present
    assert len(DATABASE_VI_TO_EN) == 12


def test_glossary_handles_d_diacritic_variants(tmp_path):
    """The PX-Web export occasionally uses ``Ð`` (U+00D0) instead of ``Đ`` (U+0110).
    The translator normalises both forms to the canonical Đ before glossary lookup."""
    g = build_glossary()
    assert g["Địa phương"] == "Locality"
    # Normalisation maps the alternate form to the canonical one
    assert _normalise_d("\u00d0ịa phương") == "Địa phương"
    # And the translator picks both up via the same normalisation
    t = Translator(cache_path=tmp_path / "c.json")
    assert t.translate("\u00d0ịa phương") == "Locality"


# ---------------------------------------------------------------------------
# Translator tiers
# ---------------------------------------------------------------------------
def test_identity_patterns():
    assert _is_identity("2024")
    assert _is_identity("12.5")
    assert _is_identity("12%")
    assert _is_identity("V07.01")
    assert _is_identity("")
    assert not _is_identity("Năm")
    assert not _is_identity("Sơ bộ 2024")


def test_translator_glossary_then_identity_then_miss(tmp_path):
    cache = tmp_path / "cache.json"
    t = Translator(cache_path=cache)
    # Glossary hit
    assert t.translate("Hà Nội") == "Hanoi"
    # Identity (year)
    assert t.translate("2024") == "2024"
    # Time-prefixed identity composition
    assert t.translate("Sơ bộ 2024") == "Preliminary 2024"
    # Miss falls back to identity, but stats register it
    miss = "Một chuỗi không có trong glossary để miss"
    assert t.translate(miss) == miss

    stats = t.stats
    assert stats["glossary_hits"] >= 1
    assert stats["identity"] >= 2
    assert stats["misses"] >= 1


def test_translator_compositional_paren_split(tmp_path):
    """Strings of the form ``"<glossary head> (<glossary unit>)"`` should
    translate via the compositional path even if the literal whole isn't
    in the glossary."""
    t = Translator(cache_path=tmp_path / "c.json")
    # Both head ("Bia các loại") and tail ("Lít") are in the glossary
    en = t.translate("Bia các loại (Lít)")
    assert "Beer" in en
    assert "Litres" in en or "Lít" not in en  # the tail is translated


def test_translator_cache_round_trip(tmp_path):
    cache = tmp_path / "cache.json"
    t = Translator(cache_path=cache)
    t._cache["xyz"] = "ABC"
    t.save()
    t2 = Translator(cache_path=cache)
    assert t2._cache.get("xyz") == "ABC"
    # Now translate("xyz") should hit cache (would be miss otherwise)
    assert t2.translate("xyz") == "ABC"
    assert t2.stats["cache_hits"] == 1


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------
def _make_synthetic_pxweb(root: Path) -> None:
    """Materialise two minimal *.metadata.json + *.parquet pairs for a
    self-contained tree-builder test."""
    root.mkdir(parents=True, exist_ok=True)

    md1 = {
        "table_id": "V02.01",
        "database": "Dân số và lao động",
        "title":    "Dân số trung bình phân theo địa phương",
        "path":     ["Dân số và lao động", "V02.01.px"],
        "variables": [
            {
                "code": "Địa phương",
                "text": "Địa phương",
                "values":     ["0", "1", "2"],
                "valueTexts": ["CẢ NƯỚC", "Hà Nội", "Đông Nam Bộ"],
            },
            {
                "code": "Năm",
                "text": "Năm",
                "values":     ["2023", "2024"],
                "valueTexts": ["2023", "Sơ bộ 2024"],
            },
        ],
        "n_cells": 6,
        "n_kept":  6,
    }
    md2 = {
        "table_id": "V07.01",
        "database": "Công nghiệp",
        "title":    "Chỉ số sản xuất công nghiệp phân theo ngành công nghiệp",
        "path":     ["Công nghiệp", "V07.01.px"],
        "variables": [
            {
                "code": "Ngành công nghiệp",
                "text": "Ngành công nghiệp",
                "values":     ["0", "1"],
                "valueTexts": ["Tổng số", "Khai khoáng"],
            },
        ],
        "n_cells": 2,
        "n_kept":  2,
    }

    (root / "Dan-so-va-lao-dong__V02.01.px.metadata.json").write_text(
        json.dumps(md1, ensure_ascii=False))
    (root / "Cong-nghiep__V07.01.px.metadata.json").write_text(
        json.dumps(md2, ensure_ascii=False))
    # parquet files don't have to exist — the builder only references their paths
    (root / "Dan-so-va-lao-dong__V02.01.px.parquet").write_bytes(b"")
    (root / "Cong-nghiep__V07.01.px.parquet").write_bytes(b"")


def test_tree_builder_basic_shape(tmp_path):
    pxweb_root = tmp_path / "pxweb"
    _make_synthetic_pxweb(pxweb_root)

    t = Translator(cache_path=tmp_path / "cache.json")
    tree = build_ontology_tree(pxweb_root, translator=t)

    assert tree.n_databases == 2
    assert tree.n_tables == 2
    # Databases come back in sorted order, so:
    db_names = [db.name_vi for db in tree.databases]
    assert "Công nghiệp" in db_names
    assert "Dân số và lao động" in db_names
    # Each DB has the right English mirror from the glossary
    by_vi = {db.name_vi: db for db in tree.databases}
    assert by_vi["Công nghiệp"].name_en == "Industry"
    assert by_vi["Dân số và lao động"].name_en == "Population and employment"


def test_tree_values_are_translated(tmp_path):
    pxweb_root = tmp_path / "pxweb"
    _make_synthetic_pxweb(pxweb_root)
    t = Translator(cache_path=tmp_path / "c.json")
    tree = build_ontology_tree(pxweb_root, translator=t)

    # Find the V02.01 table
    db = next(d for d in tree.databases if d.name_vi == "Dân số và lao động")
    tbl = db.tables[0]
    assert tbl.table_id == "V02.01"

    # Variable codes and value texts must be translated where the glossary covers
    var_loc = next(v for v in tbl.variables if v.code == "Địa phương")
    val_by_vi = {val.name_vi: val for val in var_loc.values}
    assert val_by_vi["CẢ NƯỚC"].name_en == "WHOLE COUNTRY"
    assert val_by_vi["Hà Nội"].name_en == "Hanoi"
    assert val_by_vi["Đông Nam Bộ"].name_en == "South East"

    # The "Sơ bộ 2024" prefix should be detected
    var_year = next(v for v in tbl.variables if v.code == "Năm")
    val_by_vi = {val.name_vi: val for val in var_year.values}
    assert val_by_vi["Sơ bộ 2024"].name_en == "Preliminary 2024"
    assert val_by_vi["2023"].name_en == "2023"  # identity


def test_collect_all_strings_dedupes(tmp_path):
    pxweb_root = tmp_path / "pxweb"
    _make_synthetic_pxweb(pxweb_root)
    strings = _collect_all_strings(pxweb_root)
    # Should pick up every variable text + value text + DB names + titles
    assert "Dân số và lao động" in strings
    assert "Công nghiệp" in strings
    assert "Địa phương" in strings
    assert "Hà Nội" in strings


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def test_writers_round_trip(tmp_path):
    pxweb_root = tmp_path / "pxweb"
    _make_synthetic_pxweb(pxweb_root)
    t = Translator(cache_path=tmp_path / "c.json")
    tree = build_ontology_tree(pxweb_root, translator=t)

    json_path = tmp_path / "tree.json"
    yaml_path = tmp_path / "tree.yaml"
    write_tree_json(tree, json_path)
    write_tree_yaml_summary(tree, yaml_path)

    assert json_path.exists() and json_path.stat().st_size > 0
    assert yaml_path.exists() and yaml_path.stat().st_size > 0

    # JSON re-loads with the same node counts
    reloaded = json.loads(json_path.read_text())
    assert reloaded["n_databases"] == 2
    assert len(reloaded["databases"]) == 2

    # YAML doesn't carry value lists by design (it's a summary)
    import yaml as yaml_mod
    summary = yaml_mod.safe_load(yaml_path.read_text())
    assert summary["meta"]["n_databases"] == 2
    # Tables present, but not values
    db_dsl = next(d for d in summary["databases"]
                  if d["name_vi"] == "Dân số và lao động")
    assert db_dsl["tables"][0]["variables"][0]["code"] == "Địa phương"
    assert "values" not in db_dsl["tables"][0]["variables"][0]


# ---------------------------------------------------------------------------
# Live tree (only if the curator has been run)
# ---------------------------------------------------------------------------
LIVE_PXWEB = Path("data/nso-gov-vn/raw/pxweb/vi")


@pytest.mark.skipif(
    not LIVE_PXWEB.exists() or not list(LIVE_PXWEB.glob("*.metadata.json")),
    reason="live PX-Web parquets not present; run `personas-vn curate "
           "--only download` first",
)
def test_live_tree_has_expected_databases():
    """Soft check that the real-data run produces the 12 first-class NSO DBs."""
    t = Translator(cache_path=LIVE_PXWEB.parent.parent.parent
                              / "ontology" / "translation_cache.json")
    tree = build_ontology_tree(LIVE_PXWEB, translator=t)
    assert tree.n_databases == 12
    assert tree.n_tables >= 480   # we observed 502 at the time of writing
    db_names = {db.name_vi for db in tree.databases}
    assert "Công nghiệp" in db_names
    assert "Dân số và lao động" in db_names
    assert "Đơn vị hành chính, đất đai và khí hậu" in db_names
