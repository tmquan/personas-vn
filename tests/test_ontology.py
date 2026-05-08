"""Ontology unit tests.

We exercise three things here:

* The registry loads the YAML config and produces a clean topological order
  for the persona dimensions.
* NSO category labels (English + Vietnamese, with diacritics) all map to
  the right stable domain ids.
* The registry catches obvious config mistakes (cycles, dangling parents).
"""

from __future__ import annotations

import pytest

from packages.common.config import Config
from packages.ontology.registry import OntologyRegistry, _normalise, load_registry


def test_load_registry_topological_order():
    reg = load_registry()
    order = reg.topological_order()
    seen: set[str] = set()
    for dim_id in order:
        dim = reg.get_dimension(dim_id)
        assert all(p in seen for p in dim.parents), (
            f"dimension {dim_id} placed before its parents {dim.parents}"
        )
        seen.add(dim_id)


@pytest.mark.parametrize(
    "label,slug,expected",
    [
        ("Population", "dan-so", "population"),
        ("Dân số", "dan-so", "population"),
        ("Employment", "employment", "employment"),
        ("Lao động", "lao-dong", "employment"),
        ("Education", "education", "education"),
        ("Giáo dục", "giao-duc", "education"),
        ("Industry", "industry", "industry"),
        ("Công nghiệp", "cong-nghiep", "industry"),
        ("Doanh nghiệp", "doanh-nghiep", "enterprises"),
        ("Đầu tư và Xây dựng", "dau-tu-va-xay-dung", "investment"),
        ("National Accounts", "national-accounts", "national_accounts"),
        ("Some unmapped category", "weird-slug-9999", "other"),
        (None, None, "other"),
    ],
)
def test_map_nso_category(label, slug, expected):
    reg = load_registry()
    assert reg.map_nso_category(label, slug) == expected


def test_normalise_strips_vietnamese_diacritics():
    assert _normalise("Dân số") == "dan-so"
    assert _normalise("Đầu tư và Xây dựng") == "dau-tu-va-xay-dung"


def test_registry_rejects_cycle():
    bad = Config(
        {
            "domains": [
                {"id": "x", "label_en": "X", "label_vi": "X", "aliases": []},
            ],
            "persona_dimensions": [
                {"id": "a", "domain": "x", "parents": ["b"], "cardinality": 2},
                {"id": "b", "domain": "x", "parents": ["a"], "cardinality": 2},
            ],
        }
    )
    with pytest.raises(ValueError, match="Cycle"):
        OntologyRegistry(bad)


def test_registry_rejects_dangling_parent():
    bad = Config(
        {
            "domains": [
                {"id": "x", "label_en": "X", "label_vi": "X", "aliases": []},
            ],
            "persona_dimensions": [
                {"id": "a", "domain": "x", "parents": ["does_not_exist"], "cardinality": 2},
            ],
        }
    )
    with pytest.raises(ValueError, match="Cycle"):
        OntologyRegistry(bad)
