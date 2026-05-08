"""Persona generator tests (SDG-PGMs-grounded).

We exercise the user-visible contract:

* Generation is deterministic given a seed.
* Every persona has all required structured fields, and the values are
  in the canonical Vietnamese vocabulary published by NSO PX-Web.
* Every Vietnamese categorical field has its English mirror populated.
* Marginal distributions look sane (more populous regions dominate).

The PX-Web parquets must be present under ``data/nso-gov-vn/raw/pxweb/vi/``;
the test is skipped otherwise (matches the project policy of allowing
``[curator]`` extras to be optional in CI).
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import pytest

from packages.common.paths import resolve

PXWEB_AVAILABLE = (resolve("data/nso-gov-vn/raw/pxweb/vi") / "Dan-so-va-lao-dong__V02.41.px.parquet").exists()
SDG_PGMS_AVAILABLE = True
try:
    import pgms  # noqa: F401
except ImportError:
    SDG_PGMS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not (PXWEB_AVAILABLE and SDG_PGMS_AVAILABLE),
    reason="needs sdg-pgms + downloaded PX-Web parquets (run `personas-vn curate --only download`)",
)


@pytest.fixture(scope="module")
def small_batch():
    warnings.filterwarnings("ignore")
    from packages.ontology.registry import load_registry
    from packages.personas.pgm import generate_personas

    return generate_personas(60, registry=load_registry(), seed=11)


def test_generate_is_deterministic_given_seed():
    warnings.filterwarnings("ignore")
    from packages.personas.pgm import generate_personas

    a = generate_personas(40, seed=42)
    b = generate_personas(40, seed=42)
    assert [p.persona_id for p in a.personas] == [p.persona_id for p in b.personas]
    assert [p.region for p in a.personas] == [p.region for p in b.personas]


def test_every_persona_has_bilingual_structured_fields(small_batch):
    """Every Vietnamese categorical column has an English mirror populated."""
    from packages.personas.pgm.distributions import (
        EDUCATION_LEVELS_VI,
        REGIONS_VI,
        SEXES_VI,
        URBANICITY_VI,
    )

    for p in small_batch.personas:
        assert p.region in REGIONS_VI
        assert p.region_en
        assert p.urbanicity in URBANICITY_VI
        assert p.urbanicity_en in ("urban", "rural")
        assert p.sex in SEXES_VI
        assert p.sex_en in ("male", "female")
        assert p.education_level in EDUCATION_LEVELS_VI
        assert p.education_level_en
        assert p.occupation_en  # ISCO label or fallback
        assert p.employment_status_en
        assert p.industry_sector  # may be the synthetic 'Không áp dụng' for non-workers
        assert isinstance(p.age, int) and 15 <= p.age <= 80
        assert p.persona_id.startswith("vn-")


def test_bilingual_bios_are_present_and_distinct(small_batch):
    for p in small_batch.personas:
        assert p.bio_vi and p.bio_en
        assert p.bio_vi != p.bio_en
        # Sanity: each bio should mention the persona's region.
        assert p.region in p.bio_vi
        assert p.region_en in p.bio_en
        # Templated bios start out unenriched.
        assert p.enriched is False


def test_marginal_distribution_not_uniform():
    """With 800 personas the populous Red River Delta + South-East should
    clearly outweigh the Central Highlands.
    """
    warnings.filterwarnings("ignore")
    from packages.personas.pgm import generate_personas

    batch = generate_personas(800, seed=11)
    counts = Counter(p.region for p in batch.personas)
    # Ranking depends on V02.01 — we only check the qualitative property.
    populous = ("Đồng bằng sông Hồng", "Đông Nam Bộ", "Đồng bằng sông Cửu Long")
    sparse = ("Tây Nguyên",)
    populous_total = sum(counts[r] for r in populous)
    sparse_total = sum(counts[r] for r in sparse)
    assert populous_total > sparse_total * 2


def test_persona_batch_round_trips_through_json(tmp_path: Path, small_batch):
    from packages.ontology import PersonaBatch

    p = tmp_path / "batch.json"
    p.write_text(small_batch.model_dump_json(), encoding="utf-8")
    reloaded = PersonaBatch.model_validate_json(p.read_text(encoding="utf-8"))
    assert reloaded.n == small_batch.n
    # Round trip preserves the bilingual fields and the enrichment flag.
    for original, copy in zip(small_batch.personas, reloaded.personas):
        assert original.region == copy.region
        assert original.region_en == copy.region_en
        assert original.bio_vi == copy.bio_vi
        assert original.bio_en == copy.bio_en
        assert copy.enriched is False
