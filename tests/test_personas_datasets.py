"""Tests for the Nemotron-Personas-Vietnam dataset builder.

We focus on the **invariants** a downstream consumer relies on:

* **Schema fidelity** — 22 columns in canonical order, dtypes match
  the published `nvidia/Nemotron-Personas-Japan` schema.
* **Reproducibility** — same ``seed`` produces the same DataFrame
  contents on every run (no leak from ``random``, ``os.urandom``, or
  Python's hash randomisation).
* **Bilingual pairing** — ``-vi-large`` and ``-en-large`` share UUIDs
  row-for-row; same for the two ``-small`` parquets.
* **Subset identity** — the ``-small-*`` UUIDs are a strict subset of
  the ``-large-*`` UUIDs (vi and en agree).
* **NSO grounding** — every region label, area label, and province
  label sits inside the canonical NSO partitions.

These tests run at a small ``large_size`` (≤2 000 rows) so the suite
stays fast; they still exercise the same code path the 3 M-row build
goes through.
"""

from __future__ import annotations

import importlib.util

import pytest

# Skip the whole module if pgmpy/pyarrow aren't present (i.e. the user
# didn't install the curator extras). Either dependency is required for
# the builder's structured-stream and parquet-write steps.
if importlib.util.find_spec("pgmpy") is None or importlib.util.find_spec("pyarrow") is None:
    pytest.skip("pgmpy / pyarrow not available; install [curator] extras",
                allow_module_level=True)
# vn-fullname-generator is a core dep but skip gracefully if it's missing.
if importlib.util.find_spec("vn_fullname_generator") is None:
    pytest.skip("vn-fullname-generator missing; pip install vn-fullname-generator",
                allow_module_level=True)

import pyarrow.parquet as pq

from packages.personas.datasets import (
    NEMOTRON_PERSONAS_VIETNAM_COLUMNS,
    NEMOTRON_PERSONAS_VIETNAM_FEATURES,
    build_nemotron_personas_vietnam_datasets,
)
from packages.personas.datasets.provinces import (
    PROVINCE_TO_REGION_VI,
    PROVINCE_VI_TO_EN,
)
from packages.personas.datasets.schema import REGIONS_VI_TO_EN


# Pre-build a tiny dataset once and share it across every test in this
# module — pgmpy initialisation costs a few seconds and we don't want
# to pay it per test.
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("Nemotron-Personas-Vietnam")
    summary = build_nemotron_personas_vietnam_datasets(
        out_dir=str(out_dir),
        large_size=600,
        small_size=80,
        chunk_size=200,
        seed=42,
    )
    return summary


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_canonical_22_columns():
    assert len(NEMOTRON_PERSONAS_VIETNAM_COLUMNS) == 22
    # Every Nemotron variant has these (we drop USA's bachelors_field /
    # zipcode and use Japan's region/area + province for the geo trio).
    must_have = {
        "uuid", "professional_persona", "sports_persona", "arts_persona",
        "travel_persona", "culinary_persona", "persona",
        "cultural_background",
        "skills_and_expertise", "skills_and_expertise_list",
        "hobbies_and_interests", "hobbies_and_interests_list",
        "career_goals_and_ambitions",
        "sex", "age", "marital_status", "education_level", "occupation",
        "region", "area", "province", "country",
    }
    assert set(NEMOTRON_PERSONAS_VIETNAM_COLUMNS) == must_have
    assert NEMOTRON_PERSONAS_VIETNAM_FEATURES["age"] == "int64"
    assert all(v == "string" for k, v in NEMOTRON_PERSONAS_VIETNAM_FEATURES.items() if k != "age")


def test_parquet_columns_and_dtypes(built):
    """All four parquets carry exactly the canonical schema."""
    for label, path in built["paths"].items():
        tbl = pq.read_table(path)
        assert tbl.column_names == list(NEMOTRON_PERSONAS_VIETNAM_COLUMNS), label
        # dtypes
        ages = tbl.column("age")
        assert str(ages.type) == "int64", f"age dtype in {label}"
        for col in tbl.column_names:
            if col == "age":
                continue
            assert str(tbl.column(col).type) == "string", f"{col} dtype in {label}"


def test_row_counts(built):
    assert built["rows"][("vi", "large")] == 600
    assert built["rows"][("en", "large")] == 600
    assert built["rows"][("vi", "small")] == 80
    assert built["rows"][("en", "small")] == 80


# ---------------------------------------------------------------------------
# Bilingual + subset invariants
# ---------------------------------------------------------------------------
def _uuids(p: str) -> list[str]:
    return pq.read_table(p, columns=["uuid"]).to_pandas()["uuid"].tolist()


def test_vi_en_large_share_uuids(built):
    """``-vi-large`` and ``-en-large`` describe the same persona row-for-row."""
    u_vi = _uuids(built["paths"]["vi-large"])
    u_en = _uuids(built["paths"]["en-large"])
    assert u_vi == u_en, "vi-large vs en-large UUID order divergence"


def test_vi_en_small_share_uuids(built):
    u_vi = _uuids(built["paths"]["vi-small"])
    u_en = _uuids(built["paths"]["en-small"])
    assert u_vi == u_en, "vi-small vs en-small UUID order divergence"


def test_small_is_subset_of_large(built):
    big = set(_uuids(built["paths"]["vi-large"]))
    small = set(_uuids(built["paths"]["vi-small"]))
    assert small.issubset(big)


def test_uuid_strings_are_uuid4_shaped(built):
    """UUIDs follow the canonical ``8-4-4-4-12`` hex layout."""
    import re
    pat = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    sample = _uuids(built["paths"]["vi-large"])[:50]
    assert all(pat.match(u) for u in sample)


# ---------------------------------------------------------------------------
# Reproducibility (the property the user explicitly asked us to enforce)
# ---------------------------------------------------------------------------
def test_same_seed_produces_identical_content(tmp_path):
    out_a = tmp_path / "A"
    out_b = tmp_path / "B"
    sumA = build_nemotron_personas_vietnam_datasets(
        out_dir=str(out_a), large_size=300, small_size=40,
        chunk_size=150, seed=7,
    )
    sumB = build_nemotron_personas_vietnam_datasets(
        out_dir=str(out_b), large_size=300, small_size=40,
        chunk_size=150, seed=7,
    )
    for label in ("vi-large", "en-large", "vi-small", "en-small"):
        a = pq.read_table(sumA["paths"][label]).to_pandas()
        b = pq.read_table(sumB["paths"][label]).to_pandas()
        assert a.equals(b), f"reproducibility broken in {label}"


def test_different_seed_produces_different_content(tmp_path):
    sumA = build_nemotron_personas_vietnam_datasets(
        out_dir=str(tmp_path / "A"), large_size=300, small_size=40,
        chunk_size=150, seed=7,
    )
    sumB = build_nemotron_personas_vietnam_datasets(
        out_dir=str(tmp_path / "B"), large_size=300, small_size=40,
        chunk_size=150, seed=8,
    )
    a = pq.read_table(sumA["paths"]["vi-large"]).to_pandas()
    b = pq.read_table(sumB["paths"]["vi-large"]).to_pandas()
    # We deliberately don't require all rows to differ — at small N a
    # handful of rows may coincide. We just want the frames *not* to be
    # equal.
    assert not a.equals(b), "different seeds produced identical content"


def test_chunk_size_does_not_affect_content(tmp_path):
    """Reproducibility contract requires chunk size to be irrelevant."""
    sumA = build_nemotron_personas_vietnam_datasets(
        out_dir=str(tmp_path / "A"), large_size=500, small_size=80,
        chunk_size=100, seed=42,
    )
    sumB = build_nemotron_personas_vietnam_datasets(
        out_dir=str(tmp_path / "B"), large_size=500, small_size=80,
        chunk_size=250, seed=42,
    )
    for label in ("vi-large", "en-large", "vi-small", "en-small"):
        a = pq.read_table(sumA["paths"][label]).to_pandas()
        b = pq.read_table(sumB["paths"][label]).to_pandas()
        assert a.equals(b), f"chunk size leaked into {label} output"


# ---------------------------------------------------------------------------
# NSO grounding — categorical values must come from the canonical sets
# ---------------------------------------------------------------------------
def test_region_labels_within_nso_partition(built):
    df = pq.read_table(built["paths"]["vi-large"], columns=["region"]).to_pandas()
    assert set(df["region"]).issubset(set(REGIONS_VI_TO_EN))

    df_en = pq.read_table(built["paths"]["en-large"], columns=["region"]).to_pandas()
    assert set(df_en["region"]).issubset(set(REGIONS_VI_TO_EN.values()))


def test_province_labels_are_canonical(built):
    df = pq.read_table(built["paths"]["vi-large"], columns=["province"]).to_pandas()
    assert set(df["province"]).issubset(set(PROVINCE_TO_REGION_VI))

    df_en = pq.read_table(built["paths"]["en-large"], columns=["province"]).to_pandas()
    assert set(df_en["province"]).issubset(set(PROVINCE_VI_TO_EN.values()))


def test_province_consistent_with_region(built):
    """Every province sits inside the macro-region the row claims."""
    df = pq.read_table(
        built["paths"]["vi-large"], columns=["province", "region"]
    ).to_pandas()
    for prov, reg in zip(df["province"].astype(str), df["region"].astype(str)):
        assert PROVINCE_TO_REGION_VI[prov] == reg, (prov, reg)


def test_area_values_are_canonical(built):
    df = pq.read_table(built["paths"]["vi-large"], columns=["area"]).to_pandas()
    assert set(df["area"]).issubset({"Thành thị", "Nông thôn"})
    df_en = pq.read_table(built["paths"]["en-large"], columns=["area"]).to_pandas()
    assert set(df_en["area"]).issubset({"urban", "rural"})


def test_age_in_adult_range(built):
    """The PGM samples 5-yr buckets 15-19 .. 65+; integer age stays in [15, 99]."""
    df = pq.read_table(built["paths"]["vi-large"], columns=["age"]).to_pandas()
    assert df["age"].min() >= 15
    assert df["age"].max() <= 99


def test_country_label_is_correct(built):
    vi = pq.read_table(built["paths"]["vi-large"], columns=["country"]).to_pandas()
    en = pq.read_table(built["paths"]["en-large"], columns=["country"]).to_pandas()
    assert set(vi["country"]) == {"Việt Nam"}
    assert set(en["country"]) == {"Vietnam"}


# ---------------------------------------------------------------------------
# Narrative columns are not empty + bilingual rows describe the same person
# ---------------------------------------------------------------------------
NARRATIVE_COLS = (
    "professional_persona", "sports_persona", "arts_persona",
    "travel_persona", "culinary_persona", "persona",
    "cultural_background",
    "skills_and_expertise", "skills_and_expertise_list",
    "hobbies_and_interests", "hobbies_and_interests_list",
    "career_goals_and_ambitions",
)


def test_narratives_nonempty(built):
    df = pq.read_table(built["paths"]["vi-large"]).to_pandas()
    for col in NARRATIVE_COLS:
        empty = df[col].astype(str).str.strip().eq("").sum()
        assert empty == 0, f"{col} has empty values"


def test_bilingual_rows_describe_same_person(built):
    """The vi & en versions of row i must agree on every language-neutral
    fact (uuid, age) and the language-specific labels must be each
    other's canonical mirrors. Vietnamese names appear verbatim in
    both versions (we don't romanise names) so a final assertion
    confirms each row's name carries through into the en narrative."""
    df_vi = pq.read_table(built["paths"]["vi-large"]).to_pandas()
    df_en = pq.read_table(built["paths"]["en-large"]).to_pandas()
    assert (df_vi["uuid"].values == df_en["uuid"].values).all()
    assert (df_vi["age"].values == df_en["age"].values).all()

    # Province mirrors agree row-for-row (same physical place).
    for i in range(len(df_vi)):
        prov_vi = df_vi["province"].iloc[i]
        prov_en = df_en["province"].iloc[i]
        assert PROVINCE_VI_TO_EN[prov_vi] == prov_en, (prov_vi, prov_en)

    # Region mirrors agree row-for-row.
    for i in range(len(df_vi)):
        assert REGIONS_VI_TO_EN[df_vi["region"].iloc[i]] == df_en["region"].iloc[i]

    # The Vietnamese name carries verbatim into the en narrative.
    for i in range(len(df_vi)):
        name_vi = str(df_vi["persona"].iloc[i]).split(",")[0]
        assert name_vi in str(df_en["persona"].iloc[i]), (
            f"row {i}: name {name_vi!r} not present in en persona"
        )


def test_skills_list_is_comma_separated(built):
    df = pq.read_table(built["paths"]["vi-large"], columns=["skills_and_expertise_list"]).to_pandas()
    sample = df["skills_and_expertise_list"].iloc[0]
    assert ", " in sample, f"skills_and_expertise_list not comma-separated: {sample}"
    items = [s.strip() for s in sample.split(",")]
    assert len(items) == len(set(items)), "duplicate skills in skills_and_expertise_list"


# ---------------------------------------------------------------------------
# Stratified subset is *representative*
# ---------------------------------------------------------------------------
def test_small_subset_covers_every_region(built):
    df = pq.read_table(built["paths"]["vi-small"], columns=["region"]).to_pandas()
    assert df["region"].nunique() == 6, (
        f"small subset missed regions: covered={df['region'].nunique()} expected 6"
    )
