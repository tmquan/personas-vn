"""Schema for the Nemotron-Personas-Vietnam datasets.

Single source of truth for:

* the **22-column layout** (matching the published Nemotron-Personas
  schema, regional variant, with the Japan-style ``region`` / ``area`` /
  ``province`` geographic trio adapted for Vietnam),
* the **pyarrow schema** used by the streaming parquet writer,
* the **bilingual label dictionaries** mapping every categorical value
  between its canonical Vietnamese form (matching the labels NSO
  publishes in PX-Web) and an English mirror.

We deliberately keep this module **stateless and import-cheap** — no
pandas, no I/O — so it can be imported by tests without dragging in the
generator's heavyweight dependencies (pgmpy, sentence-transformers, …).

Schema cross-reference (verified Dec 2025 via
`https://datasets-server.huggingface.co/info?dataset=nvidia%2F<name>`):

==========================  ==========================================
Country / Region            Geographic columns
==========================  ==========================================
USA                         city, state, zipcode (+ bachelors_field)
Japan                       region, area, prefecture
Korea                       district, province (+ many extras)
Brazil                      municipality, state
France                      commune, departement (+ household_type)
India                       zone, state, district (+ language fields)
**Vietnam (this package)**  **region, area, province**
==========================  ==========================================

Vietnam's NSO publishes population at three nested levels:

1. **6 macro-regions** (V02.01 ``Đồng bằng sông Hồng``, …),
2. **63 provinces / cities** (V02.01),
3. **urban / rural** classification (V02.02).

Mapping these onto Japan's ``region`` / ``area`` / ``prefecture`` trio
keeps the schema topologically identical to the published Japan dataset
— ``province`` is the admin-1 unit (analogous to a Japanese prefecture
or Korean province), ``area`` is urban vs rural, and ``region`` is the
macro-region a province rolls up into.
"""

from __future__ import annotations

from typing import Final

import pyarrow as pa

# ---------------------------------------------------------------------------
# Column layout (22 fields)
# ---------------------------------------------------------------------------
# Order matters: this list defines the on-disk column order in the
# parquet output. Same order across all 4 (language × size) variants so
# downstream consumers can `pd.concat` or schema-cast freely.
NEMOTRON_PERSONAS_VIETNAM_COLUMNS: Final[tuple[str, ...]] = (
    # 1. globally-unique identifier (UUID4 hex string)
    "uuid",
    # 2-7. six narrative persona fields (long natural-language)
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "persona",
    # 8-13. five contextual narrative fields + two "_list" mirrors
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
    # 14-18. structured demographic / socio-economic columns
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    # 19-22. geography (Vietnam-specific trio + country)
    "region",
    "area",
    "province",
    "country",
)
assert len(NEMOTRON_PERSONAS_VIETNAM_COLUMNS) == 22, "Nemotron-Personas variants are 22 columns"


# ---------------------------------------------------------------------------
# Hugging Face Datasets-style features (mirrors what the published parquets
# advertise via /datasets-server/info). Plain dict so callers don't need the
# `datasets` package to introspect the schema.
# ---------------------------------------------------------------------------
NEMOTRON_PERSONAS_VIETNAM_FEATURES: Final[dict[str, str]] = {
    col: ("int64" if col == "age" else "string")
    for col in NEMOTRON_PERSONAS_VIETNAM_COLUMNS
}


# ---------------------------------------------------------------------------
# pyarrow schema for the streaming ParquetWriter
# ---------------------------------------------------------------------------
PYARROW_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field(col, pa.int64() if col == "age" else pa.string())
        for col in NEMOTRON_PERSONAS_VIETNAM_COLUMNS
    ]
)


# ---------------------------------------------------------------------------
# Bilingual label dictionaries
# ---------------------------------------------------------------------------
# Each dict is keyed by the **canonical Vietnamese label** (the form NSO
# publishes in PX-Web) and yields an English mirror. The English mirrors
# are deliberately written in NSO/Statistics-Bureau style so the parallel
# pair reads naturally in either language.
#
# Re-exporting here (rather than re-using packages.personas.pgm.distributions)
# to keep this module side-effect free and importable from tests without
# pulling pgmpy.

REGIONS_VI_TO_EN: Final[dict[str, str]] = {
    "Đồng bằng sông Hồng":                   "Red River Delta",
    "Trung du và miền núi phía Bắc":         "Northern Midlands and Mountains",
    "Bắc Trung Bộ và Duyên hải miền Trung":  "North Central and Central Coastal",
    "Tây Nguyên":                            "Central Highlands",
    "Đông Nam Bộ":                           "South East",
    "Đồng bằng sông Cửu Long":               "Mekong River Delta",
}

AREA_VI_TO_EN: Final[dict[str, str]] = {
    "Thành thị": "urban",
    "Nông thôn": "rural",
}

SEX_VI_TO_EN: Final[dict[str, str]] = {
    "Nam": "male",
    "Nữ": "female",
}

MARITAL_VI_TO_EN: Final[dict[str, str]] = {
    "Chưa kết hôn": "never married",
    "Đã kết hôn":   "married",
    "Goá":          "widowed",
    "Ly hôn":       "divorced",
}

EDUCATION_VI_TO_EN: Final[dict[str, str]] = {
    # Matches the V02.54 NSO trained-labour qualification ladder.
    "Không có trình độ CMKT": "no formal qualification",
    "Sơ cấp":                 "primary vocational",
    "Trung cấp":              "intermediate vocational",
    "Cao đẳng":               "junior college",
    "Đại học trở lên":        "bachelor's degree or higher",
}

OCCUPATION_VI_TO_EN: Final[dict[str, str]] = {
    # Canonical NSO V02.43 labels (verified against the latest-year
    # parquet under data/nso-gov-vn/raw/pxweb/vi/). Plus
    # `Không áp dụng` ("not applicable"), the post-processing-injected
    # value for personas who aren't currently in employment.
    "Nhà lãnh đạo":                                                    "managers",
    "Chuyên môn kỹ thuật bậc cao":                                     "professionals",
    "Chuyên môn kỹ thuật bậc trung":                                   "associate professionals",
    "Nhân viên":                                                       "clerical support",
    "Dịch vụ cá nhân, bảo vệ bán hàng":                                "service and sales workers",
    "Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản":    "skilled agricultural, forestry and fishery workers",
    "Thợ thủ công và các thợ khác có liên quan":                       "craft and related trades workers",
    "Thợ lắp ráp và vận hành máy móc, thiết bị":                       "plant and machine operators and assemblers",
    "Nghề giản đơn":                                                   "elementary occupations",
    "Khác":                                                            "other",
    "Không áp dụng":                                                   "not applicable",
}

EMPLOYMENT_VI_TO_EN: Final[dict[str, str]] = {
    "Làm công ăn lương":             "wage and salary worker",
    "Chủ cơ sở sản xuất kinh doanh": "employer",
    "Tự làm":                        "own-account worker",
    "Lao động gia đình":             "contributing family worker",
    "Xã viên hợp tác xã":            "cooperative member",
    "Học nghề":                      "apprentice",
    "Khác":                          "other",
}


# Country labels — single string per language.
COUNTRY_VI: Final[str] = "Việt Nam"
COUNTRY_EN: Final[str] = "Vietnam"


def column_index(name: str) -> int:
    """Return the 0-based position of ``name`` in :data:`NEMOTRON_PERSONAS_VIETNAM_COLUMNS`.

    Useful when ordering DataFrame columns before writing — keeps the
    column order assertion in one place.
    """
    return NEMOTRON_PERSONAS_VIETNAM_COLUMNS.index(name)
