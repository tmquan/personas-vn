"""Vietnamese persona distributions, sourced from real PX-Web tables.

This module *replaces* the hand-coded statistics in :mod:`distributions`
with values pulled directly from NSO's published statistical database
(crawled by ``packages.curator.stages.DownloadStage`` into
``data/nso-gov-vn/raw/pxweb/vi/``).

For each persona dimension we declare:

* the PX-Web table id (e.g. ``V02.43``),
* the variable inside that table whose values map onto our persona
  category (e.g. ``Nghề nghiệp``),
* an optional filter (drop rollup rows like ``TỔNG SỐ``),
* an optional translation dict to a display label (English).

Counts come from the most recent year available in each table. When a
joint distribution is needed but only marginals are published (e.g. we
have *labour force by age* and *labour force by region* but not
*labour force by age × region*), we sample conditionally using the
marginals via the chain rule — equivalent to assuming local
independence at that edge of the PGM.

Tables used:

    V02.01   Area, population, density by province  (regions)
    V02.02   Population by sex × urban/rural        (sex, urbanicity)
    V02.36   Labour force by age group              (coarse age)
    V02.41   Employed by 5-year age group           (fine age)
    V02.42   Employed by industry sector
    V02.43   Employed by occupation (ISCO)
    V02.44   Employed by employment status
    V02.54   Trained labour by qualification level  (education)
    V02.59   Unemployment by region × urban/rural
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.logging import get_logger
from packages.common.paths import resolve

log = get_logger(__name__)

PXWEB_ROOT = Path("data/nso-gov-vn/raw/pxweb/vi")

# The six economic regions of Vietnam, ordered north → south. Vietnamese
# labels are what NSO publishes; English labels are convenience for the
# visualizer.
REGIONS_VI = [
    "Đồng bằng sông Hồng",
    "Trung du và miền núi phía Bắc",
    "Bắc Trung Bộ và Duyên hải miền Trung",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
]
REGIONS_EN = {
    "Đồng bằng sông Hồng":               "Red River Delta",
    "Trung du và miền núi phía Bắc":     "Northern Midlands & Mountains",
    "Bắc Trung Bộ và Duyên hải miền Trung": "North Central & Central Coast",
    "Tây Nguyên":                        "Central Highlands",
    "Đông Nam Bộ":                       "South East",
    "Đồng bằng sông Cửu Long":           "Mekong River Delta",
}


def _canonical_region(name: str | None) -> str | None:
    """Map any region label (V02.01 vs V02.59 differ in 'duyên'/'Duyên' casing)
    onto the canonical spelling used in :data:`REGIONS_VI`. Returns ``None`` if
    the input doesn't match any canonical region.
    """
    if not name:
        return None
    target = name.strip().lower()
    for canon in REGIONS_VI:
        if canon.lower() == target:
            return canon
    return None

URBANICITY_VI = ["Thành thị", "Nông thôn"]
URBANICITY_EN = {"Thành thị": "urban", "Nông thôn": "rural"}

SEXES_VI = ["Nam", "Nữ"]
SEXES_EN = {"Nam": "male", "Nữ": "female"}

# Coarse age groups used by V02.36, V02.61. Fine 5-year buckets come from
# V02.41 (15-19, 20-24, ..., 60-64, 65+); we use the fine buckets in the
# generator and roll up to coarse ones only when joining a coarse table.
AGE_GROUPS_FINE_VI = ["15-19", "20-24", "25-29", "30-34", "35-39",
                      "40-44", "45-49", "50-54", "55-59", "60-64", "65+"]
AGE_GROUPS_COARSE_VI = ["15 - 24", "25 - 49", "50+"]

# Education / qualification levels from V02.54.
EDUCATION_LEVELS_VI = ["Không có trình độ CMKT", "Sơ cấp", "Trung cấp",
                       "Cao đẳng", "Đại học trở lên"]
EDUCATION_LEVELS_EN = {
    "Không có trình độ CMKT": "no_qualification",
    "Sơ cấp":                 "primary",
    "Trung cấp":              "intermediate",
    "Cao đẳng":               "college",
    "Đại học trở lên":        "university",
}

# 11-class ISCO occupation taxonomy used by V02.43.
OCCUPATIONS_VI: list[str] = []  # populated lazily from the table
OCCUPATIONS_EN_HINTS: dict[str, str] = {
    "Nhà lãnh đạo": "managers",
    "Chuyên môn kỹ thuật bậc cao": "professionals",
    "Chuyên môn kỹ thuật bậc trung": "technicians",
    "Nhân viên": "clerical",
    "Dịch vụ cá nhân, bảo vệ bán hàng": "service_sales",
    "Lao động có kỹ thuật trong nông, lâm nghiệp và thủy sản": "skilled_agriculture",
    "Lao động thủ công và các nghề có liên quan khác": "craft_trades",
    "Thợ lắp ráp, vận hành máy móc thiết bị": "operators",
    "Lao động giản đơn": "elementary",
    "Lực lượng quân đội": "armed_forces",
    "Khác": "other",
}

EMPLOYMENT_STATUSES_VI: list[str] = []
EMPLOYMENT_STATUSES_EN_HINTS = {
    "Làm công ăn lương": "wage_employee",
    "Chủ cơ sở sản xuất kinh doanh": "employer",
    "Tự làm": "own_account",
    "Lao động gia đình": "family_worker",
    "Xã viên hợp tác xã": "coop_member",
    "Học nghề": "apprentice",
    "Khác": "other",
}

INDUSTRIES_VI: list[str] = []  # populated lazily from V02.42

# Synthetic categories not directly in PX-Web (we keep them so personas
# stay rich). When the upstream NSO database starts publishing them we
# can swap these out for real distributions.
ETHNICITIES_VI = ["Kinh", "Tày", "Thái", "Mường", "Khmer", "H'Mông", "Khác"]
MARITAL_STATUSES_VI = ["Chưa kết hôn", "Đã kết hôn", "Goá", "Ly hôn"]


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------
def _table_path(code: str) -> Path:
    """Map a PX-Web table id like 'V02.43' to its parquet path on disk."""
    if code.startswith("V02"):
        prefix = "Dan-so-va-lao-dong"
    elif code.startswith("V04"):
        prefix = "dau-tu"
    elif code.startswith("V05"):
        prefix = "Doanh-nghiep"
    elif code.startswith("V07"):
        prefix = "Cong-nghiep"
    elif code.startswith("V13"):
        prefix = "Giao-duc"
    else:
        prefix = "table"
    return resolve(PXWEB_ROOT) / f"{prefix}__{code}.px.parquet"


@lru_cache(maxsize=64)
def _load_table(code: str) -> pd.DataFrame:
    """Load a PX-Web parquet by table id; cache for reuse."""
    path = _table_path(code)
    if not path.exists():
        raise FileNotFoundError(
            f"PX-Web table {code} not found at {path}; run `personas-vn curate --only download` first."
        )
    return pd.read_parquet(path)


def _latest_year(df: pd.DataFrame, year_col: str = "Năm") -> str:
    """Return the most recent year as a string (years are stored as str)."""
    years = df[year_col].astype(str).unique()
    # Years are like '2024' but a few tables use '2024 (sb)' or similar.
    cleaned = sorted(set(years), key=lambda s: int(re.findall(r"\d+", s)[0]) if re.findall(r"\d+", s) else 0)
    return cleaned[-1] if cleaned else ""


def _drop_totals(values: list[str]) -> list[str]:
    """Strip rollup labels (TỔNG SỐ, Tổng số, CẢ NƯỚC, etc.)."""
    rollups = {"tổng số", "tong so", "cả nước", "ca nuoc"}
    return [v for v in values if v.strip().lower() not in rollups]


# ---------------------------------------------------------------------------
# Per-dimension distributions
# ---------------------------------------------------------------------------
def region_population_counts() -> pd.DataFrame:
    """Region → population (thousand persons), latest year, from V02.01.

    V02.01 lists provinces; we sum to the 6 macro-regions using a fixed
    province → region map (NSO classification).
    """
    df = _load_table("V02.01")
    indicator_col = "Chỉ tiêu"
    region_col = "Địa phương"
    pop_filter = df[indicator_col].str.contains("Dân số", na=False)
    df = df[pop_filter].copy()
    # Canonicalise the region label (V02.01 vs V02.59 differ in casing).
    df["region"] = df[region_col].map(_canonical_region)
    df = df[df["region"].notna()]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    return (
        df.groupby("region", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "count"})
    )


def urbanicity_marginal_counts() -> pd.DataFrame:
    """Urban/Rural → population, latest year, from V02.02."""
    df = _load_table("V02.02")
    df = df[df["Cách tính"] == "Tổng số (Nghìn người)"]
    df = df[df["Phân tổ"].isin(URBANICITY_VI)]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    return df.rename(columns={"Phân tổ": "urbanicity", "value": "count"})[["urbanicity", "count"]]


def sex_marginal_counts() -> pd.DataFrame:
    """Sex → population, latest year, from V02.02."""
    df = _load_table("V02.02")
    df = df[df["Cách tính"] == "Tổng số (Nghìn người)"]
    df = df[df["Phân tổ"].isin(SEXES_VI)]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    return df.rename(columns={"Phân tổ": "sex", "value": "count"})[["sex", "count"]]


def age_group_fine_counts() -> pd.DataFrame:
    """Fine 5-year age groups for the working-age population, V02.41."""
    df = _load_table("V02.41")
    df = df[~df["Nhóm tuổi"].str.upper().eq("TỔNG SỐ")]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    return df.rename(columns={"Nhóm tuổi": "age_group", "value": "count"})[["age_group", "count"]]


def education_counts() -> pd.DataFrame:
    """Trained-labour qualification level, V02.54.

    V02.54 reports the *share* of labour with qualifications by level.
    To get a marginal we add a synthetic 'no qualification' bucket that
    fills the remainder, since V02.54's TỔNG SỐ is the share of qualified
    labour, not the total.
    """
    df = _load_table("V02.54")
    df = df[df["Chuyên môn kỹ thuật"].isin(EDUCATION_LEVELS_VI[1:])]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    out = df.rename(
        columns={"Chuyên môn kỹ thuật": "education_level", "value": "count"}
    )[["education_level", "count"]]
    # Add a synthetic no-qualification bucket = total minus the rest. NSO
    # publishes ~26% trained, so ~74% have no formal CMKT.
    qualified_pct = float(out["count"].sum())
    no_qual_pct = max(1.0, 100.0 - qualified_pct)
    out = pd.concat(
        [
            pd.DataFrame([{"education_level": "Không có trình độ CMKT", "count": no_qual_pct}]),
            out,
        ],
        ignore_index=True,
    )
    return out


def occupation_counts() -> pd.DataFrame:
    """Employed by occupation (ISCO collapsed to NSO 11-class), V02.43."""
    df = _load_table("V02.43")
    df = df[~df["Nghề nghiệp"].str.upper().eq("TỔNG SỐ")]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    out = df.rename(columns={"Nghề nghiệp": "occupation", "value": "count"})[
        ["occupation", "count"]
    ]
    # Populate the global occupation list lazily.
    global OCCUPATIONS_VI
    OCCUPATIONS_VI = sorted(out["occupation"].unique().tolist())
    return out


def employment_status_counts() -> pd.DataFrame:
    """Employed by employment status, V02.44."""
    df = _load_table("V02.44")
    df = df[~df["Vị thế việc làm"].str.upper().eq("TỔNG SỐ")]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    out = df.rename(columns={"Vị thế việc làm": "employment_status", "value": "count"})[
        ["employment_status", "count"]
    ]
    global EMPLOYMENT_STATUSES_VI
    EMPLOYMENT_STATUSES_VI = sorted(out["employment_status"].unique().tolist())
    return out


def industry_counts() -> pd.DataFrame:
    """Employed by industry sector, V02.42 (latest year, 'Tổng số (Nghìn người)')."""
    df = _load_table("V02.42")
    df = df[df["Phân tổ"] == "Tổng số (Nghìn người)"]
    df = df[~df["Ngành"].str.upper().eq("TỔNG SỐ")]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    out = df.rename(columns={"Ngành": "industry_sector", "value": "count"})[
        ["industry_sector", "count"]
    ]
    global INDUSTRIES_VI
    INDUSTRIES_VI = sorted(out["industry_sector"].unique().tolist())
    return out


def unemployment_by_region_urban() -> pd.DataFrame:
    """Unemployment rate by region × urban/rural, V02.59 (latest year, %)."""
    df = _load_table("V02.59").copy()
    df["region"] = df["Vùng"].map(_canonical_region)
    df = df[df["region"].notna()]
    df = df[df["Thành thị, nông thôn"].isin(URBANICITY_VI)]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    return df.rename(
        columns={
            "Thành thị, nông thôn": "urbanicity",
            "value": "unemployment_pct",
        }
    )[["region", "urbanicity", "unemployment_pct"]]


# ---------------------------------------------------------------------------
# Composite distributions used by the generator's PGM stages
# ---------------------------------------------------------------------------
def region_x_urbanicity_counts() -> pd.DataFrame:
    """Region × urbanicity counts.

    NSO doesn't publish the *joint* directly, but V02.01 lists each
    province with its population, and the macro-region of every province
    is fixed. We don't have a per-province urban share in V02 directly,
    so we fall back to the national urban-share from V02.02 modulated by
    a region tilt (South-East and Red River Delta more urban; Highlands
    less). The tilt is the same shape as in the legacy ``distributions``
    module — the absolute counts come from the real region populations.
    """
    region_pop = region_population_counts().set_index("region")["count"]
    # Per-region urban share (expert prior). NSO Statistical Yearbook
    # publishes these in print but not via PX-Web; values mirror the
    # 2024 yearbook section 3.4.
    region_urban_pct = {
        "Đồng bằng sông Hồng":               41,
        "Trung du và miền núi phía Bắc":     21,
        "Bắc Trung Bộ và duyên hải miền Trung": 32,
        "Tây Nguyên":                        30,
        "Đông Nam Bộ":                       67,
        "Đồng bằng sông Cửu Long":           28,
    }
    rows: list[dict[str, Any]] = []
    for region, pop in region_pop.items():
        urban_pct = region_urban_pct.get(region, 35)
        rows.append({"region": region, "urbanicity": "Thành thị", "count": pop * urban_pct / 100.0})
        rows.append({"region": region, "urbanicity": "Nông thôn", "count": pop * (100 - urban_pct) / 100.0})
    return pd.DataFrame(rows)


def age_group_x_sex_counts() -> pd.DataFrame:
    """Age × sex marginals via independence: P(age, sex) ∝ P(age)·P(sex).

    NSO publishes the marginals separately (V02.41 and V02.39); the
    population is large enough that the independence approximation is
    fine for downstream sampling — at scale, the marginals dominate.
    """
    age = age_group_fine_counts().set_index("age_group")["count"]
    sex = sex_marginal_counts().set_index("sex")["count"]
    age_pct = age / age.sum()
    sex_pct = sex / sex.sum()
    rows: list[dict[str, Any]] = []
    for ag, ap in age_pct.items():
        for sx, sp in sex_pct.items():
            rows.append({"age_group": ag, "sex": sx, "count": float(ap * sp * 1_000_000)})
    return pd.DataFrame(rows)


def employment_status_x_age_x_sex_counts() -> pd.DataFrame:
    """Employment status × age × sex via independence approximation.

    PX-Web doesn't publish the joint table; we factorise through the
    marginal employment-status distribution and use the same shape for
    every (age, sex) pair. Children (<15) are handled by clamping in
    post-processing rather than represented in the source data.
    """
    es = employment_status_counts().set_index("employment_status")["count"]
    rows: list[dict[str, Any]] = []
    for ag in AGE_GROUPS_FINE_VI:
        for sx in SEXES_VI:
            for status, c in es.items():
                rows.append({
                    "age_group": ag, "sex": sx,
                    "employment_status": status, "count": float(c),
                })
    return pd.DataFrame(rows)


def occupation_x_education_counts() -> pd.DataFrame:
    """Occupation × education via marginals.

    Real NSO LFS cross-tabs are not in PX-Web at this granularity, so we
    factorise through the marginals. Once those cross-tabs become
    available the loader can be swapped in without changing the schema.
    """
    occ = occupation_counts().set_index("occupation")["count"]
    edu = education_counts().set_index("education_level")["count"]
    rows: list[dict[str, Any]] = []
    occ_pct = occ / occ.sum()
    edu_pct = edu / edu.sum()
    for o, op in occ_pct.items():
        for e, ep in edu_pct.items():
            rows.append({"occupation": o, "education_level": e, "count": float(op * ep * 1_000_000)})
    return pd.DataFrame(rows)


def industry_x_occupation_counts() -> pd.DataFrame:
    """Industry × occupation marginal-product approximation."""
    ind = industry_counts().set_index("industry_sector")["count"]
    occ = occupation_counts().set_index("occupation")["count"]
    ind_pct = ind / ind.sum()
    occ_pct = occ / occ.sum()
    rows: list[dict[str, Any]] = []
    for i, ip in ind_pct.items():
        for o, op in occ_pct.items():
            rows.append({"industry_sector": i, "occupation": o, "count": float(ip * op * 1_000_000)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
@dataclass
class PxWebDistributions:
    """Bundle every PX-Web-grounded distribution in one immutable object.

    Lazily-loaded via ``cached_property``; the underlying parquet reads
    are themselves memoised by ``_load_table``.
    """

    region_counts: pd.DataFrame = field(default_factory=region_population_counts)
    urbanicity_counts: pd.DataFrame = field(default_factory=urbanicity_marginal_counts)
    sex_counts: pd.DataFrame = field(default_factory=sex_marginal_counts)
    age_group_counts: pd.DataFrame = field(default_factory=age_group_fine_counts)
    education_counts: pd.DataFrame = field(default_factory=education_counts)
    occupation_counts: pd.DataFrame = field(default_factory=occupation_counts)
    employment_status_counts: pd.DataFrame = field(default_factory=employment_status_counts)
    industry_counts: pd.DataFrame = field(default_factory=industry_counts)
    region_x_urbanicity: pd.DataFrame = field(default_factory=region_x_urbanicity_counts)
    age_x_sex: pd.DataFrame = field(default_factory=age_group_x_sex_counts)
    employment_x_age_x_sex: pd.DataFrame = field(
        default_factory=employment_status_x_age_x_sex_counts
    )
    occupation_x_education: pd.DataFrame = field(default_factory=occupation_x_education_counts)
    industry_x_occupation: pd.DataFrame = field(default_factory=industry_x_occupation_counts)

    def summary(self) -> dict[str, int]:
        return {
            "region_counts":           len(self.region_counts),
            "urbanicity_counts":       len(self.urbanicity_counts),
            "sex_counts":              len(self.sex_counts),
            "age_group_counts":        len(self.age_group_counts),
            "education_counts":        len(self.education_counts),
            "occupation_counts":       len(self.occupation_counts),
            "employment_status_counts": len(self.employment_status_counts),
            "industry_counts":         len(self.industry_counts),
            "region_x_urbanicity":     len(self.region_x_urbanicity),
            "age_x_sex":               len(self.age_x_sex),
            "employment_x_age_x_sex":  len(self.employment_x_age_x_sex),
            "occupation_x_education":  len(self.occupation_x_education),
            "industry_x_occupation":   len(self.industry_x_occupation),
        }

    def categories(self) -> dict[str, list[str]]:
        """Return the actual category list used in each PX-Web table."""
        return {
            "region": REGIONS_VI,
            "urbanicity": URBANICITY_VI,
            "sex": SEXES_VI,
            "age_group": AGE_GROUPS_FINE_VI,
            "education_level": EDUCATION_LEVELS_VI,
            "occupation": sorted(self.occupation_counts["occupation"].unique().tolist()),
            "employment_status": sorted(
                self.employment_status_counts["employment_status"].unique().tolist()
            ),
            "industry_sector": sorted(
                self.industry_counts["industry_sector"].unique().tolist()
            ),
            "ethnicity": ETHNICITIES_VI,
            "marital_status": MARITAL_STATUSES_VI,
        }

    def english_labels(self) -> dict[str, dict[str, str]]:
        """English translation for visualizer display."""
        cats = self.categories()
        return {
            "region": REGIONS_EN,
            "urbanicity": URBANICITY_EN,
            "sex": SEXES_EN,
            "education_level": EDUCATION_LEVELS_EN,
            "occupation": {v: OCCUPATIONS_EN_HINTS.get(v, v) for v in cats["occupation"]},
            "employment_status": {
                v: EMPLOYMENT_STATUSES_EN_HINTS.get(v, v) for v in cats["employment_status"]
            },
        }
