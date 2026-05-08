"""Vietnamese persona generator (built on NVIDIA SDG-PGMs).

This module is a concrete :class:`pgms.PGMGenerator` subclass. It plugs
the PX-Web-grounded distribution tables in
:mod:`packages.personas.pgm.distributions` into NVIDIA SDG-PGMs'
cascaded Bayesian-network sampler.

Stage layout (mirrors ``configs/ontology.yaml``):

    Stage 1 — geography:        region → urbanicity
    Stage 2 — demographics:     age_group, sex, ethnicity, marital_status
    Stage 3 — socio-economic:   education_level, employment_status,
                                occupation, industry_sector
    (Post-processing after each stage adds derived columns: integer
     age, full Vietnamese name, bilingual bio.)

Categorical values are the canonical Vietnamese labels published by the
NSO PX-Web statistical database. The :func:`generate_personas` wrapper
maps them onto our typed :class:`packages.ontology.Persona` instances
together with English mirrors and a templated bilingual bio.

If real NSO PX-Web parquets aren't on disk yet (``data/nso-gov-vn/raw/``
hasn't been populated), the generator raises a clear error directing the
user to run ``personas-vn curate --only download`` first.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pydantic import BaseModel, ConfigDict

from packages.common.logging import get_logger
from packages.ontology import Persona, PersonaBatch
from packages.ontology.registry import OntologyRegistry, load_registry
from packages.personas.pgm.base import Edge, PGMGenerator, TarFileMixin
from packages.personas.pgm.distributions import (
    AGE_GROUPS_FINE_VI,
    EDUCATION_LEVELS_EN,
    EDUCATION_LEVELS_VI,
    EMPLOYMENT_STATUSES_EN_HINTS,
    ETHNICITIES_VI,
    MARITAL_STATUSES_VI,
    OCCUPATIONS_EN_HINTS,
    REGIONS_EN,
    REGIONS_VI,
    SEXES_EN,
    SEXES_VI,
    URBANICITY_EN,
    URBANICITY_VI,
    PxWebDistributions,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pgmpy")

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Vietnamese name pools
# ---------------------------------------------------------------------------
_FAMILY_NAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh",
]
_MIDDLE_NAMES_M = ["Văn", "Hữu", "Đức", "Quốc", "Minh", "Anh", "Thanh", "Trung"]
_MIDDLE_NAMES_F = ["Thị", "Thu", "Thuỳ", "Ngọc", "Hồng", "Mỹ", "Bích", "Kim"]
_GIVEN_NAMES_M = ["An", "Bảo", "Cường", "Dũng", "Đạt", "Hùng", "Khoa", "Long",
                   "Minh", "Nam", "Phúc", "Quang", "Sơn", "Tâm", "Tuấn", "Vinh"]
_GIVEN_NAMES_F = ["Anh", "Bích", "Châu", "Diệu", "Giang", "Hà", "Hoa", "Hương",
                   "Lan", "Mai", "Ngân", "Nhi", "Phương", "Quỳnh", "Trang", "Yến"]


def _slug_id(idx: int, name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_marks = no_marks.replace("đ", "d").replace("Đ", "d")
    slug = re.sub(r"[^a-z0-9]+", "-", no_marks.lower()).strip("-")
    return f"vn-{idx:06d}-{slug}"


# ---------------------------------------------------------------------------
# Income quintile (synthetic) — PX-Web does not publish per-individual income
# quintiles, so we sample these from a simple education-conditional prior.
# ---------------------------------------------------------------------------
INCOME_QUINTILES_VI = ["Q1", "Q2", "Q3", "Q4", "Q5"]
INCOME_QUINTILES_EN = {q: q for q in INCOME_QUINTILES_VI}

# Education-conditional shape (same for every region; the NSO Household
# Living Standards Survey breakdown isn't in PX-Web, but the directional
# pattern is well-known).
_INCOME_BY_EDUCATION = {
    "Không có trình độ CMKT": [38, 28, 18, 11,  5],
    "Sơ cấp":                  [22, 26, 24, 18, 10],
    "Trung cấp":               [12, 22, 28, 24, 14],
    "Cao đẳng":                [ 6, 16, 28, 30, 20],
    "Đại học trở lên":         [ 3,  8, 18, 32, 39],
}


# ---------------------------------------------------------------------------
# Data bank — Pydantic + TarFileMixin (SDG-PGMs canonical pattern)
# ---------------------------------------------------------------------------
class VNPersonaData(BaseModel, TarFileMixin):
    """Bundle every count-DataFrame into a single Pydantic data bank.

    Mirrors the ``MyData(BaseModel, TarFileMixin)`` example in the
    SDG-PGMs README. Each field is a ``count`` DataFrame with one or
    more dimension columns plus a ``count`` column; CPDs are built
    from these via :meth:`PGMGenerator.fill_na_and_gen_cpd`.

    Serialisable to/from a tar archive of parquet files via the
    ``TarFileMixin`` parent — this is what the ``data_path`` argument
    of :class:`VNPersonaGenerator` consumes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    region_counts: pd.DataFrame
    region_x_urbanicity: pd.DataFrame
    age_group_x_region: pd.DataFrame
    sex_x_age_group: pd.DataFrame
    ethnicity_x_region: pd.DataFrame
    marital_x_age_x_sex: pd.DataFrame
    education_x_age_x_urbanicity: pd.DataFrame
    employment_x_age_x_sex: pd.DataFrame
    occupation_x_education: pd.DataFrame
    industry_x_occupation: pd.DataFrame
    income_x_education: pd.DataFrame


_TAR_MAPPINGS = {
    "region_counts":              "region_counts.parquet",
    "region_x_urbanicity":        "region_x_urbanicity.parquet",
    "age_group_x_region":         "age_group_x_region.parquet",
    "sex_x_age_group":            "sex_x_age_group.parquet",
    "ethnicity_x_region":         "ethnicity_x_region.parquet",
    "marital_x_age_x_sex":        "marital_x_age_x_sex.parquet",
    "education_x_age_x_urbanicity": "education_x_age_x_urbanicity.parquet",
    "employment_x_age_x_sex":     "employment_x_age_x_sex.parquet",
    "occupation_x_education":     "occupation_x_education.parquet",
    "industry_x_occupation":      "industry_x_occupation.parquet",
    "income_x_education":         "income_x_education.parquet",
}


def _build_data_from_pxweb() -> VNPersonaData:
    """Build a :class:`VNPersonaData` from the on-disk PX-Web parquets.

    The downloaded PX-Web tables only publish marginals for several
    persona dimensions; we factorise jointly-needed tables (e.g.
    age × region) through the marginals using the chain rule. When NSO
    later publishes the joint cross-tabs the loader can be swapped in
    without touching the generator.

    SDG-PGMs requires every variable column on every count DataFrame to
    be a :class:`pandas.Categorical` with the same ordered category
    list. We do that conversion at the bottom of this function.
    """
    d = PxWebDistributions()

    # ----- Stage 2 marginals reshaped as (parent, child, count) tables -----
    # age_group | region: same fine age distribution for every region.
    age = d.age_group_counts.set_index("age_group")["count"]
    rows = []
    for region in REGIONS_VI:
        for ag, c in age.items():
            rows.append({"region": region, "age_group": ag, "count": float(c)})
    age_group_x_region = pd.DataFrame(rows)

    # sex | age_group: NSO ratios are remarkably flat (~50/50); use the
    # national marginal everywhere.
    sex = d.sex_counts.set_index("sex")["count"]
    sex_pct = sex / sex.sum()
    rows = []
    for ag in AGE_GROUPS_FINE_VI:
        for s, p in sex_pct.items():
            rows.append({"age_group": ag, "sex": s, "count": float(p * 1_000_000)})
    sex_x_age_group = pd.DataFrame(rows)

    # ethnicity | region: PX-Web doesn't publish this directly, so we use
    # the well-known 2019-Census shape from the legacy module shaped to a
    # 7-bucket taxonomy.
    eth_by_region = {
        "Đồng bằng sông Hồng":               [97, 1, 0, 1, 0, 0,  1],
        "Trung du và miền núi phía Bắc":     [44, 12, 11, 12, 0, 11, 10],
        "Bắc Trung Bộ và Duyên hải miền Trung": [89, 1, 1, 4, 0, 1,  4],
        "Tây Nguyên":                        [62, 1, 0, 1, 0, 5, 31],
        "Đông Nam Bộ":                       [93, 0, 0, 0, 2, 0,  5],
        "Đồng bằng sông Cửu Long":           [89, 0, 0, 0, 7, 0,  4],
    }
    rows = []
    for region, dist in eth_by_region.items():
        for eth, c in zip(ETHNICITIES_VI, dist):
            rows.append({"region": region, "ethnicity": eth, "count": float(c)})
    ethnicity_x_region = pd.DataFrame(rows)

    # marital | age × sex: synthetic — single under 25, married 25-54,
    # widowed/older skew for women 55+. PX-Web V02.28-V02.30 publishes
    # marriage-rate aggregates, not marital-status crosstabs.
    marital_template = [
        # (age_group, sex, single, married, widowed, divorced)
        ("15-19", "Nam", 99, 1, 0, 0), ("15-19", "Nữ", 96, 4, 0, 0),
        ("20-24", "Nam", 88, 12, 0, 0), ("20-24", "Nữ", 78, 22, 0, 0),
        ("25-29", "Nam", 35, 64, 0, 1), ("25-29", "Nữ", 25, 73, 0, 2),
        ("30-34", "Nam", 12, 86, 0, 2), ("30-34", "Nữ",  9, 88, 1, 2),
        ("35-39", "Nam",  6, 91, 0, 3), ("35-39", "Nữ",  4, 92, 1, 3),
        ("40-44", "Nam",  4, 92, 1, 3), ("40-44", "Nữ",  3, 91, 2, 4),
        ("45-49", "Nam",  3, 91, 2, 4), ("45-49", "Nữ",  2, 87, 6, 5),
        ("50-54", "Nam",  3, 90, 4, 3), ("50-54", "Nữ",  2, 82, 12, 4),
        ("55-59", "Nam",  2, 88, 7, 3), ("55-59", "Nữ",  1, 73, 22, 4),
        ("60-64", "Nam",  1, 86, 10, 3), ("60-64", "Nữ", 1, 60, 35, 4),
        ("65+",   "Nam",  1, 78, 18, 3), ("65+",   "Nữ", 1, 41, 55, 3),
    ]
    rows = []
    statuses = MARITAL_STATUSES_VI  # ['Chưa kết hôn', 'Đã kết hôn', 'Goá', 'Ly hôn']
    for age_group, sex, *dist in marital_template:
        for status, c in zip(statuses, dist):
            rows.append({"age_group": age_group, "sex": sex,
                         "marital_status": status, "count": float(c)})
    marital_x_age_x_sex = pd.DataFrame(rows)

    # education | age × urbanicity: PX-Web V02.54 has the *trained labour*
    # marginal but no age/urbanicity split. The shape below is consistent
    # with that marginal (~14% university, ~4% college, etc.) but tilts
    # urban higher and older lower per the long-running pattern.
    edu_marginal_pct = d.education_counts.set_index("education_level")["count"]
    # Urban → 1.4× the marginal share for college+; Rural → 0.7×. Younger
    # cohorts have higher modern-era enrolment.
    age_factor = {
        "15-19": 0.4, "20-24": 1.5, "25-29": 1.6, "30-34": 1.4,
        "35-39": 1.2, "40-44": 1.0, "45-49": 0.8, "50-54": 0.6,
        "55-59": 0.4, "60-64": 0.3, "65+":   0.2,
    }
    urb_factor = {"Thành thị": 1.6, "Nông thôn": 0.6}
    edu_levels = list(edu_marginal_pct.index)
    rows = []
    for ag in AGE_GROUPS_FINE_VI:
        for urb in URBANICITY_VI:
            for level in edu_levels:
                base = float(edu_marginal_pct[level])
                if level == "Không có trình độ CMKT":
                    weight = base * (1 / age_factor[ag]) * (1 / urb_factor[urb])
                else:
                    weight = base * age_factor[ag] * urb_factor[urb]
                rows.append({
                    "age_group": ag, "urbanicity": urb,
                    "education_level": level, "count": max(weight, 0.5),
                })
    education_x_age_x_urbanicity = pd.DataFrame(rows)

    # employment | age × sex: pulled straight from the PX-Web factorisation.
    employment_x_age_x_sex = d.employment_x_age_x_sex.copy()

    # occupation | education + industry | occupation: marginal-product approx.
    occupation_x_education = d.occupation_x_education.copy()
    industry_x_occupation = d.industry_x_occupation.copy()

    # income_quintile | education: synthetic (NSO HLSS)
    rows = []
    for level, dist in _INCOME_BY_EDUCATION.items():
        for q, c in zip(INCOME_QUINTILES_VI, dist):
            rows.append({"education_level": level, "income_quintile": q, "count": float(c)})
    income_x_education = pd.DataFrame(rows)

    # SDG-PGMs requires every variable column to be a `pd.Categorical`
    # with a globally-consistent set of categories. Build one shared
    # CategoricalDtype per variable, then apply it to every DataFrame
    # that contains that variable.
    occ_cats = sorted(occupation_x_education["occupation"].unique().tolist())
    ind_cats = sorted(industry_x_occupation["industry_sector"].unique().tolist())
    emp_cats = sorted(employment_x_age_x_sex["employment_status"].unique().tolist())

    cat_dtypes = {
        "region":            pd.CategoricalDtype(categories=REGIONS_VI, ordered=False),
        "urbanicity":        pd.CategoricalDtype(categories=URBANICITY_VI, ordered=False),
        "age_group":         pd.CategoricalDtype(categories=AGE_GROUPS_FINE_VI, ordered=False),
        "sex":               pd.CategoricalDtype(categories=SEXES_VI, ordered=False),
        "ethnicity":         pd.CategoricalDtype(categories=ETHNICITIES_VI, ordered=False),
        "marital_status":    pd.CategoricalDtype(categories=MARITAL_STATUSES_VI, ordered=False),
        "education_level":   pd.CategoricalDtype(categories=EDUCATION_LEVELS_VI, ordered=False),
        "occupation":        pd.CategoricalDtype(categories=occ_cats, ordered=False),
        "industry_sector":   pd.CategoricalDtype(categories=ind_cats, ordered=False),
        "employment_status": pd.CategoricalDtype(categories=emp_cats, ordered=False),
        "income_quintile":   pd.CategoricalDtype(categories=INCOME_QUINTILES_VI, ordered=False),
    }

    def _categoricalise(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, dt in cat_dtypes.items():
            if col in out.columns:
                out[col] = out[col].astype(dt)
        return out

    return VNPersonaData(
        region_counts=_categoricalise(d.region_counts),
        region_x_urbanicity=_categoricalise(d.region_x_urbanicity),
        age_group_x_region=_categoricalise(age_group_x_region),
        sex_x_age_group=_categoricalise(sex_x_age_group),
        ethnicity_x_region=_categoricalise(ethnicity_x_region),
        marital_x_age_x_sex=_categoricalise(marital_x_age_x_sex),
        education_x_age_x_urbanicity=_categoricalise(education_x_age_x_urbanicity),
        employment_x_age_x_sex=_categoricalise(employment_x_age_x_sex),
        occupation_x_education=_categoricalise(occupation_x_education),
        industry_x_occupation=_categoricalise(industry_x_occupation),
        income_x_education=_categoricalise(income_x_education),
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class VNPersonaGenerator(PGMGenerator):
    """SDG-PGMs persona generator grounded in NSO PX-Web statistics."""

    def __init__(self, data_path: str | None = None, seed: int | None = None) -> None:
        # SDG-PGMs' base __init__ runs the full template-method lifecycle
        # (get_data → get_variables → get_mappings → get_cpds → build).
        # ``seed`` is consumed by ``generate_samples`` later, not by ``__init__``.
        self._seed = seed
        super().__init__(data_path=data_path)

    # ----- abstract methods ------------------------------------------------
    def get_data(self, data_path: str | None = None) -> VNPersonaData:
        if data_path:
            return VNPersonaData.from_tarfile(data_path, _TAR_MAPPINGS)
        return _build_data_from_pxweb()

    def get_variables(self, data: VNPersonaData) -> dict[str, list[str]]:
        return {
            "region":            REGIONS_VI,
            "urbanicity":        URBANICITY_VI,
            "age_group":         AGE_GROUPS_FINE_VI,
            "sex":               SEXES_VI,
            "ethnicity":         ETHNICITIES_VI,
            "marital_status":    MARITAL_STATUSES_VI,
            "education_level":   EDUCATION_LEVELS_VI,
            "employment_status": sorted(data.employment_x_age_x_sex["employment_status"].unique().tolist()),
            "occupation":        sorted(data.occupation_x_education["occupation"].unique().tolist()),
            "industry_sector":   sorted(data.industry_x_occupation["industry_sector"].unique().tolist()),
            "income_quintile":   INCOME_QUINTILES_VI,
        }

    def get_edges(self, *args, **kwargs) -> list[list[Edge]]:
        return [
            # Stage 1 — geography
            [
                Edge(start="region", end="urbanicity"),
            ],
            # Stage 2 — demographics
            [
                Edge(start="region", end="age_group"),
                Edge(start="age_group", end="sex"),
                Edge(start="region", end="ethnicity"),
                Edge(start="age_group", end="marital_status"),
                Edge(start="sex", end="marital_status"),
            ],
            # Stage 3 — socio-economic
            [
                Edge(start="age_group", end="education_level"),
                Edge(start="urbanicity", end="education_level"),
                Edge(start="age_group", end="employment_status"),
                Edge(start="sex", end="employment_status"),
                Edge(start="education_level", end="occupation"),
                Edge(start="occupation", end="industry_sector"),
                Edge(start="education_level", end="income_quintile"),
            ],
        ]

    def get_cpds(self, data: VNPersonaData) -> list[dict[str, TabularCPD]]:
        v = self.variables  # populated by base.__init__
        # ----- Stage 1: region (root) + urbanicity | region -----
        region_cpd = self._marginal_cpd(
            counts=data.region_counts,
            variable="region",
            categories=v["region"],
        )
        urbanicity_cpd = self.fill_na_and_gen_cpd(
            counts=data.region_x_urbanicity, variable_name="urbanicity",
            evidence=["region"],
        )
        # ----- Stage 2: demographics -----
        age_cpd = self.fill_na_and_gen_cpd(
            counts=data.age_group_x_region, variable_name="age_group",
            evidence=["region"],
        )
        sex_cpd = self.fill_na_and_gen_cpd(
            counts=data.sex_x_age_group, variable_name="sex",
            evidence=["age_group"],
        )
        ethnicity_cpd = self.fill_na_and_gen_cpd(
            counts=data.ethnicity_x_region, variable_name="ethnicity",
            evidence=["region"],
        )
        marital_cpd = self.fill_na_and_gen_cpd(
            counts=data.marital_x_age_x_sex, variable_name="marital_status",
            evidence=["age_group", "sex"],
        )
        # ----- Stage 3: socio-economic -----
        education_cpd = self.fill_na_and_gen_cpd(
            counts=data.education_x_age_x_urbanicity, variable_name="education_level",
            evidence=["age_group", "urbanicity"],
        )
        employment_cpd = self.fill_na_and_gen_cpd(
            counts=data.employment_x_age_x_sex, variable_name="employment_status",
            evidence=["age_group", "sex"],
        )
        occupation_cpd = self.fill_na_and_gen_cpd(
            counts=data.occupation_x_education, variable_name="occupation",
            evidence=["education_level"],
        )
        industry_cpd = self.fill_na_and_gen_cpd(
            counts=data.industry_x_occupation, variable_name="industry_sector",
            evidence=["occupation"],
        )
        income_cpd = self.fill_na_and_gen_cpd(
            counts=data.income_x_education, variable_name="income_quintile",
            evidence=["education_level"],
        )

        return [
            {"region": region_cpd, "urbanicity": urbanicity_cpd},
            {
                "age_group": age_cpd,
                "sex": sex_cpd,
                "ethnicity": ethnicity_cpd,
                "marital_status": marital_cpd,
            },
            {
                "education_level": education_cpd,
                "employment_status": employment_cpd,
                "occupation": occupation_cpd,
                "industry_sector": industry_cpd,
                "income_quintile": income_cpd,
            },
        ]

    def get_postprocessing_steps(self) -> list[list[Callable] | None] | None:
        return [
            None,                             # Stage 1: nothing
            [self._derive_age],               # Stage 2: integer age
            [self._nullify_for_non_workers,   # Stage 3: clean up + naming + bio
             self._add_name,
             self._add_bilingual_bio],
        ]

    def get_mappings(self, data: VNPersonaData) -> dict[str, pd.DataFrame]:
        # Lookup tables for vi → en label translation; the post-processing
        # step uses these to add ``_en`` mirror columns.
        return {
            "region_lookup": pd.DataFrame(
                [{"region": k, "region_en": v} for k, v in REGIONS_EN.items()]
            ),
            "urbanicity_lookup": pd.DataFrame(
                [{"urbanicity": k, "urbanicity_en": v} for k, v in URBANICITY_EN.items()]
            ),
            "sex_lookup": pd.DataFrame(
                [{"sex": k, "sex_en": v} for k, v in SEXES_EN.items()]
            ),
            "education_lookup": pd.DataFrame(
                [{"education_level": k, "education_level_en": v}
                 for k, v in EDUCATION_LEVELS_EN.items()]
            ),
            "occupation_lookup": pd.DataFrame(
                [{"occupation": k, "occupation_en": OCCUPATIONS_EN_HINTS.get(k, k)}
                 for k in self.variables["occupation"]]
            ),
            "employment_lookup": pd.DataFrame(
                [{"employment_status": k,
                  "employment_status_en": EMPLOYMENT_STATUSES_EN_HINTS.get(k, k)}
                 for k in self.variables["employment_status"]]
            ),
        }

    # ----- helpers ---------------------------------------------------------
    @staticmethod
    def _marginal_cpd(counts: pd.DataFrame, variable: str, categories: list[str]) -> TabularCPD:
        """Build a marginal :class:`TabularCPD` from a counts DataFrame."""
        c = counts.set_index(variable)["count"].reindex(categories).fillna(1.0).values
        c = c / c.sum()
        return TabularCPD(
            variable=variable,
            variable_card=len(categories),
            values=c.reshape(-1, 1),
            state_names={variable: list(categories)},
        )

    # ----- post-processing -------------------------------------------------
    AGE_MIDPOINTS = {"15-19": 17, "20-24": 22, "25-29": 27, "30-34": 32,
                      "35-39": 37, "40-44": 42, "45-49": 47, "50-54": 52,
                      "55-59": 57, "60-64": 62, "65+": 70}

    def _derive_age(self, samples: pd.DataFrame) -> pd.DataFrame:
        if "age_group" not in samples.columns:
            return samples
        rng = np.random.default_rng(self._seed)
        mids = samples["age_group"].map(self.AGE_MIDPOINTS).fillna(30).astype(int).to_numpy()
        samples["age"] = np.maximum(15, mids + rng.integers(-2, 3, size=len(samples)))
        return samples

    @staticmethod
    def _nullify_for_non_workers(samples: pd.DataFrame) -> pd.DataFrame:
        non_workers = ~samples["employment_status"].isin(["Làm công ăn lương", "Tự làm",
                                                            "Chủ cơ sở sản xuất kinh doanh"])
        for col in ("occupation", "industry_sector"):
            if col in samples.columns:
                samples.loc[non_workers, col] = "Không áp dụng"
        return samples

    def _add_name(self, samples: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed)
        n = len(samples)
        family_arr = np.array(_FAMILY_NAMES)
        male_mid = np.array(_MIDDLE_NAMES_M)
        female_mid = np.array(_MIDDLE_NAMES_F)
        male_giv = np.array(_GIVEN_NAMES_M)
        female_giv = np.array(_GIVEN_NAMES_F)

        family = family_arr[rng.integers(0, len(family_arr), size=n)]
        is_male = (samples["sex"].to_numpy() == "Nam")
        middle = np.where(is_male,
                          male_mid[rng.integers(0, len(male_mid), size=n)],
                          female_mid[rng.integers(0, len(female_mid), size=n)])
        given = np.where(is_male,
                         male_giv[rng.integers(0, len(male_giv), size=n)],
                         female_giv[rng.integers(0, len(female_giv), size=n)])
        full = pd.Series(family).str.cat(pd.Series(middle), sep=" ").str.cat(pd.Series(given), sep=" ")
        samples["name"] = full.to_numpy()
        samples["persona_id"] = [_slug_id(i, n_) for i, n_ in enumerate(full.to_numpy())]
        return samples

    @staticmethod
    def _add_bilingual_bio(samples: pd.DataFrame) -> pd.DataFrame:
        """Add ``_en`` mirror columns + bilingual templated bios."""
        # Lookup-based label translation (mappings populated in get_mappings).
        samples["region_en"] = samples["region"].map(REGIONS_EN).fillna(samples["region"])
        samples["urbanicity_en"] = samples["urbanicity"].map(URBANICITY_EN).fillna(samples["urbanicity"])
        samples["sex_en"] = samples["sex"].map(SEXES_EN).fillna(samples["sex"])
        samples["education_level_en"] = samples["education_level"].map(EDUCATION_LEVELS_EN).fillna(samples["education_level"])
        samples["occupation_en"] = samples["occupation"].map(OCCUPATIONS_EN_HINTS).fillna(samples["occupation"])
        samples["employment_status_en"] = samples["employment_status"].map(EMPLOYMENT_STATUSES_EN_HINTS).fillna(samples["employment_status"])
        samples["industry_sector_en"] = samples["industry_sector"]   # untranslated; many free-form labels
        samples["ethnicity_en"] = samples["ethnicity"]
        marital_en = {"Chưa kết hôn": "single", "Đã kết hôn": "married",
                      "Goá": "widowed", "Ly hôn": "divorced"}
        samples["marital_status_en"] = samples["marital_status"].map(marital_en).fillna(samples["marital_status"])
        samples["income_quintile_en"] = samples["income_quintile"]

        def col(name: str) -> pd.Series:
            return samples[name].astype(str)

        samples["bio_vi"] = (
            "Một người " + col("sex") + " " + col("age").astype(str) + " tuổi, "
            + "thuộc dân tộc " + col("ethnicity") + ", "
            + "sống ở " + col("urbanicity") + " thuộc " + col("region") + ". "
            + "Tình trạng hôn nhân: " + col("marital_status") + ". "
            + "Trình độ chuyên môn kỹ thuật: " + col("education_level") + ". "
            + "Vị thế việc làm: " + col("employment_status")
            + "; nghề nghiệp: " + col("occupation")
            + "; ngành: " + col("industry_sector")
            + "; nhóm thu nhập: " + col("income_quintile") + "."
        )
        samples["bio_en"] = (
            "A " + col("age").astype(str) + "-year-old "
            + col("sex_en") + " of " + col("ethnicity_en") + " ethnicity "
            + "living in a " + col("urbanicity_en") + " area of the "
            + col("region_en") + ". "
            + "Marital status: " + col("marital_status_en") + ". "
            + "Education: " + col("education_level_en") + ". "
            + "Employment status: " + col("employment_status_en")
            + "; occupation: " + col("occupation_en")
            + "; industry: " + col("industry_sector_en")
            + "; income quintile: " + col("income_quintile") + "."
        )
        return samples


# ---------------------------------------------------------------------------
# Public entry point — wraps DataFrame samples in typed Persona objects
# ---------------------------------------------------------------------------
def generate_personas(
    n: int,
    *,
    registry: OntologyRegistry | None = None,
    seed: int | None = None,
    scrape_manifest: dict[str, Any] | None = None,
    data_path: str | None = None,
) -> PersonaBatch:
    """Generate ``n`` personas via the SDG-PGMs cascaded sampler."""
    registry = registry or load_registry()
    seed = seed if seed is not None else int(registry.output.get("random_seed", 0))
    log.info("generating %d personas via SDG-PGMs (seed=%s)", n, seed)

    gen = VNPersonaGenerator(data_path=data_path, seed=seed)
    df = gen.generate_samples(size=n, seed=seed, disable_progress_bar=True)

    sources = sorted({f"nso:{ep}" for ep in (scrape_manifest or {}).get("endpoints", {})})
    if not sources:
        # Default provenance: the PX-Web tables we built distributions from.
        sources = ["pxweb:V02.01", "pxweb:V02.02", "pxweb:V02.41", "pxweb:V02.43",
                   "pxweb:V02.44", "pxweb:V02.54", "pxweb:V02.42"]

    records = df.to_dict(orient="records")
    personas: list[Persona] = []
    for r in records:
        personas.append(Persona(
            persona_id=r["persona_id"], name=r["name"],
            region=r["region"], region_en=r["region_en"],
            urbanicity=r["urbanicity"], urbanicity_en=r["urbanicity_en"],
            age_group=r["age_group"], age=int(r["age"]),
            sex=r["sex"], sex_en=r["sex_en"],
            ethnicity=r["ethnicity"], ethnicity_en=r["ethnicity_en"],
            marital_status=r["marital_status"], marital_status_en=r["marital_status_en"],
            education_level=r["education_level"], education_level_en=r["education_level_en"],
            employment_status=r["employment_status"], employment_status_en=r["employment_status_en"],
            occupation=r["occupation"], occupation_en=r["occupation_en"],
            industry_sector=r["industry_sector"], industry_sector_en=r["industry_sector_en"],
            income_quintile=r["income_quintile"], income_quintile_en=r["income_quintile_en"],
            bio_vi=r.get("bio_vi"), bio_en=r.get("bio_en"),
            enriched=False,
            sources=sources,
        ))

    return PersonaBatch(
        generated_at=datetime.now(timezone.utc),
        n=len(personas), seed=seed,
        ontology_version=registry.version,
        scrape_manifest=scrape_manifest or {},
        enrichment={},
        personas=personas,
    )
