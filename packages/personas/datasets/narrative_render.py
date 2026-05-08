"""Vectorised renderers for the 11 Nemotron narrative columns.

Public entry point :func:`render_narratives` takes a *structured*
DataFrame (the output of the PGM generator + name + province draw) and
returns a new DataFrame whose columns are the 11 narrative strings the
Nemotron schema requires, in the requested language (``"vi"`` or
``"en"``).

Design choices
==============
* **Vectorisation** — every per-row draw is a single numpy call against
  a pre-allocated lookup array. No Python-level loops over rows.
* **Reproducibility** — all random draws come from a single
  :class:`numpy.random.Generator` passed in by the builder. The same
  ``rng`` produces identical narrative variants on every run.
* **Bilingual symmetry** — for a given row, the *vi* render and the
  *en* render select the same variant index in every lookup. This
  means ``persona-vi[i]`` and ``persona-en[i]`` describe the same
  fictional person, just in different languages. The builder ensures
  symmetry by passing the *same* ``rng`` state (a re-seeded copy) when
  rendering each language.
* **Grounded** — every variant cites a structured fact (region cuisine,
  occupation skills, …) so each narrative remains anchored in the NSO
  data the structured fields are themselves drawn from.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from packages.personas.datasets.narrative_data import (
    GOALS_EN,
    GOALS_VI,
    HOBBIES_RURAL_EN,
    HOBBIES_RURAL_VI,
    HOBBIES_URBAN_EN,
    HOBBIES_URBAN_VI,
    OCCUPATION_SKILLS_EN,
    OCCUPATION_SKILLS_VI,
    REGION_CULTURE_EN,
    REGION_CULTURE_VI,
    REGION_DISHES_EN,
    REGION_DISHES_VI,
    REGION_SCENIC_EN,
    REGION_SCENIC_VI,
    REGION_SPORTS_EN,
    REGION_SPORTS_VI,
)
from packages.personas.datasets.schema import (
    AREA_VI_TO_EN,
    EDUCATION_VI_TO_EN,
    MARITAL_VI_TO_EN,
    OCCUPATION_VI_TO_EN,
    REGIONS_VI_TO_EN,
    SEX_VI_TO_EN,
)

Lang = Literal["vi", "en"]


# ---------------------------------------------------------------------------
# Prose-only English forms with proper articles.
#
# The Nemotron schema's ``occupation`` / ``education_level`` columns hold the
# canonical (plural / article-less) ISCO and qualification labels — that
# format matches Nemotron-Personas-USA, -Japan, etc., and downstream
# consumers expect exactly those strings. But those labels read awkwardly
# when dropped into English prose ("works as professionals"). The maps
# below rewrite them into a singular, article-prefixed *narrative form*
# used **only** by the English renderer; the structured ``occupation`` /
# ``education_level`` cells are unaffected.
# ---------------------------------------------------------------------------
OCCUPATION_NARRATIVE_EN: dict[str, str] = {
    "managers":                                              "a manager",
    "professionals":                                         "a professional",
    "associate professionals":                               "an associate professional",
    "clerical support":                                      "a clerical support worker",
    "service and sales workers":                             "a service-and-sales worker",
    "skilled agricultural, forestry and fishery workers":    "a skilled agriculture, forestry or fishery worker",
    "craft and related trades workers":                      "a craft and trades worker",
    "plant and machine operators and assemblers":            "a plant and machine operator",
    "elementary occupations":                                "an elementary-occupation worker",
    "armed forces":                                          "a member of the armed forces",
    "other":                                                 "a worker",
    "not applicable":                                        "primarily occupied with home and family duties",
}


EDUCATION_NARRATIVE_EN: dict[str, str] = {
    "no formal qualification":         "no formal qualification",
    "primary vocational":              "a primary vocational qualification",
    "intermediate vocational":         "an intermediate vocational qualification",
    "junior college":                  "a junior-college qualification",
    "bachelor's degree or higher":     "a bachelor's degree or higher",
}


AREA_INDEFINITE_EN: dict[str, str] = {
    "urban": "an urban",
    "rural": "a rural",
}


def _occ_prose_en(series: pd.Series) -> pd.Series:
    return series.map(OCCUPATION_NARRATIVE_EN).fillna("a worker")


def _edu_prose_en(series: pd.Series) -> pd.Series:
    return series.map(EDUCATION_NARRATIVE_EN).fillna(series)


def _area_prose_en(series: pd.Series) -> pd.Series:
    return series.map(AREA_INDEFINITE_EN).fillna("a " + series.astype(str))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Universe size for the pre-allocated per-row permutation matrices
# (large enough to cover every per-key pool we look up). 16 fits
# comfortably ahead of the largest pool today (HOBBIES_*, len 10).
PERM_UNIVERSE: int = 16


def precompute_narrative_picks(
    n_rows: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Pre-allocate per-row permutation matrices for every concern.

    The renderer has **9 separate concerns** that each draw 2-4 items
    from a per-key pool. Giving each its own independent permutation
    array makes the picks decorrelated across concerns within a row
    (otherwise ``professional_persona`` would repeat skills already in
    ``skills_and_expertise``).

    Memory: ``9 × n_rows × 16 × uint8`` ≈ 432 MB at the 3 M scale —
    transient (only held until rendering finishes), and dwarfed by the
    output parquet bytes.

    Returns a dict keyed by *concern name*; values are ``uint8`` arrays
    of shape ``(n_rows, PERM_UNIVERSE)`` holding a per-row permutation
    of ``range(PERM_UNIVERSE)``. The downstream renderer projects each
    permutation into its actual per-key pool by keeping only entries
    ``< pool_size`` (so the picks within a row are guaranteed distinct).

    Reproducibility: row ``i``'s picks are a deterministic function of
    ``(seed, i, concern_name)`` only — independent of chunk size.
    """
    def _row_perm() -> np.ndarray:
        return rng.permuted(
            np.broadcast_to(
                np.arange(PERM_UNIVERSE, dtype=np.uint8), (n_rows, PERM_UNIVERSE)
            ).copy(),
            axis=1,
        )

    # Order matters: rng is consumed in this order, so renaming or
    # reordering keys here would break reproducibility.
    return {
        # occupation-conditional concerns
        "professional": _row_perm(),
        "skills":       _row_perm(),
        "goals":        _row_perm(),
        # region-conditional concerns
        "sports":       _row_perm(),
        "arts":         _row_perm(),
        "travel":       _row_perm(),
        "culinary":     _row_perm(),
        "culture":      _row_perm(),
        # area-conditional concerns
        "hobbies":      _row_perm(),
    }


def _resolve_picks_by_key(
    keys: pd.Series,
    table: dict[str, list[str]],
    perm_chunk: np.ndarray,
    n_picks: int,
) -> np.ndarray:
    """Resolve pre-allocated permutations into n_picks items per row.

    For each row ``i`` we look up the pool ``table[keys[i]]`` (size
    ``s``), filter ``perm_chunk[i]`` to keep only entries ``< s`` (so
    each survivor is a *distinct* index in ``[0, s)``), then take the
    first ``n_picks`` survivors. If ``s < n_picks`` (rare — only for
    pools deliberately defined with fewer items than callers ask for)
    we wrap by repetition.
    """
    n_rows = perm_chunk.shape[0]
    if n_rows == 0:
        return np.empty((0, n_picks), dtype=object)

    pools = {k: np.asarray(v, dtype=object) for k, v in table.items()}
    fallback_key = next(
        (k for k in ("Khác", "Không áp dụng") if k in pools),
        next(iter(pools)),
    )
    out = np.empty((n_rows, n_picks), dtype=object)

    keys_pos = keys.reset_index(drop=True)
    for key, idx_arr in keys_pos.groupby(keys_pos, observed=True).indices.items():
        pool = pools.get(key, pools[fallback_key])
        size = len(pool)
        if size == 0:
            continue

        sub_perm = perm_chunk[idx_arr]
        # Keep only indices that fall within the pool. ``argsort(~mask)``
        # is a stable partition that places valid (True) entries first
        # while preserving their relative order — so the per-row pick
        # order is the same as the order they appeared in the original
        # universe permutation. That is what makes the function purely
        # a function of (perm_chunk, table, key).
        mask = sub_perm < size
        order = np.argsort(~mask, axis=1, kind="stable")
        gathered = np.take_along_axis(sub_perm, order, axis=1)

        if size >= n_picks:
            picks = gathered[:, :n_picks]
        else:
            # Pool smaller than caller asked: wrap by repetition. Rare.
            reps = (n_picks + size - 1) // size
            picks = np.tile(gathered[:, :size], (1, reps))[:, :n_picks]

        out[idx_arr] = pool[picks]
    return out


def _resolve_en(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``*_en`` mirror columns derived from the canonical Vietnamese ones."""
    out = df.copy()
    out["region_en"] = out["region"].map(REGIONS_VI_TO_EN).fillna(out["region"])
    out["area_en"] = out["area"].map(AREA_VI_TO_EN).fillna(out["area"])
    out["sex_en"] = out["sex"].map(SEX_VI_TO_EN).fillna(out["sex"])
    out["marital_status_en"] = out["marital_status"].map(MARITAL_VI_TO_EN).fillna(
        out["marital_status"]
    )
    out["education_level_en"] = out["education_level"].map(EDUCATION_VI_TO_EN).fillna(
        out["education_level"]
    )
    out["occupation_en"] = out["occupation"].map(OCCUPATION_VI_TO_EN).fillna(
        out["occupation"]
    )
    return out


# ---------------------------------------------------------------------------
# Per-field renderers (return one Series each)
# ---------------------------------------------------------------------------
def _render_culinary(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> pd.Series:
    table = REGION_DISHES_VI if lang == "vi" else REGION_DISHES_EN
    picks = _resolve_picks_by_key(df["region"], table, picks_chunk["culinary"], n_picks=2)
    name = df["name"].astype(str).to_numpy()
    region = df["region_en" if lang == "en" else "region"].astype(str).to_numpy()
    if lang == "vi":
        out = (
            name + " thường thưởng thức ẩm thực " + region
            + ", đặc biệt yêu thích " + picks[:, 0].astype(str)
            + " và " + picks[:, 1].astype(str)
            + "; vào cuối tuần thường tự nấu các món truyền thống"
            " để chiêu đãi gia đình."
        )
    else:
        out = (
            name + " enjoys the cuisine of the " + region
            + ", with a particular fondness for " + picks[:, 0].astype(str)
            + " and " + picks[:, 1].astype(str)
            + "; on weekends they like to cook traditional dishes for the family."
        )
    return pd.Series(out, index=df.index)


def _render_sports(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> pd.Series:
    table = REGION_SPORTS_VI if lang == "vi" else REGION_SPORTS_EN
    picks = _resolve_picks_by_key(df["region"], table, picks_chunk["sports"], n_picks=2)
    name = df["name"].astype(str).to_numpy()
    age = df["age"].astype(int).astype(str).to_numpy()
    if lang == "vi":
        out = (
            "Ở tuổi " + age + ", " + name
            + " duy trì sức khoẻ qua " + picks[:, 0].astype(str)
            + " và thỉnh thoảng tham gia " + picks[:, 1].astype(str) + "."
        )
    else:
        out = (
            "At " + age + ", " + name
            + " stays active through " + picks[:, 0].astype(str)
            + " and occasionally takes part in " + picks[:, 1].astype(str) + "."
        )
    return pd.Series(out, index=df.index)


def _render_arts(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> pd.Series:
    table = REGION_CULTURE_VI if lang == "vi" else REGION_CULTURE_EN
    picks = _resolve_picks_by_key(df["region"], table, picks_chunk["arts"], n_picks=2)
    name = df["name"].astype(str).to_numpy()
    if lang == "vi":
        out = (
            name + " yêu thích các loại hình nghệ thuật truyền thống quê hương: "
            + picks[:, 0].astype(str) + " và " + picks[:, 1].astype(str)
            + ", thường tham gia hoặc theo dõi vào những dịp lễ hội trong năm."
        )
    else:
        out = (
            name + " loves the traditional arts of their home region — "
            + picks[:, 0].astype(str) + " and " + picks[:, 1].astype(str)
            + " — and joins or follows them whenever a festival comes around."
        )
    return pd.Series(out, index=df.index)


def _render_travel(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> pd.Series:
    table = REGION_SCENIC_VI if lang == "vi" else REGION_SCENIC_EN
    picks = _resolve_picks_by_key(df["region"], table, picks_chunk["travel"], n_picks=2)
    name = df["name"].astype(str).to_numpy()
    region = df["region_en" if lang == "en" else "region"].astype(str).to_numpy()
    if lang == "vi":
        out = (
            name + " thường nghỉ ngơi tại các điểm đến nổi tiếng của "
            + region + " như " + picks[:, 0].astype(str)
            + " và " + picks[:, 1].astype(str)
            + "; những chuyến đi ngắn cuối tuần là cách họ cân bằng nhịp sống."
        )
    else:
        out = (
            name + " unwinds at well-known destinations in the "
            + region + " such as " + picks[:, 0].astype(str)
            + " and " + picks[:, 1].astype(str)
            + "; short weekend trips are how they keep life in balance."
        )
    return pd.Series(out, index=df.index)


def _render_professional(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> pd.Series:
    table = OCCUPATION_SKILLS_VI if lang == "vi" else OCCUPATION_SKILLS_EN
    picks = _resolve_picks_by_key(df["occupation"], table, picks_chunk["professional"], n_picks=3)
    name = df["name"].astype(str).to_numpy()
    occ_prose = _occ_prose_en(df["occupation_en"]).to_numpy()
    edu_vi = df["education_level"].astype(str).to_numpy()
    edu_prose = _edu_prose_en(df["education_level_en"]).to_numpy()
    province = df["province"].astype(str).to_numpy()
    if lang == "vi":
        out = (
            name + " làm việc trong nhóm nghề " + df["occupation"].astype(str).to_numpy()
            + " tại " + province
            + ". Trình độ chuyên môn của họ là " + edu_vi
            + ", và họ thành thạo " + picks[:, 0].astype(str)
            + ", " + picks[:, 1].astype(str)
            + " và " + picks[:, 2].astype(str)
            + ". Họ làm việc cẩn thận, có trách nhiệm và luôn học hỏi cái mới."
        )
    else:
        out = (
            name + " works as " + occ_prose
            + " in " + province
            + ", with " + edu_prose
            + ", and is skilled in " + picks[:, 0].astype(str)
            + ", " + picks[:, 1].astype(str)
            + ", and " + picks[:, 2].astype(str)
            + ". They are careful, responsible, and continually keen to learn."
        )
    return pd.Series(out, index=df.index)


def _render_skills(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> tuple[pd.Series, pd.Series]:
    """Returns (skills_and_expertise prose, skills_and_expertise_list str)."""
    table = OCCUPATION_SKILLS_VI if lang == "vi" else OCCUPATION_SKILLS_EN
    picks = _resolve_picks_by_key(df["occupation"], table, picks_chunk["skills"], n_picks=4)
    sep = ", "
    list_str = pd.Series(
        [sep.join(row) for row in picks.tolist()],
        index=df.index,
    )
    if lang == "vi":
        prose = pd.Series(
            "Các kỹ năng nổi bật bao gồm: " + list_str + ".",
            index=df.index,
        )
    else:
        prose = pd.Series(
            "Their core skills include: " + list_str + ".",
            index=df.index,
        )
    return prose, list_str


def _render_hobbies(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> tuple[pd.Series, pd.Series]:
    """Hobbies are conditioned on area (urban vs rural)."""
    if lang == "vi":
        urban_pool = np.asarray(HOBBIES_URBAN_VI, dtype=object)
        rural_pool = np.asarray(HOBBIES_RURAL_VI, dtype=object)
    else:
        urban_pool = np.asarray(HOBBIES_URBAN_EN, dtype=object)
        rural_pool = np.asarray(HOBBIES_RURAL_EN, dtype=object)

    n = len(df)
    is_urban = (df["area"] == "Thành thị").to_numpy()
    n_picks = 3
    out = np.empty((n, n_picks), dtype=object)

    # Use the precomputed permutation; project into the urban / rural
    # pool sizes by keeping only entries < pool_size and taking the
    # first ``n_picks`` survivors.
    perm = picks_chunk["hobbies"]

    def _project(pool_size: int, mask_urban: bool) -> np.ndarray:
        sel_rows = np.where(is_urban) if mask_urban else np.where(~is_urban)
        sub_perm = perm[sel_rows]
        valid = sub_perm < pool_size
        order = np.argsort(~valid, axis=1, kind="stable")
        return np.take_along_axis(sub_perm, order, axis=1)[:, :n_picks]

    out[is_urban] = urban_pool[_project(len(urban_pool), True)]
    out[~is_urban] = rural_pool[_project(len(rural_pool), False)]

    list_str = pd.Series([", ".join(row) for row in out.tolist()], index=df.index)
    if lang == "vi":
        prose = pd.Series(
            "Sở thích thường ngày của họ bao gồm: " + list_str + ".",
            index=df.index,
        )
    else:
        prose = pd.Series(
            "Their everyday interests include: " + list_str + ".",
            index=df.index,
        )
    return prose, list_str


def _render_goals(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> pd.Series:
    table = GOALS_VI if lang == "vi" else GOALS_EN
    picks = _resolve_picks_by_key(df["occupation"], table, picks_chunk["goals"], n_picks=2)
    name = df["name"].astype(str).to_numpy()
    if lang == "vi":
        out = (
            "Trong vài năm tới, " + name + " mong muốn "
            + picks[:, 0].astype(str)
            + ", đồng thời " + picks[:, 1].astype(str) + "."
        )
    else:
        out = (
            "Over the next few years, " + name + " hopes to "
            + picks[:, 0].astype(str)
            + ", and to " + picks[:, 1].astype(str) + "."
        )
    return pd.Series(out, index=df.index)


def _render_culture(df: pd.DataFrame, lang: Lang, picks_chunk: dict) -> pd.Series:
    table = REGION_CULTURE_VI if lang == "vi" else REGION_CULTURE_EN
    picks = _resolve_picks_by_key(df["region"], table, picks_chunk["culture"], n_picks=2)
    name = df["name"].astype(str).to_numpy()
    province = df["province"].astype(str).to_numpy()
    if lang == "vi":
        region = df["region"].astype(str).to_numpy()
        area_label = df["area"].astype(str).to_numpy()
        out = (
            name + " lớn lên ở khu vực " + area_label + " thuộc " + province
            + " (" + region + "). Văn hoá địa phương để lại dấu ấn rõ rệt: "
            + picks[:, 0].astype(str) + " và " + picks[:, 1].astype(str)
            + " là những giá trị họ luôn trân trọng và truyền lại cho thế hệ sau."
        )
    else:
        region = df["region_en"].astype(str).to_numpy()
        area_prose = _area_prose_en(df["area_en"]).to_numpy()
        out = (
            name + " grew up in " + area_prose + " part of " + province
            + " in the " + region + ". The local culture left a clear imprint: "
            + picks[:, 0].astype(str) + " and " + picks[:, 1].astype(str)
            + " are values they cherish and pass on to the next generation."
        )
    return pd.Series(out, index=df.index)


def _render_summary(df: pd.DataFrame, lang: Lang) -> pd.Series:
    """The short ``persona`` field — 1 dense sentence spanning everything."""
    name = df["name"].astype(str).to_numpy()
    age = df["age"].astype(int).astype(str).to_numpy()
    if lang == "vi":
        sex = df["sex"].astype(str).to_numpy()
        marital = df["marital_status"].astype(str).to_numpy()
        edu = df["education_level"].astype(str).to_numpy()
        occ = df["occupation"].astype(str).to_numpy()
        province = df["province"].astype(str).to_numpy()
        region = df["region"].astype(str).to_numpy()
        area = df["area"].astype(str).to_numpy()
        out = (
            name + ", " + age + " tuổi, " + sex.astype(object).astype(str)
            + ", " + marital + ", trình độ " + edu
            + ", làm việc trong nhóm nghề " + occ
            + ", sinh sống tại khu vực " + area
            + " thuộc " + province + " (" + region + ")."
        )
    else:
        sex = df["sex_en"].astype(str).to_numpy()
        marital = df["marital_status_en"].astype(str).to_numpy()
        edu_prose = _edu_prose_en(df["education_level_en"]).to_numpy()
        occ_prose = _occ_prose_en(df["occupation_en"]).to_numpy()
        province = df["province"].astype(str).to_numpy()
        region = df["region_en"].astype(str).to_numpy()
        area_prose = _area_prose_en(df["area_en"]).to_numpy()
        out = (
            name + ", a " + age + "-year-old " + sex
            + ", " + marital + " with " + edu_prose
            + ", works as " + occ_prose
            + " and lives in " + area_prose + " part of "
            + province + " in the " + region + "."
        )
    return pd.Series(out, index=df.index)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render_narratives(
    structured: pd.DataFrame,
    lang: Lang,
    picks_chunk: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Render the 11 narrative columns for one language.

    Parameters
    ----------
    structured:
        Must contain at least these columns (Vietnamese canonical
        labels): ``region``, ``area``, ``province``, ``sex``, ``age``,
        ``marital_status``, ``education_level``, ``occupation``,
        ``name``. The function adds the ``*_en`` mirrors itself.
    lang:
        ``"vi"`` for Vietnamese cell values, ``"en"`` for English.
    picks_chunk:
        Dict of pre-computed permutation arrays produced by
        :func:`precompute_narrative_picks` (and sliced for this
        chunk). Each value is shape ``(len(structured), PERM_UNIVERSE)``.
        Passing the same ``picks_chunk`` to a vi and an en call yields
        the parallel-translation pairing required for the
        small/large × vi/en datasets to describe the same row.
        Crucially, this also makes the renderer **chunk-size invariant**
        — slicing the picks differently doesn't change a row's pick.
    """
    if structured.empty:
        cols = [
            "professional_persona", "sports_persona", "arts_persona",
            "travel_persona", "culinary_persona", "persona",
            "cultural_background", "skills_and_expertise",
            "skills_and_expertise_list", "hobbies_and_interests",
            "hobbies_and_interests_list", "career_goals_and_ambitions",
        ]
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})

    # Reset to a clean 0..n-1 RangeIndex so every per-renderer numpy
    # array stays positionally aligned regardless of how the caller
    # sliced ``structured`` (chunked builds pass an ``iloc``-slice with
    # an offset RangeIndex). The caller's own outer index is preserved
    # via the original_index variable returned to it.
    original_index = structured.index
    df = _resolve_en(structured.reset_index(drop=True))

    professional = _render_professional(df, lang, picks_chunk)
    sports = _render_sports(df, lang, picks_chunk)
    arts = _render_arts(df, lang, picks_chunk)
    travel = _render_travel(df, lang, picks_chunk)
    culinary = _render_culinary(df, lang, picks_chunk)
    culture = _render_culture(df, lang, picks_chunk)
    skills_prose, skills_list = _render_skills(df, lang, picks_chunk)
    hobbies_prose, hobbies_list = _render_hobbies(df, lang, picks_chunk)
    goals = _render_goals(df, lang, picks_chunk)
    summary = _render_summary(df, lang)

    out = pd.DataFrame({
        "professional_persona":         professional,
        "sports_persona":               sports,
        "arts_persona":                 arts,
        "travel_persona":               travel,
        "culinary_persona":             culinary,
        "persona":                      summary,
        "cultural_background":          culture,
        "skills_and_expertise":         skills_prose,
        "skills_and_expertise_list":    skills_list,
        "hobbies_and_interests":        hobbies_prose,
        "hobbies_and_interests_list":   hobbies_list,
        "career_goals_and_ambitions":   goals,
    })
    # Restore caller's original index so the builder can `concat`
    # narrative columns alongside the structured chunk without index
    # mismatches.
    out.index = original_index
    return out
