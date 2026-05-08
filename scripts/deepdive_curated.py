"""Deep-dive analysis of the curated NSO PX-Web data.

Produces every figure referenced in ``DATAANALYSIS.md``:

    docs/figures/analysis/01_population_timeseries.png
    docs/figures/analysis/02_urban_share_evolution.png
    docs/figures/analysis/03_regional_population_growth.png
    docs/figures/analysis/04_age_pyramid_shift.png
    docs/figures/analysis/05_education_attainment_trend.png
    docs/figures/analysis/06_occupation_structure_shift.png
    docs/figures/analysis/07_unemployment_region_urban.png
    docs/figures/analysis/08_unemployment_by_education.png
    docs/figures/analysis/09_industry_employment_share.png
    docs/figures/analysis/10_data_coverage_matrix.png

Each figure is independent — running the script with no args produces
all of them. Outputs land under ``docs/figures/analysis/``.

Usage:
    python -m scripts.deepdive_curated
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from packages.common.paths import REPO_ROOT, ensure_dir
from scripts._paper_style import paper_axes, save, use_paper_style

PXWEB_DIR = REPO_ROOT / "data" / "nso-gov-vn" / "raw" / "pxweb" / "vi"
OUT_DIR = ensure_dir(REPO_ROOT / "docs" / "figures" / "analysis")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(table_id: str) -> pd.DataFrame:
    """Load a Population & Labour PX-Web parquet by id (e.g. 'V02.43')."""
    db = "Dan-so-va-lao-dong" if table_id.startswith("V02") else (
        "Giao-duc" if table_id.startswith("V13") else
        "Doanh-nghiep" if table_id.startswith("V05") else
        "Cong-nghiep" if table_id.startswith("V07") else
        "dau-tu"
    )
    return pd.read_parquet(PXWEB_DIR / f"{db}__{table_id}.px.parquet")


def _year_int(value: object) -> int | None:
    """Pull the integer year out of NSO's labels ('2024', 'Sơ bộ 2024')."""
    m = re.search(r"\d{4}", str(value))
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------------------
# Figure 01 — population time-series, V02.02
# ---------------------------------------------------------------------------
def fig_population_timeseries() -> None:
    df = _load("V02.02")
    df = df[df["Cách tính"] == "Tổng số (Nghìn người)"].copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df.dropna(subset=["year"])
    pivot = df.pivot_table(index="year", columns="Phân tổ", values="value", aggfunc="first")
    out = OUT_DIR / "01_population_timeseries.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(pivot.index, pivot["Tổng số"] / 1000, color="black",
            linewidth=2.0, label="Total (millions)")
    ax.plot(pivot.index, pivot["Thành thị"] / 1000, color="black",
            linewidth=1.4, linestyle="--", label="Urban (millions)")
    ax.plot(pivot.index, pivot["Nông thôn"] / 1000, color="black",
            linewidth=1.4, linestyle=":", label="Rural (millions)")
    ax.set_title("Vietnam population 1990–2024 (NSO V02.02)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population (millions)")
    ax.legend()
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def fig_urban_share_evolution() -> None:
    df = _load("V02.02")
    df = df[df["Cách tính"] == "Tổng số (Nghìn người)"].copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df.dropna(subset=["year"])
    pivot = df.pivot_table(index="year", columns="Phân tổ", values="value", aggfunc="first")
    pivot["urban_pct"] = pivot["Thành thị"] / pivot["Tổng số"] * 100
    out = OUT_DIR / "02_urban_share_evolution.png"
    with paper_axes(figsize=(10, 5.0),
                     title="Urban share of population, 1990–2024 (NSO V02.02)",
                     xlabel="Year",
                     ylabel="% urban") as ax:
        ax.fill_between(pivot.index, 0, pivot["urban_pct"],
                         color="black", alpha=0.18)
        ax.plot(pivot.index, pivot["urban_pct"], color="black", linewidth=2)
        # Annotate endpoints.
        for x, kind in ((pivot.index[0], "start"), (pivot.index[-1], "end")):
            y = pivot.loc[x, "urban_pct"]
            ax.annotate(f"{x}: {y:.1f}%", xy=(x, y),
                         xytext=(0, 10 if kind == "end" else -16),
                         textcoords="offset points", ha="center", fontsize=10)
        ax.set_ylim(0, 50)
    save(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 03 — regional population growth, V02.20
# ---------------------------------------------------------------------------
_MACRO_REGIONS = (
    "Đồng bằng sông Hồng",
    "Trung du và miền núi phía Bắc",
    "Bắc Trung Bộ và Duyên hải miền Trung",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
)


def fig_regional_population_growth() -> None:
    df = _load("V02.20").copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df.dropna(subset=["year"])
    df = df[df["Tỉnh/Thành phố"].isin(_MACRO_REGIONS)]
    pivot = df.pivot_table(index="year", columns="Tỉnh/Thành phố",
                            values="value", aggfunc="first")
    out = OUT_DIR / "03_regional_population_growth.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1))]
    for region, style in zip(pivot.columns, styles):
        ax.plot(pivot.index, pivot[region], color="black", linewidth=1.6,
                linestyle=style, label=region)
    ax.set_title("Annual population growth rate by macro-region (NSO V02.20)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Growth rate (%)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 04 — age pyramid shift V02.41 (oldest year vs latest year)
# ---------------------------------------------------------------------------
def fig_age_pyramid_shift() -> None:
    df = _load("V02.41").copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df[~df["Nhóm tuổi"].str.upper().eq("TỔNG SỐ")]
    df = df.dropna(subset=["year"])
    age_order = ["15-19", "20-24", "25-29", "30-34", "35-39",
                 "40-44", "45-49", "50-54", "55-59", "60-64", "65+"]
    df = df[df["Nhóm tuổi"].isin(age_order)]
    earliest, latest = df["year"].min(), df["year"].max()

    early = df[df["year"] == earliest].set_index("Nhóm tuổi")["value"].reindex(age_order)
    late = df[df["year"] == latest].set_index("Nhóm tuổi")["value"].reindex(age_order)

    out = OUT_DIR / "04_age_pyramid_shift.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(age_order))
    ax.bar(x - 0.2, early.values, width=0.4, color="white",
            edgecolor="black", linewidth=0.8, label=f"{earliest}")
    ax.bar(x + 0.2, late.values, width=0.4, color="black",
            edgecolor="black", linewidth=0.4, label=f"{latest}")
    ax.set_xticks(x)
    ax.set_xticklabels(age_order, rotation=20)
    ax.set_title("Working-age (15+) employed cohort, "
                  f"{earliest} vs {latest} (NSO V02.41)")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Employed (thousand persons)")
    ax.legend()
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 05 — education attainment trend, V02.54
# ---------------------------------------------------------------------------
def fig_education_attainment_trend() -> None:
    df = _load("V02.54").copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df[~df["Chuyên môn kỹ thuật"].str.upper().eq("TỔNG SỐ")]
    df = df.dropna(subset=["year"])
    pivot = df.pivot_table(index="year", columns="Chuyên môn kỹ thuật",
                            values="value", aggfunc="first")
    pivot = pivot.sort_index()
    out = OUT_DIR / "05_education_attainment_trend.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    palette = {"Sơ cấp": "white", "Trung cấp": "lightgrey",
                "Cao đẳng": "dimgrey", "Đại học trở lên": "black"}
    bottom = np.zeros(len(pivot))
    for col in ("Sơ cấp", "Trung cấp", "Cao đẳng", "Đại học trở lên"):
        if col in pivot.columns:
            vals = pivot[col].fillna(0).to_numpy()
            ax.bar(pivot.index, vals, bottom=bottom,
                    color=palette[col], edgecolor="black",
                    linewidth=0.4, label=col)
            bottom = bottom + vals
    ax.set_title("Trained-labour share by qualification level "
                  "(NSO V02.54, %)")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of labour force")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 06 — occupation structure shift, V02.43
# ---------------------------------------------------------------------------
def fig_occupation_structure_shift() -> None:
    df = _load("V02.43").copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df[~df["Nghề nghiệp"].str.upper().eq("TỔNG SỐ")]
    df = df.dropna(subset=["year"])
    earliest, latest = df["year"].min(), df["year"].max()

    early = df[df["year"] == earliest].set_index("Nghề nghiệp")["value"]
    late = df[df["year"] == latest].set_index("Nghề nghiệp")["value"]

    # The taxonomy evolves over time (e.g. armed forces / military bucket
    # added later). Use the union of categories so both bars line up.
    all_occs = sorted(set(early.index) | set(late.index))
    early = (early.reindex(all_occs).fillna(0) / early.sum() * 100)
    late = (late.reindex(all_occs).fillna(0) / late.sum() * 100)

    out = OUT_DIR / "06_occupation_structure_shift.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6.0))
    y = np.arange(len(all_occs))
    ax.barh(y - 0.2, early.values, height=0.4, color="white",
            edgecolor="black", linewidth=0.8, label=f"{earliest} (%)")
    ax.barh(y + 0.2, late.values, height=0.4, color="black",
            edgecolor="black", linewidth=0.4, label=f"{latest} (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(all_occs, fontsize=9)
    ax.set_title(f"Occupation structure shift {earliest} - {latest} "
                  "(NSO V02.43, % of employed)")
    ax.set_xlabel("Share of employed (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 07 — unemployment region × urban/rural, V02.59
# ---------------------------------------------------------------------------
def fig_unemployment_region_urban() -> None:
    df = _load("V02.59").copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df.dropna(subset=["year"])
    df = df[df["Vùng"].isin(_MACRO_REGIONS)]
    df = df[df["Thành thị, nông thôn"].isin(("Thành thị", "Nông thôn"))]
    latest = df["year"].max()
    df = df[df["year"] == latest]
    pivot = df.pivot(index="Vùng", columns="Thành thị, nông thôn", values="value")
    pivot = pivot.reindex(_MACRO_REGIONS)
    out = OUT_DIR / "07_unemployment_region_urban.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(pivot.index))
    ax.bar(x - 0.2, pivot["Thành thị"].values, width=0.4,
            color="black", edgecolor="black", linewidth=0.4,
            label="Thành thị (urban)")
    ax.bar(x + 0.2, pivot["Nông thôn"].values, width=0.4,
            color="white", edgecolor="black", linewidth=0.8,
            label="Nông thôn (rural)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right", fontsize=9)
    ax.set_title(f"Unemployment rate by macro-region × urban/rural, "
                  f"{latest} (NSO V02.59, %)")
    ax.set_ylabel("Unemployment rate (%)")
    ax.legend()
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 08 — unemployment by qualification, V02.62
# ---------------------------------------------------------------------------
def fig_unemployment_by_education() -> None:
    df = _load("V02.62").copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df.dropna(subset=["year"])
    df = df[~df["Trình độ chuyên môn kỹ thuật"].str.upper().eq("TỔNG SỐ")]
    pivot = df.pivot_table(index="year",
                            columns="Trình độ chuyên môn kỹ thuật",
                            values="value", aggfunc="first")
    pivot = pivot.sort_index()
    out = OUT_DIR / "08_unemployment_by_education.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1))]
    for col, style in zip(pivot.columns, styles):
        ax.plot(pivot.index, pivot[col], color="black", linewidth=1.6,
                 linestyle=style, label=col)
    ax.set_title("Unemployment rate by qualification level (NSO V02.62, %)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment rate (%)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 09 — industry share of employment, V02.42
# ---------------------------------------------------------------------------
def fig_industry_employment_share() -> None:
    df = _load("V02.42").copy()
    df["year"] = df["Năm"].map(_year_int)
    df = df[df["Phân tổ"] == "Cơ cấu (%)"]
    df = df[~df["Ngành"].str.upper().eq("TỔNG SỐ")]
    df = df.dropna(subset=["year"])
    latest = df["year"].max()
    df = df[df["year"] == latest].sort_values("value", ascending=True)
    out = OUT_DIR / "09_industry_employment_share.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(11, 7.0))
    ax.barh(df["Ngành"], df["value"], color="black",
             edgecolor="black", linewidth=0.4)
    for i, (_label, v) in enumerate(zip(df["Ngành"], df["value"])):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_title(f"Employment share by industry sector, {latest} "
                  "(NSO V02.42, %)")
    ax.set_xlabel("Share of employed (%)")
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 10 — data coverage matrix (which dimensions appear in which tables)
# ---------------------------------------------------------------------------
def fig_data_coverage_matrix() -> None:
    """Build a (table × dimension) availability matrix for the 71 V02.* tables."""
    import json

    rows = []
    for parquet in sorted(PXWEB_DIR.glob("Dan-so-va-lao-dong__V02.*.px.parquet")):
        meta = parquet.with_name(parquet.name.replace(".parquet", ".metadata.json"))
        if not meta.exists():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        rows.append({
            "table_id": m.get("table_id"),
            "variables": [v.get("code", "") for v in m.get("variables") or []],
        })
    if not rows:
        print("  skip 10_data_coverage_matrix: no V02.* metadata found")
        return
    df = pd.DataFrame(rows)
    target_dims = [
        "Năm", "Tỉnh, thành phố", "Tỉnh/Thành phố", "Địa phương", "Vùng",
        "Phân tổ", "Cách tính", "Nhóm tuổi", "Giới tính",
        "Thành thị, nông thôn", "Nghề nghiệp",
        "Vị thế việc làm", "Chuyên môn kỹ thuật",
        "Trình độ chuyên môn kỹ thuật", "Ngành",
    ]
    matrix = pd.DataFrame(0, index=df["table_id"], columns=target_dims, dtype=int)
    for _, row in df.iterrows():
        for dim in row["variables"]:
            if dim in matrix.columns:
                matrix.loc[row["table_id"], dim] = 1
    # Sort by row-sum descending so the matrix reads cleanly.
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    out = OUT_DIR / "10_data_coverage_matrix.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(11, 16))
    ax.imshow(matrix.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right", fontsize=9)
    ax.set_title("Dimension coverage across V02.* tables\n(black = dimension present)")
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
def main() -> int:
    if not PXWEB_DIR.exists():
        print(f"PX-Web dir missing: {PXWEB_DIR}\n"
               "Run `personas-vn curate --only download` first.")
        return 2
    fig_population_timeseries()
    fig_urban_share_evolution()
    fig_regional_population_growth()
    fig_age_pyramid_shift()
    fig_education_attainment_trend()
    fig_occupation_structure_shift()
    fig_unemployment_region_urban()
    fig_unemployment_by_education()
    fig_industry_employment_share()
    fig_data_coverage_matrix()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
