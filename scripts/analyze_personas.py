"""Analyse the synthetic persona dataset.

Produces every figure referenced in ``DATASYNTHESIS.md``:

    docs/figures/personas/01_marginal_region.png
    docs/figures/personas/02_marginal_education.png
    docs/figures/personas/03_marginal_occupation.png
    docs/figures/personas/04_age_pyramid.png
    docs/figures/personas/05_region_x_education.png
    docs/figures/personas/06_age_x_employment.png
    docs/figures/personas/07_embedding_scatter.png
    docs/figures/personas/08_marginal_vs_pxweb.csv
    docs/figures/personas/09_geographic_choropleth.html  (interactive)

Run after ``personas-vn generate --n 100000`` (and optionally
``personas-vn embed-personas``).

Usage:
    python -m scripts.analyze_personas
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from packages.common.paths import REPO_ROOT, ensure_dir
from scripts._paper_style import paper_axes, save, use_paper_style

PERSONAS_PATH = REPO_ROOT / "data" / "personas" / "personas.json"
ENRICHED_PATH = REPO_ROOT / "data" / "personas" / "personas_enriched.json"
REDUCED_PATH = REPO_ROOT / "data" / "personas" / "reduced.parquet"
OUT_DIR = ensure_dir(REPO_ROOT / "docs" / "figures" / "personas")


def load_personas() -> pd.DataFrame:
    src = ENRICHED_PATH if ENRICHED_PATH.exists() else PERSONAS_PATH
    if not src.exists():
        raise SystemExit(f"personas not found: {src}\n"
                         "Run `personas-vn generate --n 100000` first.")
    print(f"loading {src.relative_to(REPO_ROOT)}...")
    payload = json.loads(src.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["personas"])
    print(f"  n = {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_marginal_region(df: pd.DataFrame) -> None:
    agg = df.groupby("region").size().reset_index(name="personas")
    agg = agg.sort_values("personas", ascending=False)
    out = OUT_DIR / "01_marginal_region.png"
    with paper_axes(figsize=(10, 5.2),
                     title="Personas per macro-region (NSO labels)",
                     xlabel="Region (Vietnamese)",
                     ylabel="Personas") as ax:
        ax.bar(agg["region"], agg["personas"], color="black",
               edgecolor="black", linewidth=0.4)
        for x, y in zip(agg["region"], agg["personas"]):
            ax.text(x, y + 200, f"{int(y):,}", ha="center", va="bottom",
                    fontsize=10)
        plt.xticks(rotation=15, ha="right")
    save(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_marginal_education(df: pd.DataFrame) -> None:
    agg = df.groupby("education_level").size().reset_index(name="personas")
    out = OUT_DIR / "02_marginal_education.png"
    with paper_axes(figsize=(9, 4.8),
                     title="Personas by education level (PX-Web V02.54)",
                     xlabel="Trình độ chuyên môn kỹ thuật",
                     ylabel="Personas") as ax:
        ax.bar(agg["education_level"], agg["personas"], color="black",
               edgecolor="black", linewidth=0.4)
        for x, y in zip(agg["education_level"], agg["personas"]):
            ax.text(x, y + 200, f"{int(y):,}", ha="center", va="bottom",
                    fontsize=10)
        plt.xticks(rotation=15, ha="right")
    save(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_marginal_occupation(df: pd.DataFrame) -> None:
    workers = df[df["occupation"] != "Không áp dụng"].copy()
    agg = workers.groupby("occupation_en").size().reset_index(name="personas")
    agg = agg.sort_values("personas", ascending=False)
    out = OUT_DIR / "03_marginal_occupation.png"
    with paper_axes(figsize=(10, 5.6),
                     title="Workers by occupation (ISCO collapsed, PX-Web V02.43)",
                     xlabel="Occupation (English mirror)",
                     ylabel="Personas") as ax:
        ax.bar(agg["occupation_en"], agg["personas"], color="black",
               edgecolor="black", linewidth=0.4)
        plt.xticks(rotation=30, ha="right")
    save(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_age_pyramid(df: pd.DataFrame) -> None:
    age_order = ["15-19", "20-24", "25-29", "30-34", "35-39",
                 "40-44", "45-49", "50-54", "55-59", "60-64", "65+"]
    male = df[df["sex"] == "Nam"].groupby("age_group").size().reindex(
        age_order, fill_value=0
    )
    female = df[df["sex"] == "Nữ"].groupby("age_group").size().reindex(
        age_order, fill_value=0
    )
    out = OUT_DIR / "04_age_pyramid.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(age_order))
    ax.barh(y, -male.values, color="black", edgecolor="black",
            linewidth=0.4, label="Nam (male)")
    ax.barh(y, female.values, color="white", edgecolor="black",
            linewidth=0.6, label="Nữ (female)")
    ax.set_yticks(y)
    ax.set_yticklabels(age_order)
    ax.set_xlabel("Personas (mirrored)")
    ax.set_ylabel("Age group")
    ax.set_title("Persona age pyramid (5-year bands)")
    # Make x-tick labels positive on both sides.
    xticks = ax.get_xticks()
    ax.set_xticklabels([f"{int(abs(x)):,}" for x in xticks])
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_region_x_education(df: pd.DataFrame) -> None:
    pivot = (
        df.groupby(["region", "education_level"]).size().reset_index(name="n")
        .pivot(index="region", columns="education_level", values="n")
        .fillna(0)
    )
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    out = OUT_DIR / "05_region_x_education.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Greys")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=9, color=("white" if v > 35 else "black"))
    ax.set_title("Education breakdown by region (% within region)")
    fig.colorbar(im, ax=ax, label="% within region")
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_age_x_employment(df: pd.DataFrame) -> None:
    age_order = ["15-19", "20-24", "25-29", "30-34", "35-39",
                 "40-44", "45-49", "50-54", "55-59", "60-64", "65+"]
    pivot = (
        df.groupby(["age_group", "employment_status"]).size().reset_index(name="n")
        .pivot(index="age_group", columns="employment_status", values="n")
        .reindex(age_order)
        .fillna(0)
    )
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    out = OUT_DIR / "06_age_x_employment.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Greys")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=9, color=("white" if v > 35 else "black"))
    ax.set_title("Employment status by age group (% within age group)")
    fig.colorbar(im, ax=ax, label="% within age group")
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_embedding_scatter() -> None:
    if not REDUCED_PATH.exists():
        print(f"  skip embedding scatter: {REDUCED_PATH.name} not present "
              "(run `personas-vn embed-personas` first)")
        return
    df = pd.read_parquet(REDUCED_PATH)
    out = OUT_DIR / "07_embedding_scatter.png"
    use_paper_style()
    fig, ax = plt.subplots(figsize=(11, 7.2))
    occs = sorted(df["occupation_en"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    for i, occ in enumerate(occs):
        sub = df[df["occupation_en"] == occ]
        ax.scatter(sub["x"], sub["y"], s=8, alpha=0.7,
                    edgecolors="black", linewidths=0.2,
                    color=cmap(i % cmap.N), label=occ)
    ax.set_title(f"Persona UMAP — {len(df)} embedded bios, "
                  "coloured by occupation")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=9, frameon=True)
    fig.tight_layout()
    save(out, fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def write_marginal_csv(df: pd.DataFrame) -> None:
    rows = []
    for axis in ("region", "education_level", "sex", "urbanicity",
                  "occupation_en"):
        if axis not in df.columns:
            continue
        pct = df[axis].value_counts(normalize=True) * 100
        for k, v in pct.items():
            rows.append({"axis": axis, "category": k,
                         "persona_pct": round(float(v), 2)})
    out = OUT_DIR / "08_marginal_vs_pxweb.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def write_geo_choropleth(df: pd.DataFrame) -> None:
    """Region-level distribution as an interactive Plotly bar map.

    Plotly is used here only for the *interactive* HTML (no Kaleido-
    based PNG rendering). Open the resulting file in a browser.
    """
    try:
        import plotly.express as px
    except ImportError:
        print("  skip geographic choropleth: plotly not installed")
        return
    agg = df.groupby(["region", "region_en"]).size().reset_index(name="personas")
    agg = agg.sort_values("personas", ascending=False)
    fig = px.bar(
        agg, x="personas", y="region_en", orientation="h",
        title="Persona geographic distribution (6 NSO macro-regions)",
        labels={"region_en": "Macro-region", "personas": "Personas"},
        text="personas",
    )
    fig.update_traces(marker_color="black", marker_line_color="black",
                       textposition="outside")
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Computer Modern, CMU Serif, serif", color="black"),
    )
    out = OUT_DIR / "09_geographic_choropleth.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  wrote {out.relative_to(REPO_ROOT)} (open in a browser)")


def main() -> int:
    df = load_personas()
    figure_marginal_region(df)
    figure_marginal_education(df)
    figure_marginal_occupation(df)
    figure_age_pyramid(df)
    figure_region_x_education(df)
    figure_age_x_employment(df)
    figure_embedding_scatter()
    write_marginal_csv(df)
    write_geo_choropleth(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
