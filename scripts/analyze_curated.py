"""Analyse the curated NSO PX-Web dataset.

Produces every figure referenced in ``DATAPROCESSING.md``:

    docs/figures/curated/01_tables_per_database.png
    docs/figures/curated/02_cells_per_table.png
    docs/figures/curated/03_variable_frequency.png
    docs/figures/curated/05_geographic_coverage.png
    docs/figures/curated/06_pxweb_summary.csv

Run after ``personas-vn curate --only download`` has populated
``data/nso-gov-vn/raw/pxweb/vi/``.

Usage:
    python -m scripts.analyze_curated
"""

from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from packages.common.paths import REPO_ROOT, ensure_dir
from scripts._paper_style import paper_axes, save

PXWEB_DIR = REPO_ROOT / "data" / "nso-gov-vn" / "raw" / "pxweb" / "vi"
OUT_DIR = ensure_dir(REPO_ROOT / "docs" / "figures" / "curated")


def load_catalog() -> pd.DataFrame:
    rows = []
    for meta_path in sorted(PXWEB_DIR.glob("*.metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append({
            "table_id":    meta.get("table_id"),
            "database":    meta.get("database"),
            "title":       meta.get("title"),
            "n_cells":     int(meta.get("n_kept") or meta.get("n_cells") or 0),
            "n_variables": len(meta.get("variables") or []),
            "variables":   [v.get("code", "") for v in meta.get("variables") or []],
            "parquet":     meta_path.with_name(
                meta_path.name.replace(".metadata.json", ".parquet")
            ),
        })
    return pd.DataFrame(rows)


def figure_tables_per_database(catalog: pd.DataFrame) -> None:
    agg = catalog.groupby("database").size().reset_index(name="n_tables")
    agg = agg.sort_values("n_tables", ascending=False)
    out = OUT_DIR / "01_tables_per_database.png"
    with paper_axes(figsize=(9, 5),
                     title="PX-Web tables crawled per NSO database",
                     xlabel="NSO database", ylabel="Tables") as ax:
        ax.bar(agg["database"], agg["n_tables"], color="black",
               edgecolor="black", linewidth=0.5)
        for x, y in zip(agg["database"], agg["n_tables"]):
            ax.text(x, y + 0.4, str(int(y)), ha="center", va="bottom",
                    fontsize=10)
        plt.xticks(rotation=15, ha="right")
    save(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_cells_per_table(catalog: pd.DataFrame) -> None:
    out = OUT_DIR / "02_cells_per_table.png"
    with paper_axes(figsize=(9, 4.6),
                     title="Distribution of data cells per PX-Web table",
                     xlabel="Data cells (long-format rows, log-scale)",
                     ylabel="Tables") as ax:
        cells = catalog["n_cells"].clip(lower=1)
        bins = np.logspace(np.log10(cells.min()), np.log10(cells.max()), 30)
        ax.hist(cells, bins=bins, color="black", edgecolor="black",
                linewidth=0.4)
        ax.set_xscale("log")
        ax.axvline(cells.median(), color="0.4", linestyle="--", linewidth=1,
                    label=f"median = {int(cells.median())}")
        ax.legend()
    save(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_variable_frequency(catalog: pd.DataFrame) -> None:
    counter: Counter[str] = Counter()
    for variables in catalog["variables"]:
        counter.update(variables)
    top = counter.most_common(20)
    df = pd.DataFrame(top, columns=["variable", "tables"])
    out = OUT_DIR / "03_variable_frequency.png"
    with paper_axes(figsize=(10, 5.4),
                     title="Top-20 PX-Web variable codes (across all 187 tables)",
                     xlabel="Variable code (Vietnamese)",
                     ylabel="Tables containing it") as ax:
        ax.bar(df["variable"], df["tables"], color="black",
               edgecolor="black", linewidth=0.4)
        plt.xticks(rotation=30, ha="right")
    save(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def figure_geographic_coverage(catalog: pd.DataFrame) -> None:
    province_counter: Counter[str] = Counter()
    region_cols = ("Tỉnh, thành phố", "Tỉnh/Thành phố", "Địa phương", "Vùng")
    for parquet in catalog["parquet"]:
        try:
            df = pd.read_parquet(parquet)
        except Exception:
            continue
        for col in region_cols:
            if col in df.columns:
                province_counter.update(df[col].dropna().unique())
                break
    drop = {"TỔNG SỐ", "Tổng số", "CẢ NƯỚC", "Cả nước",
            "Tổng số (Nghìn người)"}
    for d in drop:
        province_counter.pop(d, None)
    top = pd.DataFrame(
        [{"location": k, "tables": v}
         for k, v in province_counter.most_common(25)]
    )
    out = OUT_DIR / "05_geographic_coverage.png"
    with paper_axes(figsize=(10, 5.4),
                     title="Top-25 administrative units by table coverage",
                     xlabel="Province / region",
                     ylabel="Tables containing it") as ax:
        ax.bar(top["location"], top["tables"], color="black",
               edgecolor="black", linewidth=0.4)
        plt.xticks(rotation=45, ha="right")
    save(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def write_summary_csv(catalog: pd.DataFrame) -> None:
    summary = catalog[["table_id", "database", "title", "n_variables",
                        "n_cells"]].copy()
    summary["variables"] = catalog["variables"].map(lambda xs: ", ".join(xs))
    out = OUT_DIR / "06_pxweb_summary.csv"
    summary.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> int:
    if not PXWEB_DIR.exists():
        print(f"PX-Web directory not found: {PXWEB_DIR}\n"
              "Run `personas-vn curate --only download` first.")
        return 2
    catalog = load_catalog()
    print(f"loaded {len(catalog)} tables across "
          f"{catalog['database'].nunique()} databases")
    print(f"total cells: {catalog['n_cells'].sum():,}")

    figure_tables_per_database(catalog)
    figure_cells_per_table(catalog)
    figure_variable_frequency(catalog)
    figure_geographic_coverage(catalog)
    write_summary_csv(catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
