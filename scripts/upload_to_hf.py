"""End-to-end pipeline that publishes ``data/nso-gov-vn/`` (and optionally
the generated persona dataset) as HuggingFace Datasets.

Modelled 1-to-1 on tmquan/ViLA's
`data/anle.toaan.gov.vn/_to_hf.py`
( https://github.com/tmquan/ViLA/tree/main/data/anle.toaan.gov.vn ),
which is the NeMo-Curator best-practice template for a curated, multi-stage
dataset on the Hub.

What ViLA does (and what we mirror)
-----------------------------------
ViLA's pipeline has three steps:

    1. **consolidate** the per-document shards under ``jsonl/`` /
       ``parquet/embeddings/`` / ``parquet/reduced/`` into a small, typed
       parquet bundle under ``_hf/data/`` — one parquet per pipeline
       stage (``parse``, ``extract``, ``embed``, ``reduce``).
       DuckDB is used because PyArrow occasionally fails on the per-doc
       shards with ``Repetition level histogram size mismatch``.
    2. **render assets** — call ``_render_assets.py`` to refresh the
       static PNGs embedded in the dataset card and the ``_stats.json``
       snapshot.
    3. **upload** every artefact to the Hub at the right path. They use
       ``HfApi.upload_folder`` directly with explicit ``path_in_repo``
       (and atomic ``delete_patterns`` for stale files), not
       ``hf upload-large-folder`` (which has no path control).

What we adapt
-------------
* The "documents" in our pipeline are **PX-Web matrices** (502 of them)
  rather than legal documents. Each matrix has a native schema (its own
  variable codes, its own categorical bucket lists), so we cannot
  fold them all into a single union schema without losing fidelity.
  Instead we publish:

    - a 502-row **catalog** parquet (one row per matrix with metadata)
    - a per-table parquet bundle under ``tables/`` (the matrices'
      native long-format data, one parquet per matrix id)
    - per-database glob configs (load every V02 matrix at once,
      every V14 matrix at once, ...)

  The user's question "dataset split by table is better?" is answered
  yes: per-table preserves each matrix's native schema, and HuggingFace
  configs can still group them by database via glob ``data_files``.

* Personas + persona embeddings get an optional second push when
  ``--push personas`` is set. They go into a *separate* HF repo (the
  persona schema is a Pydantic model, not a tabular matrix, so mixing
  it with the PX-Web bundle would be confusing).

Hub layout — `personas-vn-pxweb`
--------------------------------
::

    .
    ├── README.md                      # dataset card (auto-generated)
    ├── _stats.json                    # numbers/tables the README quotes
    ├── data/
    │   ├── catalog.parquet                # 502-row index
    │   └── cells.parquet                  # 316K-row long-format union
    ├── tables/
    │   ├── V01.01.parquet                 # one parquet per matrix
    │   ├── V01.02.parquet
    │   └── ... (502 files)
    ├── assets/                            # the 38 NVIDIA-styled PNGs
    │                                      # straight from
    │                                      # docs/figures/analysis/
    ├── notebook.ipynb                     # DATAANALYSIS.ipynb
    └── raw/
        └── pxweb/vi/                      # original parquets + metadata

HuggingFace configs declared in the README YAML frontmatter::

    load_dataset("you/personas-vn-pxweb", "catalog")            # 502 rows
    load_dataset("you/personas-vn-pxweb", "cells")              # 316K rows
    load_dataset("you/personas-vn-pxweb", "V02_population_labour")  # 63 tables
    load_dataset("you/personas-vn-pxweb", "V14_health_society")     # 97 tables

CLI
---
::

    # Full pipeline (consolidate + assets + upload)
    python -m scripts.upload_to_hf

    # Only build the bundle locally; don't push
    python -m scripts.upload_to_hf --no-upload

    # Skip consolidation if data/nso-gov-vn/_hf/data/ is already built
    python -m scripts.upload_to_hf --skip-consolidate

    # Push the personas dataset to a different repo
    python -m scripts.upload_to_hf --push personas \
        --repo you/personas-vn-personas
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from packages.common.logging import get_logger
from packages.common.paths import REPO_ROOT, ensure_dir

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths — match ViLA's structure but rooted under data/nso-gov-vn/
# ---------------------------------------------------------------------------
DATASET_ROOT = REPO_ROOT / "data" / "nso-gov-vn"
PXWEB_VI_DIR = DATASET_ROOT / "raw" / "pxweb" / "vi"
PERSONAS_DIR = REPO_ROOT / "data" / "personas"
ANALYSIS_DIR = REPO_ROOT / "docs" / "figures" / "analysis"

# All four notebooks document the same nso-gov-vn dataset from
# different angles; we publish all four so the HF dataset card can
# point users at any one of them as a getting-started view.
NOTEBOOKS: dict[str, Path] = {
    "DATAANALYSIS.ipynb":      REPO_ROOT / "DATAANALYSIS.ipynb",
    "DATAVISUALIZATION.ipynb": REPO_ROOT / "DATAVISUALIZATION.ipynb",
    "DATAEXPLORATION.ipynb":   REPO_ROOT / "DATAEXPLORATION.ipynb",
    "DATASYNTHESIS.ipynb":     REPO_ROOT / "DATASYNTHESIS.ipynb",
}

# Companion markdown narrative docs — they explain the curator pipeline
# (DATAPROCESSING), the 39 analysis figures (DATAANALYSIS), the 11
# choropleth maps (DATAVISUALIZATION), the curator-embedding UMAP
# (DATAEXPLORATION), the persona-generator pipeline (DATASYNTHESIS), the
# bilingual ontology tree (ONTOLOGY), and the persona-dataset schema
# (DATASETS).
NARRATIVE_DOCS: dict[str, Path] = {
    "DATAPROCESSING.md":   REPO_ROOT / "DATAPROCESSING.md",
    "DATAANALYSIS.md":     REPO_ROOT / "DATAANALYSIS.md",
    "DATAVISUALIZATION.md": REPO_ROOT / "DATAVISUALIZATION.md",
    "DATAEXPLORATION.md":  REPO_ROOT / "DATAEXPLORATION.md",
    "DATASYNTHESIS.md":    REPO_ROOT / "DATASYNTHESIS.md",
    "ONTOLOGY.md":         REPO_ROOT / "ONTOLOGY.md",
    "DATASETS.md":         REPO_ROOT / "DATASETS.md",
}

# Each figure section lives under ``docs/figures/<section>/`` with a
# matching PNG + HTML pair per figure (PNG = brand-coloured static
# image for markdown embeds; HTML = self-contained interactive Plotly
# with pan / zoom / hover, ideal for HF dataset-card 'open in
# browser' download links).
#   key = section slug used inside the bundle (and in the README)
#   val = source directory (we rename ``maps`` → ``visualization``
#         in the bundle to align with DATAVISUALIZATION.ipynb's name)
FIGURE_SECTIONS: dict[str, Path] = {
    "analysis":      REPO_ROOT / "docs" / "figures" / "analysis",
    "visualization": REPO_ROOT / "docs" / "figures" / "maps",
    "exploration":   REPO_ROOT / "docs" / "figures" / "exploration",
    "synthesis":     REPO_ROOT / "docs" / "figures" / "synthesis",
}

HF_DIR = DATASET_ROOT / "_hf"
HF_DATA_DIR = HF_DIR / "data"
HF_TABLES_DIR = HF_DIR / "tables"
HF_ASSETS_DIR = HF_DIR / "assets"
HF_NOTEBOOKS_DIR = HF_DIR / "notebooks"
HF_DOCS_DIR = HF_DIR / "docs"
HF_FIGURES_DIR = HF_DIR / "figures"

DEFAULT_REPO_PXWEB = "tmquan/personas-vn-pxweb"
DEFAULT_REPO_PERSONAS = "tmquan/personas-vn-personas"

# Database id -> human-friendly slug used in HF config names. Keep these
# stable — the YAML frontmatter in README.md references them directly.
DB_SLUGS: dict[str, str] = {
    "Đơn vị hành chính, đất đai và khí hậu": "V01_geography",
    "Dân số và lao động":                    "V02_population_labour",
    "Tài khoản quốc gia":                    "V03_national_accounts",
    "Đầu tư":                                "V04_investment",
    "Doanh nghiệp":                          "V05_enterprises",
    "Nông, lâm nghiệp và thủy sản":           "V06_agriculture",
    "Công nghiệp":                           "V07_industry",
    "Thương mại, giá cả":                    "V08_trade_prices_tourism",
    "Vận tải và bưu điện":                   "V12_transport_postal",
    "Giáo dục":                              "V13_education",
    "Y tế, văn hóa và đời sống":             "V14_health_society",
    "Thống kê nước ngoài":                   "V15_international",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_str(v: Any) -> str | None:
    """Normalise mixed string/NaN columns: NaN/None -> None, else str()."""
    if v is None:
        return None
    if isinstance(v, float):
        try:
            if math.isnan(v):
                return None
        except Exception:
            pass
        return str(v)
    return v if isinstance(v, str) else str(v)


def _slug_id(table_id: str) -> str:
    """Turn 'V02.03-07' into 'V02_03-07' so it works as a filename stem."""
    return table_id.replace(".", "_")


def _ascii_slug(name: str) -> str:
    """ViLA-style ASCII slug for filenames (NFKD + strip combining)."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_combine = "".join(c for c in nfkd if not unicodedata.combining(c))
    return (no_combine.replace("đ", "d").replace("Đ", "D")
                       .replace(",", "").replace(".", "")
                       .replace(" ", "-").lower())


def _read_metadata(meta_path: Path) -> dict[str, Any]:
    """Read a PX-Web *.metadata.json next to its parquet, with safe defaults."""
    j = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "table_id": _coerce_str(j.get("table_id")) or meta_path.stem,
        "database": _coerce_str(j.get("database")) or "Unknown",
        "title":    _coerce_str(j.get("title")) or "",
        "source":   _coerce_str(j.get("source")) or "json_api",
        "variables": [_coerce_str(v.get("code")) for v in j.get("variables") or []],
        "n_cells":   int(j.get("n_kept") or j.get("n_cells") or 0),
        "url":       _coerce_str(j.get("url")),
        "downloaded_at": _coerce_str(j.get("downloaded_at")),
    }


# ---------------------------------------------------------------------------
# Step 1 — consolidate
# ---------------------------------------------------------------------------
def build_catalog() -> pd.DataFrame:
    """One row per PX-Web matrix on disk. Mirrors ViLA's per-doc index.

    Schema:
        table_id      str           "V02.43"
        database      str           "Dân số và lao động"
        config_id     str           HF config slug ("V02_population_labour")
        title         str
        source        str           "json_api" | "html_form"
        variables     list[str]     ["Nghề nghiệp", "Năm"]
        n_variables   int
        n_cells       int
        parquet_path  str           relative to repo root
        metadata_path str
    """
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(PXWEB_VI_DIR.glob("*.metadata.json")):
        m = _read_metadata(meta_path)
        parquet_path = meta_path.with_name(
            meta_path.name.replace(".metadata.json", ".parquet")
        )
        if not parquet_path.exists():
            continue
        rows.append({
            **m,
            "config_id": DB_SLUGS.get(m["database"], "other"),
            "n_variables": len(m["variables"]),
            "parquet_path":  str(parquet_path.relative_to(REPO_ROOT)),
            "metadata_path": str(meta_path.relative_to(REPO_ROOT)),
        })
    df = pd.DataFrame(rows).sort_values("table_id").reset_index(drop=True)
    log.info("catalog: %d matrices across %d databases (%d cells)",
              len(df), df["database"].nunique(), df["n_cells"].sum())
    return df


def consolidate() -> dict[str, Any]:
    """Step 1 — write the parquet bundle under ``_hf/`` mirroring the
    Hub layout. Returns a small stats blob the README will quote.
    """
    log.info("[1/3] Consolidating per-matrix shards into _hf/ …")
    ensure_dir(HF_DATA_DIR)
    ensure_dir(HF_TABLES_DIR)
    ensure_dir(HF_ASSETS_DIR)

    # 1.a — catalog.parquet (502 rows, browsing-friendly index)
    catalog = build_catalog()
    catalog_pq = HF_DATA_DIR / "catalog.parquet"
    catalog.to_parquet(catalog_pq, compression="zstd", index=False)
    log.info("  wrote %s (%.1f KB)", catalog_pq.relative_to(REPO_ROOT),
              catalog_pq.stat().st_size / 1024)

    # 1.b — per-table parquets (one file per matrix id) — plus we union
    #       them into a long-format ``cells.parquet`` for users who want a
    #       single read.
    HF_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cells_chunks: list[pd.DataFrame] = []
    for _, row in catalog.iterrows():
        src_pq = REPO_ROOT / row["parquet_path"]
        dst_pq = HF_TABLES_DIR / f"{_slug_id(row['table_id'])}.parquet"
        df = pd.read_parquet(src_pq)
        df.to_parquet(dst_pq, compression="zstd", index=False)
        # Build the long-format cells row with dimensions captured in JSON.
        if "value" in df.columns:
            non_value_cols = [c for c in df.columns if c != "value"]
            tall = df.copy()
            tall["dimensions"] = tall[non_value_cols].apply(
                lambda r: json.dumps({k: _coerce_str(v) for k, v in r.items()},
                                       ensure_ascii=False),
                axis=1,
            )
            cells_chunks.append(pd.DataFrame({
                "table_id":   row["table_id"],
                "database":   row["database"],
                "config_id":  row["config_id"],
                "dimensions": tall["dimensions"],
                "value":      tall["value"],
            }))
    cells = pd.concat(cells_chunks, ignore_index=True) if cells_chunks else pd.DataFrame()
    cells_pq = HF_DATA_DIR / "cells.parquet"
    cells.to_parquet(cells_pq, compression="zstd", index=False)
    log.info("  wrote %s (%d rows, %.1f MB)", cells_pq.relative_to(REPO_ROOT),
              len(cells), cells_pq.stat().st_size / 1e6)

    # 1.b.2 — per-database splits of cells.parquet, one parquet per
    # config slug. Different matrices in the same NSO database have
    # heterogeneous native schemas (e.g. V02.01 dims by `Địa phương`
    # while V02.02 dims by `Phân tổ` + `Cách tính`), so a glob-based HF
    # config over `tables/V02_*.parquet` would fail when `datasets`
    # tries to concatenate them. The unified `(table_id, database,
    # config_id, dimensions, value)` schema in cells.parquet sidesteps
    # this — slice it by config_id and every per-database config gets a
    # consistent schema.
    if not cells.empty:
        for config_id, sub in cells.groupby("config_id"):
            out = HF_DATA_DIR / f"cells_{config_id}.parquet"
            sub.to_parquet(out, compression="zstd", index=False)
        log.info("  wrote %d per-database cell splits to data/cells_*.parquet",
                  cells["config_id"].nunique())

    log.info("  wrote %d per-table parquets to %s/",
              len(catalog), HF_TABLES_DIR.relative_to(REPO_ROOT))

    # 1.c — copy the four notebooks, six narrative docs, and the
    # PNG + HTML interactive figure pairs.
    copy_artefacts()

    # 1.d — _stats.json that the README quotes mechanically
    stats = build_stats(catalog, cells)
    (HF_DIR / "_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    log.info("  wrote %s", (HF_DIR / "_stats.json").relative_to(REPO_ROOT))

    # 1.e — README.md (dataset card with HF config frontmatter)
    write_readme(stats)
    log.info("  wrote %s", (HF_DIR / "README.md").relative_to(REPO_ROOT))

    return stats


def copy_artefacts() -> dict[str, int]:
    """Sync notebooks, narrative docs, and PNG + HTML figure pairs into
    the ``_hf/`` bundle. Returns a small {kind: count} blob the caller
    can log.

    Layout produced::

        _hf/
        ├── assets/                        # README-embedded PNGs (analysis only)
        ├── notebooks/{4 notebooks}.ipynb
        ├── docs/{7 narrative docs}.md
        └── figures/
            ├── analysis/      {39 PNGs + 39 HTMLs}
            ├── visualization/ {11 PNGs + 11 HTMLs}  (← from docs/figures/maps/)
            ├── exploration/   {11 PNGs + 11 HTMLs}
            └── synthesis/     { 5 PNGs +  5 HTMLs}

    Why two homes for analysis PNGs (``assets/`` AND ``figures/analysis/``):
    HF dataset cards embed images via relative path; we keep ``assets/``
    short + flat so the README's ``![](assets/01_*.png)`` links stay
    legible. ``figures/`` is the canonical content tree (one
    directory per notebook) and is what ``DATAANALYSIS.md`` etc.
    embed when the docs are read on GitHub.
    """
    counts = {"notebooks": 0, "docs": 0, "png": 0, "html": 0, "csv": 0}

    # 1) Notebooks — all four DATA*.ipynb files. The legacy bundle
    # also keeps a top-level ``notebook.ipynb`` symlink-equivalent
    # (a fresh copy of DATAANALYSIS) for back-compat with downstream
    # tools that hard-coded that path.
    HF_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in NOTEBOOKS.items():
        if src.exists():
            shutil.copy2(src, HF_NOTEBOOKS_DIR / name)
            counts["notebooks"] += 1
        else:
            log.warning("  notebook missing: %s", src.relative_to(REPO_ROOT))
    legacy = NOTEBOOKS.get("DATAANALYSIS.ipynb")
    if legacy and legacy.exists():
        shutil.copy2(legacy, HF_DIR / "notebook.ipynb")

    # 2) Narrative markdown docs — companion explanations for every
    # notebook + the DATASETS / ONTOLOGY references.
    HF_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in NARRATIVE_DOCS.items():
        if src.exists():
            shutil.copy2(src, HF_DOCS_DIR / name)
            counts["docs"] += 1
        else:
            log.warning("  doc missing: %s", src.relative_to(REPO_ROOT))

    # 3) Figures — preserve PNG + HTML pairs in section subfolders.
    for section, src_dir in FIGURE_SECTIONS.items():
        if not src_dir.exists():
            log.warning("  figure section missing: %s", src_dir.relative_to(REPO_ROOT))
            continue
        dst_dir = HF_FIGURES_DIR / section
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.iterdir()):
            if not src.is_file():
                continue
            ext = src.suffix.lower()
            if ext not in (".png", ".html", ".csv"):
                continue
            shutil.copy2(src, dst_dir / src.name)
            counts[ext.lstrip(".")] = counts.get(ext.lstrip("."), 0) + 1

    # 4) Flat ``assets/`` folder (README-embedded analysis PNGs only).
    HF_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(ANALYSIS_DIR.glob("*.png")):
        shutil.copy2(src, HF_ASSETS_DIR / src.name)
    csv_src = ANALYSIS_DIR / "16_full_table_inventory.csv"
    if csv_src.exists():
        shutil.copy2(csv_src, HF_ASSETS_DIR / csv_src.name)

    log.info("  artefacts: %d notebooks, %d docs, %d PNGs, %d HTMLs",
              counts["notebooks"], counts["docs"], counts["png"], counts["html"])
    return counts


# Back-compat alias — keep the old name working for any external
# script that imports it. ``copy_assets`` now does the full bundle.
copy_assets = copy_artefacts


def build_stats(catalog: pd.DataFrame, cells: pd.DataFrame) -> dict[str, Any]:
    """Numbers / tables the README quotes — mirrors ViLA's _stats.json.

    Kept small + flat so the dataset card can be regenerated mechanically.
    """
    by_db = (catalog.groupby("database")
                       .agg(n_tables=("table_id", "count"),
                             n_cells=("n_cells", "sum"))
                       .sort_values("n_cells", ascending=False)
                       .reset_index())
    by_source = catalog["source"].value_counts().to_dict()
    return {
        "total_tables":   len(catalog),
        "total_cells":    int(catalog["n_cells"].sum()),
        "total_databases": int(catalog["database"].nunique()),
        "by_source":      by_source,
        "by_database":    by_db.to_dict(orient="records"),
        "db_prefixes":    _db_prefixes(catalog),
        "n_variables_unique": int(catalog["variables"].explode().nunique()),
        "earliest_year": _earliest_year_in(catalog),
        "latest_year":   _latest_year_in(catalog),
    }


def _earliest_year_in(catalog: pd.DataFrame) -> int | None:
    return _year_extreme(catalog, kind="min")


def _latest_year_in(catalog: pd.DataFrame) -> int | None:
    return _year_extreme(catalog, kind="max")


def _year_extreme(catalog: pd.DataFrame, *, kind: str) -> int | None:
    """Scan every parquet for 4-digit years — cheap and resilient."""
    import re
    years: list[int] = []
    for _, row in catalog.iterrows():
        path = REPO_ROOT / row["parquet_path"]
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        for col in df.columns:
            if col == "value":
                continue
            for val in df[col].astype(str).head(2000):
                m = re.search(r"(19|20)\d{2}", val)
                if m:
                    years.append(int(m.group(0)))
    if not years:
        return None
    return min(years) if kind == "min" else max(years)


# ---------------------------------------------------------------------------
# README dataset card
# ---------------------------------------------------------------------------
def _db_prefixes(catalog: pd.DataFrame) -> dict[str, list[str]]:
    """database_name -> sorted list of V-prefixes appearing under it.

    Most NSO databases have a single matrix prefix (V02 = Population &
    Labour, V03 = National Accounts, ...), but a few span multiple
    (Thương mại, giá cả publishes under V08-V11). The README's HF config
    glob has to accommodate those.
    """
    df = catalog.assign(prefix=catalog["table_id"].str.extract(r"^(V\d+)")[0])
    return (df.groupby("database")["prefix"]
                .apply(lambda s: sorted(set(s.dropna())))
                .to_dict())


def write_readme(stats: dict[str, Any]) -> None:
    """Auto-generate ``_hf/README.md`` with the dataset-card YAML frontmatter
    HuggingFace expects, declaring per-database configs via glob patterns.

    Configs we register:

    * ``catalog`` — the 502-row index (default config).
    * ``cells``   — the 316 K-row long-format union.
    * one config per NSO database (``V02_population_labour`` etc.) — glob
      over ``tables/{prefix}.*.parquet`` so users can read every matrix
      from a single database with one ``load_dataset`` call.
    """
    by_db = stats["by_database"]
    stats["db_prefixes"]   # injected by build_stats

    # Dataset-card YAML configs section. Per-database configs use a
    # glob over `tables/<prefix>_*.parquet`. A database that publishes
    # under multiple matrix prefixes (e.g. Thương mại, giá cả =
    # V08+V09+V10+V11) gets one path: line per prefix.
    yaml_configs = [
        "configs:",
        "- config_name: catalog",
        "  data_files:",
        "  - split: train",
        "    path: data/catalog.parquet",
        "  default: true",
        "- config_name: cells",
        "  data_files:",
        "  - split: train",
        "    path: data/cells.parquet",
    ]
    for db_row in by_db:
        slug = DB_SLUGS.get(db_row["database"])
        if not slug:
            continue
        # Per-database configs point at the pre-split cells parquets
        # (uniform schema) rather than at `tables/V0X_*.parquet` globs
        # (heterogeneous schemas — would crash datasets.load_dataset).
        yaml_configs.append(f"- config_name: {slug}")
        yaml_configs.append("  data_files:")
        yaml_configs.append("  - split: train")
        yaml_configs.append(f"    path: data/cells_{slug}.parquet")
    yaml_configs_str = "\n".join(yaml_configs)

    # Big per-database table for the README body.
    db_table_lines = [
        "| Database (VN) | Database (EN) | Tables | Cells | HF config |",
        "| ------------- | ------------- | -----: | ----: | --------- |",
    ]
    for db_row in by_db:
        slug = DB_SLUGS.get(db_row["database"], "other")
        en_label = (slug.split("_", 1)[1].replace("_", " ").title()
                     if "_" in slug else slug)
        db_table_lines.append(
            f"| {db_row['database']} | {en_label} | "
            f"{db_row['n_tables']} | {db_row['n_cells']:,} | `{slug}` |"
        )
    db_table_md = "\n".join(db_table_lines)

    # Usage examples in Python
    usage = """
```python
from datasets import load_dataset

# 1) The 502-row catalogue (default config) - for browsing
catalog = load_dataset("REPO_ID")["train"]
print(catalog.column_names)
# -> ['table_id', 'database', 'config_id', 'title', 'source',
#     'variables', 'n_variables', 'n_cells', 'parquet_path', ...]

# 2) The 316K long-format union, every cell across all 502 matrices
cells = load_dataset("REPO_ID", "cells")["train"]
print(cells[0])
# -> {'table_id': 'V02.43', 'database': 'Dân số và lao động',
#     'dimensions': '{"Nghề nghiệp": "Nhân viên", "Năm": "2024"}',
#     'value': 4.21}

# 3) Every matrix in NSO's Population & Labour database (V02), native schema
v02 = load_dataset("REPO_ID", "V02_population_labour")["train"]

# 4) A single matrix in its NATIVE schema. Per-database configs use a
#    unified long-format schema (table_id, database, dimensions JSON,
#    value), so if you need V02.43's native columns ("Nghề nghiệp",
#    "Năm", "value") download the per-table parquet directly. NSO's
#    matrix id "V02.43" is slugged to "V02_43" on disk:
import huggingface_hub as hf
path = hf.hf_hub_download("REPO_ID", "tables/V02_43.parquet",
                            repo_type="dataset")
import pandas as pd
df = pd.read_parquet(path)
print(df.columns.tolist())   # -> ['Nghề nghiệp', 'Năm', 'value']
```
""".strip()

    body = f"""---
language:
- vi
- en
license: cc-by-nc-4.0
size_categories:
- 100K<n<1M
task_categories:
- tabular-classification
- tabular-regression
- text-classification
tags:
- vietnamese
- nso
- statistics
- pxweb
- demographics
- official-statistics
{yaml_configs_str}
---

# personas-vn-pxweb — Vietnamese NSO PX-Web mirror

A complete, table-by-table mirror of the **General Statistics Office of
Vietnam (NSO / GSO)** PX-Web statistical database, scraped from
https://pxweb.nso.gov.vn and restructured for downstream ML use.

This is the **input dataset** for the
[`personas-vn`](https://github.com/tmquan/personas-vn) Vietnamese
persona-generation pipeline. It is also a useful standalone resource
for any work that needs authoritative Vietnamese national statistics
in a clean parquet form.

## At a glance

| Stat                                 | Value             |
| ------------------------------------ | ----------------- |
| PX-Web matrices                      | **{stats['total_tables']}**       |
| Long-format cells                    | **{stats['total_cells']:,}**       |
| First-class NSO databases            | **{stats['total_databases']}**          |
| Earliest year of data                | **{stats['earliest_year']}**       |
| Latest year of data                  | **{stats['latest_year']}**       |
| Tables fetched via JSON REST API     | **{stats['by_source'].get('json_api', 0)}**  |
| Tables fetched via legacy HTML form  | **{stats['by_source'].get('html_form', 0)}** |
| Languages in source labels           | Vietnamese (canonical) + English |

## What's on the Hub

```
.
├── README.md                     (this file)
├── _stats.json                   numbers / tables this card quotes
├── data/
│   ├── catalog.parquet           {stats['total_tables']}-row index of every matrix
│   └── cells.parquet             {stats['total_cells']:,}-row long-format union
├── tables/
│   ├── V01_01.parquet            one parquet per NSO matrix id, native schema
│   ├── V02_43.parquet
│   └── ... ({stats['total_tables']} files)
├── notebooks/                    four executed Jupyter notebooks (interactive Plotly)
│   ├── DATAANALYSIS.ipynb        — 39 analytical figures across 13 NSO domains
│   ├── DATAVISUALIZATION.ipynb   — 11 Vietnam choropleth maps (Mapbox-GL)
│   ├── DATAEXPLORATION.ipynb     — 11 UMAP embedding views of the 502-table catalog
│   └── DATASYNTHESIS.ipynb       — 5 persona-generator dataset views (PGM + OCEAN)
├── docs/                         seven companion narrative docs (≈ 140 KB)
│   ├── DATAPROCESSING.md         curator pipeline (download → parse → extract → embed → reduce)
│   ├── DATAANALYSIS.md           analytical walkthrough — figure by figure
│   ├── DATAVISUALIZATION.md      Vietnam choropleth atlas
│   ├── DATAEXPLORATION.md        catalog UMAP exploration
│   ├── DATASYNTHESIS.md          persona generator + OCEAN priors
│   ├── ONTOLOGY.md               bilingual NSO ontology tree
│   └── DATASETS.md               persona-dataset schema (joins this catalog)
├── figures/                      66 PNG + 66 HTML pairs (interactive Plotly)
│   ├── analysis/      ({{01..39}}_*.png + .html)
│   ├── visualization/ ({{01..11}}_*.png + .html)   ← Vietnam choropleth maps
│   ├── exploration/   ({{01..11}}_*.png + .html)   ← curator UMAP scatters
│   └── synthesis/     ({{01..05}}_*.png + .html)   ← persona-generator UMAPs
├── assets/                       README-embedded analysis PNGs + inventory CSV
└── raw/
    └── pxweb/vi/                 original parquets + metadata.json (502 matrices)
```

Every figure under ``figures/`` ships **both formats**: the static PNG
for inline rendering in this dataset card, plus a self-contained
**interactive HTML** with the full Plotly toolkit (pan, zoom, hover
tooltips, legend toggle). Click any HTML link below or download
``figures/<section>/<figure>.html`` to open it in a browser.

## Per-database breakdown

{db_table_md}

## Representative figures

A curated tour of the dataset, one figure per source notebook. Each
embed below is the **static PNG** rendered into the dataset card; click
the *interactive* link beneath it (or download from
``figures/<section>/<file>.html``) to open the **fully interactive
Plotly version** with pan / zoom / hover tooltips / legend toggle in
your browser.

### From [`notebooks/DATAANALYSIS.ipynb`](notebooks/DATAANALYSIS.ipynb) — when each NSO database has data

![Time-coverage Gantt per NSO database](figures/analysis/04_time_coverage.png)

*Interactive: [`figures/analysis/04_time_coverage.html`](figures/analysis/04_time_coverage.html)*

### From [`notebooks/DATAVISUALIZATION.ipynb`](notebooks/DATAVISUALIZATION.ipynb) — population by province (choropleth)

![Vietnam population choropleth, latest year](figures/visualization/02_population_by_province.png)

*Interactive: [`figures/visualization/02_population_by_province.html`](figures/visualization/02_population_by_province.html) — pan, zoom, hover any province for the exact value*

### From [`notebooks/DATAEXPLORATION.ipynb`](notebooks/DATAEXPLORATION.ipynb) — catalog semantic map (UMAP × ontology domain)

![Curator UMAP coloured by ontology domain](figures/exploration/06_umap_by_domain.png)

*Interactive: [`figures/exploration/06_umap_by_domain.html`](figures/exploration/06_umap_by_domain.html) — every dot is one PX-Web matrix, hover for title + database*

### From [`notebooks/DATASYNTHESIS.ipynb`](notebooks/DATASYNTHESIS.ipynb) — synthetic persona UMAP × region

![Persona embedding UMAP × region](figures/synthesis/02_umap_by_region.png)

*Interactive: [`figures/synthesis/02_umap_by_region.html`](figures/synthesis/02_umap_by_region.html)*

### Full index — 66 figures total

Every figure ships in **both formats** under ``figures/<section>/``: a
static PNG for inline embedding (already shown above for the four
flagship views) and a self-contained interactive HTML with the full
Plotly toolkit. The 14 most illustrative views, by section:

| Section | File (HTML) | What it shows |
| --- | --- | --- |
| analysis      | `04_time_coverage.html`               | per-database year-coverage Gantt (above) |
| analysis      | `08_population_timeseries.html`       | Vietnam population 1990–2024, total / urban / rural |
| analysis      | `13_occupation_structure_shift.html`  | ISCO 10-class shift earliest vs latest year |
| analysis      | `17_gdp_current_prices.html`          | GDP at current prices time-series |
| analysis      | `35_income_by_quintile.html`          | monthly income per capita by region × strata |
| analysis      | `37_gdp_per_capita_asean.html`        | Vietnam vs ASEAN-9 PPP GDP/capita |
| visualization | `02_population_by_province.html`      | Choropleth: population by province, latest year (above) |
| visualization | `10_enterprises_iip_bubble.html`      | bubble map: enterprises × IIP growth |
| visualization | `11_macro_regions.html`               | the 6 NSO macro-regions colour-coded |
| exploration   | `06_umap_by_domain.html`              | UMAP × ontology-domain, 12 colours (above) |
| exploration   | `08_umap_by_cellcount.html`           | UMAP × log10(cells per table) |
| exploration   | `11_semantic_neighbours.html`         | nearest-neighbour cosine search demo |
| synthesis     | `02_umap_by_region.html`              | persona UMAP × NSO macro-region (above) |
| synthesis     | `04_umap_by_education.html`           | persona UMAP × education_level |

The four ``notebooks/*.ipynb`` files contain the **executed Plotly
outputs** as well — open them in JupyterLab (or directly in HF's
notebook viewer) for the full walk-through that produced each figure.

## Usage

{usage.replace('REPO_ID', '{repo_id}')}

## Citation policy

The underlying data belongs to the **General Statistics Office of
Vietnam**. When using this mirror, please credit them as the original
publisher and respect the source license terms at
https://pxweb.nso.gov.vn.

If this mirror or the analysis figures are useful in academic work,
please also cite:

```bibtex
@misc{{nso_2026,
  title        = {{Thống kê — National Statistics Office}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/tmquan/nso-gov-vn}}}},
  note         = {{Mirror of the NSO corpus published at https://nso.gov.vn}}
}}
```

## Reproducibility

Every artefact in this repo is regenerable. From the source repo:

```bash
git clone https://github.com/tmquan/personas-vn
cd personas-vn
conda create -n pgm python=3.11 -y && conda activate pgm    # one-time
pip install -e ".[curator,viz,dev]"                          # one-time
python -m packages.pipeline.cli curate                       # downloads from NSO PX-Web
python -m scripts.upload_to_hf                               # builds + pushes this repo
```

See the source repo's [`DATAPROCESSING.md`](https://github.com/tmquan/personas-vn/blob/main/DATAPROCESSING.md)
for the full crawl protocol and [`DATAANALYSIS.md`](https://github.com/tmquan/personas-vn/blob/main/DATAANALYSIS.md)
for the analytical walkthrough.
"""

    (HF_DIR / "README.md").write_text(body)


# ---------------------------------------------------------------------------
# Step 2 — render assets (we already have the 38 PNGs from notebook)
# ---------------------------------------------------------------------------
def render_assets() -> None:
    """Step 2 — make sure the figure / notebook / docs folders are fresh.

    Unlike ViLA we don't re-execute the notebooks here — the canonical
    figures are produced by ``DATA{ANALYSIS,VISUALIZATION,EXPLORATION,
    SYNTHESIS}.ipynb`` themselves (each is self-contained and runs in
    1–4 minutes). We just verify the on-disk inventory is complete,
    otherwise warn loudly. ``--skip-assets`` bypasses even the
    verification.

    The expected on-disk inventory is **66 PNG + 66 HTML pairs across
    four sections** (analysis 39, maps→visualization 11, exploration
    11, synthesis 5) plus four notebooks and seven markdown docs.
    """
    log.info("[2/3] Verifying notebook + figure inventory …")
    n_total_png = 0
    for section, src_dir in FIGURE_SECTIONS.items():
        if not src_dir.exists():
            log.warning("  no %s/ folder — re-run that notebook first",
                         src_dir.relative_to(REPO_ROOT))
            continue
        n_png = len(list(src_dir.glob("*.png")))
        n_html = len(list(src_dir.glob("*.html")))
        n_total_png += n_png
        log.info("  %s: %d PNGs, %d HTMLs", section, n_png, n_html)
    if n_total_png < 60:
        log.warning("  expected ~66 PNGs across 4 sections; only found %d — "
                     "did all four notebooks execute?", n_total_png)
    copy_artefacts()
    log.info("  refreshed %s/", HF_DIR.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Step 3 — upload (mirrors ViLA's Upload dataclass + UPLOADS table)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Upload:
    """A single ``HfApi.upload_folder`` / ``upload_file`` invocation.

    Same contract as ViLA's ``_to_hf.Upload`` — see
    https://github.com/tmquan/ViLA/blob/main/data/anle.toaan.gov.vn/_to_hf.py
    for the rationale.
    """
    local: Path
    in_repo: str
    allow: tuple[str, ...] | None = None
    delete: tuple[str, ...] | None = None
    message: str = ""


def _uploads_for_pxweb() -> tuple[Upload, ...]:
    """Order matters: small / fast first (README + parquet bundle), then
    the heavier per-table folder, then notebooks, docs, figures, and
    finally raw. The per-table folder uses
    ``delete_patterns=("*.parquet",)`` so that a re-run with fewer
    matrices atomically wipes stale ones in the same commit.

    Bundle layout pushed to the Hub::

        ./README.md, _stats.json
        data/{catalog,cells,cells_<config>}.parquet
        tables/V01_01.parquet … V15_06.parquet  (502 files)
        notebooks/{DATAANALYSIS,DATAVISUALIZATION,DATAEXPLORATION,DATASYNTHESIS}.ipynb
        docs/{DATAPROCESSING,DATAANALYSIS,DATAVISUALIZATION,DATAEXPLORATION,
              DATASYNTHESIS,ONTOLOGY,DATASETS}.md
        figures/{analysis,visualization,exploration,synthesis}/{*.png, *.html}
        assets/{0..38}_*.png, 16_full_table_inventory.csv
        notebook.ipynb           (legacy: copy of DATAANALYSIS.ipynb at root)
        raw/pxweb/vi/*.parquet   (the 502 raw matrices + metadata.json)
    """
    return (
        Upload(local=HF_DIR, in_repo=".",
                allow=("README.md", "_stats.json"),
                message="Refresh dataset card + stats snapshot"),
        Upload(local=HF_DATA_DIR, in_repo="data",
                delete=("*.parquet",),
                message="Refresh consolidated parquet bundle "
                        "(catalog.parquet + cells.parquet)"),
        Upload(local=HF_TABLES_DIR, in_repo="tables",
                delete=("*.parquet",),
                message="Refresh per-matrix parquet shards"),
        Upload(local=HF_NOTEBOOKS_DIR, in_repo="notebooks",
                delete=("*.ipynb",),
                message="Refresh interactive Jupyter notebooks "
                        "(DATAANALYSIS, DATAVISUALIZATION, DATAEXPLORATION, DATASYNTHESIS)"),
        Upload(local=HF_DOCS_DIR, in_repo="docs",
                delete=("*.md",),
                message="Refresh narrative markdown docs"),
        Upload(local=HF_FIGURES_DIR, in_repo="figures",
                delete=("*.png", "*.html", "*.csv"),
                message="Refresh interactive Plotly figures "
                        "(PNG + HTML pairs across 4 sections)"),
        Upload(local=HF_ASSETS_DIR, in_repo="assets",
                message="Refresh analysis figures + inventory"),
        Upload(local=HF_DIR / "notebook.ipynb", in_repo="notebook.ipynb",
                message="Refresh top-level notebook (legacy alias for DATAANALYSIS.ipynb)"),
        Upload(local=PXWEB_VI_DIR, in_repo="raw/pxweb/vi",
                message="Refresh raw PX-Web parquets + metadata"),
    )


def _uploads_for_personas() -> tuple[Upload, ...]:
    """Optional secondary push for the persona dataset.

    Goes to a *separate* HF repo because the schema is unrelated to the
    PX-Web matrices.
    """
    candidates = [
        PERSONAS_DIR / "personas_enriched.json",
        PERSONAS_DIR / "personas.json",
    ]
    sources = [p for p in candidates if p.exists()]
    if not sources:
        log.warning("  no personas.json / personas_enriched.json found in %s",
                     PERSONAS_DIR)
        return ()

    # Bundle: a single parquet of personas + (if present) embedding/reduced parquets.
    HF_PERSONAS = HF_DIR.parent / "_hf_personas"
    ensure_dir(HF_PERSONAS / "data")
    primary = sources[0]
    j = json.loads(primary.read_text(encoding="utf-8"))
    rows = j.get("personas", j) if isinstance(j, dict) else j
    df = pd.DataFrame(rows)
    out = HF_PERSONAS / "data" / "personas.parquet"
    df.to_parquet(out, compression="zstd", index=False)
    log.info("  wrote %s (%d personas)", out.relative_to(REPO_ROOT), len(df))

    for sidecar in ("personas_embeddings.parquet", "personas_reduced.parquet"):
        src = PERSONAS_DIR / sidecar
        if src.exists():
            shutil.copy2(src, HF_PERSONAS / "data" / sidecar)
            log.info("  + %s", sidecar)

    return (
        Upload(local=HF_PERSONAS / "data", in_repo="data",
                delete=("*.parquet",),
                message="Refresh persona parquet bundle"),
    )


def upload(repo: str, push: str = "curated") -> None:
    """Step 3 — push everything to ``repo``."""
    from huggingface_hub import HfApi, create_repo

    log.info("[3/3] Uploading to %s …", repo)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    api = HfApi()
    create_repo(repo, repo_type="dataset", exist_ok=True)
    log.info("  repo ready: https://huggingface.co/datasets/%s", repo)

    uploads = _uploads_for_pxweb() if push == "curated" else _uploads_for_personas()

    for u in uploads:
        if not u.local.exists():
            log.warning("  skip %s (not found)",
                         u.local.relative_to(REPO_ROOT))
            continue
        log.info("  pushing %s -> %s",
                  u.local.relative_to(REPO_ROOT), u.in_repo)
        if u.local.is_file():
            api.upload_file(
                path_or_fileobj=str(u.local),
                path_in_repo=u.in_repo,
                repo_id=repo,
                repo_type="dataset",
                commit_message=u.message or f"Refresh {u.in_repo}",
            )
        else:
            api.upload_folder(
                folder_path=str(u.local),
                path_in_repo=u.in_repo,
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=list(u.allow) if u.allow else None,
                delete_patterns=list(u.delete) if u.delete else None,
                commit_message=u.message or f"Refresh {u.in_repo}",
            )
    log.info("  done. https://huggingface.co/datasets/%s", repo)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build + upload personas-vn HuggingFace dataset(s).",
    )
    ap.add_argument("--push", choices=("curated", "personas"),
                     default="curated",
                     help="Which dataset to push (default: curated PX-Web mirror)")
    ap.add_argument("--repo",
                     help=f"Hub repo id (default: {DEFAULT_REPO_PXWEB} for curated, {DEFAULT_REPO_PERSONAS} for personas)")
    ap.add_argument("--skip-consolidate", action="store_true",
                     help="Skip the parquet roll-up (use existing _hf/data/)")
    ap.add_argument("--skip-assets", action="store_true",
                     help="Skip refreshing the analysis assets folder")
    ap.add_argument("--no-upload", action="store_true",
                     help="Build the bundle locally but don't push to the Hub")
    args = ap.parse_args()

    repo = args.repo or (DEFAULT_REPO_PXWEB if args.push == "curated"
                          else DEFAULT_REPO_PERSONAS)

    if args.push == "curated":
        if not args.skip_consolidate:
            consolidate()
        if not args.skip_assets:
            render_assets()
    elif args.push == "personas":
        # Personas pipeline is much smaller — just stage the parquet
        # bundle in `_hf_personas/data/` if needed.
        pass

    if not args.no_upload:
        upload(repo, push=args.push)
    log.info("All done.")


if __name__ == "__main__":
    main()
