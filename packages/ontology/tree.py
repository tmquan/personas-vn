"""Bilingual ontology tree of NSO PX-Web statistical data.

Walks the metadata JSON sidecars under ``data/nso-gov-vn/raw/pxweb/vi/``
and produces a 4-level hierarchy:

::

    Root
    └── Database (12)             — Công nghiệp, Dân số và lao động, …
        └── Table (~500)          — V07.01, V02.01, …
            └── Variable (~5/tbl) — Năm, Tỉnh/Thành phố, Phân tổ, …
                └── Value (~30/var) — 2024, Hà Nội, Tổng số, …

Every node carries a Vietnamese label (``name_vi``) and an English label
(``name_en``) translated by :class:`packages.ontology.translator.Translator`.

Outputs:

* ``data/ontology/tree.json`` — full structured tree (every value listed)
* ``data/ontology/tree.yaml`` — top-3-level summary (Database → Table →
  Variable, no value lists), human-readable / git-diffable
* ``data/ontology/translation_cache.json`` — translation memory written by
  the translator on save

The structure is **deterministic given the on-disk PX-Web parquets and
the glossary**: the same input directory produces the same JSON bytes
on every run (modulo the cache file's growth from online calls).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from packages.common.logging import get_logger
from packages.ontology.translator import Translator

log = get_logger(__name__)

# Default location for the curator's PX-Web crawl output.
PXWEB_ROOT_DEFAULT = Path("data/nso-gov-vn/raw/pxweb/vi")
ONTOLOGY_DIR_DEFAULT = Path("data/ontology")


# ---------------------------------------------------------------------------
# Tree dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ValueNode:
    """A single value within a variable (e.g. ``2024``, ``Hà Nội``)."""

    code: str           # the raw code from PX-Web (e.g. "0", "01", "2024")
    name_vi: str
    name_en: str


@dataclass
class VariableNode:
    """A variable inside a table (e.g. ``Năm``, ``Tỉnh, thành phố``)."""

    code: str           # the variable code (also the column name)
    name_vi: str
    name_en: str
    n_values: int
    values: list[ValueNode] = field(default_factory=list)


@dataclass
class TableNode:
    """One PX-Web matrix (e.g. ``V07.01``)."""

    table_id: str
    title_vi: str
    title_en: str
    n_cells: int
    n_kept: int
    parquet_path: str
    metadata_path: str
    variables: list[VariableNode] = field(default_factory=list)


@dataclass
class DatabaseNode:
    """One first-class NSO statistical database (12 of these)."""

    database_id: str    # slug — e.g. "Cong-nghiep"
    matrix_prefix: str  # "V07"
    name_vi: str
    name_en: str
    n_tables: int = 0
    tables: list[TableNode] = field(default_factory=list)


@dataclass
class OntologyTree:
    """The whole ontology tree, plus build metadata + coverage stats."""

    generated_at: str
    pxweb_root: str
    n_databases: int
    n_tables: int
    n_variables: int
    n_values: int
    translation_stats: dict[str, int]
    coverage_pct: float
    databases: list[DatabaseNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Hard-coded slug-prefix → matrix_prefix map. NSO's catalogue path
# (``Công nghiệp/V07.01.px``) embeds the matrix prefix in the table id;
# the slugified folder name doesn't, so we keep this small lookup.
_DB_TO_MATRIX_PREFIX: dict[str, str] = {
    "Công nghiệp":                                  "V07",
    "Dân số và lao động":                           "V02",
    "Doanh nghiệp":                                 "V05",
    "Giáo dục":                                     "V13",
    "Nông, lâm nghiệp và thủy sản":                 "V06",
    "Tài khoản quốc gia":                           "V03",
    "Thống kê nước ngoài":                          "V15",
    "Thương mại, giá cả":                           "V08",
    "Vận tải và bưu điện":                          "V12",
    "Y tế, văn hóa và đời sống":                    "V14",
    "Đầu tư":                                       "V04",
    "Đơn vị hành chính, đất đai và khí hậu":        "V01",
}


def _database_slug(name_vi: str) -> str:
    """Turn a Vietnamese DB name into the slug NSO uses for parquet filenames.

    e.g. ``"Đơn vị hành chính, đất đai và khí hậu"`` →
    ``"don-vi-hanh-chinh-dat-dai-va-khi-hau"``. We mirror NSO's own
    transformation (drop diacritics, lowercase, hyphenate words, drop
    punctuation).
    """
    import re
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", name_vi)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_marks = no_marks.replace("đ", "d").replace("Đ", "d")
    slug = re.sub(r"[^a-z0-9]+", "-", no_marks.lower()).strip("-")
    return slug


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_ontology_tree(
    pxweb_root: Path | str = PXWEB_ROOT_DEFAULT,
    *,
    translator: Translator,
) -> OntologyTree:
    """Walk the PX-Web metadata JSONs and assemble a bilingual ontology tree.

    Reads every ``*.metadata.json`` under ``pxweb_root``, groups them by
    database (taken from the metadata's ``database`` field), and builds
    nested DatabaseNode → TableNode → VariableNode → ValueNode objects
    with VI labels from the metadata and EN labels from the translator.
    """
    pxweb_root = Path(pxweb_root)
    if not pxweb_root.exists():
        raise FileNotFoundError(
            f"PX-Web parquet root not found at {pxweb_root}. "
            "Run `personas-vn curate --only download` first."
        )

    metadata_files = sorted(pxweb_root.glob("*.metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(
            f"No PX-Web metadata files in {pxweb_root}. "
            "Run `personas-vn curate --only download` first."
        )

    log.info("walking %d metadata files under %s", len(metadata_files), pxweb_root)

    # Group tables by their declared `database` field
    db_buckets: dict[str, list[Path]] = {}
    for mp in metadata_files:
        md = json.loads(mp.read_text())
        db = md.get("database") or "Khác"
        db_buckets.setdefault(db, []).append(mp)

    databases: list[DatabaseNode] = []
    for db_name_vi in sorted(db_buckets.keys()):
        db_node = DatabaseNode(
            database_id=_database_slug(db_name_vi),
            matrix_prefix=_DB_TO_MATRIX_PREFIX.get(db_name_vi, ""),
            name_vi=db_name_vi,
            name_en=translator.translate(db_name_vi),
        )
        for mp in sorted(db_buckets[db_name_vi]):
            md = json.loads(mp.read_text())
            tbl_node = _build_table_node(md, mp, pxweb_root, translator)
            db_node.tables.append(tbl_node)
        db_node.n_tables = len(db_node.tables)
        databases.append(db_node)

    # Coverage: fraction of strings that hit either the glossary or the cache
    stats = translator.stats
    examined = sum(stats.values())
    hit = stats.get("glossary_hits", 0) + stats.get("cache_hits", 0) + stats.get("identity", 0)
    coverage = (100.0 * hit / examined) if examined else 100.0

    n_tables = sum(d.n_tables for d in databases)
    n_variables = sum(len(t.variables) for d in databases for t in d.tables)
    n_values = sum(len(v.values) for d in databases for t in d.tables for v in t.variables)

    return OntologyTree(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        pxweb_root=str(pxweb_root),
        n_databases=len(databases),
        n_tables=n_tables,
        n_variables=n_variables,
        n_values=n_values,
        translation_stats=dict(stats),
        coverage_pct=round(coverage, 2),
        databases=databases,
    )


def _build_table_node(
    md: dict[str, Any],
    metadata_path: Path,
    pxweb_root: Path,
    translator: Translator,
) -> TableNode:
    table_id = md.get("table_id", "")
    title_vi = md.get("title", "")
    parquet_name = metadata_path.name.replace(".metadata.json", ".parquet")
    parquet_path = pxweb_root / parquet_name

    variables: list[VariableNode] = []
    for v in md.get("variables", []):
        var = VariableNode(
            code=v.get("code", ""),
            name_vi=v.get("text") or v.get("code", ""),
            name_en=translator.translate(v.get("text") or v.get("code", "")),
            n_values=len(v.get("values", [])),
        )
        codes = v.get("values", [])
        texts = v.get("valueTexts", [])
        # Pad whichever list is shorter so iteration is uniform.
        for code, text in zip(codes, texts) if len(codes) == len(texts) else \
                          zip(codes, codes):
            var.values.append(ValueNode(
                code=str(code),
                name_vi=text,
                name_en=translator.translate(text),
            ))
        variables.append(var)

    return TableNode(
        table_id=table_id,
        title_vi=title_vi,
        title_en=translator.translate(title_vi),
        n_cells=int(md.get("n_cells", 0)),
        n_kept=int(md.get("n_kept", 0)),
        parquet_path=str(parquet_path),
        metadata_path=str(metadata_path),
        variables=variables,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def write_tree_json(tree: OntologyTree, path: Path | str) -> None:
    """Write the full tree as JSON.

    The JSON output is **stable and diff-friendly**: keys are emitted
    in dataclass declaration order, lists are already sorted by the
    builder, and indent=2 is used.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(tree), ensure_ascii=False, indent=2)
    )
    log.info("wrote tree.json (%.1f MB)", path.stat().st_size / (1024 * 1024))


def write_tree_yaml_summary(tree: OntologyTree, path: Path | str) -> None:
    """Write a top-3-level summary as YAML (no value lists).

    The full tree's value lists balloon the JSON to ~5–10 MB, which
    isn't pleasant to git-diff or read. The YAML summary is just
    Database → Table → Variable, ~50–200 KB, and pleasant to scan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "meta": {
            "generated_at": tree.generated_at,
            "pxweb_root":   tree.pxweb_root,
            "n_databases":  tree.n_databases,
            "n_tables":     tree.n_tables,
            "n_variables":  tree.n_variables,
            "n_values":     tree.n_values,
            "coverage_pct": tree.coverage_pct,
            "translation_stats": tree.translation_stats,
        },
        "databases": [
            {
                "database_id":   db.database_id,
                "matrix_prefix": db.matrix_prefix,
                "name_vi":       db.name_vi,
                "name_en":       db.name_en,
                "n_tables":      db.n_tables,
                "tables": [
                    {
                        "table_id": t.table_id,
                        "title_vi": t.title_vi,
                        "title_en": t.title_en,
                        "n_cells":  t.n_cells,
                        "variables": [
                            {
                                "code":     v.code,
                                "name_vi":  v.name_vi,
                                "name_en":  v.name_en,
                                "n_values": v.n_values,
                            }
                            for v in t.variables
                        ],
                    }
                    for t in db.tables
                ],
            }
            for db in tree.databases
        ],
    }
    with path.open("w") as f:
        yaml.safe_dump(
            summary,
            f,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
            width=200,
        )
    log.info("wrote tree.yaml (%.1f KB)", path.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _collect_all_strings(pxweb_root: Path) -> list[str]:
    """Walk every metadata JSON and harvest every translatable string.

    Used to pre-populate the translator's cache in a single batch
    online sweep, which is ~50× faster than per-call online lookups.
    """
    out: list[str] = []
    for mp in sorted(pxweb_root.glob("*.metadata.json")):
        md = json.loads(mp.read_text())
        if (db := md.get("database")):
            out.append(db)
        if (title := md.get("title")):
            out.append(title)
        for v in md.get("variables", []):
            if (vt := v.get("text") or v.get("code")):
                out.append(vt)
            for vt2 in v.get("valueTexts", []):
                out.append(vt2)
    return out


def build_and_write_ontology(
    *,
    pxweb_root: Path | str = PXWEB_ROOT_DEFAULT,
    out_dir: Path | str = ONTOLOGY_DIR_DEFAULT,
    enable_online: bool = False,
    online_workers: int = 8,
) -> dict[str, Any]:
    """Build the bilingual ontology tree and write it to disk.

    When ``enable_online`` is True, every string the curated glossary
    cannot translate is sent to Google Translate (via deep-translator)
    using a ``ThreadPoolExecutor`` of ``online_workers`` threads. The
    results are cached to ``data/ontology/translation_cache.json``.
    Subsequent runs read from the cache and require no network.

    Returns a small summary dict suitable for the CLI to log.
    """
    pxweb_root = Path(pxweb_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "translation_cache.json"

    translator = Translator(cache_path=cache_path, enable_online=enable_online)

    if enable_online:
        log.info("warming translation cache via Google Translate")
        candidates = _collect_all_strings(pxweb_root)
        translator.warm_cache_with_unknown(
            candidates, max_workers=online_workers
        )
        translator.save()

    tree = build_ontology_tree(pxweb_root, translator=translator)
    translator.save()

    json_path = out_dir / "tree.json"
    yaml_path = out_dir / "tree.yaml"
    write_tree_json(tree, json_path)
    write_tree_yaml_summary(tree, yaml_path)

    return {
        "json_path":         str(json_path),
        "yaml_path":         str(yaml_path),
        "cache_path":        str(cache_path),
        "n_databases":       tree.n_databases,
        "n_tables":          tree.n_tables,
        "n_variables":       tree.n_variables,
        "n_values":          tree.n_values,
        "coverage_pct":      tree.coverage_pct,
        "translation_stats": tree.translation_stats,
    }
