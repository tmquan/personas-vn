# Bilingual NSO ontology tree

This document describes the **complete bilingual ontology tree** that
`packages.ontology.tree` builds from the raw NSO PX-Web parquet
metadata under `data/nso-gov-vn/raw/pxweb/vi/`. Every node in the tree
carries a Vietnamese label (the canonical NSO label) **and** an
English label translated by the curated glossary +
optional Google Translate fallback.

90-second build:

```bash
conda activate pgm                                       # one-time setup: see README

# Offline (curated glossary only): ~6 s, ~76 % EN coverage
python -m packages.pipeline.cli build-ontology

# Online (curated glossary + Google for the long tail): ~90 s, 100 % EN coverage
python -m packages.pipeline.cli build-ontology --online

# Subsequent runs are offline-fast: cache → 100 % EN coverage in 6 s
python -m packages.pipeline.cli build-ontology   # uses translation_cache.json from previous --online run
```

Outputs land under `data/ontology/`:

| File | Purpose |
|---|---|
| `tree.json` | Full structured tree — every value listed (~2 MB) |
| `tree.yaml` | Top-3-level summary (DB → Table → Variable, no value lists) — diff-friendly (~270 KB) |
| `translation_cache.json` | Online-translation memory (~1 200 entries, ~160 KB), grows over re-runs |

---

## 1. Tree shape

The hierarchy mirrors NSO's own PX-Web nav structure:

```
Root (n_databases=12, n_tables=502, n_variables=1 091, n_values=9 690)
└── Database (12)             — Công nghiệp, Dân số và lao động, …
    └── Table (502)           — V07.01, V02.01, …
        └── Variable (1 091)  — Năm, Tỉnh/Thành phố, Phân tổ, …
            └── Value (9 690) — 2024, Hà Nội, Tổng số, …
```

Counts above are from the corpus shipped with the repo (Dec 2025
crawl). Each node carries:

| Level | Fields per node |
|---|---|
| Database | `database_id`, `matrix_prefix`, `name_vi`, `name_en`, `n_tables`, `tables[]` |
| Table | `table_id`, `title_vi`, `title_en`, `n_cells`, `n_kept`, `parquet_path`, `metadata_path`, `variables[]` |
| Variable | `code`, `name_vi`, `name_en`, `n_values`, `values[]` |
| Value | `code`, `name_vi`, `name_en` |

The root node also carries `generated_at`, `pxweb_root`, the four
running counts, the `translation_stats` dict, and `coverage_pct`.

---

## 2. Translation pipeline (3 tiers)

`packages/ontology/translator.py` implements a 3-tier cascade. Each
input string flows through the tiers in order until one produces a
translation:

```
                ┌────────────────────────────┐
   Vietnamese ─►│  1. Curated glossary       ├─► English
   string       │     (offline, ~280 entries)│
                └─────────────┬──────────────┘
                              │ miss
                              ▼
                ┌────────────────────────────┐
                │  1b. Compositional rules   ├─► English
                │  "X (Y)", "A, B", "A / B"  │
                │  (recurse into parts)      │
                └─────────────┬──────────────┘
                              │ miss
                              ▼
                ┌────────────────────────────┐
                │  2. On-disk cache          ├─► English
                │  (results from any prior   │
                │   online run)              │
                └─────────────┬──────────────┘
                              │ miss
                              ▼
                ┌────────────────────────────┐
                │  2b. Google Translate      ├─► English (cached)
                │  (deep-translator,         │
                │   only if --online)        │
                └─────────────┬──────────────┘
                              │ unavailable / fail
                              ▼
                ┌────────────────────────────┐
                │  3. Identity fallback      ├─► <input>
                │  (numbers, codes, %)       │
                └────────────────────────────┘
```

### Tier 1 — Curated glossary (`packages/ontology/glossary.py`)

A hand-curated VI→EN dictionary of statistical vocabulary. Entries are
grouped by category for auditability:

| Category | Count | Examples |
|---|---|---|
| Databases | 12 | `Công nghiệp` → `Industry` |
| Variable codes | ~70 | `Năm` → `Year`, `Tỉnh, thành phố` → `Province/city` |
| Provinces | 63 | `Hà Nội` → `Hanoi`, `TP.Hồ Chí Minh` → `Ho Chi Minh City` |
| Macro-regions | 6 | `Đông Nam Bộ` → `South East` |
| Aggregates | ~12 | `TỔNG SỐ` → `TOTAL`, `Cả nước` → `Whole country` |
| Time prefixes | 3 | `Sơ bộ` → `Preliminary`, `Ước tính` → `Estimated` |
| Education | ~20 | `Đại học` → `University`, `Mầm non` → `Pre-school` |
| ISCO occupations | ~15 | NSO V02.43 categories |
| ISIC industry sectors | ~50 | NSO V02.42 / V07 categories |
| Units | ~30 | `Triệu đồng` → `Million VND`, `Lít` → `Litres` |
| Edu vocabulary | ~12 | `Trường học` → `Schools`, `Học sinh` → `Students` |
| Common products | ~25 | `Bia các loại` → `Beer (all types)`, `Xi măng` → `Cement` |
| Indicator phrases | ~15 | `Chỉ số sản xuất công nghiệp` → `Industrial production index` |

Total: **~280 entries** (some categories share keys after diacritic
normalisation). Every English mirror has been hand-checked against
NSO's own English yearbooks for terminology consistency.

The glossary also handles the `Đ` (U+0110) ↔ `Ð` (U+00D0) diacritic
ambiguity NSO occasionally exhibits in its PX-Web exports — both
forms map to the same English label.

### Tier 1b — Compositional rules

Glossary lookup recurses into compound strings of the form:

* `"<Head> (<Tail>)"` — translates Head and Tail independently and
  recomposes. Example: `"Bia các loại (Lít)"` → `"Beer (all types) (Litres)"`.
* `"A, B, C"` and `"A / B / C"` — translates each part if the glossary
  knows them all.

Compositional translation only succeeds if **every** sub-part is a
real glossary hit (no identity-mixing), so mistranslations are
avoided.

### Tier 2 — On-disk cache (`data/ontology/translation_cache.json`)

Whatever Tier 2b (Google) produces is pinned to disk in a sorted JSON
file. The cache is the source of truth for re-runs: once you've run
`python -m packages.pipeline.cli build-ontology --online` once, subsequent runs read from the
cache and produce **identical bytes** without network access. This
restores determinism (Google occasionally rephrases the same input
across calls).

Delete the cache to refresh from Google.

### Tier 2b — Google Translate (`deep-translator`)

Only active when the build was started with `--online` and
`pip install deep-translator` succeeded. Uses a `ThreadPoolExecutor`
with 8–10 concurrent requests for ~14 strings/sec throughput; the full
2 700-string long-tail finishes in ~90 s.

(`translate_batch()` was tried first but turned out to hang
indefinitely on certain inputs — the per-call thread pool is a
deliberate workaround.)

### Tier 3 — Identity fallback

Numeric strings (years, percentages, codes like `V07.01`) bypass
translation entirely. Strings the prior tiers all miss are also
returned as-is and counted in `translation_stats.misses` so the
coverage report can flag them for manual curation.

---

## 3. Coverage results (Dec 2025 crawl, 502 tables)

Stats are reported in the build summary's `translation_stats` field.

| Tier | Strings | Share |
|---|---:|---:|
| Glossary hit | 6 288 | 55.7 % |
| Cache hit (online previously)¹ | 2 635 | 23.3 % |
| Identity (numbers / codes) | 2 372 | 21.0 % |
| Miss (offline-only, no cache) | 0² | 0.0 % |
| **Total examined** | 11 295 | 100.0 % |

¹ When `translation_cache.json` is empty (first offline-only run), this
counts as miss instead. Coverage is still **76.7 %** without any online
calls — the curated glossary alone handles ~80 % of value-text
occurrences and ~95 % of variable-code occurrences.

² After one full online warm-up. The cache is committed to disk so
subsequent offline runs reach 100 % coverage in 6 s.

| Mode | Coverage | Wall time | Network |
|---|---|---|---|
| Offline first run | 76.7 % | ~6 s | none |
| Online warm-up | 100.0 % | ~90 s | ~1 200 calls |
| Offline rebuild (cache present) | 100.0 % | ~6 s | none |

---

## 4. Sample output

A snippet from `data/ontology/tree.yaml`:

```yaml
- database_id: cong-nghiep
  matrix_prefix: V07
  name_vi: Công nghiệp
  name_en: Industry
  n_tables: 9
  tables:
  - table_id: V07.01
    title_vi: Chỉ số sản xuất công nghiệp phân theo ngành công nghiệp*
    title_en: Industrial production index by industry*
    n_cells: 468
    variables:
    - code: Ngành công nghiệp
      name_vi: Ngành công nghiệp
      name_en: Industry sector
      n_values: 36
    - code: Năm
      name_vi: Năm
      name_en: Year
      n_values: 13
```

A few interesting cases (from the full `tree.json`):

* `"Sơ bộ 2024"` → `"Preliminary 2024"` (time-prefix detection)
* `"Hà Nội"` → `"Hanoi"` (NSO English yearbook romanisation)
* `"Đông Nam Bộ"` → `"South East"`
* `"Bia các loại (Lít)"` → `"Beer (all types) (Litres)"` (compositional)
* `"Số trường học, lớp học, giáo viên và học sinh mẫu giáo"` → `"Number of schools, classrooms, teachers and kindergarten students"` (Google fallback, cached)

---

## 5. Extending the glossary

If `translation_stats.misses > 0` after an offline run, the
unmatched strings are the long tail. To add them to the curated
glossary, edit `packages/ontology/glossary.py`. The dictionaries are
grouped by category (databases, variables, units, products, …) — pick
whichever matches the term's domain.

After editing, regenerate:

```bash
python -m packages.pipeline.cli build-ontology   # offline; new glossary entries take effect immediately
```

To regenerate **everything from scratch** including the online cache:

```bash
rm -f data/ontology/translation_cache.json
python -m packages.pipeline.cli build-ontology --online
```

---

## 6. Where to look in code

| If you want to… | Read |
|---|---|
| Curate / extend the VI→EN dictionary | [`packages/ontology/glossary.py`](packages/ontology/glossary.py) |
| Tweak the translation cascade | [`packages/ontology/translator.py`](packages/ontology/translator.py) |
| Change the tree shape or serialiser | [`packages/ontology/tree.py`](packages/ontology/tree.py) |
| Change CLI options | [`packages/pipeline/cli.py`](packages/pipeline/cli.py) |
| Run from CLI | `python -m packages.pipeline.cli build-ontology` (add `--online` for the long-tail Google fallback) |
| Tests | [`tests/test_ontology_tree.py`](tests/test_ontology_tree.py) |

---

## 7. Relationship to the existing `configs/ontology.yaml`

This new tree is **descriptive** — the *full* set of every NSO PX-Web
table, variable, and value, generated automatically from the raw
crawl. It is the source of truth about what data NSO publishes.

The pre-existing [`configs/ontology.yaml`](configs/ontology.yaml) is
**prescriptive** — a much smaller hand-curated layer that:

1. Tags each of the 12 databases with one or more "domain" labels
   (`population`, `employment`, …) for downstream UX.
2. Declares the 11 persona dimensions used by
   `packages/personagen/vn_persona_generator.py` plus the directed-
   acyclic edges of the cascaded Bayesian network.

Both files coexist. The descriptive tree (`tree.json`/`tree.yaml`)
documents what's available; the prescriptive YAML (`ontology.yaml`)
declares what we use.

---

## 8. Disclaimer

The Vietnamese labels in `tree.json` come straight from NSO's PX-Web
metadata exports (`*.metadata.json` sidecars under
`data/nso-gov-vn/raw/pxweb/vi/`). The English labels in the **glossary**
have been hand-checked against NSO's own English yearbooks; the
English labels in the **cache** were generated by Google Translate
and may contain phrasing differences from NSO's official English
terminology. For publication-ready labels of those long-tail entries,
override the cache JSON manually or extend the curated glossary.
