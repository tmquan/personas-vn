# Data processing — curating NSO statistics with NeMo Curator

This document walks through how `personas-vn` turns raw Vietnamese
National Statistics Office (NSO) data into a clean, embedded, semantically
explorable corpus. It's pedagogical: every step has a *why* before the
*how*, every command is copy-pasteable, every figure can be regenerated
with one script.

If you only have 90 seconds, the whole pipeline is:

```bash
conda activate pgm                                       # one-time setup: see README
pip install -e ".[curator,dev]"                          # one-time
python -m packages.pipeline.cli curate                   # download → parse → extract → embed → reduce
python -m scripts.analyze_curated                        # high-level catalog figures
python -m scripts.deepdive_curated                       # deep-dive figures + tables
```

Outputs land under `data/nso-gov-vn/`. The five-stage layout is the same
one used by [tmquan/ViLA's `packages/datasites/anle/`](https://github.com/tmquan/ViLA/tree/main/packages/datasites/anle):

| Stage    | Reads                          | Writes                                      |
| -------- | ------------------------------ | ------------------------------------------- |
| download | NSO PX-Web (API + HTML)        | `raw/pxweb/{lang}/*.parquet` + `*.metadata.json` |
| parse    | `raw/pxweb/`                   | `parsed/parsed.jsonl`                       |
| extract  | `parsed/parsed.jsonl`          | `extracted/extracted.jsonl`                 |
| embed    | `extracted/extracted.jsonl`    | `embedded/embedded.parquet`                 |
| reduce   | `embedded/embedded.parquet`    | `reduced/reduced.parquet`                   |

The orchestrator (`packages/curator/pipeline.py`) chains them locally by
default, and can hand them off to a real NeMo Curator `Pipeline` +
`InProcessExecutor` / `XennaExecutor` / `RayActorPoolExecutor` with
`--backend nemo_curator`. The on-disk layout is identical either way.

What you end up with after a clean run:

- **502 PX-Web tables** across **12 NSO databases**
- **316,108 long-format data cells** in parquet
- 12 metadata JSONs + 1 master `_catalog.json` per language

---

## 1. Why two distinct sources

NSO publishes data through two completely different surfaces:

1. **`https://www.nso.gov.vn/`** — a WordPress site with news, press
   releases, and a few PDF downloads. Useful for *prose* (semantic
   search, document embeddings) but not for actual statistics.
2. **`https://pxweb.nso.gov.vn/`** — a [PX-Web v2](https://www.scb.se/en/services/statistical-programs-and-international-cooperation/px-web/)
   instance serving structured PC-Axis tables. **This is where the real
   numbers live.**

Earlier versions of `personas-vn` scraped only WordPress, which is why
the persona generator was originally hand-coded against generic priors.
The current curator targets PX-Web.

---

## 2. The PX-Web download stage — both surfaces

The PX-Web instance turns out to expose its data through **two different
mechanisms**, and we use both:

### 2.1 The clean JSON REST API

```
GET  /api/v1/{lang}/                   → list of databases
GET  /api/v1/{lang}/{db}/              → list of folders + tables
GET  /api/v1/{lang}/{db}/{table}.px    → table metadata (variables, value labels)
POST /api/v1/{lang}/{db}/{table}.px    → table data, JSON body {"query": [], "response": {"format": "json"}}
```

This is what we'd love to use for everything. **It works for 5 of NSO's
12 databases:**

| Matrix prefix | Database (VN)              | Database (EN)               | Tables |
| ------------- | -------------------------- | --------------------------- | -----: |
| V02           | Dân số và lao động         | Population and Employment   |    63  |
| V04           | Đầu tư                     | Investment                  |    25  |
| V05           | Doanh nghiệp               | Enterprise                  |    56  |
| V07           | Công nghiệp                | Industry                    |     9  |
| V13           | Giáo dục                   | Education                   |    34  |

187 tables total, fetched cleanly as JSON, parsed into a long-format
DataFrame by [`packages/scraper/pxweb.py:PxWebTable.to_records()`](packages/scraper/pxweb.py).

### 2.2 The hidden HTML-form databases

The other **7 databases are visible on the website but the JSON API
returns 404** — they exist in the IIS file system but aren't indexed by
the PX-Web REST service:

| Matrix prefix | Database (VN)                                   | Database (EN)                              | Tables |
| ------------- | ----------------------------------------------- | ------------------------------------------ | -----: |
| V01           | Đơn vị hành chính, đất đai và khí hậu          | Administrative Unit and Climate            |    18  |
| V03           | Tài khoản quốc gia                              | National Accounts and State budget         |    24  |
| V06           | Nông, lâm nghiệp và thủy sản                    | Agriculture, Forestry and Fishing          |    70  |
| V08           | Thương mại, giá cả                              | Trade, Price and Tourist                   |    72  |
| V12           | Vận tải và bưu điện                             | Transport, Postal Services & Telecom       |    24  |
| V14           | Y tế, văn hóa và đời sống                       | Health, Culture, Sport & Living standards  |    97  |
| V15           | Thống kê nước ngoài                             | International Statistics                   |    10  |

These 315 tables are **two-thirds of the total catalog** and include
some of NSO's most important publications (national accounts, state
budget, agriculture, prices, health, justice). Skipping them would
amount to ignoring more than half of what NSO publishes.

How do we get them? The legacy ASP.NET PX-Web web UI works just fine —
it uses a four-step form workflow that
[`packages/scraper/pxweb_html.py:PxWebHtmlClient`](packages/scraper/pxweb_html.py)
drives end-to-end:

```
1. GET   /pxweb/vi/{db}/{db}/{table}.px/?rxid=…
        → returns an ASP.NET variable-selection form with __VIEWSTATE
2. parse the form: every <select multiple> + every <input hidden>
3. POST  the same URL with all variable values selected + ButtonViewTable
        → server stores the selection in IIS session state
4. GET   /pxweb/vi/{db}/{db}/{table}.px/table/tableViewLayout1/
              ?rxid=…&downloadfile=FileTypeExcelX
        → an .xlsx download (Vietnamese-encoding-clean)
5. xlsx → long-format records via xlsx_to_records()
```

We use **xlsx** rather than CSV because NSO's CSV export goes through a
CP1258 → ASCII round-trip on the server that loses several Vietnamese
diacritics; the xlsx export preserves them perfectly.

The xlsx parser handles three NSO matrix layouts:

| Layout            | Example          | Shape                                                   |
| ----------------- | ---------------- | ------------------------------------------------------- |
| 2-D wide          | V02.43 (most)    | rows = first dim, cols = second dim, inner = numeric    |
| 3-D with sections | V06.36, V06.37   | col 0 = section header (sparse), col 1 = sub-label, header row[2:] = third dim |
| 1-D single-column | V14.62, V14.66+  | one value per labelled row, no header row               |

### 2.3 What the merged crawl looks like

```bash
personas-vn curate --only download
```

Configured in [`configs/curator.yaml`](configs/curator.yaml):

```yaml
download:
  source: "pxweb"
  base_url: "https://pxweb.nso.gov.vn"
  api_path: "/api/v1"
  langs: ["vi"]
  include_html_dbs: true   # walk the 7 hidden DBs via the HTML form
```

The download stage runs the JSON-API path first (fast — 2 minutes for
187 tables) then the HTML-form path (slower because each table needs a
fresh form-POST session — 4-5 minutes for 315 tables). On-disk layout:

```
data/nso-gov-vn/
├── manifest.json                                       # provenance
└── raw/
    ├── _cache/                                         # per-URL HTTP cache
    └── pxweb/vi/
        ├── _catalog.json                               # one row per matrix
        # API-path tables — single VI-prefix in the slug
        ├── Dan-so-va-lao-dong__V02.01.px.parquet
        ├── Dan-so-va-lao-dong__V02.43.px.parquet
        ├── ... (185 more)
        # HTML-path tables — doubled prefix (the path is {db}/{db}/...)
        ├── Tai-khoan-quoc-gia__Tai-khoan-quoc-gia__V03.01.px.parquet
        ├── Y-te-van-hoa-va-doi-song__Y-te-van-hoa-va-doi-song__V14.43.px.parquet
        ├── ... (313 more)
```

Each parquet is paired with a `*.metadata.json` recording the variable
codes, the data source (`api` or `html_form`), the original PX-Web title,
and the cell count.

### 2.4 What we got

```
=== summary ===
  source: pxweb
  total tables:    502
  total cells:     316,108

  by access path:
    JSON API:       187 tables   ~130K cells
    HTML form:      315 tables   ~186K cells
```

| Database (VN)                            | Tables | Cells   |
| ---------------------------------------- | -----: | ------: |
| Nông, lâm nghiệp và thủy sản (V06)       |     70 | 76,541  |
| Giáo dục (V13)                           |     34 | 57,742  |
| Dân số và lao động (V02)                 |     63 | 48,203  |
| Doanh nghiệp (V05)                       |     56 | 37,391  |
| Y tế, văn hóa và đời sống (V14)          |     97 | 23,934  |
| Thương mại, giá cả (V08)                 |     72 | 23,213  |
| Vận tải và bưu điện (V12)                |     24 | 17,380  |
| Thống kê nước ngoài (V15)                |     10 | 11,502  |
| Công nghiệp (V07)                        |      9 |  5,532  |
| Đầu tư (V04)                             |     25 |  5,522  |
| Đơn vị hành chính, đất đai và khí hậu (V01) | 18 |  4,705  |
| Tài khoản quốc gia (V03)                 |     24 |  4,443  |
| **Total**                                | **502**| **316,108** |

### 2.5 Reliability features

* **On-disk per-URL cache** keyed by canonical query — re-runs of
  `python -m packages.pipeline.cli curate` are near-instant after the first run.
* **Exponential-backoff retries** via `tenacity`, 5xx-only.
* **Content-type sanity check** — if the upstream returns HTML where
  JSON is expected (Cloudflare challenge / network filter) we raise
  `HttpError` cleanly instead of crashing on `r.json()`.
* **Per-table error isolation** — a single malformed PC-Axis matrix
  never sinks the rest of the crawl.
* **Fresh client per HTML-form fetch** — PX-Web's IIS session is sticky
  and occasionally returns the wrong selection if we reuse one across
  tables; using a new `httpx.Client` per table sidesteps that.
* **Three xlsx layouts handled** — see §2.2 above; the parser falls
  through `2-D → 3-D-with-sections → single-column` automatically.

---

## 3. Ontology — what the 502 tables map onto

The schema config at [`configs/ontology.yaml`](configs/ontology.yaml)
has three layers, designed so any persona attribute can be traced back
to a specific NSO PX-Web matrix in one lookup.

### 3.1 Layer 1 — PX-Web databases (12 first-class entities)

Each database carries:

* **`id`** — short stable slug we use internally.
* **`name_vi`** / **`name_en`** — canonical NSO labels, both languages.
* **`matrix_prefix`** — the V-prefix shared by every matrix in the DB
  (V01, V02, V03, V04, V05, V06, V07, V08, V12, V13, V14, V15).
* **`access`** — `json_api` (5 DBs) or `html_form` (7 DBs).
* **`n_tables`** — anti-regression count for the catalogue walk.
* **`domain` / `domains`** — one or many statistical-domain ids
  (Layer 2). Several DBs span multiple domains (V03 holds national
  accounts + banking + state budget; V08 holds trade + prices +
  tourism; V14 holds health + society + environment + justice +
  living standards).

### 3.2 Layer 2 — Statistical domains (21 buckets)

Language-neutral buckets used by the visualizer's facets and by the
NSO-category mapper for the legacy WordPress source. New since the
last revision: `trade`, `prices`, `tourism`, `transport`, `society`,
`living_standards`, `justice`, `environment`, `international` — all
needed to cleanly cover the 7 newly-crawled HTML-form databases.

### 3.3 Layer 3 — Persona dimensions (with PX-Web table provenance)

Every persona dimension now lists the specific NSO matrix ids whose
distributions feed its conditional probability table:

```yaml
- id: "occupation"
  domain: "employment"
  parents: ["education_level"]
  cardinality: 10
  sources: ["V02.43"]   # Employed by occupation (NSO ISCO collapse)

- id: "industry_sector"
  domain: "industry"
  parents: ["occupation"]
  cardinality: 21
  sources: ["V02.42"]   # Employed by sector (kinds of economic activity)
```

Two dimensions still have empty `sources` lists (`ethnicity`,
`income_quintile`) — those joints aren't published in PX-Web yet, so the
generator falls back to a hand-coded prior (the 2019 Census shape for
ethnicity, NSO HLSS quintiles for income). When NSO publishes them, we
fill in the source list and the loader in
[`packages/personagen/distributions_pxweb.py`](packages/personagen/distributions_pxweb.py)
swaps the synthetic prior for the real table without changes elsewhere.

The end result is a clean provenance trail: a sampled persona's
`sources` field automatically lists the 6-10 PX-Web matrices its
attributes were grounded in.

---

## 4. Parse stage

### 4.1 Why a parse stage exists at all

The PX-Web download produces structured numeric tables — there's no HTML
to clean, no language to detect. The parse stage is a no-op for the
PX-Web backend.

It does come alive for the **legacy WordPress backend**: running the
curator with `download.source: "wordpress"` parses titles + excerpts + body HTML
into a normalised `parsed.jsonl` (used by the curator's text-embedding
demo). We keep the stage in the cascade because:

1. NeMo Curator's `Pipeline` API expects a chain of `ProcessingStage`s —
   having a no-op slot makes the WordPress backend a drop-in replacement.
2. Future cross-source dedup (e.g. mapping a province name written as
   "Hà Nội" in some tables and "Hà nội" in others to a canonical
   spelling) will live here.

---

## 5. Extract stage

### 5.1 What it does

For each parsed record (currently the WordPress backend only) we want:

* The **stable ontology domain id** — `population`, `employment`,
  `industry`, etc. — so downstream tools can filter without doing fuzzy
  string matching every time.
* A short list of **TF-IDF top-N keywords** — useful for the visualizer
  (rendered next to titles) and for retrieval-augmented prompts when we
  enrich personas later.

### 5.2 Domain mapping

`packages.ontology.registry.OntologyRegistry.map_nso_category()` does
the work. It normalises the input (strips Vietnamese diacritics +
lowercases + collapses non-alnum runs), then matches against the alias
list declared in [`configs/ontology.yaml`](configs/ontology.yaml). A few
examples:

| Input                         | Domain id           |
| ----------------------------- | ------------------- |
| `Dân số`                      | `population`        |
| `Population`                  | `population`        |
| `Lao động`                    | `employment`        |
| `Đầu tư và Xây dựng`           | `investment`        |
| `Tài khoản quốc gia`           | `national_accounts` |
| `Y tế`                        | `health`            |
| `(unmatched)`                 | `other`             |

---

## 6. Embed stage

### 6.1 Pluggable backends

The embed stage uses
[`packages.personas.embed.backends.make_embedding_backend(model, ...)`](packages/personagen/embed_backends.py).
Default stays `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(free, offline, 384-dim) so the curator runs without an API key. Set
`embed.model: nvidia/...` in the config and the backend automatically
flips to NVIDIA NIM hosted embeddings.

| Model id                                              | Backend | Dim   | Notes                            |
| ----------------------------------------------------- | ------- | ----- | -------------------------------- |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (default) | local | 384 | Multilingual SBERT  |
| `nvidia/llama-3.2-nv-embedqa-1b-v2`                   | NIM     | 2048 | EmbedQA-tuned                     |
| `nvidia/llama-nemotron-embed-1b-v2`                   | NIM     | 2048 | Latest Nemotron embed             |
| `nvidia/llama-3.2-nemoretriever-300m-embed-v1`        | NIM     | 2048 | Lightweight retriever             |

NIM models need `PERSONAS_VN_LLM_API_KEY` (or `NVIDIA_API_KEY`) in the
environment.

### 6.2 What gets embedded

For PX-Web the curator pipeline doesn't project tabular cells back into
prose — that's out of scope (the persona generator's bio is the canonical
text artefact). The embed stage is mostly there for the *WordPress*
backend, which has actual prose to embed. For the `pxweb` source the
default `n=0` is correct; downstream tabs in the visualizer use the
parquet files directly without embeddings.

---

## 7. Reduce stage

UMAP (default) → 2-D coordinates per record. Density-based HDBSCAN
clustering is **opt-in** via `reduce.cluster: true` in
`configs/curator.yaml` (default `false`); when on, the stage emits
an extra `cluster` column with `-1` reserved for low-density noise
points. The default skips clustering because most consumers (the
visualiser's domain / database / language colourings, the analysis
notebooks' raw-column scatter plots) don't read cluster ids.

```yaml
reduce:
  algorithm: "umap"     # umap | pca | tsne
  n_components: 2
  n_neighbors: 15
  min_dist: 0.1
  metric: "cosine"
```

---

## 8. End-to-end run

```bash
# 1. Download (≈ 2 min API + 4 min HTML form on a residential connection)
personas-vn curate --only download

# 2. The downstream stages are PX-Web no-ops; run them just to confirm
#    the pipeline manifest is well-formed.
personas-vn curate --skip download
```

Pipeline manifest snippet:

```json
{
  "dataset": "nso-gov-vn",
  "backend": "local",
  "stages": {
    "download": {
      "status": "ok",
      "source":  "pxweb",
      "total_tables": 502,
      "total_cells":  316108,
      "langs": {
        "vi": {
          "n_tables": 502,
          "n_cells":  316108,
          "n_tables_api":  187,
          "n_tables_html": 315,
          "catalog": "data/nso-gov-vn/raw/pxweb/vi/_catalog.json"
        }
      }
    }
  }
}
```

---

## 9. Analysis

Two analysis scripts walk through the data:

```bash
# 1. Catalogue overview — figures referenced inline in this doc
python -m scripts.analyze_curated

# 2. Deep-dive — the 10 figures referenced in DATAANALYSIS.md
python -m scripts.deepdive_curated
```

For interactive deep-dive analysis with NVIDIA-styled Plotly figures
(white background, NVIDIA Green `#76B900` data series, NVIDIA Sans
typography), open
[`DATAANALYSIS.ipynb`](DATAANALYSIS.ipynb) — the notebook produces a
matched static markdown rendering at
[`DATAANALYSIS.md`](DATAANALYSIS.md) along with PNG snapshots under
`docs/figures/analysis/`.

### 9.1 Coverage by database

```python
catalog.groupby("database").size()
```

Most of the persona pipeline draws from V02 (Population & Labour,
63 tables) and V13 (Education, 34 tables); the other 10 databases
provide context for the visualizer + the Curator semantic-search tab.

### 9.2 Cells-per-table histogram

Median is around 200 cells; the long tail (V02.03-07, V14.16, V13.16)
are the rich province × year × sex × urbanicity matrices the persona
generator condition-samples on.

### 9.3 Variable frequency

Tells you which dimensions are *actually* available across the
catalogue. `Năm` (year) and `Tỉnh, thành phố` / `Địa phương` (province)
appear in nearly every table; `Vùng` (macro-region), `Phân tổ`
(breakdown), `Cách tính` (calculation method), `Nhóm tuổi` (age group)
in many.

### 9.4 Per-table summary CSV

`docs/figures/curated/06_pxweb_summary.csv` — one row per matrix with
id, database, title, n_variables, n_cells, comma-joined variable
codes. Easy to grep when you're trying to find "the table with both age
group and urbanicity".

---

## 10. Uploading to HuggingFace

We follow the same upload pattern that NVIDIA SDG-PGMs uses for the
[Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA)
dataset.

### 10.1 One-time setup

```bash
pip install huggingface_hub datasets
huggingface-cli login
```

### 10.2 Push

```bash
export HF_TOKEN=hf_xxx
python -m scripts.upload_to_hf \
    --repo your-user/personas-vn-pxweb \
    --push curated
```

What that does (see [`scripts/upload_to_hf.py`](scripts/upload_to_hf.py)):

1. Materialises a 502-row DataFrame: `(table_id, database, title,
   variables, n_cells, parquet_path)` — this is the indexable
   dataset that drives HuggingFace's Dataset Viewer.
2. `Dataset.from_pandas(df).push_to_hub(repo_id, ...)` uploads the
   index as `train.parquet` at the repo root.
3. `HfApi.upload_folder(...)` copies every per-table parquet +
   metadata JSON into `pxweb/vi/`.
4. Uploads the manifest + a generated `README.md` with YAML frontmatter
   declaring `language: [vi, en]`, `license: cc-by-nc-4.0`, relevant tags.

Resulting repo structure:

```
huggingface.co/datasets/your-user/personas-vn-pxweb/
├── README.md
├── train.parquet                              # 502-row index
├── manifest.json
└── pxweb/vi/
    ├── Dan-so-va-lao-dong__V02.01.px.parquet
    ├── ...
```

### 10.3 Citation policy

The downloaded data belongs to the General Statistics Office of Vietnam.
Always credit them as the original publisher.

---

## 11. Working with NeMo Curator's real backend

By default `personas-vn curate` runs the stages sequentially in the
local Python process. To exercise NVIDIA NeMo Curator's actual
`ProcessingStage` + `Pipeline` machinery (which scales out to Ray
clusters in production), pass `--backend nemo_curator`:

```bash
personas-vn curate --backend nemo_curator
```

The on-disk artefacts are bit-identical between the two backends — the
local executor exists to keep `python -m packages.pipeline.cli curate`
and the unit tests cheap; the NeMo Curator executor exists for when the
same pipeline runs across a cluster.

---

## 12. Configuration cheat sheet

[`configs/curator.yaml`](configs/curator.yaml) — every knob:

| Section            | Key                       | What                                    |
| ------------------ | ------------------------- | --------------------------------------- |
| `dataset`          | `name`, `root`            | Where outputs land                      |
| `download`         | `source`                  | `pxweb` (default) or `wordpress`        |
| `download`         | `langs`, `only_dbs`, `skip_dbs` | Catalog filters                  |
| `download`         | `include_html_dbs`        | Walk the 7 hidden HTML-form DBs?        |
| `download`         | `request_timeout_s`, `retries`, `retry_backoff_s`, `delay_between_requests_s` | Reliability      |
| `extract`          | `top_keywords`, `vectorizer`, `ngram_range`, `max_df`, `min_df` | TF-IDF |
| `embed`            | `model`, `backend`        | Local SBERT vs NIM (auto)               |
| `embed`            | `base_url`, `api_key_env`, `truncate`, `request_delay_s` | NIM-only       |
| `embed`            | `max_records`             | Cap for fast iteration                  |
| `reduce`           | `algorithm`, `n_components`, `n_neighbors`, `min_dist`, `metric` | UMAP / PCA / TSNE |

---

## 13. Where to look in code

| If you want to...                   | Read                                          |
| ----------------------------------- | --------------------------------------------- |
| Understand the JSON-API client      | [`packages/scraper/pxweb.py`](packages/scraper/pxweb.py) |
| Understand the HTML-form scraper    | [`packages/scraper/pxweb_html.py`](packages/scraper/pxweb_html.py) |
| Add a new stage / change a stage    | [`packages/curator/stages.py`](packages/curator/stages.py) |
| Plug in a different executor        | [`packages/curator/pipeline.py`](packages/curator/pipeline.py) |
| Add / rename a statistical domain   | [`configs/ontology.yaml`](configs/ontology.yaml) |
| Trace a persona attribute back to NSO | [`configs/ontology.yaml`](configs/ontology.yaml) — `persona_dimensions[*].sources` |
| Add a new embedding model           | [`packages/personagen/embed_backends.py`](packages/personagen/embed_backends.py) |
| Re-create catalogue figures         | [`scripts/analyze_curated.py`](scripts/analyze_curated.py) |
| Re-create deep-dive figures         | [`scripts/deepdive_curated.py`](scripts/deepdive_curated.py) |
| Run the interactive deep-dive       | [`DATAANALYSIS.ipynb`](DATAANALYSIS.ipynb) |
| Push to HuggingFace                 | [`scripts/upload_to_hf.py`](scripts/upload_to_hf.py) |

See [`DATASYNTHESIS.md`](DATASYNTHESIS.md) for the matching walkthrough
of the synthetic-persona side of the pipeline,
[`DATAANALYSIS.md`](DATAANALYSIS.md) / [`DATAANALYSIS.ipynb`](DATAANALYSIS.ipynb)
for what the data actually says about Vietnam,
[`DATAEXPLORATION.md`](DATAEXPLORATION.md) /
[`DATAEXPLORATION.ipynb`](DATAEXPLORATION.ipynb) for the
curator-pipeline terminal-artefact tour, and
[`DATAVISUALIZATION.md`](DATAVISUALIZATION.md) /
[`DATAVISUALIZATION.ipynb`](DATAVISUALIZATION.ipynb) for the
geographic atlas.
