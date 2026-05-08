# personas-vn

Vietnamese-personas pipeline: a data scraper for the **Vietnamese National Statistics Office** (https://www.nso.gov.vn/ and https://pxweb.nso.gov.vn/), an **ontology framework** that normalises the scraped data, a **PGM-style synthetic persona generator** grounded in NSO statistics, a **NeMo Curator-compatible curation pipeline** (download → parse → extract → embed → reduce) for semantic exploration of the source documents and the generated personas, and a **Gradio visualizer** that surfaces all of it.

Four long-form pedagogical guides walk through the system end-to-end:

* [`DATAPROCESSING.md`](DATAPROCESSING.md) — curating the NSO PX-Web tabular data with the NeMo Curator-style staged pipeline (the same five-stage layout used by [tmquan/ViLA's `packages/datasites/anle/`](https://github.com/tmquan/ViLA/tree/main/packages/datasites/anle)), analysis of the curated data with ready-to-run figure scripts, and HuggingFace upload instructions.
* [`DATAANALYSIS.md`](DATAANALYSIS.md) / [`DATAANALYSIS.ipynb`](DATAANALYSIS.ipynb) — what the curated data actually says: 35 years of Vietnamese demographic + labour-force trends, urbanisation, education attainment, occupation structure, regional inequality, and an honest accounting of where the published statistical surface is too coarse for the persona PGM (with the fallbacks documented).
* [`DATAEXPLORATION.md`](DATAEXPLORATION.md) / [`DATAEXPLORATION.ipynb`](DATAEXPLORATION.ipynb) — interactive tour of the curator pipeline's terminal artefact: 502 PX-Web tables in a 2-D UMAP space, 12 ontology domains, 12 NSO databases, density-based HDBSCAN clusters, and Vietnamese-language semantic neighbour search.
* [`DATAVISUALIZATION.md`](DATAVISUALIZATION.md) / [`DATAVISUALIZATION.ipynb`](DATAVISUALIZATION.ipynb) — geographic atlas of the curated data over Vietnam's 63 admin-1 provinces (population, GRDP per capita, IIP, tourism, income, …) with the two archipelagos drawn per NSO cartographic convention.
* [`DATASYNTHESIS.md`](DATASYNTHESIS.md) / [`DATASYNTHESIS.ipynb`](DATASYNTHESIS.ipynb) — generating 100K bilingual Vietnamese personas with [NVIDIA SDG-PGMs](https://github.com/NVIDIA-NeMo/SDG-PGMs), optional LLM enrichment via Nemotron / Qwen / GPT-OSS, optional sentence-transformers / NIM embedding, and a runnable `parse → extract → embed → reduce` walkthrough on a 10 % subset.
* [`DATASETS.md`](DATASETS.md) — building the four `Nemotron-Personas-Vietnam` parquet datasets (3M / 300K × vi / en) with the schema, NSO data sources per column, the `vn-fullname-generator` integration, and the single-seed reproducibility contract.
* [`ONTOLOGY.md`](ONTOLOGY.md) — the bilingual NSO ontology tree (12 databases / 502 tables / 1 091 variables / 9 690 values), built from the raw PX-Web metadata with a curated VI→EN glossary plus optional Google Translate fallback; output as `data/ontology/tree.json` (full) and `data/ontology/tree.yaml` (summary).

The repo layout follows the monorepo philosophy of [tmquan/ViLA](https://github.com/tmquan/ViLA): a flat `packages/` namespace plus runnable entry points under `apps/`. The persona-generation core mirrors [NVIDIA-NeMo/SDG-PGMs](https://github.com/NVIDIA-NeMo/SDG-PGMs)' contract — `PGMGenerator` with `get_data` / `get_variables` / `get_edges` / `get_cpds` — re-implemented in pure numpy so 100K personas generate in < 5 seconds.

## Architecture

```
personas-vn/
├── apps/
│   └── visualizer/         # Gradio explorer
│                           # (Overview / Ontology / Datasets / Personas /
│                           #  Persona embeddings / Curator / Distributions)
├── packages/
│   ├── common/             # shared infra: config loader, HTTP client, logging, paths
│   ├── ontology/           # Pydantic schema + persona dimensions + NSO-category registry
│   ├── scraper/            # WordPress REST client + ontology mapper + offline fixture
│   ├── personagen/         # PGM-inspired cascaded sampler (vectorised numpy)
│   │                       # + persona-bio embedding & UMAP pipeline
│   ├── curator/            # NeMo Curator-style staged pipeline
│   │                       # (download → parse → extract → embed → reduce)
│   │                       # Runs locally OR via real NeMo Curator ProcessingStages
│   └── pipeline/           # orchestrator + CLI
├── configs/
│   ├── scraper.yaml        # legacy scraper config (used by personas-vn scrape)
│   ├── ontology.yaml       # statistical domains + persona-dimension DAG
│   ├── curator.yaml        # full-crawl curator pipeline (download → reduce)
│   └── visualizer.yaml     # Gradio host/port/theme/page-size
├── data/                   # (gitignored)
│   ├── raw/                #   legacy scrape JSONs
│   ├── ontology/           #   normalised Datasets
│   ├── personas/           #   personas.json + embedded.parquet + reduced.parquet
│   └── nso-gov-vn/         #   curator artefacts (raw/parsed/extracted/embedded/reduced)
└── tests/                  # 29 tests: ontology / scraper / personas / curator / e2e
```

### Data flow

```
nso.gov.vn  ──HTTP──►  packages.scraper.nso  ──┐
                                               ├─►  packages.scraper.mapper  ──►  data/ontology/datasets.json                                    │ 
                        (offline fixture)  ────┘                                       │
                                                                                       ▼
                                                                        packages.personas.pgm.VNPersonaGenerator
                                                                                       │
                                                                                       ▼
                                                                            data/personas/personas.json
                                                                                       │
                                                                                       ▼
                                                                              apps.visualizer (Gradio)
```

## Quickstart

> **zsh tip:** the snippets below contain inline `#` comments. zsh ignores
> them only if you've run `setopt interactivecomments` once (or once per
> shell). Otherwise, just paste the bare command lines (no `#` tails).

**1. Create the conda environment and install** (Python ≥ 3.10, named `pgm`):

```bash
conda create -n pgm python=3.11 -y
conda activate pgm
pip install -e ".[viz,curator,dev]"
```

The `[curator]` extra pulls in `torch`, `transformers`, `sentence-transformers`,
`umap-learn`, and `nemo-curator[text_cpu]` — about 3 GB. From here on every
command in this README assumes the `pgm` env is active (`conda activate pgm`).

**2a. Light path — legacy scrape + persona generation only:**

```bash
python -m packages.pipeline.cli run-all
```

**2b. Full path — NSO curator + 100K personas + persona embeddings:**

```bash
python -m packages.pipeline.cli curate
python -m packages.pipeline.cli generate --n 100000
python -m packages.pipeline.cli embed-personas --max-records 10000
```

**2c. Build the four `Nemotron-Personas-Vietnam` parquet datasets (3M / 300K × vi / en):**

```bash
# 3M / 300K full build, ~27 min
python -m packages.pipeline.cli build-nemotron --large-size 3000000 --small-size 300000

# smoke run, ~5s
python -m packages.pipeline.cli build-nemotron --large-size 10000 --small-size 1000
```

See [`DATASETS.md`](DATASETS.md) for schema, sources, and reproducibility details.

**2d. Build the bilingual NSO ontology tree:**

```bash
# offline, ~6s, ~76% EN coverage from curated glossary
python -m packages.pipeline.cli build-ontology

# +Google Translate for the long tail, ~90s, 100% coverage; cached
python -m packages.pipeline.cli build-ontology --online
```

Outputs land at `data/ontology/tree.json` (full) and `data/ontology/tree.yaml` (summary).
See [`ONTOLOGY.md`](ONTOLOGY.md) for schema, methodology, and translation pipeline details.

Approximate timings on an M-series Mac (no GPU): `curate` ≈ 4 minutes
end-to-end (cached after first run), `generate --n 100000` ≈ 5 seconds,
`embed-personas --max-records 10000` ≈ 2 minutes.

**3. Launch the visualizer** at <http://127.0.0.1:7860/>:

```bash
python -m apps.visualizer
```

It auto-loads any artefacts present under `data/` (datasets, personas,
curator parquets, persona embeddings); missing pieces just show empty
placeholder panels with instructions.

End-to-end example output (real numbers from a recent run on an M-series Mac, no GPU):

| Step                                                     | Records             | Wall time |
| -------------------------------------------------------- | ------------------- | --------- |
| `curate --only download` (full crawl)                    | 11,638 raw          | 3:24      |
| `curate --skip download`                                 | 11,464 → 3,000 emb. | 1:00      |
| `generate --n 100000`                                    | 100,000 personas    | 0:05      |
| `embed-personas --max-records 10000`                     | 10,000 embedded     | 2:14      |

(Each row is `python -m packages.pipeline.cli <step>` from the active `pgm` env.)

The first command writes:

* `data/raw/nso_*.json` — raw WordPress responses (one file per endpoint).
* `data/raw/cache/` — per-URL HTTP cache, so re-runs are near-instant.
* `data/ontology/datasets.json` — normalised `Dataset` records, one per NSO post.
* `data/ontology/manifest.json` — provenance: when the scrape ran, which language, did it use the offline fallback?
* `data/personas/personas.json` — a typed `PersonaBatch` of synthetic Vietnamese personas.

## Ontology

Two halves, both declared in [`configs/ontology.yaml`](configs/ontology.yaml) and validated by [`packages/ontology/registry.py`](packages/ontology/registry.py):

1. **Data ontology** — `Source`, `StatisticalDomain`, `Dataset`, `Indicator`. NSO's WordPress categories (English + Vietnamese, with diacritics) are mapped onto a stable list of 12 statistical domains: `population`, `employment`, `education`, `health`, `industry`, `agriculture`, `enterprises`, `investment`, `banking`, `national_accounts`, `geography`, `other`.
2. **Persona ontology** — 11 categorical dimensions arranged into a Bayesian-network-style DAG. The sampler walks the DAG in topological order, conditioning each variable on its parents:

   ```
   region ─┬─► urbanicity
           ├─► age_group ─┬─► sex ─────► marital_status
           │              ├──────────────► employment_status
           │              └─► education_level ─► occupation ─► industry_sector
           └─► ethnicity                                   └─► income_quintile
   ```

## Persona generator

The generator follows the SDG-PGMs contract — `PGMGenerator` with `get_data` / `get_variables` / `get_edges` / `get_cpds` / `get_postprocessing_steps` — but uses pure numpy so it installs cleanly without pgmpy or torch. See [`packages/personagen/base.py`](packages/personagen/base.py) for the abstract base and [`packages/personagen/vn_persona_generator.py`](packages/personagen/vn_persona_generator.py) for the concrete `VNPersonaGenerator`.

Distribution tables — region populations, urbanisation rates, age pyramids by region, marital status by age × sex, occupation by education × urbanicity, etc. — live in [`packages/personagen/distributions.py`](packages/personagen/distributions.py) with the NSO publications they were sourced from cited inline.

A typical persona looks like this:

```json
{
  "persona_id": "vn-000042-nguyen-thi-mai",
  "name": "Nguyễn Thị Mai",
  "region": "Red River Delta",
  "urbanicity": "urban",
  "age_group": "25-34",
  "age": 28,
  "sex": "female",
  "ethnicity": "Kinh",
  "marital_status": "married",
  "education_level": "tertiary",
  "employment_status": "employed",
  "occupation": "professionals",
  "industry_sector": "services_other",
  "income_quintile": "Q4",
  "bio": "28-year-old female from the Red River Delta (urban). Ethnicity: Kinh. ...",
  "sources": ["nso:categories", "nso:posts", "nso:tags"]
}
```

## Names and genders

Sex and full name are produced by **two independent samplers** — sex is
the third node in the PGM's topological walk, full name is grafted on
afterwards by a vectorised draw from `vn-fullname-generator`'s three
underlying pools. Both branches are seeded from the builder's master
RNG so the entire (sex, name) pair is bit-reproducible.

### Sex (NSO V02.02 marginal)

Sex is a **two-class categorical** (`Nam` / `Nữ` ↔ `male` / `female`)
sampled from the NSO V02.02 *Population by sex × urban-rural* table —
specifically the `Tổng số (Nghìn người)` row, latest year. NSO publishes
a stable ~50.5 % female / 49.5 % male split nationally; the PGM treats
sex as *independent of region and age* at the marginal level (the
underlying age × sex independence assumption is documented in
[`packages/personas/pgm/distributions.py:age_group_x_sex_counts`](packages/personas/pgm/distributions.py)
— at population scale the marginals dominate).

```225:236:packages/personas/pgm/distributions.py
SEXES_VI = ["Nam", "Nữ"]
SEXES_EN = {"Nam": "male", "Nữ": "female"}

...

def sex_marginal_counts() -> pd.DataFrame:
    """Sex → population, latest year, from V02.02."""
    df = _load_table("V02.02")
    df = df[df["Cách tính"] == "Tổng số (Nghìn người)"]
    df = df[df["Phân tổ"].isin(SEXES_VI)]
    year = _latest_year(df)
    df = df[df["Năm"].astype(str) == year]
    return df.rename(columns={"Phân tổ": "sex", "value": "count"})[["sex", "count"]]
```

Sex feeds *downstream* CPDs (marital_status, occupation, OCEAN
personality traits) — not just the name generator. That's why it's
sampled inside the PGM cascade, before the name pass.

### Vietnamese full names (`vn-fullname-generator`)

Vietnamese full names use the
[`vn-fullname-generator`](https://pypi.org/project/vn-fullname-generator/)
PyPI package, which itself bundles a curated Vietnamese name dictionary
from [`duyet/vietnamese-namedb`](https://github.com/duyet/vietnamese-namedb).
We don't call its `generate(gender)` function directly — that uses
Python's global `random` (not seedable per-call without clobbering
other libraries' RNG state) and runs at ~3 Python function calls per
name (too slow at 3 M-row scale).

Instead, [`packages/personas/datasets/names.py:generate_names`](packages/personas/datasets/names.py)
reads the upstream library's three name pools directly and samples
them with a vectorised `numpy.random.Generator`:

```42:62:packages/personas/datasets/names.py
@lru_cache(maxsize=1)
def _name_pools() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (firstnames, malenames, femalenames) as numpy arrays.

    Re-reads vn_fullname_generator's bundled ``.name`` text files. The
    arrays are clean (no empty strings, NFC-normalised) so the caller
    can do straight ``np.random.choice`` against them.
    """
    from vn_fullname_generator.generator import (
        femalenames,
        firstnames,
        malenames,
    )

    def _clean(seq: list[str]) -> np.ndarray:
        out = [unicodedata.normalize("NFC", s).strip() for s in seq]
        out = [s for s in out if s]
        return np.asarray(out, dtype=object)

    return _clean(firstnames), _clean(malenames), _clean(femalenames)
```

The three pools (terminology mirrored from the upstream library):

| Pool | Role | Example entries |
|---|---|---|
| `firstnames` | Vietnamese **surnames** (họ) — sampled regardless of sex | `Nguyễn`, `Trần`, `Lê`, `Phạm`, `Hoàng`, `Phan`, `Đoàn`, … |
| `malenames` | Male **given names** (tên) — used when `sex == "Nam"` | `Văn Minh`, `Đức Anh`, `Quốc Trường`, `Trung Việt`, … |
| `femalenames` | Female **given names** — used when `sex == "Nữ"` | `Thị Hoa`, `Bích Hồng`, `Thị Mai`, `Kim Thịnh`, … |

The actual draw is three vectorised `rng.integers(...)` calls plus a
mask-based `np.where(...)` to pick the right given-name pool by sex —
~100× faster than a per-row Python loop:

```96:104:packages/personas/datasets/names.py
    firstnames, malenames, femalenames = _name_pools()

    surnames = firstnames[rng.integers(0, len(firstnames), size=n)]
    male_given = malenames[rng.integers(0, len(malenames), size=n)]
    female_given = femalenames[rng.integers(0, len(femalenames), size=n)]

    is_male = sexes == "Nam"
    given = np.where(is_male, male_given, female_given)

    return np.asarray([f"{s} {g}" for s, g in zip(surnames, given)], dtype=object)
```

Output is the canonical `"<surname> <given>"` string with full
diacritics — `Nguyễn Thị Mai`, `Đoàn Quốc Trường`, `Phan Bích Hồng`.
We **don't romanise**, even in the English parquet: the same string
appears in both `…-vi.parquet` and `…-en.parquet`.

### Reproducibility

Both samplers are driven by independent `numpy.random.Generator` streams
spawned from the master `--seed` via `numpy.random.SeedSequence.spawn(7)`.
That's important because **adding a new narrative variant must not
perturb names**, and **changing chunk size must not perturb anything**.
The seven streams are documented in
[`DATASETS.md` § 4](DATASETS.md#4-reproducibility-contract):

```text
seed
 ├─ spawn[0] → structured personas (PGM, including sex)
 ├─ spawn[1] → uuids
 ├─ spawn[2] → province (region-conditional V02.01 weights)
 ├─ spawn[3] → vn-fullname-generator name pools  ← names
 ├─ spawn[4] → narrative variants (vi & en share picks)
 ├─ spawn[5] → small-subset stratified sample
 └─ spawn[6] → OCEAN traits (Big Five, age/sex-conditional priors)
```

Names live **inside** the narrative columns rather than as a top-level
`name` / `first_name` / `last_name` column — matching the convention
NVIDIA's published `Nemotron-Personas-Japan` and -US datasets use. So
`persona[i]`, `professional_persona[i]`, `cultural_background[i]`, etc.
all carry `Phan Bích Hồng` woven into the prose; there's no standalone
name column to join against.

## Scraper

The NSO site exposes a public WordPress REST API at `/wp-json/wp/v2/`. The scraper walks `posts`, `categories`, and `tags`, paginating via the `X-WP-TotalPages` header, capped at `fetch.max_pages` in [`configs/scraper.yaml`](configs/scraper.yaml).

Reliability features:

* **On-disk HTTP cache** keyed by canonical URL — re-runs are free.
* **Exponential-backoff retries** via `tenacity`, 5xx-only.
* **Graceful end-of-pages handling** — WP returns `400 rest_post_invalid_page_number` past the last page; we treat that as a clean stop.
* **Offline fallback** — if the live calls fail and `offline_fallback: true` is set, the scraper transparently loads a bundled NSO fixture so the rest of the pipeline always has data to process.
* **Per-post error isolation** — a single malformed post never sinks the whole mapping.

## Visualizer

Five tabs:

* **Overview** — pipeline status (last scrape, ontology version, dataset/persona counts) plus a bar chart of datasets per domain.
* **Ontology** — table of every statistical domain plus an interactive Plotly DAG of the persona dimensions.
* **Datasets** — searchable / domain-filterable table of every NSO dataset, with click-through links.
* **Personas** — generate / inspect / filter a persona batch, with region- and occupation-breakdown plots and CSV export.
* **Distributions** — peek at the underlying NSO-grounded count tables feeding the generator.

Launch:

```bash
python -m apps.visualizer                   # uses configs/visualizer.yaml
```

## Curator pipeline

The curator mirrors ViLA's `packages/datasites/anle/` five-stage layout
(`download → parse → extract → embed → reduce`) and is wire-compatible with
NeMo Curator's `ProcessingStage` / `Pipeline` API.

* **download** — full WP REST crawl across `posts`, `pages`, `categories`,
  `tags` for both `lang=en` and `lang=vi`. ~11K raw records, deterministic
  pagination via `X-WP-TotalPages`, on-disk per-URL HTTP cache.
* **parse** — strips HTML, runs cheap diacritic-based language detection,
  unifies posts + pages into a single `parsed.jsonl`.
* **extract** — maps each record onto an ontology domain via the same
  `OntologyRegistry` used by the legacy scraper, runs TF-IDF (1-2 grams,
  Vietnamese-friendly token pattern) for top-N keyword extraction.
* **embed** — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  on CPU (384-d), capped at `embed.max_records` for fast iteration. Bumps
  to `null` for the full corpus.
* **reduce** — UMAP → 2-D coordinates per record. Density-based
  HDBSCAN clustering is **opt-in** via `reduce.cluster: true` in
  `configs/curator.yaml` (default `false`); when on, an extra
  `cluster` column lands in `reduced.parquet` with `-1` reserved
  for low-density noise points. Written as parquet for the
  visualizer.

Two execution backends are available: `local` (default — sequential, what
the tests exercise) and `nemo_curator` (wraps each stage as a real
`ProcessingStage` and runs it through NeMo Curator's `InProcessExecutor`,
ready to swap in `XennaExecutor`/`RayDataExecutor` for distributed runs).

Run individual stages:

```bash
personas-vn curate                                # all 5 stages
personas-vn curate --only download                # just the crawl
personas-vn curate --skip download                # everything except crawl
personas-vn curate --backend nemo_curator         # use NeMo Curator's executor
```

## Persona embedding

After generating a `PersonaBatch`, embed the bios + project to 2-D so the
visualizer can render a semantic map. Two backends ship out of the box:

```bash
# 1. Local sentence-transformers (default — free, offline, 384-dim)
personas-vn generate --n 100000 --seed 42
personas-vn embed-personas --max-records 10000

# 2. NVIDIA NIM hosted embeddings (2048-dim; needs an API key)
export PERSONAS_VN_LLM_API_KEY=$NVIDIA_API_KEY  # or just set $NVIDIA_API_KEY
personas-vn embed-personas \
    --model nvidia/llama-3.2-nv-embedqa-1b-v2 \
    --max-records 10000
```

Verified NIM models (auto-dispatched when model id starts with `nvidia/`):

| Model | Dim | Notes |
| --- | --- | --- |
| `nvidia/llama-3.2-nv-embedqa-1b-v2` | 2048 | EmbedQA-tuned, default NIM choice |
| `nvidia/llama-nemotron-embed-1b-v2` | 2048 | Latest Nemotron embed |
| `nvidia/llama-3.2-nemoretriever-300m-embed-v1` | 2048 | Lightweight retriever |
| `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` | 2048 | Vision-language |
| `nvidia/nv-embedqa-e5-v5` / `nv-embedqa-mistral-7b-v2` | varies | Older EmbedQA family |

The same backend abstraction also powers the **curator** pipeline's
`EmbedStage` for NSO documents — set `embed.model` in
[`configs/curator.yaml`](configs/curator.yaml) and the right backend is
selected automatically. The visualizer's "Curator → semantic neighbour
search" reads that config too, so query vectors live in the same space
as the corpus.

Outputs:

* `data/personas/personas.json`     — typed `PersonaBatch`
* `data/personas/embedded.parquet`  — N-d vectors per sampled persona
* `data/personas/reduced.parquet`   — 2-D UMAP coords + cluster id

The visualizer's "Persona embeddings" tab plots the 2-D map with on-the-fly
recolouring by occupation / region / urbanicity / education / cluster /
etc. and hovers showing the bio for each point.

## CLI reference

```bash
personas-vn scrape          [--scraper-config PATH] [--ontology-config PATH]
personas-vn generate        [--n N] [--seed S] [--ontology-config PATH]
personas-vn run-all         [--n N] [--seed S]
personas-vn embed-personas  [--personas PATH] [--max-records N] [--min-cluster-size K]
                            [--model MODEL] [--backend auto|local|nim]
                            [--base-url URL] [--api-key-env VAR]
personas-vn curate          [--only STAGES...] [--skip STAGES...]
                            [--backend local|nemo_curator]
```

Or via the module:

```bash
python -m packages.pipeline.cli scrape
python -m packages.pipeline.cli generate --n 1000 --seed 42
python -m packages.pipeline.cli run-all  --n 500
```

## Configuration

Three YAML files under `configs/` — kept small and human-readable:

* [`configs/scraper.yaml`](configs/scraper.yaml) — base URL, language, pagination, retries, offline-fallback policy, list of WP endpoints.
* [`configs/ontology.yaml`](configs/ontology.yaml) — statistical-domain definitions (with English + Vietnamese aliases for category mapping) and persona-dimension DAG.
* [`configs/visualizer.yaml`](configs/visualizer.yaml) — Gradio host/port/theme/share + paths to ontology / persona / fixture files.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The test suite covers:

* **Ontology registry** — topological order, NSO-category mapping (incl. Vietnamese diacritics), cycle / dangling-parent detection.
* **Scraper mapper** — fixture shape, post → dataset mapping, HTML stripping, error isolation.
* **Persona generator** — determinism under a fixed seed, every field in its declared domain, marginal distribution sanity.
* **End-to-end pipeline** — `run_all` against the offline fixture, JSON round-trip through `PersonaBatch`.

No tests touch the network — the end-to-end test forces the scraper into offline-fallback mode.

## Dependencies

Core (always installed):

* `httpx` — HTTP client (sync, with retries via `tenacity`).
* `pydantic` ≥ 2 — schema validation, JSON serialisation.
* `pandas` + `numpy` — distribution tables, sampling.
* `PyYAML` — config files.
* `rich` — pretty CLI logging.

Visualizer extras (`pip install -e ".[viz]"`):

* `gradio` ≥ 4.40 — Web UI.
* `plotly` — charts and the DAG visual.
* `networkx` — graph utilities (reserved for future ontology-graph extensions).

Dev extras (`pip install -e ".[dev]"`): `pytest`, `pytest-cov`, `ruff`.

## Disclaimer

The personas this project generates are **synthetic**. They are sampled from aggregate NSO statistics and do not correspond to real individuals. The data scraped from `nso.gov.vn` belongs to the General Statistics Office of Vietnam; please respect their terms of use and apply reasonable rate limits when running the scraper at scale.
