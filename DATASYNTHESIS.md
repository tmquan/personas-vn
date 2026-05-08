# Data synthesis — generating Vietnamese personas with SDG-PGMs

This document walks through how `personas-vn` turns the curated NSO
PX-Web tables (see [`DATAPROCESSING.md`](DATAPROCESSING.md)) into
**100,000 bilingual synthetic Vietnamese personas**. It's pedagogical:
every step has a *why*, every command runs, every figure regenerates
with one script.

90-second version:

```bash
conda activate pgm                                       # one-time setup: see README
pip install -e ".[curator,dev]"                          # one-time
python -m packages.pipeline.cli curate                   # download the PX-Web tables
python -m packages.pipeline.cli generate --n 100000      # 100K personas in ~13 seconds
python -m packages.pipeline.cli embed-personas --max-records 10000
                                                          # optional: 2-D semantic map
python -m packages.pipeline.cli enrich-personas --max-records 200 \
    --model nvidia/nemotron-3-super-120b-a12b             # optional: LLM-rich bilingual bios
python -m scripts._build_datasynthesis_nb                 # rebuild the notebook
jupyter nbconvert --to notebook --execute DATASYNTHESIS.ipynb --inplace
```

Outputs land under `data/personas/`.

| File                                   | What                                               |
| -------------------------------------- | -------------------------------------------------- |
| `data/personas/personas.json`          | Typed `PersonaBatch` (100K personas)               |
| `data/personas/personas_enriched.json` | Same batch with LLM-rewritten bio_vi / bio_en      |
| `data/personas/embedded.parquet`       | N-d float32 vector per sampled persona             |
| `data/personas/reduced.parquet`        | 2-D UMAP coordinates + cluster labels              |

---

## 1. Why SDG-PGMs

The persona generator is a direct subclass of NVIDIA SDG-PGMs'
[`PGMGenerator`](https://github.com/NVIDIA-NeMo/SDG-PGMs#-architecture)
(installed via `pip install git+https://github.com/NVIDIA-NeMo/SDG-PGMs.git`),
not a hand-rolled re-implementation. Three reasons:

1. **Faithful contract.** The same template-method lifecycle
   (`get_data` / `get_variables` / `get_edges` / `get_cpds` /
   `get_postprocessing_steps` / `get_mappings` / `get_latents` /
   `fill_na_and_gen_cpd`) means anyone familiar with SDG-PGMs (or with
   the Nemotron-Personas-USA pipeline) reads our generator instantly.
2. **Cascaded networks for free.** SDG-PGMs ships exactly the
   "multi-stage Bayesian network with deterministic post-processing
   between stages" pattern we need.
3. **Real `pgmpy.TabularCPD`s.** Probability tables go through
   `pgmpy.factors.discrete.TabularCPD`, a battle-tested
   implementation we don't have to reinvent.

The implementation file: [`packages/personagen/vn_persona_generator.py`](packages/personagen/vn_persona_generator.py).

---

## 2. The persona PGM — variables and edges

### 2.1 What a persona is

```python
class Persona(BaseModel):
    persona_id: str            # vn-NNNNNN-slug
    name: str                  # Vietnamese: "Nguyễn Thị Mai"

    # Geography
    region: str                # Vietnamese (canonical NSO label)
    region_en: str             # English mirror
    urbanicity: str            # "Thành thị" / "Nông thôn"
    urbanicity_en: str

    # Demographics
    age_group: str; age: int
    sex: str; sex_en: str
    ethnicity: str; ethnicity_en: str
    marital_status: str; marital_status_en: str

    # Socio-economic
    education_level: str; education_level_en: str
    employment_status: str; employment_status_en: str
    occupation: str; occupation_en: str
    industry_sector: str; industry_sector_en: str
    income_quintile: str; income_quintile_en: str

    # Narrative + provenance
    bio_vi: str | None
    bio_en: str | None
    enriched: bool             # True once an LLM has rewritten the bios
    sources: list[str]         # ["pxweb:V02.43", "pxweb:V02.44", ...]
```

Every categorical field is bilingual: the canonical NSO Vietnamese
label, plus an English mirror. That means downstream tools never need
to maintain their own translation table — the persona itself carries
both.

### 2.2 The cascaded network

Three stages, eleven variables, eleven CPDs. Edges declared in
[`get_edges()`](packages/personagen/vn_persona_generator.py):

```
Stage 1 — geography
    region ─► urbanicity

Stage 2 — demographics
    region ─► age_group ─► sex ─► marital_status
    region ─► ethnicity                ▲
    age_group ─────────────────────────┘

Stage 3 — socio-economic
    age_group   ─┐
    urbanicity  ─┴► education_level ─► occupation ─► industry_sector
    age_group   ─┐
    sex         ─┴► employment_status
    education_level ─► income_quintile
```

Each stage is sampled in topological order. SDG-PGMs handles this for
us via `BayesianNetwork.simulate(...)` per stage, then runs the user's
post-processing steps before moving to the next stage. By the end
every persona has all eleven categorical attributes plus an integer
age and bilingual templated bios.

### 2.3 The data bank

[`packages/personagen/vn_persona_generator.py`](packages/personagen/vn_persona_generator.py) defines the canonical SDG-PGMs `BaseModel + TarFileMixin`
data bank:

```python
class VNPersonaData(BaseModel, TarFileMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    region_counts:               pd.DataFrame
    region_x_urbanicity:         pd.DataFrame
    age_group_x_region:          pd.DataFrame
    sex_x_age_group:             pd.DataFrame
    ethnicity_x_region:          pd.DataFrame
    marital_x_age_x_sex:         pd.DataFrame
    education_x_age_x_urbanicity:pd.DataFrame
    employment_x_age_x_sex:      pd.DataFrame
    occupation_x_education:      pd.DataFrame
    industry_x_occupation:       pd.DataFrame
    income_x_education:          pd.DataFrame
```

Each DataFrame is a count table (`columns: dimensions...; count`) with
`pd.Categorical` dtypes — the format SDG-PGMs'
`fill_na_and_gen_cpd(counts, variable_name, evidence)` consumes. Tar
serialisation comes for free from `TarFileMixin`, so a frozen data bank
can be shipped to HuggingFace and replayed elsewhere with
`VNPersonaData.from_tarfile(path, _TAR_MAPPINGS)`.

### 2.4 Where the numbers come from

Every count table is built from real NSO PX-Web parquets in
`data/nso-gov-vn/raw/pxweb/vi/`. The mapping is in
[`packages/personagen/distributions_pxweb.py`](packages/personagen/distributions_pxweb.py)
and one helper per variable:

| Persona variable    | Source PX-Web table | Notes                                        |
| ------------------- | ------------------- | -------------------------------------------- |
| `region`            | V02.01              | Population by province → rolled up to 6 macro-regions |
| `urbanicity`        | V02.02              | National urban / rural marginal              |
| `age_group`         | V02.41              | 5-year buckets, 15–64 + 65+                  |
| `sex`               | V02.02              | Sex marginal (NSO publishes ~50/50)          |
| `ethnicity`         | (synthetic)         | NSO doesn't publish ethnicity-by-region in PX-Web; we use the 2019-Census shape |
| `marital_status`    | (synthetic)         | NSO publishes marriage-rate aggregates only — synthetic age × sex priors |
| `education_level`   | V02.54 + tilts      | Trained-labour by qualification + age/urbanicity tilts |
| `employment_status` | V02.44              | Employment-status marginal, broadcast to age × sex |
| `occupation`        | V02.43              | ISCO collapsed (10-class)                    |
| `industry_sector`   | V02.42              | ISIC sector marginal × occupation            |
| `income_quintile`   | (synthetic)         | NSO HLSS quintiles by education (not in PX-Web yet) |

When NSO later publishes the joint cross-tabs (e.g. occupation by age
by region), the loader can swap in the real table without changing
anything in the generator.

---

## 3. Generating personas

### 3.1 The CLI

```bash
personas-vn generate --n 100000 --seed 11
```

Wall time on an M-series Mac (no GPU): **~13 seconds for 100K
personas**. The cascaded sampler is fully vectorised — every variable in
every stage is sampled with one `BayesianNetwork.simulate(N)` call, then
post-processing runs once on the resulting DataFrame.

### 3.2 The post-processing chain

Three small steps run between stages:

1. After stage 2 — `_derive_age`: turns the `age_group` enum into an
   integer age via the bracket midpoint plus a small ±2 jitter.
2. After stage 3 — `_nullify_for_non_workers`: students /
   out-of-labour-force personas have their `occupation` and
   `industry_sector` set to `Không áp dụng` ("Not applicable") because
   the cascade samples occupation independently of employment status
   (NSO's V02.43 conditions occupation on education only).
3. After stage 3 — `_add_name`: vectorised draw from a Vietnamese
   family-name + sex-conditional middle/given-name pool; produces a
   stable `persona_id` slug.
4. After stage 3 — `_add_bilingual_bio`: maps every categorical column
   to its English mirror via `get_mappings()` lookup tables, then
   writes templated `bio_vi` and `bio_en`.

Both the structured columns and the bios are deterministic given a
seed; re-running with the same seed reproduces the same 100K rows.

### 3.3 What you actually get

A snippet from the 100K batch (showing one row from each region):

```
vn-000000-le-trung-vinh
  region=Đồng bằng sông Hồng | Red River Delta
  age=25 (25-29)  sex=Nam | male
  edu=Trung cấp | intermediate
  occ=Thợ lắp ráp và vận hành máy móc, thiết bị
  bio_vi: Một người Nam 25 tuổi, thuộc dân tộc Kinh, sống ở Nông thôn
          thuộc Đồng bằng sông Hồng. Tình trạng hôn nhân: Chưa kết hôn.
          Trình độ chuyên môn kỹ thuật: Trung cấp. Vị thế việc làm:
          Làm công ăn lương; nghề nghiệp: Thợ lắp ráp và vận hành máy
          móc, thiết bị; ngành: Công nghiệp chế biến, chế tạo; nhóm
          thu nhập: Q3.
  bio_en: A 25-year-old male of Kinh ethnicity living in a rural area
          of the Red River Delta. Marital status: single. Education:
          intermediate. Employment status: wage_employee; occupation:
          ...
```

---

## 4. LLM enrichment (optional)

Templated bios are faithful but read mechanically. The optional
`personas-vn enrich-personas` pass rewrites a (configurable) subset of
personas with an LLM-written Vietnamese narrative + parallel English
translation, both grounded in the same structured fields.

### 4.1 Provider-agnostic by design

The enrichment client is OpenAI-compatible
([`packages/personagen/llm_enrich.py`](packages/personagen/llm_enrich.py)), so any of the following works:

| Provider                    | Model id (example)                            | Endpoint                                         |
| --------------------------- | --------------------------------------------- | ------------------------------------------------ |
| **NVIDIA NIM**              | `nvidia/nemotron-3-super-120b-a12b`           | `https://integrate.api.nvidia.com/v1`            |
|                             | `nvidia/llama-3.3-nemotron-super-49b-v1`      | (same)                                           |
| **OpenRouter / Together**   | `openai/gpt-oss-120b`                         | `https://openrouter.ai/api/v1`                   |
|                             | `qwen/qwen3-235b-a22b`                        | (same)                                           |
| **Local vLLM / Ollama**     | any                                           | `http://localhost:8000/v1`                       |

```bash
export PERSONAS_VN_LLM_API_KEY=$NVIDIA_API_KEY    # or another provider's key

personas-vn enrich-personas \
    --max-records 200 \
    --model nvidia/nemotron-3-super-120b-a12b \
    --base-url https://integrate.api.nvidia.com/v1
```

### 4.2 What the prompt looks like

System (Vietnamese):

> Bạn là một trợ lý chuyên viết tiểu sử nhân vật ảo dựa trên dữ liệu
> thống kê dân số chính thức của Việt Nam. Mỗi tiểu sử cần ngắn gọn
> (3-5 câu), mang đậm văn hoá Việt Nam, và phản ánh trung thực các
> thuộc tính nhân khẩu học - kinh tế xã hội đã cung cấp. Bạn LUÔN trả
> lời bằng JSON hợp lệ duy nhất với hai trường: `{"bio_vi": ..., "bio_en": ...}`.

User: a structured listing of every persona attribute (Vietnamese label
+ English mirror) — see `_build_user_prompt` in `llm_enrich.py`.

### 4.3 Sample output

Real result from `nvidia/nemotron-3-super-120b-a12b` against a 47-year-old
divorced wage-earner in the Red River Delta:

> **bio_vi:** Võ Minh Bảo, 47 tuổi, là người Kinh sinh sống ở một làng
> nông thôn trong đồng bằng sông Hồng. Sau khi ly hôn, ông sống đơn độc
> và làm công ăn lương như bảo vệ bán hàng và cung cấp dịch vụ cá nhân
> để duy trì đời sống. Ông không có trình độ chuyên môn kỹ thuật, nhưng
> có kinh nghiệm trong nông nghiệp, lâm nghiệp và thuỷ sản, thường giúp
> đỡ các gia đình trong làng trong vụ mùa. Thu nhập của ông nằm trong
> nhóm Q2, phản ánh mức sống vừa phải phổ biến với người lao động nông
> thôn vùng này.

> **bio_en:** Võ Minh Bảo is a 47-year-old Kinh man living in a rural
> village of the Red River Delta. After his divorce, he lives alone
> and earns a living as a wage-earning security guard and provider of
> personal services. He has no formal technical qualifications, but
> he has experience in agriculture, forestry and fishery, often
> assisting fellow villagers during planting and harvest seasons. His
> income falls into the second quintile (Q2), reflecting a modest
> livelihood typical of rural workers in the region.

The model honours every structured fact: divorced, rural, Red River
Delta, no formal qualification, wage employee, occupation = service /
sales / security, Q2 income. The bilingual pair stays semantically
parallel.

### 4.4 Cost guidance

For 100K personas at typical pricing (~400 input + 300 output
tokens each, $0.50 / $1.50 per 1M tokens for hosted Nemotron-3 Super):
roughly **$65** of API spend. The default cap is 50; use `--max-records 0`
to enrich the full batch.

---

## 5. Persona embeddings (optional)

After enrichment (or just after generation if you skipped it), embed
the bios + project to 2-D so the visualizer can render a semantic map.

### 5.1 Same backend abstraction as the curator

```bash
# Local sentence-transformers (default — free, offline, 384-d)
personas-vn embed-personas --max-records 10000

# NVIDIA NIM hosted (2048-d)
personas-vn embed-personas \
    --model nvidia/llama-3.2-nv-embedqa-1b-v2 \
    --max-records 10000
```

The model id auto-routes to the right backend (any `nvidia/...` →
NIM). Other recommended NIM models:

* `nvidia/llama-nemotron-embed-1b-v2` — latest, 2048-d
* `nvidia/llama-3.2-nemoretriever-300m-embed-v1` — lightweight
* `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` — vision-language

### 5.2 Outputs

```python
embedded.parquet:    # shape (n_sampled, 25 + 1)
    persona_id, name, bio_vi, bio_en, region, region_en, ...,
    income_quintile, vector

reduced.parquet:     # shape (n_sampled, 25 + 3)
    persona_id, ..., income_quintile, x, y, cluster
```

Wall times on a residential M-series Mac:

| Backend            | Model                                       | 10K personas |
| ------------------ | ------------------------------------------- | ------------ |
| local              | paraphrase-multilingual-MiniLM-L12-v2 (384-d) | ~2:14        |
| NIM (network-bound)| nv-embedqa-1b-v2 (2048-d)                   | ~3-5 min     |

---

## 6. Subset pipeline (parse → extract → embed → reduce)

Once the four parquets exist (or even just the 300K small variant),
[`DATASYNTHESIS.ipynb`](DATASYNTHESIS.ipynb) runs the same four
post-generation stages the curator uses on NSO source documents over
a **10 % subset of `Nemotron-Personas-Vietnam-small-vi`** (= 30K
personas). Every figure below was generated by re-executing that
notebook; the matching declarative builder is
[`scripts/_build_datasynthesis_nb.py`](scripts/_build_datasynthesis_nb.py).

| Stage    | What it does                                              | Mirror in code                                              |
| -------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| parse    | load + schema-validate + build unified narrative text     | `packages.curator.stages.ParseStage`                        |
| extract  | TF-IDF top-N keywords per persona + per region            | `packages.curator.stages.ExtractStage`                      |
| embed    | sentence-transformers (384-d) over the narrative          | `packages.personas.embed.pipeline.embed_personas`           |
| reduce   | UMAP → 2-D (HDBSCAN clustering optional, off by default)  | `packages.curator.stages.ReduceStage`                       |

Stage outputs land in `data/Nemotron-Personas-Vietnam/_subset_pipeline/`
(`parsed.parquet`, `embedded.parquet`, `reduced.parquet`) — isolated
from the canonical 300K small datasets. Wall-clock on an M-series Mac,
CPU-only: parse <1s, extract ~10s, embed ~6 min, reduce ~1-2 min. Dial
`SAMPLE_FRAC` in §1 down to `0.01` (3K rows) for sub-minute iteration.

### 6.1 Extract — TF-IDF keywords per macro-region

A multilingual-friendly TF-IDF (1–2 grams,
`token_pattern=r'(?u)\b[\wÀ-ỹ]{3,}\b'`) over each persona's narrative
gives the eight most-distinctive tokens per row. Grouping those by
macro-region surfaces what's linguistically distinctive in each
region's narratives — `nội` ("Nội") for Đồng bằng sông Hồng (Hà Nội),
`chí minh` for Đông Nam Bộ, `trung` for the Central Coast, etc.

![TF-IDF keywords per region](docs/figures/synthesis/01_tfidf_keywords_per_region.png)

### 6.2 Embed + reduce — UMAP scatter × macro-region

A multilingual SBERT model (`paraphrase-multilingual-MiniLM-L12-v2`,
384-d) encodes each narrative; UMAP projects the 30K vectors to 2-D.
Colouring by macro-region reveals coarse geographic separation in the
embedding — South East and Red River Delta narratives sit in
distinguishable parts of the plane because province names and
urban / rural cues bleed into the freeform text.

![UMAP × region](docs/figures/synthesis/02_umap_by_region.png)

### 6.3 UMAP × occupation (10-class ISCO)

The strongest splitter. Same UMAP layout, recoloured by occupation:
agricultural / forestry / fishery workers, professionals, and
elementary occupations all land in distinct neighbourhoods, and
clerical-support + service-sales mix in the centre — the shape NSO
V02.43 publishes, surfaced by an embedding that knew nothing about
the structured columns.

![UMAP × occupation](docs/figures/synthesis/03_umap_by_occupation.png)

### 6.4 UMAP × education level

Same UMAP plane coloured by education level. The
"Đại học trở lên" / "bachelor's or higher" tail forms a tight
cluster; "Không có trình độ CMKT" (no formal qualification) — the
synthetic-priors bucket that fills the long tail of NSO V02.54 — is
diffuse, exactly as you would expect from a population *without* a
strong professional-narrative anchor.

![UMAP × education](docs/figures/synthesis/04_umap_by_education.png)

### 6.5 UMAP × area (urban / rural)

Last view: the same UMAP layout coloured by `area` — the binary
urban (`Thành thị`) / rural (`Nông thôn`) classification. Unlike
region/occupation/education, area doesn't carve the embedding into
clean blobs because the freeform narrative templates mention
urban-vs-rural cues only sparingly; instead urban and rural points
inter-mix inside every occupational cluster. That's the *expected*
shape — area is a coarse geographic axis, not a narrative one.

Density-based HDBSCAN clustering is **opt-in** in §5 of the
notebook (default off, matching `PersonaEmbedConfig.cluster=False`
in the production pipeline and the `embed-personas --cluster` flag
on the CLI). When enabled, §7 of the notebook prints a modal-
demographic-per-cluster summary table; otherwise §7 prints a
"clustering: off" notice and skips. The four scatters here are
unaffected — they always colour by raw persona columns regardless.

![UMAP × area](docs/figures/synthesis/05_umap_by_area.png)

---

## 7. Uploading to HuggingFace

We follow the same pattern used by NVIDIA SDG-PGMs for the
[Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA)
dataset.

### 7.1 One-time setup

```bash
pip install huggingface_hub datasets
huggingface-cli login
```

### 7.2 Push

```bash
export HF_TOKEN=hf_xxx
python -m scripts.upload_to_hf \
    --repo your-user/personas-vn-personas \
    --push personas
```

What that does ([`scripts/upload_to_hf.py`](scripts/upload_to_hf.py)):

1. Loads `data/personas/personas_enriched.json` if present, else
   `personas.json`.
2. Materialises a 100K-row DataFrame with every structured column +
   `bio_vi` / `bio_en` / `enriched`.
3. `Dataset.from_pandas(df).push_to_hub(repo_id, ...)` uploads the
   personas as `train.parquet` at the repo root.
4. Side-cars `embedded.parquet` and `reduced.parquet` if they exist.
5. Generates a `README.md` dataset card with YAML frontmatter
   (`language: [vi, en]`, `license: cc-by-nc-4.0`, tags including
   `synthetic-data`, `personas`, `sdg-pgms`, `nemotron`, `bilingual`).

Resulting repo:

```
huggingface.co/datasets/your-user/personas-vn-personas/
├── README.md                     # dataset card
├── train.parquet                 # 100K personas
├── embedded.parquet              # 10K × 384 (or 2048) vectors
└── reduced.parquet               # 10K × (x, y, cluster)
```

### 7.3 Loading the published dataset

```python
from datasets import load_dataset
ds = load_dataset("your-user/personas-vn-personas", split="train")

# bilingual filter — only LLM-enriched personas in the South-East
mask = (ds["enriched"] == True) & (ds["region_en"] == "South East")  # noqa: E712
print(ds.filter(lambda r: r["enriched"] and r["region_en"] == "South East"))
```

### 7.4 Disclaimer

Synthetic personas. They are sampled from aggregate NSO statistics and
do not correspond to real individuals. The `_README.md` template in
`scripts/upload_to_hf.py` includes that disclaimer and a citation
back to the General Statistics Office of Vietnam — keep both.

---

## 8. Quality checks

Three quick sanity checks worth running before publishing:

```python
import pandas as pd, json
df = pd.DataFrame(json.load(open("data/personas/personas.json"))["personas"])

# 1. No NaN in any structured column
assert df.drop(columns=["bio_vi", "bio_en"]).notna().all().all()

# 2. Every Vietnamese label has an English mirror
pairs = [("region", "region_en"), ("urbanicity", "urbanicity_en"),
         ("sex", "sex_en"), ("education_level", "education_level_en")]
for vi, en in pairs:
    assert (df[vi].notna() == df[en].notna()).all(), (vi, en)

# 3. Children (15-19) are never employed in waged jobs at high rates
youth = df[df["age_group"] == "15-19"]
employed_youth = (youth["employment_status"] == "Làm công ăn lương").mean()
assert employed_youth < 0.35, f"youth waged-employment too high: {employed_youth:.0%}"
```

Plus the test suite covers:

* `tests/test_personagen.py` — determinism, schema completeness,
  bilingual fields, marginal sanity, JSON round-trip.
* `tests/test_embed_backends.py` — auto-dispatch local vs NIM, missing-
  key error paths, SBERT round-trip.

```bash
python -m pytest -q
```

---

## 9. Where to look in code

| If you want to...                                        | Read                                                        |
| -------------------------------------------------------- | ----------------------------------------------------------- |
| Change the persona PGM (variables / edges / CPDs)        | [`packages/personagen/vn_persona_generator.py`](packages/personagen/vn_persona_generator.py) |
| Plug in a new PX-Web table for an existing variable      | [`packages/personagen/distributions_pxweb.py`](packages/personagen/distributions_pxweb.py) |
| Add a new bilingual field to the schema                  | [`packages/ontology/persona.py`](packages/ontology/persona.py) |
| Change the LLM prompt                                    | [`packages/personagen/llm_enrich.py`](packages/personagen/llm_enrich.py) |
| Add a new embedding model                                | [`packages/personagen/embed_backends.py`](packages/personagen/embed_backends.py) |
| Re-create every figure in §6 (subset pipeline)           | [`scripts/_build_datasynthesis_nb.py`](scripts/_build_datasynthesis_nb.py) → [`DATASYNTHESIS.ipynb`](DATASYNTHESIS.ipynb) |
| Re-create the persona-marginal analysis figures           | [`scripts/analyze_personas.py`](scripts/analyze_personas.py) |
| Push to HuggingFace                                      | [`scripts/upload_to_hf.py`](scripts/upload_to_hf.py) |

See [`DATAPROCESSING.md`](DATAPROCESSING.md) for the matching
walkthrough of the curator side, [`DATAANALYSIS.md`](DATAANALYSIS.md)
for the analytical companion (38 figures across the 12 NSO databases),
[`DATAEXPLORATION.md`](DATAEXPLORATION.md) for the curator-pipeline
terminal-artefact tour, and [`DATAVISUALIZATION.md`](DATAVISUALIZATION.md)
for the geographic atlas.
