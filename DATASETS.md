# Nemotron-Personas-Vietnam datasets

This document walks through the four bilingual Vietnamese persona
datasets that `packages.personas.datasets` ships and how each column is grounded
in real NSO PX-Web data.

| Dataset | Rows | Language | File |
| --- | ---: | --- | --- |
| `Nemotron-Personas-Vietnam-large-vi` | 3 000 000 | Vietnamese | `Nemotron-Personas-Vietnam-large-vi.parquet` |
| `Nemotron-Personas-Vietnam-large-en` | 3 000 000 | English | `Nemotron-Personas-Vietnam-large-en.parquet` |
| `Nemotron-Personas-Vietnam-small-vi` | 300 000 | Vietnamese | `Nemotron-Personas-Vietnam-small-vi.parquet` |
| `Nemotron-Personas-Vietnam-small-en` | 300 000 | English | `Nemotron-Personas-Vietnam-small-en.parquet` |

The two `-small-` parquets are **stratified subsets** of the matching
`-large-` parquets — every `uuid` in `-small-vi` also appears in
`-large-vi`, and `-small-vi[i].uuid == -small-en[i].uuid`. So the four
files are pairwise mergeable on `uuid`.

## Two pipelines

The builder ships **two interchangeable narrative engines** for the eleven Nemotron narrative columns. The structured PGM half (region, age, sex, education, occupation, …) is identical in both paths.

| Path | Renderer | Personality | Network | Cost / 1M rows | Quality |
|---|---|---|---|---|---|
| **Default — templated** | Deterministic, NSO-grounded templates over curated bilingual lookup tables | none | none | $0 | Faithful but mechanical prose |
| **`--llm` — canonical Data Designer** | Two-stage LLM pipeline: PGM + OCEAN → **LLM A** (4 attribute fields) → **LLM B** (6 persona fields) | **Big Five (OCEAN)** sampled from peer-reviewed psychology priors | NVIDIA NIM (or any OpenAI-compatible endpoint) | ~$2-5K depending on model | Diverse, culturally-rich; mirrors `nvidia/Nemotron-Personas-Japan` |

```bash
conda activate pgm                                        # one-time setup: see README

# Templated (offline, default): 3M / 300K, ~27 minutes wall on an M-series Mac
python -m packages.pipeline.cli build-nemotron \
    --large-size 3000000 --small-size 300000

# Templated smoke run (~5s):
python -m packages.pipeline.cli build-nemotron \
    --large-size 10000 --small-size 1000 --seed 42

# Canonical Data Designer pipeline (PGM + OCEAN → LLM A → LLM B):
export PERSONAS_VN_LLM_API_KEY=$NVIDIA_API_KEY            # or any OpenAI-compat endpoint
python -m packages.pipeline.cli build-nemotron --llm \
    --large-size 100 --small-size 20                       # tiny LLM smoke (~30s, ~$0.50)
python -m packages.pipeline.cli build-nemotron --llm \
    --llm-model qwen/qwen3.5-397b-a17b \
    --large-size 1000 --small-size 200
```

Both paths produce **bit-identical 22-column parquet schemas**; downstream tools see the same columns and dtypes regardless of which engine generated them.

Outputs land in `data/Nemotron-Personas-Vietnam/` by default.

---

## 1. Schema

The 22-column layout is identical to `nvidia/Nemotron-Personas-Japan`,
with Japan's `region` / `area` / `prefecture` geographic trio adapted
for Vietnam (six NSO macro-regions, urban/rural, sixty-three NSO
provinces).

| # | Column | dtype | Description |
| --- | --- | --- | --- |
| 1 | `uuid` | `string` | UUID4 (canonical 8-4-4-4-12 hex). Same `uuid` in vi/en pair and small ⊂ large. |
| 2 | `professional_persona` | `string` | Templated paragraph driven by `occupation`, `education_level`, `province`. |
| 3 | `sports_persona` | `string` | Region-conditioned. |
| 4 | `arts_persona` | `string` | Region-conditioned. |
| 5 | `travel_persona` | `string` | Region-conditioned. |
| 6 | `culinary_persona` | `string` | Region-conditioned (regional Vietnamese cuisine). |
| 7 | `persona` | `string` | One-sentence summary spanning every structured field. |
| 8 | `cultural_background` | `string` | Region + area cultural traits. |
| 9 | `skills_and_expertise` | `string` | Prose form of column 10. |
| 10 | `skills_and_expertise_list` | `string` | 4 comma-separated skills, occupation-conditioned. |
| 11 | `hobbies_and_interests` | `string` | Prose form of column 12. |
| 12 | `hobbies_and_interests_list` | `string` | 3 comma-separated hobbies, area-conditioned. |
| 13 | `career_goals_and_ambitions` | `string` | Occupation- & age-conditioned. |
| 14 | `sex` | `string` | NSO V02.02 (`Nam`/`Nữ` ↔ `male`/`female`). |
| 15 | `age` | `int64` | Integer, 15–99. NSO V02.41 (5-year buckets) jittered ±2 for variety. |
| 16 | `marital_status` | `string` | NSO V02.43-style 4-class (`Chưa kết hôn`/`Đã kết hôn`/`Goá`/`Ly hôn` ↔ `never married`/`married`/`widowed`/`divorced`). |
| 17 | `education_level` | `string` | NSO V02.54 trained-labour qualification ladder. |
| 18 | `occupation` | `string` | NSO V02.43 ISCO-collapsed 10-class taxonomy. |
| 19 | `region` | `string` | NSO V02.01 6 macro-regions. |
| 20 | `area` | `string` | `Thành thị`/`Nông thôn` ↔ `urban`/`rural`. NSO V02.02. |
| 21 | `province` | `string` | One of the 63 NSO admin-1 provinces. |
| 22 | `country` | `string` | `Việt Nam` (vi) / `Vietnam` (en). |

The schema is locked down in `packages/datagen/nemotron_schema.py`:

```45:60:packages/datagen/nemotron_schema.py
NEMOTRON_PERSONAS_VIETNAM_COLUMNS: Final[tuple[str, ...]] = (
    # 1. globally-unique identifier (UUID4 hex string)
    "uuid",
    # 2-7. six narrative persona fields (long natural-language)
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "persona",
    # 8-13. five contextual narrative fields + two "_list" mirrors
    "cultural_background",
    "skills_and_expertise",
```

The table was cross-referenced against the published Hugging Face
schemas via `https://datasets-server.huggingface.co/info?dataset=...`
for USA / Japan / Korea / Brazil / France / India / Singapore — the
non-geographic 19 fields are identical across every regional variant,
which is why we adopted them verbatim.

---

## 2. Data sources — every column is NSO-grounded

Structured columns (sex, age, marital, education, occupation, region,
area, province) are **directly sampled** from the NSO PX-Web parquets
that `personas-vn curate --only download` writes under
`data/nso-gov-vn/raw/pxweb/vi/`.

| Column | NSO PX-Web table | What we use |
| --- | --- | --- |
| `region` | V02.01 *Area, population, density by province* | 6 macro-region population marginal |
| `area` | V02.02 *Population by sex × urban/rural* | urban/rural marginal |
| `province` | V02.01 | per-province population, conditional on region |
| `age` | V02.41 *Employed by 5-year age group* | 5-year buckets 15-19 … 65+, jittered to integer |
| `sex` | V02.02 | sex marginal (≈ 50 / 50) |
| `marital_status` | V02.43 (synthetic priors per the project's existing PGM) | 4-class age × sex priors |
| `education_level` | V02.54 *Trained labour by qualification level* | with synthetic "no formal CMKT" bucket filling the remainder |
| `occupation` | V02.43 *Employed by occupation (ISCO)* | 10-class collapsed taxonomy |

The structured-persona join happens via the existing
`packages.personas.pgm.VNPersonaGenerator` (the SDG-PGMs cascaded sampler
that the project already uses for its 100 K-row personas dataset). No
new data sources are introduced — we reuse exactly the same NSO-derived
distribution tables.

The narrative columns (`professional_persona`, `culinary_persona`,
`hobbies_and_interests`, …) are **deterministic templates over the
structured fields**. They are not LLM-generated — that's intentional:
each narrative cell cites a structured fact (region cuisine, occupation
skills, …) so the prose stays grounded in the NSO-derived categorical
draws. See `packages/datagen/narrative_data.py` for the per-region
cuisine, scenic-spot, sport, and cultural-motif lookup tables, and
`packages/datagen/narrative_render.py` for the renderer.

---

## 3. Names

Vietnamese full names use the
[`vn-fullname-generator`](https://pypi.org/project/vn-fullname-generator/)
PyPI package (which itself draws from a curated Vietnamese name
dictionary at <https://github.com/duyet/vietnamese-namedb>). We sample
from the underlying name pools using a vectorised `numpy.random.Generator`
seeded from the builder's master `seed` — see
`packages/datagen/name_generator.py`. This is ~100× faster than
calling `vn_fullname_generator.generator.generate(gender)` per row at
the 3 M-row scale, **and** removes the upstream library's reliance on
Python's global `random` module (which can't be seeded without
clobbering other libraries' RNG state).

Vietnamese names appear *verbatim* in both the `-vi` and `-en` parquets;
we don't romanise. So `-large-vi.persona[i]` and `-large-en.persona[i]`
both contain the same string `"Nguyễn Thị Mai"`.

Names are infused **into** the narrative columns rather than published as
their own column — matching the convention all published
Nemotron-Personas datasets use (no `name`, `first_name`, or `last_name`
columns; names live inside the `persona` and `professional_persona`
prose).

---

## 4. Reproducibility contract

A single integer seed (`--seed`, default `42`) drives the entire build.
Same seed → identical content on every machine, every OS, every Python
hash-randomisation. Seven independent RNG streams are spawned from the
master seed via `numpy.random.SeedSequence.spawn(7)` so that adding a
new narrative variant doesn't perturb the structured demographics, and
changing chunk size doesn't perturb any per-row narrative.

```text
seed
 ├─ spawn[0] → structured personas (pgmpy via VNPersonaGenerator)
 ├─ spawn[1] → uuids
 ├─ spawn[2] → province (region-conditional V02.01 weights)
 ├─ spawn[3] → vn-fullname-generator name pools
 ├─ spawn[4] → narrative variants — templated path (vi & en share picks)
 ├─ spawn[5] → small-subset stratified sample
 └─ spawn[6] → OCEAN traits (Big Five, age/sex-conditional priors)
```

The LLM pipeline path is non-deterministic by nature (LLMs sample), but
its **per-row JSONL cache** in `data/Nemotron-Personas-Vietnam/_llm_cache/` pins each
row's outputs the first time they're generated. Re-runs are cache-hit
and **bit-equal** as long as `seed`, `model`, and the LLM prompts are
unchanged.

The narrative variants stream is itself **chunk-size invariant**: we
pre-compute a per-row permutation matrix for every concern *before* the
chunk loop starts, then slice into it per chunk. So `--chunk-size 50000`
and `--chunk-size 250000` produce **bit-equal output content** at the
same seed (parquet metadata may still differ — creation timestamp etc.
— but the row data is identical).

The test suite checks all five reproducibility properties:

```bash
python -m pytest -q tests/test_datagen.py    # end-to-end determinism + schema checks
```

* `test_same_seed_produces_identical_content`
* `test_different_seed_produces_different_content`
* `test_chunk_size_does_not_affect_content`
* `test_vi_en_large_share_uuids` (vi/en describe the same persona row-for-row)
* `test_vi_en_small_share_uuids`
* `test_small_is_subset_of_large`

plus structural invariants (column order, dtypes, NSO-canonical region/
province/area labels, age range, country label, narrative non-emptiness,
small-subset region coverage).

---

## 5. Stratified small subsets

The two `-small-*-300K-` parquets are sampled from the corresponding
`-large-*-3M-` parquets with **proportional allocation by `(region ×
sex × education_level)`**. Each stratum's quota is its share of the
parent population (rounded to ≥ 1 so even tiny strata are represented),
then trimmed deterministically back to the target size if the cumulative
quotas overshoot.

The same UUID set is enforced across the vi and en small parquets, so
joining `Nemotron-Personas-Vietnam-small-vi` to `Nemotron-Personas-Vietnam-small-en`
on `uuid` yields a 1:1 bilingual pair, and joining either to the
matching `-large-*` is a clean inner join.

---

## 6. Inspecting a row

```python
import pyarrow.parquet as pq
df_vi = pq.read_table("data/Nemotron-Personas-Vietnam/Nemotron-Personas-Vietnam-large-vi.parquet").to_pandas()
df_en = pq.read_table("data/Nemotron-Personas-Vietnam/Nemotron-Personas-Vietnam-large-en.parquet").to_pandas()

# Random sanity check: the row at index 17 is the same persona in both
# languages, and the structured fields match the NSO categorical sets.
print(df_vi.iloc[17][["uuid", "sex", "age", "occupation", "region", "province"]])
print(df_en.iloc[17][["uuid", "sex", "age", "occupation", "region", "province"]])
print()
print("VI persona:")
print(" ", df_vi.iloc[17]["persona"])
print("EN persona:")
print(" ", df_en.iloc[17]["persona"])
```

Sample row (`large-vi[0]` from a `seed=42` build):

```
uuid:                          97401f13-cb79-4d77-8ef4-b12e2b13e40b
sex:                           Nam
age:                           27
marital_status:                Chưa kết hôn
education_level:               Không có trình độ CMKT
occupation:                    Thợ thủ công và các thợ khác có liên quan
region:                        Trung du và miền núi phía Bắc
area:                          Nông thôn
province:                      Lạng Sơn
country:                       Việt Nam

persona (VI):
  Đoàn Quốc Trường, 27 tuổi, Nam, Chưa kết hôn, trình độ Không có
  trình độ CMKT, làm việc trong nhóm nghề Thợ thủ công và các thợ
  khác có liên quan, sinh sống tại khu vực Nông thôn thuộc Lạng Sơn
  (Trung du và miền núi phía Bắc).

persona (EN):
  Đoàn Quốc Trường, a 27-year-old male, never married with no formal
  qualification, works as a craft and trades worker and lives in a
  rural part of Lạng Sơn in the Northern Midlands and Mountains.
```

### 6.1 Full row dumps — every column, two personas, both languages

Two rows from the same `seed=42` build, in both `vi` and `en`. The rows
are bit-aligned across the four parquet files: `small-vi[i]`,
`small-en[i]`, `large-vi[i+offset]`, `large-en[i+offset]` all describe
the same persona, just rendered in different prose. Long fields wrap
with a 2-space hanging indent for readability.

#### Row 0 — Phan Bích Hồng (Quảng Ngãi · North Central · professional · F23)

**`small-vi[0]`**

```
uuid: 75fff514-d080-46d0-9f10-d31be4066f92
professional_persona:
  Phan Bích Hồng làm việc trong nhóm nghề Chuyên môn kỹ thuật bậc cao tại Quảng Ngãi.
    Trình độ chuyên môn của họ là Sơ cấp, và họ thành thạo sử dụng phần mềm chuyên
    ngành, thuyết trình kết quả cho khách hàng và đồng nghiệp và đào tạo và hướng dẫn
    chuyên môn. Họ làm việc cẩn thận, có trách nhiệm và luôn học hỏi cái mới.
sports_persona:
  Ở tuổi 23, Phan Bích Hồng duy trì sức khoẻ qua đua thuyền truyền thống và thỉnh thoảng
    tham gia bơi biển và lướt ván.
arts_persona:
  Phan Bích Hồng yêu thích các loại hình nghệ thuật truyền thống quê hương: bài chòi và
    lễ hội Cầu Ngư, thường tham gia hoặc theo dõi vào những dịp lễ hội trong năm.
travel_persona:
  Phan Bích Hồng thường nghỉ ngơi tại các điểm đến nổi tiếng của Bắc Trung Bộ và Duyên
    hải miền Trung như động Phong Nha và phố cổ Hội An; những chuyến đi ngắn cuối tuần
    là cách họ cân bằng nhịp sống.
culinary_persona:
  Phan Bích Hồng thường thưởng thức ẩm thực Bắc Trung Bộ và Duyên hải miền Trung, đặc
    biệt yêu thích bánh xèo miền Trung và cơm hến; vào cuối tuần thường tự nấu các món
    truyền thống để chiêu đãi gia đình.
persona:
  Phan Bích Hồng, 23 tuổi, Nữ, Chưa kết hôn, trình độ Sơ cấp, làm việc trong nhóm nghề
    Chuyên môn kỹ thuật bậc cao, sinh sống tại khu vực Thành thị thuộc Quảng Ngãi (Bắc
    Trung Bộ và Duyên hải miền Trung).
cultural_background:
  Phan Bích Hồng lớn lên ở khu vực Thành thị thuộc Quảng Ngãi (Bắc Trung Bộ và Duyên hải
    miền Trung). Văn hoá địa phương để lại dấu ấn rõ rệt: lễ hội Cầu Ngư và ca Huế trên
    sông Hương là những giá trị họ luôn trân trọng và truyền lại cho thế hệ sau.
skills_and_expertise:
  Các kỹ năng nổi bật bao gồm: phân tích chuyên sâu trong lĩnh vực hành nghề, thiết kế
    giải pháp dựa trên cơ sở khoa học, sử dụng phần mềm chuyên ngành, thuyết trình kết
    quả cho khách hàng và đồng nghiệp.
skills_and_expertise_list:
  phân tích chuyên sâu trong lĩnh vực hành nghề, thiết kế giải pháp dựa trên cơ sở khoa
    học, sử dụng phần mềm chuyên ngành, thuyết trình kết quả cho khách hàng và đồng
    nghiệp
hobbies_and_interests:
  Sở thích thường ngày của họ bao gồm: nấu ăn theo công thức trên YouTube, học tiếng Anh
    qua ứng dụng, uống cà phê và đọc sách buổi sáng.
hobbies_and_interests_list:
  nấu ăn theo công thức trên YouTube, học tiếng Anh qua ứng dụng, uống cà phê và đọc
    sách buổi sáng
career_goals_and_ambitions:
  Trong vài năm tới, Phan Bích Hồng mong muốn lấy thêm chứng chỉ chuyên môn quốc tế,
    đồng thời công bố thêm bài báo khoa học hoặc dự án trọng điểm.
sex: Nữ
age: 23
marital_status: Chưa kết hôn
education_level: Sơ cấp
occupation: Chuyên môn kỹ thuật bậc cao
region: Bắc Trung Bộ và Duyên hải miền Trung
area: Thành thị
province: Quảng Ngãi
country: Việt Nam
```

**`small-en[0]`** (same `uuid`, same persona, English narrative)

```
uuid: 75fff514-d080-46d0-9f10-d31be4066f92
professional_persona:
  Phan Bích Hồng works as a professional in Quảng Ngãi, with a primary vocational
    qualification, and is skilled in using domain-specific software, presenting results
    to clients and peers, and professional training and mentorship. They are careful,
    responsible, and continually keen to learn.
sports_persona:
  At 23, Phan Bích Hồng stays active through traditional boat racing and occasionally
    takes part in sea swimming and surfing.
arts_persona:
  Phan Bích Hồng loves the traditional arts of their home region — bài chòi sung-card
    folk art and the Cầu Ngư fishermen's festival — and joins or follows them whenever
    a festival comes around.
travel_persona:
  Phan Bích Hồng unwinds at well-known destinations in the North Central and Central
    Coastal such as Phong Nha Cave and Hội An Ancient Town; short weekend trips are how
    they keep life in balance.
culinary_persona:
  Phan Bích Hồng enjoys the cuisine of the North Central and Central Coastal, with a
    particular fondness for Central-style crispy pancake and cơm hến baby-clam rice; on
    weekends they like to cook traditional dishes for the family.
persona:
  Phan Bích Hồng, a 23-year-old female, never married with a primary vocational
    qualification, works as a professional and lives in an urban part of Quảng Ngãi in
    the North Central and Central Coastal.
cultural_background:
  Phan Bích Hồng grew up in an urban part of Quảng Ngãi in the North Central and
    Central Coastal. The local culture left a clear imprint: the Cầu Ngư fishermen's
    festival and ca Huế river-boat singing are values they cherish and pass on to the
    next generation.
skills_and_expertise:
  Their core skills include: deep professional analysis in the field, designing
    science-based solutions, using domain-specific software, presenting results to
    clients and peers.
skills_and_expertise_list:
  deep professional analysis in the field, designing science-based solutions, using
    domain-specific software, presenting results to clients and peers
hobbies_and_interests:
  Their everyday interests include: cooking from YouTube recipes, studying English
    through mobile apps, morning coffee with a book.
hobbies_and_interests_list:
  cooking from YouTube recipes, studying English through mobile apps, morning coffee
    with a book
career_goals_and_ambitions:
  Over the next few years, Phan Bích Hồng hopes to earn additional internationally
    recognised credentials, and to publish further peer-reviewed work or lead a
    flagship project.
sex: female
age: 23
marital_status: never married
education_level: primary vocational
occupation: professionals
region: North Central and Central Coastal
area: urban
province: Quang Ngai
country: Vietnam
```

#### Row 17 — Dương Trung Việt (Bạc Liêu · Mekong Delta · agriculture · M34)

**`small-vi[17]`**

```
uuid: 641e587c-04bf-4bde-87e3-e374abff5020
professional_persona:
  Dương Trung Việt làm việc trong nhóm nghề Lao động có kỹ năng trong nông nghiệp, lâm
    nghệp và thủy sản tại Bạc Liêu. Trình độ chuyên môn của họ là Không có trình độ
    CMKT, và họ thành thạo đọc dự báo thời tiết và lên kế hoạch ứng phó, thu hoạch, sơ
    chế và bảo quản nông sản và lập lịch mùa vụ và luân canh. Họ làm việc cẩn thận, có
    trách nhiệm và luôn học hỏi cái mới.
sports_persona:
  Ở tuổi 34, Dương Trung Việt duy trì sức khoẻ qua bơi sông và bơi hồ và thỉnh thoảng
    tham gia đua ghe ngo.
arts_persona:
  Dương Trung Việt yêu thích các loại hình nghệ thuật truyền thống quê hương: đua ghe
    ngo và chợ nổi, thường tham gia hoặc theo dõi vào những dịp lễ hội trong năm.
travel_persona:
  Dương Trung Việt thường nghỉ ngơi tại các điểm đến nổi tiếng của Đồng bằng sông Cửu
    Long như miệt vườn Cái Bè và rừng tràm Trà Sư; những chuyến đi ngắn cuối tuần là
    cách họ cân bằng nhịp sống.
culinary_persona:
  Dương Trung Việt thường thưởng thức ẩm thực Đồng bằng sông Cửu Long, đặc biệt yêu
    thích bánh tét lá cẩm và lẩu cá linh bông điên điển; vào cuối tuần thường tự nấu
    các món truyền thống để chiêu đãi gia đình.
persona:
  Dương Trung Việt, 34 tuổi, Nam, Chưa kết hôn, trình độ Không có trình độ CMKT, làm
    việc trong nhóm nghề Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản,
    sinh sống tại khu vực Nông thôn thuộc Bạc Liêu (Đồng bằng sông Cửu Long).
cultural_background:
  Dương Trung Việt lớn lên ở khu vực Nông thôn thuộc Bạc Liêu (Đồng bằng sông Cửu
    Long). Văn hoá địa phương để lại dấu ấn rõ rệt: vọng cổ và đờn ca tài tử Nam Bộ là
    những giá trị họ luôn trân trọng và truyền lại cho thế hệ sau.
skills_and_expertise:
  Các kỹ năng nổi bật bao gồm: sử dụng phân bón và thuốc bảo vệ thực vật an toàn, vận
    hành máy nông cơ nhỏ, đọc dự báo thời tiết và lên kế hoạch ứng phó, lập lịch mùa
    vụ và luân canh.
skills_and_expertise_list:
  sử dụng phân bón và thuốc bảo vệ thực vật an toàn, vận hành máy nông cơ nhỏ, đọc dự
    báo thời tiết và lên kế hoạch ứng phó, lập lịch mùa vụ và luân canh
hobbies_and_interests:
  Sở thích thường ngày của họ bao gồm: chăm sóc gia súc, gia cầm, câu cá ở ao làng,
    chăm sóc vườn rau và cây ăn quả.
hobbies_and_interests_list:
  chăm sóc gia súc, gia cầm, câu cá ở ao làng, chăm sóc vườn rau và cây ăn quả
career_goals_and_ambitions:
  Trong vài năm tới, Dương Trung Việt mong muốn áp dụng kỹ thuật mới (VietGAP, hữu
    cơ), đồng thời mở rộng diện tích canh tác hoặc đàn nuôi.
sex: Nam
age: 34
marital_status: Chưa kết hôn
education_level: Không có trình độ CMKT
occupation: Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản
region: Đồng bằng sông Cửu Long
area: Nông thôn
province: Bạc Liêu
country: Việt Nam
```

**`small-en[17]`** (same `uuid`, same persona, English narrative)

```
uuid: 641e587c-04bf-4bde-87e3-e374abff5020
professional_persona:
  Dương Trung Việt works as a skilled agriculture, forestry or fishery worker in Bạc
    Liêu, with no formal qualification, and is skilled in reading weather forecasts and
    planning around them, harvest, preliminary processing, and storage, and seasonal
    planning and crop rotation. They are careful, responsible, and continually keen to
    learn.
sports_persona:
  At 34, Dương Trung Việt stays active through river and pool swimming and
    occasionally takes part in ngo-boat racing.
arts_persona:
  Dương Trung Việt loves the traditional arts of their home region — ngo-boat racing
    and floating markets — and joins or follows them whenever a festival comes around.
travel_persona:
  Dương Trung Việt unwinds at well-known destinations in the Mekong River Delta such
    as Cái Bè orchards and Trà Sư cajuput forest; short weekend trips are how they keep
    life in balance.
culinary_persona:
  Dương Trung Việt enjoys the cuisine of the Mekong River Delta, with a particular
    fondness for purple-leaf bánh tét and linh-fish and dien-dien-flower hotpot; on
    weekends they like to cook traditional dishes for the family.
persona:
  Dương Trung Việt, a 34-year-old male, never married with no formal qualification,
    works as a skilled agriculture, forestry or fishery worker and lives in a rural
    part of Bạc Liêu in the Mekong River Delta.
cultural_background:
  Dương Trung Việt grew up in a rural part of Bạc Liêu in the Mekong River Delta. The
    local culture left a clear imprint: vọng cổ blues-style ballads and Mekong đờn ca
    tài tử music are values they cherish and pass on to the next generation.
skills_and_expertise:
  Their core skills include: safe use of fertilizers and pesticides, operating small
    farm machinery, reading weather forecasts and planning around them, seasonal
    planning and crop rotation.
skills_and_expertise_list:
  safe use of fertilizers and pesticides, operating small farm machinery, reading
    weather forecasts and planning around them, seasonal planning and crop rotation
hobbies_and_interests:
  Their everyday interests include: looking after livestock and poultry, fishing at
    the village pond, tending the home garden and fruit trees.
hobbies_and_interests_list:
  looking after livestock and poultry, fishing at the village pond, tending the home
    garden and fruit trees
career_goals_and_ambitions:
  Over the next few years, Dương Trung Việt hopes to adopt modern standards (VietGAP,
    organic), and to expand the farmed area or herd.
sex: male
age: 34
marital_status: never married
education_level: no formal qualification
occupation: skilled agricultural, forestry and fishery workers
region: Mekong River Delta
area: rural
province: Bac Lieu
country: Vietnam
```

What to notice in these two dumps:

* **All 22 columns are populated** for every row — there are no
  language-specific NaN gaps. The 12 long-form text fields
  (`professional_persona`, `sports_persona`, `arts_persona`,
  `travel_persona`, `culinary_persona`, `persona`,
  `cultural_background`, `skills_and_expertise`,
  `skills_and_expertise_list`, `hobbies_and_interests`,
  `hobbies_and_interests_list`, `career_goals_and_ambitions`) are
  all rendered for both `vi` and `en`, even when the structured
  fields below them carry English / Vietnamese-only enums.
* **Identical UUIDs** across `small-vi` and `small-en` confirm the
  bit-aligned-by-index reproducibility contract from §4.
* **Region-conditioned content** — the bài chòi / Cầu Ngư references
  for the North Central row vs. ngo-boat racing / vọng cổ for the
  Mekong row come from `packages/personas/datasets/lookups.py`'s
  per-region lookup tables, themselves grounded in NSO V14.45 cultural
  surveys.
* **Diacritics survive** the round-trip — even the English narratives
  retain the Vietnamese name (`Dương Trung Việt`, `Phan Bích Hồng`)
  and place names (`Quảng Ngãi`, `Bạc Liêu`) with full tone marks. The
  ASCII `province` field strips them (`Quang Ngai`, `Bac Lieu`) so
  downstream tools that key on bare-ASCII names still work.

---

## 7. Canonical Data Designer pipeline (`--llm`)

When you pass `--llm`, the builder swaps the templated narrative
renderer for the two-LLM pipeline that mirrors NVIDIA's published
Nemotron-Personas Data Designer architecture:

```
PGM (NSO-grounded structured fields) ──┐
                                       ├──►  LLM A  ──►  cultural_background
OCEAN (peer-reviewed Big Five priors) ─┘             skills_and_expertise{,_list}
                                                     hobbies_and_interests{,_list}
                                                     career_goals_and_ambitions
                                                              │
                                                              ▼
        PGM + OCEAN + LLM-A outputs ────────────────────► LLM B ─►  persona
                                                                    professional_persona
                                                                    arts_persona
                                                                    sports_persona
                                                                    travel_persona
                                                                    culinary_persona
```

### 7.1 OCEAN — sourcing and methodology

NSO does not publish psychometric data (verified empirically — every
table title across all 502 PX-Web matrices is socio-economic). Big Five
trait scores are sampled from **peer-reviewed psychology priors**, the
same approach `nvidia/Nemotron-Personas-USA` uses (US Census also has
no OCEAN data). The two papers driving the priors are cited inline in
[`packages/datagen/ocean.py`](packages/datagen/ocean.py):

* **Roberts, Walton & Viechtbauer (2006)** *Patterns of mean-level
  change in personality traits across the life course: A meta-analysis
  of longitudinal studies.* Psychological Bulletin 132(1), 1-25.
  — provides the **age effects** (Conscientiousness ↑, Agreeableness ↑,
  Neuroticism ↓, Openness slowly ↓, Extraversion small ↓ across the lifespan).
* **Schmitt, Realo, Voracek & Allik (2008)** *Why can't a man be more
  like a woman? Sex differences in Big Five personality traits across
  55 cultures.* JPSP 94(1), 168-182. — provides the **sex effects**
  (women slightly higher on Neuroticism, Agreeableness, Conscientiousness;
  men slightly higher on Openness; Extraversion ≈ 0). The 55-culture
  sample includes Vietnam and effects replicate locally.

Each of the five traits is sampled independently from a Gaussian whose
mean depends on (age, sex), with σ = 0.7 (typical NEO-FFI population
SD on a 5-pt scale), then clipped to [1, 5]. We deliberately do **not**
inject occupation- or region-level personality correlations (the
literature shows them to be small, r ≈ 0.1–0.2) to avoid over-claiming.

The downstream `personality_summary_vi` / `personality_summary_en`
columns turn each row's five trait bands into a one-line LLM-friendly
summary (e.g. `"high Openness, mid Conscientiousness, low Neuroticism"`).

### 7.2 LLM model menu

Configurable via `--llm-model`. Default is `nvidia/nemotron-3-super-120b-a12b`.

| Model id | Provider | Endpoint | Notes |
|---|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | NVIDIA | NIM | Default — most capable Nemotron MoE |
| `qwen/qwen3.5-122b-a10b` | Alibaba (NIM) | NIM | Smaller MoE, cheaper |
| `qwen/qwen3.5-397b-a17b` | Alibaba (NIM) | NIM | Largest MoE, highest quality |
| `openai/gpt-oss-120b` | OpenAI | NIM | Apache-2.0 open weights |

Any OpenAI-compatible endpoint works — set `--llm-base-url` to e.g.
`http://localhost:8000/v1` for a local vLLM, or to OpenRouter / Together /
Fireworks endpoints. The base URL defaults to NVIDIA's NIM gateway.

### 7.3 Cost estimate (3M rows × 2 languages)

Per-row token budget (prompt + completion):

* **LLM A**: ~600 input + ~400 output ≈ 1 000 tokens
* **LLM B**: ~1200 input + ~1500 output ≈ 2 700 tokens
* **Total per row per language**: ~3 700 tokens

For the full 3M / 300K product (3.3M rows × 2 languages = ~6.6M LLM-A + ~6.6M LLM-B
calls), expect ~22 B tokens. At indicative NVIDIA NIM Nemotron pricing
(~$0.5/M input, ~$1.5/M output — verify current pricing on
build.nvidia.com):

| Phase | Tokens | Approx cost |
|---|---:|---:|
| Input | ~7 B | ~$3 500 |
| Output | ~5 B | ~$7 500 |
| **Total per language** | | **~$11 000** |
| **Total (vi + en)** | | **~$22 000** |

The smaller `qwen3.5-122b-a10b` typically costs ~30-50% of the Nemotron
Super line; `gpt-oss-120b` is the cheapest. Use the per-row JSONL cache
(`data/Nemotron-Personas-Vietnam/_llm_cache/{llm_a,llm_b}.jsonl`) to start small —
`python -m packages.pipeline.cli build-nemotron --llm --large-size 1000 --small-size 200`
is ~$7 — and scale up incrementally without re-paying for cached rows.

### 7.4 Resumability

Every successful LLM response is appended to the JSONL cache at
write-time. A killed build resumes by reading the cache on next start
and skipping completed `(uuid, stage, lang, model)` keys; the model
fingerprint in the cache key means flipping the model produces fresh
outputs without serving stale ones. The cache survives Ctrl-C, OOM,
network failure, and process crashes — at worst the row that was being
written when the crash hit is lost (one line of JSONL).

### 7.5 Validation + fallback

Each LLM response must be a **JSON object** with the exact six (LLM A)
or six (LLM B) required keys, all non-empty strings. Responses that
fail validation are retried up to `LLMConfig.retries` times, then
the row is filled in from the templated renderer (with a warning) so
the final parquet always has all 22 columns populated. Set
`LLMPipelineConfig.fallback_to_templates=False` to abort the build
on the first row that exhausts its retries.

---

## 8. Disclaimer

All four datasets are **synthetic**. Every persona is sampled from
aggregate NSO statistics and does not correspond to a real individual.
Any similarity to a living person is purely coincidental. The structured
data the personas are conditioned on belongs to the General Statistics
Office of Vietnam — please respect their terms of use when redistributing
the source PX-Web parquets.

The OCEAN trait scores attached to each persona are sampled from
peer-reviewed psychology priors (Roberts 2006, Schmitt 2008) and are
**not** Vietnam-specific psychometric data — Vietnam's NSO does not
publish that. Treat the trait scores as plausible Big Five priors for
synthetic-persona conditioning, not as ground truth about Vietnamese
personality distributions.

License: same as the parent project (see `LICENSE`).
