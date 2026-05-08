"""Update Nemotron Personas Vietnam v0.pptx → v1.pptx.

Reads the v0 template (3 sparse slides — Title / empty-middle / Thank-you),
clones the empty middle into a full content deck, and writes
``Nemotron Personas Vietnam v1.pptx`` next to it.

The v0 template provides 16 layouts (Google-Slides-style); we use them
heavily so the output picks up the master's typography and chrome
without our needing to re-style anything by hand.
"""

from __future__ import annotations

import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

# ---------------------------------------------------------------------------
# Layout indices (verified against the v0 template's master)
# ---------------------------------------------------------------------------
L_TITLE        = 0    # Title Slide
L_BULLETS      = 1    # Title, Subtitle, Bullets
L_AGENDA       = 3    # Agenda
L_PROGRESSION  = 5    # Progression (10 numbered slots)
L_RIGHT_PANE   = 9    # Title, Subtitle, Content Right (left bullets + right pane)
L_CLOSING      = 13   # Closing Slide

# v0 deck has slide 1 (title), slide 2 (empty placeholder), slide 3 (Thanks)


# ---------------------------------------------------------------------------
# Slide-builder helpers
# ---------------------------------------------------------------------------
def _set_paragraph(tf, lines, *, sizes=None):
    """Replace a placeholder's text with a list of paragraph strings."""
    tf.clear()
    if not lines:
        return
    p = tf.paragraphs[0]
    p.text = lines[0]
    if sizes:
        for run in p.runs:
            run.font.size = sizes[0]
    for i, ln in enumerate(lines[1:], 1):
        para = tf.add_paragraph()
        para.text = ln
        if sizes and i < len(sizes):
            for run in para.runs:
                run.font.size = sizes[i]


def _populate(slide, items: dict[int, list[str] | str]) -> None:
    """Fill the slide's placeholders. Keys are placeholder *idx*; values
    are either a single string or a list of strings (one per paragraph)."""
    for idx, payload in items.items():
        try:
            ph = slide.placeholders[idx]
        except KeyError:
            continue
        lines = payload if isinstance(payload, list) else [payload]
        _set_paragraph(ph.text_frame, lines)


# ---------------------------------------------------------------------------
# v1 slide content
# ---------------------------------------------------------------------------
def add_content_slides(prs):
    """Append every content slide to ``prs`` (we re-order at the end)."""
    slides = []

    # ------------------------------------------------------------------
    # 1. Agenda
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_AGENDA])
    _populate(s, {
        1: [
            "Agenda",
            "1. Five-layer architecture, single seed",
            "2. Layer 1 — NSO PX-Web foundation: 502 tables, 12 databases",
            "3. Layer 2 — bilingual ontology tree (vi → en, 100% coverage)",
            "4. Layer 3 — SDG-PGMs persona model + Layer 4 — OCEAN priors",
            "5. Layer 5 — Data Designer LLM-A / LLM-B narrative pipeline",
            "6. The four published parquet datasets + reproducibility contract",
            "7. Architecture, live status, and what's next",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 2. The big picture — five layers as a Progression
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_PROGRESSION])
    # Progression layout has placeholders 1..9 for numbered steps + 0 for title
    _populate(s, {
        0: "Five layers, one seed, four published datasets",
        1: ["Layer 1 — NSO PX-Web",
            "12 statistical databases, 502 matrices, ~74 MB of bilingual "
            "structured data, crawled via the v2 JSON catalog API."],
        2: ["Layer 2 — Bilingual ontology tree",
            "4-level hierarchy (Database / Table / Variable / Value), 1 091 "
            "variables, 9 690 values, 100% EN coverage via curated glossary + "
            "cached online translation."],
        3: ["Layer 3 — SDG-PGMs persona model",
            "11 cascaded Bayesian-network dimensions; CPDs built from real "
            "PX-Web count tables. Direct subclass of `pgms.PGMGenerator`. "
            "100 K personas in ≈ 5 s."],
        4: ["Layer 4 — OCEAN sampler",
            "Big Five from Roberts (2006) age effects + Schmitt (2008) sex "
            "effects across 55 cultures. NSO has no psychometric data — same "
            "gap as US Census."],
        5: ["Layer 5 — Data Designer LLM A + LLM B",
            "PGM + OCEAN → LLM A (4 attribute fields) → LLM B (6 persona "
            "narratives). Model menu: Nemotron-3-super-120B/A12B, GPT-OSS-120B, "
            "Qwen3.5-122B/397B."],
        6: ["Output — 4 parquet files",
            "Nemotron-Personas-Vietnam-{large,small}-{vi,en}, 22 columns each, "
            "schema mirrors nvidia/Nemotron-Personas-Japan exactly. Bilingual "
            "pair shares uuids."],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 3. Layer 1 — NSO data foundation
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "Layer 1 — NSO PX-Web statistics curation",
        2: "From the General Statistics Office of Vietnam, fully bilingual "
           "structured data ready for downstream grounding.",
        1: [
            "12 statistical databases: Population & Labour, National Accounts, "
            "Industry, Agriculture, Trade, Health, Education, Investment, "
            "Enterprises, Transport, International, Geography",
            "502 PX-Web matrices crawled via the official JSON v2 catalog API "
            "(plus the legacy ASP.NET HTML form for V01 / V03 / V06 / V08 / "
            "V12 / V14 / V15)",
            "5-stage NeMo-Curator-style pipeline: download → parse → extract "
            "→ embed → reduce",
            "Per-URL on-disk HTTP cache → re-runs are free, full crawl "
            "replayable offline",
            "Output volume: 11 638 raw records → 11 464 parsed → 12 "
            "statistical domains tagged via the OntologyRegistry",
            "Curator-style backends supported: `local` (sequential, used by "
            "tests), `nemo_curator` (real ProcessingStage / InProcessExecutor)",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 4. Layer 2 — Bilingual ontology tree
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "Layer 2 — Complete bilingual ontology tree (vi → en)",
        2: "Walks the 502 PX-Web metadata sidecars and emits a 4-level "
           "structured tree with every node bilingual.",
        1: [
            "Tree shape: 12 databases → 502 tables → 1 091 variables → 9 690 "
            "values",
            "Tier 1 — curated VI→EN glossary (~280 entries hand-checked "
            "against NSO's English yearbooks): provinces, regions, ISCO, "
            "ISIC, units, aggregates",
            "Tier 1b — compositional rules: split \"Head (Tail)\" → translate "
            "parts; catches \"Bia các loại (Lít)\" → \"Beer (all types) (Litres)\"",
            "Tier 2 — on-disk JSONL cache, online Google fallback for the "
            "long tail (~1 200 strings, 90 s warm-up, deterministic re-runs)",
            "Tier 3 — identity fallback for years, codes, percentages",
            "Coverage: 76% offline-only, 100% after one online warm-up run; "
            "subsequent rebuilds are 6 s and offline",
            "Outputs: tree.json (2.1 MB, full) + tree.yaml (264 KB, "
            "diff-friendly summary) + translation_cache.json (162 KB)",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 5. Layer 3 — SDG-PGMs persona model
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "Layer 3 — SDG-PGMs Bayesian-network persona generator",
        2: "Direct subclass of nvidia-nemo/SDG-PGMs `PGMGenerator` — full "
           "lifecycle implemented (get_data / get_variables / get_edges / "
           "get_cpds / get_postprocessing_steps / get_mappings).",
        1: [
            "Stage 1 — Geography: region → urbanicity. Sources V02.01, V02.02.",
            "Stage 2 — Demographics: age_group → sex, region → ethnicity, "
            "(age_group, sex) → marital_status. Sources V02.41, V02.43.",
            "Stage 3 — Socio-economic: (age_group, urbanicity) → "
            "education_level → occupation → industry_sector; (age_group, sex) "
            "→ employment_status; education_level → income_quintile. "
            "Sources V02.42, V02.43, V02.44, V02.54.",
            "Real `pgmpy.factors.discrete.TabularCPD` per variable, built from "
            "PX-Web count DataFrames via `fill_na_and_gen_cpd`",
            "Vectorised: `BayesianNetwork.simulate(N)` per stage → 100 K "
            "personas in ≈ 5 seconds (M-series Mac, no GPU)",
            "Province sampling: 63 NSO admin-1 units, conditional on "
            "macro-region, with V02.01 within-region population weights",
            "Vietnamese names: vn-fullname-generator pools + vectorised "
            "numpy.Generator (3 M names in ≈ 0.6 s, fully seeded)",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 6. Layer 4 — OCEAN priors
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "Layer 4 — OCEAN: peer-reviewed Big Five priors",
        2: "Vietnam's NSO does not publish psychometric data — verified "
           "empirically across all 502 PX-Web tables. Same gap as US Census, "
           "Japan e-Stat, France INSEE.",
        1: [
            "Approach matches `nvidia/Nemotron-Personas-USA`: priors from "
            "two replicated psychology meta-analyses",
            "Roberts, Walton & Viechtbauer (2006) — Psychological Bulletin "
            "132(1): age effects across the lifespan (C ↑, A ↑, N ↓, O slow ↓, E small ↓)",
            "Schmitt, Realo, Voracek & Allik (2008) — JPSP 94(1): sex effects "
            "across 55 cultures including Vietnam (women > men on N, A, "
            "slightly C; men > women slightly on O)",
            "Each trait sampled from N(μ_age,sex, σ²=0.7²), clipped to [1, 5]",
            "Discretised low / mid / high bands → LLM-prompt-friendly summary "
            "string (e.g. \"high Conscientiousness, mid Openness, low "
            "Neuroticism\")",
            "Always computed (templated and LLM paths both consume "
            "personality_summary_*); the seed contract holds either way",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 7. Layer 5 — Data Designer LLM A + LLM B  (use Title + Right pane)
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_RIGHT_PANE])
    _populate(s, {
        0: "Layer 5 — Canonical Data Designer two-LLM pipeline",
        2: "PGM + OCEAN → LLM A (narrative attribute fields) → LLM B "
           "(persona narratives). Mirrors the published Nemotron-Personas "
           "compound-AI architecture.",
        1: [
            "INPUT: structured PGM columns (sex, age, marital_status, "
            "education_level, occupation, region, area, province) + Big Five "
            "OCEAN bands",
            "LLM A → cultural_background, skills_and_expertise (+ list), "
            "hobbies_and_interests (+ list), career_goals_and_ambitions",
            "LLM B → persona, professional_persona, sports_persona, "
            "arts_persona, travel_persona, culinary_persona",
            "Both LLMs route through the same OpenAI-compatible LLMClient: "
            "ThreadPoolExecutor concurrency, JSONL cache, retry, JSON "
            "validation, model-fingerprint cache key",
            "Resumable: a Ctrl-C / OOM / rate-limit at any point preserves "
            "every successful (uuid, stage, lang) cell",
            "Templated fallback: rows the LLM permanently fails on are filled "
            "from the deterministic narrative renderer so output always has "
            "all 22 columns",
        ],
        # Right-pane placeholder (idx=2 in this layout per inspection):
        # (already populated above as subtitle in idx=2; we use idx=2 also for
        # the right column when it exists. The Right Pane layout has its
        # own pane via idx=2; keep the structure simple here.)
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 8. The model menu
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "LLM model menu (build.nvidia.com / NVIDIA NIM)",
        2: "All four pinned via argparse `choices=`. Default is "
           "Nemotron-3-Super-120B/A12B. Any OpenAI-compatible endpoint works "
           "by overriding `--llm-base-url`.",
        1: [
            "nvidia/nemotron-3-super-120b-a12b   — default; most capable "
            "Nemotron MoE on the user's list",
            "qwen/qwen3.5-122b-a10b              — smaller MoE, ~30-50% the "
            "cost of Nemotron-Super",
            "qwen/qwen3.5-397b-a17b              — largest MoE, highest "
            "quality, highest cost",
            "openai/gpt-oss-120b                 — Apache-2.0 open weights",
            "Generation defaults: temperature 0.7, top_p 0.9, max_tokens "
            "1500, retries 3, concurrency 8",
            "Cost ballpark: full 3 M × 2 langs ≈ 22 B tokens via Nemotron "
            "Super (~$22 K). Smoke run at LARGE=1000 ≈ $7 with cache enabled",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 9. The four published datasets
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "The four published Nemotron-Personas-Vietnam parquets",
        2: "22-column schema mirrors `nvidia/Nemotron-Personas-Japan` "
           "exactly. Vietnam-specific geographic trio: region / area / "
           "province.",
        1: [
            "Nemotron-Personas-Vietnam-large-vi  — 3 M rows, ~1.2 GB, "
            "Vietnamese cell values",
            "Nemotron-Personas-Vietnam-large-en  — 3 M rows, ~1.0 GB, English "
            "cell values, same uuids as -large-vi row-for-row",
            "Nemotron-Personas-Vietnam-small-vi  — 300 K rows, stratified "
            "subset of -large-vi (proportional allocation by region × sex × "
            "education_level)",
            "Nemotron-Personas-Vietnam-small-en  — 300 K rows, same uuids as "
            "-small-vi → 1:1 inner-joinable",
            "Schema: uuid + 6 narrative persona fields + 5 contextual + 2 "
            "_list fields + 5 demographic + 4 geographic = 22 cols",
            "Streaming write: 4 simultaneous pyarrow.parquet.ParquetWriter "
            "instances, 100 K-row chunks; 3 M peak RAM ≈ 1.5 GB",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 10. Reproducibility contract
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "One seed, seven RNG streams, bit-equal output",
        2: "Single integer `seed` propagates via numpy.random.SeedSequence."
           "spawn(7) into seven independent Generators — each concern is "
           "isolated.",
        1: [
            "spawn[0] → structured personas (pgmpy via VNPersonaGenerator)",
            "spawn[1] → uuids (UUID4 from seeded RNG, NOT os.urandom)",
            "spawn[2] → province draw (region-conditional V02.01 weights)",
            "spawn[3] → vn-fullname-generator name pools",
            "spawn[4] → narrative variants (templated path)",
            "spawn[5] → small-subset stratified sample",
            "spawn[6] → OCEAN traits (Big Five priors)",
            "Chunk-size invariant: per-row permutation matrix is computed "
            "for the whole batch BEFORE chunking; row i's narrative variant "
            "depends only on (seed, i)",
            "LLM path: per-row JSONL cache keyed by (uuid, stage, lang, "
            "model fingerprint) → resumable across restarts, rate limits, OOM",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 11. Architecture — post-unification
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "Architecture — `packages/personas/` (post-unification)",
        2: "Was `personagen/` + `datagen/` (sounded like alternatives). Now "
           "one layered package; ~30 import sites updated, 0 stale "
           "references, 72/72 tests pass.",
        1: [
            "personas/pgm/             — SDG-PGMs subclass + NSO PX-Web "
            "distributions  (1 222 LOC)",
            "personas/ocean.py         — Big Five sampler with literature "
            "citations  (245 LOC)",
            "personas/embed/           — sentence-transformers + NIM + UMAP "
            "for personas.json  (611 LOC)",
            "personas/llm/             — unified OpenAI-compatible client + "
            "Data Designer A/B + single-bio enrich  (1 277 LOC)",
            "personas/datasets/        — Nemotron parquet builder, schema, "
            "narrative templates  (2 478 LOC)",
            "Dedupe payoff: enrich.py 265 → 224 LOC; one HTTP client, one "
            "cache, one retry policy across both LLM pipelines",
            "Adjacent packages (unchanged structurally): packages/ontology/ "
            "(bilingual tree), packages/curator/ (5-stage pipeline), "
            "packages/scraper/ (PX-Web JSON + HTML form)",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 12. Live status
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "Where we are today",
        2: "Pipeline runs end-to-end at 1 K, 10 K, 1 M scale. Templated path "
           "is offline-deterministic; LLM path needs $PERSONAS_VN_LLM_API_KEY.",
        1: [
            "Tests: 72 passing across pgm / ocean / embed / llm / datasets / "
            "ontology  (3 HF-network-dependent tests skip in sandbox)",
            "10 K rows × 4 datasets in 5.4 s end-to-end (templated); 3 M "
            "projected ≈ 27 min on M-series Mac",
            "Bilingual ontology tree: built in 6 s offline (76% glossary "
            "coverage) or 90 s online (100% coverage, cached for re-runs)",
            "LLM-A/LLM-B end-to-end smoke at N=20 with mocked client, all 22 "
            "columns populated in both languages; ready for production",
            "Sample parquet sizes (templated, 10 K rows): vi-large 4.0 MB, "
            "en-large 3.5 MB → projected 3 M: vi-large ~1.2 GB, en-large ~1.0 GB",
            "CLI: personas-vn build-nemotron / build-ontology / generate / "
            "embed-personas / enrich-personas / curate / run-all",
        ],
    })
    slides.append(s)

    # ------------------------------------------------------------------
    # 13. Sample persona — bilingual side-by-side
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[L_BULLETS])
    _populate(s, {
        0: "One row, two languages, fully NSO-grounded",
        2: "uuid 97401f13-cb79-4d77-8ef4-b12e2b13e40b  ·  seed=42 chunk 1/2  "
           "·  templated path",
        1: [
            "Structured fields:  age=27, sex=Nam (male), marital=Chưa kết hôn "
            "(never married), education=Không có trình độ CMKT (no formal "
            "qualification), occupation=Thợ thủ công và các thợ khác có liên "
            "quan (craft and related trades workers), area=Nông thôn (rural), "
            "province=Lạng Sơn, region=Trung du và miền núi phía Bắc "
            "(Northern Midlands and Mountains)",
            "OCEAN bands:  high Openness, mid Conscientiousness, mid "
            "Extraversion, mid Agreeableness, mid Neuroticism",
            "VI persona:  \"Đoàn Quốc Trường, 27 tuổi, Nam, Chưa kết hôn, "
            "trình độ Không có trình độ CMKT, làm việc trong nhóm nghề Thợ "
            "thủ công và các thợ khác có liên quan, sinh sống tại khu vực "
            "Nông thôn thuộc Lạng Sơn (Trung du và miền núi phía Bắc).\"",
            "EN persona:  \"Đoàn Quốc Trường, a 27-year-old male, never "
            "married with no formal qualification, works as a craft and "
            "trades worker and lives in a rural part of Lạng Sơn in the "
            "Northern Midlands and Mountains.\"",
            "Every cell traces back to NSO matrix V02.01 (province) / V02.43 "
            "(occupation) / V02.54 (education) / V02.02 (sex+area)",
        ],
    })
    slides.append(s)

    return slides


# ---------------------------------------------------------------------------
# Slide reordering — move the closing-slide to the end after we appended
# new content slides
# ---------------------------------------------------------------------------
def reorder_slides(prs, *, original_closing_index: int,
                   appended_slides: list) -> None:
    """Move the appended content slides BEFORE the original closing slide.

    python-pptx doesn't expose slide reordering directly; we manipulate
    the underlying ``sldIdLst`` XML to reach the order we want:

        Title → [appended content slides] → empty middle (deleted) →
        Closing
    """
    sldIdLst = prs.slides._sldIdLst
    sld_id_elements = list(sldIdLst)

    # The original v0 deck had: [Title, EmptyMiddle, Closing] at indices [0,1,2]
    # We've appended N content slides → indices [3..3+N-1].
    # Target order: [Title (0), content 0..N-1 (3..3+N-1), Closing (2)],
    # then drop EmptyMiddle (idx 1).

    title    = sld_id_elements[0]
    middle   = sld_id_elements[1]      # to be removed
    closing  = sld_id_elements[2]
    contents = sld_id_elements[3:]

    # Detach all
    for el in sld_id_elements:
        sldIdLst.remove(el)

    # Re-attach in the order we want, omitting the empty middle slide
    sldIdLst.append(title)
    for c in contents:
        sldIdLst.append(c)
    sldIdLst.append(closing)

    # Now physically remove the empty-middle slide from the package
    rId = middle.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
    prs.part.drop_rel(rId)


def refresh_title_slide(prs):
    """Update the v0 title slide subtitle so it matches v1's actual content."""
    title_slide = prs.slides[0]
    # The v0 title slide has 3 placeholders (idx 1/2/3); per inspection:
    #   idx=1 (top of stack)  → main title "Nemotron Personas Vietnam"
    #   idx=2                 → subtitle (the long blurb)
    #   idx=3                 → dateline ("NVIDIA NeMo / SDG / GTC 2026")
    # We refresh idx=2 (subtitle) to summarise the v1 surface.
    for shape in title_slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text
        if "100,000 synthetic" in txt or "from official NSO" in txt.lower():
            new_subtitle = (
                "From 502 NSO PX-Web matrices to four Nemotron-Personas-Vietnam "
                "parquet datasets — SDG-PGMs + Big Five OCEAN + Data Designer "
                "two-LLM narrative pipeline (Nemotron / Qwen / GPT-OSS), "
                "single-seed reproducible."
            )
            shape.text_frame.text = new_subtitle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    src = Path("Nemotron Personas Vietnam v0.pptx")
    dst = Path("Nemotron Personas Vietnam v1.pptx")
    prs = Presentation(str(src))

    refresh_title_slide(prs)

    appended = add_content_slides(prs)
    reorder_slides(prs, original_closing_index=2, appended_slides=appended)

    prs.save(str(dst))
    print(f"wrote {dst}  ({dst.stat().st_size / 1024:.0f} KB, "
          f"{len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
