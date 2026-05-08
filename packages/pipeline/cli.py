"""Tiny stdlib-only CLI for the pipeline.

We avoid Click / Typer to keep dependencies lean — all we need is three
subcommands.

Usage:

    python -m packages.pipeline.cli scrape
    python -m packages.pipeline.cli generate --n 500 --seed 42
    python -m packages.pipeline.cli run-all  --n 1000

Or via the installed entry point:

    personas-vn scrape
    personas-vn generate --n 500
    personas-vn run-all
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from packages.common.logging import get_logger
from packages.pipeline.runner import run_all, run_generate, run_scrape

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="personas-vn",
        description="NSO scraper + ontology + persona generator + curator pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scrape", help="Scrape NSO and write ontology/datasets.json")
    s.add_argument("--scraper-config", default="configs/scraper.yaml")
    s.add_argument("--ontology-config", default="configs/ontology.yaml")

    g = sub.add_parser("generate", help="Generate synthetic Vietnamese personas")
    g.add_argument("--n", type=int, default=None, help="Number of personas to generate")
    g.add_argument("--seed", type=int, default=None, help="Override the RNG seed")
    g.add_argument("--ontology-config", default="configs/ontology.yaml")

    a = sub.add_parser("run-all", help="Scrape then generate")
    a.add_argument("--n", type=int, default=None)
    a.add_argument("--seed", type=int, default=None)
    a.add_argument("--scraper-config", default="configs/scraper.yaml")
    a.add_argument("--ontology-config", default="configs/ontology.yaml")

    e = sub.add_parser(
        "embed-personas",
        help="Embed generated persona bios + project to 2-D for the visualizer",
    )
    e.add_argument("--personas", default="data/personas/personas.json",
                   help="Path to a persisted PersonaBatch JSON file")
    e.add_argument("--out-dir", default=None,
                   help="Where to write embedded.parquet + reduced.parquet "
                        "(default: ontology config's output.personas_dir)")
    e.add_argument("--max-records", type=int, default=10000,
                   help="Cap personas embedded (random sample). 0 = all.")
    e.add_argument("--cluster", action="store_true", default=False,
                   help="Opt in to density-based HDBSCAN clustering (off by "
                        "default — only the 2-D UMAP coords are written). "
                        "When on, adds a ``cluster`` column to reduced.parquet.")
    e.add_argument("--min-cluster-size", type=int, default=None,
                   help="HDBSCAN min_cluster_size when --cluster is on "
                        "(default: auto, max(5, n // 80)). Smaller values "
                        "yield more, finer-grained clusters.")
    e.add_argument("--model", default=None,
                   help="Embedding model id. Default: sentence-transformers/"
                        "paraphrase-multilingual-MiniLM-L12-v2 (local, 384-d). "
                        "NIM options: nvidia/llama-3.2-nv-embedqa-1b-v2, "
                        "nvidia/llama-nemotron-embed-1b-v2, "
                        "nvidia/llama-3.2-nemoretriever-300m-embed-v1.")
    e.add_argument("--backend", default="auto",
                   choices=("auto", "local", "nim"),
                   help="auto = pick from model prefix (nvidia/* → nim, else local)")
    e.add_argument("--base-url", default=None,
                   help="NIM-only: override the OpenAI-compatible base URL "
                        "(default https://integrate.api.nvidia.com/v1)")
    e.add_argument("--api-key-env", default="PERSONAS_VN_LLM_API_KEY",
                   help="NIM-only: env var holding the API key "
                        "(falls back to $NVIDIA_API_KEY if unset)")
    e.add_argument("--ontology-config", default="configs/ontology.yaml")

    enrich = sub.add_parser(
        "enrich-personas",
        help="Rewrite persona bios with an LLM (bilingual: bio_vi + bio_en)",
    )
    enrich.add_argument("--personas", default="data/personas/personas.json",
                        help="Path to a persisted PersonaBatch JSON file")
    enrich.add_argument("--out", default="data/personas/personas_enriched.json",
                        help="Where to write the enriched PersonaBatch")
    enrich.add_argument("--max-records", type=int, default=50,
                        help="Cap personas enriched (random sample). 0 = all.")
    enrich.add_argument("--model", default=None,
                        help="Override the LLM model (e.g. nvidia/nemotron-3-super-49b-instruct-v2, "
                             "openai/gpt-oss-120b, qwen/qwen3-coder-122b-a10b)")
    enrich.add_argument("--base-url", default=None,
                        help="Override the OpenAI-compatible API base URL")
    enrich.add_argument("--api-key-env", default="PERSONAS_VN_LLM_API_KEY",
                        help="Environment variable that holds the API key")
    enrich.add_argument("--temperature", type=float, default=0.7)
    enrich.add_argument("--max-tokens", type=int, default=600)

    bo = sub.add_parser(
        "build-ontology",
        help="Walk the NSO PX-Web parquet metadata and emit a bilingual "
             "(VI + EN) 4-level ontology tree (database → table → variable "
             "→ value) with curated glossary translation; optionally fall "
             "through to Google Translate for the long tail.",
    )
    bo.add_argument("--pxweb-root", default="data/nso-gov-vn/raw/pxweb/vi",
                    help="Directory containing the *.metadata.json sidecars")
    bo.add_argument("--out-dir", default="data/ontology",
                    help="Where to write tree.json + tree.yaml + "
                         "translation_cache.json")
    bo.add_argument("--online", action="store_true",
                    help="Enable Google Translate fallback for strings not "
                         "covered by the curated glossary. Requires "
                         "`pip install deep-translator` and network access. "
                         "First run takes ~5–10 min for the long tail; "
                         "results cache to disk so subsequent runs are free.")
    bo.add_argument("--workers", type=int, default=8,
                    help="Online tier: number of concurrent Google requests")

    bn = sub.add_parser(
        "build-nemotron",
        help="Build the four Nemotron-Personas-Vietnam parquet datasets "
             "(large/small × vi/en) from NSO PX-Web statistics.",
    )
    bn.add_argument("--out-dir", default="data/Nemotron-Personas-Vietnam",
                    help="Directory to write the four parquet files into")
    bn.add_argument("--large-size", type=int, default=3_000_000,
                    help="Row count for the *-large-* datasets (default 3M)")
    bn.add_argument("--small-size", type=int, default=300_000,
                    help="Row count for the *-small-* datasets, sampled "
                         "stratified-by-region from large (default 300K)")
    bn.add_argument("--chunk-size", type=int, default=100_000,
                    help="Chunk size for the streaming parquet writer "
                         "(higher = more RAM, fewer flushes; default 100K)")
    bn.add_argument("--seed", type=int, default=42,
                    help="Single integer seed; all internal RNG streams "
                         "(structured personas, uuids, provinces, names, "
                         "narrative variants, subset selection, OCEAN) "
                         "derive from this via numpy SeedSequence.spawn for "
                         "fully reproducible output")
    bn.add_argument("--pxweb-root", default=None,
                    help="Override the NSO PX-Web parquet directory "
                         "(default: data/nso-gov-vn/raw/pxweb/vi)")
    bn.add_argument("--compression", default="snappy",
                    help="Parquet compression codec (default: snappy, "
                         "matching the published Nemotron-Personas datasets)")
    # ---- LLM pipeline (canonical Data Designer PGM + OCEAN → LLM A → LLM B) ----
    bn.add_argument("--llm", action="store_true",
                    help="Use the canonical two-LLM Data Designer pipeline "
                         "(PGM + OCEAN → LLM A → LLM B) instead of the "
                         "deterministic templated narrative renderer. "
                         "Requires $PERSONAS_VN_LLM_API_KEY to be set.")
    bn.add_argument(
        "--llm-model",
        default="nvidia/nemotron-3-super-120b-a12b",
        choices=[
            "nvidia/nemotron-3-super-120b-a12b",
            "qwen/qwen3.5-122b-a10b",
            "qwen/qwen3.5-397b-a17b",
            "openai/gpt-oss-120b",
        ],
        help="LLM model id. Defaults to NVIDIA Nemotron-3 Super 120B/A12B.",
    )
    bn.add_argument("--llm-base-url", default="https://integrate.api.nvidia.com/v1",
                    help="OpenAI-compatible base URL "
                         "(default: NVIDIA NIM at integrate.api.nvidia.com)")
    bn.add_argument("--llm-api-key-env", default="PERSONAS_VN_LLM_API_KEY",
                    help="Env var holding the API key (also tries $NVIDIA_API_KEY)")
    bn.add_argument("--llm-workers", type=int, default=8,
                    help="Concurrent in-flight LLM requests (default 8). "
                         "Higher = faster but more rate-limit risk.")
    bn.add_argument("--llm-temperature", type=float, default=0.7)
    bn.add_argument("--llm-max-tokens", type=int, default=1500)
    bn.add_argument("--llm-cache-dir", default=None,
                    help="JSONL cache directory (default: <out-dir>/_llm_cache). "
                         "Cache makes runs resumable: a Ctrl-C mid-build can "
                         "be resumed without re-paying for completed rows.")

    c = sub.add_parser(
        "curate",
        help="Run the staged NSO curation pipeline (download → parse → extract → embed → reduce)",
    )
    c.add_argument("--curator-config", default="configs/curator.yaml")
    c.add_argument("--only", nargs="+", default=None,
                   choices=("download", "parse", "extract", "embed", "reduce"),
                   help="Only run these stages")
    c.add_argument("--skip", nargs="+", default=None,
                   choices=("download", "parse", "extract", "embed", "reduce"),
                   help="Skip these stages")
    c.add_argument("--backend", default="local", choices=("local", "nemo_curator"),
                   help="Execution backend; nemo_curator wraps each stage as a ProcessingStage")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "scrape":
        artefacts = run_scrape(
            scraper_config=args.scraper_config,
            ontology_config=args.ontology_config,
        )
        log.info("scrape complete: datasets=%s manifest=%s",
                 artefacts.datasets_path, artefacts.manifest_path)
    elif args.command == "generate":
        artefacts = run_generate(
            n=args.n,
            seed=args.seed,
            ontology_config=args.ontology_config,
        )
        log.info("generate complete: personas=%s", artefacts.personas_path)
    elif args.command == "run-all":
        artefacts = run_all(
            n=args.n,
            seed=args.seed,
            scraper_config=args.scraper_config,
            ontology_config=args.ontology_config,
        )
        log.info("run-all complete: datasets=%s personas=%s",
                 artefacts.datasets_path, artefacts.personas_path)
    elif args.command == "embed-personas":
        from packages.personas.pgm import embed_personas_from_disk

        summary = embed_personas_from_disk(
            personas_path=args.personas,
            output_dir=args.out_dir,
            ontology_config=args.ontology_config,
            max_records=(args.max_records if args.max_records > 0 else None),
            cluster=args.cluster,
            min_cluster_size=args.min_cluster_size,
            model=args.model,
            backend=args.backend,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
        log.info("embed-personas complete: %s", summary)
    elif args.command == "enrich-personas":
        from packages.personas.llm.enrich import (
            LLMEnrichConfig,
            enrich_personas_from_disk,
        )

        cfg_kwargs: dict[str, Any] = {
            "max_records": args.max_records if args.max_records > 0 else None,
            "api_key_env": args.api_key_env,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.model:
            cfg_kwargs["model"] = args.model
        if args.base_url:
            cfg_kwargs["base_url"] = args.base_url
        cfg = LLMEnrichConfig(**cfg_kwargs)
        summary = enrich_personas_from_disk(
            personas_path=args.personas,
            out_path=args.out,
            config=cfg,
        )
        log.info("enrich-personas complete: %s", summary)
    elif args.command == "build-ontology":
        from packages.ontology.tree import build_and_write_ontology

        summary = build_and_write_ontology(
            pxweb_root=args.pxweb_root,
            out_dir=args.out_dir,
            enable_online=args.online,
            online_workers=args.workers,
        )
        log.info("build-ontology complete: %s", summary)
    elif args.command == "build-nemotron":
        from packages.personas.datasets import build_nemotron_personas_vietnam_datasets

        llm_pipeline_config = None
        if args.llm:
            import os

            from packages.personas.llm.client import LLMConfig
            from packages.personas.llm.pipeline import LLMPipelineConfig

            api_key_env = args.llm_api_key_env
            if not os.environ.get(api_key_env) and os.environ.get("NVIDIA_API_KEY"):
                # Convenience: pick up $NVIDIA_API_KEY too.
                api_key_env = "NVIDIA_API_KEY"

            cache_dir = args.llm_cache_dir or f"{args.out_dir}/_llm_cache"
            llm_pipeline_config = LLMPipelineConfig(
                llm=LLMConfig(
                    model=args.llm_model,
                    base_url=args.llm_base_url,
                    api_key_env=api_key_env,
                    temperature=args.llm_temperature,
                    max_tokens=args.llm_max_tokens,
                    concurrency=args.llm_workers,
                ),
                cache_dir=cache_dir,
            )

        summary = build_nemotron_personas_vietnam_datasets(
            out_dir=args.out_dir,
            large_size=args.large_size,
            small_size=args.small_size,
            chunk_size=args.chunk_size,
            seed=args.seed,
            pxweb_root=args.pxweb_root,
            parquet_compression=args.compression,
            use_llm_pipeline=args.llm,
            llm_pipeline_config=llm_pipeline_config,
        )
        log.info("build-nemotron complete: paths=%s rows=%s",
                 summary["paths"], summary["rows"])
    elif args.command == "curate":
        # Local import keeps the curator stack (torch, sentence-transformers,
        # umap-learn, ...) optional for users who only want the persona side.
        from packages.curator import run_curation

        artefacts = run_curation(
            config_path=args.curator_config,
            only=args.only,
            skip=args.skip,
            backend=args.backend,
        )
        log.info("curate complete: root=%s manifest=%s",
                 artefacts.root, artefacts.manifest_path)
    else:  # pragma: no cover - argparse ensures one of the above
        parser.error(f"unknown command: {args.command}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
