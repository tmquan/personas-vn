"""End-to-end pipeline test that runs the full scrape+generate cycle
against the bundled fixture (no network).

We assert the pipeline writes the expected files and that the persona JSON
round-trips through the typed Pydantic model.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from packages.common.paths import REPO_ROOT, resolve
from packages.ontology import PersonaBatch
from packages.pipeline import run_all


@pytest.fixture
def tmp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build an isolated workspace that mirrors the repo's config layout
    but writes outputs into ``tmp_path``. We monkey-patch ``REPO_ROOT`` so
    ``resolve()`` lands inside the temp dir.
    """
    # Copy configs/ into the temp workspace.
    cfg_src = REPO_ROOT / "configs"
    cfg_dst = tmp_path / "configs"
    shutil.copytree(cfg_src, cfg_dst)

    # Re-write the scraper config to force the offline path: a clearly
    # bogus base URL means every live call fails, then `offline_fallback`
    # kicks in.
    scraper_yaml = cfg_dst / "scraper.yaml"
    cfg = yaml.safe_load(scraper_yaml.read_text())
    cfg["source"]["base_url"] = "https://localhost.invalid"
    cfg["fetch"]["retries"] = 1
    cfg["fetch"]["request_timeout_s"] = 2
    cfg["offline_fallback"] = True
    scraper_yaml.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr("packages.common.paths.REPO_ROOT", tmp_path, raising=False)
    # The Config / paths modules cache nothing besides the registry; clear
    # the registry's lru_cache so the fresh YAML files are re-read.
    from packages.ontology import registry as _reg

    _reg.load_registry.cache_clear()
    return tmp_path


def test_run_all_offline_writes_expected_artefacts(tmp_workspace: Path):
    artefacts = run_all(
        n=25,
        seed=123,
        scraper_config=str(tmp_workspace / "configs" / "scraper.yaml"),
        ontology_config=str(tmp_workspace / "configs" / "ontology.yaml"),
    )
    # The ontology dataset file exists and has > 0 records.
    assert artefacts.datasets_path is not None
    payload = json.loads(artefacts.datasets_path.read_text(encoding="utf-8"))
    assert payload["datasets"], "no datasets written"
    assert payload["scrape_manifest"].get("used_fallback") is True

    # Personas round-trip cleanly.
    assert artefacts.personas_path is not None
    batch = PersonaBatch.model_validate_json(artefacts.personas_path.read_text(encoding="utf-8"))
    assert batch.n == 25
    assert len(batch.personas) == 25
    assert batch.seed == 123

    # And the in-memory registry still resolves.
    assert resolve("data") == tmp_workspace / "data"
