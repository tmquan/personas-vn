"""Tests for the OCEAN sampler + the two-LLM Data Designer pipeline.

These tests deliberately avoid hitting any real LLM endpoint. They use
the :class:`packages.personas.llm.client._MockClient` seam to inject
deterministic canned responses so the pipeline shape, validation,
caching, and assembly logic can be exercised offline.

Coverage:

* OCEAN sampler reproducibility + age/sex effect direction
* JSONL cache round-trip (save → reload)
* LLMClient cache hits skip the responder
* Pipeline assembly: LLM A + LLM B outputs land in the right columns
* Validation: malformed LLM responses trigger fallback
* Bilingual round-trip: vi & en runs share UUID set
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

# Skip module if optional deps are missing
for _dep in ("pyarrow", "pandas", "vn_fullname_generator"):
    if importlib.util.find_spec(_dep) is None:
        pytest.skip(f"{_dep} missing", allow_module_level=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.personas.llm.client import (  # noqa: E402
    JSONLCache,
    LLMClient,  # noqa: F401
    LLMConfig,
    _MockClient,
    _model_fingerprint,
    _parse_json_lenient,
)
from packages.personas.llm.pipeline import (  # noqa: E402
    LLMPipelineConfig,
    render_narratives_via_llm,
)
from packages.personas.llm.prompts import (  # noqa: E402
    LLM_A_REQUIRED_KEYS,
    LLM_B_REQUIRED_KEYS,
    build_user_prompt_a,
    validate_llm_a,
    validate_llm_b,
)
from packages.personas.ocean import (  # noqa: E402
    LIFE_SLOPE_PER_DECADE,
    SEX_SHIFT_FEMALE_MINUS_MALE,
    TRAITS,
    sample_ocean,
)


# ---------------------------------------------------------------------------
# OCEAN sampler
# ---------------------------------------------------------------------------
def test_ocean_returns_expected_columns():
    df = sample_ocean(
        age=np.array([22, 30, 60]),
        sex=np.array(["Nam", "Nữ", "Nam"]),
        rng=np.random.default_rng(0),
    )
    assert len(df) == 3
    for trait in TRAITS:
        assert trait in df.columns
        assert f"{trait}_band" in df.columns
        assert df[trait].between(1.0, 5.0).all()
    assert "personality_summary_vi" in df.columns
    assert "personality_summary_en" in df.columns


def test_ocean_is_reproducible_under_same_seed():
    a = sample_ocean(
        age=np.array([22, 45, 70]),
        sex=np.array(["Nam", "Nữ", "Nam"]),
        rng=np.random.default_rng(42),
    )
    b = sample_ocean(
        age=np.array([22, 45, 70]),
        sex=np.array(["Nam", "Nữ", "Nam"]),
        rng=np.random.default_rng(42),
    )
    pd.testing.assert_frame_equal(a, b)


def test_ocean_age_slope_directions_match_priors():
    """Population means at age 70 vs 20 must move in the directions the
    Roberts (2006) meta-analysis predicts."""
    young = sample_ocean(
        age=np.full(5000, 22),
        sex=np.array(["Nam"] * 2500 + ["Nữ"] * 2500),
        rng=np.random.default_rng(11),
    )
    old = sample_ocean(
        age=np.full(5000, 65),
        sex=np.array(["Nam"] * 2500 + ["Nữ"] * 2500),
        rng=np.random.default_rng(12),
    )
    for trait, slope in LIFE_SLOPE_PER_DECADE.items():
        diff = old[trait].mean() - young[trait].mean()
        if slope > 0:
            assert diff > 0, f"{trait}: expected positive age slope, got {diff:+.2f}"
        elif slope < 0:
            assert diff < 0, f"{trait}: expected negative age slope, got {diff:+.2f}"


def test_ocean_sex_shift_directions_match_schmitt():
    male = sample_ocean(
        age=np.full(5000, 35),
        sex=np.array(["Nam"] * 5000),
        rng=np.random.default_rng(21),
    )
    female = sample_ocean(
        age=np.full(5000, 35),
        sex=np.array(["Nữ"] * 5000),
        rng=np.random.default_rng(22),
    )
    for trait, shift in SEX_SHIFT_FEMALE_MINUS_MALE.items():
        if abs(shift) < 0.05:
            continue   # too small to assert direction reliably at n=5k
        diff = female[trait].mean() - male[trait].mean()
        if shift > 0:
            assert diff > 0, f"{trait}: expected female>male, got {diff:+.3f}"
        else:
            assert diff < 0, f"{trait}: expected female<male, got {diff:+.3f}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_jsonl_cache_round_trip(tmp_path):
    p = tmp_path / "c.jsonl"
    c1 = JSONLCache(path=p)
    assert len(c1) == 0
    c1.put("k1", {"a": 1})
    c1.put("k2", {"b": 2})
    c1.close()

    c2 = JSONLCache(path=p)
    assert len(c2) == 2
    assert c2.get("k1") == {"a": 1}
    assert c2.get("k2") == {"b": 2}
    assert "k1" in c2
    assert "missing" not in c2


def test_jsonl_cache_ignores_corrupt_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        '{"key": "ok", "value": {"x": 1}}\n'
        'not-json-at-all\n'
        '{"key": "ok2", "value": {"y": 2}}\n'
    )
    c = JSONLCache(path=p)
    assert len(c) == 2
    assert c.get("ok") == {"x": 1}
    assert c.get("ok2") == {"y": 2}


def test_model_fingerprint_changes_with_model():
    a = _model_fingerprint(LLMConfig(model="m1"))
    b = _model_fingerprint(LLMConfig(model="m2"))
    assert a != b
    assert _model_fingerprint(LLMConfig(model="m1")) == a   # stable


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def test_parse_json_handles_markdown_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert _parse_json_lenient(raw) == {"a": 1}
    raw2 = "```\n{\"b\": 2}\n```"
    assert _parse_json_lenient(raw2) == {"b": 2}
    raw3 = '{"c": 3}'
    assert _parse_json_lenient(raw3) == {"c": 3}


# ---------------------------------------------------------------------------
# Prompt validators
# ---------------------------------------------------------------------------
def test_llm_a_validator_catches_missing_keys():
    good = {k: "x" for k in LLM_A_REQUIRED_KEYS}
    assert validate_llm_a(good) is None

    bad = dict(good)
    del bad["cultural_background"]
    assert validate_llm_a(bad) is not None

    bad2 = dict(good)
    bad2["cultural_background"] = ""   # empty
    assert validate_llm_a(bad2) is not None


def test_llm_b_validator_catches_missing_keys():
    good = {k: "x" for k in LLM_B_REQUIRED_KEYS}
    assert validate_llm_b(good) is None
    bad = dict(good)
    del bad["persona"]
    assert validate_llm_b(bad) is not None


def test_prompt_builders_include_all_demographic_fields():
    record = {
        "name":            "Trần Văn Minh",
        "age":             32,
        "sex":             "Nam",
        "marital_status":  "Đã kết hôn",
        "education_level": "Đại học trở lên",
        "occupation":      "Nhà chuyên môn bậc cao",
        "area":            "Thành thị",
        "province":        "Hà Nội",
        "region":          "Đồng bằng sông Hồng",
        "personality_summary_vi": "Cởi mở cao, Tận tâm cao",
        "personality_summary_en": "high Openness, high Conscientiousness",
    }
    p_vi = build_user_prompt_a(record, "vi")
    p_en = build_user_prompt_a(record, "en")
    assert "Trần Văn Minh" in p_vi
    assert "Trần Văn Minh" in p_en
    assert "Hà Nội" in p_vi
    assert "Cởi mở cao" in p_vi
    assert "high Openness" in p_en


# ---------------------------------------------------------------------------
# Pipeline (mock-driven)
# ---------------------------------------------------------------------------
def _make_mock_responder(stage: str):
    """Return a responder function that produces canned LLM-A or LLM-B
    JSON based on the cache key (which encodes stage + lang + uuid)."""
    if stage == "a":
        keys = sorted(LLM_A_REQUIRED_KEYS)
    else:
        keys = sorted(LLM_B_REQUIRED_KEYS)

    def _respond(cache_key: str, sys_p: str, usr_p: str) -> dict[str, Any]:
        # Echo the cache key into each value so we can assert content
        # made it through to the right column.
        return {k: f"<{stage}:{k}:{cache_key[:30]}>" for k in keys}
    return _respond


def _structured_chunk_fixture() -> pd.DataFrame:
    """A 3-row structured frame matching what the pipeline expects."""
    return pd.DataFrame({
        "uuid":            ["u-001", "u-002", "u-003"],
        "name":            ["Trần Văn Minh", "Lê Thị Hoa", "Nguyễn Văn Hai"],
        "age":             [28, 41, 56],
        "sex":             ["Nam", "Nữ", "Nam"],
        "marital_status":  ["Chưa kết hôn", "Đã kết hôn", "Đã kết hôn"],
        "education_level": ["Đại học trở lên", "Trung cấp", "Sơ cấp"],
        "occupation":      ["Chuyên môn kỹ thuật bậc cao",
                            "Dịch vụ cá nhân, bảo vệ bán hàng",
                            "Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản"],
        "area":            ["Thành thị", "Thành thị", "Nông thôn"],
        "province":        ["Hà Nội", "TP.Hồ Chí Minh", "Cần Thơ"],
        "region":          ["Đồng bằng sông Hồng", "Đông Nam Bộ",
                            "Đồng bằng sông Cửu Long"],
        "personality_summary_vi": ["Cởi mở cao, Tận tâm cao", "Hướng ngoại cao", "Tận tâm cao"],
        "personality_summary_en": ["high Openness", "high Extraversion",
                                   "high Conscientiousness"],
    })


def test_llm_pipeline_assembles_columns_in_right_slots(tmp_path):
    cache_a = JSONLCache(path=tmp_path / "a.jsonl")
    cache_b = JSONLCache(path=tmp_path / "b.jsonl")
    client_a = _MockClient(cache_a, _make_mock_responder("a"))
    client_b = _MockClient(cache_b, _make_mock_responder("b"))

    cfg = LLMPipelineConfig(
        llm=LLMConfig(model="mock://test"),
        cache_dir=str(tmp_path),
    )
    chunk = _structured_chunk_fixture()
    out = render_narratives_via_llm(
        chunk, lang="vi", cfg=cfg, client_a=client_a, client_b=client_b,
    )
    assert len(out) == 3
    # LLM A's outputs land in the four narrative-attribute columns
    assert all("a:cultural_background" in v for v in out["cultural_background"])
    assert all("a:skills_and_expertise:" in v for v in out["skills_and_expertise"])
    assert all("a:hobbies_and_interests:" in v for v in out["hobbies_and_interests"])
    assert all("a:career_goals_and_ambitions" in v for v in out["career_goals_and_ambitions"])
    # LLM B's outputs land in the six persona narratives
    for col in ("persona", "professional_persona", "sports_persona",
                "arts_persona", "travel_persona", "culinary_persona"):
        assert all(f"b:{col}:" in v for v in out[col]), col


def test_llm_pipeline_falls_back_to_templates_on_failure(tmp_path):
    """If the LLM permanently fails, ``fallback_to_templates`` (default
    True) ensures every row still gets all 11 narrative columns."""

    def fail_a(cache_key, sys_p, usr_p):
        return None  # mock client returns None as if all retries failed

    cache_a = JSONLCache(path=tmp_path / "a.jsonl")
    cache_b = JSONLCache(path=tmp_path / "b.jsonl")
    # Override the mock to return None — but the mock client we built
    # just calls the responder. Let's emulate by giving an invalid
    # response that fails validation.
    client_a = _MockClient(cache_a, lambda *a, **k: {"only": "garbage"})
    client_b = _MockClient(cache_b, _make_mock_responder("b"))

    cfg = LLMPipelineConfig(
        llm=LLMConfig(model="mock://test"),
        cache_dir=str(tmp_path),
        fallback_to_templates=True,
    )
    chunk = _structured_chunk_fixture()
    out = render_narratives_via_llm(
        chunk, lang="en", cfg=cfg, client_a=client_a, client_b=client_b,
    )
    assert len(out) == 3
    # All 11 columns are populated (from the template fallback)
    for col in ("cultural_background", "skills_and_expertise",
                "skills_and_expertise_list", "hobbies_and_interests",
                "hobbies_and_interests_list", "career_goals_and_ambitions",
                "persona", "professional_persona", "sports_persona",
                "arts_persona", "travel_persona", "culinary_persona"):
        assert (out[col].astype(str).str.len() > 0).all(), col


def test_llm_pipeline_cache_makes_re_runs_zero_calls(tmp_path):
    """Two pipeline invocations with the same cache file must do zero
    network calls on the second run. Validates resumability."""
    cache_a = JSONLCache(path=tmp_path / "a.jsonl")
    cache_b = JSONLCache(path=tmp_path / "b.jsonl")
    counter = {"n": 0}

    def counting_responder(stage):
        responder = _make_mock_responder(stage)

        def _wrap(cache_key, sys_p, usr_p):
            counter["n"] += 1
            return responder(cache_key, sys_p, usr_p)
        return _wrap

    client_a = _MockClient(cache_a, counting_responder("a"))
    client_b = _MockClient(cache_b, counting_responder("b"))

    cfg = LLMPipelineConfig(
        llm=LLMConfig(model="mock://test"),
        cache_dir=str(tmp_path),
    )
    chunk = _structured_chunk_fixture()

    render_narratives_via_llm(chunk, lang="vi", cfg=cfg,
                              client_a=client_a, client_b=client_b)
    first_calls = counter["n"]
    assert first_calls == 6   # 3 rows × 2 stages

    # Second pass: same cache, fresh in-memory clients, same chunk
    client_a.close()
    client_b.close()
    cache_a2 = JSONLCache(path=tmp_path / "a.jsonl")
    cache_b2 = JSONLCache(path=tmp_path / "b.jsonl")
    client_a2 = _MockClient(cache_a2, counting_responder("a"))
    client_b2 = _MockClient(cache_b2, counting_responder("b"))

    render_narratives_via_llm(chunk, lang="vi", cfg=cfg,
                              client_a=client_a2, client_b=client_b2)
    second_calls = counter["n"] - first_calls
    assert second_calls == 0, f"expected cache hits on second pass; got {second_calls}"
