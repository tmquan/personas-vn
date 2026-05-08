"""Tests for the pluggable embedding backends.

We don't hit the NIM endpoint here — that requires a live API key and
network. We exercise:

* The auto-dispatch logic (model name → backend class).
* The NIM backend's missing-key error path.
* The local sentence-transformers backend's basic encode round-trip
  (only when ``sentence-transformers`` is installed).

The actual NIM API smoke test is covered by the live
``personas-vn embed-personas`` command in the README — that exercises
the full `make_embedding_backend` → ``encode`` path end-to-end.
"""

from __future__ import annotations

import importlib.util

import pytest

from packages.personas.embed.backends import (
    LOCAL_DEFAULT_MODEL,
    NIM_EMBEDDING_MODELS,
    NimEmbeddingsBackend,
    NimEmbeddingsConfig,
    SentenceTransformersBackend,
    make_embedding_backend,
)


def test_auto_dispatch_local_for_sentence_transformers(monkeypatch):
    # We can't actually instantiate SentenceTransformersBackend without the
    # ST model on disk, but we can verify the *class* dispatch happens.
    import packages.personas.embed.backends as eb

    captured = {}

    class _FakeST:
        def __init__(self, model: str, *, device: str = "cpu") -> None:
            captured["model"] = model
            captured["device"] = device
            self.name = "sentence-transformers"
            self.model = model

    monkeypatch.setattr(eb, "SentenceTransformersBackend", _FakeST)
    backend = make_embedding_backend(model=LOCAL_DEFAULT_MODEL)
    assert isinstance(backend, _FakeST)
    assert captured["model"] == LOCAL_DEFAULT_MODEL


def test_auto_dispatch_nim_for_nvidia_prefix(monkeypatch):
    import packages.personas.embed.backends as eb

    class _FakeNim:
        def __init__(self, model: str, config=None) -> None:
            self.model = model
            self.cfg = config

    monkeypatch.setattr(eb, "NimEmbeddingsBackend", _FakeNim)
    for model in (
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "nvidia/llama-nemotron-embed-1b-v2",
        "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    ):
        backend = make_embedding_backend(model=model)
        assert isinstance(backend, _FakeNim), model
        assert backend.model == model


def test_explicit_backend_pins_choice(monkeypatch):
    import packages.personas.embed.backends as eb

    class _FakeNim:
        def __init__(self, model: str, config=None) -> None:
            self.model = model

    monkeypatch.setattr(eb, "NimEmbeddingsBackend", _FakeNim)
    # Even with a non-nvidia model id, backend="nim" forces the NIM client.
    backend = make_embedding_backend(model="my-custom-endpoint-model", backend="nim")
    assert isinstance(backend, _FakeNim)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        make_embedding_backend(model="x", backend="quantum-cpu")


def test_nim_backend_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("PERSONAS_VN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="missing NIM API key"):
        NimEmbeddingsBackend(
            "nvidia/llama-3.2-nv-embedqa-1b-v2",
            config=NimEmbeddingsConfig(api_key_env="PERSONAS_VN_LLM_API_KEY"),
        )


def test_nim_catalogue_listed():
    """The known-good model list shouldn't shrink unintentionally."""
    expected = {
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "nvidia/llama-nemotron-embed-1b-v2",
        "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    }
    assert expected.issubset(set(NIM_EMBEDDING_MODELS))


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed",
)
def test_local_backend_encode_roundtrip():
    """Encode 3 short bilingual strings; check shape + normalisation."""
    backend = SentenceTransformersBackend(LOCAL_DEFAULT_MODEL, device="cpu")
    try:
        v = backend.encode(
            ["Việt Nam thống kê", "Vietnam statistics", "thống kê quốc gia"],
            batch_size=3, normalize=True,
        )
    finally:
        backend.close()
    assert v.shape == (3, backend.dim)
    # Normalised vectors have unit norm.
    import numpy as np

    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
