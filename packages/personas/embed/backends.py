"""Pluggable text-embedding backends.

Two backends ship out of the box:

* :class:`SentenceTransformersBackend` — runs a multilingual SBERT model
  locally (default: ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``).
  Free, offline, fast on CPU; 384-dim vectors.

* :class:`NimEmbeddingsBackend` — talks to NVIDIA's hosted embeddings
  endpoint at ``https://integrate.api.nvidia.com/v1/embeddings``. Each
  request is OpenAI-compatible JSON. Verified working with:

      nvidia/llama-3.2-nv-embedqa-1b-v2          (2048-dim, EmbedQA)
      nvidia/llama-3.2-nemoretriever-300m-embed-v1
      nvidia/llama-nemotron-embed-1b-v2          (2048-dim, latest Nemotron)

  These models all accept ``input_type`` of ``"passage"`` (for documents
  to be searched) or ``"query"`` (for the search query). Some have small
  per-request batch limits (~50), so the backend chunks transparently.

Auto-dispatch: :func:`make_embedding_backend` picks the right backend
based on the model name unless you explicitly pin ``backend=...``.
Anything starting with ``nvidia/`` (or a configured prefix list) goes to
NIM; everything else is treated as a local sentence-transformers model.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from packages.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Catalogue (purely advisory — used by the CLI / docs)
# ---------------------------------------------------------------------------
NIM_EMBEDDING_MODELS = [
    "nvidia/llama-nemotron-embed-1b-v2",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
    "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
    "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/nv-embedqa-mistral-7b-v2",
    "nvidia/nv-embed-v1",
    "nvidia/embed-qa-4",
]

LOCAL_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NIM_DEFAULT_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
NIM_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class EmbeddingBackend(ABC):
    """Stateful encoder; subclasses cache models / clients in __init__."""

    name: str = "embedding-backend"
    model: str = ""
    dim: int | None = None

    @abstractmethod
    def encode(self, texts: list[str], *, normalize: bool = True,
                input_type: str = "passage", batch_size: int = 64,
                show_progress: bool = False) -> np.ndarray:
        """Return ``(len(texts), dim)`` float32 ndarray."""

    def encode_query(self, text: str, *, normalize: bool = True) -> np.ndarray:
        """Convenience: encode a single query string."""
        v = self.encode([text], normalize=normalize, input_type="query", batch_size=1,
                         show_progress=False)
        return v[0]

    def close(self) -> None:
        """Release any held resources (HF model on GPU/CPU, HTTPX clients)."""


# ---------------------------------------------------------------------------
# Local backend (sentence-transformers)
# ---------------------------------------------------------------------------
class SentenceTransformersBackend(EmbeddingBackend):
    """Local multilingual SBERT — works offline, no API key needed."""

    name = "sentence-transformers"

    def __init__(self, model: str = LOCAL_DEFAULT_MODEL, *, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the local embedding backend; "
                "install with `pip install -e \".[curator]\"`."
            ) from exc
        log.info("loading local embedding model: %s (device=%s)", model, device)
        self.model = model
        self._device = device
        self._st = SentenceTransformer(model, device=device)

    def encode(self, texts: list[str], *, normalize: bool = True,
                input_type: str = "passage", batch_size: int = 64,
                show_progress: bool = False) -> np.ndarray:
        # input_type is informative-only for sentence-transformers; the model
        # is symmetric. We still honour the parameter name for API parity.
        del input_type
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        v = self._st.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)
        if self.dim is None:
            self.dim = v.shape[1]
        return v

    def close(self) -> None:
        self._st = None  # let the GC reclaim ~120MB


# ---------------------------------------------------------------------------
# NVIDIA NIM hosted backend
# ---------------------------------------------------------------------------
@dataclass
class NimEmbeddingsConfig:
    base_url: str = NIM_DEFAULT_BASE_URL
    api_key_env: str = "PERSONAS_VN_LLM_API_KEY"  # set or fallback to NVIDIA_API_KEY
    timeout_s: float = 60.0
    retries: int = 2
    retry_backoff_s: float = 1.5
    request_delay_s: float = 0.0
    truncate: str = "END"  # "NONE" | "START" | "END" — server-side truncation
    encoding_format: str = "float"
    # NIM has soft batch caps; 50 is safe across all current embedding models.
    max_batch_size: int = 50


class NimEmbeddingsBackend(EmbeddingBackend):
    """OpenAI-compatible embeddings against a NIM endpoint.

    Verified live against ``nvidia/llama-3.2-nv-embedqa-1b-v2``,
    ``nvidia/llama-nemotron-embed-1b-v2``, and
    ``nvidia/llama-3.2-nemoretriever-300m-embed-v1`` (all 2048-dim).

    The backend chunks input into ``max_batch_size`` slices, retries 5xx
    + transport errors with exponential backoff, and surfaces a clean
    :class:`RuntimeError` when the API key is missing.
    """

    name = "nim-embeddings"

    def __init__(self, model: str = NIM_DEFAULT_MODEL,
                 config: NimEmbeddingsConfig | None = None) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a core dep
            raise RuntimeError("httpx is required for the NIM backend") from exc
        self.model = model
        self._cfg = config or NimEmbeddingsConfig()
        api_key = os.environ.get(self._cfg.api_key_env) or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"missing NIM API key: set ${self._cfg.api_key_env} (or $NVIDIA_API_KEY) "
                f"to a key valid for {self._cfg.base_url}"
            )
        self._client = httpx.Client(
            base_url=self._cfg.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self._cfg.timeout_s,
        )
        log.info("NIM embeddings backend ready: %s @ %s", model, self._cfg.base_url)

    def encode(self, texts: list[str], *, normalize: bool = True,
                input_type: str = "passage", batch_size: int = 64,
                show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if input_type not in ("passage", "query"):
            raise ValueError(f"input_type must be 'passage' or 'query', got {input_type!r}")
        chunk = min(int(batch_size), int(self._cfg.max_batch_size))
        all_vecs: list[np.ndarray] = []
        n_done = 0
        for batch in _chunks(texts, chunk):
            v = self._post_embeddings(batch, input_type=input_type)
            if normalize:
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                v = v / norms
            all_vecs.append(v)
            n_done += len(batch)
            if show_progress:
                log.info("  NIM encode: %d/%d", n_done, len(texts))
            if self._cfg.request_delay_s:
                time.sleep(self._cfg.request_delay_s)
        out = np.vstack(all_vecs).astype(np.float32) if all_vecs else np.zeros((0, 0), dtype=np.float32)
        if self.dim is None and out.size:
            self.dim = out.shape[1]
        return out

    def close(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._client.close()

    # ----- internals -------------------------------------------------------
    def _post_embeddings(self, batch: list[str], *, input_type: str) -> np.ndarray:
        import httpx

        body = {
            "model": self.model,
            "input": batch,
            "input_type": input_type,
            "encoding_format": self._cfg.encoding_format,
            "truncate": self._cfg.truncate,
        }
        last_exc: Exception | None = None
        for attempt in range(1, self._cfg.retries + 2):
            try:
                r = self._client.post("/embeddings", json=body)
                r.raise_for_status()
                payload = r.json()
                rows = payload.get("data") or []
                if len(rows) != len(batch):
                    raise RuntimeError(
                        f"NIM returned {len(rows)} rows for batch of {len(batch)}"
                    )
                vecs = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
                return vecs
            except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt > self._cfg.retries:
                    break
                wait = self._cfg.retry_backoff_s * attempt
                log.debug("NIM embeddings attempt %d failed (%s); retrying in %.1fs",
                          attempt, exc, wait)
                time.sleep(wait)
        raise RuntimeError(
            f"NIM /embeddings failed after {self._cfg.retries + 1} attempts: {last_exc}"
        )


def _chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_embedding_backend(
    *,
    model: str | None = None,
    backend: str = "auto",
    device: str = "cpu",
    nim_config: NimEmbeddingsConfig | None = None,
) -> EmbeddingBackend:
    """Build an :class:`EmbeddingBackend` for ``model``.

    ``backend="auto"`` (default) inspects the model name: anything
    starting with ``nvidia/`` goes to the NIM backend; anything else is
    a local sentence-transformers model.
    """
    chosen_model = model or LOCAL_DEFAULT_MODEL
    if backend == "auto":
        backend = "nim" if chosen_model.lower().startswith("nvidia/") else "local"
    if backend in ("nim", "nim-embeddings", "nvidia"):
        return NimEmbeddingsBackend(chosen_model, config=nim_config)
    if backend in ("local", "sentence-transformers", "st"):
        return SentenceTransformersBackend(chosen_model, device=device)
    raise ValueError(f"unknown embedding backend: {backend!r}")
