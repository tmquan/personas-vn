"""Three-tier Vietnamese → English translator for ontology nodes.

Tier 1 — :func:`packages.ontology.glossary.build_glossary`. Curated,
deterministic, fully offline. Covers ~80% of NSO PX-Web vocabulary by
occurrence (provinces, regions, common variable codes, aggregates,
units).

Tier 2 — *optional* `deep-translator` Google backend, used only for
strings the glossary doesn't cover. The translator caches every result
to a JSON file on disk so repeat runs are free (and bit-equal).

Tier 3 — identity fallback. Numeric strings, years, codes and any string
no other tier produces a translation for keep their original Vietnamese
text and are flagged so they can be added to the glossary later.

Determinism caveat
==================
The Tier 2 backend is non-deterministic across runs (Google rephrases
identical input from time to time). The cache pins the *first*
translation we ever saw for each VI string and re-uses it on subsequent
runs, which restores determinism in practice. Delete
``data/ontology/translation_cache.json`` to refresh.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from packages.common.logging import get_logger
from packages.ontology.glossary import _normalise_d, build_glossary

log = get_logger(__name__)

# Strings matching this regex bypass translation entirely and are
# returned identity. Catches years (``2024``, ``Sơ bộ 2024`` AFTER the
# prefix is stripped), pure-numeric codes, percentages, etc.
_IDENTITY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^\s*\d+([.,]\d+)?\s*$"),                # 2024, 12.5
    re.compile(r"^\s*\d{4}([-/]\d{2,4})?\s*$"),          # 2024-25, 2024/25
    re.compile(r"^\s*[A-Z0-9._-]+\s*$"),                 # codes like V07.01
    re.compile(r"^\s*\d+([.,]\d+)?\s*%\s*$"),            # 12%
    re.compile(r"^\s*$"),                                # whitespace
)

# Compositional patterns: a string matching one of these can be split
# into translatable parts. Order matters — we try most-specific first.
# Each pattern has a (regex, formatter) where formatter takes a re.Match
# and a child-translate function.
_PAREN_RE = re.compile(r"^(.+?)\s*\(([^()]+)\)\s*(.*)$")
# Generic comma/slash splits — translated component-wise.
_LIST_SEP_RE = re.compile(r"\s*([,/])\s*")


def _is_identity(text: str) -> bool:
    return any(p.match(text) for p in _IDENTITY_PATTERNS)


def _strip_time_prefix(text: str, glossary: dict[str, str]) -> tuple[str, str | None]:
    """Detach a leading ``Sơ bộ`` / ``Ước tính`` / ``Chính thức`` prefix.

    Returns ``(remainder, prefix_en)``. Used so we translate
    ``Sơ bộ 2024`` as ``"Preliminary 2024"`` rather than missing it.
    """
    for vi_prefix in ("Sơ bộ", "Ước tính", "Chính thức"):
        if text.startswith(vi_prefix + " "):
            en = glossary.get(vi_prefix)
            if en is not None:
                return text[len(vi_prefix):].lstrip(), en
    return text, None


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------
@dataclass
class Translator:
    """Glossary-first VI→EN translator with on-disk cache.

    Parameters
    ----------
    cache_path:
        Where the persistent translation cache lives. The file is
        loaded on construction and re-written on :meth:`save`.
    enable_online:
        If True and `deep-translator` is importable, fall through to
        Google Translate for strings the glossary doesn't cover.
    """

    cache_path: Path
    enable_online: bool = False
    _glossary: dict[str, str] = field(default_factory=dict)
    _cache: dict[str, str] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "glossary_hits": 0,
        "cache_hits":    0,
        "online_calls":  0,
        "identity":      0,
        "misses":        0,
    })

    def __post_init__(self) -> None:
        self._glossary = build_glossary()
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
                log.info("loaded translation cache: %d entries", len(self._cache))
            except json.JSONDecodeError:
                log.warning("translation cache at %s is corrupt; ignoring",
                            self.cache_path)
                self._cache = {}

    # -- public API -----------------------------------------------------
    def translate(self, vi: str) -> str:
        """Translate a single VI string to EN. Pure function modulo the
        cache and the call-count statistics."""
        return self._translate(vi)

    def translate_many(self, texts: list[str]) -> list[str]:
        """Translate a list, in order. Convenience over `translate`."""
        return [self._translate(t) for t in texts]

    def warm_cache_with_unknown(
        self,
        candidates: list[str],
        *,
        max_workers: int = 8,
        max_strings: int | None = None,
    ) -> dict[str, int]:
        """Translate every candidate not already covered by glossary/cache.

        Implementation note: we use a small ``ThreadPoolExecutor``
        instead of ``translate_batch``. The deep-translator batch path
        on the Google backend is unreliable in practice — it sometimes
        hangs indefinitely on certain inputs. Per-call requests with a
        modest concurrency (8 by default) get us most of the throughput
        without the hang risk, and let us emit progress every N
        completions.

        Cache is persisted to disk every 100 successful translations
        so a Ctrl-C at any point preserves progress.

        Returns a dict of stat counts.
        """
        if not self.enable_online:
            return {"skipped_offline": len(candidates)}

        # Filter to strings the glossary / cache / identity tier can't
        # already serve. Deduplicate while preserving first-seen order.
        seen: set[str] = set()
        unknown: list[str] = []
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            normalised = unicodedata.normalize("NFC", c).strip()
            if (
                _is_identity(normalised)
                or normalised in self._cache
                or normalised in self._glossary
                or _normalise_d(normalised) in self._glossary
                or self._try_compositional(normalised) is not None
            ):
                continue
            unknown.append(normalised)

        if not unknown:
            log.info("warm_cache: no unknown strings to translate")
            return {"online_calls": 0}

        if max_strings is not None:
            unknown = unknown[:max_strings]

        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            from deep_translator import GoogleTranslator
        except ImportError:
            log.warning("deep-translator not installed; can't warm cache")
            return {"skipped_no_dep": len(unknown)}

        log.info(
            "warm_cache: translating %d uncovered strings via Google "
            "(workers=%d) — progress logged every 50 completions",
            len(unknown), max_workers,
        )

        def _one(vi: str) -> tuple[str, str | None]:
            try:
                return vi, GoogleTranslator(source="vi", target="en").translate(vi)
            except Exception as exc:
                log.debug("translate %r failed: %s", vi[:50], exc)
                return vi, None

        successful = 0
        completed = 0
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_one, s) for s in unknown]
            for fut in as_completed(futures):
                vi, en = fut.result()
                completed += 1
                if en and en.strip():
                    self._cache[vi] = en
                    successful += 1
                if completed % 50 == 0 or completed == len(unknown):
                    elapsed = time.perf_counter() - start
                    rate = completed / max(elapsed, 0.001)
                    eta = (len(unknown) - completed) / max(rate, 0.001)
                    log.info(
                        "  warm_cache: %d/%d (%.0f%%, %d ok, %.1f/s, ETA %.0fs)",
                        completed, len(unknown),
                        100 * completed / len(unknown),
                        successful, rate, eta,
                    )
                if successful and successful % 100 == 0:
                    self.save()

        elapsed = time.perf_counter() - start
        log.info(
            "warm_cache: done — %d/%d translated in %.1fs (%.1f/s)",
            successful, len(unknown), elapsed, successful / max(elapsed, 0.001),
        )
        return {
            "online_calls": successful,
            "n_unknown":    len(unknown),
            "elapsed_s":    round(elapsed, 1),
        }

    def save(self) -> None:
        """Persist the cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True)
        )
        log.info("wrote translation cache: %d entries → %s",
                 len(self._cache), self.cache_path)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # -- internals ------------------------------------------------------
    def _translate(self, vi_raw: str) -> str:
        if vi_raw is None:
            return ""

        vi = unicodedata.normalize("NFC", vi_raw).strip()
        if not vi:
            return ""

        if _is_identity(vi):
            self._stats["identity"] += 1
            return vi

        # Detach an optional "Preliminary "/"Estimated "/"Official "
        # prefix so e.g. ``Sơ bộ 2024`` translates as
        # ``"Preliminary 2024"`` rather than as a single uncovered key.
        remainder, prefix_en = _strip_time_prefix(vi, self._glossary)
        if prefix_en is not None and _is_identity(remainder):
            self._stats["identity"] += 1
            return f"{prefix_en} {remainder}"

        # Tier 1 — glossary
        en = self._glossary.get(vi) or self._glossary.get(_normalise_d(vi))
        if en is not None:
            self._stats["glossary_hits"] += 1
            return en

        # Tier 1b — compositional rules. Try to recurse into a string of
        # the form "Head (Tail)" or "A, B" or "A / B" by translating each
        # part independently. This catches frequent NSO patterns like
        # "Bia các loại (Lít)" → "Beer (all types) (Litres)".
        composed = self._try_compositional(vi)
        if composed is not None:
            self._stats["glossary_hits"] += 1
            return composed

        # Tier 2 — cache (which itself stores online-translation results)
        if vi in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[vi]

        # Tier 2b — optional online backend
        if self.enable_online:
            online_en = _online_translate(vi)
            if online_en is not None and online_en.strip():
                self._stats["online_calls"] += 1
                self._cache[vi] = online_en
                return online_en

        # Tier 3 — identity fallback (with miss accounting)
        self._stats["misses"] += 1
        return vi

    def _try_compositional(self, vi: str) -> str | None:
        """Try to translate a compound phrase by recursing into parts.

        Returns the assembled English phrase if **every** atomic part was
        glossary-translatable, otherwise ``None`` (so the caller can
        try the next tier). We deliberately avoid identity fallback at
        this level — a compositional translation only counts if every
        sub-part was a real hit.
        """
        # "X (Tail)" — translate X and Tail separately, recompose.
        m = _PAREN_RE.match(vi)
        if m and not m.group(3):           # only matches if "(Tail)" is the suffix
            head, tail = m.group(1).strip(), m.group(2).strip()
            head_en = self._lookup_strict(head)
            tail_en = self._lookup_strict(tail)
            if head_en is not None and tail_en is not None:
                return f"{head_en} ({tail_en})"

        # "A, B, C" or "A / B / C" — translate each part.
        if _LIST_SEP_RE.search(vi):
            # Split on the first separator type encountered.
            sep_match = _LIST_SEP_RE.search(vi)
            sep = sep_match.group(1)
            parts = [p.strip() for p in vi.split(sep)]
            if len(parts) >= 2:
                pieces: list[str] = []
                ok = True
                for p in parts:
                    en = self._lookup_strict(p)
                    if en is None:
                        ok = False
                        break
                    pieces.append(en)
                if ok:
                    return f"{sep} ".join(pieces)

        return None

    def _lookup_strict(self, vi: str) -> str | None:
        """Glossary-only (no compositional, no online, no identity)
        lookup. Used inside the compositional path."""
        if not vi:
            return ""
        if _is_identity(vi):
            return vi
        return self._glossary.get(vi) or self._glossary.get(_normalise_d(vi))


# ---------------------------------------------------------------------------
# Optional Google Translate backend (deep-translator)
# ---------------------------------------------------------------------------
_ONLINE_AVAILABLE: bool | None = None


def _online_translate(vi: str) -> str | None:
    """Translate a single VI string via Google Translate (deep-translator).

    Returns ``None`` if the dependency is missing or the call fails. We
    catch broadly because every failure mode here (network, rate-limit,
    API change) should fall back to the identity tier rather than abort
    the build.
    """
    global _ONLINE_AVAILABLE
    if _ONLINE_AVAILABLE is False:
        return None
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        if _ONLINE_AVAILABLE is None:
            log.warning(
                "deep-translator not installed — online tier disabled. "
                "Install with `pip install deep-translator` for full coverage."
            )
        _ONLINE_AVAILABLE = False
        return None

    _ONLINE_AVAILABLE = True
    try:
        return GoogleTranslator(source="vi", target="en").translate(vi)
    except Exception as exc:
        log.warning("online translation of %r failed: %s — falling back",
                    vi[:60], exc)
        return None
