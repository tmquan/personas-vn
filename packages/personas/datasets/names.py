"""Vectorised, seedable Vietnamese full-name generator.

Wraps the ``vn-fullname-generator`` PyPI package. The upstream library
exposes a single ``generator.generate(gender)`` function that calls
``random.choice`` from Python's global RNG, which means:

* it is *not* seedable per call (you'd have to reseed the global module
  state, which leaks into other libraries),
* it issues three Python-level function calls per name, which is too
  slow for the 3 M-row builds.

Both problems vanish if we read the upstream library's three name pools
(``firstnames``, ``malenames``, ``femalenames``) directly and do the
draws ourselves with a :class:`numpy.random.Generator`. That's exactly
what :func:`generate_names` does — same data, same statistical
distribution, ~100x faster, reproducible from a seed.

Usage::

    from packages.personas.datasets.names import generate_names
    rng = np.random.default_rng(42)                     # any Generator
    names = generate_names(["Nam", "Nữ", "Nam"], rng)
    # → array(['Trần Văn Minh', 'Lê Thị Hoa', 'Phạm Đức Anh'])
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Pool loading. Wrapped in a memoised function so the import cost is paid
# once per process and tests can monkey-patch it.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_names(
    sexes_vi: np.ndarray | list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an ``ndarray[str]`` of full Vietnamese names, one per row.

    Parameters
    ----------
    sexes_vi:
        Iterable of canonical Vietnamese sex labels (``"Nam"`` or
        ``"Nữ"``). Length determines the number of names returned.
    rng:
        A :class:`numpy.random.Generator` (typically spawned from the
        builder's :class:`numpy.random.SeedSequence`). Reusing the same
        ``rng`` across calls advances its state — that's expected.

    Notes
    -----
    Output is a flat ``" "``-joined string ``"<surname> <given>"`` —
    matching what ``vn_fullname_generator.generator.generate`` returns,
    matching the spelling and diacritics of the upstream pools, and
    safe to embed in narrative text.
    """
    sexes = np.asarray(sexes_vi, dtype=object)
    n = sexes.shape[0]
    if n == 0:
        return np.array([], dtype=object)

    firstnames, malenames, femalenames = _name_pools()

    surnames = firstnames[rng.integers(0, len(firstnames), size=n)]
    male_given = malenames[rng.integers(0, len(malenames), size=n)]
    female_given = femalenames[rng.integers(0, len(femalenames), size=n)]

    is_male = sexes == "Nam"
    given = np.where(is_male, male_given, female_given)

    return np.asarray([f"{s} {g}" for s, g in zip(surnames, given)], dtype=object)


def slug_id(idx: int, name: str) -> str:
    """Return a stable ``vn-NNNNNN-<asciified-name>`` slug.

    Used as a *human-readable* identifier alongside the canonical
    UUID4 column. Lifts the existing helper from
    :mod:`packages.personas.pgm.generator` into this package
    so callers don't need to import the full PGM stack.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_marks = no_marks.replace("đ", "d").replace("Đ", "d")
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", no_marks.lower()).strip("-")
    return f"vn-{idx:06d}-{slug}"
