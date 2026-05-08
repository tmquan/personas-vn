"""PGM generator base — re-export of NVIDIA-NeMo/SDG-PGMs.

Importing from this module is equivalent to importing from
``pgms.generators.base.pgm_generator`` and ``pgms.data_ingest.data_banks``;
we expose ``Edge``, ``PGMGenerator``, and ``TarFileMixin`` as a single
import surface so callers (notably
:mod:`packages.personas.pgm.generator`) only depend on one
local namespace.

We also pre-configure pgmpy's noisy ``FutureWarning`` filter at import
time. New code may import directly from :mod:`pgms` if it doesn't
need the warning silencer.
"""

from __future__ import annotations

import warnings

# Configure pgmpy *before* the first import to silence its noisy startup.
warnings.filterwarnings("ignore", category=FutureWarning)

# Re-export the real things.
from pgms.data_ingest.data_banks import TarFileMixin  # noqa: E402
from pgms.generators.base.pgm_generator import Edge, PGMGenerator  # noqa: E402

__all__ = [
    "Edge",
    "PGMGenerator",
    "TarFileMixin",
]
