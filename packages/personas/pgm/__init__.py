"""Probabilistic-graphical-model persona generator (SDG-PGMs subclass).

The PGM produces structured Vietnamese personas — region, urbanicity,
age, sex, ethnicity, marital status, education, employment status,
occupation, industry sector, income quintile — sampled via a cascaded
Bayesian network whose CPDs are built from real NSO PX-Web statistics.

Public surface (re-exported here for ergonomics):

* :class:`Edge`, :class:`PGMGenerator`, :class:`TarFileMixin` —
  re-exported from NVIDIA SDG-PGMs (``pip install sdg-pgms``).
* :class:`VNPersonaData`, :class:`VNPersonaGenerator`,
  :func:`generate_personas` — the Vietnamese-NSO subclass.
* :class:`PxWebDistributions` and helpers for reading the PX-Web
  parquet count tables.

This sub-package was previously :mod:`packages.personas.pgm` (pre-unification).
"""

from __future__ import annotations

from packages.personas.pgm.base import Edge, PGMGenerator, TarFileMixin
from packages.personas.pgm.distributions import PxWebDistributions
from packages.personas.pgm.generator import (
    VNPersonaData,
    VNPersonaGenerator,
    generate_personas,
)

__all__ = [
    "Edge",
    "PGMGenerator",
    "TarFileMixin",
    "PxWebDistributions",
    "VNPersonaData",
    "VNPersonaGenerator",
    "generate_personas",
]
