"""Ontology layer.

This package owns:

* :mod:`packages.ontology.schema` — the Pydantic types (``StatisticalDomain``,
  ``Source``, ``Indicator``, ``Dataset``) used to normalise raw scraper
  output.
* :mod:`packages.ontology.persona` — the persona schema (``Persona``) and
  the ``PersonaDimension`` declarations that drive the cascaded sampler.
* :mod:`packages.ontology.registry` — a small registry that loads
  ``configs/ontology.yaml`` and provides ``map_nso_category()``,
  ``list_domains()``, etc.

Nothing here knows anything about the network. The scraper hands us already-
parsed dictionaries; the persona generator reads dimensions from us. This
keeps the ontology testable in isolation.
"""

from packages.ontology.persona import (
    Persona,
    PersonaBatch,
    PersonaDimension,
)
from packages.ontology.registry import OntologyRegistry, load_registry
from packages.ontology.schema import (
    Dataset,
    Indicator,
    Source,
    StatisticalDomain,
)

__all__ = [
    "Dataset",
    "Indicator",
    "OntologyRegistry",
    "Persona",
    "PersonaBatch",
    "PersonaDimension",
    "Source",
    "StatisticalDomain",
    "load_registry",
]
