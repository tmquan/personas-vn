"""Runtime registry that ties ontology config to domain / dimension objects.

Loading ``configs/ontology.yaml`` once and wrapping it in this object is much
cheaper than re-parsing the YAML in every consumer. The registry also owns
the messy job of mapping NSO's free-form WordPress category slugs onto our
fixed list of statistical domains.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from packages.common.config import Config, load_config
from packages.common.logging import get_logger
from packages.ontology.persona import PersonaDimension
from packages.ontology.schema import StatisticalDomain

log = get_logger(__name__)


def _normalise(value: str) -> str:
    """Lower-case + strip Vietnamese diacritics + collapse non-alnum."""
    nfkd = unicodedata.normalize("NFKD", value)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Special-case: Vietnamese 'đ' / 'Đ' do NOT decompose under NFKD.
    no_marks = no_marks.replace("đ", "d").replace("Đ", "d")
    return re.sub(r"[^a-z0-9]+", "-", no_marks.lower()).strip("-")


class OntologyRegistry:
    """Registry of domains + persona dimensions, with NSO-category mapping."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._domains: dict[str, StatisticalDomain] = {}
        self._dimensions: dict[str, PersonaDimension] = {}
        self._alias_index: dict[str, str] = {}

        for raw in config.get("domains", []):
            d = raw.to_dict() if isinstance(raw, Config) else dict(raw)
            domain = StatisticalDomain(**d)
            self._domains[domain.id] = domain
            for alias in [domain.id, domain.label_en, domain.label_vi, *domain.aliases]:
                self._alias_index[_normalise(alias)] = domain.id

        for raw in config.get("persona_dimensions", []):
            d = raw.to_dict() if isinstance(raw, Config) else dict(raw)
            dim = PersonaDimension(**d)
            self._dimensions[dim.id] = dim

        # Validate: every dimension's domain must exist.
        for dim in self._dimensions.values():
            if dim.domain not in self._domains:
                raise ValueError(
                    f"Persona dimension '{dim.id}' references unknown domain '{dim.domain}'"
                )
        # Validate: every parent must exist and come earlier in topological
        # order. We build the order on the fly.
        self._topological_order = self._compute_topological_order()

    # ----- domains ----------------------------------------------------------
    def list_domains(self) -> list[StatisticalDomain]:
        return list(self._domains.values())

    def get_domain(self, domain_id: str) -> StatisticalDomain:
        return self._domains[domain_id]

    def map_nso_category(self, label: str | None, slug: str | None = None) -> str:
        """Map an NSO category (label or slug) onto a stable domain id.

        Falls back to ``"other"`` if no alias matches. We try the slug first
        because slugs are already normalised by WordPress.
        """
        for candidate in (slug, label):
            if not candidate:
                continue
            key = _normalise(candidate)
            if key in self._alias_index:
                return self._alias_index[key]
            # Try a substring match against any alias as a last resort.
            for alias_key, domain_id in self._alias_index.items():
                if alias_key and (alias_key in key or key in alias_key):
                    return domain_id
        return "other" if "other" in self._domains else next(iter(self._domains))

    # ----- persona dimensions ----------------------------------------------
    def list_dimensions(self) -> list[PersonaDimension]:
        return [self._dimensions[name] for name in self._topological_order]

    def get_dimension(self, dimension_id: str) -> PersonaDimension:
        return self._dimensions[dimension_id]

    def topological_order(self) -> list[str]:
        return list(self._topological_order)

    # ----- output settings --------------------------------------------------
    @property
    def output(self) -> Config:
        return self._config.get("output", Config())

    @property
    def version(self) -> str:
        # Bump this when persona dimensions / domain ids change in a
        # non-backwards-compatible way; PersonaBatch records it.
        return "1.0.0"

    # ----- internals --------------------------------------------------------
    def _compute_topological_order(self) -> list[str]:
        unresolved = dict(self._dimensions)
        order: list[str] = []
        # Kahn-ish: repeatedly drain dimensions whose parents are all placed.
        while unresolved:
            placed_this_round = [
                dim_id
                for dim_id, dim in unresolved.items()
                if all(parent in order for parent in dim.parents)
            ]
            if not placed_this_round:
                raise ValueError(
                    f"Cycle or missing parent in persona dimensions: {list(unresolved)}"
                )
            for dim_id in placed_this_round:
                order.append(dim_id)
                del unresolved[dim_id]
        return order


@lru_cache(maxsize=4)
def load_registry(path: str | Path = "configs/ontology.yaml") -> OntologyRegistry:
    """Load + memoise an :class:`OntologyRegistry` from a YAML config."""
    cfg = load_config(path)
    return OntologyRegistry(cfg)
