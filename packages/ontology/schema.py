"""Core ontology types for NSO data.

The mental model:

* A :class:`Source` is one publisher (here, always NSO).
* A :class:`StatisticalDomain` is a stable, language-neutral bucket, e.g.
  ``population``, ``employment``, ``industry``. NSO's WordPress categories
  map onto these.
* An :class:`Indicator` is a single measurable concept (``urban_population``,
  ``unemployment_rate_15_24``). Indicators are what personas "consume".
* A :class:`Dataset` is one publication tied to a domain — typically a NSO
  post / press release / table dump. Datasets carry provenance (URL, date)
  and a structured payload.

We use Pydantic 2 for free validation + JSON (de)serialisation. The
``model_config`` settings make the models permissive about extra raw fields
NSO might add later.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class _OntologyBase(BaseModel):
    """Base model with shared config so individual models stay short."""

    model_config = ConfigDict(
        extra="allow",        # NSO occasionally adds new top-level fields
        frozen=False,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Source / domain
# ---------------------------------------------------------------------------
class Source(_OntologyBase):
    """The publisher of a dataset. There's only one in this project, but
    modelling it explicitly keeps us honest if we ever add World Bank, UN, etc.
    """

    id: str = Field(description="Stable identifier, e.g. 'nso'")
    name: str
    url: HttpUrl
    country: str = "VN"
    language: Literal["vi", "en", "mixed"] = "mixed"


class StatisticalDomain(_OntologyBase):
    """One of the high-level statistical buckets used across the system.

    The list is fixed in ``configs/ontology.yaml``; instances of this type
    are created at registry-load time, not by the scraper.
    """

    id: str = Field(description="Stable slug, e.g. 'population'")
    label_en: str
    label_vi: str
    aliases: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Indicator / dataset
# ---------------------------------------------------------------------------
class Indicator(_OntologyBase):
    """A single measurable statistical concept.

    Indicators have *units* and *dimensions* (the axes you can slice them
    along, e.g. ``["region", "year"]``). When the persona generator pulls
    distributions, it picks indicators whose dimensions match the persona
    dimension it's filling.
    """

    id: str
    domain_id: str
    name_en: str
    name_vi: str | None = None
    unit: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    description: str | None = None


class Dataset(_OntologyBase):
    """One published artefact mapped from a NSO record (typically a WP post).

    The ``payload`` is intentionally schema-less — different NSO posts carry
    very different shapes (text article, table, embedded chart). We preserve
    the raw structure here and let downstream layers project it as needed.
    """

    id: str = Field(description="Stable id, typically 'nso:<post_id>'")
    source_id: str = "nso"
    domain_id: str
    title: str
    slug: str | None = None
    url: HttpUrl | None = None
    language: Literal["vi", "en", "mixed"] = "vi"
    published_at: datetime | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    raw_categories: list[int] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
