"""NSO scraper.

Three sub-modules:

* :mod:`packages.scraper.nso` — the WordPress REST client + paginator.
* :mod:`packages.scraper.mapper` — converts raw WP records into ontology
  :class:`Dataset` instances.
* :mod:`packages.scraper.fixtures` — bundled offline fallback so the rest
  of the pipeline always has something to chew on.
"""

from packages.scraper.fixtures import load_fixture_records
from packages.scraper.mapper import map_posts_to_datasets
from packages.scraper.nso import NSOClient, ScrapeManifest, scrape_nso

__all__ = [
    "NSOClient",
    "ScrapeManifest",
    "load_fixture_records",
    "map_posts_to_datasets",
    "scrape_nso",
]
