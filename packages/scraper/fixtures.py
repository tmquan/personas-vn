"""Bundled offline fixture for the NSO scraper.

Why this exists: the visualizer and persona generator should always have
something to render, even when:

* the dev machine has no network,
* NSO is down,
* TLS verification is misconfigured,
* the rate-limit kicks in.

We keep a small, hand-curated snapshot here. Real production runs will pull
the live site; the fixture exists so demos and tests are deterministic.
"""

from __future__ import annotations

from typing import Any

# A representative slice covering several statistical domains. Field names
# match the WordPress REST shape exactly so the mapper code path is the same
# whether we got data from the network or from here.
_FIXTURE_CATEGORIES: list[dict[str, Any]] = [
    {"id": 14, "name": "Employment", "slug": "employment", "parent": 0, "count": 247},
    {"id": 18, "name": "National Accounts", "slug": "national-accounts", "parent": 0, "count": 197},
    {"id": 24, "name": "Agriculture, Forestry and Fishery", "slug": "agriculture-forestry-and-fishery", "parent": 0, "count": 357},
    {"id": 26, "name": "Enterprises", "slug": "enterprises", "parent": 0, "count": 348},
    {"id": 28, "name": "Industry", "slug": "industry", "parent": 0, "count": 434},
    {"id": 30, "name": "Investment and Construction", "slug": "investment-and-construction", "parent": 0, "count": 354},
    {"id": 38, "name": "Education", "slug": "education", "parent": 0, "count": 148},
    {"id": 70, "name": "Dân số", "slug": "dan-so", "parent": 0, "count": 383},
    {"id": 72, "name": "Lao động", "slug": "lao-dong", "parent": 0, "count": 458},
    {"id": 84, "name": "Công nghiệp", "slug": "cong-nghiep", "parent": 0, "count": 819},
    {"id": 86, "name": "Doanh nghiệp", "slug": "doanh-nghiep", "parent": 0, "count": 803},
    {"id": 88, "name": "Đầu tư và Xây dựng", "slug": "dau-tu-va-xay-dung", "parent": 0, "count": 766},
    {"id": 622, "name": "Health, Culture, Sport, Living standards, Social order, Safety and Environment", "slug": "health-culture-sport-living-standards-social-order-safety-and-environment", "parent": 0, "count": 380},
    {"id": 629, "name": "Giáo dục", "slug": "giao-duc", "parent": 0, "count": 370},
    {"id": 719, "name": "Administrative unit, Land and Climate", "slug": "administrative-unit-land-and-climate", "parent": 0, "count": 33},
    {"id": 1387, "name": "Chủ đề khác", "slug": "chu-de-khac", "parent": 0, "count": 1003},
]

_FIXTURE_TAGS: list[dict[str, Any]] = [
    {"id": 1392, "name": "80 năm ngành Thống kê", "slug": "80-nam-thong-ke"},
    {"id": 2001, "name": "GDP", "slug": "gdp"},
    {"id": 2002, "name": "Census", "slug": "census"},
]

_FIXTURE_POSTS: list[dict[str, Any]] = [
    {
        "id": 60001,
        "date": "2026-04-15T08:00:00",
        "slug": "labour-force-survey-q1-2026",
        "link": "https://www.nso.gov.vn/employment/2026/04/labour-force-survey-q1-2026/",
        "title": {"rendered": "Labour force survey, Q1 2026"},
        "excerpt": {"rendered": "<p>Headline labour-force participation rate stable at 68.7% in Q1 2026; youth unemployment ticked up to 7.4% nationally.</p>"},
        "categories": [14, 72],
        "tags": [],
        "lang": "en",
    },
    {
        "id": 60002,
        "date": "2026-03-30T08:00:00",
        "slug": "population-projections-2026-2050",
        "link": "https://www.nso.gov.vn/dan-so/2026/03/population-projections-2026-2050/",
        "title": {"rendered": "Population projections, 2026–2050"},
        "excerpt": {"rendered": "<p>Vietnam's population is projected to peak around 107M in 2044, with the median age rising from 33 (2026) to 42 (2050).</p>"},
        "categories": [70],
        "tags": [2002],
        "lang": "en",
    },
    {
        "id": 60003,
        "date": "2026-02-14T08:00:00",
        "slug": "gdp-q4-2025-by-region",
        "link": "https://www.nso.gov.vn/national-accounts/2026/02/gdp-q4-2025-by-region/",
        "title": {"rendered": "GDP, Q4 2025, by economic region"},
        "excerpt": {"rendered": "<p>Q4 2025 GDP grew 6.9% YoY; the South-East region led at 7.4%, followed by the Red River Delta at 7.1%.</p>"},
        "categories": [18],
        "tags": [2001],
        "lang": "en",
    },
    {
        "id": 60004,
        "date": "2026-01-22T08:00:00",
        "slug": "industrial-production-index-2025",
        "link": "https://www.nso.gov.vn/cong-nghiep/2026/01/industrial-production-index-2025/",
        "title": {"rendered": "Industrial Production Index, full-year 2025"},
        "excerpt": {"rendered": "<p>IIP rose 8.3% in 2025; processing & manufacturing contributed 6.8 percentage points to the headline.</p>"},
        "categories": [28, 84],
        "tags": [],
        "lang": "en",
    },
    {
        "id": 60005,
        "date": "2025-12-10T08:00:00",
        "slug": "education-statistics-2024-2025",
        "link": "https://www.nso.gov.vn/giao-duc/2025/12/education-statistics-2024-2025/",
        "title": {"rendered": "Education statistics, school year 2024–2025"},
        "excerpt": {"rendered": "<p>Net enrolment at lower-secondary level reached 95.4%; tertiary gross enrolment was 35.7%.</p>"},
        "categories": [38, 629],
        "tags": [],
        "lang": "en",
    },
    {
        "id": 60006,
        "date": "2025-11-05T08:00:00",
        "slug": "enterprise-census-preliminary-2025",
        "link": "https://www.nso.gov.vn/doanh-nghiep/2025/11/enterprise-census-preliminary-2025/",
        "title": {"rendered": "Enterprise census, preliminary results 2025"},
        "excerpt": {"rendered": "<p>Active enterprises totalled 921,000 at end-2025; 97.6% are SMEs; 38% are based in the South-East region.</p>"},
        "categories": [26, 86],
        "tags": [],
        "lang": "en",
    },
    {
        "id": 60007,
        "date": "2025-10-12T08:00:00",
        "slug": "agriculture-yearbook-2024",
        "link": "https://www.nso.gov.vn/agriculture-forestry-and-fishery/2025/10/agriculture-yearbook-2024/",
        "title": {"rendered": "Agriculture, Forestry and Fishery yearbook 2024"},
        "excerpt": {"rendered": "<p>Rice output reached 43.7M tonnes; aquaculture production grew 4.1% YoY; coffee exports hit a record 1.94M tonnes.</p>"},
        "categories": [24],
        "tags": [],
        "lang": "en",
    },
    {
        "id": 60008,
        "date": "2025-09-20T08:00:00",
        "slug": "fdi-investment-construction-h1-2025",
        "link": "https://www.nso.gov.vn/dau-tu-va-xay-dung/2025/09/fdi-investment-construction-h1-2025/",
        "title": {"rendered": "FDI, Investment and Construction, H1 2025"},
        "excerpt": {"rendered": "<p>Realised FDI capital reached US$10.8B in H1 2025, up 8.4% YoY; manufacturing took 76% of new registrations.</p>"},
        "categories": [30, 88],
        "tags": [],
        "lang": "en",
    },
    {
        "id": 60009,
        "date": "2025-08-15T08:00:00",
        "slug": "household-living-standards-2024",
        "link": "https://www.nso.gov.vn/health-culture-sport-living-standards/2025/08/household-living-standards-2024/",
        "title": {"rendered": "Household Living Standards Survey, 2024"},
        "excerpt": {"rendered": "<p>Average monthly per-capita income reached VND 5.4M; the urban-rural ratio narrowed slightly to 1.45.</p>"},
        "categories": [622],
        "tags": [],
        "lang": "en",
    },
    {
        "id": 60010,
        "date": "2025-07-01T08:00:00",
        "slug": "administrative-units-update-2025",
        "link": "https://www.nso.gov.vn/administrative-unit-land-and-climate/2025/07/administrative-units-update-2025/",
        "title": {"rendered": "Administrative units of Vietnam, 2025 update"},
        "excerpt": {"rendered": "<p>As of mid-2025, Vietnam comprised 63 provinces / centrally-controlled cities, 705 districts and 10,599 communes.</p>"},
        "categories": [719],
        "tags": [],
        "lang": "en",
    },
]


def load_fixture_records() -> dict[str, list[dict[str, Any]]]:
    """Return the bundled fixture in the shape produced by ``scrape_nso``."""
    return {
        "categories": list(_FIXTURE_CATEGORIES),
        "tags": list(_FIXTURE_TAGS),
        "posts": list(_FIXTURE_POSTS),
    }
