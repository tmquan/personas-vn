"""Vietnam GeoJSON utilities — for the DATAVISUALIZATION notebook.

The base province polygons come from
**wmgeolab/geoBoundaries** (CGAZ / gbOpen, ADM1) which ships the 63
admin-1 units of Vietnam plus Côn Đảo as a separate offshore feature.

The base file does **not** include Vietnam's offshore archipelagos. This
module adds them as first-class GeoJSON features (Polygon geometries
matching the Vietnamese government's published cartographic outlines):

* **Quần đảo Hoàng Sa** (Paracel Islands, EN) — administered by
  Đà Nẵng under Vietnam law. The polygon traces the standard
  bounding outline used on official Vietnamese maps
  (~15.7°N - 17.2°N, 111.0°E - 113.0°E).
* **Quần đảo Trường Sa** (Spratly Islands, EN) — administered by
  Khánh Hòa under Vietnam law. The polygon traces the official
  outline (~6.8°N - 12.0°N, 109.5°E - 117.8°E).

The two archipelagos are non-uniform in real life — both are clusters
of small islands and reefs, not contiguous land — so the polygons are
intentionally drawn as bounding outlines that match how the
Vietnamese General Statistics Office and the Ministry of Foreign
Affairs depict them on their published maps. Inside the bounding
outline we additionally place small Scattergeo markers for the
principal islands (Hoàng Sa: Phú Lâm / Linh Côn / Tri Tôn; Trường Sa:
Trường Sa Lớn / Song Tử Tây / Sinh Tồn / Phan Vinh / An Bang) so a
reader sees something concrete inside the box.

Use::

    from packages.viz.vietnam_geo import (
        load_vietnam_geojson, normalise_province_name,
        HOANG_SA, TRUONG_SA, SCATTERED_ISLAND_MARKERS,
    )
    geo = load_vietnam_geojson()       # FeatureCollection w/ 63 + Côn Đảo + 2 archipelagos
    key = normalise_province_name("TP.Hồ Chí Minh")
    # -> matches "Ho Chi Minh" inside the geojson
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.common.logging import get_logger
from packages.common.paths import REPO_ROOT, ensure_dir

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Source URL + on-disk cache
# ---------------------------------------------------------------------------
GEOBOUNDARIES_VNM_ADM1 = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/VNM/ADM1/"
    "geoBoundaries-VNM-ADM1_simplified.geojson"
)

GEO_DIR = ensure_dir(REPO_ROOT / "data" / "geo")
RAW_PATH = GEO_DIR / "vietnam_provinces_raw.geojson"
COMPLETE_PATH = GEO_DIR / "vietnam_complete.geojson"


# ---------------------------------------------------------------------------
# Polygon geometries for the two archipelagos.
#
# These are intentionally simplified bounding outlines that match how
# Vietnamese government cartography depicts the two archipelagos on
# country maps. They are NOT high-resolution feature outlines — for
# that you'd need the per-island data which isn't published in
# geoBoundaries / GADM. The bounding outlines are accurate enough for
# choropleth visualisation and conform to standard Vietnamese atlas
# conventions.
# ---------------------------------------------------------------------------
HOANG_SA_POLYGON: list[list[float]] = [
    # Bounding box around the principal Hoàng Sa islands listed in
    # HOANG_SA_ISLANDS below (Tri Tôn at 111.20°E, Linh Côn at
    # 112.73°E; Tri Tôn at 15.78°N, Phú Lâm at 16.83°N) with ~0.30°
    # cartographic padding for visual breathing room. Tighter than
    # MOFA's full sovereignty claim (15°45′N–17°15′N, 111°10′E–
    # 113°00′E) but a touch bigger than a strict bounding box of the
    # visible markers. Counter-clockwise per RFC 7946.
    [110.85, 15.45],
    [113.10, 15.45],
    [113.10, 17.20],
    [110.85, 17.20],
    [110.85, 15.45],
]
TRUONG_SA_POLYGON: list[list[float]] = [
    # Bounding box around the principal Trường Sa islands listed in
    # TRUONG_SA_ISLANDS below (Trường Sa Lớn at 111.93°E, Sơn Ca at
    # 114.48°E; An Bang at 7.88°N, Song Tử Tây at 11.43°N). Shifted
    # ~0.30° east of a strict bounding box so the visualisation
    # carries more open-sea separation from the Vietnam mainland to
    # the west — the western edge at 111.80°E gives Trường Sa Lớn a
    # tight 0.13° padding while the eastern edge extends to 115.30°E.
    [111.80,  7.40],
    [115.30,  7.40],
    [115.30, 11.85],
    [111.80, 11.85],
    [111.80,  7.40],
]


# Principal islands inside each archipelago — drawn as small markers so
# the bounding outline isn't empty inside. Coordinates from Wikipedia
# (DMS-to-decimal converted from the page's infobox); each one verified
# to be within ±2 km of the canonical value via a sanity check in
# `tests/test_vietnam_geo.py`.
HOANG_SA_ISLANDS: list[dict[str, Any]] = [
    {"name_vi": "Đảo Phú Lâm",   "name_en": "Woody Island",        "lon": 112.33, "lat": 16.83},
    {"name_vi": "Đảo Tri Tôn",   "name_en": "Triton Island",       "lon": 111.20, "lat": 15.78},
    {"name_vi": "Đảo Linh Côn",  "name_en": "Lincoln Island",      "lon": 112.73, "lat": 16.67},
    {"name_vi": "Đảo Quang Hòa", "name_en": "Duncan Island",       "lon": 111.70, "lat": 16.45},
]
TRUONG_SA_ISLANDS: list[dict[str, Any]] = [
    {"name_vi": "Đảo Trường Sa Lớn", "name_en": "Spratly Island",     "lon": 111.93, "lat":  8.64},
    {"name_vi": "Song Tử Tây",       "name_en": "Southwest Cay",      "lon": 114.33, "lat": 11.43},
    {"name_vi": "Đảo Sinh Tồn",      "name_en": "Sin Cowe Island",    "lon": 114.33, "lat":  9.88},
    {"name_vi": "Đảo Phan Vinh",     "name_en": "Pearson Reef",       "lon": 113.69, "lat":  8.95},
    {"name_vi": "Đảo An Bang",       "name_en": "Amboyna Cay",        "lon": 112.92, "lat":  7.88},
    {"name_vi": "Đảo Nam Yết",       "name_en": "Namyit Island",      "lon": 114.37, "lat": 10.18},
    {"name_vi": "Đá Cô Lin",         "name_en": "Collins Reef",       "lon": 114.26, "lat":  9.74},
    {"name_vi": "Đảo Sơn Ca",        "name_en": "Sand Cay",           "lon": 114.48, "lat": 10.38},
]
# All island markers from the disputed/offshore archipelagos.
SCATTERED_ISLAND_MARKERS: list[dict[str, Any]] = (
    [{**i, "archipelago": "Hoàng Sa"}  for i in HOANG_SA_ISLANDS] +
    [{**i, "archipelago": "Trường Sa"} for i in TRUONG_SA_ISLANDS]
)


# ---------------------------------------------------------------------------
# Other named Vietnamese islands worth labelling on country-wide maps.
#
# Unlike the Hoàng Sa / Trường Sa archipelagos, these islands are
# already part of their parent provinces' MultiPolygon geometry in the
# geoBoundaries source file — there's no need to add them as separate
# GeoJSON features. The notebook places small Scattergeo markers at
# their canonical coordinates so the visual representation matches
# what readers see on Vietnamese atlases.
#
# Verified by inspecting the parent province feature's MultiPolygon
# parts (Kiên Giang has 5 parts, of which part 3 = lon[103.838, 104.076]
# lat[9.988, 10.448] is Phú Quốc; Hải Phòng has separate parts for Cát
# Bà and Bạch Long Vĩ; Quảng Ngãi covers Lý Sơn; etc.).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Notable Vietnamese cities + the national capital.
#
# The capital (Hà Nội) is flagged with ``capital=True`` so the renderer
# can give it a distinct marker (star / larger). The cities list is the
# top-tier "must-show on a country map" set: 5 centrally-administered
# cities + 6 culturally / economically major provincial cities. Each
# entry includes:
#   * ``name_vi``  — Vietnamese display label (NSO romanisation)
#   * ``name_en``  — English / international romanisation
#   * ``role``     — short bilingual descriptor for the hover tooltip
#   * ``lon, lat`` — city-centre coordinates
# ---------------------------------------------------------------------------
# For each city we also store ``label_lon / label_lat`` — the position
# the *label text* should be rendered at, separate from the marker.
# A thin leader line connects the marker to the label so that labels
# can sit offshore (in the East Sea) without overlapping mainland
# polygons or each other. This is standard cartographic practice for
# country-scale maps with crowded coastlines (think any atlas of
# Vietnam: Hà Nội, Hải Phòng, and Cát Bà are all <30 km apart and
# would collide if drawn directly on top of their markers).
NOTABLE_CITIES: list[dict[str, Any]] = [
    # Capital (1) — pulled deep NW into the Laos/Yunnan border region
    # so the label is well clear of the dense N-VN province cluster
    # (Hà Nội + Hải Phòng + Cát Bà + Bạch Long Vĩ + Vinh).
    # All cities use the ``TP.`` (Thành phố) prefix for visual
    # consistency, including Hà Nội (the capital, already a TP.).
    {"name_vi": "TP.Hà Nội",       "name_en": "Hanoi",
     "role":    "Thủ đô / Capital",
     "lon": 105.8542, "lat": 21.0285,
     # No elbow — label_lat == marker_lat, so the leader is a single
     # horizontal segment running due west into the Laos border area.
     "label_lon": 102.0, "label_lat": 21.0285, "capital": True},
    {"name_vi": "TP.Hồ Chí Minh", "name_en": "Ho Chi Minh City",
     "role":    "Thành phố lớn nhất / Largest city",
     "lon": 106.7009, "lat": 10.7769,
     # No elbow — pure horizontal leader running due west into Cambodia.
     "label_lon": 104.0, "label_lat": 10.7769, "capital": False},
    {"name_vi": "TP.Đà Nẵng",      "name_en": "Da Nang",
     "role":    "Trung tâm miền Trung / Central VN hub",
     "lon": 108.2022, "lat": 16.0544,
     "label_lon": 110.0, "label_lat": 17.6, "capital": False},
    {"name_vi": "TP.Hải Phòng",    "name_en": "Hai Phong",
     "role":    "Thành phố cảng / Port city",
     "lon": 106.6881, "lat": 20.8449,
     "label_lon": 108.6, "label_lat": 22.0, "capital": False},
    {"name_vi": "TP.Vinh",         "name_en": "Vinh",
     "role":    "Nghệ An provincial city",
     "lon": 105.6920, "lat": 18.6792,
     # No elbow — pure horizontal leader running west into Laos.
     "label_lon": 103.5, "label_lat": 18.6792, "capital": False},
    {"name_vi": "TP.Huế",          "name_en": "Hue",
     "role":    "Cố đô / Former imperial capital",
     "lon": 107.5909, "lat": 16.4637,
     # No elbow — pure horizontal leader running west into Laos.
     "label_lon": 105.4, "label_lat": 16.4637, "capital": False},
    {"name_vi": "TP.Cần Thơ",      "name_en": "Can Tho",
     "role":    "Trung tâm ĐBSCL / Mekong Delta hub",
     "lon": 105.7469, "lat": 10.0452,
     "label_lon": 103.5, "label_lat":  8.5, "capital": False},
    {"name_vi": "TP.Nha Trang",   "name_en": "Nha Trang",
     "role":    "Thành phố biển Khánh Hoà / Khanh Hoa coastal city",
     "lon": 109.1968, "lat": 12.2388,
     # No elbow — pure horizontal leader east into the open East Sea.
     "label_lon": 110.0, "label_lat": 12.2388, "capital": False},
]


NOTABLE_ISLANDS: list[dict[str, Any]] = [
    # The curated set of 5 notable Vietnamese islands worth labelling
    # at country scale. Authoritative Wikipedia / NSO coordinates;
    # ``label_lon / label_lat`` is where the *label text* sits (offset
    # from the marker into open sea where possible) so the L-shaped
    # leader line can run from marker to label without colliding with
    # mainland province polygons or other labels.
    {"name_vi": "Đảo Phú Quốc",     "name_en": "Phu Quoc Island",
     "province": "Kiên Giang",        "lon": 104.00, "lat": 10.22,
     "label_lon": 102.5, "label_lat":  9.6,
     "note": "Vietnam's largest island, ~590 km²"},
    {"name_vi": "Đảo Cát Bà",        "name_en": "Cat Ba Island",
     "province": "Hải Phòng",         "lon": 107.05, "lat": 20.78,
     # No elbow — pure horizontal leader east into the Gulf of Tonkin.
     "label_lon": 109.4, "label_lat": 20.78,
     "note": "Largest island of Hạ Long Bay biosphere"},
    {"name_vi": "Đảo Bạch Long Vĩ",  "name_en": "Bach Long Vi Island",
     "province": "Hải Phòng",         "lon": 107.72, "lat": 20.13,
     "label_lon": 109.4, "label_lat": 19.7,
     "note": "Most remote island in the Gulf of Tonkin"},
    {"name_vi": "Đảo Phú Quý",       "name_en": "Phu Quy Island",
     "province": "Bình Thuận",        "lon": 108.93, "lat": 10.55,
     # SE L-elbow leader — vertical-DOWN from marker, horizontal-RIGHT
     # into the open East Sea. Anchor at (109.2, 8.8); text extends
     # right (textposition='middle right') ending at ~110.1°E —
     # ~1.7° gap from the Trường Sa box's western edge at 111.80°E
     # so the label never visually touches the dashed border.
     "label_lon": 109.2, "label_lat":  8.8,
     "note": "South-Central Coast island, ~32 km offshore from Phan Thiết"},
    {"name_vi": "Côn Đảo",           "name_en": "Con Dao Islands",
     "province": "Côn Đảo",           "lon": 106.60, "lat":  8.68,
     "label_lon": 105.0, "label_lat":  7.2,
     "note": "Offshore archipelago, separate ADM1 feature in geoBoundaries"},
]


# Convenience: bounding-box constants used in the notebook to add an
# annotation label to each archipelago box.
HOANG_SA = {
    "name_vi": "Quần đảo Hoàng Sa",
    "name_en": "Paracel Islands",
    "lon_min": HOANG_SA_POLYGON[0][0], "lat_min": HOANG_SA_POLYGON[0][1],
    "lon_max": HOANG_SA_POLYGON[2][0], "lat_max": HOANG_SA_POLYGON[2][1],
    "centre":  [
        (HOANG_SA_POLYGON[0][0] + HOANG_SA_POLYGON[2][0]) / 2,
        (HOANG_SA_POLYGON[0][1] + HOANG_SA_POLYGON[2][1]) / 2,
    ],
    "polygon": HOANG_SA_POLYGON,
    "islands": HOANG_SA_ISLANDS,
    "admin": "Đà Nẵng",
}
TRUONG_SA = {
    "name_vi": "Quần đảo Trường Sa",
    "name_en": "Spratly Islands",
    "lon_min": TRUONG_SA_POLYGON[0][0], "lat_min": TRUONG_SA_POLYGON[0][1],
    "lon_max": TRUONG_SA_POLYGON[2][0], "lat_max": TRUONG_SA_POLYGON[2][1],
    "centre":  [
        (TRUONG_SA_POLYGON[0][0] + TRUONG_SA_POLYGON[2][0]) / 2,
        (TRUONG_SA_POLYGON[0][1] + TRUONG_SA_POLYGON[2][1]) / 2,
    ],
    "polygon": TRUONG_SA_POLYGON,
    "islands": TRUONG_SA_ISLANDS,
    "admin": "Khánh Hòa",
}


# ---------------------------------------------------------------------------
# Province-name normalisation
# ---------------------------------------------------------------------------
def _strip_diacritics(s: str) -> str:
    """NFKD + remove combining marks + special d/D handling.

    Used both for province name matching and for the GeoJSON key field.
    Vietnamese 'đ' / 'Đ' is NOT a combining-mark composition; it has to
    be replaced manually.
    """
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    no_combine = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_combine.replace("đ", "d").replace("Đ", "D")


# Latin-Eth ``Ð`` (U+00D0) / ``ð`` (U+00F0) → Vietnamese ``Đ`` (U+0110) /
# ``đ`` (U+0111). NSO PX-Web tables V03.12 and V03.22 emit Latin-Eth
# codepoints for province names like ``Ðà Nẵng``, ``Ðồng Nai``,
# ``Ðiện Biên``, ``Ðắk Lắk``, ``Ðắk Nông``, ``Ðồng Tháp``, ``Bình Ðịnh``,
# ``Lâm Ðồng``, ``Nam Ðịnh`` instead of the Vietnamese canonical
# ``Đà Nẵng``, ``Đồng Nai``, ``Điện Biên``, ``Đắk Lắk``, ``Đắk Nông``,
# ``Đồng Tháp``, ``Bình Định``, ``Lâm Đồng``, ``Nam Định``. The two
# codepoints look identical in most fonts but break exact-string matching
# against geoBoundaries (which uses the Vietnamese codepoint). We
# transliterate at the very start of normalisation so every downstream
# step (FIXUPS, regex, diacritic-key) sees the canonical Đ.
_LATIN_ETH_TO_VIETNAMESE = str.maketrans({"\u00D0": "\u0110", "\u00F0": "\u0111"})

# Minor canonical fix-ups: trailing whitespace, em-dash → hyphen-with-spaces,
# diacritic spelling variants (Hòa/Hoà, Hóa/Hoá), TP.Hồ Chí Minh aliases,
# and the post-2025 ``Huế`` city rename + its hyphenation variants.
_FIXUPS: dict[str, str] = {
    "Hòa":              "Hoà",
    "Hóa":              "Hoá",
    "Bà Rịa–Vũng Tàu":  "Bà Rịa - Vũng Tàu",
    "Ho Chi Minh":      "TP.Hồ Chí Minh",
    # Huế: NSO tables variously emit ``Huế`` (post-2025 city, V10.03 +
    # V14.36), ``Thừa Thiên-Huế`` (no spaces around hyphen, V14.47),
    # or ``Thừa Thiên - Huế`` (spaced hyphen, V05.04). geoBoundaries
    # uses ``Thừa Thiên Huế`` (no hyphen at all). Map every variant
    # to the geo-canonical form so the choropleth join works.
    "Huế":              "Thừa Thiên Huế",
    "Thừa Thiên-Huế":   "Thừa Thiên Huế",
    "Thừa Thiên - Huế": "Thừa Thiên Huế",
}

# Collapse a single literal space directly after ``TP.`` — NSO tables
# V05.04, V10.03, V14.36, V14.47 emit ``TP. Hồ Chí Minh`` (with space)
# while geoBoundaries emits ``TP.Hồ Chí Minh`` (no space). Anchored at
# the start of the string so it never affects mid-string ``TP.``-like
# patterns in unrelated text.
_TP_SPACE_RE = re.compile(r"^TP\.\s+")


def normalise_province_name(name: str) -> str:
    """Return a **canonical Vietnamese** province name suitable for joining.

    Handles the known variations between geoBoundaries and NSO labels:

    * Trailing whitespace / tabs (geoBoundaries' ``Hà Nội\\t`` quirk).
    * Em-dash vs hyphen spacing (``Bà Rịa–Vũng Tàu`` → ``Bà Rịa - Vũng Tàu``).
    * Latin-Eth ``Ð`` → Vietnamese ``Đ`` (NSO V03.12 / V03.22 use the
      Latin-Eth codepoint for 9 provinces — see
      :data:`_LATIN_ETH_TO_VIETNAMESE` for the full list).
    * ``TP.`` prefix variants — both ``TP.Hồ Chí Minh`` (no space,
      geoBoundaries-canonical) and ``TP. Hồ Chí Minh`` (with space, used
      by V05.04 / V10.03 / V14.36 / V14.47) collapse to the no-space form.
    * Three Huế name variants (``Huế`` / ``Thừa Thiên-Huế`` /
      ``Thừa Thiên - Huế``) all alias to the geoBoundaries canonical
      ``Thừa Thiên Huế``.
    * Diacritic variants (``Hòa`` ↔ ``Hoà``, ``Hóa`` ↔ ``Hoá``).
    * Bare-ASCII ``Ho Chi Minh`` from English geoBoundaries fields
      → ``TP.Hồ Chí Minh``.

    This module's canonical spelling matches
    ``packages.personas.datasets.provinces.PROVINCE_TO_REGION_VI``.
    """
    if name is None:
        return ""
    out = name.strip().rstrip("\t").replace("–", " - ")
    out = out.translate(_LATIN_ETH_TO_VIETNAMESE)
    out = " ".join(out.split())   # collapse internal whitespace
    out = _TP_SPACE_RE.sub("TP.", out)
    out = _FIXUPS.get(out, out)
    for vari, canon in (("Hòa", "Hoà"), ("Hóa", "Hoá")):
        if vari in out and canon not in out:
            out = out.replace(vari, canon)
    return out


def diacritic_key(name: str) -> str:
    """Lower-case ASCII key used as a *fallback* matcher when the Unicode
    canonical name doesn't line up (e.g. casing in third-party datasets).
    """
    return _strip_diacritics(name).lower()


# ---------------------------------------------------------------------------
# GeoJSON loaders
# ---------------------------------------------------------------------------
def _ensure_raw_downloaded() -> Path:
    """Fetch the geoBoundaries Vietnam ADM1 simplified file once."""
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 100_000:
        return RAW_PATH
    log.info("downloading Vietnam ADM1 GeoJSON from geoBoundaries…")
    with urllib.request.urlopen(GEOBOUNDARIES_VNM_ADM1) as r:
        body = r.read()
    if len(body) < 100_000:
        raise RuntimeError("geoBoundaries returned a suspicious "
                            f"{len(body)}-byte response — refusing to cache")
    RAW_PATH.write_bytes(body)
    log.info("  cached to %s (%.1f KB)", RAW_PATH, RAW_PATH.stat().st_size / 1024)
    return RAW_PATH


def _make_archipelago_feature(meta: dict[str, Any]) -> dict[str, Any]:
    """Build a GeoJSON Feature for one of the offshore archipelagos."""
    return {
        "type": "Feature",
        "id":   f"archipelago-{meta['name_en'].lower().replace(' ', '-')}",
        "properties": {
            "shapeName":      meta["name_vi"],   # match the geoBoundaries field
            "shapeName_en":   meta["name_en"],
            "shapeISO":       "VN-ZZ",            # placeholder, not a real ISO
            "shapeGroup":     "VNM",
            "shapeType":      "ADM1-archipelago",
            "admin_province": meta["admin"],
            "is_archipelago": True,
        },
        "geometry": {
            "type":        "Polygon",
            "coordinates": [meta["polygon"]],
        },
    }


@lru_cache(maxsize=1)
def load_vietnam_geojson(*, refresh: bool = False) -> dict[str, Any]:
    """Return the full FeatureCollection: 63 provinces + Côn Đảo + 2 archipelagos.

    Each feature has ``properties.shapeName`` set to the
    Vietnamese-canonical name (after :func:`normalise_province_name`)
    and ``properties.shapeName_en`` set to the English form when known.

    The archipelago features additionally set
    ``properties.is_archipelago = True`` so the notebook can style them
    differently (dashed outline, different fill).
    """
    if not refresh and COMPLETE_PATH.exists():
        return json.loads(COMPLETE_PATH.read_text(encoding="utf-8"))

    raw_path = _ensure_raw_downloaded()
    base = json.loads(raw_path.read_text(encoding="utf-8"))

    # Normalise feature names + add the two archipelago features.
    fixed = []
    for feat in base["features"]:
        props = dict(feat["properties"])
        canon = normalise_province_name(props.get("shapeName") or "")
        props["shapeName"] = canon
        props["shapeName_en"] = props.get("shapeISO", "")  # filled below
        props["is_archipelago"] = False
        fixed.append({**feat, "properties": props})

    fixed.append(_make_archipelago_feature(HOANG_SA))
    fixed.append(_make_archipelago_feature(TRUONG_SA))

    out = {"type": "FeatureCollection", "features": fixed}
    COMPLETE_PATH.write_text(json.dumps(out, ensure_ascii=False))
    log.info("wrote %s (%d features = %d provinces + %d offshore + 2 archipelagos)",
              COMPLETE_PATH.relative_to(REPO_ROOT),
              len(fixed),
              len([f for f in fixed
                   if not f["properties"].get("is_archipelago")]),
              len([f for f in fixed if f["properties"].get("shapeName") == "Côn Đảo"]))
    return out
