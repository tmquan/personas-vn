"""Vietnam province ↔ macro-region map + per-province population weights.

The NSO PX-Web table V02.01 publishes population at the **63-province**
admin-1 level *and* at the **6-macro-region** level, but **without** an
explicit child→parent mapping. The macro-region partition is, however,
fixed by Vietnamese administrative law (most recently the 2008 Hanoi
expansion) and remains stable.

This module hard-codes that partition and exposes a single function,
:func:`load_province_table`, that loads the latest year of V02.01 and
attaches the canonical region for each province along with a normalised
population share usable as a sampling weight.

Why hard-code the mapping rather than discover it? Because V02.01 lists
both provinces *and* macro-regions in the same ``Địa phương`` column;
NSO doesn't distinguish them via a separate level column. Coding the
partition here means the mapping is auditable in one place, doesn't
need a fragile string heuristic, and remains correct even if a future
NSO crawl reorders the rows.

Provincial English names follow Wikipedia's
`Provinces of Vietnam <https://en.wikipedia.org/wiki/Provinces_of_Vietnam>`_
canonical romanisation (which itself follows the General Statistics
Office's English yearbooks).
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import pandas as pd

from packages.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constant: 63 provinces grouped by macro-region.
# Order within each region: alphabetical by Vietnamese name.
# Source: Vietnam General Statistics Office classification, 2008 reform
# (the same partition used by NSO's PX-Web yearbooks).
# ---------------------------------------------------------------------------
# fmt: off
PROVINCE_TO_REGION_VI: dict[str, str] = OrderedDict([
    # ─── 1. Đồng bằng sông Hồng (Red River Delta) — 11 provinces/cities
    ("Bắc Ninh",       "Đồng bằng sông Hồng"),
    ("Hà Nam",         "Đồng bằng sông Hồng"),
    ("Hà Nội",         "Đồng bằng sông Hồng"),
    ("Hải Dương",      "Đồng bằng sông Hồng"),
    ("Hải Phòng",      "Đồng bằng sông Hồng"),
    ("Hưng Yên",       "Đồng bằng sông Hồng"),
    ("Nam Định",       "Đồng bằng sông Hồng"),
    ("Ninh Bình",      "Đồng bằng sông Hồng"),
    ("Quảng Ninh",     "Đồng bằng sông Hồng"),
    ("Thái Bình",      "Đồng bằng sông Hồng"),
    ("Vĩnh Phúc",      "Đồng bằng sông Hồng"),
    # ─── 2. Trung du và miền núi phía Bắc (Northern Midlands & Mountains) — 14
    ("Bắc Giang",      "Trung du và miền núi phía Bắc"),
    ("Bắc Kạn",        "Trung du và miền núi phía Bắc"),
    ("Cao Bằng",       "Trung du và miền núi phía Bắc"),
    ("Điện Biên",      "Trung du và miền núi phía Bắc"),
    ("Hà Giang",       "Trung du và miền núi phía Bắc"),
    ("Hoà Bình",       "Trung du và miền núi phía Bắc"),
    ("Lai Châu",       "Trung du và miền núi phía Bắc"),
    ("Lạng Sơn",       "Trung du và miền núi phía Bắc"),
    ("Lào Cai",        "Trung du và miền núi phía Bắc"),
    ("Phú Thọ",        "Trung du và miền núi phía Bắc"),
    ("Sơn La",         "Trung du và miền núi phía Bắc"),
    ("Thái Nguyên",    "Trung du và miền núi phía Bắc"),
    ("Tuyên Quang",    "Trung du và miền núi phía Bắc"),
    ("Yên Bái",        "Trung du và miền núi phía Bắc"),
    # ─── 3. Bắc Trung Bộ và Duyên hải miền Trung (North-Central & Central Coast) — 14
    ("Bình Định",      "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Bình Thuận",     "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Đà Nẵng",        "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Hà Tĩnh",        "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Khánh Hoà",      "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Nghệ An",        "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Ninh Thuận",     "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Phú Yên",        "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Quảng Bình",     "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Quảng Nam",      "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Quảng Ngãi",     "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Quảng Trị",      "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Thanh Hoá",      "Bắc Trung Bộ và Duyên hải miền Trung"),
    ("Thừa Thiên Huế", "Bắc Trung Bộ và Duyên hải miền Trung"),
    # ─── 4. Tây Nguyên (Central Highlands) — 5
    ("Đắk Lắk",        "Tây Nguyên"),
    ("Đắk Nông",       "Tây Nguyên"),
    ("Gia Lai",        "Tây Nguyên"),
    ("Kon Tum",        "Tây Nguyên"),
    ("Lâm Đồng",       "Tây Nguyên"),
    # ─── 5. Đông Nam Bộ (South East) — 6
    ("Bà Rịa - Vũng Tàu", "Đông Nam Bộ"),
    ("Bình Dương",     "Đông Nam Bộ"),
    ("Bình Phước",     "Đông Nam Bộ"),
    ("Đồng Nai",       "Đông Nam Bộ"),
    ("Tây Ninh",       "Đông Nam Bộ"),
    ("TP.Hồ Chí Minh", "Đông Nam Bộ"),
    # ─── 6. Đồng bằng sông Cửu Long (Mekong River Delta) — 13
    ("An Giang",       "Đồng bằng sông Cửu Long"),
    ("Bạc Liêu",       "Đồng bằng sông Cửu Long"),
    ("Bến Tre",        "Đồng bằng sông Cửu Long"),
    ("Cà Mau",         "Đồng bằng sông Cửu Long"),
    ("Cần Thơ",        "Đồng bằng sông Cửu Long"),
    ("Đồng Tháp",      "Đồng bằng sông Cửu Long"),
    ("Hậu Giang",      "Đồng bằng sông Cửu Long"),
    ("Kiên Giang",     "Đồng bằng sông Cửu Long"),
    ("Long An",        "Đồng bằng sông Cửu Long"),
    ("Sóc Trăng",      "Đồng bằng sông Cửu Long"),
    ("Tiền Giang",     "Đồng bằng sông Cửu Long"),
    ("Trà Vinh",       "Đồng bằng sông Cửu Long"),
    ("Vĩnh Long",      "Đồng bằng sông Cửu Long"),
])
# fmt: on
assert len(PROVINCE_TO_REGION_VI) == 63, "Vietnam has 63 admin-1 units"


# ---------------------------------------------------------------------------
# Vietnamese → English province display labels (NSO / Wikipedia romanisation).
# fmt: off
PROVINCE_VI_TO_EN: dict[str, str] = {
    "An Giang":          "An Giang",
    "Bà Rịa - Vũng Tàu": "Ba Ria - Vung Tau",
    "Bạc Liêu":          "Bac Lieu",
    "Bắc Giang":         "Bac Giang",
    "Bắc Kạn":           "Bac Kan",
    "Bắc Ninh":          "Bac Ninh",
    "Bến Tre":           "Ben Tre",
    "Bình Định":         "Binh Dinh",
    "Bình Dương":        "Binh Duong",
    "Bình Phước":        "Binh Phuoc",
    "Bình Thuận":        "Binh Thuan",
    "Cà Mau":            "Ca Mau",
    "Cao Bằng":          "Cao Bang",
    "Cần Thơ":           "Can Tho",
    "Điện Biên":         "Dien Bien",
    "Đà Nẵng":           "Da Nang",
    "Đắk Lắk":           "Dak Lak",
    "Đắk Nông":          "Dak Nong",
    "Đồng Nai":          "Dong Nai",
    "Đồng Tháp":         "Dong Thap",
    "Gia Lai":           "Gia Lai",
    "Hà Giang":          "Ha Giang",
    "Hà Nam":            "Ha Nam",
    "Hà Nội":            "Hanoi",
    "Hà Tĩnh":           "Ha Tinh",
    "Hải Dương":         "Hai Duong",
    "Hải Phòng":         "Hai Phong",
    "Hậu Giang":         "Hau Giang",
    "Hoà Bình":          "Hoa Binh",
    "Hưng Yên":          "Hung Yen",
    "Khánh Hoà":         "Khanh Hoa",
    "Kiên Giang":        "Kien Giang",
    "Kon Tum":           "Kon Tum",
    "Lai Châu":          "Lai Chau",
    "Lâm Đồng":          "Lam Dong",
    "Lạng Sơn":          "Lang Son",
    "Lào Cai":           "Lao Cai",
    "Long An":           "Long An",
    "Nam Định":          "Nam Dinh",
    "Nghệ An":           "Nghe An",
    "Ninh Bình":         "Ninh Binh",
    "Ninh Thuận":        "Ninh Thuan",
    "Phú Thọ":           "Phu Tho",
    "Phú Yên":           "Phu Yen",
    "Quảng Bình":        "Quang Binh",
    "Quảng Nam":         "Quang Nam",
    "Quảng Ngãi":        "Quang Ngai",
    "Quảng Ninh":        "Quang Ninh",
    "Quảng Trị":         "Quang Tri",
    "Sóc Trăng":         "Soc Trang",
    "Sơn La":            "Son La",
    "Tây Ninh":          "Tay Ninh",
    "Thái Bình":         "Thai Binh",
    "Thái Nguyên":       "Thai Nguyen",
    "Thanh Hoá":         "Thanh Hoa",
    "Thừa Thiên Huế":    "Thua Thien Hue",
    "Tiền Giang":        "Tien Giang",
    "TP.Hồ Chí Minh":    "Ho Chi Minh City",
    "Trà Vinh":          "Tra Vinh",
    "Tuyên Quang":       "Tuyen Quang",
    "Vĩnh Long":         "Vinh Long",
    "Vĩnh Phúc":         "Vinh Phuc",
    "Yên Bái":           "Yen Bai",
}
# fmt: on
assert set(PROVINCE_VI_TO_EN) == set(PROVINCE_TO_REGION_VI), \
    "Province en/vi maps must agree on the 63-unit set"


# ---------------------------------------------------------------------------
# V02.01 loader → per-province population weight, region-conditional.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_province_table(pxweb_root: str | None = None) -> pd.DataFrame:
    """Return one row per province with its macro-region and population.

    Columns:

    ============  ====================================================
    province      Vietnamese label (matches V02.01 ``Địa phương``)
    province_en   English label (Wikipedia / NSO yearbook romanisation)
    region        Vietnamese macro-region label
    population    Latest-year population in **thousands of persons**
                  (V02.01 ``Dân số trung bình (Nghìn người)``)
    weight        ``population / sum(population_within_region)`` — i.e.
                  a within-region categorical distribution that sums to
                  1.0 per region group, ready for ``np.random.choice``.
    ============  ====================================================
    """
    root = Path(pxweb_root) if pxweb_root else Path("data/nso-gov-vn/raw/pxweb/vi")
    path = root / "Dan-so-va-lao-dong__V02.01.px.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"V02.01 parquet not found at {path}. Run `personas-vn curate "
            "--only download` first."
        )
    df = pd.read_parquet(path)

    pop = df[df["Chỉ tiêu"].str.contains("Dân số", na=False)].copy()
    latest = pop["Năm"].astype(str).max()
    pop = pop[pop["Năm"].astype(str) == latest].copy()
    pop = pop[pop["Địa phương"].isin(PROVINCE_TO_REGION_VI)].copy()
    pop = pop.rename(columns={"Địa phương": "province", "value": "population"})

    if pop["province"].nunique() != 63:
        log.warning(
            "V02.01 latest year (%s) only covers %d/63 provinces; "
            "missing rows will use a within-region uniform fallback.",
            latest, pop["province"].nunique(),
        )

    out = (
        pd.DataFrame({"province": list(PROVINCE_TO_REGION_VI.keys())})
        .merge(pop[["province", "population"]], on="province", how="left")
    )
    out["region"] = out["province"].map(PROVINCE_TO_REGION_VI)
    out["province_en"] = out["province"].map(PROVINCE_VI_TO_EN)
    out["population"] = out["population"].fillna(out["population"].mean() or 1000.0)

    region_total = out.groupby("region")["population"].transform("sum")
    out["weight"] = out["population"] / region_total
    return out[["province", "province_en", "region", "population", "weight"]].copy()
