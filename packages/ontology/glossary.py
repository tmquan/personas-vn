"""Curated Vietnamese → English glossary for NSO PX-Web statistical text.

This is the **first tier** of the translation pipeline used by the
ontology-tree builder (:mod:`packages.ontology.tree`). It covers the
high-frequency vocabulary that NSO publishes in the PX-Web matrices:

* the 12 first-class statistical databases,
* the ~40 most common variable codes (``Năm``, ``Tỉnh, thành phố``,
  ``Phân tổ``, …),
* the 6 macro-regions and 63 provinces,
* sex / urbanicity / age-group / education / occupation labels,
* aggregate phrases (``Tổng số``, ``Cả nước``, ``Sơ bộ <year>``, …),
* common units (``Nghìn người``, ``Triệu đồng``, ``%``, ``Người``, …),
* time-period labels (months, academic years, "preliminary" prefix).

By covering the high-frequency strings in a curated dictionary we
guarantee:

1. **Determinism** — same input always maps to the same output, no
   network or service variability.
2. **Quality** — every translation has been hand-checked against
   NSO's published English glossary (see the
   `English yearbooks <https://www.nso.gov.vn/du-lieu-va-so-lieu-thong-ke/>`_
   that GSO/NSO publishes alongside the Vietnamese versions).
3. **Offline coverage** — the bulk of the corpus by occurrence
   translates correctly with no internet access.

The translator falls through to (optionally) ``deep-translator`` for
the long-tail strings this glossary doesn't cover. Coverage statistics
are reported at the end of ``personas-vn build-ontology``.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_d(text: str) -> str:
    """Map the alternate Vietnamese ``Ð`` (U+00D0 LATIN CAPITAL ETH) to
    the canonical ``Đ`` (U+0110). NSO's PX-Web export occasionally
    uses both — they look identical but compare unequal as strings.
    """
    return text.replace("\u00d0", "\u0110").replace("\u00f0", "\u0111")


# ---------------------------------------------------------------------------
# 1. Top-level statistical databases (12)
# ---------------------------------------------------------------------------
# NSO publishes 12 "first-class" statistical databases at the root of
# pxweb.nso.gov.vn. The Vietnamese names below are the canonical labels
# from the PX-Web nav; the English names mirror NSO's own English
# yearbook nomenclature.

DATABASE_VI_TO_EN: Final[dict[str, str]] = {
    "Đơn vị hành chính, đất đai và khí hậu":
        "Administrative units, land and climate",
    "Dân số và lao động":
        "Population and employment",
    "Tài khoản quốc gia":
        "National accounts and state budget",
    "Đầu tư":
        "Investment",
    "Doanh nghiệp":
        "Enterprises",
    "Nông, lâm nghiệp và thủy sản":
        "Agriculture, forestry and fishery",
    "Công nghiệp":
        "Industry",
    "Thương mại, giá cả":
        "Trade, prices and tourism",
    "Vận tải và bưu điện":
        "Transport, postal services and telecommunications",
    "Giáo dục":
        "Education",
    "Y tế, văn hóa và đời sống":
        "Health, culture, sport and living standards",
    "Thống kê nước ngoài":
        "International statistics",
}


# ---------------------------------------------------------------------------
# 2. Variable codes (the column names PX-Web variables use)
# ---------------------------------------------------------------------------
# These are the strings under metadata.json#variables[].code. Curated to
# cover ~95% of variable occurrences; everything in the 196 unique codes
# observed in the corpus that appears more than 3 times is here.

VARIABLE_VI_TO_EN: Final[dict[str, str]] = {
    # time
    "Năm":               "Year",
    "Tháng":             "Month",
    "Năm học":           "Academic year",
    "Quý":               "Quarter",
    "Kỳ":                "Period",
    # geography
    "Địa phương":         "Locality",
    "Tỉnh, thành phố":    "Province/city",
    "Tỉnh/thành phố":     "Province/city",
    "Tỉnh/Thành phố":     "Province/city",
    "Vùng":               "Region",
    "Khu vực kinh tế":    "Economic zone",
    "Quốc gia, vùng lãnh thổ":           "Country/territory",
    "Quốc tịch":          "Nationality",
    "Thành thị, nông thôn":              "Urban/rural",
    # composite (var1, var2)
    "Địa phương, Năm":                "Locality, Year",
    "Phân tổ, Năm":                   "Disaggregation, Year",
    "Năm, Nhóm hàng":                 "Year, Commodity group",
    "Năm, Tỉnh, thành phố":           "Year, Province/city",
    "Cách tính, Năm":                 "Method of calculation, Year",
    "Tỉnh, thành phố, Năm":           "Province/city, Year",
    "Tỉnh/thành phố, Năm":            "Province/city, Year",
    "Phân theo địa phương, Năm":      "By locality, Year",
    "Năm, Vùng":                      "Year, Region",
    "Năm, Tỉnh/Thành phố":            "Year, Province/city",
    # categorical / disaggregation
    "Phân tổ":            "Disaggregation",
    "Chỉ tiêu":           "Indicator",
    "Cách tính":          "Method of calculation",
    "Loại hình":          "Type",
    "Loại hình doanh nghiệp":         "Type of enterprise",
    "Loại trường":         "Type of school",
    "Loại nhà":            "Type of housing",
    "Loại cơ sở":          "Type of facility",
    "Loại thu":            "Type of revenue",
    "Cấp học":             "Education level",
    "Cấp quản lý":         "Level of management",
    "Trình độ":            "Qualification level",
    "Trình độ chuyên môn": "Professional qualification",
    "Trình độ chuyên môn kỹ thuật":   "Technical and professional qualification",
    "Trình độ công nghệ":             "Technology level",
    "Thành phần kinh tế": "Economic sector",
    "Ngành":              "Industry / sector",
    "Ngành kinh tế":      "Economic activity",
    "Ngành công nghiệp":  "Industry sector",
    "Nghề nghiệp":        "Occupation",
    "Nhóm tuổi":          "Age group",
    "Nhóm hàng":          "Commodity group",
    "Nhóm thu nhập":      "Income group",
    "Quy mô lao động":    "Labour-force size",
    "Quy mô vốn":         "Capital size",
    "Đối tác đầu tư":     "Investment partner",
    "Số dự án và tổng vốn đăng ký":   "Number of projects and total registered capital",
    "Tổng số và chỉ số phát triển":   "Total and growth index",
    "Tổng số và Chỉ số phát triển":   "Total and growth index",
    "chỉ số phát triển, Năm và Khu vực":  "Growth index, Year and Region",
    # value-bearing
    "Giá trị":            "Value",
    "Trị giá":            "Value (monetary)",
    "Sản lượng":          "Output / production",
    "Khối lượng":         "Volume / quantity",
    "Diện tích":          "Area",
    "Tỷ suất":            "Rate",
    "Chênh lệch":         "Difference",
    # specialised
    "Một số cây lâu năm":  "Selected perennial crops",
    "Trạm, Mực nước":      "Hydrological station, Water level",
    "Chỉ tiêu, Loại thu":  "Indicator, Type of revenue",
}

# After diacritic normalisation, ``Ðịa phương`` collapses to
# ``Địa phương``; the registry will look up under the normalised key.


# ---------------------------------------------------------------------------
# 3. Aggregates / sentinel values
# ---------------------------------------------------------------------------
AGGREGATE_VI_TO_EN: Final[dict[str, str]] = {
    "TỔNG SỐ":          "TOTAL",
    "Tổng số":          "Total",
    "tổng số":          "total",
    "CẢ NƯỚC":          "WHOLE COUNTRY",
    "Cả nước":          "Whole country",
    "cả nước":          "whole country",
    "Toàn quốc":        "Nationwide",
    "Trong nước":       "Domestic",
    "Nước ngoài":       "Foreign",
    "Khác":             "Other",
    "Không áp dụng":    "Not applicable",
    "Chung":            "General",
}


# ---------------------------------------------------------------------------
# 4. Common time-period prefixes
# ---------------------------------------------------------------------------
TIME_PREFIX_VI_TO_EN: Final[dict[str, str]] = {
    "Sơ bộ":          "Preliminary",
    "Ước tính":       "Estimated",
    "Chính thức":     "Official",
}


# ---------------------------------------------------------------------------
# 5. Provinces, regions, sex, urbanicity, age groups, education, occupation
# ---------------------------------------------------------------------------
# These mostly already live in `packages.personas.datasets` and `packages.personas.pgm`.
# We re-export them from a single dict here so the translator can run
# without importing the heavy generator modules.

# Imported lazily inside _build_dimension_glossary to avoid a circular
# import (the ontology package is imported by the personagen package).

def _build_dimension_glossary() -> dict[str, str]:
    out: dict[str, str] = {}
    # Provinces (63) + regions (6) + Vietnam itself
    from packages.personas.datasets.provinces import PROVINCE_VI_TO_EN
    from packages.personas.datasets.schema import (
        AREA_VI_TO_EN,
        EDUCATION_VI_TO_EN,
        MARITAL_VI_TO_EN,
        OCCUPATION_VI_TO_EN,
        REGIONS_VI_TO_EN,
        SEX_VI_TO_EN,
    )

    out.update(REGIONS_VI_TO_EN)
    out.update(PROVINCE_VI_TO_EN)
    out.update(AREA_VI_TO_EN)
    out.update(SEX_VI_TO_EN)
    out.update(MARITAL_VI_TO_EN)
    out.update(EDUCATION_VI_TO_EN)
    out.update(OCCUPATION_VI_TO_EN)
    out["Việt Nam"] = "Vietnam"
    out["Việt nam"] = "Vietnam"
    return out


# ---------------------------------------------------------------------------
# 6. Education-system specific labels (V13 — Giáo dục)
# ---------------------------------------------------------------------------
EDUCATION_LABEL_VI_TO_EN: Final[dict[str, str]] = {
    "Mầm non":             "Pre-school",
    "Tiểu học":            "Primary",
    "Trung học cơ sở":     "Lower secondary",
    "Trung học phổ thông": "Upper secondary",
    "Phổ thông":           "General education",
    "Trung cấp chuyên nghiệp":      "Professional intermediate",
    "Cao đẳng":            "Junior college",
    "Đại học":             "University",
    "Đại học, sau đại học":         "University and post-graduate",
    "Sau đại học":         "Post-graduate",
    "Không có trình độ chuyên môn kỹ thuật":      "No technical qualification",
}


# ---------------------------------------------------------------------------
# 7. Common ISCO occupation phrases that appear inside table cells
# (separate from the persona generator's 11-class collapsed taxonomy)
# ---------------------------------------------------------------------------
ISCO_PHRASES_VI_TO_EN: Final[dict[str, str]] = {
    "Nhà chuyên môn bậc cao":     "Professionals (high-level)",
    "Nhà chuyên môn bậc trung":   "Associate professionals",
    "Lực lượng quân đội":         "Armed forces",
    "Lao động giản đơn":          "Elementary occupations",
    "Lực lượng vũ trang":         "Armed forces (security)",
}


# ---------------------------------------------------------------------------
# 8. Common ISIC industry sectors (used in V07 / V02.42 / V05.x)
# ---------------------------------------------------------------------------
ISIC_VI_TO_EN: Final[dict[str, str]] = {
    "Nông, lâm nghiệp và thuỷ sản":          "Agriculture, forestry and fishery",
    "Nông, lâm nghiệp và thủy sản":          "Agriculture, forestry and fishery",
    "Khai khoáng":                           "Mining and quarrying",
    "Khai thác than cứng và than non":       "Mining of hard coal and lignite",
    "Khai thác dầu thô và khí đốt tự nhiên": "Extraction of crude petroleum and natural gas",
    "Khai thác quặng kim loại":              "Mining of metal ores",
    "Khai khoáng khác":                      "Other mining and quarrying",
    "Hoạt động dịch vụ hỗ trợ khai thác mỏ và quặng":
        "Mining support service activities",
    "Công nghiệp chế biến, chế tạo":         "Manufacturing",
    "Sản xuất, chế biến thực phẩm":          "Manufacture of food products",
    "Sản xuất đồ uống":                      "Manufacture of beverages",
    "Sản xuất sản phẩm thuốc lá":            "Manufacture of tobacco products",
    "Dệt":                                   "Manufacture of textiles",
    "Sản xuất trang phục":                   "Manufacture of wearing apparel",
    "Sản xuất da và các sản phẩm có liên quan":
        "Manufacture of leather and related products",
    "Chế biến gỗ và sản xuất sản phẩm từ gỗ, tre, nứa":
        "Manufacture of wood and wood products",
    "Sản xuất giấy và sản phẩm từ giấy":     "Manufacture of paper and paper products",
    "In, sao chép bản ghi các loại":         "Printing and reproduction",
    "Sản xuất than cốc, sản phẩm dầu mỏ tinh chế":
        "Manufacture of coke and refined petroleum products",
    "Sản xuất hoá chất và sản phẩm hoá chất": "Manufacture of chemicals",
    "Sản xuất thuốc, hoá dược và dược liệu": "Manufacture of pharmaceuticals",
    "Sản xuất sản phẩm từ cao su và plastic": "Manufacture of rubber and plastic products",
    "Sản xuất sản phẩm từ khoáng phi kim loại khác":
        "Manufacture of other non-metallic mineral products",
    "Sản xuất kim loại":                     "Manufacture of basic metals",
    "Sản xuất sản phẩm từ kim loại đúc sẵn (trừ máy móc, thiết bị)":
        "Manufacture of fabricated metal products",
    "Sản xuất sản phẩm điện tử, máy vi tính và sản phẩm quang học":
        "Manufacture of computer, electronic and optical products",
    "Sản xuất thiết bị điện":                "Manufacture of electrical equipment",
    "Sản xuất máy móc, thiết bị chưa được phân vào đâu":
        "Manufacture of machinery and equipment n.e.c.",
    "Sản xuất xe có động cơ":                "Manufacture of motor vehicles",
    "Sản xuất phương tiện vận tải khác":     "Manufacture of other transport equipment",
    "Sản xuất giường, tủ, bàn, ghế":          "Manufacture of furniture",
    "Công nghiệp chế biến, chế tạo khác":    "Other manufacturing",
    "Sửa chữa, bảo dưỡng và lắp đặt máy móc và thiết bị":
        "Repair and installation of machinery",
    "Sản xuất và phân phối điện, khí đốt, nước nóng, hơi nước và điều hoà không khí":
        "Electricity, gas, steam and air conditioning supply",
    "Cung cấp nước; hoạt động quản lý và xử lý rác thải, nước thải":
        "Water supply; sewerage, waste management",
    "Xây dựng":                              "Construction",
    "Bán buôn và bán lẻ":                    "Wholesale and retail trade",
    "Vận tải, kho bãi":                      "Transportation and storage",
    "Dịch vụ lưu trú và ăn uống":            "Accommodation and food service",
    "Thông tin và truyền thông":             "Information and communication",
    "Hoạt động tài chính, ngân hàng và bảo hiểm":
        "Financial and insurance activities",
    "Hoạt động kinh doanh bất động sản":     "Real estate activities",
    "Hoạt động chuyên môn, khoa học và công nghệ":
        "Professional, scientific and technical activities",
    "Hoạt động hành chính và dịch vụ hỗ trợ":
        "Administrative and support service activities",
    "Hoạt động của Đảng Cộng sản, tổ chức chính trị - xã hội; quản lý Nhà nước, an ninh quốc phòng; bảo đảm xã hội bắt buộc":
        "Public administration and defence; compulsory social security",
    "Giáo dục và đào tạo":                   "Education",
    "Y tế và hoạt động trợ giúp xã hội":     "Human health and social work",
    "Nghệ thuật, vui chơi và giải trí":      "Arts, entertainment and recreation",
    "Hoạt động dịch vụ khác":                "Other service activities",
    "Hoạt động làm thuê các công việc trong các hộ gia đình, sản xuất sản phẩm vật chất và dịch vụ tự tiêu dùng của hộ gia đình":
        "Activities of households as employers",
    "Hoạt động của các tổ chức và cơ quan quốc tế":
        "Activities of extraterritorial organisations",
}


# ---------------------------------------------------------------------------
# 9. Units of measure (also recognised inside "(...)" parentheticals)
# ---------------------------------------------------------------------------
UNIT_VI_TO_EN: Final[dict[str, str]] = {
    "Nghìn người":             "Thousand persons",
    "Triệu người":             "Million persons",
    "Người":                   "Persons",
    "Nghìn":                   "Thousand",
    "Triệu":                   "Million",
    "Tỷ":                      "Billion",
    "Tỷ đồng":                 "Billion VND",
    "Triệu đồng":              "Million VND",
    "Nghìn đồng":              "Thousand VND",
    "Đồng":                    "VND",
    "Phần trăm (%)":           "Percent (%)",
    "Tỷ lệ phần trăm":         "Percentage",
    "Người/km2":               "Persons per km²",
    "Người/Km2":               "Persons per km²",
    "Km2":                     "km²",
    "km2":                     "km²",
    "Ha":                      "Ha",
    "Hécta":                   "Hectare",
    "Tấn":                     "Tonne",
    "Nghìn tấn":               "Thousand tonnes",
    "Triệu tấn":               "Million tonnes",
    "Tỷ kWh":                  "Billion kWh",
    "Triệu kWh":               "Million kWh",
    "Nghìn USD":               "Thousand USD",
    "Triệu USD":               "Million USD",
    "Tỷ USD":                  "Billion USD",
    "Số":                      "Number",
    # Short unit symbols inside parentheticals (e.g. "Bia các loại (Lít)").
    "Lít":                     "Litres",
    "Kg":                      "kg",
    "kg":                      "kg",
    "Cái":                     "Pieces",
    "Đôi":                     "Pairs",
    "Bộ":                      "Sets",
    "Chiếc":                   "Pieces",
    "Kwh":                     "kWh",
    "kWh":                     "kWh",
    "M2":                      "m²",
    "M3":                      "m³",
    "m":                       "m",
    "km":                      "km",
    "Cm":                      "cm",
    "Mm":                      "mm",
}


# ---------------------------------------------------------------------------
# 10. Education vocabulary (V13 + Y-te-van-hoa-va-doi-song slices)
# ---------------------------------------------------------------------------
EDU_VOCAB_VI_TO_EN: Final[dict[str, str]] = {
    "Trường học":          "Schools",
    "Lớp học":             "Classes",
    "Giáo viên":           "Teachers",
    "Học sinh":            "Students",
    "Sinh viên":           "Tertiary students",
    "Học viên":            "Trainees",
    "Trường":              "Schools",
    "Lớp":                 "Classes",
    "Số trường học, lớp học, giáo viên và học sinh":
        "Number of schools, classes, teachers and students",
    "Số học sinh bình quân một lớp học":
        "Average students per class",
    "Số học sinh bình quân một giáo viên":
        "Average students per teacher",
    "Số sinh viên bình quân một lớp học":
        "Average tertiary students per class",
    "Số sinh viên bình quân một giáo viên":
        "Average tertiary students per teacher",
    "Mẫu giáo":            "Kindergarten",
    "Nhà trẻ":             "Nursery",
    "Nhà trẻ, mẫu giáo":   "Nursery and kindergarten",
}


# ---------------------------------------------------------------------------
# 11. Common product / commodity nouns (V07 industry-output tables)
# ---------------------------------------------------------------------------
PRODUCT_VI_TO_EN: Final[dict[str, str]] = {
    "Bia các loại":             "Beer (all types)",
    "Nước khoáng":              "Mineral water",
    "Muối biển":                "Sea salt",
    "Muối":                     "Salt",
    "Thủy sản đóng hộp":        "Canned aquatic products",
    "Thuỷ sản đóng hộp":        "Canned aquatic products",
    "Nước mắm":                 "Fish sauce",
    "Dầu thực vật tinh luyện":  "Refined vegetable oil",
    "Bột ngọt":                 "Monosodium glutamate",
    "Sữa tươi":                 "Fresh milk",
    "Sữa bột":                  "Powdered milk",
    "Đường kính":               "Granulated sugar",
    "Sợi":                      "Yarn",
    "Vải":                      "Fabric",
    "Quần áo mặc thường":       "Casual clothing",
    "Giầy, dép da":             "Leather shoes and sandals",
    "Giày, dép da":             "Leather shoes and sandals",
    "Giày thể thao":            "Sports shoes",
    "Chè chế biến":             "Processed tea",
    "Rượu mạnh và rượu trắng":  "Spirits and white wine",
    "Điện phát ra":             "Electricity produced",
    "Nước máy thương phẩm":     "Commercial tap water",
    "Xi măng":                  "Cement",
    "Thép":                     "Steel",
    "Thép cán":                 "Rolled steel",
    "Phân bón":                 "Fertiliser",
    "Phân hoá học":             "Chemical fertiliser",
    "Phân lân":                 "Phosphate fertiliser",
    "Phân đạm":                 "Nitrogen fertiliser",
    "Than sạch":                "Clean coal",
    "Dầu thô":                  "Crude petroleum",
    "Khí đốt":                  "Natural gas",
    "Xăng dầu":                 "Petroleum",
    "Sản phẩm công nghiệp bình quân đầu người":
        "Industrial output per capita",
    "Một số sản phẩm công nghiệp chủ yếu bình quân đầu người":
        "Selected major industrial products per capita",
}


# ---------------------------------------------------------------------------
# 12. Compositional indicator phrases (very common as table-title infixes)
# ---------------------------------------------------------------------------
INDICATOR_VI_TO_EN: Final[dict[str, str]] = {
    "Chỉ số sản xuất công nghiệp":         "Industrial production index",
    "Chỉ số phát triển":                   "Growth index",
    "Chỉ số phát triển (Năm trước = 100)": "Growth index (previous year = 100)",
    "Chỉ số phát triển (Năm trước = 100) - %":
        "Growth index (previous year = 100) - %",
    "Năm trước = 100":                     "previous year = 100",
    "bình quân đầu người":                 "per capita",
    "bình quân":                           "average",
    "tăng trưởng":                         "growth",
    "tỷ trọng":                            "share",
    "phân theo":                           "by",
    "chia theo":                           "by",
    "theo":                                "by",
    "và":                                  "and",
    "với":                                 "with",
    "của":                                 "of",
    "tại":                                 "at",
    "tại thời điểm":                       "as of",
    "Số":                                  "Number of",
}


# ---------------------------------------------------------------------------
# Public assembled lookup
# ---------------------------------------------------------------------------
def build_glossary() -> dict[str, str]:
    """Return the assembled VI→EN glossary, with diacritic-normalised keys.

    Combines every static dict above. The returned mapping has both the
    original and the diacritic-normalised key forms (so e.g. ``Ðịa phương``
    and ``Địa phương`` both map to ``Locality``).
    """
    glossary: dict[str, str] = {}
    for d in (
        DATABASE_VI_TO_EN,
        VARIABLE_VI_TO_EN,
        AGGREGATE_VI_TO_EN,
        TIME_PREFIX_VI_TO_EN,
        EDUCATION_LABEL_VI_TO_EN,
        ISCO_PHRASES_VI_TO_EN,
        ISIC_VI_TO_EN,
        UNIT_VI_TO_EN,
        EDU_VOCAB_VI_TO_EN,
        PRODUCT_VI_TO_EN,
        INDICATOR_VI_TO_EN,
        _build_dimension_glossary(),
    ):
        for k, v in d.items():
            glossary[k] = v
            glossary[_normalise_d(k)] = v
    return glossary
