"""Bilingual lookup tables that ground the narrative templates.

The narrative fields in the Nemotron schema (``professional_persona``,
``culinary_persona``, …) are deterministic compositions over the
**structured** persona fields plus these lookup dictionaries. Because
every narrative variant cites a structured fact (occupation, region,
education, …), the resulting strings stay grounded in the underlying
NSO statistics.

Lookup tables are organised by axis:

* ``OCCUPATION_*`` — skills, sample-tools, professional-narrative seeds
  per ISCO-collapsed occupation (NSO V02.43 categories).
* ``REGION_*`` — region-specific cuisines, scenic destinations, cultural
  motifs (Vietnam's six macro-regions per NSO V02.01).
* ``HOBBIES_*`` — age × area baskets of common Vietnamese hobbies.
* ``GOALS_*`` — career-goal seeds per (occupation, education) pair.

All structures are bilingual: ``..._VI`` is the Vietnamese-language
side, ``..._EN`` is the English mirror. They have identical key sets
and identical list lengths so the template renderers can pick the
same variant index in either language and produce parallel output.

This module is **deliberately data-only** (no functions, no I/O) so
tests can import the dictionaries and assert structural invariants
without setting up the rest of the pipeline.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# 1. Per-occupation skill / tool baskets
# ---------------------------------------------------------------------------
# Keyed by the canonical Vietnamese occupation labels NSO publishes in
# PX-Web V02.43 (and aliases that appeared in older years). Each value is
# a list of 4-8 *concrete* skills typical of the occupation. The English
# mirror uses the SAME index → same skill, translated.

OCCUPATION_SKILLS_VI: Final[dict[str, list[str]]] = {
    "Nhà lãnh đạo": [
        "lập chiến lược và lộ trình hoạt động",
        "ra quyết định dựa trên dữ liệu",
        "quản lý ngân sách và phân bổ nguồn lực",
        "lãnh đạo và xây dựng đội ngũ",
        "đàm phán hợp đồng và đối tác",
        "quản trị rủi ro và tuân thủ",
    ],
    "Chuyên môn kỹ thuật bậc cao": [
        "phân tích chuyên sâu trong lĩnh vực hành nghề",
        "viết báo cáo kỹ thuật và xuất bản",
        "thiết kế giải pháp dựa trên cơ sở khoa học",
        "thuyết trình kết quả cho khách hàng và đồng nghiệp",
        "đào tạo và hướng dẫn chuyên môn",
        "sử dụng phần mềm chuyên ngành",
    ],
    "Chuyên môn kỹ thuật bậc trung": [
        "thực hiện quy trình kỹ thuật theo quy chuẩn",
        "kiểm tra chất lượng và hiệu chuẩn thiết bị",
        "ghi chép tài liệu kỹ thuật chính xác",
        "phối hợp với chuyên gia bậc cao",
        "đào tạo nhân viên mới",
        "khắc phục sự cố vận hành cơ bản",
    ],
    "Nhân viên": [
        "soạn thảo và lưu trữ văn bản hành chính",
        "nhập liệu và tổng hợp dữ liệu trên Excel",
        "trao đổi qua email và điện thoại",
        "lên lịch họp và quản lý lịch công tác",
        "xử lý chứng từ và hồ sơ",
        "hỗ trợ khách hàng bằng tiếng Việt và tiếng Anh cơ bản",
    ],
    "Dịch vụ cá nhân, bảo vệ bán hàng": [
        "tư vấn sản phẩm và dịch vụ",
        "vận hành máy POS và xử lý thanh toán",
        "giữ vệ sinh khu vực bán hàng theo tiêu chuẩn 5S",
        "xử lý khiếu nại của khách",
        "kiểm kê hàng hóa và đặt hàng bổ sung",
        "đào tạo nhân viên mới về dịch vụ khách hàng",
    ],
    "Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản": [
        "lập lịch mùa vụ và luân canh",
        "sử dụng phân bón và thuốc bảo vệ thực vật an toàn",
        "vận hành máy nông cơ nhỏ",
        "chăm sóc gia súc gia cầm",
        "thu hoạch, sơ chế và bảo quản nông sản",
        "đọc dự báo thời tiết và lên kế hoạch ứng phó",
    ],
    "Thợ thủ công và các thợ khác có liên quan": [
        "đo đạc và đọc bản vẽ kỹ thuật cơ bản",
        "vận hành công cụ tay và máy nhỏ an toàn",
        "lắp ráp và sửa chữa cấu kiện",
        "kiểm tra chất lượng theo dung sai",
        "bảo trì thiết bị định kỳ",
        "tuân thủ quy định an toàn lao động",
    ],
    "Thợ lắp ráp và vận hành máy móc, thiết bị": [
        "vận hành máy móc theo quy trình SOP",
        "đọc đồng hồ đo và bảng điều khiển",
        "thực hiện kiểm tra trước ca và bảo dưỡng định kỳ",
        "phát hiện và xử lý sự cố cơ bản",
        "ghi nhật ký vận hành",
        "tuân thủ tiêu chuẩn an toàn và 5S",
    ],
    "Nghề giản đơn": [
        "thực hiện công việc thể chất bền bỉ",
        "tuân thủ hướng dẫn của tổ trưởng",
        "phối hợp làm việc theo nhóm",
        "giữ gìn vệ sinh và an toàn nơi làm việc",
        "vận chuyển và sắp xếp hàng hóa",
        "ghi nhớ quy trình lặp lại",
    ],
    "Khác": [
        "kỹ năng giao tiếp đa dạng",
        "thích nghi nhanh với công việc mới",
        "tổ chức và quản lý thời gian",
        "tự học qua tài liệu trực tuyến",
        "phối hợp nhóm liên chức năng",
    ],
    "Không áp dụng": [
        "kỹ năng vun vén gia đình",
        "quản lý chi tiêu hàng tháng",
        "chăm sóc người thân",
        "tham gia hoạt động cộng đồng",
        "tự học qua sách báo và internet",
    ],
}

OCCUPATION_SKILLS_EN: Final[dict[str, list[str]]] = {
    "Nhà lãnh đạo": [
        "strategic planning and roadmap setting",
        "data-driven decision making",
        "budget management and resource allocation",
        "team leadership and team building",
        "contract and partnership negotiation",
        "risk management and compliance",
    ],
    "Chuyên môn kỹ thuật bậc cao": [
        "deep professional analysis in the field",
        "writing technical reports and publications",
        "designing science-based solutions",
        "presenting results to clients and peers",
        "professional training and mentorship",
        "using domain-specific software",
    ],
    "Chuyên môn kỹ thuật bậc trung": [
        "executing technical procedures to standard",
        "quality control and equipment calibration",
        "accurate technical documentation",
        "coordinating with senior specialists",
        "training new technicians",
        "basic operational troubleshooting",
    ],
    "Nhân viên": [
        "drafting and filing administrative documents",
        "data entry and Excel-based reporting",
        "email and phone correspondence",
        "scheduling meetings and travel",
        "handling vouchers and dossiers",
        "basic English/Vietnamese customer support",
    ],
    "Dịch vụ cá nhân, bảo vệ bán hàng": [
        "product and service consultation",
        "operating POS systems and processing payments",
        "maintaining a 5S-clean sales floor",
        "handling customer complaints",
        "stocktaking and replenishment",
        "training new staff on customer service",
    ],
    "Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản": [
        "seasonal planning and crop rotation",
        "safe use of fertilizers and pesticides",
        "operating small farm machinery",
        "livestock and poultry husbandry",
        "harvest, preliminary processing, and storage",
        "reading weather forecasts and planning around them",
    ],
    "Thợ thủ công và các thợ khác có liên quan": [
        "measuring and reading basic technical drawings",
        "safe operation of hand and small power tools",
        "assembling and repairing components",
        "tolerance-based quality inspection",
        "scheduled equipment maintenance",
        "compliance with occupational-safety regulations",
    ],
    "Thợ lắp ráp và vận hành máy móc, thiết bị": [
        "operating machinery to SOP",
        "reading gauges and control panels",
        "pre-shift checks and scheduled maintenance",
        "detecting and resolving basic faults",
        "logging operational data",
        "compliance with safety and 5S standards",
    ],
    "Nghề giản đơn": [
        "sustained physical work",
        "following team-leader instructions",
        "collaborating in small teams",
        "workplace cleanliness and safety",
        "moving and stacking goods",
        "remembering repetitive procedures",
    ],
    "Khác": [
        "versatile communication skills",
        "rapid adaptation to new work",
        "personal organisation and time management",
        "self-study through online materials",
        "cross-functional team collaboration",
    ],
    "Không áp dụng": [
        "household management skills",
        "monthly budgeting",
        "caring for family members",
        "participating in community activities",
        "self-study through books and the internet",
    ],
}


# ---------------------------------------------------------------------------
# 2. Per-region cultural / culinary / scenic seeds
# ---------------------------------------------------------------------------
# One entry per macro-region (NSO 6-region partition). Each value bundles
# a few natural-language seeds the templates can mix in. Keys are the
# canonical Vietnamese region labels.

REGION_DISHES_VI: Final[dict[str, list[str]]] = {
    "Đồng bằng sông Hồng": [
        "phở bò Hà Nội", "bún chả", "bánh cuốn Thanh Trì",
        "chả cá Lã Vọng", "cốm làng Vòng", "bún riêu cua",
    ],
    "Trung du và miền núi phía Bắc": [
        "thắng cố", "xôi ngũ sắc", "phở chua", "thịt trâu gác bếp",
        "rượu ngô", "cá nướng Pa Pỉnh Tộp",
    ],
    "Bắc Trung Bộ và Duyên hải miền Trung": [
        "bún bò Huế", "mì Quảng", "cơm hến", "bánh xèo miền Trung",
        "bánh bèo", "nem chua Thanh Hoá",
    ],
    "Tây Nguyên": [
        "cơm lam", "gà nướng Bản Đôn", "rượu cần", "lẩu lá rừng",
        "thịt heo gác bếp", "cà phê Buôn Ma Thuột",
    ],
    "Đông Nam Bộ": [
        "cơm tấm Sài Gòn", "bánh mì thịt nguội", "hủ tiếu Nam Vang",
        "lẩu mắm", "gỏi cuốn", "bò lá lốt",
    ],
    "Đồng bằng sông Cửu Long": [
        "lẩu cá linh bông điên điển", "bún cá Châu Đốc", "hủ tiếu Mỹ Tho",
        "bánh xèo miền Tây", "cá lóc nướng trui", "bánh tét lá cẩm",
    ],
}

REGION_DISHES_EN: Final[dict[str, list[str]]] = {
    "Đồng bằng sông Hồng": [
        "Hanoi-style beef phở", "bún chả grilled pork with noodles",
        "Thanh Trì rolled rice cakes", "Lã Vọng turmeric fish",
        "Vòng-village young rice", "bún riêu crab noodle soup",
    ],
    "Trung du và miền núi phía Bắc": [
        "thắng cố highland horse stew", "five-colour sticky rice",
        "sour phở", "smoked buffalo from the rafters",
        "corn rice wine", "Pa Pỉnh Tộp grilled fish",
    ],
    "Bắc Trung Bộ và Duyên hải miền Trung": [
        "Huế-style spicy beef noodles", "mì Quảng turmeric noodles",
        "cơm hến baby-clam rice", "Central-style crispy pancake",
        "bánh bèo steamed rice cakes", "Thanh Hoá fermented pork",
    ],
    "Tây Nguyên": [
        "bamboo sticky rice", "Bản Đôn grilled chicken",
        "rượu cần communal jar wine", "wild-leaf hotpot",
        "highland smoked pork", "Buôn Ma Thuột coffee",
    ],
    "Đông Nam Bộ": [
        "Saigon broken-rice plates", "cold-cut bánh mì",
        "Nam Vang noodle soup", "fermented-fish hotpot",
        "fresh summer rolls", "lá lốt-wrapped beef",
    ],
    "Đồng bằng sông Cửu Long": [
        "linh-fish and dien-dien-flower hotpot",
        "Châu Đốc fish noodle soup", "Mỹ Tho-style rice noodle soup",
        "Mekong-style crispy pancake", "open-flame grilled snakehead",
        "purple-leaf bánh tét",
    ],
}


REGION_SCENIC_VI: Final[dict[str, list[str]]] = {
    "Đồng bằng sông Hồng": [
        "phố cổ Hà Nội", "vịnh Hạ Long", "Tam Cốc – Bích Động",
        "chùa Bái Đính", "Tràng An",
    ],
    "Trung du và miền núi phía Bắc": [
        "ruộng bậc thang Mù Cang Chải", "Sa Pa", "hồ Ba Bể",
        "cao nguyên đá Đồng Văn", "thác Bản Giốc",
    ],
    "Bắc Trung Bộ và Duyên hải miền Trung": [
        "phố cổ Hội An", "kinh thành Huế", "động Phong Nha",
        "biển Mỹ Khê Đà Nẵng", "Lăng Cô",
    ],
    "Tây Nguyên": [
        "thác Dray Nur", "hồ Lắk", "núi Lang Biang",
        "rừng quốc gia Yok Đôn", "buôn Đôn",
    ],
    "Đông Nam Bộ": [
        "địa đạo Củ Chi", "núi Bà Đen", "Côn Đảo",
        "Vũng Tàu", "phố đi bộ Nguyễn Huệ",
    ],
    "Đồng bằng sông Cửu Long": [
        "chợ nổi Cái Răng", "rừng tràm Trà Sư", "đảo Phú Quốc",
        "miệt vườn Cái Bè", "biển Mũi Né",
    ],
}

REGION_SCENIC_EN: Final[dict[str, list[str]]] = {
    "Đồng bằng sông Hồng": [
        "Hanoi's Old Quarter", "Hạ Long Bay", "Tam Cốc – Bích Động",
        "Bái Đính pagoda", "Tràng An scenic complex",
    ],
    "Trung du và miền núi phía Bắc": [
        "the Mù Cang Chải rice terraces", "Sa Pa", "Ba Bể Lake",
        "the Đồng Văn Karst Plateau", "Bản Giốc Waterfall",
    ],
    "Bắc Trung Bộ và Duyên hải miền Trung": [
        "Hội An Ancient Town", "the Imperial City of Huế",
        "Phong Nha Cave", "Mỹ Khê Beach in Đà Nẵng", "Lăng Cô lagoon",
    ],
    "Tây Nguyên": [
        "Dray Nur Waterfall", "Lắk Lake", "Lang Biang Mountain",
        "Yok Đôn National Park", "Buôn Đôn village",
    ],
    "Đông Nam Bộ": [
        "the Củ Chi tunnels", "Bà Đen Mountain", "Côn Đảo islands",
        "Vũng Tàu seaside", "Nguyễn Huệ pedestrian street",
    ],
    "Đồng bằng sông Cửu Long": [
        "the Cái Răng floating market", "Trà Sư cajuput forest",
        "Phú Quốc Island", "Cái Bè orchards", "Mũi Né beach",
    ],
}

REGION_CULTURE_VI: Final[dict[str, list[str]]] = {
    "Đồng bằng sông Hồng": [
        "ca trù", "chèo", "quan họ Bắc Ninh", "lễ hội đình làng",
        "tranh Đông Hồ",
    ],
    "Trung du và miền núi phía Bắc": [
        "chợ phiên vùng cao", "thổ cẩm Mông – Dao", "lễ hội Lồng Tồng",
        "khèn Mông", "ẩm thực sương sa",
    ],
    "Bắc Trung Bộ và Duyên hải miền Trung": [
        "nhã nhạc cung đình Huế", "ca Huế trên sông Hương",
        "lễ hội Cầu Ngư", "bài chòi", "đèn lồng Hội An",
    ],
    "Tây Nguyên": [
        "không gian văn hoá cồng chiêng",
        "sử thi Đam San", "lễ bỏ mả",
        "nhà rông", "đan lát mây tre",
    ],
    "Đông Nam Bộ": [
        "đờn ca tài tử", "cải lương Sài Gòn", "lễ hội bà Chúa Xứ",
        "võ cổ truyền Bình Định", "ẩm thực đường phố",
    ],
    "Đồng bằng sông Cửu Long": [
        "đờn ca tài tử Nam Bộ", "lễ Ok Om Bok của người Khmer",
        "đua ghe ngo", "đờn cò sáo trúc miệt vườn",
        "chợ nổi", "vọng cổ",
    ],
}

REGION_CULTURE_EN: Final[dict[str, list[str]]] = {
    "Đồng bằng sông Hồng": [
        "ca trù chamber music", "chèo opera", "quan họ Bắc Ninh duets",
        "village đình festivals", "Đông Hồ folk paintings",
    ],
    "Trung du và miền núi phía Bắc": [
        "highland market days", "Mông and Dao brocade",
        "the Lồng Tồng harvest festival", "the khèn pan-pipe of the Mông",
        "highland herbal cuisine",
    ],
    "Bắc Trung Bộ và Duyên hải miền Trung": [
        "Huế royal court music (nhã nhạc)",
        "ca Huế river-boat singing", "the Cầu Ngư fishermen's festival",
        "bài chòi sung-card folk art", "Hội An lanterns",
    ],
    "Tây Nguyên": [
        "the Space of Gong Culture",
        "the Đam San epic", "the lễ bỏ mả grave-leaving rite",
        "stilt-house rông communal halls", "rattan and bamboo weaving",
    ],
    "Đông Nam Bộ": [
        "đờn ca tài tử southern chamber music",
        "Saigon-style cải lương theatre",
        "the Bà Chúa Xứ Pilgrimage", "Bình Định traditional martial arts",
        "vibrant street-food culture",
    ],
    "Đồng bằng sông Cửu Long": [
        "Mekong đờn ca tài tử music",
        "the Khmer Ok Om Bok festival",
        "ngo-boat racing", "river-orchard folk-flute repertoire",
        "floating markets", "vọng cổ blues-style ballads",
    ],
}


REGION_SPORTS_VI: Final[dict[str, list[str]]] = {
    "Đồng bằng sông Hồng": [
        "bóng đá sân bảy", "cầu lông tại nhà thi đấu phường",
        "đi bộ quanh hồ Hoàn Kiếm", "đạp xe quanh Hồ Tây",
        "võ cổ truyền", "yoga tại công viên",
    ],
    "Trung du và miền núi phía Bắc": [
        "leo núi và trekking", "đua ngựa và đẩy gậy ngày hội",
        "kéo co thôn bản", "bóng chuyền sân đất",
        "đi bộ đường dài",
    ],
    "Bắc Trung Bộ và Duyên hải miền Trung": [
        "bơi biển và lướt ván", "đua thuyền truyền thống",
        "võ cổ truyền Bình Định", "bóng đá phong trào",
        "chạy bộ ven biển",
    ],
    "Tây Nguyên": [
        "leo núi", "đua voi (lễ hội)", "bóng đá đồng đội",
        "đi rừng",
        "võ cổ truyền",
    ],
    "Đông Nam Bộ": [
        "bóng đá sân bảy", "tennis sân câu lạc bộ",
        "yoga và Pilates phòng tập", "chạy bộ ở công viên Tao Đàn",
        "đạp xe đường dài cuối tuần", "bóng rổ phong trào",
    ],
    "Đồng bằng sông Cửu Long": [
        "đua ghe ngo", "bóng đá sân bảy",
        "bơi sông và bơi hồ", "đạp xe miệt vườn",
        "cầu lông sân nhà",
    ],
}

REGION_SPORTS_EN: Final[dict[str, list[str]]] = {
    "Đồng bằng sông Hồng": [
        "seven-a-side football", "badminton at the ward gym",
        "walking laps around Hoàn Kiếm Lake", "cycling around West Lake",
        "traditional Vietnamese martial arts", "park-side yoga",
    ],
    "Trung du và miền núi phía Bắc": [
        "mountain hiking and trekking", "festival horse racing and pole-pushing",
        "village tug-of-war", "dirt-pitch volleyball",
        "long-distance walking",
    ],
    "Bắc Trung Bộ và Duyên hải miền Trung": [
        "sea swimming and surfing", "traditional boat racing",
        "Bình Định martial arts", "neighbourhood football",
        "running along the seafront",
    ],
    "Tây Nguyên": [
        "mountain hiking", "festival elephant racing",
        "team football", "forest walking",
        "traditional martial arts",
    ],
    "Đông Nam Bộ": [
        "seven-a-side football", "club tennis",
        "yoga and Pilates studios", "jogging at Tao Đàn Park",
        "weekend long-distance cycling", "informal basketball",
    ],
    "Đồng bằng sông Cửu Long": [
        "ngo-boat racing", "seven-a-side football",
        "river and pool swimming", "orchard cycling routes",
        "courtyard badminton",
    ],
}


# ---------------------------------------------------------------------------
# 3. Hobbies / interests indexed by area (urban/rural). Two compact baskets.
# ---------------------------------------------------------------------------
HOBBIES_URBAN_VI: Final[list[str]] = [
    "uống cà phê và đọc sách buổi sáng",
    "đi xem phim Việt cuối tuần",
    "nấu ăn theo công thức trên YouTube",
    "tập gym ba buổi mỗi tuần",
    "chơi cờ vua online",
    "chụp ảnh phố phường bằng điện thoại",
    "đi bộ trong công viên",
    "viết blog hoặc Facebook chia sẻ trải nghiệm",
    "trồng cây cảnh trong căn hộ",
    "học tiếng Anh qua ứng dụng",
]
HOBBIES_URBAN_EN: Final[list[str]] = [
    "morning coffee with a book",
    "weekend trips to the cinema for Vietnamese films",
    "cooking from YouTube recipes",
    "the gym three times a week",
    "online chess",
    "phone-camera street photography",
    "walks in the city park",
    "writing a personal Facebook or blog journal",
    "tending balcony plants",
    "studying English through mobile apps",
]

HOBBIES_RURAL_VI: Final[list[str]] = [
    "chăm sóc vườn rau và cây ăn quả",
    "câu cá ở ao làng",
    "trò chuyện với hàng xóm cuối ngày",
    "đi chợ phiên",
    "đan lát và làm việc thủ công",
    "tham gia lễ hội đình làng",
    "tự nấu rượu nếp",
    "chăm sóc gia súc, gia cầm",
    "đi xe máy thăm họ hàng",
    "nghe đài và xem tin tức buổi tối",
]
HOBBIES_RURAL_EN: Final[list[str]] = [
    "tending the home garden and fruit trees",
    "fishing at the village pond",
    "evening chats with neighbours",
    "visiting the periodic village market",
    "weaving and other handicrafts",
    "joining the local đình festival",
    "making home-brewed sticky-rice wine",
    "looking after livestock and poultry",
    "motorbike trips to visit relatives",
    "evening radio and television news",
]


# ---------------------------------------------------------------------------
# 4. Career-goal seeds, indexed by occupation (English mirror keyed the same)
# ---------------------------------------------------------------------------
GOALS_VI: Final[dict[str, list[str]]] = {
    "Nhà lãnh đạo": [
        "mở rộng quy mô tổ chức và đào tạo lớp kế cận",
        "đạt chứng chỉ quản trị quốc tế (PMP, MBA mini)",
        "đưa doanh nghiệp niêm yết trong vòng năm năm tới",
    ],
    "Chuyên môn kỹ thuật bậc cao": [
        "công bố thêm bài báo khoa học hoặc dự án trọng điểm",
        "lấy thêm chứng chỉ chuyên môn quốc tế",
        "trở thành chuyên gia tư vấn được công nhận",
    ],
    "Chuyên môn kỹ thuật bậc trung": [
        "nâng bậc lên chuyên môn cao trong ba năm tới",
        "lấy thêm chứng chỉ kỹ thuật và đào tạo",
        "phụ trách một dự án độc lập",
    ],
    "Nhân viên": [
        "thăng tiến lên vị trí trưởng nhóm",
        "học thêm kỹ năng số và phân tích dữ liệu",
        "ổn định công việc và tích luỹ tiết kiệm",
    ],
    "Dịch vụ cá nhân, bảo vệ bán hàng": [
        "lên vị trí trưởng ca hoặc cửa hàng trưởng",
        "học quản trị bán lẻ",
        "tích luỹ vốn để mở cửa hàng riêng",
    ],
    "Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản": [
        "mở rộng diện tích canh tác hoặc đàn nuôi",
        "áp dụng kỹ thuật mới (VietGAP, hữu cơ)",
        "liên kết hợp tác xã để bao tiêu sản phẩm",
    ],
    "Thợ thủ công và các thợ khác có liên quan": [
        "lấy bằng nghề bậc cao hơn",
        "mở xưởng riêng cùng vài đồng nghiệp",
        "ổn định việc làm và mua đất xây nhà",
    ],
    "Thợ lắp ráp và vận hành máy móc, thiết bị": [
        "lên vị trí ca trưởng",
        "học bảo trì cơ điện và tự động hoá",
        "ổn định thu nhập để lo cho gia đình",
    ],
    "Nghề giản đơn": [
        "tìm được công việc ổn định và lâu dài",
        "học thêm nghề để chuyển sang công việc kỹ thuật",
        "tích luỹ tiết kiệm cho con đi học",
    ],
    "Khác": [
        "tìm hiểu các cơ hội nghề mới phù hợp với sức khoẻ",
        "học thêm kỹ năng để mở rộng thu nhập",
        "ổn định cuộc sống và lo cho gia đình",
    ],
    "Không áp dụng": [
        "ưu tiên chăm sóc gia đình và sức khoẻ bản thân",
        "tham gia hoạt động cộng đồng địa phương",
        "tự học để duy trì nhịp sống tích cực",
    ],
}

GOALS_EN: Final[dict[str, list[str]]] = {
    "Nhà lãnh đạo": [
        "grow the organisation and develop the next leadership tier",
        "earn an international management credential such as a PMP or mini-MBA",
        "take the company public within the next five years",
    ],
    "Chuyên môn kỹ thuật bậc cao": [
        "publish further peer-reviewed work or lead a flagship project",
        "earn additional internationally recognised credentials",
        "become a recognised consultant in the field",
    ],
    "Chuyên môn kỹ thuật bậc trung": [
        "promote to senior specialist within three years",
        "earn further technical certifications and training",
        "lead an independent project",
    ],
    "Nhân viên": [
        "promote to team-leader",
        "build digital and data-analysis skills",
        "stabilise the role and accumulate savings",
    ],
    "Dịch vụ cá nhân, bảo vệ bán hàng": [
        "promote to shift supervisor or shop manager",
        "study retail management",
        "save towards opening a small shop",
    ],
    "Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản": [
        "expand the farmed area or herd",
        "adopt modern standards (VietGAP, organic)",
        "join a cooperative for guaranteed offtake",
    ],
    "Thợ thủ công và các thợ khác có liên quan": [
        "earn a higher trade certificate",
        "co-found a small workshop with a few colleagues",
        "achieve stable work and save towards a family home",
    ],
    "Thợ lắp ráp và vận hành máy móc, thiết bị": [
        "promote to shift leader",
        "study mechatronics and automation maintenance",
        "stabilise income to support the family",
    ],
    "Nghề giản đơn": [
        "find stable, long-term work",
        "learn a trade to step up to skilled work",
        "save for the children's education",
    ],
    "Khác": [
        "explore new career options suited to one's health",
        "learn skills that broaden the income base",
        "stabilise life and provide for the family",
    ],
    "Không áp dụng": [
        "prioritise family care and personal health",
        "join local community activities",
        "study to keep an active rhythm to daily life",
    ],
}


# Validation that the bilingual lookup tables agree key-for-key. We do
# this at import time so a stale lookup is caught loudly during dev.
def _check_parallel(left: dict[str, list[str]], right: dict[str, list[str]],
                    name: str) -> None:
    if left.keys() != right.keys():
        missing_left = right.keys() - left.keys()
        missing_right = left.keys() - right.keys()
        raise AssertionError(
            f"Bilingual mismatch in {name}: "
            f"left missing {missing_left}, right missing {missing_right}"
        )
    for k in left:
        if len(left[k]) != len(right[k]):
            raise AssertionError(
                f"Bilingual mismatch in {name}[{k!r}]: "
                f"len(vi)={len(left[k])} vs len(en)={len(right[k])}"
            )


_check_parallel(OCCUPATION_SKILLS_VI, OCCUPATION_SKILLS_EN, "OCCUPATION_SKILLS")
_check_parallel(REGION_DISHES_VI, REGION_DISHES_EN, "REGION_DISHES")
_check_parallel(REGION_SCENIC_VI, REGION_SCENIC_EN, "REGION_SCENIC")
_check_parallel(REGION_CULTURE_VI, REGION_CULTURE_EN, "REGION_CULTURE")
_check_parallel(REGION_SPORTS_VI, REGION_SPORTS_EN, "REGION_SPORTS")
_check_parallel(GOALS_VI, GOALS_EN, "GOALS")
assert len(HOBBIES_URBAN_VI) == len(HOBBIES_URBAN_EN), "urban hobbies bilingual length mismatch"
assert len(HOBBIES_RURAL_VI) == len(HOBBIES_RURAL_EN), "rural hobbies bilingual length mismatch"
