"""Build DATAANALYSIS.ipynb programmatically.

Run this script to (re-)generate the notebook. It walks all 12 NSO PX-Web
databases (across 15 matrix prefixes V01..V15), each section being:

    - A markdown intro that connects the section back to the new ontology
      layers in `configs/ontology.yaml`.
    - Working Python cells that load the relevant parquets and produce
      Plotly figures styled with `scripts._nvidia_style`.
    - Each figure cell calls `save_figure(...)` so the PNG snapshot lands
      under `docs/figures/analysis/` for `DATAANALYSIS.md` to embed.

We build the notebook this way (instead of editing it cell-by-cell)
because the analysis is large (~38 figures) and a single declarative
build script is far easier to keep in sync with the ontology than 38
manual edits.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "DATAANALYSIS.ipynb"


def md(src: str) -> dict:
    """Return a markdown cell."""
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str) -> dict:
    """Return a code cell."""
    return nbf.v4.new_code_cell(src.strip("\n"))


# ---------------------------------------------------------------------------
# Cell list — every cell of the notebook in order.
# ---------------------------------------------------------------------------
cells: list[dict] = []


cells.append(md(r"""
# Data analysis — what the curated NSO PX-Web data actually says

Companion notebook to [`DATAPROCESSING.md`](DATAPROCESSING.md) and
[`DATASYNTHESIS.md`](DATASYNTHESIS.md). Where those documents describe
how the system *works*, this notebook is a deep guided tour of **what
the 502 PX-Web tables / 316,108 long-format cells actually say** about
Vietnam — and where the gaps are.

Coverage:

* **All 12 NSO databases** of the new ontology (15 PX-Web matrix
  prefixes V01..V15) get their own section.
* **All 21 statistical domains** are represented at least once.
* **All 11 persona dimensions** in `configs/ontology.yaml` are connected
  back to their PX-Web source(s) in §15.
* The full 502-row catalogue is summarised in §16.

Every figure follows the
[NVIDIA brand guidelines](https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/):

* **White** background
* **NVIDIA Green** `#76B900` for the primary data series
* **Black** for text + axis chrome
* **NVIDIA Sans** (with safe fallbacks) typography
* Plotly for interactive HTML; Kaleido-rendered PNG snapshots saved to
  `docs/figures/analysis/` for [`DATAANALYSIS.md`](DATAANALYSIS.md) to
  embed inline.

Run prerequisites:

```bash
make curate                              # populates data/nso-gov-vn/raw/pxweb/vi/
pip install -e ".[curator,viz]"          # plotly + kaleido v1+ + pandas + openpyxl
```
"""))


cells.append(md("## §0 — Setup"))


cells.append(code(r"""
'''Setup — imports, NVIDIA Plotly theme, helper to load any PX-Web table by id.'''
from __future__ import annotations

import json
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings('ignore')

import sys
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._nvidia_style import (
    apply_nvidia_style, save_figure,
    NV_GREEN, NV_GREEN_DARK, NV_GREEN_SOFT,
    NV_BLACK, NV_DARK, NV_GREY, NV_LIGHT_GREY, NV_FAINT,
    NV_WHITE, NV_DISCRETE, NV_SEQUENTIAL, NV_FONT_FAMILY,
)

PXWEB_DIR = REPO_ROOT / 'data' / 'nso-gov-vn' / 'raw' / 'pxweb' / 'vi'
OUT_DIR   = REPO_ROOT / 'docs' / 'figures' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

pio.templates['nvidia'] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=13),
        colorway=NV_DISCRETE,
    )
)
pio.templates.default = 'nvidia'

print(f'PX-Web directory: {PXWEB_DIR}')
print(f'output directory: {OUT_DIR}')
"""))


cells.append(code(r"""
'''Bilingual labelling helpers — every figure title, axis label, and legend
   entry is rendered as "Vietnamese / English" so the analysis is readable
   in either language.

   - `vi_en(vi, en)` — compose a paired label "<vi> / <en>".
   - `tr(label)` — translate a known PX-Web category to bilingual form;
     unknown categories pass through unchanged so we never silently lose
     a province or industry name.
'''

# Curated translations for the categorical dimensions used throughout the
# analysis. Keep this list short on purpose — translating every Vietnamese
# province name would just clutter the charts.
VI_EN_DICT: dict[str, str] = {
    # Macro-regions (NSO's six economic regions of Vietnam)
    'Đồng bằng sông Hồng':                     'Red River Delta',
    'Trung du và miền núi phía Bắc':           'Northern midlands & mountains',
    'Bắc Trung Bộ và Duyên hải miền Trung':    'North Central & Central Coast',
    'Tây Nguyên':                              'Central Highlands',
    'Đông Nam Bộ':                             'South East',
    'Đồng bằng sông Cửu Long':                 'Mekong Delta',

    # Sex / urbanicity
    'Nam':         'Male',
    'Nữ':          'Female',
    'Thành thị':   'Urban',
    'Nông thôn':   'Rural',

    # Generic markers
    'Tổng số':                'Total',
    'TỔNG SỐ':                'Total',
    'CẢ NƯỚC':                'Whole country',
    'Phân tổ':                'Breakdown',
    'Cách tính':              'Indicator',
    'Năm':                    'Year',
    'Năm học':                'Academic year',
    'Tỉnh, thành phố':        'Province / city',
    'Tỉnh/thành phố':         'Province / city',
    'Vùng':                   'Region',
    'Thành thị, nông thôn':   'Urban / Rural',
    'Giới tính':              'Sex',
    'Nhóm tuổi':              'Age group',
    'Nghề nghiệp':            'Occupation',
    'Ngành':                  'Industry',
    'Ngành kinh tế':          'Economic sector',
    'Ngành công nghiệp':      'Industrial sector',
    'Cấp học':                'Education level',
    'Chuyên môn kỹ thuật':    'Qualification',
    'Trình độ chuyên môn kỹ thuật': 'Qualification level',
    'Quốc tịch':              'Nationality',
    'Quốc gia, vùng lãnh thổ': 'Country / territory',
    'Nhóm thu nhập':          'Income quintile',
    'Số trường':              'Number of schools',

    # Education attainment buckets
    'Sơ cấp':         'Elementary',
    'Trung cấp':      'Intermediate',
    'Cao đẳng':       'College',
    'Đại học trở lên': 'University+',
    'Không có trình độ CMKT': 'No formal qualification',

    # Investment sector types (V04.01)
    'Kinh tế Nhà nước':                   'State sector',
    'Kinh tế ngoài Nhà nước':             'Non-state sector',
    'Khu vực có vốn đầu tư nước ngoài':   'FDI sector',

    # Income quintiles (V14.30)
    'Nhóm 1': 'Q1 (lowest)', 'Nhóm 2': 'Q2',
    'Nhóm 3': 'Q3',          'Nhóm 4': 'Q4',
    'Nhóm 5': 'Q5 (highest)',

    # Top ASEAN peer countries (V15.05). NSO PX-Web exports country
    # names in phonetic-hyphenated Vietnamese rather than the
    # short-form Latin spellings (``Singapore``, ``Malaysia`` etc.).
    # We keep both forms here so external code can look up either
    # convention and get the canonical English name.
    'Việt Nam':         'Vietnam',
    'Thái Lan':         'Thailand',
    'Xin-ga-po':        'Singapore',
    'In-đô-nê-xi-a':    'Indonesia',
    'Phi-li-pin':       'Philippines',
    'Ma-lai-xi-a':      'Malaysia',
    'Mi-an-ma':         'Myanmar',
    'Cam-pu-chia':      'Cambodia',
    'Lào':              'Laos',
    'Bru-nây':          'Brunei',
    'Đông Ti Mo':       'Timor-Leste',
    # Latin-Vietnamese aliases for backwards compat (older code paths).
    'Singapore':        'Singapore',
    'Inđônêxia':        'Indonesia',
    'Philíppin':        'Philippines',
    'Malaysia':         'Malaysia',
    'Mianma':           'Myanmar',

    # PX-Web database names (used as y-axis ticks in fig 04)
    'Đơn vị hành chính, đất đai và khí hậu': 'Administrative Units, Land & Climate',
    'Dân số và lao động':                    'Population & Employment',
    'Tài khoản quốc gia':                    'National Accounts & State Budget',
    'Đầu tư':                                'Investment',
    'Doanh nghiệp':                          'Enterprises',
    'Nông, lâm nghiệp và thủy sản':           'Agriculture, Forestry & Fishery',
    'Công nghiệp':                           'Industry',
    'Thương mại, giá cả':                    'Trade, Prices & Tourism',
    'Vận tải và bưu điện':                   'Transport, Postal & Telecom',
    'Giáo dục':                              'Education',
    'Y tế, văn hóa và đời sống':             'Health, Culture & Living Standards',
    'Thống kê nước ngoài':                   'International Statistics',

    # Livestock & poultry species (V06.41 — note PX-Web annotation marks)
    'Trâu':                            'Buffalo',
    'Bò':                              'Cattle',
    'Lợn':                             'Pig',
    'Lợn (*)':                         'Pig',
    'Gia cầm':                         'Poultry',
    'Gia cầm (**) (Triệu con)':        'Poultry (millions)',
    'Gà':                              'Chicken',
    'Vịt':                             'Duck',
    'Ngựa':                            'Horse',
    'Dê':                              'Goat',
    'Dê, cừu':                         'Goat & sheep',
    'Cừu':                             'Sheep',

    # Livestock product output (V06.46)
    'Sản lượng thịt trâu hơi xuất chuồng (Nghìn tấn)':
        'Buffalo meat output (thousand tonnes)',
    'Sản lượng thịt bò hơi xuất chuồng (Nghìn tấn)':
        'Beef output (thousand tonnes)',
    'Sản lượng thịt lợn hơi xuất chuồng (Nghìn tấn)':
        'Pork output (thousand tonnes)',
    'Sản lượng thịt gia cầm hơi giết, bán (Nghìn tấn)':
        'Poultry meat output (thousand tonnes)',
    'Sản lượng sữa tươi (Triệu lít)':
        'Fresh milk (million litres)',
    'Trứng gia cầm (Triệu quả)':
        'Poultry eggs (million)',
    'Sản lượng kén tằm (Tấn)':
        'Silkworm cocoon (tonnes)',
    'Sản lượng mật ong (Tấn)':
        'Honey (tonnes)',

    # Industrial sectors (V07.01 — major collapsed groups)
    'Toàn ngành công nghiệp':                       'All industry',
    'Khai khoáng':                                  'Mining & quarrying',
    'Công nghiệp chế biến, chế tạo':                'Manufacturing',
    'Sản xuất và phân phối điện':                   'Electricity & gas',
    'Cung cấp nước; hoạt động quản lý và xử lý rác thải, nước thải':
        'Water supply & waste treatment',

    # Education levels (V13.04 — cấp học)
    'Mầm non':            'Pre-school',
    'Tiểu học':           'Primary',
    'Trung học cơ sở':    'Lower secondary',
    'Trung học phổ thông': 'Upper secondary',
    'Phổ thông':          'General education',
    'Trung cấp chuyên nghiệp': 'Professional intermediate',
    'Đại học':            'University',
    'Cao đẳng nghề':      'Vocational college',
    'Trung cấp nghề':     'Vocational intermediate',
    'Số nữ giáo viên và học sinh': 'Female teachers & pupils',
    'Số trường, lớp, giáo viên và học sinh':
        'Schools, classes, teachers, pupils',

    # ISCO occupation groups (V02.43 — Nghề nghiệp)
    'Nhà lãnh đạo':                        'Managers',
    'Chuyên môn kỹ thuật bậc cao':         'Professionals',
    'Chuyên môn kỹ thuật bậc trung':       'Technicians & associate professionals',
    'Nhân viên':                           'Clerical support',
    'Dịch vụ cá nhân, bảo vệ bán hàng':    'Service & sales workers',
    'Lao động có kỹ năng trong nông nghiệp, lâm nghệp và thủy sản':
        'Skilled agricultural / forestry / fishery workers',
    'Thợ thủ công và các thợ khác có liên quan':
        'Craft & related trades workers',
    'Thợ lắp ráp và vận hành máy móc, thiết bị':
        'Plant & machine operators',
    'Nghề giản đơn':                       'Elementary occupations',
    'Khác':                                'Other',

    # ISIC sectors (V02.42 / V02.57 / V04.04 / V05.01 / V09.02 — Ngành / Ngành kinh tế)
    'Nông nghiệp, lâm nghiệp và thủy sản': 'Agriculture, forestry & fishery',
    'Nông nghiệp, lâm nghiệp và thuỷ sản': 'Agriculture, forestry & fishery',
    'Khai khoáng':                         'Mining & quarrying',
    'Công nghiệp chế biến, chế tạo':        'Manufacturing',
    'Sản xuất và phân phối điện, khí đốt, nước nóng, hơi nước và điều hòa không khí':
        'Electricity, gas & air-conditioning supply',
    'Sản xuất và phân phối điện, khí đốt, nước nóng, hơi nước và điều hoà không khí':
        'Electricity, gas & air-conditioning supply',
    'Sản xuất và phân phối điện, nước; cung cấp nước; hoạt động quản lý và xử lý rác thải, nước thải':
        'Utilities (electricity, water, waste)',
    'Cung cấp nước; hoạt động quản lý và xử lý rác thải, nước thải':
        'Water supply & waste management',
    'Xây dựng':                            'Construction',
    'Bán buôn và bán lẻ; sửa chữa ô tô, mô tô, xe máy và xe có động cơ khác':
        'Wholesale & retail, vehicle repair',
    'Vận tải, kho bãi':                    'Transport & storage',
    'Vận tải kho bãi':                     'Transport & storage',
    'Dịch vụ lưu trú và ăn uống':           'Accommodation & food service',
    'Thông tin và truyền thông':           'Information & communication',
    'Hoạt động tài chính, ngân hàng và bảo hiểm':
        'Finance, banking & insurance',
    'Hoạt động kinh doanh bất động sản':   'Real estate activities',
    'Hoạt động chuyên môn, khoa học và công nghệ':
        'Professional, scientific & technical',
    'Hoạt động hành chính và dịch vụ hỗ trợ':
        'Administrative & support services',
    'Hoạt động của Đảng Cộng sản, tổ chức chính trị - xã hội; quản lý Nhà nước, an ninh quốc phòng; đảm bảo xã hội bắt buộc':
        'Public administration & defense',
    'Giáo dục và đào tạo':                 'Education',
    'Y tế và hoạt động trợ giúp xã hội':   'Health & social work',
    'Nghệ thuật, vui chơi và giải trí':    'Arts, entertainment & recreation',
    'Hoạt động dịch vụ khác':              'Other service activities',
    'Hoạt động khác':                      'Other activities',
    'Hoạt động làm thuê các công việc trong các hộ gia đình, sản xuất sản phẩm, vật chất và dịch vụ tiêu dùng của hộ gia đình':
        'Activities of households as employers',
    'Không phân tổ được':                   'Unclassified',
    'Công nghiệp':                         'Industry',
    'Công nghiệp và xây dựng':              'Industry & construction',
    'Dịch vụ':                             'Services',

    # ISIC sub-sectors of Industry (V07.01 — Ngành công nghiệp)
    'Toàn ngành công nghiệp':                            'All industry',
    'Công nghiệp chế biến, chế tạo khác':                 'Other manufacturing',
    'Khai khoáng khác':                                  'Other mining',
    'Khai thác than cứng và than non':                    'Coal & lignite mining',
    'Khai thác dầu thô và khí đốt tự nhiên':              'Crude oil & natural gas',
    'Khai thác quặng kim loại':                           'Metal-ore mining',
    'Khai thác, xử lý và cung cấp nước':                  'Water collection & supply',
    'Hoạt động dịch vụ hỗ trợ khai thác mỏ và quặng':     'Mining support services',
    'Hoạt động thu gom, xử lý và tiêu huỷ rác thải; tái chế phế liệu':
        'Waste collection, treatment & recycling',
    'Sản xuất, chế biến thực phẩm':                       'Food processing',
    'Sản xuất đồ uống':                                  'Beverages',
    'Sản xuất sản phẩm thuốc lá':                         'Tobacco products',
    'Dệt':                                               'Textiles',
    'Sản xuất trang phục':                                'Wearing apparel',
    'Sản xuất da và các sản phẩm có liên quan':           'Leather & related products',
    'Chế biến gỗ và sản xuất sản phẩm từ gỗ, tre, nứa (trừ giường, tủ, bàn ghế); sản xuất sản phẩm từ rơm, rạ và vật liệu tết bện':
        'Wood, cork, straw & bamboo products',
    'Sản xuất giấy và sản phẩm từ giấy':                  'Paper & paper products',
    'In, sao chép bản ghi các loại':                       'Printing & reproduction',
    'Sản xuất than cốc, sản phẩm dầu mỏ tinh chế':         'Coke & refined petroleum',
    'Sản xuất hoá chất và sản phẩm hoá chất':             'Chemicals',
    'Sản xuất thuốc, hoá dược và dược liệu':              'Pharmaceuticals',
    'Sản xuất sản phẩm từ cao su và plastic':             'Rubber & plastics',
    'Sản xuất sản phẩm từ khoáng phi kim loại khác':      'Other non-metallic mineral products',
    'Sản xuất kim loại':                                  'Basic metals',
    'Sản xuất sản phẩm từ kim loại đúc sẵn (trừ máy móc, thiết bị)':
        'Fabricated metal products',
    'Sản xuất sản phẩm điện tử, máy vi tính và sản phẩm quang học':
        'Computer, electronic & optical products',
    'Sản xuất thiết bị điện':                             'Electrical equipment',
    'Sản xuất máy móc, thiết bị chưa được phân vào đâu':
        'Other machinery & equipment',
    'Sản xuất xe có động cơ, rơ moóc':                    'Motor vehicles & trailers',
    'Sản xuất phương tiện vận tải khác':                   'Other transport equipment',
    'Sản xuất giường, tủ, bàn, ghế':                      'Furniture',
    'Sửa chữa, bảo dưỡng và lắp đặt máy móc, thiết bị':   'Repair & installation of machinery',

    # Vietnamese centrally-administered cities + major provinces (V03.22, V05.04 etc.)
    # The English form is the official ISO 3166-2:VN romanisation (no diacritics).
    'TP.Hà Nội':           'Hanoi',
    'TP. Hà Nội':          'Hanoi',
    'Hà Nội':              'Hanoi',
    'TP. Hồ Chí Minh':     'Ho Chi Minh City',
    'TP.Hồ Chí Minh':      'Ho Chi Minh City',
    'Hồ Chí Minh':         'Ho Chi Minh City',
    'Hải Phòng':           'Hai Phong',
    'Đà Nẵng':             'Da Nang',
    'Cần Thơ':             'Can Tho',
    'Huế':                 'Hue',
    'Bắc Ninh':            'Bac Ninh',
    'Bình Dương':          'Binh Duong',
    'Hải Dương':           'Hai Duong',
    'Đồng Nai':            'Dong Nai',
    'Quảng Ninh':          'Quang Ninh',
    'Vĩnh Phúc':           'Vinh Phuc',
    'Hà Nam':              'Ha Nam',
    'Thái Nguyên':         'Thai Nguyen',
    'Bắc Giang':           'Bac Giang',
    'Bà Rịa - Vũng Tàu':   'Ba Ria - Vung Tau',
    'Hưng Yên':            'Hung Yen',
    'Long An':             'Long An',
    'Phú Thọ':             'Phu Tho',
    'Cả nước':             'Whole country',
    # Climate stations (V01.13) — VN city / town names
    'Lai Châu':            'Lai Chau',
    'Sơn La':              'Son La',
    'Tuyên Quang':         'Tuyen Quang',
    'Bãi Cháy':            'Bai Chay',
    'Nam Định':            'Nam Dinh',
    'Vinh':                'Vinh',
    'Đà Lạt':              'Da Lat',
    'Pleiku':              'Pleiku',
    'Quy Nhơn':            'Quy Nhon',
    'Nha Trang':           'Nha Trang',
    'Vũng Tàu':            'Vung Tau',
    'Cà Mau':              'Ca Mau',

    # Tourist nationalities (V10.05)
    'Hàn Quốc':            'South Korea',
    'CHND Trung Hoa':      'China (PRC)',
    'Trung Quốc':          'China',
    'Đài Loan':            'Taiwan',
    'Hoa Kỳ':              'USA',
    'Mỹ':                  'USA',
    'Nhật Bản':            'Japan',
    'Ma-lai-xi-a':         'Malaysia',
    'Ôx-trây-li-a':        'Australia',
    'Australia':           'Australia',
    'Xin-ga-po':           'Singapore',
    'Vương quốc Anh':      'United Kingdom',
    'Anh':                 'United Kingdom',
    'Pháp':                'France',
    'Phi-li-pin':          'Philippines',
    'Đức':                 'Germany',
    'Liên bang Nga':       'Russia',
    'Nga':                 'Russia',
    'In-đô-nê-xi-a':       'Indonesia',
    'Ca-na-đa':            'Canada',
    'Tây Ban Nha':         'Spain',
    'Hà Lan':              'Netherlands',
    'Ý':                   'Italy',
    'Italia':              'Italy',
    'Niu Di-lân':          'New Zealand',
    'Thụy Điển':           'Sweden',
    'Thụy Sĩ':             'Switzerland',
    'Bỉ':                  'Belgium',
    'Áo':                  'Austria',
    'Đan Mạch':            'Denmark',
    'Na Uy':               'Norway',
    'Phần Lan':            'Finland',
    'Ấn Độ':               'India',
    'Hồng Kông':           'Hong Kong',
}


# When the combined "VI / EN" label exceeds this many characters we wrap
# it onto two lines (VI on the first line, EN on the second). Plotly
# renders <br> as a newline in titles, axis titles, legends, and tick
# labels — keeps the chart readable when sub-sector names get long.
_MAX_INLINE = 48


def vi_en(vi: str, en: str, max_inline: int = _MAX_INLINE) -> str:
    '''Compose an explicit bilingual label.

    Returns ``"<vi> / <en>"`` when the joined length is short, otherwise
    breaks onto two lines as ``"<vi><br><en>"``.
    '''
    one_line = f'{vi} / {en}'
    if len(one_line) <= max_inline:
        return one_line
    return f'{vi}<br>{en}'


def tr(label, max_inline: int = _MAX_INLINE) -> str:
    '''Translate a known categorical value to a bilingual form.

    Falls through unchanged if no translation is registered for ``label``;
    wraps to two lines when the joined "VI / EN" form would be too long.
    '''
    if label is None:
        return ''
    s = str(label)
    en = VI_EN_DICT.get(s)
    if not en:
        return s
    return vi_en(s, en, max_inline=max_inline)


def tr_series(values):
    '''Vectorised version of `tr` for pandas Series / lists.'''
    return [tr(v) for v in values]


def _year_int(value):
    '''Pull the 4-digit year out of a string like "2010", "2010.0", "2009-2010", etc.'''
    m = re.search(r'\d{4}', str(value))
    return int(m.group(0)) if m else None


def load_table(table_id: str) -> pd.DataFrame:
    '''Load a parquet by NSO matrix id (works for both single- and doubled-prefix slugs).'''
    matches = list(PXWEB_DIR.glob(f'*{table_id}.px.parquet'))
    if not matches:
        raise FileNotFoundError(f'PX-Web matrix {table_id} not on disk under {PXWEB_DIR}')
    df = pd.read_parquet(matches[0])
    if 'Năm' in df.columns:
        df = df.copy()
        df['year'] = df['Năm'].map(_year_int)
    return df


def load_catalog() -> pd.DataFrame:
    '''One row per PX-Web matrix on disk, both naming layouts handled.'''
    rows = []
    for meta_path in sorted(PXWEB_DIR.glob('*.metadata.json')):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        parquet = meta_path.with_name(meta_path.name.replace('.metadata.json', '.parquet'))
        if not parquet.exists():
            continue
        rows.append({
            'table_id':  meta.get('table_id'),
            'database':  meta.get('database'),
            'title':     meta.get('title'),
            'source':    meta.get('source', 'json_api'),
            'n_cells':   int(meta.get('n_kept') or meta.get('n_cells') or 0),
            'variables': [v.get('code', '') for v in meta.get('variables') or []],
            'parquet':   parquet,
        })
    df = pd.DataFrame(rows)
    df['prefix'] = df['table_id'].str.extract(r'^(V\d+)')
    return df

catalog = load_catalog()
print(f'loaded {len(catalog)} matrices  /  {catalog["n_cells"].sum():,} cells')
print(f'  prefixes: {sorted(catalog["prefix"].dropna().unique())}')
print(f'  databases: {catalog["database"].nunique()} unique  /  by access path: '
       f'{catalog["source"].value_counts().to_dict()}')
"""))


# ---------------------------------------------------------------------------
# §1 catalogue overview
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §1 — Catalogue overview

Before diving into specific topics, what does the catalog *as a whole*
look like? Three angles: tables-per-database, cells-per-table, and the
most-used variable codes.

The colour split of fig 01 (`json_api` vs `html_form`) is the key
insight — two-thirds of the catalog lives behind the legacy ASP.NET
form workflow; if we hadn't built the
[`pxweb_html.py`](packages/scraper/pxweb_html.py) scraper we'd be
missing 315 of 502 tables.
"""))


cells.append(code(r"""
'''Figure 01 — tables per database, coloured by access path.'''
db_summary = (
    catalog.groupby(['database', 'source']).size()
    .reset_index(name='n_tables')
    .sort_values('n_tables', ascending=False)
)

fig01 = px.bar(
    db_summary, x='database', y='n_tables', color='source',
    title=vi_en('Số bảng PX-Web theo từng cơ sở dữ liệu NSO',
                 'PX-Web tables per NSO database (502 total, 12 databases)'),
    labels={'database': vi_en('Cơ sở dữ liệu NSO', 'NSO database'),
             'n_tables': vi_en('Số bảng', 'Tables'),
             'source':   vi_en('Đường truy cập', 'Access path')},
    color_discrete_map={'json_api': NV_GREEN, 'html_form': NV_DARK},
)
fig01.update_layout(barmode='stack', xaxis_tickangle=-25, height=560)
apply_nvidia_style(fig01)
save_figure(fig01, '01_tables_per_database', width=1100, height=560)
fig01.show()
"""))


cells.append(code(r"""
'''Figure 02 — distribution of long-format cells per table (log-x histogram).

We compute the histogram explicitly with ``numpy.logspace`` bins + a
single ``go.Bar`` rather than using ``px.histogram(log_x=True)``, which
in our plotly + kaleido versions bins in *linear* space and then shows
the bars on a log axis — so 30 bins of width ~350 each cluster against
the leftmost edge while the axis stretches uselessly past 10^33. The
explicit-logspace path also lets us set the y-axis label to the
Vietnamese caption (``Số bảng`` instead of plotly's default ``count``)
and the bar marker colour without having to monkey-patch the ``px``
output.
'''
import numpy as np

n_cells = catalog['n_cells'].clip(lower=1)              # avoid log10(0)
log_min, log_max = np.floor(np.log10(n_cells.min())), np.ceil(np.log10(n_cells.max()))
bins = np.logspace(log_min, log_max, 30)
counts, edges = np.histogram(n_cells, bins=bins)
centres = np.sqrt(edges[:-1] * edges[1:])               # geometric mid-points
widths  = np.diff(edges) * 0.92                          # 8% gap between bars

fig02 = go.Figure(go.Bar(
    x=centres, y=counts, width=widths,
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.5)),
    hovertemplate=('[%{customdata[0]:,.0f} – %{customdata[1]:,.0f}] '
                    + vi_en('cells/bảng', 'cells/table') +
                    '<br>%{y} ' + vi_en('bảng', 'tables') + '<extra></extra>'),
    customdata=np.column_stack([edges[:-1], edges[1:]]),
))
fig02.update_layout(
    title=vi_en('Phân phối số ô dữ liệu trên mỗi bảng PX-Web',
                 'Distribution of long-format cells per PX-Web table'),
    xaxis=dict(title=vi_en('Số ô / bảng (log)', 'Cells per table (log scale)'),
                type='log'),
    yaxis=dict(title=vi_en('Số bảng', 'Tables')),
    bargap=0,
)
apply_nvidia_style(fig02)
# Pin the log-x range to the actual data extent and disable the
# zero-line. ``apply_nvidia_style`` defaults ``zeroline=True`` (sensible
# on a linear axis), but on a log axis the "zero line" sits at x=0
# which is -∞ in log space — kaleido responds by auto-extending the
# axis to ~10^297 and squashing every bar against the leftmost edge.
# Setting an explicit ``range`` (in log10 units) and ``zeroline=False``
# keeps the histogram readable.
fig02.update_xaxes(range=[float(log_min), float(log_max)], zeroline=False)
_med = int(catalog['n_cells'].median())
fig02.add_vline(
    x=_med, line_dash='dash', line_color=NV_GREY,
    annotation_text=vi_en(f'trung vị = {_med:,}', f'median = {_med:,}'),
    annotation_position='top right',
)
save_figure(fig02, '02_cells_per_table', width=1000, height=480)
fig02.show()

print(f'  {len(catalog)} matrices  /  {catalog["n_cells"].sum():,} long-format cells')
print(f'  median = {_med:,} cells/table  /  range = {catalog["n_cells"].min():,}'
       f' – {catalog["n_cells"].max():,}')
"""))


cells.append(code(r"""
'''Figure 03 — top-30 variable codes across all 502 tables.'''
counter = Counter()
for variables in catalog['variables']:
    counter.update(variables)
top = pd.DataFrame(counter.most_common(30), columns=['variable', 'tables'])

fig03 = px.bar(
    top, x='variable', y='tables',
    title=vi_en('30 mã biến PX-Web phổ biến nhất',
                 'Top-30 PX-Web variable codes (count of containing tables)'),
    labels={'variable': vi_en('Mã biến (tiếng Việt)', 'Variable code (Vietnamese)'),
             'tables':   vi_en('Số bảng chứa biến', 'Tables containing it')},
)
fig03.update_traces(marker_color=NV_GREEN, marker_line_color=NV_BLACK)
fig03.update_xaxes(tickangle=-35)
apply_nvidia_style(fig03)
save_figure(fig03, '03_variable_frequency', width=1300, height=600)
fig03.show()
"""))


cells.append(code(r"""
'''Figure 04 — time-coverage span per database (when does each DB start / end?).'''
def _years_in(parquet_path: Path) -> tuple[int | None, int | None]:
    '''Scan every column of a parquet for 4-digit years; return (min, max).'''
    df = pd.read_parquet(parquet_path)
    years = []
    for col in df.columns:
        if col == 'value':
            continue
        for val in df[col].astype(str).head(2000):
            m = re.search(r'(19|20)\d{2}', val)
            if m:
                years.append(int(m.group(0)))
    if not years:
        return None, None
    return min(years), max(years)

rows = []
for _, r in catalog.iterrows():
    y0, y1 = _years_in(r['parquet'])
    rows.append({'database': r['database'], 'table': r['table_id'],
                  'y0': y0, 'y1': y1})
spans = pd.DataFrame(rows).dropna()

fig04 = go.Figure()
db_order = (spans.groupby('database')['y0'].min()
              .sort_values().index.tolist())
for i, db in enumerate(db_order):
    sub = spans[spans['database'] == db]
    for _, r in sub.iterrows():
        fig04.add_trace(go.Scatter(
            x=[r['y0'], r['y1']], y=[i, i],
            mode='lines',
            line=dict(color=NV_GREEN, width=4),
            opacity=0.55, showlegend=False,
            hovertemplate=f"<b>{r['table']}</b><br>{r['y0']}–{r['y1']}<extra></extra>",
        ))
fig04.update_layout(
    title=vi_en('Khoảng thời gian dữ liệu của từng bảng PX-Web theo cơ sở dữ liệu',
                 'Time-coverage span of every PX-Web table, grouped by database'),
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis=dict(tickmode='array',
                tickvals=list(range(len(db_order))),
                ticktext=[tr(db) for db in db_order]),
    height=600,
)
apply_nvidia_style(fig04)
save_figure(fig04, '04_time_coverage', width=1200, height=600)
fig04.show()

print(f'  earliest year on record: {spans["y0"].min()}')
print(f'  latest   year on record: {spans["y1"].max()}')
"""))


# ---------------------------------------------------------------------------
# §2 V01 geography & climate
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §2 — V01: Geography, Land & Climate (18 tables / 4,705 cells)

Maps onto the `geography` statistical domain and the `region` /
`urbanicity` persona dimensions. V01 is small (3 % of the cell volume)
but anchors every spatial join downstream — every other database joins
its province codes against V01.01.
"""))


cells.append(code(r"""
'''Figure 05 — land-use composition (V01.04, % of provincial area, latest snapshot).'''
df = load_table('V01.04')
prov_col = 'Phân theo địa phương, Năm'
use_col  = 'Đất sử dụng'
national = df[df[prov_col] == 'CẢ NƯỚC'].copy()
# The "Năm" lives inside `Đất sử dụng` for this matrix. Keep all
# non-snapshot rows (= categorical land-use class) for a single date.
latest_date = sorted(set(national[use_col]),
                      key=lambda s: re.findall(r'\d{4}', s) or [''])[-1]
snap = (df[df[use_col] != latest_date].groupby(use_col)['value'].sum()
          .sort_values(ascending=True))
fig05 = go.Figure()
fig05.add_trace(go.Bar(
    y=snap.index, x=snap.values, orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.5)),
    text=[f'{v:,.0f}' for v in snap.values], textposition='outside',
))
fig05.update_layout(
    title=vi_en('Cơ cấu sử dụng đất theo loại',
                 'Land-use composition by category') +
           ' (NSO V01.04)',
    xaxis_title=vi_en('Tổng diện tích (đơn vị tương đối)',
                        'Total area share (relative units)'),
    height=520,
)
apply_nvidia_style(fig05)
save_figure(fig05, '05_land_use_composition', width=1100, height=520)
fig05.show()
"""))


cells.append(code(r"""
'''Figure 06 — average annual temperature trend, top 8 stations (V01.13).'''
df = load_table('V01.13')
prov_col = 'Tỉnh, thành phố'
df = df.dropna(subset=['year', 'value'])
top_stations = df['Tỉnh, thành phố'].value_counts().head(8).index.tolist()
df = df[df[prov_col].isin(top_stations)]

fig06 = go.Figure()
palette = [NV_GREEN, NV_BLACK, NV_GREEN_DARK, NV_GREY,
            '#888888', NV_GREEN_SOFT, NV_LIGHT_GREY, '#444444']
for station, color in zip(top_stations, palette):
    sub = df[df[prov_col] == station].sort_values('year')
    fig06.add_trace(go.Scatter(
        x=sub['year'], y=sub['value'], mode='lines+markers',
        name=tr(station),
        line=dict(color=color, width=2.4), marker=dict(size=5),
    ))
fig06.update_layout(
    title=vi_en('Nhiệt độ trung bình năm tại 8 trạm quan trắc lớn',
                 'Average annual air temperature at top-8 climate stations') +
           ' (NSO V01.13, °C)',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Nhiệt độ trung bình (°C)', 'Mean temperature (°C)'),
    height=520,
)
apply_nvidia_style(fig06)
save_figure(fig06, '06_temperature_trend', width=1200, height=520)
fig06.show()
"""))


cells.append(code(r"""
'''Figure 07 — total annual rainfall, latest year vs earliest (V01.08).'''
df = load_table('V01.08').dropna(subset=['year', 'value'])
prov_col = 'Tỉnh, thành phố'
y0, y1 = df['year'].min(), df['year'].max()
e = df[df['year'] == y0].set_index(prov_col)['value']
l = df[df['year'] == y1].set_index(prov_col)['value']
common = sorted(set(e.index) & set(l.index))
e, l = e.reindex(common), l.reindex(common)
delta = (l - e).sort_values()

fig07 = go.Figure()
colors = [NV_GREEN if d >= 0 else NV_BLACK for d in delta.values]
fig07.add_trace(go.Bar(
    y=delta.index, x=delta.values, orientation='h',
    marker=dict(color=colors, line=dict(color=NV_BLACK, width=0.4)),
))
fig07.update_layout(
    title=vi_en('Biến động lượng mưa năm',
                 'Change in total annual rainfall') +
           f' {int(y0)} → {int(y1)} (NSO V01.08, mm)',
    xaxis_title=vi_en('Δ lượng mưa (mm)', 'Δ rainfall (mm)'),
    height=620,
)
apply_nvidia_style(fig07)
save_figure(fig07, '07_rainfall_change', width=1100, height=620)
fig07.show()
"""))


# ---------------------------------------------------------------------------
# §3 V02 population & labour
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §3 — V02: Population & Labour (63 tables / 48,203 cells)

The single most important database for the persona generator: nearly
every persona dimension (`region`, `urbanicity`, `age_group`, `sex`,
`education_level`, `employment_status`, `occupation`, `industry_sector`)
draws its CPD from a V02 matrix. 7 figures, walking through:
demographics → labour force → unemployment.
"""))


cells.append(code(r"""
'''Figure 08 — Vietnam population 1990–2024, total / urban / rural (V02.02).'''
df = load_table('V02.02')
df = df[df['Cách tính'] == 'Tổng số (Nghìn người)'].dropna(subset=['year'])
piv = df.pivot_table(index='year', columns='Phân tổ', values='value', aggfunc='first')
piv = piv[['Tổng số', 'Thành thị', 'Nông thôn']] / 1000

fig08 = go.Figure()
fig08.add_trace(go.Scatter(x=piv.index, y=piv['Tổng số'], mode='lines',
                            name=tr('Tổng số'),
                            line=dict(color=NV_GREEN, width=3.5)))
fig08.add_trace(go.Scatter(x=piv.index, y=piv['Nông thôn'], mode='lines',
                            name=tr('Nông thôn'),
                            line=dict(color=NV_BLACK, width=2, dash='dot')))
fig08.add_trace(go.Scatter(x=piv.index, y=piv['Thành thị'], mode='lines',
                            name=tr('Thành thị'),
                            line=dict(color=NV_GREEN_DARK, width=2.5, dash='dash')))
fig08.update_layout(
    title=vi_en('Dân số Việt Nam 1990–2024',
                 'Vietnam population 1990–2024') + ' (NSO V02.02)',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Dân số (triệu người)', 'Population (millions)'),
    height=520,
)
apply_nvidia_style(fig08)
save_figure(fig08, '08_population_timeseries', width=1100, height=520)
fig08.show()

print(f"  Total: {piv['Tổng số'].iloc[0]:.1f}M ({piv.index.min()}) → "
       f"{piv['Tổng số'].iloc[-1]:.1f}M ({piv.index.max()}) "
       f"= +{(piv['Tổng số'].iloc[-1]/piv['Tổng số'].iloc[0]-1)*100:.1f}%")
"""))


cells.append(code(r"""
'''Figure 09 — urban share of population, 1990–2024 (V02.02).'''
df = load_table('V02.02')
df = df[df['Cách tính'] == 'Tổng số (Nghìn người)'].dropna(subset=['year'])
piv = df.pivot_table(index='year', columns='Phân tổ', values='value', aggfunc='first')
piv['urban_pct'] = piv['Thành thị'] / piv['Tổng số'] * 100

fig09 = go.Figure()
fig09.add_trace(go.Scatter(
    x=piv.index, y=piv['urban_pct'], mode='lines',
    name=vi_en('Tỷ lệ thành thị %', 'Urban share %'),
    fill='tozeroy', line=dict(color=NV_GREEN, width=3),
    fillcolor='rgba(118,185,0,0.15)',
))
fig09.update_layout(
    title=vi_en('Tỷ lệ dân số thành thị, 1990–2024',
                 'Urban share of population, 1990–2024') +
           ' (NSO V02.02)',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('% thành thị', '% urban'),
    yaxis=dict(range=[0, 50]), height=460,
)
for x in (piv.index[0], piv.index[-1]):
    y = piv.loc[x, 'urban_pct']
    fig09.add_annotation(x=x, y=y, text=f'<b>{x}</b>: {y:.1f}%',
                          showarrow=False, yshift=14,
                          font=dict(family=NV_FONT_FAMILY, color=NV_BLACK))
apply_nvidia_style(fig09)
save_figure(fig09, '09_urban_share_evolution', width=1000, height=460)
fig09.show()
"""))


cells.append(code(r"""
'''Figure 10 — life expectancy, latest year by region × residence (V02.24).'''
df = load_table('V02.24').dropna(subset=['year'])
latest = df['year'].max()
sub = df[df['year'] == latest]
# Match the right columns dynamically — schema varies
cols = list(sub.columns)
piv = sub.pivot_table(values='value',
                       index=[c for c in cols if c not in ('year', 'Năm', 'value')],
                       aggfunc='first')

fig10 = go.Figure()
flat = piv.reset_index()
top10 = flat.sort_values('value', ascending=True).tail(20)
def _strata_label(row):
    parts = [tr(v) for v in row[:-1]]
    return ' / '.join(str(p) for p in parts)
fig10.add_trace(go.Bar(
    y=[_strata_label(row) for row in top10.values],
    x=top10['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    text=[f'{v:.1f}' for v in top10['value']], textposition='outside',
))
fig10.update_layout(
    title=vi_en('Tuổi thọ trung bình tính từ lúc sinh',
                 'Life expectancy at birth') +
           f' — {int(latest)} (NSO V02.24, ' +
           vi_en('20 nhóm cao nhất', 'top 20 strata') + ')',
    xaxis_title=vi_en('Số năm', 'Years'),
    height=620,
)
apply_nvidia_style(fig10)
save_figure(fig10, '10_life_expectancy', width=1400, height=620)
fig10.show()
"""))


cells.append(code(r"""
'''Figure 11 — employed cohort by 5-year age band, earliest vs latest year (V02.41).'''
df = load_table('V02.41')
df = df[~df['Nhóm tuổi'].str.upper().eq('TỔNG SỐ')].dropna(subset=['year'])
# NSO V02.41 collapses everyone aged 50+ into a single open-ended top
# bucket (``'50+'``), not five-year cohorts up to ``'65+'``. Eight bars
# in total: seven 5-year cohorts from 15-19 to 45-49 plus the ``50+``
# top bucket. Hardcoding ``50-54``/``55-59``/``60-64``/``65+`` filters
# them all out as not-in-data and the age-pyramid loses its right tail.
age_order = ['15-19','20-24','25-29','30-34','35-39','40-44',
              '45-49','50+']
df = df[df['Nhóm tuổi'].isin(age_order)]
y0, y1 = df['year'].min(), df['year'].max()
e = df[df['year']==y0].set_index('Nhóm tuổi')['value'].reindex(age_order)
l = df[df['year']==y1].set_index('Nhóm tuổi')['value'].reindex(age_order)

fig11 = go.Figure()
fig11.add_trace(go.Bar(x=age_order, y=e.values, name=f'{int(y0)}',
                        marker=dict(color=NV_LIGHT_GREY, line=dict(color=NV_BLACK, width=1))))
fig11.add_trace(go.Bar(x=age_order, y=l.values, name=f'{int(y1)}',
                        marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=1))))
fig11.update_layout(
    title=vi_en('Lao động có việc làm theo nhóm tuổi 5 năm',
                 'Employed cohort by 5-year age band') +
           f' — {int(y0)} vs {int(y1)} (NSO V02.41)',
    xaxis_title=vi_en('Nhóm tuổi', 'Age group'),
    yaxis_title=vi_en('Lao động có việc làm (nghìn người)',
                        'Employed (thousand persons)'),
    barmode='group', height=540,
)
apply_nvidia_style(fig11)
save_figure(fig11, '11_age_pyramid_shift', width=1100, height=540)
fig11.show()
"""))


cells.append(code(r"""
'''Figure 12 — trained-labour share by qualification, stacked over time (V02.54).

NSO V02.54 exports the four qualification levels (Sơ cấp / Trung cấp /
Cao đẳng / Đại học trở lên) plus a ``TỔNG SỐ`` row per year. For
**2010–2014** the PX-Web export *omits* the ``Trung cấp`` cell — only
the other three categories and the total are published — so a naive
filter of ``TỔNG SỐ`` + ``fillna(0)`` produces visually-stunted bars
~5pp shorter than 2009 and 2015+, looking like missing data.

We back-fill the gap analytically: for years where ``Trung cấp`` is
missing but ``TỔNG SỐ`` is reported, infer ``Trung cấp = TỔNG SỐ -
(Sơ cấp + Cao đẳng + Đại học trở lên)``. The result for 2010–2014 is
~5.2–5.6pp, perfectly continuous with the 5.4pp reported in 2015 — the
missing values almost certainly were measured but not split out by
NSO's PX-Web export.
'''
df = load_table('V02.54').dropna(subset=['year'])
piv_full = df.pivot_table(index='year', columns='Chuyên môn kỹ thuật',
                            values='value', aggfunc='first').sort_index()

order   = ['Sơ cấp', 'Trung cấp', 'Cao đẳng', 'Đại học trở lên']
palette = [NV_FAINT, NV_LIGHT_GREY, NV_GREEN_SOFT, NV_GREEN]

# Back-fill missing Trung cấp from the published TỔNG SỐ. ``min_count``
# guards against propagating-NaN: only impute when ALL three other
# categories are present, otherwise leave the cell as NaN.
if {'Trung cấp', 'TỔNG SỐ'}.issubset(piv_full.columns):
    others = ['Sơ cấp', 'Cao đẳng', 'Đại học trở lên']
    others_sum = piv_full[others].sum(axis=1, min_count=len(others))
    inferred = piv_full['TỔNG SỐ'] - others_sum
    piv_full['Trung cấp'] = piv_full['Trung cấp'].fillna(inferred)
piv = piv_full.drop(columns=['TỔNG SỐ'], errors='ignore')

fig12 = go.Figure()
for col, color in zip(order, palette):
    if col in piv.columns:
        fig12.add_trace(go.Bar(
            x=piv.index, y=piv[col].fillna(0), name=tr(col),
            marker=dict(color=color, line=dict(color=NV_BLACK, width=0.5)),
        ))
fig12.update_layout(
    title=vi_en('Tỷ lệ lao động đã qua đào tạo theo trình độ',
                 'Trained-labour share by qualification level') +
           ' (NSO V02.54, % ' +
           vi_en('lực lượng lao động', 'of labour force') + ')',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('% lực lượng lao động', '% of labour force'),
    barmode='stack', height=520,
)
apply_nvidia_style(fig12)
save_figure(fig12, '12_education_attainment_trend', width=1100, height=520)
fig12.show()

print(f'  Trained share: {piv.iloc[0].sum():.1f}% ({piv.index.min()}) → '
       f'{piv.iloc[-1].sum():.1f}% ({piv.index.max()})')
"""))


cells.append(code(r"""
'''Figure 13 — occupation structure shift, 2009 vs latest (V02.43).'''
df = load_table('V02.43')
df = df[~df['Nghề nghiệp'].str.upper().eq('TỔNG SỐ')].dropna(subset=['year'])
y0, y1 = df['year'].min(), df['year'].max()
e = df[df['year']==y0].set_index('Nghề nghiệp')['value']
l = df[df['year']==y1].set_index('Nghề nghiệp')['value']
all_occs = sorted(set(e.index) | set(l.index))
e = e.reindex(all_occs).fillna(0) / max(e.sum(), 1) * 100
l = l.reindex(all_occs).fillna(0) / max(l.sum(), 1) * 100

y_labels = [tr(o) for o in all_occs]
fig13 = go.Figure()
fig13.add_trace(go.Bar(y=y_labels, x=e.values, name=str(int(y0)), orientation='h',
                        marker=dict(color=NV_LIGHT_GREY, line=dict(color=NV_BLACK, width=1))))
fig13.add_trace(go.Bar(y=y_labels, x=l.values, name=str(int(y1)), orientation='h',
                        marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=1))))
fig13.update_layout(
    title=vi_en('Chuyển dịch cơ cấu nghề nghiệp',
                 'Occupation structure shift') +
           f' {int(y0)}–{int(y1)} (NSO V02.43, % ' +
           vi_en('lao động có việc làm', 'of employed') + ')',
    xaxis_title=vi_en('Tỷ lệ lao động có việc làm (%)',
                        'Share of employed (%)'),
    barmode='group', height=620,
)
apply_nvidia_style(fig13)
save_figure(fig13, '13_occupation_structure_shift', width=1400, height=620)
fig13.show()
"""))


cells.append(code(r"""
'''Figure 14 — labour productivity by economic sector (V02.57).'''
df = load_table('V02.57').dropna(subset=['year'])
sect_col = 'Ngành kinh tế' if 'Ngành kinh tế' in df.columns else 'Ngành'
df = df[~df[sect_col].str.upper().eq('TỔNG SỐ')]
latest = df['year'].max()
sub = df[df['year'] == latest].sort_values('value', ascending=True).tail(15)

fig14 = go.Figure()
fig14.add_trace(go.Bar(
    y=[tr(s) for s in sub[sect_col]], x=sub['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    text=[f'{v:,.0f}' for v in sub['value']], textposition='outside',
))
fig14.update_layout(
    title=vi_en('Năng suất lao động theo ngành kinh tế',
                 'Labour productivity by economic sector') +
           f' — {int(latest)} (NSO V02.57)',
    xaxis_title=vi_en('Năng suất (triệu VND / người / năm)',
                        'Productivity (million VND / worker / year)'),
    height=600,
)
apply_nvidia_style(fig14)
save_figure(fig14, '14_labour_productivity_by_sector', width=1400, height=600)
fig14.show()
"""))


cells.append(code(r"""
'''Figure 15 — unemployment rate by macro-region × urban/rural, latest year (V02.59).

V02.59's ``Vùng`` column emits ``'Bắc Trung Bộ và duyên hải miền Trung'``
with **lowercase** ``d``, while every other NSO table (and our canonical
form) uses **upper-case** ``D``. Without normalisation that one row is
silently dropped by the ``.isin(MACRO_REGIONS)`` filter and the central
region disappears from the chart.
'''
MACRO_REGIONS = (
    'Đồng bằng sông Hồng',
    'Trung du và miền núi phía Bắc',
    'Bắc Trung Bộ và Duyên hải miền Trung',
    'Tây Nguyên',
    'Đông Nam Bộ',
    'Đồng bằng sông Cửu Long',
)
df = load_table('V02.59').dropna(subset=['year'])
df['Vùng'] = df['Vùng'].astype(str).str.replace(
    'duyên hải miền Trung', 'Duyên hải miền Trung', regex=False)
df = df[df['Vùng'].isin(MACRO_REGIONS)]
df = df[df['Thành thị, nông thôn'].isin(('Thành thị', 'Nông thôn'))]
latest = df['year'].max()
df = df[df['year'] == latest]
piv = df.pivot(index='Vùng', columns='Thành thị, nông thôn', values='value').reindex(MACRO_REGIONS)

fig15 = go.Figure()
fig15.add_trace(go.Bar(x=[tr(r) for r in piv.index], y=piv['Thành thị'],
                        name=tr('Thành thị'),
                        marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=1))))
fig15.add_trace(go.Bar(x=[tr(r) for r in piv.index], y=piv['Nông thôn'],
                        name=tr('Nông thôn'),
                        marker=dict(color=NV_LIGHT_GREY, line=dict(color=NV_BLACK, width=1))))
fig15.update_layout(
    title=vi_en('Tỷ lệ thất nghiệp theo vùng × thành thị / nông thôn',
                 'Unemployment rate by macro-region × urban / rural') +
           f' — {int(latest)} (NSO V02.59, %)',
    xaxis_title=vi_en('Vùng kinh tế', 'Macro-region'),
    yaxis_title=vi_en('Tỷ lệ thất nghiệp (%)', 'Unemployment rate (%)'),
    barmode='group', xaxis_tickangle=-15, height=540,
)
apply_nvidia_style(fig15)
save_figure(fig15, '15_unemployment_region_urban', width=1100, height=540)
fig15.show()
"""))


cells.append(code(r"""
'''Figure 16 — unemployment rate by qualification level over time (V02.62).'''
df = load_table('V02.62').dropna(subset=['year'])
df = df[~df['Trình độ chuyên môn kỹ thuật'].str.upper().eq('TỔNG SỐ')]
piv = df.pivot_table(index='year', columns='Trình độ chuyên môn kỹ thuật',
                       values='value', aggfunc='first').sort_index()
order = ['Đại học trở lên'] + [c for c in piv.columns if c != 'Đại học trở lên']
palette = [NV_GREEN, NV_BLACK, NV_GREEN_DARK, NV_GREY, '#888888', NV_LIGHT_GREY]
fig16 = go.Figure()
for col, color in zip(order, palette):
    if col not in piv.columns: continue
    fig16.add_trace(go.Scatter(
        x=piv.index, y=piv[col], mode='lines+markers', name=tr(col),
        line=dict(color=color, width=2.6), marker=dict(size=6),
    ))
fig16.update_layout(
    title=vi_en('Tỷ lệ thất nghiệp theo trình độ chuyên môn',
                 'Unemployment rate by qualification level') +
           ' (NSO V02.62, %)',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Tỷ lệ thất nghiệp (%)', 'Unemployment rate (%)'),
    height=520,
)
apply_nvidia_style(fig16)
save_figure(fig16, '16_unemployment_by_education', width=1100, height=520)
fig16.show()

print(f"\nLatest year ({piv.index.max()}): "
       f"university+ unemployment = {piv.loc[piv.index.max(), 'Đại học trở lên']:.2f}%")
"""))


# ---------------------------------------------------------------------------
# §4 V03 national accounts & banking
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §4 — V03: National Accounts & Banking (24 tables / 4,443 cells)

V03 covers GDP, state budget, banking, and insurance — i.e. every macro
indicator a persona's `income_quintile` could be conditioned on. Today
the persona generator uses a synthetic income prior; if NSO ever
publishes the household-survey × quintile joint, V03 is where to look.
"""))


cells.append(code(r"""
'''Figure 17 — Vietnam GDP at current prices (V03.01).

NSO V03.01's PX-Web export covers 2000–2002 then **jumps to 2009–2024**,
skipping 2003–2008 entirely. With Plotly's default numeric x-axis the
six missing years would render as a wide blank strip between the two
clusters of bars — that's the visual "lag" you'd otherwise see at the
left edge of the chart. Two adjustments:

* Filter on the explicit GDP-current-prices indicator (``Tổng sản phẩm
  trong nước theo giá hiện hành  - Tỷ đồng``) instead of taking
  ``groupby('year').max()`` over every "Tỷ đồng" indicator — the latter
  silently mixes GDP, GNI, total consumption, etc., depending on which
  happens to be largest each year.
* Use ``xaxis.type='category'`` so the x-axis shows only the 19 years
  that actually have data, side-by-side, with no empty gap. Years stay
  in chronological order via the ``categoryorder`` setting.
'''
df = load_table('V03.01').dropna(subset=['year'])
gdp_indicator = next(
    (c for c in df['Chỉ tiêu'].dropna().unique()
     if 'Tổng sản phẩm trong nước' in c and 'Tỷ đồng' in c),
    None,
)
sub = df[df['Chỉ tiêu'] == gdp_indicator] if gdp_indicator else \
      df[df['Chỉ tiêu'].str.contains('Tỷ đồng', na=False)]
total = sub.groupby('year', as_index=False)['value'].max().sort_values('year')
total['trillion_vnd'] = total['value'] / 1000
total['year_label'] = total['year'].astype(int).astype(str)

fig17 = go.Figure()
fig17.add_trace(go.Bar(
    x=total['year_label'], y=total['trillion_vnd'],
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.5)),
    text=[f'{v:,.0f}' for v in total['trillion_vnd']], textposition='outside',
))
fig17.update_layout(
    title=vi_en('Tổng sản phẩm trong nước (GDP) theo giá hiện hành',
                 'Vietnam GDP at current prices') +
           ' (NSO V03.01, ' + vi_en('nghìn tỷ VND', 'trillion VND') + ')',
    xaxis=dict(title=vi_en('Năm', 'Year'),
                type='category',
                categoryorder='array',
                categoryarray=list(total['year_label'])),
    yaxis_title=vi_en('GDP (nghìn tỷ VND)', 'GDP (trillion VND)'),
    height=520,
)
apply_nvidia_style(fig17)
save_figure(fig17, '17_gdp_current_prices', width=1100, height=520)
fig17.show()
"""))


cells.append(code(r"""
'''Figure 18 — state budget revenue by category (V03.13).'''
df = load_table('V03.13').dropna(subset=['year'])
sub = df[df['Chỉ tiêu, Loại thu'] == 'Giá trị (Tỷ đồng)']
sub = sub.groupby('year', as_index=False)['value'].sum().sort_values('year')
sub['trillion'] = sub['value'] / 1000

fig18 = go.Figure()
fig18.add_trace(go.Scatter(
    x=sub['year'], y=sub['trillion'], mode='lines+markers',
    name=vi_en('Tổng thu', 'Total revenue'),
    line=dict(color=NV_GREEN, width=3.5), marker=dict(size=8, color=NV_GREEN_DARK),
    fill='tozeroy', fillcolor='rgba(118,185,0,0.18)',
))
fig18.update_layout(
    title=vi_en('Tổng thu ngân sách Nhà nước',
                 'State budget revenue, total') +
           ' (NSO V03.13, ' + vi_en('nghìn tỷ VND', 'trillion VND') + ')',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Tổng thu (nghìn tỷ VND)', 'Revenue (trillion VND)'),
    height=480,
)
apply_nvidia_style(fig18)
save_figure(fig18, '18_state_budget_revenue', width=1100, height=480)
fig18.show()
"""))


cells.append(code(r"""
'''Figure 19 — social-insurance penetration by province, latest year (V03.22).

V03.22 has two latent quirks the previous version of this cell ignored:

1. **Latin-Eth ``Ð`` (U+00D0)** instead of Vietnamese ``Đ`` (U+0110)
   in 9 province names (``Ðồng Nai``, ``Ðà Nẵng``, ``Bình Ðịnh``, ...
   the same set we already handle for the visualisation notebook).
   The labels render with a visually-wrong codepoint and break exact-
   string matching against any canonical filter list.
2. **Aggregate rows mixed in with provinces** — ``Cả nước`` (national
   total), the 6 macro-regions, all sit in the same ``Tỉnh, thành phố``
   column and routinely land inside the top-20 (``Ðông Nam Bộ`` at 54%
   was ranking #6, ahead of most actual provinces). The chart's title
   says "by province" but it was silently mixing aggregation levels.

The fix transliterates ``Ð → Đ`` first, then drops the known non-
province rows before sorting and slicing the top-20.
'''
df = load_table('V03.22').dropna(subset=['year']).copy()
df['Tỉnh, thành phố'] = df['Tỉnh, thành phố'].astype(str).str.translate(
    str.maketrans({'\u00D0': '\u0110', '\u00F0': '\u0111'}))
NON_PROVINCE = {
    'Cả nước', 'CẢ NƯỚC', 'Tổng số', 'TỔNG SỐ',
    'Đồng bằng sông Hồng', 'Trung du và miền núi phía Bắc',
    'Bắc Trung Bộ và Duyên hải miền Trung', 'Tây Nguyên',
    'Đông Nam Bộ', 'Đồng bằng sông Cửu Long',
}
df = df[~df['Tỉnh, thành phố'].isin(NON_PROVINCE)]
latest = df['year'].max()
sub = (df[df['year'] == latest]
        .sort_values('value', ascending=True).tail(20))
fig19 = go.Figure()
fig19.add_trace(go.Bar(
    y=[tr(p) for p in sub['Tỉnh, thành phố']], x=sub['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    text=[f'{v:.1f}%' for v in sub['value']], textposition='outside',
))
fig19.update_layout(
    title=vi_en('Tỷ lệ tham gia bảo hiểm xã hội theo tỉnh',
                 'Social-insurance participation rate by province') +
           f' — {int(latest)} (NSO V03.22, ' +
           vi_en('20 tỉnh dẫn đầu', 'top-20 provinces') + ', %)',
    xaxis_title=vi_en('Tỷ lệ tham gia (%)', 'Participation rate (%)'),
    height=620,
)
apply_nvidia_style(fig19)
save_figure(fig19, '19_social_insurance_by_province', width=1400, height=620)
fig19.show()
"""))


# ---------------------------------------------------------------------------
# §5 V04 investment
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §5 — V04: Investment (25 tables / 5,522 cells)

Maps onto the `investment` domain. Investment intensity has a strong
shape-effect on regional `urbanicity` distributions over time, and is
where the persona generator picks up macro-economic context for richer
bios.
"""))


cells.append(code(r"""
'''Figure 20 — total investment by economic sector type (V04.01).

Two NSO V04.01 quirks the previous version of this cell tripped over —
the chart silently rendered as 0 traces (empty figure):

1. The PX-Web export uses **double spaces** inside three of the four
   ``Thành phần kinh tế`` values (``'Kinh tế  Nhà nước'``,
   ``'Kinh tế ngoài  nhà nước'``, ``'Khu vực có vốn  đầu tư nước
   ngoài'``) and a stray lower-case ``n`` in ``ngoài  nhà nước``. The
   stable ``order`` list of canonical labels with single spaces never
   matches any column in the pivot, so every ``if col in piv.columns``
   guard is False and no bars get added.
2. The ``Cách tính`` column has FOUR units, two of which contain
   ``Tỷ đồng`` — both the current-prices series (``Giá thực tế (Tỷ
   đồng)``) and the constant-prices series (``Giá so sánh 2010 (Tỷ
   đồng)``). Filtering on substring ``'Tỷ đồng'`` mixes the two,
   which would (silently) double-count any sector that has values in
   both. We restrict to *current prices* — the standard reporting
   metric for this NSO indicator.

Fix: normalise the sector strings (``\\s+`` → single space + casefold
for matching), pin to the current-prices unit, and drop the ``Tổng
số`` aggregate before pivoting.
'''
df = load_table('V04.01').dropna(subset=['year'])
df = df[df['Cách tính'] == 'Giá thực tế (Tỷ đồng)']
# Collapse PX-Web double-spaces inside the sector label so we can match
# against the canonical ``order`` list. Strip and de-duplicate spaces;
# preserve diacritics + case for the displayed legend.
df = df.assign(sector=df['Thành phần kinh tế'].str.replace(
    r'\s+', ' ', regex=True).str.strip())
df = df[df['sector'].str.casefold() != 'tổng số']
piv = df.pivot_table(index='year', columns='sector',
                       values='value', aggfunc='first').sort_index()
piv = piv.dropna(how='all').fillna(0) / 1000          # → trillion VND

# The sector labels in V04.01 are inconsistently cased (``Kinh tế Nhà
# nước`` vs ``Kinh tế ngoài nhà nước``). Match the canonical names
# below against the pivot columns case-insensitively to pick up either.
fig20 = go.Figure()
order = ['Kinh tế Nhà nước', 'Kinh tế ngoài Nhà nước', 'Khu vực có vốn đầu tư nước ngoài']
palette = [NV_BLACK, NV_GREEN, NV_GREEN_DARK]
casefold_lookup = {c.casefold(): c for c in piv.columns}
for col, color in zip(order, palette):
    actual = casefold_lookup.get(col.casefold())
    if actual is None:
        continue
    fig20.add_trace(go.Bar(
        x=piv.index, y=piv[actual], name=tr(col),
        marker=dict(color=color, line=dict(color=NV_BLACK, width=0.5)),
    ))
fig20.update_layout(
    title=vi_en('Vốn đầu tư xã hội theo thành phần kinh tế',
                 'Total social investment by sector type') +
           ' (NSO V04.01, ' + vi_en('nghìn tỷ VND', 'trillion VND') + ')',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Vốn đầu tư (nghìn tỷ VND)', 'Investment (trillion VND)'),
    barmode='stack', height=520,
)
apply_nvidia_style(fig20)
save_figure(fig20, '20_investment_by_sector_type', width=1100, height=520)
fig20.show()

print(f'  total social investment {int(piv.index.min())}: '
       f'{piv.loc[piv.index.min()].sum():,.0f} nghìn tỷ VND  →  '
       f'{int(piv.index.max())}: {piv.loc[piv.index.max()].sum():,.0f} nghìn tỷ VND')
"""))


cells.append(code(r"""
'''Figure 21 — investment by industry, latest year (V04.04).'''
df = load_table('V04.04').dropna(subset=['year'])
df = df[~df['Ngành kinh tế'].str.upper().eq('TỔNG SỐ')]
latest = df['year'].max()
sub = (df[df['year'] == latest]
        .sort_values('value', ascending=True).tail(15))

fig21 = go.Figure()
fig21.add_trace(go.Bar(
    y=[tr(s) for s in sub['Ngành kinh tế']], x=sub['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    text=[f'{v:,.0f}' for v in sub['value']], textposition='outside',
))
fig21.update_layout(
    title=vi_en('Vốn đầu tư theo ngành kinh tế',
                 'Total investment by industry') +
           f' — {int(latest)} (NSO V04.04, ' +
           vi_en('15 ngành dẫn đầu', 'top-15') + ', ' +
           vi_en('tỷ VND', 'billion VND') + ')',
    xaxis_title=vi_en('Vốn đầu tư (tỷ VND)', 'Investment (billion VND)'),
    height=600,
)
apply_nvidia_style(fig21)
save_figure(fig21, '21_investment_by_industry', width=1400, height=600)
fig21.show()
"""))


# ---------------------------------------------------------------------------
# §6 V05 enterprises
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §6 — V05: Enterprises (56 tables / 37,391 cells)

V05 is the second-largest cell pool after V14. Its province × industry
× year matrices are how the persona generator's `industry_sector` CPD
gets ground-truth shape — workers cluster where enterprises cluster.
"""))


cells.append(code(r"""
'''Figure 22 — enterprises in Vietnam, total time-series (V05.04).'''
df = load_table('V05.04').dropna(subset=['year'])
nat = df[df['Tỉnh/thành phố'].str.upper().eq('CẢ NƯỚC')].copy()
nat = nat.sort_values('year')

fig22 = go.Figure()
fig22.add_trace(go.Scatter(
    x=nat['year'], y=nat['value']/1000, mode='lines+markers',
    name=vi_en('Số doanh nghiệp (nghìn)',
                'Number of enterprises (thousands)'),
    line=dict(color=NV_GREEN, width=3.5), marker=dict(size=8, color=NV_GREEN_DARK),
    fill='tozeroy', fillcolor='rgba(118,185,0,0.18)',
))
fig22.update_layout(
    title=vi_en('Số doanh nghiệp đang hoạt động — cả nước',
                 'Active enterprises in Vietnam — national total') +
           ' (NSO V05.04, ' + vi_en('nghìn doanh nghiệp', 'thousands') + ')',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Doanh nghiệp (×1.000)', 'Enterprises (×1,000)'),
    height=480,
)
apply_nvidia_style(fig22)
save_figure(fig22, '22_enterprises_national', width=1100, height=480)
fig22.show()
"""))


cells.append(code(r"""
'''Figure 23 — enterprises by industry, latest year (V05.01).'''
df = load_table('V05.01').dropna(subset=['year'])
df = df[~df['Ngành'].str.upper().eq('TỔNG SỐ')]
latest = df['year'].max()
sub = (df[df['year'] == latest]
        .sort_values('value', ascending=True).tail(15))

fig23 = go.Figure()
fig23.add_trace(go.Bar(
    y=[tr(s) for s in sub['Ngành']], x=sub['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    text=[f'{v:,.0f}' for v in sub['value']], textposition='outside',
))
fig23.update_layout(
    title=vi_en('Số doanh nghiệp theo ngành',
                 'Enterprises by industry') +
           f' — {int(latest)} (NSO V05.01, ' +
           vi_en('15 ngành dẫn đầu', 'top-15') + ')',
    xaxis_title=vi_en('Số doanh nghiệp', 'Number of enterprises'),
    height=620,
)
apply_nvidia_style(fig23)
save_figure(fig23, '23_enterprises_by_industry', width=1400, height=620)
fig23.show()
"""))


# ---------------------------------------------------------------------------
# §7 V06 agriculture
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §7 — V06: Agriculture, Forestry & Fishery (70 tables / 76,541 cells)

V06 is the **largest cell volume in the catalog** — 24 % of all 316,108
cells. Most of those are fine-grained province × year × crop matrices.
Vietnam's rural labour force is the largest single occupation group in
the country; V06's province-level rice / livestock / aquaculture
output drives the persona generator's regional `industry_sector`
distribution for the agricultural class.
"""))


cells.append(code(r"""
'''Figure 24 — rice production by macro-region, latest year (V06.13).'''
df = load_table('V06.13').dropna(subset=['year'])
prov_col = 'Tỉnh, thành phố'
latest = df['year'].max()
sub = df[df['year'] == latest].copy()
# Pull only macro-regions
MACRO_REGIONS = (
    'Đồng bằng sông Hồng',
    'Trung du và miền núi phía Bắc',
    'Bắc Trung Bộ và Duyên hải miền Trung',
    'Tây Nguyên',
    'Đông Nam Bộ',
    'Đồng bằng sông Cửu Long',
)
sub = sub[sub[prov_col].isin(MACRO_REGIONS)].sort_values('value', ascending=True)

fig24 = go.Figure()
fig24.add_trace(go.Bar(
    y=[tr(v) for v in sub[prov_col]], x=sub['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.5)),
    text=[f'{v:,.0f}' for v in sub['value']], textposition='outside',
))
fig24.update_layout(
    title=vi_en('Sản lượng lúa theo vùng kinh tế',
                 'Rice production by macro-region') +
           f' — {int(latest)} (NSO V06.13, ' +
           vi_en('nghìn tấn', 'thousand tonnes') + ')',
    xaxis_title=vi_en('Sản lượng (nghìn tấn)', 'Production (thousand tonnes)'),
    height=440,
)
apply_nvidia_style(fig24)
save_figure(fig24, '24_rice_production_by_region', width=1400, height=440)
fig24.show()
"""))


cells.append(code(r"""
'''Figure 25 — livestock & poultry product output time-series (V06.46).

V06.41 (raw stock counts) ships in PX-Web with the year dimension
collapsed into a combined column header that our parser flattens out;
V06.46 (product output) uses a clean ``(product × year × value)``
schema so it's the cleaner table to plot here.
'''
df = load_table('V06.46').dropna(subset=['year', 'value'])
prod_col = 'Sản phẩm chăn nuôi chủ yếu'
piv = (df.pivot_table(index='year', columns=prod_col,
                        values='value', aggfunc='first')
          .sort_index())
# Restrict to the meat-output series (comparable units, "thousand tonnes")
meat_cols = [c for c in piv.columns if 'thịt' in c.lower()]
piv = piv[meat_cols]

fig25 = go.Figure()
palette = [NV_GREEN, NV_BLACK, NV_GREEN_DARK, NV_GREY, '#888888', NV_GREEN_SOFT]
for (col, color) in zip(piv.columns, palette):
    fig25.add_trace(go.Scatter(
        x=piv.index, y=piv[col], mode='lines+markers', name=tr(col),
        line=dict(color=color, width=2.4), marker=dict(size=5),
    ))
fig25.update_layout(
    title=vi_en('Sản lượng thịt theo loại — cả nước',
                 'Meat output by livestock type — national totals') +
           ' (NSO V06.46)',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Sản lượng (nghìn tấn)', 'Output (thousand tonnes)'),
    height=520,
)
apply_nvidia_style(fig25)
save_figure(fig25, '25_livestock_timeseries', width=1200, height=520)
fig25.show()
"""))


cells.append(code(r"""
'''Figure 26 — total fishery output by macro-region, latest year (V06.63).'''
df = load_table('V06.63').dropna(subset=['year'])
prov_col = 'Tỉnh, thành phố'
latest = df['year'].max()
MACRO_REGIONS = (
    'Đồng bằng sông Hồng',
    'Trung du và miền núi phía Bắc',
    'Bắc Trung Bộ và Duyên hải miền Trung',
    'Tây Nguyên',
    'Đông Nam Bộ',
    'Đồng bằng sông Cửu Long',
)
sub = (df[(df['year'] == latest) & (df[prov_col].isin(MACRO_REGIONS))]
        .sort_values('value', ascending=True))

fig26 = go.Figure()
fig26.add_trace(go.Bar(
    y=[tr(v) for v in sub[prov_col]], x=sub['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.5)),
    text=[f'{v:,.0f}' for v in sub['value']], textposition='outside',
))
fig26.update_layout(
    title=vi_en('Tổng sản lượng thủy sản (đánh bắt + nuôi trồng)',
                 'Total aquaculture + capture fishery output') +
           f' — {int(latest)} (NSO V06.63, ' +
           vi_en('nghìn tấn', 'thousand tonnes') + ')',
    xaxis_title=vi_en('Sản lượng (nghìn tấn)', 'Production (thousand tonnes)'),
    height=440,
)
apply_nvidia_style(fig26)
save_figure(fig26, '26_fishery_by_region', width=1400, height=440)
fig26.show()
"""))


# ---------------------------------------------------------------------------
# §8 V07 industry
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §8 — V07: Industry (9 tables / 5,532 cells)

V07 is the smallest in cell count but anchors the *industrial-output*
view of `industry_sector`. The IIP (V07.01) shows year-on-year movement
in manufacturing, mining, electricity, and water utilities.
"""))


cells.append(code(r"""
'''Figure 27 — IIP by industry sector over time (V07.01).

V07.01 lists 36 ``Ngành công nghiệp`` rows mixing the four VSIC tier-1
industry headers (mining, manufacturing, electricity, water) with ~30
tier-2 sub-sectors. The previous version of this cell took
``piv.columns[:7]`` (alphabetical NSO order), which selected a noisy
mix of high-level and niche sub-sectors and produced legend labels up
to **130 characters** of Vietnamese — wider than the data plot itself.

The fix is to pick a curated short-list (the 4 tier-1 sectors plus 2
high-signal tier-2 sectors — electronics and textiles, both of which
are major Vietnamese export industries), label the legend with concise
English headers (max ~17 chars), and keep the full bilingual VI/EN name
in the hover tooltip so the source taxonomy is still discoverable.
'''
# (canonical_VN_label, short_legend_label) — order = legend order.
SECTOR_SHORT = [
    ('Công nghiệp chế biến, chế tạo',                                  'Manufacturing'),
    ('Khai khoáng',                                                    'Mining'),
    ('Sản xuất và phân phối điện, khí đốt, nước nóng, hơi nước và điều hoà không khí',
                                                                       'Electricity & gas'),
    ('Cung cấp nước; hoạt động quản lý và xử lý rác thải, nước thải',  'Water & waste'),
    ('Sản xuất sản phẩm điện tử, máy vi tính và sản phẩm quang học',   'Electronics'),
    ('Dệt',                                                            'Textiles'),
]

df = load_table('V07.01').dropna(subset=['year'])
keep = [k for k, _ in SECTOR_SHORT]
df = df[df['Ngành công nghiệp'].isin(keep)]
piv = df.pivot_table(index='year', columns='Ngành công nghiệp',
                       values='value', aggfunc='first').sort_index()

fig27 = go.Figure()
palette = [NV_GREEN, NV_BLACK, NV_GREEN_DARK, NV_GREY, '#888888', NV_GREEN_SOFT]
for (col, short), color in zip(SECTOR_SHORT, palette):
    if col not in piv.columns:
        continue
    fig27.add_trace(go.Scatter(
        x=piv.index, y=piv[col], mode='lines+markers',
        name=short,
        line=dict(color=color, width=2.6), marker=dict(size=4),
        # Full bilingual VI/EN in the hover so the legend stays compact
        # but the source taxonomy is still discoverable.
        hovertemplate=(
            '<b>' + tr(col) + '</b><br>'
            '%{x}: %{y:.1f}<extra></extra>'
        ),
    ))
fig27.update_layout(
    title=vi_en('Chỉ số sản xuất công nghiệp (IIP) theo ngành',
                 'Index of Industrial Production (IIP) by sector') +
           ' (NSO V07.01, ' +
           vi_en('năm trước = 100', 'prev year = 100') + ')',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Chỉ số IIP', 'IIP'),
    height=520,
)
apply_nvidia_style(fig27)
save_figure(fig27, '27_iip_by_sector', width=1200, height=520)
fig27.show()
"""))


# ---------------------------------------------------------------------------
# §9 V08-V11 trade, prices, tourism
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §9 — V08–V11: Trade, Exports, Tourism, Prices (72 tables / 23,213 cells)

The four matrix prefixes V08, V09, V10, V11 all live inside the single
*Thương mại, giá cả* database in the ontology — they share the
`trade` / `prices` / `tourism` statistical-domain umbrella. Four
flagship views: retail, exports, tourism, and the CPI heatmap.
"""))


cells.append(code(r"""
'''Figure 28 — retail sales by macro-region time-series (V08.02).'''
df = load_table('V08.02').dropna(subset=['year'])
prov_col = 'Tỉnh, thành phố'
MACRO_REGIONS = (
    'Đồng bằng sông Hồng',
    'Trung du và miền núi phía Bắc',
    'Bắc Trung Bộ và Duyên hải miền Trung',
    'Tây Nguyên',
    'Đông Nam Bộ',
    'Đồng bằng sông Cửu Long',
)
df = df[df[prov_col].isin(MACRO_REGIONS)]
piv = df.pivot_table(index='year', columns=prov_col,
                       values='value', aggfunc='first').sort_index() / 1000   # → trillion VND

fig28 = go.Figure()
order = ['Đông Nam Bộ'] + [r for r in MACRO_REGIONS if r != 'Đông Nam Bộ']
palette = [NV_GREEN, NV_BLACK, NV_GREY, '#888888', NV_LIGHT_GREY, NV_GREEN_DARK]
for region, color in zip(order, palette):
    if region not in piv.columns: continue
    fig28.add_trace(go.Scatter(
        x=piv.index, y=piv[region], mode='lines+markers',
        name=tr(region), line=dict(color=color, width=2.4), marker=dict(size=5),
    ))
fig28.update_layout(
    title=vi_en('Tổng mức bán lẻ hàng hóa theo vùng kinh tế',
                 'Retail sales by macro-region') +
           ' (NSO V08.02, ' + vi_en('nghìn tỷ VND, giá hiện hành',
                                       'trillion VND, current prices') + ')',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Tổng mức bán lẻ (nghìn tỷ VND)',
                        'Retail sales (trillion VND)'),
    height=520,
)
apply_nvidia_style(fig28)
save_figure(fig28, '28_retail_by_region', width=1200, height=520)
fig28.show()
"""))


cells.append(code(r"""
'''Figure 29 — exports by industry, latest year (V09.02).'''
df = load_table('V09.02').dropna(subset=['year'])
df = df[~df['Ngành kinh tế'].str.upper().eq('TỔNG SỐ')]
latest = df['year'].max()
sub = (df[df['year'] == latest]
        .sort_values('value', ascending=True).tail(12))

fig29 = go.Figure()
fig29.add_trace(go.Bar(
    y=[tr(s) for s in sub['Ngành kinh tế']], x=sub['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    text=[f'{v:,.0f}' for v in sub['value']], textposition='outside',
))
fig29.update_layout(
    title=vi_en('Trị giá xuất khẩu hàng hóa theo ngành kinh tế',
                 'Goods exports by industry') +
           f' — {int(latest)} (NSO V09.02, ' +
           vi_en('12 ngành dẫn đầu, triệu USD',
                  'top-12, million USD') + ')',
    xaxis_title=vi_en('Trị giá xuất khẩu (triệu USD)',
                        'Export value (million USD)'),
    height=540,
)
apply_nvidia_style(fig29)
save_figure(fig29, '29_exports_by_industry', width=1400, height=540)
fig29.show()
"""))


cells.append(code(r"""
'''Figure 30 — international visitor arrivals by nationality (V10.05).'''
df = load_table('V10.05').dropna(subset=['year'])
df = df[~df['Quốc tịch'].str.upper().eq('TỔNG SỐ')]
latest = df['year'].max()
sub = (df[df['year'] == latest]
        .sort_values('value', ascending=True).tail(15))

fig30 = go.Figure()
fig30.add_trace(go.Bar(
    y=[tr(n) for n in sub['Quốc tịch']], x=sub['value'], orientation='h',
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.4)),
    text=[f'{v:,.0f}' for v in sub['value']], textposition='outside',
))
fig30.update_layout(
    title=vi_en('Khách quốc tế đến Việt Nam theo quốc tịch',
                 'International visitor arrivals by nationality') +
           f' — {int(latest)} (NSO V10.05, ' +
           vi_en('nghìn lượt', 'thousands') + ')',
    xaxis_title=vi_en('Lượt khách (nghìn người)',
                        'Arrivals (thousand persons)'),
    height=580,
)
apply_nvidia_style(fig30)
save_figure(fig30, '30_tourist_arrivals', width=1400, height=580)
fig30.show()
"""))


cells.append(code(r"""
'''Figure 31 — monthly CPI heatmap, year × month (V11.01, prev-month = 100).'''
df = load_table('V11.01').dropna(subset=['year'])
month_col = 'Các tháng (tháng trước = 100)'
piv = df.pivot_table(index='year', columns=month_col,
                       values='value', aggfunc='first').sort_index()
month_order = [f'Tháng {i}' for i in range(1, 13)]
piv = piv.reindex(columns=[m for m in month_order if m in piv.columns])

fig31 = go.Figure(data=go.Heatmap(
    z=piv.values, x=piv.columns, y=piv.index,
    colorscale=NV_SEQUENTIAL,
    colorbar=dict(title=vi_en('CPI tháng (%)', 'CPI mom (%)'),
                   tickfont=dict(family=NV_FONT_FAMILY)),
    hovertemplate='<b>%{y} %{x}</b><br>CPI %{z:.2f}<extra></extra>',
))
fig31.update_layout(
    title=vi_en('Chỉ số giá tiêu dùng theo tháng',
                 'Monthly CPI (prev. month = 100)') +
           ' — ' + vi_en('năm × tháng', 'year × month heatmap') +
           ' (NSO V11.01)',
    xaxis_title=vi_en('Tháng', 'Month'),
    yaxis_title=vi_en('Năm', 'Year'),
    height=620,
)
apply_nvidia_style(fig31, axes=False)
fig31.update_xaxes(showticklabels=True, tickfont=dict(family=NV_FONT_FAMILY, size=11))
fig31.update_yaxes(showticklabels=True, tickfont=dict(family=NV_FONT_FAMILY, size=11))
save_figure(fig31, '31_cpi_monthly_heatmap', width=1100, height=620)
fig31.show()
"""))


# ---------------------------------------------------------------------------
# §10 V12 transport
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §10 — V12: Transport, Postal, Telecommunications (24 tables / 17,380 cells)

Maps onto the `transport` statistical domain. We pull the cargo-volume
time series — useful for grounding the persona generator's understanding
of regional trade intensity (a worker in Đông Nam Bộ is statistically
more likely to be in logistics than one in Tây Nguyên).
"""))


cells.append(code(r"""
'''Figure 32 — passenger transport volume by sector, time-series (V12.05).'''
import contextlib
candidates = ['V12.05', 'V12.07', 'V12.09', 'V12.11', 'V12.13']
df = None
for tid in candidates:
    with contextlib.suppress(FileNotFoundError):
        df = load_table(tid)
        loaded = tid
        break
if df is not None:
    yc = next((c for c in df.columns if 'Năm' in c), None)
    if yc:
        df['year'] = df[yc].map(_year_int)
        df = df.dropna(subset=['year', 'value'])
    sub = df.groupby('year')['value'].sum().reset_index().sort_values('year')

    fig32 = go.Figure()
    fig32.add_trace(go.Scatter(
        x=sub['year'], y=sub['value'], mode='lines+markers',
        line=dict(color=NV_GREEN, width=3.5), marker=dict(size=8, color=NV_GREEN_DARK),
        fill='tozeroy', fillcolor='rgba(118,185,0,0.18)',
        name=tr('Tổng số'),
    ))
    fig32.update_layout(
        title=vi_en('Khối lượng hành khách / hàng hóa luân chuyển',
                     'Passenger / cargo volume time-series') +
               f' (NSO {loaded})',
        xaxis_title=vi_en('Năm', 'Year'),
        yaxis_title=vi_en('Khối lượng (đơn vị NSO)', 'Volume (NSO units)'),
        height=480,
    )
    apply_nvidia_style(fig32)
    save_figure(fig32, '32_transport_volume', width=1100, height=480)
    fig32.show()
else:
    print('No V12 long-format table found.')
"""))


# ---------------------------------------------------------------------------
# §11 V13 education
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §11 — V13: Education (34 tables / 57,742 cells)

Second-largest cell volume in the catalog (18 %). Provides
*institutional* context for the persona generator's `education_level`
attribute — V02.54 says how many adults have university degrees, V13
says where they got them.
"""))


cells.append(code(r"""
'''Figure 33 — number of schools by level over time (V13.04).

V13.04's ``Cấp học`` column has **8 distinct values** mixing three
different categorisation schemes:

* The 3 modern, current-Vietnamese-curriculum levels (``Tiểu học`` /
  primary, ``Trung học cơ sở`` / lower secondary, ``Trung học phổ thông``
  / upper secondary) — full 30-year coverage 1995–2024.
* 2 legacy aggregates (``Phổ thông cơ sở``, ``Trung học``) reported
  1995–2019 only, replaced by the modern split from 2020 onwards.
* 3 multi-level combined-school types (``Tiểu học và trung học cơ sở``,
  ``Trung học cơ sở và trung học phổ thông``, ``Tiểu học, trung học cơ sở
  và trung học phổ thông``) reported 2020–2024 only.

Plotting all 8 produces a cluttered legend with five short noisy lines
hugging the x-axis and dropping to zero at either end. Filtering to the
3 modern levels gives a clean, comparable across-time picture.
'''
df = load_table('V13.04')
df['year'] = df['Năm học'].map(_year_int)
df = df.dropna(subset=['year'])
sub = df[df['Tổng số và chỉ số phát triển'] == 'Số trường'].copy()
KEEP_LEVELS = ['Tiểu học', 'Trung học cơ sở', 'Trung học phổ thông']
sub = sub[sub['Cấp học'].isin(KEEP_LEVELS)]
piv = sub.pivot_table(index='year', columns='Cấp học',
                       values='value', aggfunc='first').sort_index()
piv = piv.reindex(columns=KEEP_LEVELS)              # canonical legend order

fig33 = go.Figure()
palette = [NV_GREEN, NV_GREEN_DARK, NV_BLACK]
for col, color in zip(KEEP_LEVELS, palette):
    if col not in piv.columns:
        continue
    fig33.add_trace(go.Scatter(
        x=piv.index, y=piv[col], mode='lines+markers', name=tr(col),
        line=dict(color=color, width=2.8), marker=dict(size=5),
    ))
fig33.update_layout(
    title=vi_en('Số trường học theo cấp học',
                 'Number of schools by level') +
           ' (NSO V13.04)',
    xaxis_title=vi_en('Năm (đầu năm học)', 'Year (start of academic year)'),
    yaxis_title=vi_en('Số trường', 'Schools'),
    height=520,
)
apply_nvidia_style(fig33)
save_figure(fig33, '33_schools_by_level', width=1200, height=520)
fig33.show()

print(f'  schools 1995 vs 2024:')
for level in KEEP_LEVELS:
    if level in piv.columns:
        first, last = piv[level].dropna().iloc[[0, -1]]
        print(f'    {tr(level):40s}  {int(first):>6,d} → {int(last):>6,d}'
               f'  ({(last/first - 1) * 100:+.0f}%)')
"""))


# ---------------------------------------------------------------------------
# §12 V14 health & living standards
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §12 — V14: Health, Society, Living Standards (97 tables / 23,934 cells)

V14 is the **largest table count** in the catalog and spans **5
statistical domains** simultaneously: `health`, `society`,
`environment`, `justice`, `living_standards`. Three flagship views:
healthcare facilities, income by quintile, poverty rate.
"""))


cells.append(code(r"""
'''Figure 34 — healthcare facilities national total (V14.04).

V14.04's PX-Web schema is mislabelled: the column called
``'Loại cơ sở, Năm'`` actually holds the **facility type** (8 values
including ``'TỔNG SỐ'``), and the column called ``'Cấp quản lý'``
actually holds the **year** (6 values: 2010, 2011, 2014, 2015, 2016,
2017). On top of the swap, NSO reports the national total
inconsistently — the ``TỔNG SỐ`` row alternates between *full
inventory* (2010, 2014, 2016: ~13,000 facilities each) and *new
construction that year* (2011, 2015, 2017: 46–753), with no flag in
the data telling them apart. The chart used to plot all six side by
side which made the bars look randomly missing.

We threshold on a value floor (``>= 1,000``) to keep only the three
inventory years — the resulting bar chart shows the actual stock of
healthcare facilities at three sampled points across the published
window. The print-out below the figure flags the dropped delta-only
years for transparency.
'''
df = load_table('V14.04').dropna(subset=['value']).copy()
type_col = 'Loại cơ sở, Năm'                            # actually facility type
year_col = 'Cấp quản lý'                                # actually the year
df['year'] = df[year_col].map(_year_int)
nat = df[(df[type_col].str.upper() == 'TỔNG SỐ')].dropna(subset=['year']).sort_values('year')
# Drop the alternating "new construction this year" rows (~50–750 each)
# and keep the three full-inventory snapshots (~13k each).
INVENTORY_FLOOR = 1_000
inventory = nat[nat['value'] >= INVENTORY_FLOOR]
delta = nat[nat['value'] < INVENTORY_FLOOR]

fig34 = go.Figure()
fig34.add_trace(go.Bar(
    x=inventory['year'].astype(int).astype(str), y=inventory['value'],
    marker=dict(color=NV_GREEN, line=dict(color=NV_BLACK, width=0.5)),
    text=[f'{v:,.0f}' for v in inventory['value']], textposition='outside',
))
fig34.update_layout(
    title=vi_en('Tổng số cơ sở y tế — cả nước',
                 'Total healthcare facilities — national') +
           ' (NSO V14.04, ' +
           vi_en('các năm có kiểm kê đầy đủ',
                  'inventory-snapshot years only') + ')',
    xaxis=dict(title=vi_en('Năm', 'Year'), type='category',
                categoryorder='array',
                categoryarray=list(inventory['year'].astype(int).astype(str))),
    yaxis_title=vi_en('Số cơ sở', 'Facilities'),
    height=480,
)
apply_nvidia_style(fig34)
save_figure(fig34, '34_healthcare_facilities', width=1100, height=480)
fig34.show()

if not delta.empty:
    print('  V14.04 also reports new-construction-only rows for the in-between years '
           '(dropped from chart):')
    for _, r in delta.iterrows():
        print(f'    {int(r["year"])}: {int(r["value"])} new facilities reported')
"""))


cells.append(code(r"""
'''Figure 35 — monthly per-capita income by region × urban-rural over time (V14.26).

NSO publishes the quintile-level breakdown in V14.30, but the public
PX-Web XLSX export of V14.30 collapses the quintile dimension into the
header so our long-format parser can't recover it. We use V14.26 here
instead — same survey series, same units (nghìn VND / tháng), but with
a clean ``(stratum × year × value)`` schema.
'''
df = load_table('V14.26').dropna(subset=['year', 'value'])
strata_col = 'Phân tổ'
piv = (df.pivot_table(index='year', columns=strata_col,
                        values='value', aggfunc='first')
          .sort_index())

# Highlight the 4 most-informative strata.
HIGHLIGHT = ['CẢ NƯỚC', 'Thành thị', 'Nông thôn', 'Đông Nam Bộ']
fig35 = go.Figure()
palette = [NV_BLACK, NV_GREEN_DARK, NV_GREY, NV_GREEN]
for col, color in zip(HIGHLIGHT, palette):
    if col not in piv.columns: continue
    width = 3.5 if col == 'CẢ NƯỚC' else 2.4
    fig35.add_trace(go.Scatter(
        x=piv.index, y=piv[col], mode='lines+markers',
        name=tr(col),
        line=dict(color=color, width=width), marker=dict(size=6),
    ))
fig35.update_layout(
    title=vi_en('Thu nhập bình quân đầu người một tháng',
                 'Monthly income per capita') +
           ' (NSO V14.26, ' + vi_en('nghìn VND', 'thousand VND') + ')',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Thu nhập (nghìn VND/tháng)',
                        'Income (thousand VND/month)'),
    height=520,
)
apply_nvidia_style(fig35)
save_figure(fig35, '35_income_by_quintile', width=1100, height=520)
fig35.show()
"""))


cells.append(code(r"""
'''Figure 36 — multidimensional poverty rate by macro-region (V14.45).

V14.45 (along with V03.12 and V03.22) emits **Latin-Eth ``Ð``**
(U+00D0) instead of Vietnamese ``Đ`` (U+0110) in three of the six
macro-region names — ``Ðồng bằng sông Hồng``, ``Ðông Nam Bộ``,
``Ðồng bằng sông Cửu Long``. The two codepoints look identical in most
fonts but break exact-string matching against the canonical Vietnamese
form, so half the lines disappear from the chart. We translate
``Ð → Đ`` (and lowercase ``ð → đ`` for symmetry) on the
``Phân tổ`` column before the ``.isin(...)`` filter.
'''
df = load_table('V14.45').dropna(subset=['year'])
df['Phân tổ'] = df['Phân tổ'].astype(str).str.translate(
    str.maketrans({'\u00D0': '\u0110', '\u00F0': '\u0111'}))
MACRO_REGIONS = (
    'Đồng bằng sông Hồng',
    'Trung du và miền núi phía Bắc',
    'Bắc Trung Bộ và Duyên hải miền Trung',
    'Tây Nguyên',
    'Đông Nam Bộ',
    'Đồng bằng sông Cửu Long',
)
sub = df[df['Phân tổ'].isin(MACRO_REGIONS)]
piv = sub.pivot_table(index='year', columns='Phân tổ',
                       values='value', aggfunc='first').sort_index()

fig36 = go.Figure()
order = MACRO_REGIONS
palette = [NV_GREEN, NV_BLACK, NV_GREY, '#888888', NV_LIGHT_GREY, NV_GREEN_DARK]
for region, color in zip(order, palette):
    if region not in piv.columns: continue
    fig36.add_trace(go.Scatter(
        x=piv.index, y=piv[region], mode='lines+markers', name=tr(region),
        line=dict(color=color, width=2.4), marker=dict(size=5),
    ))
fig36.update_layout(
    title=vi_en('Tỷ lệ hộ nghèo đa chiều theo vùng kinh tế',
                 'Multidimensional poverty rate by macro-region') +
           ' (NSO V14.45, % ' + vi_en('hộ', 'of households') + ')',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('Tỷ lệ hộ nghèo (%)', 'Poverty rate (%)'),
    height=520,
)
apply_nvidia_style(fig36)
save_figure(fig36, '36_poverty_by_region', width=1200, height=520)
fig36.show()
"""))


# ---------------------------------------------------------------------------
# §13 V15 international
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §13 — V15: International Statistics (10 tables / 11,502 cells)

V15 isn't used by the persona generator (we don't generate non-Vietnamese
personas), but it provides peer-country context that a tab in the
visualizer can surface — "where does Vietnam sit on PPP-adjusted GDP per
capita relative to ASEAN?".
"""))


cells.append(code(r"""
'''Figure 37 — PPP-adjusted GDP per capita, Vietnam vs ASEAN (V15.05).

Two NSO V15.05 quirks the previous version of this cell tripped over —
the chart used to render only 5 of the 10 lines:

1. The PX-Web export uses **hyphenated phonetic** Vietnamese country
   names rather than the short Latin spellings most readers expect
   (``Xin-ga-po`` rather than ``Singapore``, ``In-đô-nê-xi-a`` rather
   than ``Inđônêxia``, ``Phi-li-pin`` rather than ``Philíppin``,
   ``Ma-lai-xi-a`` rather than ``Malaysia``, ``Mi-an-ma`` rather than
   ``Mianma``). Five of the original ten ``PEERS`` entries silently
   failed the ``.isin(...)`` filter, so the chart had only 5 lines —
   easy to miss because the legend just got shorter.
2. **Bru-nây has a NaN at 2006** (single missing year). Plotly's
   ``Scatter`` default is ``connectgaps=False``, so the Brunei line
   breaks into two disconnected segments. Setting ``connectgaps=True``
   bridges the single-year gap with a straight line; the trend reads
   as a continuous curve across 1999–2023.

Restricted to the 10 ASEAN peers — also drop the pre-2009 segment
where NSO data has obvious unit-scale anomalies (e.g. Thái Lan jumps
from 7,595 USD in 2003 to 4,850 in 2004 to 54,947 in 2005, a
suspicious 11x spike that doesn't match any external source).
'''
df = load_table('V15.05').dropna(subset=['year'])
PEERS = ['Việt Nam', 'Thái Lan', 'Xin-ga-po', 'In-đô-nê-xi-a',
         'Phi-li-pin', 'Ma-lai-xi-a', 'Mi-an-ma', 'Cam-pu-chia',
         'Lào', 'Bru-nây']
sub = df[df['Quốc gia, vùng lãnh thổ'].isin(PEERS)]
sub = sub[sub['year'] >= 2009]                         # drop noisy pre-2009 window
piv = sub.pivot_table(index='year', columns='Quốc gia, vùng lãnh thổ',
                       values='value', aggfunc='first').sort_index()

fig37 = go.Figure()
order = ['Việt Nam'] + [c for c in piv.columns if c != 'Việt Nam']
palette = [NV_GREEN, NV_BLACK, NV_GREEN_DARK, NV_GREY, '#888888',
            NV_LIGHT_GREY, NV_GREEN_SOFT, '#444444', '#222222', '#bbbbbb']
for col, color in zip(order, palette):
    if col not in piv.columns: continue
    width = 4 if col == 'Việt Nam' else 1.6
    fig37.add_trace(go.Scatter(
        x=piv.index, y=piv[col], mode='lines', name=tr(col),
        line=dict(color=color, width=width),
        connectgaps=True,                              # bridge any single-year NaN
    ))
fig37.update_layout(
    title=vi_en('GDP bình quân đầu người PPP — Việt Nam và ASEAN',
                 'PPP-adjusted GDP per capita — Vietnam in ASEAN context') +
           ' (NSO V15.05, USD)',
    xaxis_title=vi_en('Năm', 'Year'),
    yaxis_title=vi_en('GDP/người (PPP, USD)', 'GDP per capita (PPP, USD)'),
    height=540,
)
apply_nvidia_style(fig37)
save_figure(fig37, '37_gdp_per_capita_asean', width=1200, height=540)
fig37.show()

print(f'  series rendered: {len(fig37.data)} / {len(PEERS)} ASEAN peers')
"""))


# ---------------------------------------------------------------------------
# §14 coverage matrix + provenance
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §14 — Cross-database coverage matrix

Which (database, key-variable) pairs actually have tables behind them?
This is the chart that tells the persona generator which CPDs are
ground-truthable from PX-Web vs which need a synthetic prior.
"""))


cells.append(code(r"""
'''Figure 38 — coverage heatmap (database × variable code).'''
KEY_VARS = ['Năm', 'Tỉnh/Thành phố', 'Tỉnh, thành phố', 'Vùng',
             'Thành thị, nông thôn', 'Phân tổ', 'Giới tính', 'Nhóm tuổi',
             'Nghề nghiệp', 'Ngành', 'Chuyên môn kỹ thuật',
             'Trình độ chuyên môn kỹ thuật', 'Cách tính', 'Nhóm thu nhập']

rows = []
for db, group in catalog.groupby('database'):
    var_count = Counter()
    for variables in group['variables']:
        var_count.update(variables)
    for var in KEY_VARS:
        rows.append({'database': db, 'variable': var, 'tables': var_count.get(var, 0)})
heat = pd.DataFrame(rows).pivot(index='database', columns='variable', values='tables')
heat = heat.reindex(columns=KEY_VARS)

fig38 = go.Figure(data=go.Heatmap(
    z=heat.values, x=heat.columns, y=heat.index,
    colorscale=NV_SEQUENTIAL,
    colorbar=dict(title=vi_en('Số bảng', '# tables'),
                   tickfont=dict(family=NV_FONT_FAMILY)),
    text=heat.values, texttemplate='%{text}',
    hovertemplate='<b>%{y}</b> × %{x}<br>%{z} tables<extra></extra>',
))
fig38.update_layout(
    title=vi_en('Ma trận độ phủ PX-Web — cơ sở dữ liệu × mã biến chính',
                 'PX-Web coverage matrix — databases × key variable codes'),
    xaxis_title='', yaxis_title='', height=620,
)
fig38.update_xaxes(tickangle=-30)
apply_nvidia_style(fig38, axes=False)
fig38.update_xaxes(showticklabels=True, tickfont=dict(family=NV_FONT_FAMILY, size=11))
fig38.update_yaxes(showticklabels=True, tickfont=dict(family=NV_FONT_FAMILY, size=11))
save_figure(fig38, '38_coverage_matrix', width=1300, height=620)
fig38.show()
"""))


# ---------------------------------------------------------------------------
# §15 provenance
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §15 — Persona-attribute provenance

Which NSO matrix grounds which persona dimension? This sankey diagram
is built from `configs/ontology.yaml::persona_dimensions[*].sources`
and shows the full provenance chain: every time the persona generator
samples (say) an `occupation`, it traces back to NSO matrix V02.43.
"""))


cells.append(code(r"""
'''Figure 39 — provenance Sankey: persona dimension → PX-Web matrix.'''
import yaml
ont = yaml.safe_load((REPO_ROOT / 'configs' / 'ontology.yaml').read_text())
dims = ont['persona_dimensions']

labels, sources, targets, values = [], [], [], []
def _idx(label):
    if label not in labels:
        labels.append(label)
    return labels.index(label)

for d in dims:
    if not d.get('sources'):
        continue
    dim_idx = _idx(f'persona/{d["id"]}')
    for src in d['sources']:
        m = _idx(f'matrix/{src}')
        sources.append(dim_idx); targets.append(m); values.append(1)

fig39 = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(
        pad=18, thickness=18,
        line=dict(color=NV_BLACK, width=0.7),
        label=labels,
        color=[NV_GREEN if l.startswith('persona/') else NV_LIGHT_GREY for l in labels],
    ),
    link=dict(
        source=sources, target=targets, value=values,
        color='rgba(118,185,0,0.32)',
    ),
)])
fig39.update_layout(
    title=vi_en('Nguồn gốc thuộc tính persona — bảng NSO làm nền cho mỗi chiều',
                 'Persona-attribute provenance — NSO matrix that grounds each dimension'),
    height=620, font=dict(family=NV_FONT_FAMILY, color=NV_BLACK, size=12),
    paper_bgcolor='white', plot_bgcolor='white',
)
save_figure(fig39, '39_persona_provenance', width=1200, height=620)
fig39.show()
"""))


# ---------------------------------------------------------------------------
# §16 full inventory
# ---------------------------------------------------------------------------
cells.append(md(r"""
## §16 — Full 502-table inventory

A summary table of every PX-Web matrix on disk: id, database, title,
cell count, variable codes. Useful for grep'ing when you're trying to
find "the table with both age group and urbanicity" or similar.

The same data is dumped to a CSV at
`docs/figures/analysis/16_full_table_inventory.csv` for downstream
queries (and HF dataset card generation).
"""))


cells.append(code(r"""
'''Cell — render the full 502-row inventory + CSV side-effect.'''
inv = catalog.copy()
inv['n_vars']    = inv['variables'].str.len()
inv['variables'] = inv['variables'].str.join(' / ')
inv['parquet']   = inv['parquet'].astype(str).str.replace(str(REPO_ROOT) + '/', '', regex=False)
inv = inv[['table_id', 'prefix', 'database', 'title', 'source', 'n_cells', 'n_vars', 'variables', 'parquet']]
inv = inv.sort_values('table_id').reset_index(drop=True)

csv_path = OUT_DIR / '16_full_table_inventory.csv'
inv.to_csv(csv_path, index=False)
print(f'wrote {csv_path}  ({len(inv)} rows, {inv["n_cells"].sum():,} total cells)')

with pd.option_context('display.max_colwidth', 80, 'display.max_rows', 502):
    display(inv)
"""))


cells.append(code(r"""
'''Cell — per-database aggregate summary (cells, time span, spatial granularity).'''
agg = (catalog.groupby('database')
                .agg(n_tables=('table_id', 'count'),
                     n_cells=('n_cells', 'sum'),
                     access_paths=('source', lambda x: ' / '.join(sorted(set(x)))))
                .sort_values('n_cells', ascending=False)
                .reset_index())
agg.loc['Total'] = ['Total',
                     agg['n_tables'].sum(), agg['n_cells'].sum(), '']
display(agg)
"""))


cells.append(md(r"""
---

## Summary — what this analysis demonstrated

* **Catalogue coverage**: 502 matrices across 12 NSO databases (15
  PX-Web prefixes), 316 K cells. The HTML-form scraper accounts for
  63 % of the tables — without it we'd be missing the entire economy
  side of the data.
* **Time depth**: earliest 1990, latest 2024 — a 35-year window suitable
  for trend analysis on every macro indicator.
* **Spatial depth**: every persona-relevant table has province-level
  granularity (`Tỉnh, thành phố`) — the NSO 63-province codes.
* **Persona grounding**: 9 of 11 persona dimensions are fully grounded
  in PX-Web data (see fig 39); the remaining 2 (`ethnicity`,
  `income_quintile`) use Census + HLSS priors awaiting NSO's joint
  publications.
* **NVIDIA brand styling**: every figure is a Plotly chart on a white
  background using NVIDIA Green `#76B900` as the primary data colour
  with the NVIDIA Sans typography fallback, as required by the
  [logo & brand guidelines](https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/).

For the static-markdown view of these results, see
[`DATAANALYSIS.md`](DATAANALYSIS.md). For the persona-generation side,
see [`DATASYNTHESIS.md`](DATASYNTHESIS.md).
"""))


# ---------------------------------------------------------------------------
# Build the notebook
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
nbf.write(nb, NB_PATH)
print(f'wrote {NB_PATH}  ({len(cells)} cells)')
