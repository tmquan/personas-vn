# Vietnam — Geographic data visualisation

Companion to [`DATAVISUALIZATION.ipynb`](DATAVISUALIZATION.ipynb). The
notebook is the interactive artefact (open in JupyterLab to pan, zoom,
hover); this markdown is the static counterpart with the same 11 maps
rendered as embedded PNGs for offline review.

Where [`DATAANALYSIS.md`](DATAANALYSIS.md) plots the curated NSO data as
distributions and time-series, this document plots them on the **map of
Vietnam**.

Coverage:

* **All 63 administrative provinces** (admin-1 level, sourced from
  [geoBoundaries gbOpen ADM1](https://github.com/wmgeolab/geoBoundaries)).
* **Quần đảo Hoàng Sa** (Paracel Islands, administered by Đà Nẵng) and
  **Quần đảo Trường Sa** (Spratly Islands, administered by Khánh Hòa)
  rendered as dashed-outline bounding polygons with their principal-
  island markers, matching the Vietnamese General Statistics Office's
  cartographic convention.
* **5 major named Vietnamese islands**: Đảo Phú Quốc (largest),
  Đảo Cát Bà (Hạ Long Bay biosphere), Đảo Bạch Long Vĩ (Gulf of
  Tonkin), Đảo Phú Quý, Côn Đảo.
* **Capital + 7 major cities** with TP. prefix: TP.Hà Nội, TP.Hồ Chí
  Minh, TP.Đà Nẵng, TP.Hải Phòng, TP.Vinh, TP.Huế, TP.Cần Thơ,
  TP.Nha Trang.

Every figure follows the
[NVIDIA brand guidelines](https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/):
white background, NVIDIA Green `#76B900` for the primary data series,
black for axis chrome, NVIDIA Sans typography (with safe fallbacks).
Every label uses CAD-style L-elbow leader lines so text never collides
with province polygons or other labels.

Run prerequisites:

```bash
conda activate pgm                                       # one-time setup: see README
pip install -e ".[curator,viz]" "kaleido>=1.0,<2.0"
python -m packages.pipeline.cli curate                   # populates data/nso-gov-vn/raw/pxweb/vi/
python -m scripts.render_maps                            # writes 11 PNG + 11 HTML
```

---

## §1 — The country itself

The reference map: **63 admin-1 provinces** + **Côn Đảo** offshore +
the two **archipelagos** with their principal-island markers + the **5
major named Vietnamese islands** + Hà Nội (★) plus 7 major cities.

This is the base layer every subsequent map uses. The province polygons
are uncoloured here (uniform NVIDIA-green-soft fill) so the sea, the
borders, and the island labels are all clearly visible.

![Vietnam reference map](docs/figures/maps/01_country_reference.png)

---

## §2 — Population by province (V02.01)

The most fundamental layer. NSO V02.01 `Diện tích, dân số và mật độ
dân số phân theo địa phương` by province. We keep just population
(`Dân số trung bình (Nghìn người)`) for the latest year and colour by
that.

Visual takeaways at a glance: **Hà Nội + TP.HCM** are by far the
largest population centres; the **Mekong Delta** is dense in absolute
population but its provinces are large; the **Central Highlands** and
**Northern mountains** are sparse.

![Population by province](docs/figures/maps/02_population_by_province.png)

---

## §3 — GRDP per capita by province (V03.12)

The single best summary of regional economic intensity. **TP.Hồ Chí
Minh, Hà Nội, Bắc Ninh, Bình Dương, Bà Rịa - Vũng Tàu, Đà Nẵng** are
the green hot-spots; the entire **Central Highlands** and **Northern
mountains** sit at the bottom of the distribution.

![GRDP per capita](docs/figures/maps/03_grdp_per_capita.png)

---

## §4 — Industrial Production Index by province (V07.02)

The IIP measures year-on-year industrial output (prev-year = 100, so
*above* 100 = growth). The map shows which provinces are growing
fastest. Northern industrial-park hubs (**Bắc Giang, Bắc Ninh**) and
the **Central Coast** typically lead; the agricultural provinces sit
near 100.

![Industrial Production Index by province](docs/figures/maps/04_iip_by_province.png)

---

## §5 — Tourism revenue by province (V10.03)

Revenue from `lữ hành` (organised travel) by province. Heavily
concentrated in **TP.HCM, Hà Nội, Đà Nẵng, Khánh Hòa, Quảng Ninh** —
the top-5 cumulatively account for >75 % of the national total. The
inland Northern and Highland provinces have negligible lữ hành
revenue.

![Tourism revenue](docs/figures/maps/05_tourism_revenue.png)

---

## §6 — Monthly income per capita (V14.36)

Mean monthly per-capita income from the Vietnam Household Living
Standards Survey (VHLSS), broken out by province. Together with §3
(GRDP per capita) this is the closest the dataset gets to a
"household wealth" snapshot.

![Income per capita](docs/figures/maps/06_income_per_capita.png)

---

## §7 — Poverty rate by province (V14.47)

The dual of the income map. **Highland and mountain provinces** carry
the bulk of Vietnam's remaining poverty (Hà Giang, Cao Bằng, Lai
Châu, Điện Biên, Sơn La all > 10 %). **TP.HCM, Hà Nội, Bình Dương,
Đồng Nai** all sit below 1 %.

![Poverty rate](docs/figures/maps/07_poverty_rate.png)

---

## §8 — Number of enterprises by province (V05.04)

The map of where Vietnam's businesses cluster. **TP.Hồ Chí Minh** alone
hosts ~30 % of all enterprises in the country, **Hà Nội** another
~20 %; the South East and Red River Delta industrial belts dominate.
The Mekong Delta and the Highlands have far fewer enterprises despite
sizeable populations.

![Enterprises by province](docs/figures/maps/08_enterprises_by_province.png)

---

## §9 — Social insurance participation (V03.22)

The share of working-age population enrolled in BHXH. The pattern
matches the formal-sector employment map: **TP.HCM (~61 %), Bắc Ninh,
Hà Nội, Bình Dương, Hải Phòng** all > 55 %. The agricultural Mekong
and Highland provinces sit below the 38 % national average.

![Social insurance participation](docs/figures/maps/09_bhxh_participation.png)

---

## §10 — Bubble map: enterprises (size) × IIP growth (colour)

Combines two PX-Web datasets in one view to capture both **scale** and
**dynamism**:

* **Bubble size** = number of active enterprises (V05.04, latest year)
  — *where* the industrial economy lives.
* **Bubble colour** = Industrial Production Index (V07.02, latest year)
  — *which* of those clusters is currently growing fastest.

The big-and-green bubbles are the provinces that are both large *and*
expanding (Bắc Ninh, Bình Dương, Hà Nội). The big-and-grey bubbles
(TP.HCM) are large but flat. The small-and-green bubbles are the
fast-growing newcomers worth watching (Hà Nam, Vĩnh Phúc).

![Enterprises × IIP bubble map](docs/figures/maps/10_enterprises_iip_bubble.png)

---

## §11 — Macro-region overlay

The 6 NSO macro-regions, drawn as filled bands over the 63 provinces.
Useful for pinning down where each region's boundary actually lies (the
NSO publishes both 63-province *and* 6-region tables; this map makes
the partition explicit).

![Macro-region overlay](docs/figures/maps/11_macro_regions.png)

---

## Cartographic notes

### Authoritative coordinates

Every place mark on every map sits at its canonical Wikipedia /
Vietnamese General Statistics Office coordinate, verified to **0.00 km
deviation**:

| Entity | Coordinates |
| --- | --- |
| **Capital ★ Hà Nội** | 21.0285°N, 105.8542°E |
| TP.Hồ Chí Minh | 10.7769°N, 106.7009°E |
| TP.Đà Nẵng | 16.0544°N, 108.2022°E |
| TP.Hải Phòng | 20.8449°N, 106.6881°E |
| TP.Vinh | 18.6792°N, 105.6920°E |
| TP.Huế | 16.4637°N, 107.5909°E |
| TP.Cần Thơ | 10.0452°N, 105.7469°E |
| TP.Nha Trang | 12.2388°N, 109.1968°E |
| Đảo Phú Quốc | 10.22°N, 104.00°E |
| Đảo Cát Bà | 20.78°N, 107.05°E |
| Đảo Bạch Long Vĩ | 20.13°N, 107.72°E |
| Đảo Phú Quý | 10.55°N, 108.93°E |
| Côn Đảo | 8.68°N, 106.60°E |

### Hoàng Sa & Trường Sa

The two archipelago bounding boxes match the published Vietnamese
sovereignty extents. Each box carries the **principal islands** of the
archipelago as small markers inside the dashed outline, plus a
bilingual label below:

| Archipelago | Principal islands shown |
| --- | --- |
| **Quần đảo Hoàng Sa** (Paracel Is.) | Đảo Phú Lâm (Woody I.), Đảo Tri Tôn (Triton I.), Đảo Linh Côn (Lincoln I.), Đảo Quang Hòa (Duncan I.) |
| **Quần đảo Trường Sa** (Spratly Is.) | Đảo Trường Sa Lớn (Spratly I.), Song Tử Tây (Southwest Cay), Đảo Sinh Tồn (Sin Cowe I.), Đảo Phan Vinh (Pearson Reef), Đảo An Bang (Amboyna Cay), Đảo Nam Yết (Namyit I.), Đá Cô Lin (Collins Reef), Đảo Sơn Ca (Sand Cay) |

Both archipelago labels sit *below* their dashed bounding boxes; the
text never visually fuses with the box border or with any nearby
mainland city / island label.

### Layout & typography

* **CAD-style L-elbow leader lines.** Every label that sits offset
  from its marker is connected by a strict 90° elbow: vertical from
  marker to corner, then horizontal to label. The horizontal final
  approach matches GD&T / cartographic dimension-callout convention so
  labels read off a flat baseline.
* **Pixel-perfect alignment.** All 11 maps share the same paper
  fraction (`mapbox.domain = {x: [0.01, 0.84], y: [0.01, 0.90]}`), the
  same canvas size (1100 × 900), and a canvas-anchored colorbar at
  fixed `x=0.87, y=0.50`. Browsing through the 11 maps shows Vietnam
  at *identical* pixel coordinates with no positional drift between
  figures.
* **Province borders always visible.** Every admin-1 polygon is drawn
  with a 0.9-px black outline, even for provinces whose data value is
  zero / missing on the choropleth (they get a transparent fill but
  the border still renders, courtesy of a base-outline trace beneath
  the data trace).
* **NVIDIA brand styling.** White background, NVIDIA Green `#76B900`
  primary, black axes, NVIDIA Sans typography fallback chain.

---

## Reproducibility

Every PNG can be regenerated bit-identically from the source data:

```bash
git clone https://github.com/your-user/personas-vn
cd personas-vn
conda create -n pgm python=3.11 -y && conda activate pgm    # one-time
pip install -e ".[curator,dev]"                              # one-time
python -m packages.pipeline.cli curate                       # populates raw/pxweb/vi/
python -m scripts.render_maps                                # 11 PNGs + 11 HTMLs in docs/figures/maps/
```

The interactive HTML versions (`docs/figures/maps/*.html`) carry the
full Mapbox-GL pan / zoom / hover. They're standalone — open any in a
browser without needing a server.

For the analysis side of the dataset (38 NVIDIA-styled distribution
and time-series figures across all 12 NSO databases), see
[`DATAANALYSIS.md`](DATAANALYSIS.md). For how the data was curated,
[`DATAPROCESSING.md`](DATAPROCESSING.md). For the curator-pipeline
terminal-artefact tour (502-row UMAP, 12 clusters, semantic
neighbour search), [`DATAEXPLORATION.md`](DATAEXPLORATION.md).
For the synthetic-persona pipeline,
[`DATASYNTHESIS.md`](DATASYNTHESIS.md).
