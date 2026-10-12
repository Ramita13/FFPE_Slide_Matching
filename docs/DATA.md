# Dataset

## Source

The images come from the public dataset released with Tang *et al.* (2026),
*"An AI framework for automated quality control of paraffin block and slide
consistency"*, Virchows Archiv. Dataset DOI:
[10.6084/m9.figshare.31451491](https://doi.org/10.6084/m9.figshare.31451491).
Collected at the Second Affiliated Hospital of Soochow University (Jan–May 2025).

The data is **not redistributed** in this repository. Download it from figshare
and place it as:

```
data/
  LK/   CaseXXXX_LK.jpg   # paraffin block photographs
  QP/   CaseXXXX_QP.jpg   # H&E slide preview scans
```

## Pairing

A block and its slide share the case ID: `Case0001_LK.jpg` ↔ `Case0001_QP.jpg`.

| | Count |
|---|---|
| Block images (`LK`) | 2,218 |
| Slide images (`QP`) | 2,216 |
| Complete pairs | **2,216** |
| Blocks without slide | `Case0356`, `Case1751` |

## Acquisition (per the paper)

- **Block (LK):** Huawei Mate60 phone camera on a fixed stand, 90° ring light;
  originally 4160×3120 JPEG.
- **Slide (QP):** DMetrix WSI scanner, fast preview mode, 1712×596 PNG.
- Specimen mix in the original study: ~1,587 surgical resections, ~633 small
  biopsies (endoscopic, cervical, needle core, fragmented tissue).

## Observations on the released images

- Both modalities are **already cropped to the tissue region** (typically
  ~300–650 px per side), consistent with the paper's YOLOv11 ROI step.
- Block images: brown/beige tissue on grey-white wax, low contrast, specular
  highlights, blur; separate fragments often look fused.
- Slide images: H&E on white, high contrast, fine detail; fragments can be
  torn, folded or split compared with the block.
- **There are no labels for missing tissue.** All pairs are presumed to be
  genuine matches; whether any contain real omissions is unknown. See
  [PROBLEM.md](PROBLEM.md) for how we construct evaluation labels.

Detailed statistics are produced by the EDA in
[research/05_dataset_eda.md](research/05_dataset_eda.md).
