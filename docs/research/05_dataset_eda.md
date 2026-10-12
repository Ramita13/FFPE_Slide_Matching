# 05 — Dataset Exploratory Data Analysis

*Scope:* all 2,218 block photos (LK) and 2,216 slide previews (QP) of the Tang *et al.*
(2026) release. *Script:* [`scripts/eda.py`](../../scripts/eda.py). *Outputs:*
`results/eda/eda_stats.csv` (one row per case), `results/eda/eda_quantiles.csv`,
`results/eda/eda_summary.png`, `results/eda/eda_specimen_space.png`, and contact sheets
in `results/eda/inspect/` (PNGs are git-ignored; regenerate them with the commands below).

---

## 1. Summary

The points that matter most for method design:

1. **Both modalities share a common pixel scale.** The geometric-mean crop-size ratio
   `crop_scale = sqrt(area_QP / area_LK)` has a median of **1.05** (IQR 1.00–1.10,
   p5–p95 0.93–1.27), and the long sides of LK and QP crops correlate at r = 0.96. The
   release appears to have rescaled the block photos to the slide-preview resolution
   (about 23 px/mm, given that the 596 px preview height covers a ~26 mm slide). **Scale
   is therefore close to known (≈1.0–1.2×, since sections expand slightly on the water
   bath)**. That is a strong prior for registration, and a crop that departs a lot
   from it is itself a QC signal.
2. **Orientation is essentially arbitrary.** Of the 856 pairs where both crops are
   clearly elongated, 501 (59%) swap portrait↔landscape, so they are rotated by about
   90°. Visual inspection also shows 180° rotations and probable mirror images. Methods
   must handle the full rotation group and reflections.
3. **Block segmentation is the bottleneck.** Block images are low-contrast (median grey
   std 24 vs 29 for slides; 22% of blocks have Laplacian variance < 2, which is very
   blurry). Wax colour varies (grey, pink, bluish), fatty tissue is nearly
   wax-coloured, and a simple threshold finds **no tissue in 38 blocks (1.7%)**, almost
   all of them tiny biopsies.
4. **Fragments fuse in the block and split on the slide.** The block shows one
   fragment in 63% of cases (median 1). The slide shows a median of 2, and 40% of pairs
   have equal counts. Fragment counts are only weakly concordant (Spearman ρ = 0.26).
   A count comparison is far too noisy to be a detector on its own.
5. **Very pale tissue is common on slides.** Fat, oedematous stroma and mucus show up
   as near-white lace (saturation ≈ 10 vs ≈ 8 for glass). On roughly 1–2% of slides
   (e.g. Case0113, Case0627, Case1747) almost nothing is stained, and a colour threshold
   finds next to nothing. "Faint" must not be read as "missing".
6. **Repeated levels / ribbons exist but are uncommon.** We confirmed them visually in
   Case0074, Case0182, Case0190, Case1365 and Case1950. They concentrate in slide crops
   that are much wider than the block (78 pairs with `qp_w > 1.4 × lk_long_side`; about
   20% of a sample of those showed repeats), which puts the overall prevalence at about
   1–3%. In several of these the second level is **cut off by the preview/ROI edge**.
7. **Crop/ROI artefacts can mimic omission.** Some QP crops contain only part of the
   section or none of it (Case0654 is a slide corner with no tissue). 64 slides hit the
   596 px preview height cap. A pipeline needs a "crop invalid" output that is separate
   from "tissue missing".
8. **We found a handful of plausible real omissions** (§5), mostly small biopsies where
   the block shows more fragments than the slide. The most convincing are Case1072,
   Case1131, Case1535 and Case1561.
9. **A biopsy/resection proxy from crop size works.** "LK long side < 250 px, or < 400 px
   with ≥ 4 fragments" labels **27.3% as biopsy**, close to the 28.5% (633/2,220) the
   source study reports.

---

## 2. Method

All metrics are computed per image at native resolution unless noted. Masks are
**deliberately simple proxies** for EDA, not production segmentations.

| Metric | Definition |
|---|---|
| `h, w, aspect, elong` | size; `aspect = w/h`; `elong = long/short` (rotation-invariant) |
| `brightness, contrast` | mean / std of grey level |
| `blur` | variance of Laplacian after resizing the long side to 512 px (lower = blurrier; small crops are upsampled and so score low) |
| `lk_glare` | fraction of pixels with V ≥ 245 and S ≤ 30 (for slides this is just white glass, stored as `qp_white_frac`) |
| Block mask | Wax colour = median Lab of the most wax-like half (bright, low chroma) of a 4% border ring. Tissue = chroma-weighted Lab distance from wax > max(Otsu, 14). Glare is treated as background. Open/close, fill holes. |
| Slide mask | "Stained": S > max(Otsu, 25) or V < bg − 60. "Pale": smoothed S > bg + 3 **or** local grey-level std > bg + 3. Union, open/close (kernel 3% of the short side), fill holes. `qp_pale_frac` = share of the mask found only by the pale rule; `qp_stained_frac` = stained-only area fraction. |
| `tissue_frac, n_frag, n_frag_small, largest_frac, solidity` | Connected components: `n_frag` counts components ≥ 0.2% of image area, `n_frag_small` counts 0.03–0.2%. Solidity of the union mask is computed against its convex hull. |
| `border_touch` | fraction of the outermost pixel ring covered by mask (tissue truncated by the crop, or a slide frame line) |
| `qp_n_similar_pairs` | pairs of slide components with area ratio ≥ 0.75 and Hu-moment `matchShapes` < 0.15 (intended as a repeated-level proxy, see §4.4) |
| `qp_pen_frac` | blue/green/near-black pixel fraction (intended for pen ink, see §4.6) |
| Pair | `elong_ratio = qp_elong/lk_elong`; `frag_diff = qp_n_frag − lk_n_frag`; `area_ratio = qp_tissue_frac/lk_tissue_frac`; `crop_scale = sqrt(qp_area/lk_area)`; `orient_swap` (both elongated > 1.2, different orientation); `specimen_proxy` |

**Visual inspection.** About 150 pairs were inspected on contact sheets showing image
and mask contours side by side (`--montage`):

- outliers by `frag_diff` (both tails), `area_ratio` (both tails), `crop_scale`,
  `elong_ratio`, `qp_n_similar_pairs`, blur, empty block masks, preview-height-capped
  slides and pen-ink score;
- **58 randomly sampled pairs** (seeds 0 and 1) used to estimate prevalence;
- a random 12 of the 78 "wide slide" pairs.

---

## 3. Statistics

### 3.1 Quantiles (all cases)

| metric | n | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lk_h` | 2218 | 57 | 76 | 242 | 430 | 539 | 656 | 712 |
| `lk_w` | 2218 | 56 | 75 | 248 | 412 | 518 | 639 | 699 |
| `qp_h` | 2216 | 69 | 87 | 244 | 414 | 508 | 579 | 596 |
| `qp_w` | 2216 | 67 | 90 | 262 | 478 | 617 | 745 | 852 |
| `lk_elong` | 2218 | 1 | 1.02 | 1.09 | 1.19 | 1.34 | 1.68 | 2.08 |
| `qp_elong` | 2216 | 1 | 1.02 | 1.1 | 1.21 | 1.4 | 1.74 | 2.1 |
| `lk_brightness` | 2218 | 89 | 104 | 132 | 145 | 153 | 166 | 180 |
| `qp_brightness` | 2216 | 182 | 193 | 209 | 219 | 226 | 233 | 236 |
| `lk_contrast` | 2218 | 3.1 | 6.2 | 13.8 | 24.1 | 36.7 | 58.5 | 69.7 |
| `qp_contrast` | 2216 | 9.7 | 15.1 | 23 | 29.1 | 35.1 | 44.7 | 51.5 |
| `lk_blur` | 2218 | 0.63 | 0.87 | 2.32 | 8.16 | 34.6 | 182 | 464 |
| `qp_blur` | 2216 | 1.15 | 1.38 | 4.19 | 15.8 | 33.2 | 82.7 | 126 |
| `lk_glare` | 2218 | 0 | 0 | 0 | 0 | 2e-5 | 2e-4 | 6.5e-4 |
| `lk_tissue_frac` | 2218 | 0 | 0.073 | 0.283 | 0.438 | 0.552 | 0.678 | 0.75 |
| `qp_tissue_frac` | 2216 | 0.19 | 0.279 | 0.44 | 0.546 | 0.633 | 0.746 | 0.821 |
| `qp_pale_frac` | 2216 | 0.10 | 0.14 | 0.29 | 0.45 | 0.59 | 0.81 | 0.92 |
| `lk_n_frag` | 2218 | 0 | 1 | 1 | 1 | 2 | 4 | 6 |
| `qp_n_frag` | 2216 | 1 | 1 | 1 | 2 | 3 | 6 | 8 |
| `lk_n_frag_small` | 2218 | 0 | 0 | 0 | 0 | 1 | 3 | 6 |
| `qp_n_frag_small` | 2216 | 0 | 0 | 1 | 2 | 4 | 8 | 13 |
| `lk_largest_frac` | 2218 | 0 | 0.45 | 0.86 | 0.999 | 1 | 1 | 1 |
| `qp_largest_frac` | 2216 | 0.33 | 0.50 | 0.90 | 0.988 | 0.997 | 1 | 1 |
| `qp_border_touch` | 2216 | 0 | 0.007 | 0.032 | 0.062 | 0.126 | 0.342 | 0.57 |
| `crop_scale` | 2216 | 0.83 | 0.93 | 1.00 | 1.05 | 1.10 | 1.27 | 1.60 |
| `elong_ratio` | 2216 | 0.75 | 0.86 | 0.96 | 1.01 | 1.08 | 1.26 | 1.49 |
| `frag_diff` | 2216 | −4 | −2 | 0 | 0 | 1 | 4 | 7 |
| `area_ratio` | 2184 | 0.49 | 0.79 | 1.06 | 1.21 | 1.58 | 4.27 | 28.1 |

Notes:

- `lk_glare` is close to zero because block highlights in the released JPEGs are grey
  flecks (V ≈ 220–240) rather than saturated pixels. Glare looks like small bright
  specks and scratches, so it should be handled as texture noise, not as large
  saturated regions.
- Slide crop height is capped at 596 px, the preview height.
- The `area_ratio` tail above about 4 is almost entirely block-mask failure on faint
  biopsies, not real differences.

### 3.2 Fragment-count agreement (counts clipped at 5)

| LK \ QP | 1 | 2 | 3 | 4 | ≥5 |
|---|---:|---:|---:|---:|---:|
| 0 | 10 | 13 | 10 | 2 | 3 |
| 1 | 701 | 361 | 166 | 67 | 99 |
| 2 | 133 | 141 | 84 | 30 | 51 |
| 3 | 44 | 48 | 32 | 20 | 30 |
| 4 | 17 | 23 | 19 | 9 | 21 |
| ≥5 | 10 | 20 | 14 | 13 | 25 |

`frag_diff` is 0 for 40% of pairs, |diff| ≥ 3 for 14%, and **≤ −2 (block has more
pieces than slide) for 142 pairs**. Visual review shows that most of these 142 come
from slide-mask merging (large closing kernel) or fragments that look separate in the
block but lie close together on the slide. They are not omissions.

### 3.3 By specimen proxy

| group | n | % | median `lk_blur` | median `lk_contrast` | median `lk_tissue_frac` | median `area_ratio` | `frag_diff` < 0 | `lk_n_frag` = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| biopsy | 604 | 27.3 | 1.3 | 13.3 | 0.22 | 1.94 | 22.2% | 6.0% |
| resection | 1612 | 72.7 | 17.6 | 29.0 | 0.50 | 1.13 | 13.6% | 0.1% |

Biopsy crops are tiny (often < 100 px, 156 cases), upsampled-blurry and low-contrast.
Their block masks are much less reliable.

---

## 4. Qualitative observations

### 4.1 Block (LK) appearance
- **Wax colour varies by batch or lighting.** Grey-white is most common, but pink/mauve
  (Case1592, Case1655, Case1721, Case1747, Case1984) and bluish-grey (Case2068,
  Case2072, Case2095) also occur. Background must be estimated per image.
- **Tissue contrast ranges** from black (Case0510, Case0941, Case1061, Case1535: ink or
  pigment) through brown/red (blood) to near-wax pale fat (Case0097, Case0113,
  Case0159, Case0200, Case1300). Fat is the dominant failure mode in both modalities.
- **Some blocks show eosin-pink tissue** (Case0431, Case0571, Case0573, Case1351,
  Case0609), i.e. pre-stained biopsies. Tiny pink biopsies are the most blurred images
  in the set (Laplacian variance < 1).
- **Surface artefacts:** white wax flecks and scratches (almost every image), small
  bubbles/highlights (Case0431, Case0433), embedded hairs/fibres (Case0255, Case0389),
  and crystalline fat glints (Case1578, Case1659, Case1953).
- **Fused fragments:** adjacent pieces merge under the halo of tissue-tinted wax, for
  example Case0001 (8 slide pieces vs 1 block blob), Case0004–0006, Case0870 and
  Case1265 (two lymph nodes joined in the block). The block mask returns 1 component
  in 63% of cases.
- **Masks fail on** the 38 empty blocks (Case0431–0451 range, Case1471, Case1721,
  Case2065, Case2072) and on pink-wax images with faint tissue (Case1592, Case1747).

### 4.2 Slide (QP) appearance
- Tissue splits, tears and folds relative to the block: Case0002, Case0255 (fat around
  nodes disintegrates into lace), Case1416, Case0105 (an edge strip separates).
- **Pale fat or loose stroma** is barely visible: Case0113, Case0200, Case0627,
  Case0885, Case1243, Case1300, Case1301, Case1747.
- **The slide frame or coverslip edge appears inside some crops** as thin lines along
  the crop border (Case1013, Case1198, Case1300, Case0627, Case1171). These are what
  `qp_pen_frac` and `qp_border_touch` mostly pick up.
- **Incomplete face / partial sectioning:** the slide section is visibly smaller or
  differently shaped than the block face (Case2068, where oval block pieces appear as
  arrow-shaped sections; Case0994, where only a strip of the block face appears).
  Tissue is present but reduced, which is distinct from "missing".

### 4.3 Orientation
- About 90° rotation is very common: Case0166, Case1541, Case1825, Case2007, Case2182,
  Case2216 and Case0856 were all confirmed visually. 59% of elongated pairs swap
  orientation.
- 180° rotations and mirror images are plausible in many multi-fragment cases (e.g.
  Case0941, Case2030), and reflection must be assumed.

### 4.4 Multiple levels / ribbons (important)
- **Confirmed repeats:**
  - Case1950: two near-identical sections side by side.
  - Case0190: three sections, with the dark nodule repeated.
  - Case1365: two levels of a fragmented biopsy.
  - Case0074 and Case0182: second level partly truncated at the crop edge.
- **Possible:** Case0159, Case0843, Case0870, Case2053.
- **No clear repeats in the 58 random pairs**, so they are uncommon (~1–3% overall).
  They are enriched in wide slide crops (`qp_w / lk_long_side > 1.4`, 78 cases).
- The Hu-moment `qp_n_similar_pairs` proxy **does not work** for this. Its top hits
  (Case0495, Case1265, Case1351) are several similar lymph nodes or biopsy pieces, not
  repeated levels.
- **Implication:** the slide can contain *more* tissue instances than the block, so
  matching must allow many-to-one (slide → block) and report coverage of each block
  fragment by *any* level.

### 4.5 ROI / crop issues
- **Case0654:** the QP crop is a slide corner with **no tissue**, while the block shows
  a small pink biopsy. This is a detection or crop error in the release.
- **Low `crop_scale` (< 0.8, 14 cases)** usually means the QP crop covers less than the
  block: Case1072, Case1131, Case1561, Case1587, Case0449. Some of these are real
  omission candidates (§5).
- **64 slides at the 596 px height cap.** Their tissue spans the preview, and the crop
  can truncate tissue (Case0074, Case0182, Case1262).

### 4.6 Pen marks, labels, bubbles
- **No pen marks or labels** were seen in any inspected crop. Both modalities are
  already cropped to the tissue ROI, which excludes the slide label and cassette text.
  The pixels flagged by `qp_pen_frac` are frame/coverslip lines.
- Air bubbles on slides were rare at preview resolution (possible small ones on the
  right edge of Case1302).

---

## 5. Candidate real-omission cases

These are hypotheses for manual review at full resolution, not labels. Ordered by
confidence:

| Case | Observation | Alternative explanation |
|---|---|---|
| **Case1072** | Block: two well-separated similar-sized pieces. Slide: one piece only (`crop_scale` 0.68). | QP ROI crop dropped the second section |
| **Case1131** | Block: two pieces, a right piece plus a left piece with a thin dark rim. Slide: one piece (`crop_scale` 0.61). | ROI crop; or the left piece is wax/rim only |
| **Case1535** | Block: five large pieces (four inked black, one pale). Slide: four small fragments, much smaller than the block pieces. | Incomplete face (block not trimmed deep enough) |
| **Case1561** | Block: main cluster plus a separate dark fragment at the bottom. Slide: one pale piece (`crop_scale` 0.53). | ROI crop; pale tissue |
| **Case1481** | Block: large piece, pale strip, and a small dark nodule on the right. Slide: only the large piece and the strip. | Nodule is a blood clot / wax inclusion |
| **Case0389** | Block: the largest piece (dark, granular, lower right) has no clear slide counterpart of similar size. Slide: small nodes in pale fat. | Fatty piece sectioned pale; partial face |
| **Case0488 / Case0994** | Block shows extensive tissue with dark foci. Slide shows only a strip or a few foci, and most of the face is blank. | Fat sections poorly (very common in breast/fatty tissue) |
| **Case0654** | Slide crop contains no tissue. | Almost certainly a data/crop error |
| **Case0938, Case1222** | Block shows one or two more discrete small pieces than are visible on the slide. | Fragments merged or pale on the slide |

**Prior estimate.** Even if every listed case were real, true omissions would be below
about 0.5% of the dataset. Real positives are therefore far too rare to train on, which
confirms the synthetic-omission strategy in [PROBLEM.md](../PROBLEM.md). A dedicated
review of `crop_scale < 0.8`, biopsy cases with `frag_diff ≤ −2`, and the list above
is a cheap way to seed the real held-out set.

---

## 6. Implications for method design

1. **Use the scale prior.** Constrain registration and matching to a scale of about
   0.9–1.3 (slide relative to block). Treat crops outside 0.8–1.6 as "check crop"
   rather than forcing a fit.
2. **Search the full rotation range plus reflection** (O(2), not SO(2)). Do not
   assume the default orientation.
3. **Allow partial and many-to-one correspondence.** Block fragments may be fused
   (1 block → many slide), and slides may carry repeated levels (many slide → 1
   block). The missing-tissue decision should be **per block region coverage**
   ("is this block region explained by any slide tissue?"), not a fragment-count
   difference.
4. **Block segmentation needs a learned model or a robust prior.** Per-image wax
   estimation helps but fails on faint biopsies and fat. Options:
   - a small U-Net trained on pseudo-labels from the confident cases here;
   - interactive/SAM-style prompting;
   - outputting "block unreadable" (about 2% of cases, mostly tiny biopsies) instead
     of a decision.
5. **Detect pale tissue explicitly on slides** (texture plus slight saturation, or a
   learned tissue detector). Otherwise fatty specimens will produce false
   "missing" alarms.
6. **Report separate states:**
   - crop invalid (no tissue, or truncated at the 596 px cap or crop border);
   - incomplete face (all fragments present but smaller), so that it is not confused
     with a missing fragment;
   - true omission.
7. **Synthetic omissions should be realistic:**
   - remove whole slide fragments, not random blobs, with emphasis on small and
     peripheral fragments in multi-fragment biopsies;
   - generate hard negatives that mimic the observed splitting, lace-like
     disintegration of fat, ~90° rotations, flips, repeated levels and pale
     staining.
8. **Stratify evaluation** by `specimen_proxy`. Biopsies (27%) are the clinically
   important class for omissions, but also have the worst block image quality.

### Stratification heuristic

```
lk_long = max(lk_h, lk_w)            # px; ~23 px/mm at the shared preview scale
n_frag  = max(lk_n_frag, qp_n_frag)
biopsy  = lk_long < 250 or (lk_long < 400 and n_frag >= 4)
```

This labels 604 cases (27.3%) as biopsy, against 28.5% in the source study. Errors to
expect: large multi-fragment curettings fall under "resection", and small single
resections (e.g. skin punches) under "biopsy". For a finer split, add a
"multi-fragment" flag (`n_frag ≥ 3`) as a second stratification axis.

---

## 7. How to reproduce

```bash
# full statistics + plots (~1 min on 8 CPU cores)
python scripts/eda.py --data-dir data --out-dir results/eda

# contact sheet for chosen cases: rows of [LK | LK mask | QP | QP mask]
python scripts/eda.py --data-dir data --out-dir results/eda/inspect \
    --montage Case1072 Case1131 Case1535 Case1561 --montage-name candidates.png --montage-only
```

Key options: `--min-frag-frac` (default 0.002), `--min-small-frac` (0.0003),
`--workers`, `--limit`. Dependencies: numpy, opencv-python, pandas, matplotlib
(Python 3.14, CPU only).
