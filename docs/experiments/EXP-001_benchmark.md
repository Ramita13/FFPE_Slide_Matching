# EXP-001: Phase 2 benchmark (splits, mask-space omissions, harness) + EXP-000 sanity baseline

| | |
|---|---|
| **Date** | 2026-10-12 |
| **Status** | done: gate passed |
| **Code** | `scripts/make_splits.py`, `src/ffpe_match/{synth,evaluate,masks,segment}.py`, `scripts/render_synth_samples.py`, `experiments/exp000_area_ratio.py` |

## Goal
A cheap, reproducible benchmark that every pipeline (P1–P4 in
[00_synthesis](../research/00_synthesis.md)) is scored on, with a pass/fail gate:
*synthetic samples look realistic, and the task cannot be solved by tissue area alone.*

## Design

**Splits:** case-level 60/20/20 stratified by the EDA specimen proxy (seed 20261012):
train 1331 / val 443 / test 442. A 200-case `dev` subset of val is used for iteration.
The test split stays untouched until final evaluation. File: `splits/splits.csv`.

**Masks:** classical segmentation (`ffpe_match.segment`), cached in `cache/masks/` (gitignored).
Pairs where either mask is empty are reported as **unassessable**, not scored.

**Synthetic omissions (mask space, `ffpe_match.synth`):** deterministic from `crc32(case:idx)`.

| Stage | Applies to | Parameters |
|---|---|---|
| Whole-fragment removal | positives (p=0.5 if a removable fragment exists) | P ∝ 1/√area; fragments ≥ 50% of tissue never removed |
| Partial cut along irregular boundary | other positives | log-uniform 2–30% of tissue |
| Tear (split + pull apart) | **both labels**, p=0.4 | gap 1–4% of image size |
| Shrink ("present but smaller") | **both labels**, p=0.3 | 5–30% area loss of one fragment, never < 50% |
| Elastic warp | both | 0.5–2% amplitude |
| Rotation / flip / scale | both | 0–360°, p_flip 0.5, scale 0.90–1.25 (EDA crop-scale range) |
| Crop to bbox + margin, boundary noise | both | margin 2–5%, open/close r=1–2 |

**Tracks** (`ffpe_match.evaluate`):
- `self`: reference = clean slide mask. Tests matching logic only; exact localisation ground truth.
- `cross`: reference = **block** mask. Realistic setting including block-segmentation error.

**Metrics:** AUROC; sensitivity at 95% specificity; AUROC per missing-area bin
(0–2, 2–5, 5–10, 10–25, ≥25%) and per specimen type; localisation IoU (`self` track).

## Fixes found during visual review
- Slide glass/coverslip edges plus crop-border lines closed loops with tissue, and hole-filling
  flooded whole regions (Case0074 mask fraction 0.71 → 0.43). Added `straight_lines()` removal and
  capped hole-filling at 2% of image area.
- Fragment removal could delete 99.6% of tissue → capped at fragments < 50% of tissue.
- Tear gaps looked too wide → reduced from 2–6% to 1–4%.

## Results (dev subset, 199 assessable cases × 4 samples, `exp000_area_ratio`)

Score = 1 − area(query)/area(reference).

| Track | AUROC | Sens@95%spec | 0–2% | 2–5% | 5–10% | 10–25% | ≥25% |
|---|---|---|---|---|---|---|---|
| self | 0.644 | 0.25 | 0.53 | 0.56 | 0.64 | 0.73 | 0.91 |
| cross | 0.579 | 0.11 | 0.47 | 0.54 | 0.60 | 0.64 | 0.73 |

Positive mix: 143 whole-fragment (median 1.3% of tissue), 255 partial (median 7.0%).
Nuisance balance (neg/pos): tear 0.39/0.43, shrink 0.35/0.30. Unassessable: Case1587.

## Analysis
- Area alone is near chance below 5% missing area. That band is the clinically important one,
  so the benchmark rewards real correspondence rather than area bookkeeping. **Gate passed.**
- The `cross` track is worse because block and slide areas differ naturally. Matching on
  block masks will be bounded by block segmentation quality (Phase 3).

## Known limitations
- Very pale slides (e.g. Case0200) are under-segmented by the classical method → Phase 3 (GrandQC).
- Repeated sections on one slide (~1–3% of cases, e.g. Case0074): cutting one copy is not a true
  omission, so these introduce small label noise in positives.
- Mask-space only: no image texture. Appropriate for P1–P3; P4 (learned, image input) will need
  an image-space generator with the leakage controls in [04](../research/04_learned_and_synthetic.md).
- No real labelled omissions exist, and only the pre-cropped images are available
  (no uncropped thumbnails; confirmed by the data owner). Real-world validation will rely on manual review of flagged cases.

## Next
Phase 3: segmentation benchmark (slide: classical vs GrandQC; block: classical vs SAM-refined),
then Phase 4 P1 (rigid search + coverage) on this harness.
