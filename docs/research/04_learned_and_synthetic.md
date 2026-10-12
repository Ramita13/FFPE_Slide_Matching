# 04 — Learned / end-to-end approaches and synthetic defect data

*Research track 4. Scope: learned architectures for "is any block tissue missing from the slide?",
a synthetic omission generator (needed because there are no labels), and an evaluation protocol.
Constraints: CPU only, Python 3.14 (torch 2.11.0+cpu, torchvision 0.26, OpenCV 5.0 are installed;
`timm` and `scikit-image` are not).*

---

## 1. Summary

- **The generator and its leakage controls matter more than the architecture.** With no
  labels, every learned model is trained and first evaluated on synthetic omissions. The main risk is
  a model that detects *editing traces* (inpainting texture, JPEG re-compression, crop margins)
  rather than missing tissue. The generator below is built so that every trace a positive carries
  also appears in negatives ("sham edits"). It also includes audits that measure leakage directly.
- **Standard change-detection networks (FC-Siam, BIT, ChangeFormer, SNUNet) cannot be used as-is.**
  They assume co-registered, same-modality image pairs. Here the pose is unknown (rotation, flip,
  mirrored mounting), the deformation is non-rigid and the two images come from different
  modalities. They work only after a **rigid or similarity pre-alignment that cannot absorb a
  missing fragment** (no CPD or other non-rigid warp; see PROBLEM.md). Alignment-free
  **cross-attention** models avoid this problem.
- **Recommended learned baseline:** take frozen, pre-extracted **DINOv2/v3 ViT-S dense tokens**
  from both images and feed them to a small **two-stream cross-attention transformer**. It has an
  image-level "missing" head and a per-block-token "unmatched" head. Run a **mask-only twin** of the
  same model as a control that has no artifacts. Both train in minutes per epoch on 8 CPU cores once
  features are cached.
- **Fragment-token matching (SuperGlue-style with a dustbin / partial optimal transport)** is the
  right structural prior: "unmatched" is a first-class output. It depends on block segmentation,
  which is the weakest link, so use region (over-segmented) tokens rather than fragment tokens.
- **Deprioritise:** CycleGAN-style modality translation (it can hallucinate or erase tissue, which is
  exactly the error we must detect, and it is slow on CPU). Also deprioritise pathology foundation
  models (UNI, CONCH, Virchow): they are trained on 20x histology patches, these images are macro
  photos and thumbnails, the models are large ViT-L/H networks and their weights are gated.
- **Real test set:** random sampling is inefficient if real omissions are rare (about 1,500 cases
  must be reviewed to find 30 positives at 2% prevalence). Use **score-enriched + random stratified
  sampling** with inverse-probability weights, and a simple side-by-side labelling app (about 10 s
  per pair).

### Dataset facts measured for this report (used to set generator parameters)

| Quantity (QP slides, n=2,216) | Value |
|---|---|
| Image size (median, h×w) | 414 × 478 px (5th–95th percentile: ~88 to ~580×745) |
| Tissue fraction of crop (saturation-Otsu mask) | median 0.28 (5th–95th percentile 0.07–0.57) |
| Connected tissue components (≥0.2% of image) | 1: 27%, 2: 23%, 3: 15%, 4: 10%, 5+: 25% |
| Smallest component share of tissue (multi-component slides, n=1,615) | median **2.3%**, quartiles 1.1%–8.8% |
| Background (non-tissue) pixel mean / std | 235.7 / 9.9 grey levels |
| Crop margin, tissue bbox to border, per side | median ~3–4%, 95th percentile ~15%, 5th percentile 0 |
| JPEG quantisation (mean luma table) | QP: 5.77 for every file (one encoder setting); LK: 17.3 (heavier compression) |
| QP/LK long-side ratio | median 1.05 (0.93–1.28) |

Visual inspection of 10 pairs (e.g. Case0001, 0007, 0150, 0500, 1200) confirms the following. Block
fragments are low-contrast and often fused (Case0001 shows about 8 blurry blobs in the block and 9
crisp fragments on the slide). Black ink can saturate the block (Case0007). Some blocks show a
pink-tinted background. Relative rotation of about 90° occurs (Case0150). Slides frequently show
torn or split tissue compared with the block. **Small fragments are typical, not exceptional**:
in half of the multi-fragment slides the smallest piece is ≤2.3% of the tissue. A global similarity
score will barely move when such a piece is lost.

---

## 2. Architecture options

CPU timings were measured on this machine (8 threads, batch 8–32, 224² RGB unless noted). They are
in ms per image:

| Model | Train (fwd+bwd) | Inference |
|---|---|---|
| MobileNetV3-S | 40 | 4.4 |
| ResNet-18 | 87 | 34 |
| ConvNeXt-Tiny | 245 | 60 |
| ViT-B/16 | 724 | 216 (so ViT-S/14 ≈ 50–60, estimated) |
| Tiny U-Net, w=32, 4-ch 128² masks | 84 | 27 |
| Tiny U-Net, w=32, 6-ch 256² RGB | 391 | 156 |

| # | Approach | How it applies here | Pros | Cons / risks | Label needs | CPU compute | Verdict |
|---|---|---|---|---|---|---|---|
| A | **Siamese / contrastive global embedding** (Tang et al. ConvNeXt-T) | One embedding per image; distance gives match/mismatch | Simple; prior work exists; robust to pose with augmentation | Global pooling is dominated by large fragments. A 2% fragment barely changes the embedding. No localisation. Trained for identity, not completeness | Synthetic pairs (or identity pairs) | Fine-tuning ConvNeXt-T: ~0.5 s/pair, ~2.8 h/epoch for 20k pairs (marginal). Frozen features + head: cheap | **Baseline only.** Use it as the "global score" reference that other models must beat on small-area strata |
| B | **Two-stream cross-attention transformer** on dense tokens (LoFTR-like self/cross layers) | Block tokens attend to slide tokens. Each block token predicts P(unmatched). Max/top-k pooling gives P(missing) | Alignment-free, so it handles rotation, flip and non-rigid deformation. Per-token output gives localisation. Small fragments get their own tokens | Needs pose augmentation (D4 + arbitrary rotation). Cross-modal token similarity depends on how good the frozen features are | Synthetic image-level labels; token labels when the generator knows block-space masks (see §3.6) | Cached ViT-S tokens (16×16 grid per image), 4 layers, d=256: ~5–10 ms/pair to train, about 5–8 min per 50k-pair epoch | **Primary recommendation** |
| C | **Bi-temporal change detection** (FC-Siam-diff/conc, SNUNet, BIT, ChangeFormer) | "What is in A but not in B" segmentation in block space | Mature architectures; dense output; FC-Siam is tiny | Needs co-registered inputs. The pre-alignment must be rigid or similarity only (plus D4 search), or a missing fragment is absorbed (CPD failure mode). Cross-modal: the literature on heterogeneous change detection (SAR↔optical) shows plain Siamese weight-sharing fails, so use pseudo-Siamese streams or code-aligned autoencoders | Synthetic masks in block space | Tiny U-Net on masks at 128²: 84 ms/img (w=32), ~25 ms (w=16). RGB 256²: 0.4 s/img | **Second option, mask input, after rigid alignment from the classical track.** Not ChangeFormer/BIT at full size on CPU |
| D | **Set-to-set fragment/region matching** (SuperGlue/LightGlue with a dustbin; Set Transformer; Sinkhorn / partial OT) | Tokens = fragments or regions with shape descriptors (area, Hu/Fourier, convexity) + pooled DINO features + position. Self- and cross-attention, then Sinkhorn with a dustbin. Mass = area, so partial OT allows one-to-many matching (splits and tears) | Explicit "unmatched" output with area accounting. Matches the problem structure. Very fast (dozens of tokens). Interpretable | Fused block fragments break one-to-one tokenisation. Needs good block segmentation. Use superpixel/region tokens (32–64 per block) to soften this | Synthetic; token labels come for free from mask-space generation | ms per pair; trivially CPU-feasible | **Strong option, ideally fused with B.** Shares the "region token + dustbin" head |
| E | **Anomaly detection** (PatchCore, DRAEM, CutPaste, PU learning) | Treat all 2,216 genuine pairs as "normal" (mostly true) and score anomalous pairs | Uses real pairs without synthesis; PU learning matches the "unlabelled, mostly negative" reality | Fires on *any* deviation (tears, folds, ink, glare), which is exactly the hard-negative problem. Cannot distinguish missing from torn. DRAEM/CutPaste are themselves synthetic-anomaly methods | None, or synthetic | Cheap with frozen features | **Use as a pre-screener** to enrich the real labelling set (§4.3), not as the detector |
| F | **Modality translation** (CycleGAN, CUT; block↔slide or stain↔wax) | Translate block to slide style, then compare in one modality | Removes the appearance gap in principle | Unpaired translators **hallucinate or delete structures** when distributions differ (Cohen et al. 2018). Deleting a fragment is the exact error to detect. ~1–2 s/iteration on CPU for 256² with 4 networks: days of training | None (unpaired) | Heavy | **Not recommended.** If used at all, translate *masks*, not RGB |
| G | **Foundation features: DINOv2/DINOv3** | Frozen dense patch tokens for B/C/D | Strong generic dense features; good for shape and boundaries; ViT-S (21M) is CPU-viable. DINOv3 improves dense features (Gram anchoring) | Not trained on wax or H&E thumbnails. Cross-modal token correspondence is unproven, so test it. DINOv3 weights need a licence click-through on Hugging Face | — | ViT-S/14 at 224: ~55 ms/img (estimated). Caching about 20k images takes about 20 min | **Use (ViT-S/B, frozen)** |
| G' | **CLIP / SigLIP** | Global or patch features | Robust global semantics | Weak dense localisation; semantics irrelevant here (tissue vs wax) | — | Similar | Low priority |
| G'' | **Pathology FMs: UNI, CONCH, Virchow** | Patch features | Excellent at 20x histology | Domain mismatch: slide *thumbnails* (~a few µm/px wider) and **block macro photos** are outside their training distribution. ViT-L/H costs ~1 s/img on CPU. Gated weights | — | Heavy | **Not recommended** for this task |
| H | **SAM / promptable segmentation** (auxiliary) | Better block fragment masks than thresholding | Helps tokenisation for D and mask inputs for C | ViT-B SAM is slow on CPU (~seconds per image), but it runs once offline | — | ~1–3 h offline for 2.2k blocks | Useful offline tool (see track on segmentation) |

### Design notes common to B, C and D

1. **Pose:** train with the full D4 group plus uniform rotation in [0°, 360°) and scale 0.8–1.25
   applied to the *slide only*, so the model cannot assume alignment. Sections can be mounted
   mirrored, so flips are mandatory.
2. **Asymmetry:** the question is asymmetric (block ⊄ slide). Cross-attention should be
   block→slide, and the head reads block tokens. Extra tissue on the slide (floaters, deeper levels)
   must not raise the score. Include "extra slide tissue" as a hard negative (N8).
3. **Pooling for small fragments:** use top-k or log-sum-exp pooling of per-token unmatched
   probabilities, not mean pooling. Mean pooling repeats Siamese model A's failure on small
   fragments.
4. **Area-aware output:** report the estimated unmatched block-tissue fraction alongside P(missing),
   so the operating point can be set per specimen type.
5. **Twin mask-only model:** the same architecture with binary or soft tissue masks as input (one
   channel per side, plus a distance transform). It has no access to texture, so its synthetic
   performance cannot come from inpainting artifacts. The gap between the RGB and mask models on the
   **real** labelled set shows whether appearance features help or only add shortcuts.

---

## 3. Synthetic defect generator — specification

Planned location: `scripts/synth_defects.py` (generation) and `src/ffpe_match/synth/` (ops). The
generator is deterministic given `(case_id, variant_idx, global_seed)`.

### 3.1 Principles

1. **Edit the slide, not the block, for the main positive class.** Removing tissue from a slide
   whose background is clean (std ≈ 10 grey levels) is easy to make realistic. "Block has tissue
   that the slide does not" is exactly the target.
2. **Every non-label operation is applied identically to positives and negatives.** This covers
   pose, stain, re-cropping, resizing, JPEG re-encoding and **sham inpainting**. The *only*
   label-correlated factor is that tissue was removed.
3. **Generate in mask space first, then render to RGB.** Every sample stores the slide tissue mask
   before and after the edit, so mask-input models (Option C/D) see exact, artifact-free inputs and
   the RGB path is an optional rendering step.
4. **Every case contributes both classes** (in train and test). Case-level difficulty and fragment
   count must not predict the label.
5. **Use a second positive generator with the opposite artifact signature.** Paste a fragment into
   the *block* (§3.5). Positives then sometimes carry traces on the slide side and sometimes on the
   block side, so a trace detector cannot work.

### 3.2 Pre-processing (once per case)

- **Slide tissue mask** `M_s`: HSV saturation, Otsu threshold with a floor of 20, open 3×3, close
  5×5, remove components smaller than 0.2% of image area. Fill holes smaller than 0.5% of tissue
  area. Store connected components `{C_i}` with area share `a_i = |C_i| / |M_s|`.
- **Slide background bank**: pixels with `dist_to_tissue > 15 px`. Extract 32×32 patches with no
  tissue (keep ≥ 20 per slide; if there are fewer, use a pooled bank from slides with similar
  background mean ±5).
- **Block tissue mask** `M_b` (needed only for localisation labels and block-side ops). Use the
  segmentation track's method. Keep a soft mask.
- **Empirical crop-margin distribution**: per-side margins measured from the real QP set (median
  ~3–4%, long tail to ~15%).

### 3.3 Label definition

`missing = 1` if the removed slide tissue area fraction `f = removed / |M_s|` is ≥ `f_min`
(default **0.5%**) **and** the removed region is a whole component or a contiguous chunk that is
≥ 25 px² at the working resolution. Otherwise `missing = 0`. **Area strata** (fraction of total
slide tissue):

| Stratum | f range | Notes |
|---|---|---|
| S1 | 0.5–1% | tiny satellite; hardest |
| S2 | 1–2% | |
| S3 | 2–5% | median smallest fragment is 2.3%, so this is the most important stratum |
| S4 | 5–10% | |
| S5 | 10–25% | |
| S6 | 25–50% | gross omission; sanity check |

Sample strata uniformly during training, with S1–S3 **oversampled ×2**.

### 3.4 Positive operations (slide side)

| Op | p (given positive) | Procedure | Parameters |
|---|---|---|---|
| **P1 whole-component removal** | 0.45 | Pick component(s) whose `a_i` falls in the target stratum. If none fits, pick the component closest to the target and re-assign the stratum | Only for slides with ≥2 components (73%) |
| **P2 multi-component removal** | 0.10 | Remove 2–3 components whose total share is in the stratum | ≥3 components |
| **P3 partial chunk removal** | 0.30 | Cut a fragment with a random smooth curve (cubic Bézier chord, endpoints on the boundary, curvature ≤ 0.3/extent). Remove the side with the target area | Required for single-component slides (27%). The chunk must touch the original fragment boundary for ≥ 15% of its perimeter (an edge piece, not an interior hole) |
| **P4 satellite emphasis** | 0.15 | P1 restricted to components with `a_i < 2%` | biopsy-like slides |

**Rendering an RGB removal** (mask `R` = removed tissue):
1. `R' = dilate(R, r)` with `r ~ U{3..6}` px at native resolution, restricted to `R ∪ (halo
   pixels within r of R that are not in the remaining tissue)`. This removes the antialiased
   "ghost outline".
2. **Fill** with real background. Tile random bank patches with 8-px feathered seams, then apply
   **Poisson blending** (`cv2.seamlessClone`, NORMAL_CLONE) into `R'` to match low-frequency
   illumination and vignetting. For `|R'| < 400 px²` use `cv2.inpaint(TELEA, radius 5)` plus
   Gaussian noise with σ equal to the local background std.
3. Do **not** use a generative inpainter (LaMa/SD). They hallucinate tissue-like texture, and
   their artifacts are learnable.

### 3.5 Positive operation (block side, alternative generator, p = 0.25 of all positives)

**B+ fragment paste:** take a segmented block fragment from a *different* training case (same
split). Scale it so its area share of the new block tissue falls in the stratum, give it a random
rotation, and place it in the wax background at least 5 px from the existing tissue. Blend with
Poisson MIXED_CLONE and alpha-feather 3 px. Match its colour to the host tissue mean (Reinhard in
Lab, strength 0.5–1.0). The slide is unchanged, so the label is 1 and the **block-space
localisation mask is exact**. Run the same paste machinery as a sham on negatives: paste a
background-only wax patch from another block.

### 3.6 Localisation labels

- Block side (B+): exact mask.
- Slide side (P1–P4): the removed region is known in *slide* coordinates. To get block-space labels,
  map it through the rigid alignment from the classical track when its confidence is high.
  Otherwise supervise only the image-level label (MIL) and the slide-space "absent region".
- **Self-pair pretraining (optional, recommended for C/D):** make a *pseudo-block* from the slide
  mask itself. Apply a random similarity transform plus D4, morphological closing (radius 3–10 px)
  to fuse nearby fragments like real blocks, a Gaussian-blurred boundary, and an optional
  wax-coloured rendering. Then remove tissue from the *slide* copy. This gives unlimited pairs with
  exact block-space masks for mask-input models. Fine-tune on real block/slide pairs afterwards.

### 3.7 Hard negatives (label 0) — slide side

Each negative applies 1–2 of these ops. Each positive *also* receives these ops with p = 0.5, so the
ops are uninformative.

| Op | p (given negative) | Procedure | Parameters |
|---|---|---|---|
| **N1 tear** | 0.15 | Split a component along a random polyline (3–6 vertices). Translate the pieces apart and rotate them; fill the crack with background | gap 2–15 px; rotation ±5°; area kept ≥ 98% |
| **N2 fold** | 0.12 | Reflect a strip across a random line inside the fragment; darken the overlap (H&E folds look darker) | strip width 5–20% of the fragment extent; overlap multiplied by 0.6–0.85 in RGB; *visible area falls by up to ~10%* |
| **N3 split / fragmentation** | 0.12 | Break one fragment into 2–4 pieces (Voronoi cells inside the fragment) and displace each piece | displacement 3–20 px; rotation ±10° |
| **N4 elastic + stretch** | 0.20 | Random displacement field (Simard) plus anisotropic affine transform | α = 20–40 px, σ = 8–12 px at ~500 px; per-axis scale 0.85–1.15; shear ±0.1 |
| **N5 partial-face shrink** | 0.12 | Erode the mask and add "lacy" holes from thresholded smooth noise inside the tissue | erosion 1–4 px; holes remove 1–5% of tissue; **never removes a whole component** |
| **N6 stain / scanner** | 0.30 | HED colour jitter, brightness, gamma, blur | HED σ = 0.03–0.05 per channel; brightness ±10%; gamma 0.8–1.2; Gaussian blur σ 0–1 |
| **N7 debris / bubble / pen** | 0.08 | Add small dark specks, air-bubble rings and coverslip-edge lines | specks 1–10, radius 1–4 px |
| **N8 extra slide tissue** | 0.08 | Paste a small fragment from another slide (floater or deeper level). Must **not** raise the score | 0.5–5% of tissue area |
| **N9 sham edit** | 0.25 | Run the exact §3.4 fill pipeline on a **tissue-free** region of matched size and shape (a random component mask moved to empty background) | Same size distribution as positives |
| **N0 clean** | 0.15 | Only global ops (§3.9) | — |

**Ambiguity note.** P3 (edge chunk removed) and N5 (partial face) can look alike. This mirrors a
real clinical ambiguity: an incomplete face can make a section smaller. Keep P3 at ≥ 5% of the
fragment with a clean, contiguous boundary, keep N5 diffuse, and **report P3 separately**. The real
labelling protocol should have an explicit "incomplete face" label (§4.3).

### 3.8 Block-side augmentations (both classes, same distribution)

Specular glare: 0–3 elliptical highlights, radius 3–15% of the image, peak brightness 235–255,
Gaussian edge σ 5–15 px. Defocus blur σ 0–2.5. Motion blur length 0–7 px. Contrast ×0.6–1.0.
Colour temperature ±800 K equivalent, plus wax hue shift toward pink (seen in real blocks) with
p = 0.15. JPEG quality 70–90. D4, rotation ±15°, scale 0.9–1.1.

### 3.9 Global post-processing (all samples, identical)

1. Slide pose: D4 plus uniform rotation 0–360° (pad with background bank) and scale 0.8–1.25.
2. **Re-crop** both images to the tissue bbox plus a margin sampled from the empirical per-side
   margin distribution. Without this, removing a border fragment leaves a large empty margin, which
   is a trivially learnable shortcut.
3. Random resize ×0.9–1.1 so JPEG 8×8 grids are not aligned with the originals. Re-encode the slide
   with the **original QP quantisation tables** (`PIL.Image.save(qtables=orig.quantization)`) and
   the block at quality 70–90. All samples go through this, including N0 clean, so double
   compression is uninformative.

### 3.10 Mixing, volumes and storage

- Training: 50% positives / 50% negatives. Positives are 75% slide-side and 25% block-side (B+).
  Generate on the fly for mask models. For RGB/feature models, **precompute K = 8 variants per
  training case**: about 1,550 × 8 ≈ 12.4k slides plus 4 block augmentations ≈ 6.2k blocks.
  Caching ViT-S tokens takes about 20 min on CPU.
- Frozen synthetic test set: per test case 1 clean, 1 sham, 3 hard negatives and up to 6
  positives (one per feasible stratum). With ~330 test cases that is about 3.5k samples.
  Written once with a manifest.
- Manifest (JSONL), one row per sample:
  `{case_id, split, variant, seed, label, generator: "slide"|"block", ops: [{name, params}],
  removed_area_frac, stratum, n_components, slide_mask_before, slide_mask_after, removed_mask,
  block_label_mask|null}`.

### 3.11 Leakage audits (must pass before reporting any synthetic number)

| Audit | Expected if clean | Action on failure |
|---|---|---|
| **Slide-only classifier** (same backbone, no block input) on positive vs negative | AUROC ≤ ~0.6 (some leakage from priors such as lower fragment count is unavoidable; quantify it) | Inspect Grad-CAM. Fix the fill, crop or JPEG step |
| **Sham vs clean** classifier | AUROC ≈ 0.5 | Fill pipeline is detectable; rework it |
| **Block-only classifier** | AUROC ≈ 0.5 (B+ excluded: ≈ 0.5 by construction) | Sampling bias across cases |
| **Generator ablation** | A model trained only on slide-side positives should still detect B+ positives, and vice versa | A large drop means it learned generator traces |
| **Mask twin vs RGB model** on real labelled pairs | RGB ≥ mask | RGB worse means shortcut learning |

---

## 4. Evaluation protocol

### 4.1 Splits

- Case-level split, fixed once and committed as `splits/split_v1.json`: 70/15/15 (about
  1,551/332/333 cases). Stratify by slide component count (1 / 2–4 / 5+) as a proxy for resection
  vs biopsy, since specimen type is not released. Use a seeded hash of the case ID.
- Every synthetic variant of a case inherits that case's split. B+ donor fragments come from the
  same split.
- The real labelled set (§4.3) is drawn **only from the test split**, plus extra test-only cases if
  more positives are needed. Never tune on it. Use a separate small real dev set from the
  validation split for threshold selection.

### 4.2 Metrics

- Primary: **AUROC** and **sensitivity at 95% specificity**. Also report specificity at 95%
  sensitivity, because a second reader needs high sensitivity.
- Report results **stratified by area (S1–S6)**, by op type (P1–P4, B+), by component-count stratum
  and by hard-negative type (false-positive rate per N-op). A model is useful only if S2–S3
  sensitivity is good. Gross-omission AUROC is not informative.
- Report the **false-positive rate on all 2,216 real pairs** (as unlabelled, mostly negative data)
  at the chosen threshold. This bounds the real alarm rate.
- Localisation: pointing accuracy (the argmax of the unmatched map falls inside the removed region,
  dilated by 5 px) and IoU for B+ and self-pair samples.
- Calibration: ECE on the synthetic test set, and reliability on the real set.
- Uncertainty: 95% CIs by **case-level bootstrap** (resample cases, not variants: 2,000 resamples).
  Use DeLong for paired AUROC comparisons.

### 4.3 Real labelled test set

**Problem.** Real omission prevalence is unknown and probably low. At 2% prevalence, a purely random
sample needs about 1,500 reviews to collect 30 positives.

**Design: enriched stratified sampling with known inclusion probabilities.**
1. Score all test-split pairs, and optionally all 2,216, with the cheap, diverse screeners from all
   tracks: area-ratio and component-count mismatch heuristics, the anomaly score (Option E) and
   early models.
2. Label (a) the **top ~150** by an ensemble rank, (b) a **uniform random ~300** from the rest,
   and (c) about 50 random multi-fragment (5+) cases, where omission risk is highest.
3. Estimate sensitivity and specificity with **inverse-probability (Horvitz–Thompson) weights**.
   The random stratum keeps the estimates unbiased. Report "positives found per 100 reviewed" for
   the enriched stratum.
4. Caveat: the enriched stratum was selected by the same family of models being evaluated. Evaluate
   the final model on the random stratum separately, and do not use the final model as a screener.

**Labels:** `complete`, `missing` (click the location in the block, plus a size estimate:
<2%, 2–10%, >10%), `incomplete_face` (tissue present but smaller), `ungradable` (glare or blur),
and `uncertain`. Have a histotechnologist or pathologist review all `missing`/`uncertain` pairs.
Double-label a 20% subset and report Cohen's κ.

**Tooling** (thumbnails are small, so speed matters more than features):
- Recommended: a ~100-line **Gradio or Streamlit** app (or a static HTML page) showing LK | QP side
  by side with rotate and flip buttons for the slide, a click-to-mark point on the block, keyboard
  shortcuts 1–5 for labels, and CSV output. Expect about 8–15 s per pair, so 500 pairs take about
  1.5–2 h.
- For polygon localisation: **Label Studio** (multi-image labelling config) or **CVAT**. **napari**
  works for local single-user mask touch-ups.
- Limitation: QP thumbnails are low-resolution, so only omissions visible at thumbnail scale can be
  labelled. Glass slides and full WSIs are not in the release.

### 4.4 Statistical power (Wilson 95% CIs; Hanley–McNeil SE)

| Quantity | n | 95% CI / SE |
|---|---|---|
| Sensitivity 0.90 | 20 / 30 / 50 / 100 positives | 0.70–0.97 / 0.74–0.97 / 0.79–0.96 / 0.83–0.95 |
| Specificity 0.95 | 200 / 300 / 500 / 1,000 negatives | 0.91–0.97 / 0.92–0.97 / 0.93–0.97 / 0.94–0.96 |
| AUROC 0.90 | (20 pos, 200 neg) / (30, 300) / (50, 300) / (100, 500) | SE 0.047 / 0.038 / 0.030 / 0.021 |
| Zero positives found among n random cases | 100 / 300 / 500 | prevalence ≤ 3% / 1% / 0.6% (rule of three) |

Consequences:
- With the likely 20–50 real positives, the real set can **confirm transfer** (e.g. a sensitivity
  CI lower bound above 0.7) and rank models with large differences. It **cannot** resolve small
  differences. Fine-grained comparison has to use the synthetic set; the real set is a gate.
- If the random stratum yields no positives, report the prevalence bound. That result is itself
  important for the clinical case.

---

## 5. Recommendation and plan

1. **Week 1: Generator v1, mask space plus RGB rendering** (§3), with manifest, frozen synthetic
   test set and all §3.11 audits. Commit `splits/split_v1.json`.
2. **Week 1–2: Baselines on the synthetic test set.**
   - Global Siamese (Option A, frozen DINOv2 ViT-S CLS or pooled ConvNeXt-T plus an MLP on
     |e_b − e_s|). This is the reference that shows the small-fragment weakness.
   - **Mask-only cross-attention twin** (B on masks; on-the-fly generation; minutes per epoch).
3. **Week 2–3: Main model** = frozen **DINOv2 ViT-S/14** (or DINOv3 ViT-S/16 if licence access is
   granted) dense tokens on 224–448 inputs, cached for K = 8 variants. Then a 4-layer block↔slide
   cross-attention transformer (d=256, 4 heads, ~3M params). Per-block-token unmatched head with
   top-k pooling, and an area-fraction regression head. Loss: BCE (image level) + BCE on tokens
   where labels exist + a ranking loss across strata. Training budget: < 1 h per run on 8 cores.
4. **Week 3–4: Region-token matcher** (Option D) with Sinkhorn + dustbin, using region tokens from
   the segmentation track. Fuse it with B via shared tokens, or ensemble the scores.
5. **In parallel: build the real labelled set** (§4.3) with early models as screeners. Use it as
   the go/no-go gate: sensitivity at 95% specificity on real positives with a bootstrap CI, and the
   RGB model compared with the mask twin.
6. **Do not invest in:** CycleGAN/CUT translation, pathology FMs (UNI/CONCH/Virchow), full
   fine-tuning of ConvNeXt or ViT on CPU, or full-size ChangeFormer/BIT. Revisit change-detection
   networks only as a mask-input U-Net after rigid (non-absorbing) alignment.

**Main open risks:** (i) synthetic-to-real transfer, mitigated by the sham edits, the two-sided
generator, the mask twin and the real gate. (ii) Block segmentation quality, which limits D and
the block-side localisation labels. (iii) Real prevalence may be so low that the real set mainly
measures false alarms. That is still decision-relevant for deployment as a second reader.

---

## 6. References

- Tang et al. (2026). AI framework for automated QC of paraffin block and slide consistency. *Virchows Archiv*. https://doi.org/10.1007/s00428-026-04535-w · code: https://github.com/tang777-feng/Siamese-Network-with-ConvNeXt-Tiny · data: https://doi.org/10.6084/m9.figshare.31451491
- Liu et al. (2022). A ConvNet for the 2020s (ConvNeXt). https://arxiv.org/abs/2201.03545
- Daudt, Le Saux, Boulch (2018). Fully Convolutional Siamese Networks for Change Detection. https://arxiv.org/abs/1810.08462
- Chen, Qi, Shi (2021). Remote Sensing Image Change Detection with Transformers (BIT). https://arxiv.org/abs/2103.00208
- Bandara & Patel (2022). A Transformer-Based Siamese Network for Change Detection (ChangeFormer). https://arxiv.org/abs/2201.01293
- Fang et al. (2021). SNUNet-CD. *IEEE GRSL*. https://doi.org/10.1109/LGRS.2021.3056416
- Luppino et al. (2020/2024). Code-Aligned Autoencoders for Unsupervised Change Detection in Multimodal Remote Sensing Images. https://arxiv.org/abs/2004.07011
- Sarlin et al. (2020). SuperGlue: Learning Feature Matching with Graph Neural Networks. https://arxiv.org/abs/1911.11763
- Lindenberger, Sarlin, Pollefeys (2023). LightGlue. https://arxiv.org/abs/2306.13643
- Sun et al. (2021). LoFTR: Detector-Free Local Feature Matching with Transformers. https://arxiv.org/abs/2104.00680
- Lee et al. (2019). Set Transformer. https://arxiv.org/abs/1810.00825
- Cuturi (2013). Sinkhorn Distances. https://arxiv.org/abs/1306.0895
- Chapel, Alaya, Gasso (2020). Partial Optimal Transport with Applications on Positive-Unlabeled Learning. https://arxiv.org/abs/2002.08276
- Myronenko & Song (2010). Point Set Registration: Coherent Point Drift. https://arxiv.org/abs/0905.2635
- Oquab et al. (2023). DINOv2. https://arxiv.org/abs/2304.07193
- Siméoni et al. (2025). DINOv3. https://arxiv.org/abs/2508.10104 · https://github.com/facebookresearch/dinov3
- Radford et al. (2021). CLIP. https://arxiv.org/abs/2103.00020
- Chen et al. (2024). UNI: Towards a General-Purpose Foundation Model for Computational Pathology. https://arxiv.org/abs/2308.15474
- Lu et al. (2024). CONCH: A Visual-Language Foundation Model for Computational Pathology. https://arxiv.org/abs/2307.12914
- Vorontsov et al. (2024). Virchow. https://arxiv.org/abs/2309.07778
- Kirillov et al. (2023). Segment Anything. https://arxiv.org/abs/2304.02643
- Zhu et al. (2017). CycleGAN. https://arxiv.org/abs/1703.10593
- Park et al. (2020). Contrastive Unpaired Translation (CUT). https://arxiv.org/abs/2007.15651
- Cohen, Luck, Honari (2018). Distribution Matching Losses Can Hallucinate Features in Medical Image Translation. https://arxiv.org/abs/1805.08841
- Roth et al. (2022). Towards Total Recall in Industrial Anomaly Detection (PatchCore). https://arxiv.org/abs/2106.08265
- Zavrtanik, Kristan, Skočaj (2021). DRAEM. https://arxiv.org/abs/2108.07610
- Li et al. (2021). CutPaste. https://arxiv.org/abs/2104.04015
- Schlüter et al. (2022). Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization (NSA). https://arxiv.org/abs/2109.15222
- Elkan & Noto (2008). Learning Classifiers from Only Positive and Unlabeled Data. https://doi.org/10.1145/1401890.1401920
- Ghiasi et al. (2021). Simple Copy-Paste Is a Strong Data Augmentation Method. https://arxiv.org/abs/2012.07177
- Pérez, Gangnet, Blake (2003). Poisson Image Editing. https://doi.org/10.1145/1201775.882269
- Suvorov et al. (2022). LaMa: Resolution-robust Large Mask Inpainting. https://arxiv.org/abs/2109.07161
- Simard, Steinkraus, Platt (2003). Best Practices for CNNs Applied to Visual Document Analysis (elastic deformation). https://doi.org/10.1109/ICDAR.2003.1227801
- Geirhos et al. (2020). Shortcut Learning in Deep Neural Networks. https://arxiv.org/abs/2004.07780
- Bianchi & Piva (2012). Image Forgery Localization via Block-Grained Analysis of JPEG Artifacts. https://doi.org/10.1109/TIFS.2012.2187516
- Schömig-Markiefka et al. (2021). Quality control stress test for deep learning-based diagnostic model in digital pathology. *Mod Pathol*. https://doi.org/10.1038/s41379-021-00859-x
- Janowczyk et al. (2019). HistoQC. *JCO CCI*. https://doi.org/10.1200/CCI.18.00157
- Tellez et al. (2019). Quantifying the effects of data augmentation and stain color normalization (HED jitter). https://arxiv.org/abs/1902.06543
- Hanley & McNeil (1982). The meaning and use of the area under a ROC curve. https://doi.org/10.1148/radiology.143.1.7063747
- DeLong, DeLong, Clarke-Pearson (1988). Comparing the areas under correlated ROC curves. https://doi.org/10.2307/2531595
- Annotation tools: Label Studio https://labelstud.io · CVAT https://github.com/cvat-ai/cvat · napari https://napari.org · Gradio https://www.gradio.app
