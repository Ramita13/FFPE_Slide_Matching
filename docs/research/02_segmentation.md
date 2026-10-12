# 02 — Segmentation & fragment instance extraction

*Research track 2. Scope: getting tissue masks and fragment instances out of both block
photos (LK) and slide thumbnails (QP), on a CPU-only Windows / Python 3.14 machine.*

## Summary

- **Slides (QP) are easy at the semantic level but hard at the instance level.** Separating
  tissue from glass works with training-free methods, provided the threshold is low enough to
  keep pale fat and loose stroma. Otsu on saturation, the usual default, drops that tissue.
  The hard part is deciding which pieces belong to one fragment. Tears (Case0330), fat
  bridges (Case0001, Case0120) and floaters (Case2043, Case0638) make "connected component"
  an unreliable definition of a fragment.
- **Blocks (LK) are the weak link, as expected, for reasons that differ from the brief.**
  Glare is minor in the released JPEGs, which are tone-compressed with V ≤ ~215. The
  dominant problems are:
  1. **Spatially varying wax colour.** A global background model fails; a flat-field fit
     fixes it (Case0155).
  2. **Translucent halos**: tissue seen through the wax and soft edges, which fuse adjacent
     fragments (Case0001, Case0029, Case0981).
  3. **Tiny small-biopsy crops**: 311 of 2,218 blocks are under 150 px on a side, some only
     41×51 px (Case0654).
  4. **Pre-stained (eosin-tinted) biopsies.** Pink/red tissue breaks "beige tissue" colour
     priors (Case0595, Case1372, Case0638).
- **A classical baseline is strong and almost free.** It uses a flat-fielded Lab distance to
  wax plus a distance-transform watershed for blocks, and glass distance plus a texture gate
  for slides. It runs in 1–280 ms per image and gives plausible masks on 24 probe cases.
  Its main error is a core-versus-halo trade-off on blocks: it either fuses fragments or
  under-segments their pale margins.
- **MobileSAM works best as a refiner.** With a box and point prompt taken from each
  classical block instance, it recovers the pale margins the classical mask misses (Case0155,
  Case0223). The automatic mask generator, by contrast:
  - misses most low-contrast block fragments;
  - over-segments slides into histological parts;
  - is slow on CPU: 50–90 s per image at 16×16 points under load, against an encoder pass
    of about 1.5–15 s.

  Box prompts covering a whole slide crop fail.
- **TIAToolbox 2.1.3 runs on Python 3.14 with workarounds.** It needs a separate
  short-path venv and `--no-deps`, because it pins PyYAML 6.0.2 (no cp314 wheel) and
  conflicting numpy/opencv versions. It ships the **GrandQC tissue detector**.
  - **On slides, GrandQC at 2× upsampling** gives the best mask in the probe:
    IoU 0.89 vs classical, ~1 s per image. It keeps pale tissue and returns no tissue on
    ROI-failure crops.
  - **On blocks, GrandQC is empty.** Unstained tissue on wax is out of domain.
  - Its `OtsuTissueMasker` and `MorphologicalMasker` are plain grey Otsu (3–8 ms). They drop
    pale slide tissue and fuse block fragments.
  - `fcn-tissue_mask` and `efficientunet-tissue_mask` assume 2 and 8 mpp; thumbnails are
    ~40 µm/px. They took 20–40 s per image and returned empty masks.
- **Recommendation.** Use a hybrid pipeline:
  1. A classical or GrandQC semantic mask (slide).
  2. A classical flat-field mask with watershed proposals (block).
  3. MobileSAM/SAM2 box refinement on the block.
  4. Hierarchical, rather than flat, instance definitions on both sides: dense "cores"
     inside fat-connected "islands".
  5. Many-to-one matching downstream instead of hard pre-merging of slide pieces.

  Bootstrap labels from these outputs and train a small U-Net / YOLO11-seg for blocks only
  once about 200–300 corrected block masks exist.

## 1. Problem specifics observed in the data

I inspected about 30 pairs visually and ran the probe on 24. These are the specifics that
drive method choice:

| # | Observation | Cases | Consequence |
|---|---|---|---|
| 1 | **Crop sizes vary 40–740 px.** 311/2,218 blocks are < 150 px per side; small biopsies are ~50×50 px (Case0654 41×51, Case1413 48×51) | 0654, 1413, 1372, 595, 1442 | At 50 px a fragment is a few hundred pixels: no room for SAM's 1024-px encoder detail, and watershed is meaningless. Upsample before deep models, and use size-adaptive kernels. |
| 2 | **Wax background colour varies across the image** (ring-light fall-off, wax thickness, a yellow cast near tissue) | 0155, 0981, 0029 | A single global wax colour labels wax between fragments as tissue (Case0155, first run). Fitting a quadratic flat-field to wax pixels fixed it. |
| 3 | **Translucent halos around fragments.** A pale yellow/brown rim, likely tissue at or just below the face plus scattering, merges neighbouring fragments | 0001 (7–8 nodes look like one blob), 0029, 0981, 0330 | The threshold level decides the instance count: low → fused, high → split but too small. This is the central block-side difficulty. |
| 4 | **Some biopsies are pre-stained** (eosin/red) in the block | 0595, 1372, 0638, 0654 | Hue-specific "beige tissue" rules fail; use distance-from-wax rather than a tissue colour model. |
| 5 | **Glare is modest in the released images.** Measured glare (brighter than wax and near-neutral) is 0–7% of pixels; the worst are 1639 (6.6%) and 2108 (4.5%). There are also white crystal/fleck specks on the face | 1639, 2108, 0330, 0075, 0022 | Absolute V thresholds (> 235) find nothing; define glare relative to the wax. Treat glare as "unknown", fill it only when enclosed by tissue, or inpaint it before SAM. |
| 6 | **Very faint blocks**: contrast to wax ΔE ≈ 6–8 | 1442, 2043 | Otsu floor and guard needed; these are low-confidence cases to flag rather than decide. |
| 7 | **Slide: pale fat/loose stroma is real tissue but nearly glass-coloured** (10–20 grey levels darker) | 0001, 0120, 0260, 1107 | Otsu on saturation keeps only the dense cores. Including the pale tissue joins separate cores into one island (Case0001: 8 cores → 2 islands). |
| 8 | **Slide tears and cracks** split one fragment into parallel strips; the gaps are similar in size to true inter-fragment gaps | 0330 (3 slabs in block, 3–4 strips on slide), 0155 (top strip with cracks) | Proximity-based merging is ambiguous; merging belongs in the matcher (many-to-one), not as a fixed pre-processing rule. |
| 9 | **Slide floaters/debris and out-of-focus fragments** not obviously present in the block | 2043, 0638, 0260 | Unmatched slide tissue is expected noise. The QC question is one-directional (block → slide). |
| 10 | **ROI crop failures on QP**: the crop shows the slide corner with no tissue | 0654, 1372 | Need a QP sanity check (tissue fraction ≈ 0, or dark curved edge) so these route to "cannot assess", not "missing". |
| 11 | **Slide crop includes the slide edge/label strip** | 0260 | The dark-and-unsaturated artefact filter is needed. Edges still leak into the mask (thin line along the right border). |
| 12 | **Candidate real discrepancy**: the block shows a second, smaller fragment (lower left) that is not obviously separate on the slide (one lobulated mass plus a speck) | 0223 | Worth a manual look; a good seed for the real held-out set. |
| 13 | **Rotation, flip and scale differ** between crops | 0330 (block slabs horizontal, slide strips vertical), 0149 | Instance descriptors used for matching must be pose-invariant. |

Context: the slide side of the probe (§3) produces 1–10 pieces per case. A rough Otsu scan
of 317 slides (every 7th case) gave 1–15+ components, with a mode of 1–3 and a long tail
of 6–14.

## 2. Method comparison

Speeds are CPU timings at thumbnail size (~400–650 px). "Measured" means timed in this
repo's probes on this machine (8 logical cores; other agents were running, so treat these
as pessimistic). Other speeds are estimates from model size and published numbers.

| Method | Labels needed | CPU speed / image | Block (LK): expected strength / weakness | Slide (QP): expected strength / weakness | Py3.14 / Windows |
|---|---|---|---|---|---|
| **Global Otsu / HSV-S / grey threshold** | none | < 5 ms | − Low contrast, colour-varying wax, halos → fused or partial masks | + Dense tissue fine. − Drops pale fat; Otsu on grey is pulled by stain intensity | ✅ opencv |
| **Lab distance-to-wax + flat-field + Otsu (this repo)** | none | **measured 2–280 ms** (median ~40 ms) | + Handles pink pre-stained biopsies and wax colour drift; separates most fragments (0001 → 5, 0155 → 6, 0330 → 5). − Loses pale halos and margins (0120 fat, 0981 left ribbon); single threshold controls the fuse/split trade-off | n/a | ✅ |
| **Glass-distance + texture gate (this repo)** | none | **measured 1–80 ms** | n/a | + Keeps fat and loose stroma (0001, 0120, 0260). − Joins cores through fat; slide-edge leakage (0260); 3 strips → 3 pieces but 1 group (0330) | ✅ |
| **Colour deconvolution (Ruifrok–Johnston H/E/DAB)** | none | ~10 ms | n/a (no stain) | + H and E channels separate dense cellular cores from pale stroma/fat, giving a natural "core" level for the hierarchy; also lets you down-weight blood-only regions. − Stain vectors vary; preview JPEG colour is compressed | ✅ (numpy; skimage `rgb2hed` if installed) |
| **Morphology + distance-transform watershed (this repo)** | none | measured within the above | + Splits necks between convex blobs (0155, 0022 → 7). − Over-splits elongated or irregular fragments (0330 bottom slab → 3); cannot split fragments fused along a long edge | Same; better applied to "cores" than to fat-bridged islands | ✅ scipy |
| **HistoQC** (Janowczyk 2019) | none (config) | ~1–5 s per WSI; thumbnail-level steps are fast | Not applicable (assumes brightfield H&E on glass) | + Robust classical tissue mask plus artefact modules (pen, blur, folds). − Designed around OpenSlide WSIs, not JPEG crops; heavy config; you would reimplement the 2–3 relevant modules | ⚠️ installs, but pinned deps; plain JPEG input needs a workaround |
| **PathProfiler tissue segmentation** (Haghighat 2022) | pretrained (prostate WSIs, low-mag U-Net) | ~0.2–1 s | Not applicable | + Learned tissue/background at thumbnail resolution. − Single-organ training; less maintained; superseded by GrandQC | ⚠️ manual (repo scripts) |
| **TIAToolbox `OtsuTissueMasker`** | none | **measured 3 ms** | − Measured: one fused blob per multi-fragment block | − Measured: drops pale tissue (IoU 0.64 vs classical) | ✅ separate venv (tiatoolbox 2.1.3, `--no-deps`) |
| **TIAToolbox `MorphologicalMasker`** | none | **measured 5–8 ms** | Otsu + dilation + min-region filter; fusion unchanged | Holes filled and specks removed; pale-tissue problem remains (IoU 0.69) | ✅ separate venv |
| **TIAToolbox `fcn-tissue_mask` / `efficientunet-tissue_mask`** | pretrained (TIA internal data) | **measured 20–40 s** after upsampling to 2 / 8 mpp | Not applicable | − Empty masks on Case0001 with whole-image use; thumbnails are ~40 µm/px vs 2 / 8 mpp training. Not recommended | ✅ separate venv |
| **TIAToolbox `unet_tissue_mask_tsef`** | pretrained ("tissue excluding fat") | 1024² UNet-R50, ~seconds | Not applicable | Possible "core" extractor (fat excluded); untested | ✅ separate venv |
| **GrandQC tissue detector** (Weng 2024; in TIAToolbox as `grandqc_tissue_detection`) | pretrained (multi-centre WSIs, Dice 0.957 reported) | **measured 0.3 s (×1), 1.1 s (×2), 2.5 s (×4)** | − Measured: empty on unstained blocks (works only on pink pre-stained biopsies) | + **Measured best QP mask at ×2 upsampling** (IoU 0.89 vs classical; keeps pale tissue; detects ROI-failure crops). − ×1 envelopes neighbouring pieces; ×4 drops pale and blurry tissue. Semantic only | ✅ via TIAToolbox venv, or the GrandQC repo |
| **"Double-pass" classical tissue detection** (MDPI *Cancers* 2025) | none | ~ms | n/a | Reported mIoU 0.826 vs GrandQC 0.871, much faster on CPU: a good sanity baseline | ✅ (reimplement) |
| **SAM ViT-H/B automatic mask generator** | none | ViT-B encoder ~3–6 s, ViT-H ~15–30 s; plus 32×32 points decoding (tens of seconds) | − Misses low-contrast fragments; segments glare and specks | Over-segments into histological parts; needs heavy post-filtering | ✅ (torch) |
| **MobileSAM (TinyViT) — auto** | none | **measured**: encoder 1.4–15 s (contended), auto mode at 16×16 points **50–90 s** | − Measured: finds 3–17 masks, mostly dark sub-regions; misses pale fragments (0001, 0155) | Measured: part-level oversegmentation (0001: 16 masks, nodules separated nicely; 0330: strips split further); includes a glass mask | ✅ pip from GitHub + timm |
| **MobileSAM — box + point prompts from classical instances** | none | **measured** ~0.3–1 s per prompt after encoding | + **Best block masks in the probe**: grows the classical cores to the true fragment boundary (0155, 0223, 0001). − Inherits the classical instance count (cannot un-fuse); boxes that span several fragments return unions | − Fails when the box spans the whole crop (0001, 0330 groups → mask = everything); fine for piece-level boxes | ✅ |
| **SAM2 / SAM2.1 (Hiera-T/S)** | none | Hiera-T image encoder ~1–2 s CPU (est.) | Similar to SAM with slightly better boundaries; its video-propagation feature is irrelevant here | Same | ✅ (`sam2` pip, torch ≥ 2.5; may need `SAM2_BUILD_CUDA=0`) |
| **EfficientSAM / EfficientViT-SAM** | none | EfficientViT-SAM-L0 ~0.5–1 s CPU (est.) | Faster drop-in refiner; same failure modes as SAM | Same | ✅ torch (EfficientViT needs a few extra deps) |
| **SAM-HQ** | none | ≥ SAM ViT-B time | Sharper boundaries on thin structures; little benefit at 50–600 px | Better for thin fronds (0029) | ✅ |
| **MedSAM** | none (box prompt only) | ViT-B, ~3–6 s | Box-prompt refinement only; medical fine-tune is mostly radiology; no clear gain over SAM here | Same | ✅ |
| **PathoSAM** (Griebel 2025) | none | ViT-B/L/H | **Wrong scale**: trained for nuclei at 40×. Not useful for fragments | Not useful | ✅ |
| **SAM 3 (concept / text prompt "tissue")** | none | ~850M params; many seconds per image on CPU | Possibly better zero-shot "tissue piece" detection with text prompts; untested, heavy | Same | ⚠️ new, heavy |
| **FastSAM (YOLOv8-seg trained on SA-1B)** | none | ~0.2–0.5 s CPU | Fast "segment everything", but weak on low contrast | OK on slides | ✅ ultralytics |
| **DINOv2/v3 patch features + clustering / linear probe** | 0 / few (5–20 images) | ViT-S ~0.3–1 s | + Few-shot tissue-vs-wax linear probe could beat colour features on halos/blur. − 14-px patches = coarse masks at 50-px biopsies | + Few-shot fine; overkill vs classical | ✅ torch hub |
| **YOLO11-seg (n/s)** | **instance labels** (~200–500 images) | ~50–150 ms (n/s at 640) | + Learns instance splitting of fused fragments if annotated consistently. − Mask resolution coarse (160² proto), poor on tiny crops unless upsampled | Good, but classical already suffices | ✅ ultralytics (check torch 2.11 compatibility) |
| **Mask R-CNN (torchvision)** | instance labels | ~1–3 s (R50-FPN) | Strong instance separation with labels; slower | Same | ✅ torchvision |
| **U-Net / UNet++ semantic + boundary/"touching border" class + watershed** (DCAN / StarDist / Cellpose-flow style) | semantic + boundary labels (~100–300) | ~50–200 ms (small encoder) | **Best long-term option for fused fragments**: explicitly learn the thin wax gap / boundary class and split there | Same | ✅ (`segmentation_models_pytorch` pure-python + timm) |

Notes on the table:
- *Instance definition matters more than model choice on slides.* All semantic methods
  (classical, GrandQC, U-Net) give tissue/no-tissue. Instances come from connected
  components, so they depend on how pale tissue and tears are treated.
- *SAM-family models give "part" instances.* On slides, parts are often more consistent
  with what is visible in the block (dense cores) than CC-level islands are.

## 3. Probe experiments

### Probe 1: classical baseline, 24 cases
Script: [`scripts/seg_probe_classical.py`](../../scripts/seg_probe_classical.py) (numpy +
opencv + scipy only). Outputs: `results/seg_probe/CaseXXXX_seg.png` (four panels: block |
block instances with glare in cyan | slide | slide groups) and `results/seg_probe/summary.csv`.

Cases: 0001, 0022, 0029, 0036, 0075, 0120, 0155, 0223, 0260, 0330, 1107, 1639 (resections
and multi-fragment cases); 0595, 0654, 1372, 1413, 1442, 2043, 0638 (small biopsies,
41–77 px); and 0190, 0799, 1898, 2108, 0981 (from the high-component tail).

**Block pipeline:**
1. Gaussian blur.
2. Lab conversion.
3. Wax colour estimated from the 40% least-chromatic border pixels.
4. Glare = L > wax_L + 10 and chroma < wax_chroma + 6.
5. Weighted ΔE map (chroma weighted 2× over lightness).
6. Otsu with an absolute floor.
7. **Quadratic flat-field fit** of the wax colour on non-tissue pixels, with 2 re-threshold
   rounds.
8. Small closing (s/120), hole filling (glare is filled only if enclosed), opening, and
   removal of small components.
9. Distance-transform marker watershed (`scipy.ndimage.watershed_ift`; markers =
   dt ≥ 0.45·max per component).

**Slide pipeline:**
1. Glass colour = per-channel 90th percentile.
2. diff = max channel darkening; Otsu clamped to ≤ 14 grey levels.
3. A faint band (> 7 levels) is accepted only where local std > 3 (texture gate).
4. Dark-and-unsaturated artefact removal (slide edge, label).
5. Small closing (s/200), hole filling, opening, and removal of components < 0.15%.
6. **Proximity grouping** = connected components after dilation by 2% of the short side.

**Iteration history** (each change was verified visually):

| Version | Change | Effect |
|---|---|---|
| v1 | global wax colour, closing s/60; slide = Otsu(max(S, OD)) | Block: 0001 and 0330 fully fused into 1 instance; **0155 labelled wax as tissue**. Slide: pale fat missed (0001, 0120, 0260, 1107). OpenCV `cv2.watershed` (gradient priority) dropped about half the mask on 0155 → switched to `ndi.watershed_ift` on the distance map |
| v2 | slide = glass distance + texture gate; block closing s/120 | Slide fat recovered, but cores now join into islands (0001: 8 Otsu pieces → 2 islands); 0029 thin fronds kept |
| v3 | **block flat-field** | 0001: 1 → 5 instances; 0155: 1–2 → 6 (≈ the visible 5–6); 0330: 1 → 5–7 (3 slabs, over-split by watershed). Cost: pale tissue under-segmented (0120 fat, 0981 left ribbon, 1442) |
| v4 | slide closing s/80 → s/200 | 0330 strips stay as 3 pieces before grouping (grouping still merges them: gap ≈ 2% of the short side) |

**Per-case counts (v4)**, abridged from `summary.csv`:

| Case | Block instances (CC / watershed) | Slide pieces / groups | Visual judgement |
|---|---|---|---|
| 0001 | 5 / 5 | 3 / 2 | Block ≈ 6–7 fragments (2 still fused); slide 8 cores joined by fat |
| 0029 | 3 / 3 | 3 / 3 | Block 3 visible cores (≈ 3–4); slide thin fronds joined |
| 0120 | 3 / 3 | 10 / 4 | Block misses pale fat pieces; slide good |
| 0155 | 6 / 6 | 6 / 4 | Good on both |
| 0223 | 2 / 2 | 2 / 2 | Block: 2 fragments; slide: 1 mass + speck → **candidate omission or compression** |
| 0330 | 5 / 7 | 3 / 1 | Block over-split (3 slabs → 7); slide 3 strips merged by grouping |
| 0981 | 3 / 4 | 5 / 4 | Block misses the pale left ribbon |
| Small biopsies (0595, 1372, 1413, 0638) | 1 / 1 | 1 / 1 (0638: 3 incl. specks) | Good: pink pre-stain handled |
| 1442 | 1 / 1 | 1 / 1 | Block very faint (contrast 6.0): mask undersized |
| 0654, 1372 | 1 / 1 | — | **QP crop contains no tissue** (slide corner): ROI failure |

Timing: block 1.6–280 ms (larger images take longer; the flat-field lstsq dominates),
slide 1–80 ms. **The whole dataset would take about 2–3 minutes on one core.**

### Probe 2: TIAToolbox (Otsu / Morphological masker / GrandQC)
Script: [`scripts/seg_probe_tiatoolbox.py`](../../scripts/seg_probe_tiatoolbox.py). It runs
the same 24 cases next to the classical masks. Outputs: `results/seg_probe/tiatoolbox/`,
with per-case panels (image | classical | tia_otsu | tia_morph | grandqc ×1 | ×2 | ×4) and
`summary.csv` (tissue fraction, number of components, ms, IoU vs classical).

**Install on Python 3.14 / Windows (feasible, with workarounds).**
- `pip install tiatoolbox` fails: 2.1.3 pins `PyYAML==6.0.2`, which has no cp314 wheel, and
  the source build fails. It would also downgrade numpy and pull `opencv-python` 4.x next to
  the installed `opencv-python-headless` 5.
- Working recipe: a `--system-site-packages` venv at a **short path** (`.venv_tia/` in the repo
  root), `pip install --no-deps tiatoolbox`, then about 15 runtime deps (listed in the script
  docstring).
- Gotchas:
  - Inside the long temp directory, Windows MAX_PATH broke `transformers` and the numba cache.
  - Set `NUMBA_DISABLE_JIT=1`: umap/pynndescent are imported at package import time.
- The main environment is untouched.

**Pretrained tissue models shipped with TIAToolbox 2.1.3**, from `pretrained_model.yaml`:

| Name | Architecture | Training resolution | Notes |
|---|---|---|---|
| `grandqc_tissue_detection` | UNet++ / EfficientNet-B0 (GrandQC) | **10 mpp**, 512² patches | 2 classes; JPEG-q80 + ImageNet norm preproc |
| `efficientunet-tissue_mask` | EfficientUNet | 8 mpp, 512² | 1-class + 0.95 threshold |
| `fcn-tissue_mask` | UNet, ResNet-50 encoder | 2 mpp, 1024² in → 512² out | TIA internal data |
| `unet_tissue_mask_tsef` | UNet, ResNet-50, 3 classes | "baseline" | *Tissue segmentation excluding fat*: a possible "core" extractor, not tested |

**Results** (median over 24 cases; CPU timings under load from other jobs):

| Method | QP: IoU vs classical | QP: ms | LK: IoU vs classical | LK: ms | Qualitative |
|---|---|---|---|---|---|
| `OtsuTissueMasker` | 0.64 | 3 | 0.77 | 3 | QP: drops pale fat/stroma (0120, 0260, 0799), speckled fragments. LK: a single fused blob on every multi-fragment block (0001, 0155, 0330 → 1 component) |
| `MorphologicalMasker` (kernel = s/100 px, min region 0.15%) | 0.69 | 8 | 0.75 | 5 | Same as Otsu with holes filled; no help with fusion |
| GrandQC ×1 (native ~40 µm/px) | 0.87 | 330 | ≈ 0 | 290 | QP: high recall and smooth, but **envelopes neighbouring pieces with the glass between them** (0120, 0799 → one blob). LK: **empty** except pink pre-stained biopsies (0595, 0638, 1372, 0654) and dark 0223/1107 |
| **GrandQC ×2** | **0.89** | 1,100 | ≈ 0 | 1,180 | **Best QP masks in the probe**: keeps pale tissue (0260, 0029), separates pieces (0120), and correctly finds no tissue on the ROI-failure crop 1372 |
| GrandQC ×4 (≈ nominal 10 mpp) | 0.78 | 2,300–2,800 | ≈ 0 | 2,800 | Drops pale and out-of-focus tissue on blurry thumbnails (0036 → 3% vs 32%; 0799 partial). Matching the nominal mpp is *worse*: the preview images lack 10-mpp texture |
| `efficientunet-tissue_mask` ×5, `fcn-tissue_mask` ×4 (`--all-models`, Case0001 only) | — | 20,000–40,000 | — | 30,000 | Empty or near-empty masks with my whole-image integration. Possibly an integration error, but both are 20–100× slower than GrandQC and were not pursued |

**Verdict.**
- TIAToolbox's classical maskers are plain grey Otsu and add nothing over our baseline.
- **GrandQC at about 2× upsampling is a strong, label-free slide tissue model.** It agrees with
  the classical glass-distance mask at IoU ≈ 0.89, keeps pale tissue, handles ROI failures,
  and costs ~1 s per image on CPU. It is the recommended learned reference and fallback for
  QP.
- It does not transfer to block photos: wax and unstained tissue are out of domain.
- Its envelope behaviour at ×1 is a warning: learned tissue masks trained for WSI QC
  optimise area coverage, not fragment separation.

### Probe 3: MobileSAM on CPU, 6 cases
Script: [`scripts/seg_probe_mobilesam.py`](../../scripts/seg_probe_mobilesam.py).
Weights: `models/mobile_sam.pt` (40 MB). Outputs: `results/seg_probe/mobilesam/`. Each
panel shows image | classical | SAM auto | SAM box+point prompted, with timings in
`timings.csv`.

| Case/mod | Encoder (s) | Auto, pps = 16 (s) | Auto masks | Prompted (s) / prompts |
|---|---|---|---|---|
| 0001 LK | 1.4 | 61 | 12 | 31 / 5 |
| 0001 QP | 14.8 | 67 | 16 | 3.0 / 2 |
| 0155 LK | 4.2 | 62 | 12 | 5.3 / 6 |
| 0155 QP | 8.6 | 67 | 24 | 2.6 / 4 |
| 0330 LK | 4.7 | 50 | 17 | 4.2 / 7 |
| 0330 QP | 4.0 | 88 | 20 | 13.3 / 1 |
| 0223 LK / QP | 8.4 / 5.5 | 52 / 52 | 3 / 7 | 5.0 / 3.6 |
| 0029 LK / QP | 7.5 / 7.6 | 50 / 85 | 3 / 20 | 13.7 / 6.1 |
| 0120 LK / QP | 4.0 / 2.2 | 60 / 68 | 10 / 20 | 3.2 / 45 |

The machine was shared with other jobs, so the timings vary 10×. Uncontended, expect about
0.5–1.5 s per encoder pass and about 10–20 s for auto mode at 16×16 points.

**Qualitative results:**
- **Auto on blocks is poor.** On 0001, 0155 and 0330 it segments a few dark sub-regions and
  specks and misses most of the pale fragments. On 0223 its largest mask is the wax itself.
- **Auto on slides gives good part-level masks.** Each lymph-node core in 0001 becomes its
  own mask, and 0155's pieces separate well. But it over-splits strips (0330) and returns a
  glass mask that must be filtered (area > 60% or low-saturation mean).
- **Prompted refinement on blocks is the best block result in the probe.** The classical
  cores are grown to the full fragment extent, e.g. 0155's green and grey fragments and
  0223's lower-left fragment, which classical under-segments.
- **Prompted refinement on slides fails when the classical group box covers the whole
  crop** (0001, 0330): the mask becomes the full image. Prompts must be per-core or
  per-piece, never per-island.

## 4. Specific sub-problems

### 4.1 Separating fused fragments in block images
In order of cost:
1. **Flat-field the wax first** (probe v3). Most apparent fusion was really a background
   model error.
2. **Hysteresis / two-level masks.** Use high-threshold "cores" as instance seeds and a
   low-threshold "extent" mask as the region. Then use marker watershed on the ΔE map, not
   on the distance transform, so that boundaries follow the wax gaps (intensity valleys)
   rather than shape necks. The probe used distance-transform markers, which over-split
   elongated slabs (0330).
3. **SAM refinement per seed** (probe 3), with mutual-exclusion resolution between
   overlapping masks (assign contested pixels by SAM logit or by geodesic distance to the
   seed).
4. **Learned boundary class.** A U-Net with a 3-class output (tissue / boundary / wax), or a
   Cellpose-style flow model, trained on bootstrapped labels. This is the standard
   way to split touching objects (DCAN, StarDist, Cellpose) and the most promising route for
   halos.
5. **Use the slide as a prior** (cross-modal). After a global pose is found, the slide's core
   count and layout can seed block watershed markers. Caution: this risks hiding exactly the
   omission we want to detect, so use it only for splitting, never for adding or deleting
   tissue.

### 4.2 Glare
- In this release glare is limited: 0–7% of pixels, and the relative definition is needed
  because V ≤ ~215.
- Flag glare as **unknown**: exclude it from threshold statistics, fill it only when
  enclosed by tissue, and inpaint it with `cv2.inpaint` (Telea) before SAM or a CNN.
- White specks on the block face (paraffin crystals, scratches) behave like glare. A
  3–5 px median or opening handles them.
- If the acquisition protocol can ever change, cross-polarised illumination removes specular
  reflection at the source.

### 4.3 Consistent instance definitions across modalities
- Do not force a single flat definition. Build a **two-level hierarchy on both sides**:
  - **Islands** = connected tissue after a small closing, including pale fat. This is what
    the block's extent mask sees.
  - **Cores** = dense regions. On the slide, threshold the H/E-deconvolved OD or use
    SAM-auto parts. On the block, use the high-threshold seeds.
- Case0001 shows why: the block shows about 7 blobs that match the slide's 8 dense cores, not
  its 2 fat-connected islands.
- **Merge slide pieces in the matcher, not in segmentation.** Keep the over-segmented slide
  pieces and let the correspondence step assign many slide pieces to one block fragment
  (many-to-one / partial assignment). After a pose hypothesis, check the union's shape
  against the block fragment. Merging cues the matcher can use:
  1. Gap width relative to piece size.
  2. Complementary straight or parallel edges (tears: 0330, 0155).
  3. Shared pale-tissue bridges.
  4. Convex-hull overlap.
- **Area-coverage rather than counting.** The final decision ("missing?") is more robust as
  "block tissue area not covered by any matched slide piece after alignment" than as
  "#block instances ≠ #slide instances". Instance quality then matters mainly for pose
  estimation, and the semantic mask quality matters for the decision.
- **One-directional asymmetry.** Unmatched *slide* pieces (floaters, debris, 2043/0638
  specks) are not omissions. Unmatched *block* fragments are the target, so the block mask
  should err towards completeness: favour recall plus a confidence score.

## 5. Label bootstrapping plan

1. **Pseudo-labels at no cost.** Run classical-v4 on all 2,216 pairs, then MobileSAM
   box-refine the block instances (~1–2 h CPU in total).
2. **Disagreement-driven review (active learning).** Rank cases by disagreement between
   methods: classical vs SAM-refined IoU, block-vs-slide instance-count mismatch, and
   GrandQC vs classical on slides. Review the top 200–300 in CVAT or Label Studio (both
   support SAM-assisted polygon editing), stratified by size bucket (< 150 px biopsies vs
   resections).
3. **Train** a small UNet++ (EffNet-B0, `segmentation_models_pytorch`) with a
   tissue / boundary / wax head for blocks, and optionally YOLO11n-seg as an instance
   baseline. Both train on CPU in hours at 384 px.
4. **Iterate.** Re-rank by model uncertainty (entropy, MC-dropout) and add 100 labels per
   round.
5. The same reviewed masks double as the **evaluation set for segmentation**: report Dice,
   instance F1 and the count error per stratum.

## 6. Recommendation: ranked shortlist to benchmark

Benchmark on about 50 hand-annotated cases (25 resection, 25 small-biopsy), with instance
masks on both modalities.

| Rank | Pipeline | Why |
|---|---|---|
| 1 | **Slide:** glass-distance classical (v4) **vs** GrandQC tissue (TIAToolbox, **2× upsample**; ×1 envelopes pieces, ×4 drops pale tissue); cores from H&E deconvolution, `unet_tissue_mask_tsef`, or SAM-auto parts | Classical is ~10–80 ms; GrandQC ×2 is the robust learned reference (~1 s, IoU 0.89 with classical) and also flags ROI-failure crops. Pick per Dice and pale-tissue recall, or use their union/intersection as a confidence signal |
| 2 | **Block:** flat-field Lab ΔE classical + hysteresis seeds → **MobileSAM (or SAM2.1-tiny / EfficientViT-SAM) box+point refinement** | Best qualitative block result in the probe; no labels; ~1–5 s per image |
| 3 | **Block:** UNet++ (tissue / boundary / wax) trained on bootstrapped labels from #2 | Addresses fused fragments and halos directly; ~100 ms CPU inference |
| 4 | YOLO11n-seg instance model on the same labels | Fast instance baseline; tests whether direct instance learning beats semantic + watershed |
| 5 | DINOv2-S patch features + few-shot linear probe (tissue vs wax) | Cheap check that foundation features beat colour on faint blocks (1442, 2043) |
| — | Not recommended: SAM/MobileSAM auto mode on blocks; PathoSAM; TIAToolbox `fcn-`/`efficientunet-tissue_mask`; any H&E tissue model (GrandQC included) on blocks; HistoQC or PathProfiler as dependencies (GrandQC supersedes them) | Probe or scale evidence above |

Engineering notes:
- Keep the main environment free of TIAToolbox; run it from `.venv_tia/` (recipe in the
  `seg_probe_tiatoolbox.py` docstring). For production it may be simpler to vendor the
  GrandQC weights from HF `TIACentre/GrandQC_Tissue_Detection` and re-create the UNet++/B0
  with `segmentation_models_pytorch` in the main environment.
- MobileSAM installs fine on Py3.14: `pip install timm git+https://github.com/ChaoningZhang/MobileSAM.git`.
- Add a QP ROI-failure check (tissue fraction < 2% or curved dark edge) before matching.

## References

- Tang et al., *An AI framework for automated QC of paraffin block and slide consistency*, Virchows Archiv 2026. https://doi.org/10.1007/s00428-026-04535-w
- Otsu, *A threshold selection method from gray-level histograms*, IEEE TSMC 1979. https://doi.org/10.1109/TSMC.1979.4310076
- Ruifrok & Johnston, *Quantification of histochemical staining by color deconvolution*, Anal Quant Cytol Histol 2001. https://pubmed.ncbi.nlm.nih.gov/11531144/
- Janowczyk et al., *HistoQC: an open-source QC tool for digital pathology slides*, JCO CCI 2019. https://doi.org/10.1200/CCI.18.00157 · https://github.com/choosehappy/HistoQC
- Haghighat et al., *Automated quality assessment of large digitised histology cohorts by AI* (PathProfiler), Sci Rep 2022. https://doi.org/10.1038/s41598-022-08351-5 · https://github.com/MaryamHaghighat/PathProfiler
- Weng et al., *GrandQC: a comprehensive solution to quality control problem in digital pathology*, Nat Commun 2024. https://doi.org/10.1038/s41467-024-54769-y · https://github.com/cpath-ukk/grandqc
- Comparison of classical "double-pass" vs GrandQC tissue detection, *Cancers* 17(17):2918, 2025. https://www.mdpi.com/2072-6694/17/17/2918
- Pocock et al., *TIAToolbox as an end-to-end library for advanced tissue image analytics*, Commun Med 2022. https://doi.org/10.1038/s43856-022-00186-5 · pretrained models: https://tia-toolbox.readthedocs.io/en/latest/pretrained.html · GrandQC in TIAToolbox: https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.models.architecture.grandqc.GrandQCModel.html
- Kirillov et al., *Segment Anything*, ICCV 2023. https://arxiv.org/abs/2304.02643
- Ravi et al., *SAM 2: Segment Anything in Images and Videos*, 2024. https://arxiv.org/abs/2408.00714 · https://github.com/facebookresearch/sam2
- Carion et al., *SAM 3: Segment Anything with Concepts*, 2025. https://arxiv.org/abs/2511.16719
- Zhang et al., *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications* (MobileSAM), 2023. https://arxiv.org/abs/2306.14289 · https://github.com/ChaoningZhang/MobileSAM
- Xiong et al., *EfficientSAM*, CVPR 2024. https://arxiv.org/abs/2312.00863
- Zhang et al., *EfficientViT-SAM*, 2024. https://arxiv.org/abs/2402.05008
- Ke et al., *Segment Anything in High Quality* (SAM-HQ), NeurIPS 2023. https://arxiv.org/abs/2306.01567
- Ma et al., *Segment anything in medical images* (MedSAM), Nat Commun 2024. https://doi.org/10.1038/s41467-024-44824-z
- Griebel, Archit & Pape, *Segment Anything for Histopathology* (PathoSAM), MIDL 2025. https://arxiv.org/abs/2502.00408 · https://github.com/computational-cell-analytics/patho-sam
- Zhao et al., *Fast Segment Anything* (FastSAM), 2023. https://arxiv.org/abs/2306.12156
- Oquab et al., *DINOv2*, 2023. https://arxiv.org/abs/2304.07193
- He et al., *Mask R-CNN*, ICCV 2017. https://arxiv.org/abs/1703.06870
- Ronneberger et al., *U-Net*, MICCAI 2015. https://arxiv.org/abs/1505.04597 · Zhou et al., *UNet++*, 2018. https://arxiv.org/abs/1807.10165
- Ultralytics YOLO11 segmentation. https://docs.ultralytics.com/tasks/segment/
- Chen et al., *DCAN: deep contour-aware networks*, CVPR 2016. https://arxiv.org/abs/1604.02677 · Stringer et al., *Cellpose*, Nat Methods 2021. https://doi.org/10.1038/s41592-020-01018-x
- Beucher & Meyer, marker-controlled watershed, 1993. Implementation used: `scipy.ndimage.watershed_ift`.
