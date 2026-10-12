# 01 — Prior Work and Reference Implementation Review

*Research track 1. Compiled 2026-10-12. Sources were read through web search and page fetches. The reference repository was read on GitHub and was not cloned or run. Claims we could not check against a primary source are marked **[unverified]**. Statements that are our own inference are marked **[inference]**.*

---

## 1. Summary

- **No published method addresses our task.** We found no peer-reviewed method that automatically detects *which tissue fragment in a paraffin block is absent from the slide*. The closest work falls into three groups:
  1. **Identity verification.** Tang *et al.* 2026 is the only automated block↔slide study we found, and it answers "same case?", not "is anything missing?".
  2. **Imaging hardware with manual visual comparison.** SPOT Imaging BlocDoc, evaluated by L'Imperio *et al.* 2021, captures block images; a person then compares them with the slide.
  3. **Patents describing the concept.** Clarapath's "Facing and quality control in microtomy" family describes block-vs-section outline comparison for "incomplete tissue sample", but we found no published validation.
- **"Missing tissue" in commercial WSI QC is a different problem.** In products such as AIRAQc and AiosynQC, and in Atallah 2022 and Mutter 2022, the term means tissue that is on the glass but was not captured by the scanner. The reference there is the slide's macro image, not the block. These methods do not transfer directly, but they provide useful prevalence data and evaluation practice.
- **The reference code is a clean global-embedding baseline that cannot localise anything.** It pools a ConvNeXt-Tiny embedding globally, trains with a cosine contrastive loss plus in-batch hard negatives, and uses size-matched hard negatives. Its backbone and training scaffolding are reusable. Its objective is the wrong one for us: a global embedding is dominated by large fragments, which is exactly the failure mode in PROBLEM.md.
- **Registration benchmarks rely on full correspondence.** ANHIR, ACROBAT, VALIS and DeeperHistReg are strong references for cross-stain WSI alignment. All assume the two images show essentially the same tissue. Like CPD, they absorb missing content rather than reporting it.
- **The precedents we need come from partial matching.** The most relevant methods treat "unmatched" as an explicit outcome:
  - SuperGlue's dustbin
  - partial or unbalanced optimal transport
  - the explicit outlier component in CPD/BCPD, read out per point
  - Siamese change detection
  - block-face-to-histology registration in 3D reconstruction, where damaged sections are expected
- **Main recommendation.** Treat the task as **fragment-level partial assignment with an explicit "unmatched" state**, not as global similarity. Use the Tang repository only as (a) an identity and sanity pre-filter and (b) a source of pretrained features.

---

## 2. Reference implementation review: `tang777-feng/Siamese-Network-with-ConvNeXt-Tiny`

Repository: <https://github.com/tang777-feng/Siamese-Network-with-ConvNeXt-Tiny> (5 commits on `main` at time of review).
Paper: Tang L, Xie L, Yang X, Liu J, Zou X, Xia W. *An AI framework for automated quality control of paraffin block and slide consistency: a clinical evaluation and human–machine comparison study.* Virchows Arch 2026, doi:10.1007/s00428-026-04535-w, PMID 41999396.

### 2.1 Paper-level facts (from the Europe PMC abstract)

| Item | Value |
|---|---|
| Data | 2,220 retrospective block/slide pairs from the Second Affiliated Hospital of Soochow University. The public release we use has 2,216 complete pairs (see DATA.md). |
| Pipeline | YOLOv11 tissue detection, then a Siamese network for matching |
| Results | Mean AUC 0.9809 (5-fold CV); accuracy 95.05% on the independent test set vs 91.52% for manual checking |
| By specimen type | 98.01% accuracy on resections vs 88.77% on small biopsies ("fewer morphological landmarks") |
| Speed | < 0.05 s per slide, about 80× faster than humans |

The full text is paywalled, and we did not read it. All details below come from the code.

### 2.2 Repository contents

The repository contains `README.md`, `config.py`, `dataset.py`, `model.py`, `train.py`, `test.py` and `readme.py`. `readme.py` is a near-duplicate of the training script.

The following are **missing**:
- The YOLOv11 stage. The README says crops are produced beforehand.
- `requirements.txt` and `LICENSE`, although the README refers to both.
- Trained weights.
- Any reported numbers. The README's result figures are commented-out placeholders.

### 2.3 Architecture (`model.py`)

- **Backbone.** `torchvision.models.convnext_tiny` with ImageNet weights (`ConvNeXt_Tiny_Weights.DEFAULT`; `pretrained=True` by default). Only `.features` is kept, giving a 768-channel output.
- **Pooling.** `AdaptiveAvgPool2d(1)` produces a single 768-d global vector per image.
- **Projection head.** Linear(768→1024, no bias) → BatchNorm1d → ReLU → Linear(1024→512, no bias). The output is L2-normalised.
- **Siamese tower.** One shared tower serves both inputs: true weight sharing across block and slide, with no modality-specific layers.
- **Score.** Cosine similarity of the two 512-d embeddings.

### 2.4 Loss (`ContrastiveLossWithHardMining`, margin m = 0.5)

With s = cos(e₁, e₂) for each pair and label y ∈ {0,1}:

```
L_pos  = y       · (1 − s)²
L_neg  = (1 − y) · relu(s − m)²
L_hard = relu(max_{j≠i} (E1 · E2ᵀ)_ij − m)²      # in-batch, per row
L      = mean(L_pos + L_neg + 0.5 · L_hard)
```

**Issue we noticed [inference from reading the code].** Hard-negative mining treats every off-diagonal element of `E1·E2ᵀ` as a negative and does not consult `labels`. Two problems follow:
- Each block appears twice in the dataset, once in a positive pair and once in a negative pair. The *true match* of a row can therefore land off-diagonal in the same batch, where it gets pushed apart.
- For rows that are negative pairs, the diagonal, which is the actual negative, is excluded from mining.

Both effects add label noise to the hard term. This should be fixed before reusing the loss.

### 2.5 Data, preprocessing and negative pairs (`dataset.py`, `config.py`)

- **Splits** (`DataSplitter`). A fixed 20% case-level hold-out (`random_state=999`), then 5-fold `KFold` on the remaining 80% (`random_state=42`). Because the hold-out is fixed, there is **no test leakage across folds**.
- **Pairs.** Each case gives one positive pair (block, own slide) with label 1 and one negative pair with label 0. Classes are balanced 1:1.
- **Hard negatives** (`similar_neg_ratio = 0.8`):
  - With probability 0.8, sample up to 50 candidate slides from *other cases in the same split*. Rank them by L1 distance of raw image (width, height): `|w−w'| + |h−h'|`. Pick uniformly from the 5 closest.
  - Otherwise, pick a random slide from another case.
  - "Hardness" is therefore **crop-size similarity only**, with no appearance-based mining at the dataset level.
  - Python's global `random` is seeded with 31 (default).
- **Resize and pad.**
  - If either image's longest side exceeds 224, both are downscaled with `INTER_AREA` (aspect preserved). Smaller images are not upscaled.
  - Each image is then centre-padded to 224×224 with **black** (value 0).
  - At 224 px, small biopsy fragments shrink to a handful of pixels **[inference]**, which helps explain the 88.77% biopsy accuracy.
- **Augmentation** (Albumentations). HorizontalFlip, VerticalFlip and RandomRotate90 (each p = 0.5), then ImageNet normalisation. There is no colour, elastic or scale augmentation. Validation applies normalisation only.
- **Folder naming discrepancy.** The README states `QP/ = paraffin block crops` and `LK/ = slide crops`. Our DATA.md, which follows the paper's acquisition description, has the reverse: `LK` = block photograph and `QP` = slide scan. The pinyin reading supports DATA.md: *là kuài* (蜡块, wax block) → LK and *qiē piàn* (切片, section) → QP **[inference]**. The README label appears to be a documentation error. Because the network shares weights, this does not affect their results, but it matters for anyone reading their code.

### 2.6 Training and evaluation (`config.py`, `train.py`, `test.py`)

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4 |
| Schedule | CosineAnnealingLR, stepped per batch, `T_max = epochs × len(loader)`, `eta_min = 1e-6` |
| Epochs / batch | 40 / 16; fp32 (no AMP) |
| Model selection | Best validation AUC per fold → `best_model.pth` |
| Ensemble | Mean of the 5 fold models' cosine scores on the hold-out set |
| Threshold | Youden's J computed **on the test set itself** |
| Reported metrics | Accuracy, precision, recall, F1, AUC. The ensemble AUC is the mean of interpolated per-fold ROC curves. |

**Caveats [inference]:**
- Choosing the threshold on the test set makes the reported accuracy, precision, recall and F1 somewhat optimistic. AUC is unaffected.
- The validation accuracy in `train.py` thresholds raw cosine at 0.5.
- `evaluate()` silently returns (AUC 0.5, acc 0) on any exception.

### 2.7 Reusable vs not reusable for our task

| Component | Reusable? | Notes |
|---|---|---|
| Case-level splitting with a fixed hold-out | **Yes** | Same principle as our `splits/`. Synthetic variants must stay inside the same split. |
| ConvNeXt-Tiny Siamese trained on this exact dataset | **Partly** | Useful as a *pre-filter* ("is this even the right slide?") and as a domain-adapted feature extractor. We would have to retrain it, because no weights are released. |
| Global average pooling + single cosine score | **No** | It collapses everything to one vector. A missing small fragment barely moves the embedding, and the method cannot localise. We need spatial or per-fragment features. |
| Contrastive loss with in-batch hard negatives | **With fixes** | Mask hard negatives by case ID. The concept carries over: "omission vs benign deformation" is our fine-grained hard-negative problem. |
| Size-matched negatives | **Concept only** | Our hard negatives must be *same-case* perturbations (tear, fold, split, elastic warp) versus true omissions. |
| 224 px letterbox | **No** | Too coarse for small fragments. Use higher resolution or per-fragment crops. |
| Flip/rot90 augmentation | **Yes** | It reflects real mounting flips. Add colour, scale and elastic augmentation. |
| Youden-on-test threshold selection | **No** | Choose thresholds on validation and report sensitivity at a fixed specificity (PROBLEM.md). |

**License.** The README and badge declare **MIT**, but no `LICENSE` file is present in the repository listing (raw `LICENSE` returns 404). The intended license is therefore MIT, but strictly no license text has been granted. If we copy code rather than reimplement it, we should ask the authors to add the file. The dataset (figshare 10.6084/m9.figshare.31451491) has its own terms, which we could not retrieve (HTTP 403) **[unverified]**.

---

## 3. Related literature

### 3.1 Table

| # | Paper / source | Year | Task | Method | Relevance to us |
|---|---|---|---|---|---|
| 1 | Tang *et al.*, Virchows Arch | 2026 | Block↔slide **identity** verification | YOLOv11 ROI + Siamese ConvNeXt-Tiny, contrastive loss | Same dataset. A baseline and pre-filter, not an omission detector. Weaker on biopsies. |
| 2 | L'Imperio, Gibilisco, Fraggetta, *J Pathol Inform* 12:32 | 2021 | Block-face imaging at the microtome (BlocDoc), with **manual** block vs WSI comparison | Normal + polarised photograph of the cut block face, linked by barcode to the LIS and WSI | The only clinical study we found of block-vs-slide tissue concordance. 11,248 blocks imaged; "occasional" errors in 98 blocks (0.9%); near-systematic discordance in 3-D fragmented specimens (TURP, nasal polyps, morcellated myomas). This gives a prior on prevalence and on where to look. |
| 3 | Clarapath, US 11,435,268 B2 "Facing and quality control in microtomy" (+ family: 11,467,071; 11,609,162; 11,821,826; 11,898,948; 12,158,404; 12,405,194; 12,650,366) | priority 2020 | Automated microtomy QC | Image the block (e.g., under UV) and the section on tape/slide; segment by colour and intensity; compare size, shape and outline; flag "incomplete tissue sample" | The closest *conceptual* match to our task. Patent only; no published accuracy. Possible freedom-to-operate consideration for commercial use **[inference; not legal advice]**. |
| 4 | Fraggetta *et al.*, *J Pathol Inform* 9 | 2018 | WSI errors detected via the scanner macro image | Retrospective error review | Tissue-finder failures and poor scan coverage were *all* caught through the macro image. This supports comparing against an independent reference image. |
| 5 | Atallah *et al.*, *Mod Pathol* 35:903–910 | 2022 | Missing tissue in breast WSIs vs glass slides | Audit of 40,160 WSIs | Missing tissue in 2–19% of resections and 0% of cores, mostly fat at slide edges, with no diagnostic change. A good example of stratifying by specimen type and by "clinically relevant" omission. |
| 6 | Mutter *et al.*, *J Pathol Inform* 13:8 | 2022 | Scanner tissue dropouts | **Overlay of the glass-slide camera image onto the WSI** | Dropouts in 29.5% of 631 scans, mostly small "shards". A cross-modal overlay method for fragment omission, and the clearest evidence that small fragments are the common failure. |
| 7 | AIRAQc deployment (secondary report, reachmd.com; JPI paper not read) | 2026 | WSI QC incl. "missing tissue" | Deep-learning artefact segmentation, threshold-based routing | Deployed at scale (≈95k WSIs per month). "Missing tissue" here means scan coverage, not block omission **[numbers unverified]**. |
| 8 | Weng *et al.* GrandQC, *Nat Commun* 15 | 2024 | Tissue and artefact segmentation in WSIs | Segmentation CNN; open tool, test set and TCGA masks | A strong, open **slide-side tissue segmenter** (tissue Dice 0.957). It could replace ad-hoc QP thresholding. |
| 9 | Borovec *et al.* ANHIR, *IEEE TMI* | 2020 | Cross-stain non-rigid WSI registration | Challenge: 8 datasets, landmark rTRE | Top methods are coarse-robust init + multiresolution non-rigid. The evaluation design (landmarks, robustness = % landmarks improved) can be borrowed. They assume full overlap. |
| 10 | Weitz *et al.* ACROBAT 2022 (dataset arXiv 2211.13621; challenge paper 2024) | 2022–24 | IHC→H&E WSI registration, breast | Challenge, 8 algorithms; 4,212 WSIs / 1,153 patients | A large public cross-stain benchmark. Same full-overlap caveat. |
| 11 | Gatenbee *et al.* VALIS, *Nat Commun* 14:4502 | 2023 | Multi-slide WSI registration | Feature-based rigid + non-rigid, open source | Mature tooling for slide↔slide alignment. Not designed for wax photo ↔ H&E, and no explicit "unmatched" output. |
| 12 | Wodzinski *et al.* DeeperHistReg, arXiv 2404.14434 | 2024 | Robust WSI registration framework | Modular init + non-rigid, PyPI/Docker | As for VALIS. Usable as a dense refinement step *after* fragment correspondence. |
| 13 | STAR, arXiv 2509.02952 | 2025 | Rigid serial-WSI registration with QC | Reports robustness to *partial tissue overlap* on ANHIR/ACROBAT | One of the few registration papers that treats partial overlap explicitly. Worth reading in full. |
| 14 | ZeroReg3D, arXiv 2506.21923 | 2025 | Zero-shot 3D serial-section reconstruction | XFeat keypoints → affine → B-spline | Names "fragmentation or missing regions" as an open complication. Keypoint matching is training-free. |
| 15 | Feuerstein *et al.*, MICCAI | 2011 | 3D histology reconstruction using **block-face** images | Otsu contours, moment-based init, NMI registration of sections to block-face | The closest academic analogue (block-face ↔ section). Damaged sections are expected, and the block face acts as a damage-free reference. |
| 16 | Block-face + MRI-histology pipelines (JoVE 51325 / PMC4415371; Dryad dataset 10.5061/dryad.cnp5hqcgh; Gibson *et al.* JPI 2013) | 2013–2021 | Block-face as an intermediate reference for histology registration | Section-to-block-face registration | Shows that block-face images survive section damage and act as ground-truth geometry. Mostly 3-D and research-grade. |
| 17 | Rusu *et al.* RAPSODI, *Med Phys* | 2020 | Prostate MRI↔histopathology | 3-D reconstruction from histology, slice-to-slice registration | Macro↔micro registration with shrinkage, which is relevant for scale priors. |
| 18 | Schouten *et al.* PythoStitcher, *Sci Rep* | 2024 | Stitching tissue fragments into whole-mount sections | Fragment-pair fitting + genetic algorithm + per-fragment affine | Fragment-level combinatorial matching. Public fragment datasets on Zenodo (13786929). |
| 19 | TMA de-arraying (Wang *et al.* 2011 PLoS One; ATMAD 2018 BMC Bioinf) | 2011–18 | Mapping TMA cores to a design grid, with **missing cores** | Detection + TPS grid fit, explicit missing-core handling | The closest "expected set vs observed set, report absentees" setting in pathology. |
| 20 | Myronenko & Song, CPD, *TPAMI* (arXiv 0905.2635) | 2010 | Point-set registration | GMM centroids (moving set) fit to data, with a uniform outlier component of weight *w* | Explains our failure: outliers are modelled **only on the data side**, and every centroid must explain data. See §5. |
| 21 | Hirose, Bayesian CPD, *TPAMI* 43(7) | 2021 | Point-set registration | Bayesian CPD; geodesic variant TPAMI 2023 | The same caveat applies **[outlier model details not verified]**. |
| 22 | Sarlin *et al.* SuperGlue, CVPR (arXiv 1911.11763) | 2020 | Learned feature matching | GNN + differentiable OT with a **dustbin** for unmatched points | A direct template for "unmatched block fragment" as a first-class output. |
| 23 | Chapel *et al.*, Partial OT, NeurIPS (arXiv 2002.08276) | 2020 | Partial Wasserstein / Gromov-Wasserstein | Transport only part of the mass | Principled fragment-area matching in which un-transported block mass = missing tissue. Partial GW handles cross-modal feature spaces. |
| 24 | Daudt *et al.*, FC-Siamese change detection (arXiv 1810.08462) | 2018 | Pixel-wise change detection, co-registered images | Siamese FCN (concat/diff) | A template for a *localising* Siamese network once images are coarsely aligned. Omission = "change". |

### 3.2 Takeaways

1. **The Tang task and our task need different representations.** Identity tolerates losing small fragments. Omission detection is *about* small fragments. Both Tang (biopsies 88.77%) and the scanner-dropout literature (Mutter: mostly "shards"; Atallah: edge fat) point to small and peripheral fragments as the hard cases.
2. **Clinical practice uses a reference image plus human comparison.** BlocDoc and macro-image review both rely on a person. Automating that comparison is the open gap.
3. **Registration literature is mature but assumes full overlap.** We should use registration only *after* deciding correspondences, and only to refine.
4. **Partial-assignment formulations exist and are mature in computer vision.** These include the dustbin, partial OT, and outlier mixtures read out per point. They have not been applied to block↔slide QC.

---

## 4. Commercial and clinical systems

| System | Vendor | What it does | Automated comparison to slide? | Source quality |
|---|---|---|---|---|
| **BlocDoc** | SPOT Imaging | Images the cut block face at the microtome (normal + polarised; vendor also claims subsurface "uncut tissue" views). Barcode-linked to LIS/WSI. Vendor cites the CAP digital-pathology workflow question on tissue reaching the slide and WSI. | **No.** Visual comparison by technician or pathologist. | Peer-reviewed implementation study (L'Imperio 2021) + vendor pages |
| **Tissue Pattern Tracking** | SPOT Imaging | Vendor page on tracking tissue patterns block→slide | Unclear (page 404 at review time) **[unverified]** | Vendor |
| **SectionStar** | Clarapath | FDA-registered robotic microtomy. The patent family describes block/section image comparison, facing detection and "incomplete tissue" flags. | Claimed in patents. Whether the shipped product does this is **unverified**. | Patents + trade press (CLP Magazine) |
| **Tissue-Tek AutoSection** | Sakura Finetek | Automated microtome with AutoAlign/AutoTrim (senses block–blade relationship) | **No** block-face camera found in public material | Press releases, distributors |
| **AIRAQc** | (Sectra Amplifier listing; research-only there) | WSI artefact QC incl. "missing tissue" | Compares against the WSI and slide, not the block | Secondary news report + marketplace listing |
| **AiosynQC** | Aiosyn | WSI QC; "incomplete scan detection" (2025). A Radboudumc case study mentions *plans* to use scanner macro images to confirm WSI↔slide match. | WSI only, not the block | Vendor / trade press |
| **GrandQC** | Academic (open source) | Tissue + artefact segmentation | WSI only | Peer-reviewed |
| **Proscia Automated QC** | Proscia | WSI artefact QC on FFPE | WSI only | Vendor brief |
| Leica / Epredia | — | We found **no** commercial block-face imaging product from these vendors in public material | — | Negative search result |

**Implication.** The market has *capture* of block images (BlocDoc) and *slide-side* QC (AIRAQc, AiosynQC, GrandQC). Automated **block-to-slide tissue completeness** appears unaddressed except in Clarapath's patents.

---

## 5. Gaps and implications for our project

1. **Why CPD fails, and a cheap fix to try.** In CPD the moving set is the GMM centroids and the fixed set is the data. Only the data side has an outlier class (weight *w*), and all centroids share equal priors 1/M.
   - If the block is the *moving* set, the points of a missing fragment *must* be explained by slide points, which forces the warp.
   - If instead the **block is the data set**, the **slide is the moving set**, *w* is set high (e.g., 0.3–0.5), and the deformation is strongly regularised (high β/λ, or rigid/affine first), then the per-point outlier posterior on block points becomes a direct "missing" heat map. This is worth testing as a baseline before anything learned **[inference]**.
2. **Make "unmatched" a first-class output.** Prefer fragment-level partial assignment: block fragments vs slide fragments, solved by Hungarian matching with a dustbin or by partial OT on area-weighted fragments. Use shape and appearance descriptors that are invariant to rotation and flip. Report unmatched block area. SuperGlue and partial OT are the closest established formulations.
3. **The segmentation side is asymmetric.**
   - The slide (QP) can use an off-the-shelf tissue segmenter such as GrandQC.
   - The block (LK) is the weak link. No published block-photo segmenter exists that we found. Annotation or self-training on our data is needed, possibly bootstrapped from the slide masks through coarse alignment.
4. **Many-to-one is normal.** Tears and splits turn one block fragment into several slide fragments. PythoStitcher-style grouping, or OT with fragments allowed to split mass, is needed to avoid false "missing" calls.
5. **Hard negatives must be same-case.** Tang's negatives (other cases of similar crop size) teach identity. Ours must teach "deformed but complete" versus "incomplete": tears, folds, splits, elastic warps and partial-depth sectioning versus true removal. This is already planned in PROBLEM.md.
6. **Expected prevalence and stratification.** BlocDoc data suggests roughly 1% "occasional" omissions and near-universal issues in 3-D fragmented specimens. Biopsies and multi-fragment specimens should be the focus of the real review set. Expect a very low positive rate in random samples, so the review should be enriched for multi-fragment cases.
7. **Evaluation practice to borrow.**
   - From ANHIR/ACROBAT: report robustness alongside accuracy, and keep a held-out set with frozen thresholds.
   - From Atallah: stratify by whether the omission is clinically relevant (size, fat-only).
   - Avoid choosing thresholds on the test set, as Tang does.
8. **Legal and licensing.** Tang's code is MIT by declaration but has no LICENSE file; the dataset terms are unverified. The Clarapath patents cover block/section outline comparison for completeness. Check them if the project moves toward a product **[not legal advice]**.

### Strongest datasets and benchmarks

| Resource | Use for us |
|---|---|
| Tang 2026 / figshare 31451491 (our data) | The only block-photo↔H&E paired set we found. No omission labels. |
| ACROBAT (4,212 WSIs, SND 2022-190-1) and ANHIR | Pre-training or validation of cross-stain registration and features; landmark-based metrics |
| GrandQC test set + TCGA QC masks (Zenodo 14039591) | A slide-side tissue segmentation model and benchmark |
| PythoStitcher fragment datasets (Zenodo 13786929) | Testing fragment-level correspondence and stitching logic |
| Block-face/MRI-histology dataset (Dryad 10.5061/dryad.cnp5hqcgh) | The only public block-face ↔ histology pairs we found, though not clinical FFPE thumbnails **[suitability unverified]** |

### Baselines to implement

1. **Tang-style global Siamese score**, retrained on synthetic omissions. Expected to be weak on small fragments; it serves as the "global similarity" baseline.
2. **Area-ratio heuristic**: block tissue area vs slide tissue area after scale normalisation.
3. **CPD/BCPD with block as data, high *w*, and per-point outlier posterior** (§5.1).
4. **Fragment-level matching with a dustbin or partial OT** on segmented fragments. This is the proposed main direction.
5. **Change-detection Siamese network** (FC-Siam-diff style) applied after coarse alignment, for localisation.

**Metrics**: AUROC, sensitivity at 95% specificity, stratified by missing-area fraction and specimen type (biopsy vs resection), plus a localisation metric (pointing accuracy or IoU), as in PROBLEM.md. Add robustness, meaning the % of pairs where alignment succeeds, in the ANHIR sense.

---

## 6. References

1. Tang L, Xie L, Yang X, Liu J, Zou X, Xia W. An AI framework for automated quality control of paraffin block and slide consistency: a clinical evaluation and human–machine comparison study. *Virchows Arch* 2026. <https://doi.org/10.1007/s00428-026-04535-w> (abstract via Europe PMC, PMID 41999396)
2. Reference code: <https://github.com/tang777-feng/Siamese-Network-with-ConvNeXt-Tiny> — files `model.py`, `dataset.py`, `config.py`, `train.py`, `test.py` at <https://raw.githubusercontent.com/tang777-feng/Siamese-Network-with-ConvNeXt-Tiny/main/>
3. Dataset: <https://doi.org/10.6084/m9.figshare.31451491>
4. L'Imperio V, Gibilisco F, Fraggetta F. What is essential is (no more) invisible to the eyes: the introduction of BlocDoc in the digital pathology workflow. *J Pathol Inform* 2021;12:32. <https://pmc.ncbi.nlm.nih.gov/articles/PMC8529340>
5. SPOT Imaging, BlocDoc product pages. <https://spotimaging.com/sample-imaging-blocdoc/>, <https://beta.spotimaging.com/block-doc/> (vendor material)
6. Mitra PP, Sewkarran J, Yagci B, Pinto JR (Clarapath Inc.). Facing and quality control in microtomy. US 11,435,268 B2 (2022). <https://patents.google.com/patent/US11435268B2/en>
7. Clarapath SectionStar update (trade press). <https://clpmag.com/lab-essentials/lab-automation/clarapath-updates-sectionstar-robotic-microtomy-platform/>
8. Fraggetta F, *et al.* The importance of eSlide macro images for primary diagnosis with whole slide imaging. *J Pathol Inform* 2018;9. <https://doaj.org/article/f0e2e03eb1874504812fa7d483fcb725>
9. Atallah NM, *et al.* Potential quality pitfalls of digitalized whole slide image of breast pathology in routine practice. *Mod Pathol* 2022;35:903–910. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8711290/>
10. Mutter GL, Milstone DS, Hwang DH, Siegmund S, Bruce A. Measuring digital pathology throughput and tissue dropouts. *J Pathol Inform* 2022;13:8. doi:10.4103/jpi.jpi_5_21. <https://doaj.org/article/3691d4839b5e46ff864f2d72a859a9c7>
11. AIRAQc deployment report (secondary). <https://reachmd.com/news/ai-quality-control-deployment-reshapes-whole-slide-workflow/2487222/>; Sectra Amplifier listing <https://amplifiermarketplace.sectra.com/pathology/airaqc/>
12. Aiosyn incomplete-scan detection. <https://tissuepathology.com/2025/04/11/aiosyn-expands-ai-powered-quality-control-with-incomplete-scan-detection/>; Radboudumc case study <https://tissuepathology.com/2024/12/31/aiosyn-case-study-automating-radboudumcs-digital-slide-quality-control-with-aiosynqc/>
13. Weng Z, *et al.* GrandQC: a comprehensive solution to quality control problem in digital pathology. *Nat Commun* 2024;15. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11649692/>; data <https://zenodo.org/records/14039591>
14. Borovec J, *et al.* ANHIR: Automatic Non-Rigid Histological Image Registration Challenge. *IEEE TMI* 2020. doi:10.1109/TMI.2020.2986331. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7584382>
15. Weitz P, *et al.* ACROBAT — a multi-stain breast cancer histological whole-slide-image data set. arXiv 2211.13621. <https://arxiv.org/abs/2211.13621>; challenge paper <https://www.utupub.fi/items/64402d00-e70d-4ac2-908c-6b81bcca08c1/full>; data <https://snd.gu.se/en/catalogue/dataset/2022-190-1>
16. Gatenbee CD, *et al.* Virtual alignment of pathology image series for multi-gigapixel whole slide images. *Nat Commun* 2023;14:4502. <https://pubmed.ncbi.nlm.nih.gov/37495577/>
17. Wodzinski M, Marini N, Atzori M, Müller H. DeeperHistReg: Robust whole slide images registration framework. arXiv 2404.14434. <https://arxiv.org/abs/2404.14434>
18. STAR: A fast and robust rigid registration framework for serial histopathological images. arXiv 2509.02952. <https://arxiv.org/html/2509.02952v1>
19. ZeroReg3D: A zero-shot registration pipeline for 3D consecutive histopathology image reconstruction. arXiv 2506.21923. <https://arxiv.org/pdf/2506.21923>
20. Feuerstein M, *et al.* Reconstruction of 3-D histology images by simultaneous deformable registration. MICCAI 2011. <https://campar.in.tum.de/pub/feuerste2011miccai/feuerste2011miccai.pdf>
21. Block-face imaging for histology reconstruction: JoVE <https://www.jove.com/t/51325>; <https://pmc.ncbi.nlm.nih.gov/articles/PMC4415371>; Dryad dataset "Precise MRI-histology coregistration of paraffin-embedded tissue with blockface imaging" <https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqcgh>; Gibson E, *et al.* *J Pathol Inform* 2013 <https://www.slicer.org/w/img_auth.php/1/13/Gibson-JPatholInform2013.pdf>
22. Rusu M, *et al.* Registration of presurgical MRI and histopathology images from radical prostatectomy via RAPSODI. *Med Phys* 2020. doi:10.1002/mp.14337. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7586964/>
23. Schouten D, *et al.* Full resolution reconstruction of whole-mount sections from digitized individual tissue fragments. *Sci Rep* 2024. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10794243/>; data <https://zenodo.org/records/13786929>
24. Wang Y, *et al.* A TMA de-arraying method for high throughput biomarker discovery in tissue research. *PLoS One* 2011. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3189244/>; ATMAD (BMC Bioinformatics 2018) <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2111-8/figures/2>
25. Myronenko A, Song X. Point set registration: Coherent Point Drift. *IEEE TPAMI* 2010. <https://arxiv.org/abs/0905.2635>
26. Hirose O. A Bayesian formulation of coherent point drift. *IEEE TPAMI* 2021;43(7):2269–2286. doi:10.1109/TPAMI.2020.2971687
27. Sarlin P-E, DeTone D, Malisiewicz T, Rabinovich A. SuperGlue: Learning feature matching with graph neural networks. CVPR 2020. <https://arxiv.org/abs/1911.11763>
28. Chapel L, Alaya MZ, Gasso G. Partial optimal transport with applications on positive-unlabeled learning. NeurIPS 2020. <https://arxiv.org/abs/2002.08276>
29. Daudt RC, Le Saux B, Boulch A. Fully convolutional Siamese networks for change detection. 2018. <https://arxiv.org/abs/1810.08462>
30. Sakura Finetek Tissue-Tek AutoSection (product announcement). <https://thepathologist.com/product-profiles/2014/sakura-finetek-introduces-the-world-s-first-automated-and-programmable-microtome>
