# Research Synthesis & Experimental Plan

*Phase 1 output. Summarises tracks [01](01_prior_work.md)–[05](05_dataset_eda.md) and fixes the
gated plan for Phases 2–6. Numbers below come from those reports; see them for details and references.*

## 1. Key conclusions

1. **The task is open.** No published method automatically detects block tissue missing
   from the slide. The closest are BlocDoc (human comparison), Clarapath patents
   (contour comparison, unvalidated), and Tang *et al.* 2026, which solves *case identity*
   with a single global embedding and cannot localise or see small fragments. ([01](01_prior_work.md))
2. **Omissions are small relative to the whole.** In multi-fragment slides the smallest
   fragment is a median 2.3% of tissue area (IQR 1.1–8.8%). Global similarity scores are the
   wrong tool. ([04](04_learned_and_synthetic.md))
3. **Fragment counts cannot be compared.** Fragments fuse in the block (63% look like a single
   blob) and tear on the slide; count correlation is ρ = 0.26. The right question is
   *"is every block region covered by slide tissue?"* ([05](05_dataset_eda.md))
4. **Geometry is constrained.** Scale between crops is ~1.05× (90% within 0.93–1.27×), but
   rotation is arbitrary and mirroring occurs. A rigid + flip + bounded-scale search is cheap. ([05](05_dataset_eda.md))
5. **CPD fails for a structural reason, and coverage fixes it.** CPD must explain every point and
   absorbs a missing fragment into the warp (only ~22% of missing area recovered, IoU 0.14 on toy data).
   Using *slide ⊆ block*, so that slide tissue must land on block tissue and block tissue may stay
   uncovered, gives an explicit missing map. Toy AUROC: rigid-per-piece + coverage **0.93**
   (IoU 0.90), partial GW 0.84, CPD 0.89 (wrong location). ([03](03_partial_matching.md))
6. **The main confounder is "present but smaller"** (different section depth). It caps toy
   sens@95%spec at ~0.5. A fragment-level decision (each block fragment matched or not)
   is needed to separate *missing* from *shrunk*. ([03](03_partial_matching.md))
7. **Block segmentation is the bottleneck.** Blurry (22%), low contrast, uneven wax colour,
   pale halos fuse fragments, and tiny biopsies (311 blocks < 150 px). Slides are mostly solved:
   GrandQC at 2× upsampling (via TIAToolbox) is the best slide model tested; classical works too.
   Pale fat on slides is a false-alarm trap. ([02](02_segmentation.md), [05](05_dataset_eda.md))
8. **Synthetic data must avoid shortcuts.** Image-space inpainting leaves traces (uniform JPEG
   setting, crop margins). Start in **mask space**; any image-space generator needs matched
   processing of negatives and leakage checks. ([04](04_learned_and_synthetic.md))
9. **The dataset has real limits.** Slide images are pre-cropped by the authors' detector, so
   an apparent omission can be a crop artefact (e.g. Case1072, Case1131: two block pieces, one
   slide piece). Some crops contain no tissue (Case0654). The system needs a third
   output: **"cannot assess"**. Real omissions are likely < 0.5% of cases. ([05](05_dataset_eda.md))

## 2. Candidate pipelines (ranked by cost-to-test)

| Rank | Pipeline | Stages | Output | Why |
|---|---|---|---|---|
| **P1** | Rigid-search + coverage | masks → global rigid/flip/scale search on masks → per-piece rigid refinement → uncovered block area (distance-transform tolerance) | score + missing map | Best on toy data; CPU-cheap; interpretable |
| **P2** | Fragment assignment with "missing" option | block regions × slide pieces → shape + area + layout cost → many-to-one assignment with dummy "missing" node | per-fragment missing/shrunk/present | Targets the "smaller vs missing" confounder |
| **P3** | Partial (fused) Gromov-Wasserstein | area-weighted points → partial GW, no pose search → unmatched block mass | score + soft map | Rotation/flip-invariant; fallback when P1 pose search fails |
| P4 | Learned matcher | frozen DINOv2/v3 tokens (cached) → small cross-attention head with "unmatched" output, trained on synthetic omissions | score + map | Only if P1–P3 plateau below target |
| ✗ | Plain CPD / global Siamese | — | — | Documented failures; kept only as reference baselines |

Segmentation candidates feeding these:
- **Slide:** classical (HSV/Lab + texture for fat) vs GrandQC ×2.
- **Block:** classical with flat-field correction vs classical-prompted MobileSAM/SAM2;
  later a small U-Net (tissue/boundary/wax) trained on corrected pseudo-labels.

## 3. Gated plan

| Phase | Work | Scale | Gate to continue |
|---|---|---|---|
| **2. Benchmark infra** | case-level splits; **mask-space** omission generator (remove 1 fragment / x% area; hard negatives: tear, split, shrink, elastic, flip); eval harness (AUROC, sens@95%spec, by area bin & specimen type, IoU) | ~200 cases | Synthetic samples pass visual review |
| **3. Segmentation** | benchmark slide & block segmenters; hand-correct ~50 block masks as a mini ground truth | ~100 cases | Block mask quality good enough that P1 on *predicted* masks isn't dominated by segmentation errors |
| **4. Matching** | P1 → P2 → P3 on predicted masks; CPD baseline for reference | val split | Any pipeline ≥ 0.90 AUROC on synthetic + sensible on real candidates → refine it, skip P4 |
| 5. Learned (conditional) | P4 with cached features | CPU | Beats best of P1–P3 on synthetic *and* real |
| 6. Real validation | review ~300–500 real pairs (flagged + random, weighted estimates); error analysis; write-up | — | — |

Each experiment is logged with [the template](../experiments/TEMPLATE.md) and has its
pass/fail criterion written *before* it runs.

## 4. Open questions for the data owner

- Are **uncropped slide thumbnails** (full 1712×596 previews) or local-lab slides available?
  They would let us tell real omissions from crop artefacts and build a trustworthy real test set.
- Case lists for manual review: Case1072, 1131, 1535, 1561, 1481, 0389, 0488, 0994, 0223.
