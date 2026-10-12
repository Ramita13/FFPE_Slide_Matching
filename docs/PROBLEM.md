# Problem Definition

## Task

**Input:** a block photograph `B` (LK) and a slide thumbnail `S` (QP) from the same case.
**Output:**
1. `missing ∈ {0,1}`: is any block tissue absent from the slide? (primary)
2. A localisation of the missing region(s) in `B` (secondary, for explainability)
3. A confidence score so the system can work as a screening tool at a chosen operating point

The intended use is a **second reader**: high sensitivity for missing tissue,
with a false-alarm rate low enough that technicians don't ignore it.

## Why it is hard

| Challenge | Effect |
|---|---|
| Cross-modality appearance (wax photo vs H&E scan) | Pixel/texture features don't transfer; shape is the most reliable shared signal |
| Unknown rotation / flip / scale | Need invariant descriptors or pose search; sections can be mounted mirrored |
| Non-rigid deformation (stretching, folding, tearing, shrinkage) | One block fragment can appear as several slide pieces, or the reverse |
| Low block contrast / glare / blur | Block segmentation is the weakest link; fused fragments |
| Partial sectioning (cut not deep enough, "incomplete face") | Tissue is present but smaller, which is not the same as missing |
| Small fragments matter most | Global similarity scores are dominated by the large pieces |
| No ground-truth labels | Must synthesise defects and/or manually annotate a test set |

## Known failure mode: forced alignment (CPD)

Coherent Point Drift (and most dense registration) assumes **both point sets
describe the same shape**. When a fragment is missing from the slide, CPD warps
the remaining points to cover the block's extra fragment instead of leaving it
unmatched. The missing tissue is absorbed into the deformation, so nothing is
flagged. Any approach we adopt must model **partial correspondence /
outliers explicitly**, so that "unmatched block tissue" is a first-class output.

## Evaluation protocol (draft)

Because the dataset has no omission labels:

1. **Synthetic omissions.** Remove one or more fragments (or a fraction of tissue
   area) from the *slide* image by inpainting with background, at controlled
   sizes (e.g. 1%, 5%, 10%, 25% of tissue area). Clean pairs are negatives.
   Also add **hard negatives that are not omissions**: tearing, folding,
   splitting and elastic warps, so the model doesn't learn "any difference = missing".
2. **Real held-out set.** Manually review a stratified sample (biopsy vs resection)
   to label genuine omissions and check the synthetic results transfer.
3. **Metrics:** AUROC, sensitivity at 95% specificity (and the reverse),
   stratified by missing-area fraction and specimen type; localisation IoU or
   pointing accuracy for the secondary output.
4. **Splits:** case-level splits fixed once (`splits/`), with no case leakage between synthetic variants.
