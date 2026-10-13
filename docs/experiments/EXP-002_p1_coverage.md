# EXP-002: P1 — rigid pose search + per-piece refinement + coverage

| | |
|---|---|
| **Date** | 2026-10-12 → paused 2026-10-13 |
| **Status** | in progress (paused): self track done; cross track / segmenter comparison pending |
| **Code** | `src/ffpe_match/match_coverage.py`, `experiments/exp002_p1_coverage.py` |

## Hypothesis
If slide tissue is treated as a subset of block tissue (every slide pixel must land on block
tissue, block tissue may stay uncovered), the uncovered block area after a rigid alignment is a
direct missing-tissue signal. Unlike CPD, nothing forces the missing region to be "explained".
The toy study ([03](../research/03_partial_matching.md)) gave AUROC 0.93 with perfect masks.

## Method
1. Downsample both masks by a common factor (longest reference side = `WORK` px).
2. Global search: 36 rotations (10°) × flip × 5 scales (0.85–1.25); translation by FFT
   correlation. Objective `J = overlap − λ·(query on background)`, λ = 1. Local refinement ±8°, ±0.05 scale.
3. Per-piece refinement: each query piece gets ±15° and a local translation search (many-to-one allowed).
4. Missing map = reference − dilate(placed query, `tol`), opened with radius 0.5·`tol`.
   Score = largest missing component / reference area.

## Results (dev: 199 cases × 4 samples, `self` track)

| WORK | tol | AUROC | Sens@95%spec | Loc IoU | 0–2% | 2–5% | 5–10% | 10–25% | ≥25% | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — (area-ratio baseline, EXP-000) | 0.644 | 0.25 | — | 0.53 | 0.56 | 0.64 | 0.73 | 0.91 | 40 s |
| **160** | **0.02** | **0.858** | **0.45** | 0.47 | 0.69 | 0.80 | 0.92 | 0.97 | 0.99 | 6 min |
| 256 | 0.01 | 0.845 | 0.42 | 0.50 | 0.66 | 0.78 | 0.91 | 0.97 | 0.99 | 18 min |

Biopsy vs resection AUROC: 0.86 vs 0.86 (no gap on the self track).
Scoring by total instead of largest missing component: 0.851 (no gain).

## Analysis
- P1 is a large gain over area-only (0.64 → 0.86) and is near-perfect above 5% missing area.
- **Weak spot: omissions < 2% of tissue** (mostly whole tiny fragments). They *are* detected
  (only 11% score zero) but their signal is the same size as the false signal from
  **tears** in negatives (mean negative score: tear 0.016, shrink 0.009, neither 0.0025).
- Higher resolution / tighter tolerance did not help and costs 3× → resolution is not the
  bottleneck. The confusion is structural (a torn-off piece vs a missing piece), which is what
  P2 (fragment-level assignment with an explicit "missing" option) is designed to resolve.
- Decision: keep WORK=160, tol=0.02 as the P1 default; stop the resolution sweep.

## Pending (when resumed)
1. `cross` track (block mask as reference) with classical slide masks.
2. Same with GrandQC slide masks (`--slide-seg grandqc`). Masks for the 200 dev cases are
   already cached in `cache/masks_grandqc/` (7.5 min on CPU via `scripts/cache_grandqc_masks.py`).
3. Label-free segmentation check: uncovered block fraction on real, unmodified pairs.
4. Then P2.
