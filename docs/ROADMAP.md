# Roadmap

## Phase 0 — Framing ✅
- Problem definition, data description, evaluation protocol ([PROBLEM.md](PROBLEM.md), [DATA.md](DATA.md))

## Phase 1 — Research survey ✅
Parallel research tracks, each producing a report in [research/](research/):

| # | Track | Report |
|---|---|---|
| 01 | Prior work & reference implementation (Tang *et al.*) | [01_prior_work.md](research/01_prior_work.md) |
| 02 | Tissue segmentation / fragment extraction (block & slide) | [02_segmentation.md](research/02_segmentation.md) |
| 03 | Partial / outlier-aware matching & registration (the CPD problem) | [03_partial_matching.md](research/03_partial_matching.md) |
| 04 | Learned approaches & synthetic defect generation | [04_learned_and_synthetic.md](research/04_learned_and_synthetic.md) |
| 05 | Dataset exploratory analysis | [05_dataset_eda.md](research/05_dataset_eda.md) |

Output: **[research/00_synthesis.md](research/00_synthesis.md)**: conclusions, ranked pipelines (P1–P4) and the gated plan for the phases below.

> Working principle: cheapest experiment first, a pass/fail gate written before each run, scale up only after a gate passes.

## Phase 2 — Benchmark infrastructure ✅ ([EXP-001](experiments/EXP-001_benchmark.md))
- Fixed case-level splits
- **Mask-space** synthetic omission generator + hard non-omission perturbations
- ~~Small manually labelled real test set~~ → moved to Phase 6 (no real labels available)
- Common evaluation harness (AUROC, sens@spec, stratified by missing-area %)

## Phase 3 — Segmentation benchmark (in progress, paused)
Slide (classical vs GrandQC) and block (classical vs SAM-refined), judged by downstream P1 performance on the `cross` track (no hand-labelled masks; decided 2026-10-12).

## Phase 4 — Matching baselines (P1 started: [EXP-002](experiments/EXP-002_p1_coverage.md))
P1 rigid-search + coverage → P2 fragment assignment with "missing" option → P3 partial GW; CPD as reference.

## Phase 5 — Learned matcher (only if Phase 4 misses its gate)

## Phase 6 — Real-world validation & write-up

Gates and details: [research/00_synthesis.md §3](research/00_synthesis.md#3-gated-plan).
Experiments are logged in [experiments/](experiments/) using [the template](experiments/TEMPLATE.md).
