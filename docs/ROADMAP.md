# Roadmap

## Phase 0 — Framing ✅
- Problem definition, data description, evaluation protocol ([PROBLEM.md](PROBLEM.md), [DATA.md](DATA.md))

## Phase 1 — Research survey (in progress)
Parallel research tracks, each producing a report in [research/](research/):

| # | Track | Report |
|---|---|---|
| 01 | Prior work & reference implementation (Tang *et al.*) | [01_prior_work.md](research/01_prior_work.md) |
| 02 | Tissue segmentation / fragment extraction (block & slide) | [02_segmentation.md](research/02_segmentation.md) |
| 03 | Partial / outlier-aware matching & registration (the CPD problem) | [03_partial_matching.md](research/03_partial_matching.md) |
| 04 | Learned approaches & synthetic defect generation | [04_learned_and_synthetic.md](research/04_learned_and_synthetic.md) |
| 05 | Dataset exploratory analysis | [05_dataset_eda.md](research/05_dataset_eda.md) |

Output: a synthesis ([research/00_synthesis.md](research/00_synthesis.md)) with a ranked shortlist of pipelines.

## Phase 2 — Benchmark infrastructure
- Fixed case-level splits
- Synthetic omission generator + hard non-omission perturbations
- Small manually labelled real test set
- Common evaluation harness (AUROC, sens@spec, stratified by missing-area %)

## Phase 3 — Experiments
Each shortlisted pipeline is run as an experiment logged in [experiments/](experiments/)
using [the template](experiments/TEMPLATE.md).

## Phase 4 — Consolidation
Best pipeline, ablations, error analysis, write-up.
