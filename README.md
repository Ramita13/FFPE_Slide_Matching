# FFPE Block ↔ Slide Matching: Detecting Missing Tissue Fragments

An open, experimental research project on **automated quality control (QC)** for
Formalin-Fixed Paraffin-Embedded (FFPE) histology: given a photograph of a paraffin
block and the thumbnail of the corresponding whole-slide image (WSI), determine
whether **every tissue fragment present in the block was captured on the slide**.

## Why this matters

During microtomy, small fragments (biopsy pieces, satellite fragments, edges) can
fail to be picked up, be trimmed away, or float off during water-bath transfer. A
slide that silently omits a fragment can lead to under-sampling of a lesion and a
missed diagnosis. Today this check is done by eye.

## How this differs from prior work

Tang *et al.* (Virchows Archiv, 2026, [doi:10.1007/s00428-026-04535-w](https://doi.org/10.1007/s00428-026-04535-w))
built a YOLOv11 + Siamese ConvNeXt-Tiny framework that verifies **case identity**
(does this slide belong to this block?). Their code:
[tang777-feng/Siamese-Network-with-ConvNeXt-Tiny](https://github.com/tang777-feng/Siamese-Network-with-ConvNeXt-Tiny).

This project targets a **finer-grained question**: the slide *is* from the right
block, but **is any tissue missing?** That requires fragment-level correspondence
and an explicit notion of "unmatched", not just a global similarity score.

## Repository layout

| Path | Contents |
|---|---|
| [docs/PROBLEM.md](docs/PROBLEM.md) | Problem definition, challenges, evaluation protocol |
| [docs/DATA.md](docs/DATA.md) | Dataset description and how to obtain it |
| [docs/research/](docs/research/) | Literature & method surveys (one report per research track) |
| [docs/experiments/](docs/experiments/) | Experiment log — one write-up per experiment |
| [src/ffpe_match/](src/ffpe_match/) | Reusable library code |
| [experiments/](experiments/) | Runnable experiment scripts |
| [scripts/](scripts/) | Data utilities (EDA, synthetic-defect generation, …) |
| `data/` | *Not tracked.* Place `LK/` (block) and `QP/` (slide) images here |

## Status

🧪 Phase 1 (research survey) complete. Start with **[docs/research/00_synthesis.md](docs/research/00_synthesis.md)**; progress is tracked in [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup

```bash
pip install -r requirements.txt   # Python 3.14, CPU-only is sufficient
python scripts/eda.py --data-dir data
```

## Citation

If you use this work, please cite this repository and the original dataset paper (Tang *et al.*, 2026).
