"""Cache GrandQC slide tissue masks (TIAToolbox pretrained 'grandqc_tissue_detection', 2x
upsampling) as an alternative slide segmenter: cache/masks_grandqc/<case>.npz.

Must run in the TIAToolbox environment (Python 3.14 needs a separate venv; see
docs/research/02_segmentation.md):
    .venv_tia/Scripts/python scripts/cache_grandqc_masks.py --dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffpe_match.data import load_pair  # noqa: E402
from ffpe_match.evaluate import load_split  # noqa: E402
from ffpe_match.segment import _odd, fill_holes, group_pieces, remove_small  # noqa: E402
from seg_probe_tiatoolbox import model_mask  # noqa: E402
from tiatoolbox.models.architecture import get_pretrained_model  # noqa: E402

OUT = ROOT / "cache" / "masks_grandqc"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = get_pretrained_model("grandqc_tissue_detection")[0].eval()
    cases = load_split(args.split, dev_only=args.dev)
    for i, (case, _) in enumerate(cases):
        f = OUT / f"{case}.npz"
        if f.exists():
            continue
        _, qp = load_pair(case)
        m, _ = model_mask(model, cv2.cvtColor(qp, cv2.COLOR_BGR2RGB), upsample=2)
        h, w = m.shape
        m = remove_small(fill_holes(m, max_hole_frac=0.02), int(0.0015 * h * w))
        groups = group_pieces(m, gap=max(3, int(0.02 * min(h, w))))
        np.savez_compressed(f, slide_mask=m, slide_groups=groups.astype(np.int32))
        if i % 20 == 0:
            print(f"{i}/{len(cases)} {case}", flush=True)


if __name__ == "__main__":
    main()
