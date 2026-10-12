"""MobileSAM CPU probe: speed + qualitative behaviour on block (LK) and slide (QP) images.

Two prompting modes are compared against the classical probe (seg_probe_classical.py):
  * auto  : SamAutomaticMaskGenerator (grid of point prompts, points_per_side=16)
  * prompt: classical instances -> one box + one positive point (max distance-transform
            location) per instance -> SAM refines each mask ("classical proposes, SAM refines").

Requirements (CPU-only, Python 3.14 OK):
    pip install timm git+https://github.com/ChaoningZhang/MobileSAM.git
    weights: https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -> models/

Usage:
    python scripts/seg_probe_mobilesam.py --cases 1 155 330 223 29 120
Outputs: results/seg_probe/mobilesam/CaseXXXX_sam.png and timings.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_probe_classical import colorize, segment_block, segment_slide  # noqa: E402

from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry  # noqa: E402


def auto_labels(gen, rgb):
    masks = gen.generate(rgb)
    lab = np.zeros(rgb.shape[:2], np.int32)
    # paint large->small so small masks stay visible; drop near-full-image (background) masks
    for i, m in enumerate(sorted(masks, key=lambda m: -m["area"])):
        if m["area"] > 0.6 * lab.size:
            continue
        lab[m["segmentation"]] = i + 1
    return lab, len(masks)


def prompted_labels(pred, rgb, inst):
    pred.set_image(rgb)
    out = np.zeros(inst.shape, np.int32)
    for l in [l for l in np.unique(inst) if l]:
        m = (inst == l).astype(np.uint8)
        ys, xs = np.nonzero(m)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        py, px = np.unravel_index(np.argmax(dt), dt.shape)
        masks, scores, _ = pred.predict(point_coords=np.array([[px, py]]), point_labels=np.array([1]),
                                        box=box, multimask_output=False)
        out[masks[0] & (out == 0)] = l
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="*", default=[1, 155, 330, 223, 29, 120])
    ap.add_argument("--weights", default=str(ROOT / "models" / "mobile_sam.pt"))
    ap.add_argument("--pps", type=int, default=16, help="points_per_side for auto mode")
    args = ap.parse_args()
    out = ROOT / "results" / "seg_probe" / "mobilesam"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    sam = sam_model_registry["vit_t"](checkpoint=args.weights).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.pps, pred_iou_thresh=0.8,
                                    stability_score_thresh=0.85, min_mask_region_area=50)
    pred = SamPredictor(sam)

    rows = []
    for c in args.cases:
        tiles = []
        for mod in ("LK", "QP"):
            bgr = cv2.imread(str(ROOT / "data" / mod / f"Case{c:04d}_{mod}.jpg"))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cls = segment_block(bgr)["instances"] if mod == "LK" else segment_slide(bgr)["groups"]
            with torch.inference_mode():
                t0 = time.perf_counter()
                pred.set_image(rgb)
                t_enc = time.perf_counter() - t0
                t0 = time.perf_counter()
                auto, n_auto = auto_labels(gen, rgb)
                t_auto = time.perf_counter() - t0
                t0 = time.perf_counter()
                prm = prompted_labels(pred, rgb, cls)
                t_prm = time.perf_counter() - t0
            rows.append(dict(case=f"Case{c:04d}", modality=mod, hw=f"{bgr.shape[0]}x{bgr.shape[1]}",
                             encoder_s=round(t_enc, 2), auto_s=round(t_auto, 2), auto_masks=n_auto,
                             prompted_s=round(t_prm, 2), n_prompts=int(len(np.unique(cls)) - 1)))
            print(rows[-1])
            H = 300
            for im in (bgr, colorize(cls, bgr), colorize(auto, bgr), colorize(prm, bgr)):
                r = H / im.shape[0]
                tiles.append(cv2.resize(im, (int(im.shape[1] * r), H), interpolation=cv2.INTER_NEAREST))
        sep = np.zeros((300, 6, 3), np.uint8)
        top = np.hstack([tiles[0], sep, tiles[1], sep, tiles[2], sep, tiles[3]])
        bot = np.hstack([tiles[4], sep, tiles[5], sep, tiles[6], sep, tiles[7]])
        W = max(top.shape[1], bot.shape[1])
        pad = lambda a: np.hstack([a, np.full((a.shape[0], W - a.shape[1], 3), 255, np.uint8)])
        bar = np.full((24, W, 3), 255, np.uint8)
        cv2.putText(bar, f"Case{c:04d}: image | classical | SAM auto (pps={args.pps}) | SAM box+point from classical",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"Case{c:04d}_sam.png"), np.vstack([bar, pad(top), pad(bot)]))
    with open(out / "timings.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
