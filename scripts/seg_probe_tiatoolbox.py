"""TIAToolbox tissue-masking probe on block (LK) and slide (QP) images, next to the classical baseline.

Compared methods (all CPU):
  * classical  : scripts/seg_probe_classical.py (block: flat-field Lab distance; slide: glass distance)
  * tia_otsu   : tiatoolbox.tools.tissuemask.OtsuTissueMasker      (Otsu on grey, no morphology)
  * tia_morph  : tiatoolbox.tools.tissuemask.MorphologicalMasker   (Otsu + dilation + min-region filter)
  * grandqc    : pretrained "grandqc_tissue_detection" (UNet++/EfficientNet-B0, Weng et al. 2024, 10 mpp)
  * effunet    : pretrained "efficientunet-tissue_mask" (TIA, 8 mpp)
  * fcn        : pretrained "fcn-tissue_mask" (UNet-ResNet50, TIA internal data, 2 mpp)
  Whole thumbnails are run in one pass. QP thumbnails are ~40 um/px, so DL models are also run on
  upsampled input (capped at 1536 px long side) to approximate their training resolution.

TIAToolbox pins numpy/opencv versions that conflict with the main env, so install it in a separate venv:
    (tiatoolbox 2.1.3 pins PyYAML==6.0.2, which has no cp314 wheel, so install with --no-deps; the venv
     must live at a SHORT path -- Windows MAX_PATH breaks transformers/numba inside the long temp dir)
    python -m venv --system-site-packages .venv_tia
    .venv_tia/Scripts/python -m pip install --no-deps tiatoolbox
    .venv_tia/Scripts/python -m pip install scikit-image shapely "umap-learn<0.5.12" dask "zarr<3.2"         openslide-python openslide-bin glymur simpleitk defusedxml imagecodecs universal-pathlib         transformers "tqdm<4.69" tifffile
    NUMBA_DISABLE_JIT=1 .venv_tia/Scripts/python scripts/seg_probe_tiatoolbox.py
Outputs: results/seg_probe/tiatoolbox/CaseXXXX_tia.png + summary.csv
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
from seg_probe_classical import DEFAULT_CASES, colorize, segment_block, segment_slide  # noqa: E402

from tiatoolbox.tools.tissuemask import MorphologicalMasker, OtsuTissueMasker  # noqa: E402
from tiatoolbox.models.architecture import get_pretrained_model  # noqa: E402


MAX_SIDE = 1536  # cap upsampled size to keep CPU time bounded


def model_mask(model, rgb: np.ndarray, upsample: float = 1, center_crop_half: bool = False) -> tuple[np.ndarray, float]:
    """Run a TIAToolbox segmentation model on a whole (small) image.

    upsample: approximate the model's training resolution (QP thumbnails are ~40 um/px).
    center_crop_half: models whose output is the central half of the input (fcn-tissue_mask:
    1024 in -> 512 out) get the image embedded in a 2x canvas of glass colour.
    Returns (mask at original size, effective upsample factor)."""
    h, w = rgb.shape[:2]
    f = min(float(upsample), MAX_SIDE / max(h, w)) if upsample > 1 else 1.0
    im = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_CUBIC) if f != 1 else rgb
    H, W = im.shape[:2]
    if center_crop_half:
        # output = central half of the input at the same resolution -> embed the image in a
        # 2x canvas of glass colour so that the central half is exactly the (padded) image
        Hp, Wp = H + (-H) % 32, W + (-W) % 32
        glass = np.median(np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]]), axis=0).astype(np.uint8)
        canvas = np.empty((2 * Hp, 2 * Wp, 3), np.uint8)
        canvas[:] = glass
        canvas[Hp // 2:Hp // 2 + H, Wp // 2:Wp // 2 + W] = im
        im = canvas
    else:
        im = cv2.copyMakeBorder(im, 0, (-H) % 64, 0, (-W) % 64, cv2.BORDER_REPLICATE)
    x = model.preproc(im) if hasattr(model, "preproc") else im
    with torch.inference_mode():
        prob = model.infer_batch(model, torch.from_numpy(np.ascontiguousarray(x))[None], device="cpu")
    prob = np.asarray(prob[0] if isinstance(prob, (list, tuple)) else prob)[0]
    m = np.asarray(model.postproc(prob)).astype(np.uint8)[:H, :W]
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool), f


def cc_labels(mask: np.ndarray, min_frac: float = 0.0015) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = np.zeros_like(lab)
    k = 1
    for i in range(1, n):
        if st[i, 4] >= min_frac * mask.size:
            out[lab == i] = k
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="*", default=DEFAULT_CASES)
    ap.add_argument("--all-models", action="store_true", help="also run efficientunet/fcn tissue masks")
    args = ap.parse_args()
    out = ROOT / "results" / "seg_probe" / "tiatoolbox"
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for name in ("grandqc_tissue_detection",) + (("efficientunet-tissue_mask", "fcn-tissue_mask") if args.all_models else ()):
        models[name] = get_pretrained_model(name)[0].eval()
    # (label, model name, upsample, centre-crop) -- QP thumbnails ~40 um/px
    dl_runs = [("grandqc_x1", "grandqc_tissue_detection", 1, False),
               ("grandqc_x2", "grandqc_tissue_detection", 2, False),
               ("grandqc_x4", "grandqc_tissue_detection", 4, False)]   # ~10 mpp training res
    if args.all_models:  # slow (20-40 s/img) and produced empty masks on Case0001 in our whole-image use
        dl_runs += [("effunet_x5", "efficientunet-tissue_mask", 5, False),  # 8 mpp training res
                    ("fcn_x4", "fcn-tissue_mask", 4, True)]                 # 2 mpp training res (still 5x off)

    rows = []
    for c in args.cases:
        tiles_rows = []
        for mod in ("LK", "QP"):
            bgr = cv2.imread(str(ROOT / "data" / mod / f"Case{c:04d}_{mod}.jpg"))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = {}
            t0 = time.perf_counter()
            res["classical"] = (segment_block(bgr)["mask"] if mod == "LK" else segment_slide(bgr)["mask"])
            res["t_classical"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            res["tia_otsu"] = OtsuTissueMasker().fit_transform([rgb])[0].astype(bool)
            res["t_tia_otsu"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            # thumbnails have no reliable MPP metadata -> give kernel/min-region in pixels
            k = max(1, min(bgr.shape[:2]) // 100)
            res["tia_morph"] = MorphologicalMasker(kernel_size=k, min_region_size=int(0.0015 * rgb.shape[0] * rgb.shape[1])
                                                   ).fit_transform([rgb])[0].astype(bool)
            res["t_tia_morph"] = time.perf_counter() - t0
            for label, name, up, cc in dl_runs:
                t0 = time.perf_counter()
                res[label], _ = model_mask(models[name], rgb, up, cc)
                res[f"t_{label}"] = time.perf_counter() - t0
            names = ["classical", "tia_otsu", "tia_morph"] + [r[0] for r in dl_runs]
            row = dict(case=f"Case{c:04d}", modality=mod, hw=f"{bgr.shape[0]}x{bgr.shape[1]}")
            for n in names:
                row[f"{n}_frac"] = round(float(res[n].mean()), 3)
                row[f"{n}_ncc"] = int(cc_labels(res[n]).max())
                row[f"{n}_ms"] = round(1000 * res[f"t_{n}"], 1)
                if n != "classical":
                    a, b = res[n], res["classical"]
                    row[f"{n}_iou_vs_classical"] = round(float((a & b).sum() / max(1, (a | b).sum())), 3)
            rows.append(row)
            print(row)
            H = 220
            tl = []
            for n in [None] + names:
                im = bgr if n is None else colorize(cc_labels(res[n]), bgr)
                r = H / im.shape[0]
                im = cv2.resize(im, (max(1, int(im.shape[1] * r)), H), interpolation=cv2.INTER_NEAREST)
                if n:
                    cv2.putText(im, n, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                tl += [im, np.zeros((H, 4, 3), np.uint8)]
            tiles_rows.append(np.hstack(tl[:-1]))
        W = max(t.shape[1] for t in tiles_rows)
        pad = [np.hstack([t, np.full((t.shape[0], W - t.shape[1], 3), 255, np.uint8)]) for t in tiles_rows]
        cv2.imwrite(str(out / f"Case{c:04d}_tia.png"), np.vstack(pad))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
