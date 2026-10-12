"""Classical (training-free) tissue / fragment segmentation probe for both modalities.

Block (LK): background-colour distance in Lab (wax colour estimated from the image border),
            glare suppression, Otsu, morphology, then distance-transform watershed to split
            fused fragments.
Slide (QP): "not white" tissue map (max of saturation and optical density), Otsu with a floor,
            morphology, then proximity grouping of torn pieces into fragment groups.

Only depends on numpy + opencv (no scikit-image wheel needed). CPU only.

Usage:
    python scripts/seg_probe_classical.py                 # default case list
    python scripts/seg_probe_classical.py --cases 1 29 223 --out results/seg_probe

Outputs (in --out): one PNG panel per case (block | block mask+instances | slide | slide
mask+groups) and summary.csv with per-case counts, tissue fractions and timings.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CASES = [1, 22, 29, 36, 75, 120, 155, 223, 260, 330, 1107, 1639,
                 595, 654, 1372, 1413, 1442, 2043, 190, 799, 1898, 2108, 981, 638]


# Segmentation code lives in the package; this script only renders/summarises probes.
import sys
sys.path.insert(0, str(ROOT / "src"))
from ffpe_match.segment import segment_block, segment_slide  # noqa: E402



# ----------------------------------------------------------------------------- viz
def colorize(labels: np.ndarray, base: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    pal = rng.integers(40, 255, (max(labels.max() + 1, 2), 3)).astype(np.uint8)
    pal[0] = 0
    col = pal[labels]
    out = base.copy()
    m = labels > 0
    out[m] = (0.45 * base[m] + 0.55 * col[m]).astype(np.uint8)
    for l in range(1, labels.max() + 1):
        cs, _ = cv2.findContours((labels == l).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (0, 0, 255), 1)
    return out


def panel(case: int, lk, blk, qp, sld, H: int = 320) -> np.ndarray:
    def rs(im):
        r = H / im.shape[0]
        return cv2.resize(im, (max(1, int(im.shape[1] * r)), H), interpolation=cv2.INTER_NEAREST)
    ov_b = lk.copy()
    ov_b[blk["glare"]] = (255, 255, 0)
    tiles = [rs(lk), rs(colorize(blk["instances"], ov_b)), rs(qp), rs(colorize(sld["groups"], qp))]
    sep = np.zeros((H, 6, 3), np.uint8)
    row = np.hstack(sum([[t, sep] for t in tiles], [])[:-1])
    nb = len([l for l in np.unique(blk["instances"]) if l])
    ng = len([l for l in np.unique(sld["groups"]) if l])
    txt = f"Case{case:04d}  block inst={nb}  slide pieces={sld['n_pieces']} groups={ng}"
    bar = np.full((26, row.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, row])


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="*", default=DEFAULT_CASES)
    ap.add_argument("--out", default=str(ROOT / "results" / "seg_probe"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in args.cases:
        lk = cv2.imread(str(ROOT / "data" / "LK" / f"Case{c:04d}_LK.jpg"))
        qp = cv2.imread(str(ROOT / "data" / "QP" / f"Case{c:04d}_QP.jpg"))
        if lk is None or qp is None:
            print(f"Case{c:04d}: missing image, skipped")
            continue
        t0 = time.perf_counter()
        blk = segment_block(lk)
        t1 = time.perf_counter()
        sld = segment_slide(qp)
        t2 = time.perf_counter()
        cv2.imwrite(str(out / f"Case{c:04d}_seg.png"), panel(c, lk, blk, qp, sld))
        nb = int(len(np.unique(blk["instances"])) - 1)
        ng = int(len(np.unique(sld["groups"])) - 1)
        n_cc_block = cv2.connectedComponents(blk["mask"].astype(np.uint8))[0] - 1
        rows.append(dict(case=f"Case{c:04d}", lk_hw=f"{lk.shape[0]}x{lk.shape[1]}",
                         qp_hw=f"{qp.shape[0]}x{qp.shape[1]}",
                         block_tissue_frac=round(float(blk["mask"].mean()), 3),
                         block_cc=n_cc_block, block_ws_instances=nb,
                         block_glare_frac=round(float(blk["glare"].mean()), 3),
                         block_contrast=round(blk["contrast"], 1),
                         slide_tissue_frac=round(float(sld["mask"].mean()), 3),
                         slide_pieces=sld["n_pieces"], slide_groups=ng,
                         block_ms=round(1000 * (t1 - t0), 1), slide_ms=round(1000 * (t2 - t1), 1)))
        print(rows[-1])
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} panels + summary.csv to {out}")


if __name__ == "__main__":
    main()
