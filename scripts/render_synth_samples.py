"""Render synthetic samples for visual review (Phase 2 gate).

Each row: slide image | reference slide mask (removed region in red) | negative query |
positive query | block mask. Writes results/synth_review/sheet_XX.png.

Usage:  python scripts/render_synth_samples.py --n 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ffpe_match.data import load_pair  # noqa: E402
from ffpe_match.evaluate import load_split  # noqa: E402
from ffpe_match.masks import assessable, get_masks  # noqa: E402
from ffpe_match.synth import generate, sample_seed  # noqa: E402

H = 180


def _fit(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = H / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((H, H, 3), 255, np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    return out


def _m(mask, removed=None):
    v = np.where(mask[..., None], 60, 255).astype(np.uint8).repeat(3, 2)
    if removed is not None:
        v[removed] = (40, 40, 230)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--per-sheet", type=int, default=8)
    args = ap.parse_args()
    out = ROOT / "results" / "synth_review"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for case, sp in load_split("val", dev_only=True):
        if len(rows) >= args.n:
            break
        m = get_masks(case)
        if not assessable(m):
            continue
        neg = generate(m["slide_mask"], m["slide_groups"], 0, sample_seed(case, 0))
        pos = generate(m["slide_mask"], m["slide_groups"], 1, sample_seed(case, 1))
        _, qp = load_pair(case)
        row = np.hstack([_fit(qp), _fit(_m(m["slide_mask"], pos.removed_ref)), _fit(_m(neg.query)),
                         _fit(_m(pos.query)), _fit(_m(m["block_mask"]))])
        label = (f"{case} {sp} | pos: {pos.meta.get('mode')} {pos.missing_frac:.1%}"
                 f"{' tear' if pos.meta.get('tear') else ''}{' shrink' if pos.meta.get('shrink') else ''}"
                 f" | neg:{' tear' if neg.meta.get('tear') else ''}{' shrink' if neg.meta.get('shrink') else ''}")
        bar = np.full((18, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, row]))

    for i in range(0, len(rows), args.per_sheet):
        cv2.imwrite(str(out / f"sheet_{i // args.per_sheet:02d}.png"), np.vstack(rows[i:i + args.per_sheet]))
    print(f"wrote {len(rows)} rows to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
