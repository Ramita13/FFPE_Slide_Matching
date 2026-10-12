"""On-disk cache of classical segmentation masks (cache/masks/<case>.npz, gitignored)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import ROOT, load_pair
from .segment import segment_block, segment_slide

CACHE = ROOT / "cache" / "masks"


def get_masks(case: str, cache: Path = CACHE) -> dict:
    """Return block_mask, block_inst, slide_mask, slide_groups for a case (cached)."""
    f = cache / f"{case}.npz"
    if f.exists():
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    lk, qp = load_pair(case)
    blk, sld = segment_block(lk), segment_slide(qp)
    out = {"block_mask": blk["mask"], "block_inst": blk["instances"].astype(np.int32),
           "slide_mask": sld["mask"], "slide_groups": sld["groups"].astype(np.int32)}
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def assessable(m: dict) -> bool:
    """Both modalities must contain tissue; otherwise the pair is 'cannot assess'."""
    return bool(m["block_mask"].any() and m["slide_mask"].any())
