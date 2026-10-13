"""On-disk cache of classical segmentation masks (cache/masks/<case>.npz, gitignored)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import ROOT, load_pair
from .segment import segment_block, segment_slide

CACHE = ROOT / "cache" / "masks"


def get_masks(case: str, slide: str = "classical", block: str = "classical") -> dict:
    """Return block_mask, block_inst, slide_mask, slide_groups for a case.

    slide/block select the segmenter. 'classical' masks are computed on demand and cached;
    other variants (e.g. slide='grandqc') are read from cache/masks_<variant>/ and must be
    produced beforehand by their scripts (scripts/cache_<variant>_masks.py)."""
    m = _classical(case)
    if slide != "classical":
        with np.load(ROOT / "cache" / f"masks_{slide}" / f"{case}.npz") as z:
            m["slide_mask"], m["slide_groups"] = z["slide_mask"], z["slide_groups"]
    if block != "classical":
        with np.load(ROOT / "cache" / f"masks_{block}" / f"{case}.npz") as z:
            m["block_mask"], m["block_inst"] = z["block_mask"], z["block_inst"]
    return m


def _classical(case: str, cache: Path = CACHE) -> dict:
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
