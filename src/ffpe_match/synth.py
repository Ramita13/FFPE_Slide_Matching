"""Mask-space synthetic omission generator.

Given a slide tissue mask and its fragment groups, produce a *query* mask that may have
tissue removed (positive) or not (negative). Both labels pass through the identical
nuisance pipeline, so tissue removal is the only systematic difference between them:

    reference slide mask --[omission if label=1]--> --[tear]--[shrink]--[elastic]-->
        --[rotate/flip/scale]--[crop to bbox + margin]--[boundary noise]--> query mask

Working in mask space avoids the image-inpainting shortcuts described in
docs/research/04_learned_and_synthetic.md. Design and parameters: docs/experiments/EXP-001.

The ground-truth missing region is returned in the *reference* frame, so localisation can be
scored on the self-pair track (reference = clean slide mask).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SynthConfig:
    p_fragment: float = 0.5               # P(remove whole fragment | positive, >=2 fragments)
    partial_frac: tuple = (0.02, 0.30)    # log-uniform fraction of total tissue for partial cuts
    p_tear: float = 0.4                   # nuisance: split a fragment and pull halves apart
    tear_gap: tuple = (0.01, 0.04)        # gap as fraction of image size
    p_shrink: float = 0.3                 # nuisance: "present but smaller" (section depth)
    shrink_frac: tuple = (0.05, 0.30)     # area loss of the shrunk fragment
    elastic_amp: tuple = (0.005, 0.02)    # displacement amplitude as fraction of image size
    scale: tuple = (0.90, 1.25)           # matches EDA crop-scale 5-95% range
    p_flip: float = 0.5
    margin: tuple = (0.02, 0.05)          # crop margin like the dataset's ROI crops


@dataclass
class Sample:
    query: np.ndarray                     # bool mask, query (slide-like) frame
    removed_ref: np.ndarray               # bool mask, reference frame (empty for negatives)
    label: int
    missing_frac: float                   # removed area / reference tissue area
    meta: dict = field(default_factory=dict)


def sample_seed(case: str, idx: int) -> int:
    return zlib.crc32(f"{case}:{idx}".encode())


# ----------------------------------------------------------------------------- helpers
def _smooth_field(shape, rng, grid=6):
    f = rng.normal(size=(grid, grid)).astype(np.float32)
    return cv2.resize(f, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _wavy_projection(shape, theta, rng, amp):
    """Signed distance along direction theta plus a smooth random perturbation,
    so cut / tear boundaries are irregular rather than straight lines."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return xx * np.cos(theta) + yy * np.sin(theta) + amp * _smooth_field(shape, rng)


def _fragments(groups: np.ndarray) -> list[np.ndarray]:
    return [groups == g for g in np.unique(groups) if g != 0]


def _shift(mask, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.uint8), M, mask.shape[::-1], flags=cv2.INTER_NEAREST) > 0


# ----------------------------------------------------------------------------- omissions
def _remove_fragment(mask, frags, rng):
    """Remove one fragment, favouring small ones (they are the ones lost in practice).
    Fragments holding >= 50% of the tissue are never removed (that is an empty/wrong slide,
    a different failure mode)."""
    areas = np.array([f.sum() for f in frags], float)
    w = np.where(areas < 0.5 * areas.sum(), 1 / np.sqrt(areas), 0.0)
    k = rng.choice(len(frags), p=w / w.sum())
    return mask & ~frags[k], {"mode": "fragment", "frag_area_frac": float(areas[k] / mask.sum())}


def _remove_partial(mask, frags, rng, cfg):
    """Cut off a fraction of one fragment along an irregular boundary."""
    total = mask.sum()
    lo, hi = np.log(cfg.partial_frac[0]), np.log(cfg.partial_frac[1])
    target = int(np.exp(rng.uniform(lo, hi)) * total)
    areas = np.array([f.sum() for f in frags])
    ok = np.flatnonzero(areas > 1.5 * target)
    k = rng.choice(ok) if ok.size else int(np.argmax(areas))
    target = min(target, int(0.6 * areas[k]))
    proj = _wavy_projection(mask.shape, rng.uniform(0, 2 * np.pi), rng, 0.05 * max(mask.shape))
    vals = proj[frags[k]]
    thr = np.sort(vals)[-max(target, 1)]
    return mask & ~(frags[k] & (proj >= thr)), {"mode": "partial"}


# ----------------------------------------------------------------------------- nuisance
def _tear(mask, groups, rng, cfg):
    frags = _fragments(groups)
    areas = np.array([f.sum() for f in frags], float)
    f = frags[rng.choice(len(frags), p=areas / areas.sum())]
    ys, xs = np.nonzero(f)
    theta = rng.uniform(0, 2 * np.pi)
    proj = _wavy_projection(mask.shape, theta, rng, 0.03 * max(mask.shape))
    side = f & (proj > np.median(proj[ys, xs]))
    d = rng.uniform(*cfg.tear_gap) * max(mask.shape)
    moved = _shift(side, d * np.cos(theta), d * np.sin(theta))
    return (mask & ~side) | moved


def _shrink(mask, groups, rng, cfg):
    frags = _fragments(groups)
    f = frags[rng.integers(len(frags))]
    a0, goal = f.sum(), 1 - rng.uniform(*cfg.shrink_frac)
    k = np.ones((3, 3), np.uint8)
    g = f.astype(np.uint8)
    while True:
        nxt = cv2.erode(g, k)
        if nxt.sum() < goal * a0 or nxt.sum() < 0.5 * a0:  # never erase the fragment
            break
        g = nxt
    return (mask & ~f) | (g > 0)


def _elastic(mask, rng, cfg):
    h, w = mask.shape
    a = rng.uniform(*cfg.elastic_amp) * max(h, w)
    dx, dy = a * _smooth_field((h, w), rng, 4), a * _smooth_field((h, w), rng, 4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = cv2.remap(mask.astype(np.float32), xx + dx, yy + dy, cv2.INTER_LINEAR)
    return out > 0.5


def _similarity(mask, rng, cfg):
    h, w = mask.shape
    ang, s = rng.uniform(0, 360), rng.uniform(*cfg.scale)
    flip = rng.random() < cfg.p_flip
    m = mask[:, ::-1] if flip else mask
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    x0, y0 = corners.min(0)
    x1, y1 = corners.max(0)
    M[:, 2] -= (x0, y0)
    size = (int(np.ceil(x1 - x0)) + 1, int(np.ceil(y1 - y0)) + 1)
    out = cv2.warpAffine(m.astype(np.float32), M, size, flags=cv2.INTER_LINEAR) > 0.5
    return out, {"angle": float(ang), "scale": float(s), "flip": bool(flip)}


def _crop(mask, rng, cfg):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = int(rng.uniform(*cfg.margin) * max(y1 - y0, x1 - x0))
    return np.pad(mask[y0:y1, x0:x1], m)


def _boundary_noise(mask, rng):
    r = int(rng.integers(1, 3))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    op = cv2.MORPH_OPEN if rng.random() < 0.5 else cv2.MORPH_CLOSE
    return cv2.morphologyEx(mask.astype(np.uint8), op, k) > 0


# ----------------------------------------------------------------------------- public API
def generate(slide_mask: np.ndarray, slide_groups: np.ndarray, label: int, seed: int,
             cfg: SynthConfig = SynthConfig()) -> Sample:
    """Create one synthetic query from a reference slide mask. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    ref = slide_mask.astype(bool)
    groups = np.where(ref, slide_groups, 0)
    frags = _fragments(groups)
    meta: dict = {"n_frag": len(frags)}

    q = ref.copy()
    if label:
        removable = sum(f.sum() < 0.5 * ref.sum() for f in frags)
        if removable and rng.random() < cfg.p_fragment:
            q, info = _remove_fragment(q, frags, rng)
        else:
            q, info = _remove_partial(q, frags, rng, cfg)
        meta.update(info)
    removed = ref & ~q
    groups_q = np.where(q, groups, 0)

    # Nuisance pipeline: identical distribution for both labels. Pad first so tears and
    # warps never clip at the border.
    pad = int(0.15 * max(q.shape))
    q, groups_q = np.pad(q, pad), np.pad(groups_q, pad)
    if rng.random() < cfg.p_tear and groups_q.any():
        q = _tear(q, groups_q, rng, cfg); meta["tear"] = True
    if rng.random() < cfg.p_shrink and groups_q.any():
        q = _shrink(q, np.where(q, groups_q, 0), rng, cfg); meta["shrink"] = True
    q = _elastic(q, rng, cfg)
    q, meta["sim"] = _similarity(q, rng, cfg)
    q = _boundary_noise(_crop(q, rng, cfg), rng)

    return Sample(query=q, removed_ref=removed, label=int(label),
                  missing_frac=float(removed.sum() / max(ref.sum(), 1)), meta=meta)
