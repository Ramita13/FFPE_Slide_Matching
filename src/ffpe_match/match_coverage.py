"""P1: rigid pose search + per-piece refinement + coverage read-out.

Assumption (docs/research/03_partial_matching.md): slide tissue is a subset of block tissue.
So every slide (query) pixel should land on block (reference) tissue, while block tissue is
allowed to stay uncovered. Uncovered block tissue is the missing-tissue map.

1. Downsample both masks by one common factor (the two modalities share ~px/mm, EDA §scale).
2. Global search over rotation x flip x scale; translation for each by FFT correlation.
   Objective  J = overlap - lam * (query pixels off reference tissue).
3. Per-piece refinement: each query piece gets a small local rotation/translation search,
   which absorbs tears, floats and local distortion (many-to-one allowed).
4. missing = reference minus dilated placed query, opened to drop thin boundary residue.
   score = largest missing component / reference area.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import fftconvolve

WORK = 160               # working resolution: longest side of the reference
LAM = 1.0                # penalty weight for query tissue placed on wax/background
ANGLES = np.arange(0, 360, 10)
SCALES = (0.85, 0.95, 1.05, 1.15, 1.25)


def _resize(m, f):
    h, w = m.shape
    return cv2.resize(m.astype(np.float32), (max(1, round(w * f)), max(1, round(h * f))),
                      interpolation=cv2.INTER_AREA) > 0.5


def _transform(m, angle, scale=1.0, flip=False):
    """Rotate/scale/flip about the centre onto a tight, expanded canvas (float mask)."""
    if flip:
        m = m[:, ::-1]
    h, w = m.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    M[:, 2] -= c.min(0)
    size = tuple(int(np.ceil(v)) + 1 for v in (np.ptp(c[:, 0]), np.ptp(c[:, 1])))
    out = cv2.warpAffine(m.astype(np.float32), M, size, flags=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(out > 0.5)
    return (out > 0.5)[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if ys.size else out > 0.5


def _best_shift(ref, q, window=None):
    """Best placement of q on ref. Returns (J, top-left y, x). `window`=(y0, x0, r) limits
    the search to |offset - (y0, x0)| <= r."""
    ov = fftconvolve(ref.astype(np.float32), q[::-1, ::-1].astype(np.float32), mode="full")
    J = (1 + LAM) * ov - LAM * q.sum()
    if window is not None:
        y0, x0, r = window
        iy, ix = y0 + q.shape[0] - 1, x0 + q.shape[1] - 1
        mask = np.full(J.shape, -np.inf)
        mask[max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1] = 0
        J = J + mask
    i, j = np.unravel_index(np.argmax(J), J.shape)
    return float(J[i, j]), i - (q.shape[0] - 1), j - (q.shape[1] - 1)


def _paste(canvas, q, y, x):
    H, W = canvas.shape
    y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + q.shape[0], H), min(x + q.shape[1], W)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] |= q[y0 - y:y1 - y, x0 - x:x1 - x]


def align(ref: np.ndarray, query: np.ndarray):
    """Return (placed query in reference frame at work resolution, ref at work resolution)."""
    f = WORK / max(ref.shape)
    R, Q = _resize(ref, f), _resize(query, f)
    best = (-np.inf, None)
    for flip in (False, True):
        for s in SCALES:
            for a in ANGLES:
                q = _transform(Q, a, s, flip)
                J, y, x = _best_shift(R, q)
                if J > best[0]:
                    best = (J, (a, s, flip, y, x))
    a0, s0, flip, _, _ = best[1]
    # local refinement of the global pose
    for a in np.arange(a0 - 8, a0 + 9, 2):
        for s in (s0 - 0.05, s0, s0 + 0.05):
            q = _transform(Q, a, s, flip)
            J, y, x = _best_shift(R, q)
            if J > best[0]:
                best = (J, (a, s, flip, y, x))
    a, s, flip, y, x = best[1]
    placed_global = np.zeros(R.shape, bool)
    qg = _transform(Q, a, s, flip)
    _paste(placed_global, qg, y, x)

    # per-piece refinement on an enlarged canvas (pieces may sit partly outside R)
    pad = WORK // 4
    Rp = np.pad(R, pad)
    canvas = np.zeros(Rp.shape, bool)
    full = np.zeros(Rp.shape, bool)
    _paste(full, qg, y + pad, x + pad)
    n, lab = cv2.connectedComponents(full.astype(np.uint8))
    r = max(3, WORK // 16)
    for k in range(1, n):
        piece = lab == k
        if piece.sum() < 4:
            continue
        ys, xs = np.nonzero(piece)
        p = piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        bestp = _best_shift(Rp, p, window=(ys.min(), xs.min(), r)) + (p,)
        for da in (-15, -10, -5, 5, 10, 15):
            pr = _transform(p, da)
            cy, cx = ys.min() + (p.shape[0] - pr.shape[0]) // 2, xs.min() + (p.shape[1] - pr.shape[1]) // 2
            cand = _best_shift(Rp, pr, window=(cy, cx, r)) + (pr,)
            if cand[0] > bestp[0]:
                bestp = cand
        _paste(canvas, bestp[3], bestp[1], bestp[2])
    return canvas[pad:-pad, pad:-pad], R, placed_global


def coverage_score(ref: np.ndarray, query: np.ndarray, tol: float = 0.02,
                   open_frac: float = 0.5) -> dict:
    """tol: boundary tolerance (fraction of WORK) by which placed query tissue is dilated.
    open_frac: opening radius as a fraction of the tolerance; removes thin residue rings
    left by shrinkage/boundary noise while keeping compact missing regions."""
    placed, R, _ = align(ref, query)
    t = max(1, int(round(tol * WORK)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    covered = cv2.dilate(placed.astype(np.uint8), k) > 0
    miss = R & ~covered
    to = max(1, int(round(open_frac * t)))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * to + 1, 2 * to + 1))
    miss = cv2.morphologyEx(miss.astype(np.uint8), cv2.MORPH_OPEN, ko) > 0
    area = max(R.sum(), 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(miss.astype(np.uint8))
    largest = st[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
    miss_full = cv2.resize(miss.astype(np.uint8), ref.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    return {"score": float(largest / area), "x_total_missing": float(miss.sum() / area),
            "x_offref": float((placed & ~R).sum() / max(placed.sum(), 1)),
            "missing_map": miss_full & ref.astype(bool)}
