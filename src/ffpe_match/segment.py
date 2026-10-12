"""Classical (training-free) tissue / fragment segmentation for both modalities.

Block (LK): background-colour distance in Lab (wax colour estimated from the image border),
            glare suppression, Otsu, quadratic flat-field correction, morphology, then
            distance-transform watershed to split fused fragments.
Slide (QP): "not white" tissue map (max channel darkness below glass) with a low floor plus a
            texture test for pale fat, morphology, then proximity grouping of torn pieces.

Origin and qualitative evaluation: docs/research/02_segmentation.md.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------- helpers
def _odd(k: int) -> int:
    k = max(3, int(k))
    return k if k % 2 else k + 1


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def fill_holes(mask: np.ndarray, max_hole_frac: float | None = None) -> np.ndarray:
    """Fill enclosed background. With max_hole_frac, only holes smaller than that fraction of
    the image are filled: large 'holes' are usually glass enclosed by tissue plus an edge
    artefact (Case0074), not lumina inside tissue."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = ff[1:-1, 1:-1] == 0
    if max_hole_frac is not None:
        holes = remove_small(holes, 1) & ~remove_small(holes, int(max_hole_frac * h * w))
    return mask.astype(bool) | holes


def border_ring(img: np.ndarray, frac: float = 0.04) -> np.ndarray:
    h, w = img.shape[:2]
    b = max(2, int(round(min(h, w) * frac)))
    ring = np.zeros((h, w), bool)
    ring[:b] = ring[-b:] = True
    ring[:, :b] = ring[:, -b:] = True
    return ring


# ----------------------------------------------------------------------------- block (LK)
def segment_block(bgr: np.ndarray, flatfield: bool = True) -> dict:
    h, w = bgr.shape[:2]
    s = min(h, w)
    blur = cv2.GaussianBlur(bgr, (0, 0), max(1.0, s / 200))
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab).astype(np.float32)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Wax colour: the ROI crop usually leaves a wax margin; tissue may touch some border
    # pixels, so take the border pixels whose chroma is lowest (wax is near-neutral grey).
    ring = border_ring(bgr)
    ab = lab[..., 1:][ring]
    chroma = np.hypot(ab[:, 0] - 128, ab[:, 1] - 128)
    sel = chroma <= np.percentile(chroma, 40)
    bg = np.median(lab[ring][sel], axis=0)

    # Glare / specular highlights. The released JPEGs are tone-compressed (V rarely > 215),
    # so define glare relative to the wax: brighter than wax and near-neutral.
    chroma_px = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    bg_chroma = float(np.hypot(bg[1] - 128, bg[2] - 128))
    glare = (lab[..., 0] > bg[0] + 10) & (chroma_px < bg_chroma + 6)
    glare = cv2.dilate(glare.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # Distance from wax: chroma difference weighted more than lightness (tissue is mostly
    # browner/yellower/pinker, sometimes darker; lightness alone is confounded by shading).
    def wax_distance(bgmap):
        d = lab - bgmap
        return np.sqrt(0.5 * d[..., 0] ** 2 + 2.0 * d[..., 1] ** 2 + 2.0 * d[..., 2] ** 2)

    def otsu_mask(dist):
        dist = dist.copy()
        dist[glare] = 0
        d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        t, _ = cv2.threshold(d8[~glare], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Guard: if the crop is almost all wax, Otsu splits noise -> absolute floor.
        t_raw = max(float(t) / 255 * dist.max(), 6.0)
        return dist > t_raw, t_raw

    mask, t_raw = otsu_mask(wax_distance(bg))
    if flatfield:
        # Wax colour/illumination varies across the block (ring-light falloff, wax
        # thickness, tissue seen *through* translucent wax). Fit a quadratic surface per
        # Lab channel to pixels currently called wax, then re-threshold (2 rounds).
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yy /= h; xx /= w
        A = np.stack([np.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy], -1).reshape(-1, 6)
        for _ in range(2):
            wax = ~cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~glare
            if wax.sum() < 0.05 * h * w:
                break
            idx = np.flatnonzero(wax.ravel())
            idx = idx[:: max(1, idx.size // 20000)]
            coef, *_ = np.linalg.lstsq(A[idx], lab.reshape(-1, 3)[idx], rcond=None)
            bgmap = (A @ coef).reshape(h, w, 3)
            mask, t_raw = otsu_mask(wax_distance(bgmap))

    k = _odd(s / 120)  # small: larger closings bridge the thin wax channels between fragments
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    mask = fill_holes(mask | (glare & cv2.dilate(mask.astype(np.uint8), np.ones((k, k))).astype(bool)))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    mask = remove_small(mask, int(0.003 * h * w))

    inst = split_watershed(mask, bgr)
    return {"mask": mask, "instances": inst, "glare": glare, "contrast": float(t_raw)}


def split_watershed(mask: np.ndarray, bgr: np.ndarray, rel_peak: float = 0.45) -> np.ndarray:
    """Distance-transform marker watershed. Splits necks between convex blobs (fused
    fragments) while leaving single compact fragments whole. Returns int32 label map.
    Markers = per-component cores where dt >= rel_peak * max(dt in component)."""
    m8 = mask.astype(np.uint8)
    if m8.sum() == 0:
        return np.zeros(mask.shape, np.int32)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    n, cc = cv2.connectedComponents(m8)
    markers = np.zeros(mask.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        comp = cc == c
        core = comp & (dt >= rel_peak * dt[comp].max())
        k, cl = cv2.connectedComponents(core.astype(np.uint8))
        for j in range(1, k):
            markers[cl == j] = nxt
            nxt += 1
    markers[~mask] = -1  # background marker for watershed_ift
    topo = (255 - cv2.normalize(dt, None, 0, 254, cv2.NORM_MINMAX)).astype(np.uint8)
    ws = ndi.watershed_ift(topo, markers).astype(np.int32)
    ws[~mask | (ws < 0)] = 0
    return ws


# ----------------------------------------------------------------------------- slide (QP)
def segment_slide(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    s = min(h, w)
    blur = cv2.GaussianBlur(bgr, (0, 0), max(0.8, s / 300))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Glass colour = per-channel 90th percentile (glass dominates most crops). Tissue is any
    # pixel darker than glass in some channel; pale fat/loose stroma is only ~10-20 levels
    # darker, so we use a low fixed floor instead of Otsu alone (Otsu on H&E keeps only the
    # basophilic/eosinophilic cores and drops fat -- seen on Case0001/0120/0260).
    f = blur.astype(np.float32)
    glass = np.percentile(f.reshape(-1, 3), 90, axis=0)
    diff = np.clip(glass - f, 0, None).max(axis=2)
    diff8 = np.clip(diff * 2, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(float(t), 2 * 14.0)  # never require more than 14 grey levels below glass
    feat = diff8
    mask = feat > t
    # very pale tissue is textured; glass is flat -> require some local texture for the
    # faint band only (keeps fat septa, drops smooth glass shading / coverslip haze)
    faint = (diff > 7) & ~mask
    lstd = cv2.GaussianBlur(f.mean(axis=2) ** 2, (0, 0), 2) - cv2.GaussianBlur(f.mean(axis=2), (0, 0), 2) ** 2
    mask |= faint & (np.sqrt(np.clip(lstd, 0, None)) > 3.0)

    # Dark non-tissue (slide edge, pen, coverslip edge, label): dark AND unsaturated
    dark_gray = (hsv[..., 2] < 90) & (hsv[..., 1] < 60)
    mask &= ~cv2.dilate(dark_gray.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    mask &= ~straight_lines(mask, thick=_odd(s / 40))

    k = _odd(s / 200)  # keep narrow tears open; merging is done explicitly by group_pieces
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    mask = fill_holes(mask, max_hole_frac=0.02)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    mask = remove_small(mask, int(0.0015 * h * w))

    n, pieces = cv2.connectedComponents(mask.astype(np.uint8))
    groups = group_pieces(mask, gap=max(3, int(0.02 * s)))
    return {"mask": mask, "pieces": pieces, "n_pieces": n - 1, "groups": groups,
            "dark_artifact_frac": float(dark_gray.mean())}


def straight_lines(mask: np.ndarray, rel_len: float = 0.2, thick: int = 7) -> np.ndarray:
    """Long, thin, near-axis-aligned structures: glass/coverslip edges and crop borders.
    Left in the mask they close loops with tissue, and fill_holes then floods whole regions
    (Case0074, Case0123, Case0200). Tissue is kept because its thick parts survive an
    opening with a `thick` px disc."""
    m8 = mask.astype(np.uint8)
    h, w = mask.shape
    hor = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((1, max(3, int(rel_len * w))), np.uint8))
    ver = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((max(3, int(rel_len * h)), 1), np.uint8))
    body = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick, thick)))
    lines = (hor | ver).astype(bool) & ~body.astype(bool)
    return cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def group_pieces(mask: np.ndarray, gap: int) -> np.ndarray:
    """Merge slide pieces separated by < gap px (tears/cracks) into fragment groups.
    Implemented as connected components of the dilated mask, labels transferred back."""
    k = _odd(gap)
    dil = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, lab = cv2.connectedComponents(dil)
    lab[~mask] = 0
    return lab.astype(np.int32)
