"""Toy experiment: does a matcher report *unmatched* block tissue, or absorb it?

Synthetic "biopsy" blocks (2-6 star-shaped fragments) are turned into synthetic
"slides" by: independent per-fragment float (rotation/translation), per-fragment
shrink, smooth elastic warp, optional tear (fragment split into two shifted
halves), optional partial face (one fragment shrunk to ~80%), then an unknown
global rotation + mirror flip + scale.  Each case is generated twice from the
same block: NEGATIVE (all fragments present) and POSITIVE (one fragment
removed from the slide).

Every method outputs a per-block-point "unmatched" indicator u_i in [0,1].
Scores:
  global  = mean(u)                     (unmatched-area fraction)
  cluster = mass of largest spatially connected set {u_i > .5} (fraction)
Localisation (positives) = IoU of that largest set with the removed fragment.

Methods
  cpd_w0        rigid+flip init, then plain non-rigid CPD (pycpd, w=0), slide->block
  cpd_w0.5      same, w=0.5   (scored by coverage distance)
  cpd_w0.5_post same run, scored by CPD's own outlier posterior 1-Pt1
  cpd_rev       block points as GMM centroids moving to slide; u = low P1 mass
  rigid_cov     global rigid+flip (directed chamfer slide->block), coverage
  piece_cov     rigid_cov + per-slide-piece rigid refinement, coverage
  piece_pw      piece_cov pose, then partial Wasserstein deficit
  partial_w     global rigid then partial Wasserstein (POT), block marginal deficit
  partial_gw    partial Gromov-Wasserstein (POT), NO alignment (rot/flip invariant)
  srgw          semi-relaxed GW (slide marginal fixed, block marginal free)

Requires: numpy, scipy, opencv-python(-headless), POT (`pip install pot`), pycpd.
Usage: python scripts/toy_partial_matching.py --n 60 --out results/toy_partial_matching
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import ot
from pycpd import DeformableRegistration
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

CANVAS = 512
PAD = 256  # padding for distance-transform lookups


# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------
def star_polygon(rng, cx, cy, R, n=96):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones_like(th)
    for k in range(2, 6):
        r += rng.uniform(0, 0.25 / (k - 1)) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    r *= R * rng.uniform(0.7, 1.0)
    ax = rng.uniform(0.6, 1.0)  # elongation (biopsy cores are elongated)
    phi = rng.uniform(0, np.pi)
    x, y = r * np.cos(th), ax * r * np.sin(th)
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def make_block(rng):
    k = rng.integers(2, 7)
    frags, masks = [], []
    occ = np.zeros((CANVAS, CANVAS), np.uint8)
    tries = 0
    while len(frags) < k and tries < 500:
        tries += 1
        R = float(np.exp(rng.uniform(np.log(18), np.log(75))))
        cx, cy = rng.uniform(R + 20, CANVAS - R - 20, 2)
        poly = star_polygon(rng, cx, cy, R)
        m = np.zeros_like(occ)
        cv2.fillPoly(m, [poly.round().astype(np.int32)], 1)
        if (cv2.dilate(m, np.ones((15, 15), np.uint8)) & occ).any():
            continue
        occ |= m
        frags.append(poly)
        masks.append(m)
    return frags, masks


def warp_poly(poly, rng, amp):
    """smooth elastic warp of polygon vertices"""
    out = poly.copy()
    for _ in range(2):
        f = rng.uniform(1 / 120, 1 / 50)
        ph = rng.uniform(0, 2 * np.pi, 2)
        out[:, 0] += amp * np.sin(2 * np.pi * f * poly[:, 1] + ph[0])
        out[:, 1] += amp * np.sin(2 * np.pi * f * poly[:, 0] + ph[1])
    return out


def make_slide(rng, frags, drop: int | None, tear: bool, partial_face: bool):
    big = 1024
    off = (big - CANVAS) / 2
    mask = np.zeros((big, big), np.uint8)
    tear_idx = int(np.argmax([cv2.contourArea(p.astype(np.float32)) for p in frags]))
    pf_idx = int(rng.integers(len(frags)))
    for i, poly in enumerate(frags):
        if i == drop:
            continue
        c = poly.mean(0)
        a = np.deg2rad(rng.uniform(-20, 20))
        sc = rng.uniform(0.92, 1.05)
        if partial_face and i == pf_idx:
            sc *= 0.8
        Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        p = (poly - c) @ Rm.T * sc + c + rng.normal(0, 15, 2)
        p = warp_poly(p, rng, amp=0.04 * np.sqrt(cv2.contourArea(p.astype(np.float32))))
        m = np.zeros_like(mask)
        cv2.fillPoly(m, [(p + off).round().astype(np.int32)], 1)
        if tear and i == tear_idx:
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean(), xs.mean()
            t = rng.uniform(0, np.pi)
            n = np.array([np.cos(t), np.sin(t)])
            side = ((xs - cx) * n[0] + (ys - cy) * n[1]) > 0
            half = np.zeros_like(m)
            half[ys[side], xs[side]] = 1
            m[ys[side], xs[side]] = 0
            sh = n * rng.uniform(10, 20)
            M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
            m |= cv2.warpAffine(half, M, (big, big), flags=cv2.INTER_NEAREST)
        mask |= m
    # global similarity + flip
    flip = rng.random() < 0.5
    ang = rng.uniform(0, 360)
    scale = rng.uniform(0.8, 1.25)
    M = cv2.getRotationMatrix2D((big / 2, big / 2), ang, scale)
    if flip:
        mask = mask[:, ::-1].copy()
    mask = cv2.warpAffine(mask, M, (big, big), flags=cv2.INTER_NEAREST)
    return mask, scale


def grid_points(mask, h, rng):
    H, W = mask.shape
    xs, ys = np.meshgrid(np.arange(h / 2, W, h), np.arange(h / 2, H, h))
    P = np.stack([xs.ravel(), ys.ravel()], 1) + rng.uniform(-h / 4, h / 4, (xs.size, 2))
    P = P[(P[:, 0] >= 0) & (P[:, 0] < W - 1) & (P[:, 1] >= 0) & (P[:, 1] < H - 1)]
    keep = mask[P[:, 1].astype(int), P[:, 0].astype(int)] > 0
    return P[keep]


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def dt_lookup(dt, pts):
    return map_coordinates(dt, [pts[..., 1] + PAD, pts[..., 0] + PAD], order=1, mode="nearest")


def global_rigid(S, block_mask):
    """Rotation/flip/translation of slide points S (already in block units) that
    minimises the *directed* chamfer slide->block (robust to missing block tissue)."""
    bm = np.pad(block_mask, PAD)
    dt = cv2.distanceTransform((bm == 0).astype(np.uint8), cv2.DIST_L2, 5)
    bc = np.argwhere(block_mask)[:, ::-1].mean(0)
    sub = S[np.random.default_rng(0).choice(len(S), min(200, len(S)), replace=False)]
    sc = sub.mean(0)
    g = np.arange(-80, 81, 16)
    T = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    cands = []
    for flip in (False, True):
        F = np.diag([-1.0, 1.0]) if flip else np.eye(2)
        for a in np.deg2rad(np.arange(0, 360, 6)):
            P = (sub - sc) @ (rot(a) @ F).T + bc
            cost = dt_lookup(dt, P[None] + T[:, None]).mean(1) ** 2
            j = int(np.argmin(cost))
            cands.append((cost[j], flip, a, T[j]))
    cands.sort(key=lambda c: c[0])
    best = None
    for _, flip, a, t in cands[:6]:
        F = np.diag([-1.0, 1.0]) if flip else np.eye(2)

        def f(x, F=F):
            P = (sub - sc) @ (rot(x[0]) @ F).T + bc + x[1:]
            return np.mean(dt_lookup(dt, P) ** 2)

        r = minimize(f, np.r_[a, t], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
        if best is None or r.fun < best[0]:
            best = (r.fun, F, r.x)
    _, F, x = best
    A = rot(x[0]) @ F
    return lambda Q: (Q - sc) @ A.T + bc + x[1:], dt


def piecewise_refine(S_al, labels, dt):
    """per slide piece: small rigid refinement minimising directed chamfer."""
    out = S_al.copy()
    for l in np.unique(labels):
        idx = labels == l
        P0 = S_al[idx]
        c = P0.mean(0)

        def f(x):
            P = (P0 - c) @ rot(x[0]).T + c + x[1:]
            return np.mean(dt_lookup(dt, P) ** 2) + 1e-3 * (x[1] ** 2 + x[2] ** 2)

        best = None
        for a0 in np.deg2rad([-20, 0, 20]):
            r = minimize(f, np.r_[a0, 0, 0], method="Nelder-Mead", options=dict(maxiter=300))
            if best is None or r.fun < best.fun:
                best = r
        x = best.x
        out[idx] = (P0 - c) @ rot(x[0]).T + c + x[1:]
    return out


def coverage_u(B, S_al, tau):
    d, _ = cKDTree(S_al).query(B)
    return (d > tau).astype(float)


def smooth_deficit(B, rho, h):
    """rho_i ~ 1 where block point i received its share of mass; smooth over 2h."""
    tree = cKDTree(B)
    nb = tree.query_ball_point(B, 2.0 * h)
    rs = np.array([rho[n].mean() for n in nb])
    return (rs < 0.5).astype(float)


def scores(B, u, h, removed=None):
    tree = cKDTree(B)
    on = np.nonzero(u > 0.5)[0]
    cl = np.zeros(len(B), bool)
    if len(on):
        pairs = cKDTree(B[on]).query_pairs(1.8 * h, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(on), len(on)))
        _, lab = connected_components(g, directed=False)
        big = np.bincount(lab).argmax()
        cl[on[lab == big]] = True
    out = dict(glob=float(u.mean()), cluster=float(cl.mean()))
    if removed is not None:
        inter = (cl & removed).sum()
        uni = (cl | removed).sum()
        out["loc_iou"] = inter / uni if uni else 0.0
    return out


# ----------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------
def run_methods(B, S, s_labels, block_mask, h):
    res = {}
    t0 = time.perf_counter()
    tf, dt = global_rigid(S, block_mask)
    S_al = tf(S)
    tau = 2.0 * h
    res["rigid_cov"] = coverage_u(B, S_al, tau)
    S_pc = piecewise_refine(S_al, s_labels, dt)
    res["piece_cov"] = coverage_u(B, S_pc, tau)
    t_align = time.perf_counter() - t0

    # Partial W after piecewise-rigid pose (Pipeline A pose + Pipeline B score)
    a0 = np.full(len(B), 1 / len(B))
    b0 = np.full(len(S), 1 / len(B))
    M0 = ot.dist(B, S_pc)
    T0 = ot.partial.partial_wasserstein(a0, b0, M0 / M0.max(), m=0.97 * min(a0.sum(), b0.sum()), nb_dummies=50)
    res["piece_pw"] = smooth_deficit(B, T0.sum(1) * len(B), h)

    # CPD (normalised coords, pycpd defaults alpha=beta=2)
    mu, sd = B.mean(0), np.sqrt(((B - B.mean(0)) ** 2).sum(1).mean())
    Bn, Sn = (B - mu) / sd, (S_al - mu) / sd
    for w in (0.0, 0.5):
        reg = DeformableRegistration(X=Bn, Y=Sn, w=w, max_iterations=150)
        TY, _ = reg.register()
        res[f"cpd_w{w:g}"] = coverage_u(Bn, TY, tau / sd)
        if w > 0:
            res[f"cpd_w{w:g}_post"] = (reg.Pt1 < 0.5).astype(float)
    # reversed roles: block = moving GMM centroids, slide = data
    reg = DeformableRegistration(X=Sn, Y=Bn, w=0.0, max_iterations=150)
    reg.register()
    rho = reg.P1 * len(Bn) / max(len(Sn), 1)  # ~1 when each centroid gets its share
    res["cpd_rev"] = smooth_deficit(B, rho, h)

    # Partial Wasserstein after global rigid (mass per point = 1/N_B -> equal density)
    a = np.full(len(B), 1 / len(B))
    b = np.full(len(S), 1 / len(B))
    m = 0.97 * min(a.sum(), b.sum())
    M = ot.dist(B, S_al)
    T = ot.partial.partial_wasserstein(a, b, M / M.max(), m=m, nb_dummies=50)
    res["partial_w"] = smooth_deficit(B, T.sum(1) * len(B), h)

    # Partial GW: no alignment at all (intra-set distances only)
    C1, C2 = ot.dist(B, B, metric="euclidean"), ot.dist(S, S, metric="euclidean")
    nrm = max(C1.max(), C2.max())
    T = ot.gromov.partial_gromov_wasserstein(C1 / nrm, C2 / nrm, a, b, m=m, nb_dummies=50, numItermax=200000, warn=False)
    res["partial_gw"] = smooth_deficit(B, T.sum(1) * len(B), h)

    # semi-relaxed GW: slide marginal fixed, block marginal free
    T = ot.gromov.semirelaxed_gromov_wasserstein(C2 / nrm, C1 / nrm, np.full(len(S), 1 / len(S)), max_iter=2000)
    rho = T.sum(0) * len(S)  # ~1 slide point per covered block point (equal density)
    res["srgw"] = smooth_deficit(B, rho, h)
    return res, t_align


# ----------------------------------------------------------------------------
def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.r_[pos, neg]
    ranks = allv.argsort().argsort() + 1.0
    # average ties
    for v in np.unique(allv):
        k = allv == v
        ranks[k] = ranks[k].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def sens_at_spec(pos, neg, spec=0.95):
    thr = np.quantile(neg, spec)
    return float(np.mean(np.asarray(pos) > thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_pts", type=int, default=300)
    ap.add_argument("--out", default="results/toy_partial_matching")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for case in range(args.n):
        frags, masks = make_block(rng)
        if len(frags) < 2:
            continue
        block_mask = np.clip(sum(masks), 0, 1).astype(np.uint8)
        h = np.sqrt(block_mask.sum() / args.target_pts)
        B = grid_points(block_mask, h, rng)
        frag_id = np.full(len(B), -1)
        for i, m in enumerate(masks):
            frag_id[m[B[:, 1].astype(int), B[:, 0].astype(int)] > 0] = i
        tear = rng.random() < 0.4
        pface = rng.random() < 0.3
        # half the positives drop the SMALLEST fragment (small omissions matter most)
        areas = [m.sum() for m in masks]
        drop = int(np.argmin(areas)) if rng.random() < 0.5 else int(rng.integers(len(frags)))
        st = rng.bit_generator.state
        for positive in (False, True):
            rng.bit_generator.state = st  # identical deformations for the pair
            smask, scale = make_slide(rng, frags, drop if positive else None, tear, pface)
            s_est = scale * rng.uniform(0.92, 1.08)  # scale known only to +-8%
            S = grid_points(smask, h * s_est, rng) / s_est
            _, lab = cv2.connectedComponents(smask)
            s_labels = lab[(S[:, 1] * s_est).astype(int), (S[:, 0] * s_est).astype(int)]
            removed = frag_id == drop if positive else None
            t0 = time.perf_counter()
            res, _ = run_methods(B, S, s_labels, block_mask, h)
            dt_ = time.perf_counter() - t0
            miss_frac = float(masks[drop].sum() / block_mask.sum()) if positive else 0.0
            for meth, u in res.items():
                sc = scores(B, u, h, removed)
                rows.append(dict(case=case, positive=int(positive), method=meth, n_frag=len(frags),
                                 tear=int(tear), pface=int(pface), miss_frac=miss_frac,
                                 NB=len(B), NS=len(S), **sc))
            print(f"case {case:3d} pos={int(positive)} K={len(frags)} NB={len(B)} NS={len(S)} "
                  f"miss={miss_frac:.3f} tear={int(tear)} pf={int(pface)} {dt_:.1f}s", flush=True)
        if case == 0:
            np.savez(out / "example_case0.npz", B=B, S=S, frag_id=frag_id)
    with open(out / "per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + (["loc_iou"] if "loc_iou" not in rows[0] else []))
        w.writeheader()
        w.writerows(rows)
    summarise(rows, out)


def summarise(rows, out):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    lines = ["method,score,AUROC_all,Sens@95Spec_all,AUROC_miss<5%,AUROC_miss5-15%,AUROC_miss>15%,"
             "mean_neg_score,mean_pos_score,mean_loc_IoU"]
    for m in methods:
        R = [r for r in rows if r["method"] == m]
        neg = [r for r in R if not r["positive"]]
        pos = [r for r in R if r["positive"]]
        for key in ("glob", "cluster"):
            nv = [r[key] for r in neg]
            bins = []
            for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 1.01)):
                pv = [r[key] for r in pos if lo <= r["miss_frac"] < hi]
                bins.append(f"{auroc(pv, nv):.3f}" if pv else "nan")
            pv = [r[key] for r in pos]
            loc = np.mean([r["loc_iou"] for r in pos])
            lines.append(f"{m},{key},{auroc(pv, nv):.3f},{sens_at_spec(pv, nv):.3f},{','.join(bins)},"
                         f"{np.mean(nv):.3f},{np.mean(pv):.3f},{loc:.3f}")
    txt = "\n".join(lines)
    (out / "summary.csv").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
