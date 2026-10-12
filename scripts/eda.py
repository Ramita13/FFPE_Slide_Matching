#!/usr/bin/env python
"""Exploratory data analysis for FFPE block (LK) / slide (QP) image pairs.

Computes simple, interpretable per-image and per-pair statistics that inform
method design (segmentation difficulty, fragment counts, specimen type, image
quality), writes them to a CSV, renders summary plots and, optionally, visual
contact sheets (image + rough tissue mask) for chosen cases.

Per image (prefix ``lk_`` for blocks, ``qp_`` for slides):
    h, w, aspect (w/h), elong (long/short side), brightness (mean grey),
    contrast (grey std), blur (variance of Laplacian at a fixed 512 px long
    side; *lower = blurrier*), glare (fraction of near-saturated, low-chroma
    pixels), tissue_frac (rough mask area / image area), n_frag (connected
    components >= ``--min-frag-frac`` of image area), n_frag_small (components
    between ``--min-small-frac`` and ``--min-frag-frac``), largest_frac
    (largest component / total tissue), solidity of the union mask.
Slides additionally get ``qp_n_similar_pairs``: number of component pairs with
near-identical shape and size (proxy for repeated levels / ribbons), and
``qp_pen_frac`` (fraction of strongly blue/green/black ink-like pixels).

Per pair:
    elong_ratio = qp_elong / lk_elong, frag_diff = qp_n_frag - lk_n_frag,
    area_ratio = qp_tissue_frac / lk_tissue_frac,
    orient_swap = 1 if both crops are elongated (>1.2) but in different
    orientations (portrait vs landscape), i.e. an apparent ~90 deg rotation,
    specimen_proxy = "biopsy" / "resection" heuristic (see specimen_proxy()),
    crop_scale = sqrt(qp crop area / lk crop area) (~1 when both crops frame
    the same tissue at the same pixel scale).
Both: border_touch = fraction of the outermost pixel ring covered by tissue
(high = tissue probably truncated by the ROI crop).

The tissue masks are deliberately simple (thresholding + morphology) and are
NOT meant as a production segmentation; they are a proxy for EDA only.

Block mask: the wax colour is estimated from the image border (median Lab of
the brightest/least-chromatic half of a border ring, since tight crops mean the
border often touches tissue). Tissue = pixels whose (chroma-weighted) Lab
distance from the wax colour exceeds max(Otsu threshold, a floor). Glare
pixels are treated as background.

Slide mask: HSV saturation above max(Otsu, floor) OR clearly darker than the
background, followed by morphology.

Usage::

    python scripts/eda.py --data-dir data --out-dir results/eda
    python scripts/eda.py --data-dir data --out-dir results/eda \
        --montage Case0001 Case0002 --montage-only

"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

NORM_SIDE = 512  # long side used for scale-dependent metrics (blur)


# ----------------------------------------------------------------------------
# Image loading / generic metrics
# ----------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB (cv2.imread fails on some unicode paths)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long(img: np.ndarray, side: int = NORM_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = side / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def basic_stats(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grey_n = cv2.cvtColor(resize_long(rgb), cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    glare = (hsv[..., 2] >= 245) & (hsv[..., 1] <= 30)
    return {
        "h": h,
        "w": w,
        "aspect": w / h,
        "elong": max(h, w) / min(h, w),
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "blur": float(cv2.Laplacian(grey_n, cv2.CV_64F).var()),
        "glare": float(glare.mean()),
    }


# ----------------------------------------------------------------------------
# Rough tissue masks
# ----------------------------------------------------------------------------
def _border_ring(img: np.ndarray, frac: float = 0.04) -> np.ndarray:
    h, w = img.shape[:2]
    b = max(2, int(round(min(h, w) * frac)))
    ring = np.zeros((h, w), bool)
    ring[:b] = ring[-b:] = True
    ring[:, :b] = ring[:, -b:] = True
    return ring


def _otsu(values: np.ndarray) -> float:
    v = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _clean(mask: np.ndarray, open_frac: float = 0.006, close_frac: float = 0.012) -> np.ndarray:
    """Open/close with kernel sizes relative to image size, then fill holes."""
    h, w = mask.shape
    s = min(h, w)
    ko = max(3, int(s * open_frac) | 1)
    kc = max(3, int(s * close_frac) | 1)
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    # fill holes: flood from border on the inverted mask
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def block_tissue_mask(rgb: np.ndarray, floor: float = 14.0) -> tuple[np.ndarray, dict]:
    """Tissue mask for a paraffin-block photo via distance from wax colour."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    glare = (hsv[..., 2] >= 245) & (hsv[..., 1] <= 30)

    ring = _border_ring(rgb) & ~glare
    if ring.sum() < 50:
        ring = _border_ring(rgb)
    chroma = np.hypot(a, b)
    # wax = bright and low-chroma; keep the "most wax-like" half of the ring
    score = L[ring] - 2.0 * chroma[ring]
    keep = score >= np.median(score)
    bgL = float(np.median(L[ring][keep]))
    bga = float(np.median(a[ring][keep]))
    bgb = float(np.median(b[ring][keep]))

    # chroma-weighted distance; tissue is browner/yellower and usually darker
    dist = np.sqrt(0.5 * (L - bgL) ** 2 + 2.0 * ((a - bga) ** 2 + (b - bgb) ** 2))
    dist[glare] = 0
    t = max(_otsu(dist), floor)
    mask = _clean(dist > t)
    return mask, {"bg_L": bgL, "bg_a": bga, "bg_b": bgb, "thr": t}


def slide_tissue_mask(rgb: np.ndarray, floor: float = 25.0) -> tuple[np.ndarray, dict]:
    """Tissue mask for an H&E preview.

    Two components are OR-ed:
      * "stained" tissue: HSV saturation above max(Otsu, floor) or clearly
        darker than the background;
      * "pale" tissue (fat, oedematous stroma, mucus): smoothed saturation a
        few units above the background median OR elevated local grey-level
        texture. Fat is almost white in these previews (S ~10 vs ~8 for glass),
        so a pure colour threshold misses it.
    ``pale_frac`` = fraction of the final mask contributed only by the pale rule.
    ``stained_frac`` = area fraction of the stained-only mask (near zero when
    the crop contains no H&E-stained tissue at all, e.g. a mis-crop).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ring = _border_ring(rgb)
    bg_v = float(np.percentile(v[ring], 75))
    bg_s = float(np.median(s[ring]))
    t = max(_otsu(s), floor)
    stained = (s > t) | (v < bg_v - 60)

    sig = max(1.5, min(rgb.shape[:2]) / 120)
    s_smooth = cv2.GaussianBlur(s, (0, 0), sig) - bg_s
    mu = cv2.blur(g, (9, 9))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - mu * mu, 0))
    sd = cv2.GaussianBlur(sd, (0, 0), sig)
    bg_sd = float(np.median(sd[ring]))
    pale = (s_smooth > 3.0) | (sd > bg_sd + 3.0)

    strong = _clean(stained, open_frac=0.004, close_frac=0.010)
    mask = _clean(stained | pale, open_frac=0.008, close_frac=0.03)
    pale_frac = float((mask & ~strong).sum() / max(mask.sum(), 1))
    return mask, {"bg_V": bg_v, "bg_S": bg_s, "thr": t, "pale_frac": pale_frac,
                  "stained_frac": float(strong.mean())}


def pen_fraction(rgb: np.ndarray) -> float:
    """Fraction of ink-like pixels (blue/green hues or near-black), slides only."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    # OpenCV hue in [0,180): green ~35-85, blue ~90-125 (H&E purple is ~125-170)
    ink = ((hue >= 35) & (hue <= 120) & (sat > 60)) | (val < 50)
    return float(ink.mean())


# ----------------------------------------------------------------------------
# Component analysis
# ----------------------------------------------------------------------------
def components(mask: np.ndarray, min_frac: float, small_frac: float):
    """Return (stats dict, label image, ids of components >= min_frac)."""
    h, w = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    big = areas >= min_frac * h * w
    small = (areas >= small_frac * h * w) & ~big
    total = areas.sum()
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    out = {
        "border_touch": float(edge.mean()),
        "tissue_frac": float(total / (h * w)),
        "n_frag": int(big.sum()),
        "n_frag_small": int(small.sum()),
        "largest_frac": float(areas.max() / total) if total > 0 else 0.0,
    }
    # solidity of union mask (area / convex hull area); low = scattered / irregular
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is not None and len(pts) >= 3:
        hull = cv2.convexHull(pts)
        ha = cv2.contourArea(hull)
        out["solidity"] = float(total / ha) if ha > 0 else np.nan
    else:
        out["solidity"] = np.nan
    return out, lab, np.where(big)[0] + 1


def similar_component_pairs(labels: np.ndarray, ids: np.ndarray,
                            shape_tol: float = 0.15, area_tol: float = 0.75) -> int:
    """Count pairs of large components with similar size and Hu-moment shape.

    A crude proxy for slides carrying several levels / ribbons of the same tissue.
    """
    cnts = []
    for i in ids:
        m = (labels == i).astype(np.uint8)
        c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if c:
            c = max(c, key=cv2.contourArea)
            cnts.append((cv2.contourArea(c), c))
    n = 0
    for (a1, c1), (a2, c2) in itertools.combinations(cnts, 2):
        if min(a1, a2) / max(a1, a2) < area_tol:
            continue
        if cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0) < shape_tol:
            n += 1
    return n


# ----------------------------------------------------------------------------
# Per-case processing
# ----------------------------------------------------------------------------
def analyse_case(args: tuple) -> dict:
    case, lk_path, qp_path, min_frac, small_frac = args
    row: dict = {"case": case}
    for prefix, path in (("lk", lk_path), ("qp", qp_path)):
        if path is None:
            continue
        try:
            rgb = load_rgb(path)
        except IOError as e:  # pragma: no cover
            print(e, file=sys.stderr)
            continue
        st = basic_stats(rgb)
        if prefix == "qp":  # on a slide, "glare" is just white glass background
            st["white_frac"] = st.pop("glare")
        row.update({f"{prefix}_{k}": v for k, v in st.items()})
        if prefix == "lk":
            mask, info = block_tissue_mask(rgb)
        else:
            mask, info = slide_tissue_mask(rgb)
            row["qp_pen_frac"] = pen_fraction(rgb)
        row.update({f"{prefix}_{k}": v for k, v in info.items()})
        comp, labels, ids = components(mask, min_frac, small_frac)
        row.update({f"{prefix}_{k}": v for k, v in comp.items()})
        if prefix == "qp":
            row["qp_n_similar_pairs"] = similar_component_pairs(labels, ids)
    if "lk_h" in row and "qp_h" in row:
        row["elong_ratio"] = row["qp_elong"] / row["lk_elong"]
        row["frag_diff"] = row["qp_n_frag"] - row["lk_n_frag"]
        long_side = max(row["lk_h"], row["lk_w"])
        nf = max(row["lk_n_frag"], row["qp_n_frag"])
        row["orient_swap"] = int((row["lk_elong"] > 1.2 and row["qp_elong"] > 1.2)
                                 and ((row["lk_aspect"] > 1) != (row["qp_aspect"] > 1)))
        row["specimen_proxy"] = specimen_proxy(long_side, nf)
        row["crop_scale"] = float(np.sqrt(row["qp_h"] * row["qp_w"] / (row["lk_h"] * row["lk_w"])))
        row["area_ratio"] = (row["qp_tissue_frac"] / row["lk_tissue_frac"]
                             if row["lk_tissue_frac"] > 0 else np.nan)
    return row


def specimen_proxy(lk_long_side: float, n_frag: float) -> str:
    """Rough biopsy / resection label from crop size and fragment count.

    Crops are tight around the tissue and LK/QP share a common pixel scale
    (~1 px per slide-preview pixel), so the block crop's long side is a proxy
    for physical specimen size. Thresholds were chosen to reproduce the ~28%
    small-biopsy share reported for the source study; see
    docs/research/05_dataset_eda.md.
    """
    if lk_long_side < 250 or (lk_long_side < 400 and n_frag >= 4):
        return "biopsy"
    return "resection"


def list_cases(data_dir: Path) -> list[tuple[str, Path | None, Path | None]]:
    lk = {p.name.split("_")[0]: p for p in (data_dir / "LK").glob("*_LK.jpg")}
    qp = {p.name.split("_")[0]: p for p in (data_dir / "QP").glob("*_QP.jpg")}
    return [(c, lk.get(c), qp.get(c)) for c in sorted(set(lk) | set(qp))]


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 200, 0), 2)
    return out


def montage(data_dir: Path, cases: list[str], out_path: Path, with_masks: bool = True,
            tile: int = 300) -> None:
    """Render a contact sheet: one row per case, [LK | LK-mask | QP | QP-mask]."""
    rows = []
    for case in cases:
        tiles = []
        for mod in ("LK", "QP"):
            p = data_dir / mod / f"{case}_{mod}.jpg"
            if not p.exists():
                tiles += [np.full((tile, tile, 3), 128, np.uint8)] * (2 if with_masks else 1)
                continue
            rgb = load_rgb(p)
            mask = (block_tissue_mask if mod == "LK" else slide_tissue_mask)(rgb)[0]
            imgs = [rgb, _overlay(rgb, mask)] if with_masks else [rgb]
            for im in imgs:
                im = resize_long(im, tile)
                canvas = np.full((tile, tile, 3), 40, np.uint8)
                canvas[:im.shape[0], :im.shape[1]] = im
                tiles.append(canvas)
        row = np.concatenate(tiles, axis=1)
        cv2.putText(row, case, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def summary_plots(df, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def hist(ax, cols, title, log=False, bins=60):
        for c, lbl in cols:
            v = df[c].dropna()
            if log:
                v = np.log10(v.clip(lower=1e-6))
            ax.hist(v, bins=bins, alpha=0.6, label=lbl)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    fig, axs = plt.subplots(3, 4, figsize=(16, 10))
    axs = axs.ravel()
    hist(axs[0], [("lk_elong", "block"), ("qp_elong", "slide")], "elongation (long/short side)")
    hist(axs[1], [("lk_brightness", "block"), ("qp_brightness", "slide")], "mean brightness")
    hist(axs[2], [("lk_contrast", "block"), ("qp_contrast", "slide")], "contrast (grey std)")
    hist(axs[3], [("lk_blur", "block"), ("qp_blur", "slide")], "log10 Laplacian variance (blur)", log=True)
    hist(axs[4], [("lk_glare", "block glare"), ("qp_white_frac", "slide white bg")], "glare / white fraction")
    hist(axs[5], [("lk_tissue_frac", "block"), ("qp_tissue_frac", "slide")], "tissue area fraction")
    for i, (c, lbl) in enumerate([("lk_n_frag", "block"), ("qp_n_frag", "slide")]):
        v = df[c].dropna().clip(upper=20)
        axs[6].hist(v + (i * 0.4 - 0.2), bins=np.arange(-0.5, 21.5, 1), alpha=0.6, label=lbl)
    axs[6].set_title("fragment count (clipped at 20)", fontsize=9)
    axs[6].legend(fontsize=7)
    axs[7].hist(df["frag_diff"].dropna().clip(-15, 15), bins=np.arange(-15.5, 16.5, 1))
    axs[7].set_title("frag_diff = slide - block", fontsize=9)
    hist(axs[8], [("area_ratio", "slide/block")], "log10 tissue area ratio", log=True)
    hist(axs[9], [("elong_ratio", "slide/block")], "elongation ratio")
    axs[10].scatter(df["lk_n_frag"], df["qp_n_frag"], s=4, alpha=0.3)
    axs[10].plot([0, 15], [0, 15], "k--", lw=0.8)
    axs[10].set_xlim(0, 15)
    axs[10].set_ylim(0, 15)
    axs[10].set_xlabel("block n_frag")
    axs[10].set_ylabel("slide n_frag")
    axs[10].set_title("fragment counts", fontsize=9)
    axs[11].scatter(df["lk_tissue_frac"], df["qp_tissue_frac"], s=4, alpha=0.3,
                    c=np.log10(df["qp_n_frag"].clip(lower=1)), cmap="viridis")
    axs[11].set_xlabel("block tissue frac")
    axs[11].set_ylabel("slide tissue frac")
    axs[11].set_title("tissue fraction (colour = log slide n_frag)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_summary.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["qp_n_frag"].clip(upper=40), df["qp_solidity"], s=5, alpha=0.35)
    ax.set_xlabel("slide n_frag (clipped 40)")
    ax.set_ylabel("slide union-mask solidity")
    ax.set_title("specimen-type proxy space")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_specimen_space.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="folder containing LK/ and QP/")
    ap.add_argument("--out-dir", type=Path, default=Path("results/eda"))
    ap.add_argument("--min-frag-frac", type=float, default=0.002,
                    help="min component area (fraction of image) counted as a fragment")
    ap.add_argument("--min-small-frac", type=float, default=0.0003,
                    help="min area for the 'small fragment' count")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N cases")
    ap.add_argument("--montage", nargs="*", default=None,
                    help="case IDs to render as a contact sheet (montage_<first>.png)")
    ap.add_argument("--montage-name", default=None, help="file name for the contact sheet")
    ap.add_argument("--montage-only", action="store_true", help="skip statistics, only render montage")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.montage:
        name = args.montage_name or f"montage_{args.montage[0]}.png"
        montage(args.data_dir, args.montage, args.out_dir / name)
        print(f"wrote {args.out_dir / name}")
        if args.montage_only:
            return 0

    import pandas as pd

    cases = list_cases(args.data_dir)
    if args.limit:
        cases = cases[: args.limit]
    jobs = [(c, lk, qp, args.min_frag_frac, args.min_small_frac) for c, lk, qp in cases]
    print(f"analysing {len(jobs)} cases with {args.workers} workers ...")
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(analyse_case, jobs, chunksize=16))
    else:
        rows = [analyse_case(j) for j in jobs]
    df = pd.DataFrame(rows).set_index("case")
    csv = args.out_dir / "eda_stats.csv"
    df.to_csv(csv, float_format="%.5g")
    print(f"wrote {csv} ({len(df)} rows)")

    summary_plots(df.dropna(subset=["frag_diff"]), args.out_dir)
    q = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    q.to_csv(args.out_dir / "eda_quantiles.csv", float_format="%.4g")
    print(f"wrote plots and quantiles to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
