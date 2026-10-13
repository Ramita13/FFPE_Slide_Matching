"""Common evaluation harness for missing-tissue detectors.

A *method* is a picklable callable ``method(ref, query) -> dict`` taking two bool masks and
returning at least ``{"score": float}`` (higher = more likely missing). It may also return
``"missing_map"``: a bool mask in the *reference* frame, used for localisation IoU.

Tracks
------
self   reference = clean slide mask, query = synthetic slide mask. Isolates the matching logic
       from cross-modal segmentation and has exact localisation ground truth.
cross  reference = block mask, query = synthetic slide mask. The realistic setting
       (block segmentation errors included); detection metrics only.

Each case yields ``n_per_case`` samples; even indices are negatives, odd are positives.
"""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import ROOT
from .masks import assessable, get_masks
from .synth import SynthConfig, generate, sample_seed

AREA_BINS = [0.0, 0.02, 0.05, 0.10, 0.25, 1.01]


def load_split(split: str, dev_only: bool = False) -> list[tuple[str, str]]:
    """[(case, specimen_proxy)] for a split ('train'|'val'|'test')."""
    with open(ROOT / "splits" / "splits.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if dev_only:
        rows = [r for r in rows if r["dev"] == "1"]
    return [(r["case"], r["specimen_proxy"]) for r in rows]


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def _run_case(args):
    method, case, specimen, track, n_per_case, cfg, seg = args
    m = get_masks(case, **seg)
    if not assessable(m):
        return [{"case": case, "specimen": specimen, "assessable": 0}]
    ref = m["slide_mask"] if track == "self" else m["block_mask"]
    rows = []
    for i in range(n_per_case):
        s = generate(m["slide_mask"], m["slide_groups"], label=i % 2,
                     seed=sample_seed(case, i), cfg=cfg)
        out = method(ref.astype(bool), s.query)
        row = {"case": case, "specimen": specimen, "assessable": 1, "idx": i,
               "label": s.label, "missing_frac": round(s.missing_frac, 5),
               "mode": s.meta.get("mode", "none"), "tear": int(s.meta.get("tear", False)),
               "shrink": int(s.meta.get("shrink", False)), "score": float(out["score"])}
        row.update({k: float(v) for k, v in out.items() if k.startswith("x_")})
        if track == "self" and s.label and out.get("missing_map") is not None:
            row["iou"] = _iou(out["missing_map"].astype(bool), s.removed_ref)
        rows.append(row)
    return rows


def run(method, split: str = "val", track: str = "self", dev_only: bool = True,
        n_per_case: int = 2, cfg: SynthConfig = SynthConfig(), workers: int = 8,
        seg: dict | None = None) -> list[dict]:
    """seg: segmenter choice passed to get_masks, e.g. {"slide": "grandqc"}."""
    jobs = [(method, c, sp, track, n_per_case, cfg, seg or {}) for c, sp in load_split(split, dev_only)]
    with ProcessPoolExecutor(workers) as ex:
        return [r for rows in ex.map(_run_case, jobs, chunksize=4) for r in rows]


def sens_at_spec(y, s, spec=0.95):
    thr = np.quantile(s[y == 0], spec)
    return float((s[y == 1] > thr).mean()), float(thr)


def metrics(rows: list[dict]) -> dict:
    r = [x for x in rows if x.get("assessable")]
    y = np.array([x["label"] for x in r])
    s = np.array([x["score"] for x in r])
    frac = np.array([x["missing_frac"] for x in r])
    sp = np.array([x["specimen"] for x in r])
    sens, thr = sens_at_spec(y, s)
    out = {"n_samples": len(r), "n_unassessable_cases": sum(1 for x in rows if not x.get("assessable")),
           "auroc": float(roc_auc_score(y, s)), "sens@95spec": sens, "thr@95spec": thr,
           "by_area": {}, "by_specimen": {}}
    neg = y == 0
    for lo, hi in zip(AREA_BINS[:-1], AREA_BINS[1:]):
        pos = (y == 1) & (frac >= lo) & (frac < hi)
        if pos.sum() >= 5:
            yy, ss = np.r_[np.zeros(neg.sum()), np.ones(pos.sum())], np.r_[s[neg], s[pos]]
            out["by_area"][f"{lo:.0%}-{min(hi, 1):.0%}"] = {
                "n_pos": int(pos.sum()), "auroc": float(roc_auc_score(yy, ss)),
                "sens@thr": float((s[pos] > thr).mean())}
    for g in sorted(set(sp)):
        k = sp == g
        if len(set(y[k])) == 2:
            out["by_specimen"][g] = {"n": int(k.sum()), "auroc": float(roc_auc_score(y[k], s[k]))}
    ious = [x["iou"] for x in r if "iou" in x and not np.isnan(x["iou"])]
    if ious:
        out["loc_iou_mean"] = float(np.mean(ious))
    return out


def save(rows: list[dict], m: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for x in rows for k in x})
    with open(out_dir / "per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
