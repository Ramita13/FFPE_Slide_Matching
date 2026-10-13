"""EXP-002: P1 rigid pose search + per-piece refinement + coverage (self & cross tracks).

Usage:
  python experiments/exp002_p1_coverage.py --track self --tol 0.02 --work 160
  python experiments/exp002_p1_coverage.py --track cross
Results: results/exp002_p1/<track>_w<work>_t<tol>_o<open>_s-<slide seg>_b-<block seg>/
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ffpe_match import evaluate as ev  # noqa: E402
from ffpe_match import match_coverage as mc  # noqa: E402


def method(ref, query, work, tol, open_frac):
    mc.WORK = work  # module-level setting; each worker process sets it per call
    return mc.coverage_score(ref, query, tol=tol, open_frac=open_frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="self", choices=["self", "cross"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--work", type=int, default=160)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--open", type=float, default=0.5)
    ap.add_argument("--slide-seg", default="classical", help="classical | grandqc")
    ap.add_argument("--block-seg", default="classical", help="classical | <cached variant>")
    args = ap.parse_args()
    fn = functools.partial(method, work=args.work, tol=args.tol, open_frac=args.open)
    rows = ev.run(fn, split=args.split, track=args.track, dev_only=not args.full, n_per_case=4,
                  seg={"slide": args.slide_seg, "block": args.block_seg})
    m = ev.metrics(rows)
    tag = f"{args.track}_w{args.work}_t{args.tol}_o{args.open}_s-{args.slide_seg}_b-{args.block_seg}"
    ev.save(rows, m, ROOT / "results" / "exp002_p1" / tag)
    print(tag, json.dumps({k: round(m[k], 3) for k in ("auroc", "sens@95spec")}),
          "iou", round(m.get("loc_iou_mean", float("nan")), 3))
    print("   by area:", {k: round(v["auroc"], 3) for k, v in m["by_area"].items()})
    print("   by specimen:", {k: round(v["auroc"], 3) for k, v in m["by_specimen"].items()})


if __name__ == "__main__":
    main()
