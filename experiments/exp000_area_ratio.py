"""EXP-000: area-ratio sanity baseline.

score = 1 - area(query) / area(reference). No alignment, no shape information. Its purpose
is to measure how much of the benchmark can be solved by tissue area alone: if this scores
near 1.0 AUROC, the nuisance pipeline is too weak and the benchmark is too easy.

Usage:  python experiments/exp000_area_ratio.py [--split val] [--full]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ffpe_match import evaluate as ev  # noqa: E402


def area_ratio(ref, query):
    return {"score": 1.0 - query.sum() / max(ref.sum(), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--full", action="store_true", help="whole split instead of the dev subset")
    args = ap.parse_args()
    for track in ("self", "cross"):
        rows = ev.run(area_ratio, split=args.split, track=track, dev_only=not args.full, n_per_case=4)
        m = ev.metrics(rows)
        ev.save(rows, m, ROOT / "results" / "exp000_area_ratio" / track)
        print(track, json.dumps({k: m[k] for k in ("n_samples", "n_unassessable_cases", "auroc", "sens@95spec")}))
        print("   by area:", {k: round(v["auroc"], 3) for k, v in m["by_area"].items()})


if __name__ == "__main__":
    main()
