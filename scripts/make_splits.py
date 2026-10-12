"""Create fixed case-level splits (60/20/20), stratified by the EDA specimen proxy.

Also marks a 200-case development subset of the validation split ('dev') used for cheap
iteration; the test split is not touched until final evaluation.

Usage:  python scripts/make_splits.py      ->  splits/splits.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ffpe_match.data import list_pairs  # noqa: E402

SEED, DEV_N = 20261012, 200


def main() -> None:
    proxy = {}
    with open(ROOT / "results" / "eda" / "eda_stats.csv", newline="") as f:
        for r in csv.DictReader(f):
            proxy[r["case"]] = r["specimen_proxy"] or "unknown"
    cases = list_pairs()
    rng = np.random.default_rng(SEED)
    split = {}
    for stratum in sorted({proxy.get(c, "unknown") for c in cases}):
        cs = [c for c in cases if proxy.get(c, "unknown") == stratum]
        rng.shuffle(cs)
        n = len(cs)
        for i, c in enumerate(cs):
            split[c] = "train" if i < 0.6 * n else ("val" if i < 0.8 * n else "test")
    val = sorted(c for c in cases if split[c] == "val")
    dev = set(rng.choice(val, size=min(DEV_N, len(val)), replace=False))

    out = ROOT / "splits" / "splits.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "split", "dev", "specimen_proxy"])
        for c in cases:
            w.writerow([c, split[c], int(c in dev), proxy.get(c, "unknown")])
    counts = {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")}
    print(f"{len(cases)} cases -> {counts}, dev={len(dev)}  ->  {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
