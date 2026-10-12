"""Dataset access: block (LK) / slide (QP) pairs keyed by case ID."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def list_pairs(data_dir: Path = DATA_DIR) -> list[str]:
    """Case IDs (e.g. 'Case0001') that have both an LK and a QP image."""
    lk = {p.name[:-len("_LK.jpg")] for p in (data_dir / "LK").glob("*_LK.jpg")}
    qp = {p.name[:-len("_QP.jpg")] for p in (data_dir / "QP").glob("*_QP.jpg")}
    return sorted(lk & qp)


def load_pair(case: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return (block_bgr, slide_bgr) uint8 images."""
    lk = cv2.imread(str(data_dir / "LK" / f"{case}_LK.jpg"), cv2.IMREAD_COLOR)
    qp = cv2.imread(str(data_dir / "QP" / f"{case}_QP.jpg"), cv2.IMREAD_COLOR)
    if lk is None or qp is None:
        raise FileNotFoundError(case)
    return lk, qp
