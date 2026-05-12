from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OfficialWordGeometry:
    image_id: str
    words: np.ndarray
    geometry: np.ndarray


def parse_word_geometry_file(path: Path) -> OfficialWordGeometry:
    words: list[int] = []
    geometry: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        words.append(int(parts[0]))
        geometry.append([float(value) for value in parts[1:]])

    word_array = np.asarray(words, dtype=np.int32)
    # VGG official Oxford word ids are 1-based: [1, 1000000].
    # The sparse matrix code uses Python/NumPy 0-based column ids.
    if word_array.size:
        word_array = word_array - 1

    return OfficialWordGeometry(
        image_id=path.stem,
        words=word_array,
        geometry=np.asarray(geometry, dtype=np.float32),
    )


def list_official_word_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix == ".txt")


def official_geometry_to_keypoints(geometry: np.ndarray) -> np.ndarray:
    """Convert official [x, y, A, B, C] geometry to the project's keypoint layout."""
    if geometry.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    keypoints = np.zeros((len(geometry), 7), dtype=np.float32)
    keypoints[:, 0] = geometry[:, 0]
    keypoints[:, 1] = geometry[:, 1]
    # Approximate scale from ellipse area. This is only used for compatibility
    # with visualization/debug code; spatial RANSAC uses x/y coordinates.
    a = geometry[:, 2]
    b = geometry[:, 3]
    c = geometry[:, 4]
    det = np.maximum(a * c - b * b, 1e-12)
    keypoints[:, 2] = np.sqrt(1.0 / np.sqrt(det)).astype(np.float32)
    return keypoints


def crop_official_to_bbox(
    words: np.ndarray,
    geometry: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(words) == 0:
        return words, geometry
    x1, y1, x2, y2 = bbox
    xs = geometry[:, 0]
    ys = geometry[:, 1]
    mask = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    return words[mask], geometry[mask]
