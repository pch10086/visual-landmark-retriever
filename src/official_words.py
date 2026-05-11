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

    return OfficialWordGeometry(
        image_id=path.stem,
        words=np.asarray(words, dtype=np.int32),
        geometry=np.asarray(geometry, dtype=np.float32),
    )


def list_official_word_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix == ".txt")

