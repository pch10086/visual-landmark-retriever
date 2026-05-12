from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from src.index import ImageIndex, compute_tfidf, words_to_histogram
from src.official_words import (
    crop_official_to_bbox,
    official_geometry_to_keypoints,
    parse_word_geometry_file,
)


OFFICIAL_VOCAB_SIZE = 1_000_000


@dataclass
class OfficialIndex:
    image_index: ImageIndex
    geometry_paths: dict[str, str]

    @property
    def image_ids(self) -> list[str]:
        return self.image_index.image_ids


def load_official_words_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    item = parse_word_geometry_file(path)
    return item.words, item.geometry


def load_official_query(
    geometry_path: Path,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    words, geometry = load_official_words_geometry(geometry_path)
    query_words, query_geometry = crop_official_to_bbox(words, geometry, bbox)
    query_keypoints = official_geometry_to_keypoints(query_geometry)
    return query_words, query_geometry, query_keypoints


def build_official_index(word_files: list[Path]) -> OfficialIndex:
    image_ids: list[str] = []
    histograms = []
    word_assignments: dict[str, np.ndarray] = {}
    geometry_paths: dict[str, str] = {}

    for path in word_files:
        item = parse_word_geometry_file(path)
        image_ids.append(item.image_id)
        word_assignments[item.image_id] = item.words
        geometry_paths[item.image_id] = str(path)
        histograms.append(words_to_histogram(item.words, OFFICIAL_VOCAB_SIZE))

    hist = sparse.vstack(histograms).tocsr()
    matrix, idf = compute_tfidf(hist)
    image_index = ImageIndex(
        image_ids=image_ids,
        vocab_size=OFFICIAL_VOCAB_SIZE,
        idf=idf,
        matrix=matrix,
        word_assignments=word_assignments,
        feature_paths=geometry_paths,
    )
    return OfficialIndex(image_index=image_index, geometry_paths=geometry_paths)


def save_official_index(index: OfficialIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)


def load_official_index(path: Path) -> OfficialIndex:
    return joblib.load(path)

