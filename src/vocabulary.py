from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class Vocabulary:
    model: MiniBatchKMeans
    feature: str

    @property
    def size(self) -> int:
        return int(self.model.n_clusters)

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors.size == 0:
            return np.empty((0,), dtype=np.int32)
        words = self.model.predict(descriptors.astype(np.float32, copy=False))
        return words.astype(np.int32, copy=False)


def train_vocabulary(
    descriptors: np.ndarray,
    vocab_size: int,
    feature: str,
    batch_size: int = 8192,
    random_state: int = 0,
    max_iter: int = 100,
) -> Vocabulary:
    if len(descriptors) < vocab_size:
        raise ValueError(
            f"Need at least vocab_size descriptors: got {len(descriptors)}, K={vocab_size}"
        )

    model = MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=batch_size,
        random_state=random_state,
        max_iter=max_iter,
        n_init=3,
        reassignment_ratio=0.01,
        verbose=0,
    )
    model.fit(descriptors.astype(np.float32, copy=False))
    return Vocabulary(model=model, feature=feature)


def save_vocabulary(vocabulary: Vocabulary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocabulary, path)


def load_vocabulary(path: Path) -> Vocabulary:
    return joblib.load(path)
