from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class Vocabulary:
    model: Any
    feature: str
    backend: str = "sklearn"
    centroids: np.ndarray | None = None

    @property
    def size(self) -> int:
        if self.centroids is not None:
            return int(self.centroids.shape[0])
        return int(self.model.n_clusters)

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors.size == 0:
            return np.empty((0,), dtype=np.int32)
        if self.backend == "faiss":
            return faiss_predict(self.centroids, descriptors)
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
    return Vocabulary(model=model, feature=feature, backend="sklearn")


def faiss_predict(
    centroids: np.ndarray | None,
    descriptors: np.ndarray,
    use_gpu: bool = False,
    batch_size: int = 200000,
) -> np.ndarray:
    if centroids is None:
        raise ValueError("FAISS vocabulary is missing centroids.")
    if descriptors.size == 0:
        return np.empty((0,), dtype=np.int32)

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required for this vocabulary. Install faiss-gpu or faiss-cpu."
        ) from exc

    centroids = np.ascontiguousarray(centroids.astype(np.float32, copy=False))
    descriptors = np.ascontiguousarray(descriptors.astype(np.float32, copy=False))
    index = build_faiss_quantizer(centroids, use_gpu=use_gpu)
    return faiss_search(index, descriptors, batch_size=batch_size)


def build_faiss_quantizer(centroids: np.ndarray, use_gpu: bool = False):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required for this vocabulary. Install faiss-gpu or faiss-cpu."
        ) from exc

    centroids = np.ascontiguousarray(centroids.astype(np.float32, copy=False))
    index = faiss.IndexFlatL2(centroids.shape[1])
    if use_gpu:
        if not hasattr(faiss, "get_num_gpus"):
            raise RuntimeError("This FAISS build does not expose GPU support. Install faiss-gpu.")
        if faiss.get_num_gpus() <= 0:
            raise RuntimeError("FAISS GPU was requested, but no GPU is visible to FAISS.")
        try:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not move FAISS quantizer to GPU: {exc}") from exc
    index.add(centroids)
    return index


def faiss_search(index, descriptors: np.ndarray, batch_size: int = 200000) -> np.ndarray:
    if descriptors.size == 0:
        return np.empty((0,), dtype=np.int32)
    descriptors = np.ascontiguousarray(descriptors.astype(np.float32, copy=False))

    chunks = []
    for start in range(0, len(descriptors), batch_size):
        batch = descriptors[start : start + batch_size]
        _, ids = index.search(batch, 1)
        chunks.append(ids.ravel().astype(np.int32, copy=False))
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int32)


def make_faiss_vocabulary(centroids: np.ndarray, feature: str) -> Vocabulary:
    centroids = np.ascontiguousarray(centroids.astype(np.float32, copy=False))
    return Vocabulary(model=None, feature=feature, backend="faiss", centroids=centroids)


def save_vocabulary(vocabulary: Vocabulary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocabulary, path)


def load_vocabulary(path: Path) -> Vocabulary:
    return joblib.load(path)
