from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse


@dataclass
class ImageIndex:
    image_ids: list[str]
    vocab_size: int
    idf: np.ndarray
    matrix: sparse.csr_matrix
    word_assignments: dict[str, np.ndarray]
    feature_paths: dict[str, str]

    @property
    def image_to_row(self) -> dict[str, int]:
        return {image_id: idx for idx, image_id in enumerate(self.image_ids)}


def words_to_histogram(words: np.ndarray, vocab_size: int) -> sparse.csr_matrix:
    if words.size == 0:
        return sparse.csr_matrix((1, vocab_size), dtype=np.float32)
    counts = np.bincount(words.astype(np.int32), minlength=vocab_size).astype(np.float32)
    cols = np.flatnonzero(counts)
    data = counts[cols]
    rows = np.zeros_like(cols)
    return sparse.csr_matrix((data, (rows, cols)), shape=(1, vocab_size), dtype=np.float32)


def l2_normalize_csr(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    nonzero = norms > 0
    inv = np.zeros_like(norms, dtype=np.float32)
    inv[nonzero] = 1.0 / norms[nonzero]
    return sparse.diags(inv).dot(matrix).tocsr()


def compute_tfidf(hist: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    hist = hist.tocsr().astype(np.float32)
    n_docs = hist.shape[0]
    df = np.diff(hist.tocsc().indptr).astype(np.float32)
    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    tfidf = hist.multiply(idf).tocsr()
    tfidf = l2_normalize_csr(tfidf)
    return tfidf, idf.astype(np.float32)


def make_query_vector(words: np.ndarray, vocab_size: int, idf: np.ndarray) -> sparse.csr_matrix:
    hist = words_to_histogram(words, vocab_size)
    vector = hist.multiply(idf).tocsr()
    return l2_normalize_csr(vector)


def save_index(index: ImageIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)


def load_index(path: Path) -> ImageIndex:
    return joblib.load(path)

