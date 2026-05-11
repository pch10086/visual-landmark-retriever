from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features import crop_features_to_bbox
from src.index import ImageIndex, make_query_vector
from src.vocabulary import Vocabulary


@dataclass(frozen=True)
class RetrievalResult:
    image_id: str
    score: float


def rank_from_query_words(
    query_words: np.ndarray,
    index: ImageIndex,
    exclude: set[str] | None = None,
) -> list[RetrievalResult]:
    query_vec = make_query_vector(query_words, index.vocab_size, index.idf)
    scores = index.matrix.dot(query_vec.T).toarray().ravel()
    order = np.argsort(-scores)
    excluded = exclude or set()

    results: list[RetrievalResult] = []
    for row in order:
        score = float(scores[row])
        if score <= 0:
            break
        image_id = index.image_ids[int(row)]
        if image_id in excluded:
            continue
        results.append(RetrievalResult(image_id=image_id, score=score))
    return results


def query_words_from_features(
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    vocabulary: Vocabulary,
    bbox: tuple[float, float, float, float] | None = None,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bbox is not None:
        keypoints, descriptors = crop_features_to_bbox(keypoints, descriptors, bbox)
    if vocabulary.backend == "faiss":
        from src.vocabulary import faiss_predict

        words = faiss_predict(vocabulary.centroids, descriptors, use_gpu=use_gpu)
    else:
        words = vocabulary.predict(descriptors)
    return keypoints, descriptors, words
