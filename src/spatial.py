from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.features import FeatureSet, load_features
from src.index import ImageIndex
from src.retrieve import RetrievalResult


@dataclass(frozen=True)
class SpatialScore:
    image_id: str
    original_score: float
    inliers: int
    matches: int
    spatial_score: float


def group_word_indices(words: np.ndarray) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, word in enumerate(words.tolist()):
        groups[int(word)].append(idx)
    return groups


def make_word_matches(
    query_words: np.ndarray,
    candidate_words: np.ndarray,
    max_pairs_per_word: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    query_groups = group_word_indices(query_words)
    candidate_groups = group_word_indices(candidate_words)

    query_indices: list[int] = []
    candidate_indices: list[int] = []
    for word, q_indices in query_groups.items():
        c_indices = candidate_groups.get(word)
        if not c_indices:
            continue
        for q_idx in q_indices[:max_pairs_per_word]:
            for c_idx in c_indices[:max_pairs_per_word]:
                query_indices.append(q_idx)
                candidate_indices.append(c_idx)

    if not query_indices:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return np.asarray(query_indices, dtype=np.int32), np.asarray(candidate_indices, dtype=np.int32)


def geometric_inliers(
    query_points: np.ndarray,
    candidate_points: np.ndarray,
    method: str = "homography",
    reproj_threshold: float = 8.0,
) -> int:
    if len(query_points) < 4 or len(candidate_points) < 4:
        return 0

    src = query_points[:, :2].astype(np.float32)
    dst = candidate_points[:, :2].astype(np.float32)

    if method == "affine":
        _, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            maxIters=2000,
            confidence=0.995,
        )
    else:
        _, mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            maxIters=2000,
            confidence=0.995,
        )

    if mask is None:
        return 0
    return int(mask.ravel().sum())


def spatial_rerank(
    query_feature: FeatureSet,
    query_words: np.ndarray,
    results: list[RetrievalResult],
    index: ImageIndex,
    feature_dir: Path,
    top_k: int = 100,
    method: str = "homography",
    spatial_weight: float = 0.02,
    max_pairs_per_word: int = 8,
) -> tuple[list[RetrievalResult], list[SpatialScore]]:
    top = results[:top_k]
    rest = results[top_k:]
    spatial_scores: list[SpatialScore] = []

    for result in top:
        feature_path = Path(index.feature_paths[result.image_id])
        candidate = load_features(feature_path)
        candidate_words = index.word_assignments.get(result.image_id)
        if candidate_words is None:
            candidate_words = np.empty((0,), dtype=np.int32)

        q_idx, c_idx = make_word_matches(
            query_words,
            candidate_words,
            max_pairs_per_word=max_pairs_per_word,
        )
        if len(q_idx) == 0:
            inliers = 0
            matches = 0
        else:
            inliers = geometric_inliers(
                query_feature.keypoints[q_idx],
                candidate.keypoints[c_idx],
                method=method,
            )
            matches = int(len(q_idx))

        score = float(result.score + spatial_weight * inliers)
        spatial_scores.append(
            SpatialScore(
                image_id=result.image_id,
                original_score=result.score,
                inliers=inliers,
                matches=matches,
                spatial_score=score,
            )
        )

    reranked_top = [
        RetrievalResult(image_id=item.image_id, score=item.spatial_score)
        for item in sorted(spatial_scores, key=lambda item: (-item.spatial_score, -item.inliers))
    ]
    return reranked_top + rest, spatial_scores

