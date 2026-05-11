from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureSet:
    image_id: str
    image_path: str
    keypoints: np.ndarray
    descriptors: np.ndarray


def create_detector(feature: str = "sift", max_features: int = 4000):
    feature = feature.lower()
    if feature == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "OpenCV SIFT is unavailable. Install opencv-contrib-python or use --feature orb."
            )
        return cv2.SIFT_create(nfeatures=max_features)
    if feature == "orb":
        return cv2.ORB_create(nfeatures=max_features)
    raise ValueError(f"Unsupported feature type: {feature}")


def keypoints_to_array(keypoints: tuple[cv2.KeyPoint, ...] | list[cv2.KeyPoint]) -> np.ndarray:
    rows = []
    for kp in keypoints:
        rows.append(
            [
                kp.pt[0],
                kp.pt[1],
                kp.size,
                kp.angle,
                kp.response,
                float(kp.octave),
                float(kp.class_id),
            ]
        )
    if not rows:
        return np.empty((0, 7), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def extract_image_features(
    image_path: Path,
    image_id: str,
    feature: str = "sift",
    max_features: int = 4000,
) -> FeatureSet:
    detector = create_detector(feature=feature, max_features=max_features)
    gray = load_gray(image_path)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    keypoint_array = keypoints_to_array(keypoints or [])

    if descriptors is None:
        descriptor_dim = 128 if feature.lower() == "sift" else 32
        descriptors = np.empty((0, descriptor_dim), dtype=np.float32)
    elif feature.lower() == "sift":
        descriptors = descriptors.astype(np.float32, copy=False)
    else:
        descriptors = descriptors.astype(np.float32)

    return FeatureSet(
        image_id=image_id,
        image_path=str(image_path),
        keypoints=keypoint_array,
        descriptors=descriptors,
    )


def crop_features_to_bbox(
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(keypoints) == 0:
        return keypoints, descriptors
    x1, y1, x2, y2 = bbox
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
    mask = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    return keypoints[mask], descriptors[mask]


def save_features(feature_set: FeatureSet, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        image_id=feature_set.image_id,
        image_path=feature_set.image_path,
        keypoints=feature_set.keypoints,
        descriptors=feature_set.descriptors,
    )


def load_features(path: Path) -> FeatureSet:
    data = np.load(path, allow_pickle=False)
    return FeatureSet(
        image_id=str(data["image_id"]),
        image_path=str(data["image_path"]),
        keypoints=data["keypoints"].astype(np.float32, copy=False),
        descriptors=data["descriptors"].astype(np.float32, copy=False),
    )


def feature_path(feature_dir: Path, image_id: str) -> Path:
    return feature_dir / f"{image_id}.npz"

