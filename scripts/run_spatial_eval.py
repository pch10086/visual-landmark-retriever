#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from tqdm import tqdm

from src.evaluate import evaluate_query, metrics_to_dict
from src.features import FeatureSet, load_features
from src.index import load_index
from src.oxford_io import build_image_map, iter_available_queries, load_ground_truth
from src.paths import FEATURE_DIR, GT_DIR, IMAGE_DIR, METRICS_DIR, PROCESSED_DIR, VIS_DIR, ensure_dirs
from src.retrieve import query_words_from_features, rank_from_query_words
from src.spatial import spatial_rerank
from src.visualize import save_retrieval_grid
from src.vocabulary import load_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate spatial reranking on Oxford5k.")
    parser.add_argument("--feature", default="sift", help="feature directory name, e.g. sift")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--method", default="homography", choices=["homography", "affine"])
    parser.add_argument("--spatial-weight", type=float, default=0.02)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--visualize", type=int, default=5)
    args = parser.parse_args()

    ensure_dirs()
    gt = load_ground_truth(GT_DIR)
    image_map = build_image_map(IMAGE_DIR)
    queries = iter_available_queries(gt, image_map.keys())
    feature_dir = FEATURE_DIR / args.feature
    queries = [query for query in queries if (feature_dir / f"{query.query_image}.npz").exists()]
    if args.max_queries:
        queries = queries[: args.max_queries]
    if not queries:
        raise SystemExit(
            "No queries have matching extracted features. Run extract_features on query images."
        )

    vocabulary = load_vocabulary(PROCESSED_DIR / f"vocab_{args.feature}_{args.vocab_size}.joblib")
    index = load_index(PROCESSED_DIR / f"index_{args.feature}_{args.vocab_size}.joblib")

    metrics = []
    spatial_debug = {}

    for idx, query in enumerate(tqdm(queries, desc="evaluating spatial rerank")):
        query_feature_full = load_features(feature_dir / f"{query.query_image}.npz")
        query_keypoints, query_descriptors, query_words = query_words_from_features(
            query_feature_full.keypoints,
            query_feature_full.descriptors,
            vocabulary,
            bbox=query.bbox,
        )
        query_feature = FeatureSet(
            image_id=query_feature_full.image_id,
            image_path=query_feature_full.image_path,
            keypoints=query_keypoints,
            descriptors=query_descriptors,
        )
        bow_results = rank_from_query_words(
            query_words,
            index,
            exclude={query.query_image},
        )
        reranked, scores = spatial_rerank(
            query_feature=query_feature,
            query_words=query_words,
            results=bow_results,
            index=index,
            feature_dir=feature_dir,
            top_k=args.top_k,
            method=args.method,
            spatial_weight=args.spatial_weight,
        )
        metrics.append(evaluate_query(query.query_id, reranked, gt))
        spatial_debug[query.query_id] = [
            {
                "image_id": item.image_id,
                "original_score": item.original_score,
                "inliers": item.inliers,
                "matches": item.matches,
                "spatial_score": item.spatial_score,
            }
            for item in scores[: args.top_k]
        ]

        if idx < args.visualize:
            save_retrieval_grid(
                query,
                reranked,
                image_map,
                gt,
                VIS_DIR / f"spatial_{query.query_id}.jpg",
                top_n=10,
            )

    payload = metrics_to_dict(metrics)
    payload["feature"] = args.feature
    payload["vocab_size"] = args.vocab_size
    payload["top_k"] = args.top_k
    payload["method"] = args.method
    payload["spatial_weight"] = args.spatial_weight
    payload["spatial_debug"] = spatial_debug
    out_path = METRICS_DIR / f"spatial_{args.feature}_{args.vocab_size}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "spatial_debug"}, indent=2))
    print(f"saved metrics: {out_path}")


if __name__ == "__main__":
    main()
