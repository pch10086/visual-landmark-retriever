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
from src.paths import FEATURE_DIR, GT_DIR, IMAGE_DIR, METRICS_DIR, VIS_DIR, PROCESSED_DIR, ensure_dirs
from src.retrieve import query_words_from_features, rank_from_query_words
from src.visualize import save_retrieval_grid
from src.vocabulary import load_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BoVW retrieval on Oxford5k.")
    parser.add_argument("--feature", default="sift", help="feature directory name, e.g. sift")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--visualize", type=int, default=5, help="number of query grids to save")
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
    rankings = {}

    for idx, query in enumerate(tqdm(queries, desc="evaluating BoVW")):
        query_feature = load_features(feature_dir / f"{query.query_image}.npz")
        query_keypoints, query_descriptors, query_words = query_words_from_features(
            query_feature.keypoints,
            query_feature.descriptors,
            vocabulary,
            bbox=query.bbox,
        )
        query_region_feature = FeatureSet(
            image_id=query_feature.image_id,
            image_path=query_feature.image_path,
            keypoints=query_keypoints,
            descriptors=query_descriptors,
        )
        results = rank_from_query_words(
            query_words,
            index,
            exclude={query.query_image},
        )
        metrics.append(evaluate_query(query.query_id, results, gt))
        rankings[query.query_id] = [
            {"image_id": result.image_id, "score": result.score} for result in results[:1000]
        ]

        if idx < args.visualize:
            save_retrieval_grid(
                query,
                results,
                image_map,
                gt,
                VIS_DIR / f"bow_{query.query_id}.jpg",
                top_n=10,
            )

    payload = metrics_to_dict(metrics)
    payload["feature"] = args.feature
    payload["vocab_size"] = args.vocab_size
    payload["rankings"] = rankings
    out_path = METRICS_DIR / f"bow_{args.feature}_{args.vocab_size}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rankings"}, indent=2))
    print(f"saved metrics: {out_path}")


if __name__ == "__main__":
    main()
