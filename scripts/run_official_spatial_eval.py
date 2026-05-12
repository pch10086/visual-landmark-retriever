#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import _bootstrap  # noqa: F401
from tqdm import tqdm

from src.evaluate import evaluate_query, metrics_to_dict
from src.official_index import load_official_index, load_official_query, load_official_words_geometry
from src.official_words import official_geometry_to_keypoints
from src.oxford_io import build_image_map, iter_available_queries, load_ground_truth
from src.paths import GT_DIR, IMAGE_DIR, METRICS_DIR, PROCESSED_DIR, VIS_DIR, ensure_dirs
from src.retrieve import rank_from_query_words
from src.spatial import spatial_rerank_arrays
from src.visualize import save_retrieval_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial reranking with official Oxford5k hesaff_sift 1M words/geometry."
    )
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--method", default="homography", choices=["homography", "affine"])
    parser.add_argument("--spatial-weight", type=float, default=0.02)
    parser.add_argument("--max-pairs-per-word", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--visualize", type=int, default=10)
    args = parser.parse_args()

    ensure_dirs()
    gt = load_ground_truth(GT_DIR)
    image_map = build_image_map(IMAGE_DIR)
    official = load_official_index(PROCESSED_DIR / "official_index_1m.joblib")
    image_index = official.image_index

    queries = iter_available_queries(gt, image_index.image_ids)
    if args.max_queries:
        queries = queries[: args.max_queries]
    if not queries:
        raise SystemExit("No official query images found in the official index.")

    @lru_cache(maxsize=512)
    def candidate_loader(image_id: str):
        words, geometry = load_official_words_geometry(Path(official.geometry_paths[image_id]))
        return official_geometry_to_keypoints(geometry), words

    metrics = []
    spatial_debug = {}
    for idx, query in enumerate(tqdm(queries, desc="evaluating official spatial")):
        query_words, _, query_keypoints = load_official_query(
            Path(official.geometry_paths[query.query_image]),
            query.bbox,
        )
        bow_results = rank_from_query_words(
            query_words,
            image_index,
            exclude={query.query_image},
        )
        reranked, scores = spatial_rerank_arrays(
            query_keypoints=query_keypoints,
            query_words=query_words,
            results=bow_results,
            candidate_loader=candidate_loader,
            top_k=args.top_k,
            method=args.method,
            spatial_weight=args.spatial_weight,
            max_pairs_per_word=args.max_pairs_per_word,
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

        if idx < args.visualize and query.query_image in image_map:
            save_retrieval_grid(
                query,
                reranked,
                image_map,
                gt,
                VIS_DIR / f"official_spatial_{query.query_id}.jpg",
                top_n=10,
            )

    payload = metrics_to_dict(metrics)
    payload["feature"] = "official_hesaff_sift"
    payload["vocab_size"] = 1_000_000
    payload["top_k"] = args.top_k
    payload["method"] = args.method
    payload["spatial_weight"] = args.spatial_weight
    payload["max_pairs_per_word"] = args.max_pairs_per_word
    payload["spatial_debug"] = spatial_debug
    out_path = METRICS_DIR / "official_spatial_1m.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "spatial_debug"}, indent=2))
    print(f"saved metrics: {out_path}")


if __name__ == "__main__":
    main()

