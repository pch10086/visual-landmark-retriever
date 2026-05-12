#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from tqdm import tqdm

from src.evaluate import evaluate_query, metrics_to_dict
from src.official_index import load_official_index, load_official_query
from src.oxford_io import build_image_map, iter_available_queries, load_ground_truth
from src.paths import GT_DIR, IMAGE_DIR, METRICS_DIR, PROCESSED_DIR, VIS_DIR, ensure_dirs
from src.retrieve import rank_from_query_words
from src.visualize import save_retrieval_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate official Oxford5k hesaff_sift 1M word assignments."
    )
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

    metrics = []
    rankings = {}
    for idx, query in enumerate(tqdm(queries, desc="evaluating official BoVW")):
        geometry_path = Path(official.geometry_paths[query.query_image])
        query_words, _, _ = load_official_query(geometry_path, query.bbox)
        results = rank_from_query_words(
            query_words,
            image_index,
            exclude={query.query_image},
        )
        metrics.append(evaluate_query(query.query_id, results, gt))
        rankings[query.query_id] = [
            {"image_id": result.image_id, "score": result.score} for result in results[:1000]
        ]

        if idx < args.visualize and query.query_image in image_map:
            save_retrieval_grid(
                query,
                results,
                image_map,
                gt,
                VIS_DIR / f"official_bow_{query.query_id}.jpg",
                top_n=10,
            )

    payload = metrics_to_dict(metrics)
    payload["feature"] = "official_hesaff_sift"
    payload["vocab_size"] = 1_000_000
    payload["rankings"] = rankings
    out_path = METRICS_DIR / "official_bow_1m.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rankings"}, indent=2))
    print(f"saved metrics: {out_path}")


if __name__ == "__main__":
    main()

