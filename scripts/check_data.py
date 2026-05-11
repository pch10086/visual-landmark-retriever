#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.oxford_io import build_image_map, iter_available_queries, load_ground_truth
from src.paths import GT_DIR, IMAGE_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Oxford5k data layout.")
    parser.add_argument("--show-missing", type=int, default=20)
    args = parser.parse_args()

    ensure_dirs()
    gt = load_ground_truth(GT_DIR)
    image_map = build_image_map(IMAGE_DIR)
    available_queries = iter_available_queries(gt, image_map.keys())
    missing = [query.query_image for query in gt.queries if query.query_image not in image_map]

    print("Oxford5k data check")
    print(f"  ground truth queries: {len(gt.queries)}")
    print(f"  images found: {len(image_map)}")
    print(f"  query images available: {len(available_queries)}/{len(gt.queries)}")
    if missing:
        print("  missing query images:")
        for image_id in missing[: args.show_missing]:
            print(f"    {image_id}")
        if len(missing) > args.show_missing:
            print(f"    ... {len(missing) - args.show_missing} more")


if __name__ == "__main__":
    main()

