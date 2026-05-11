#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from tqdm import tqdm

from src.features import extract_image_features, feature_path, save_features
from src.oxford_io import build_image_map
from src.paths import FEATURE_DIR, IMAGE_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract local features for Oxford5k images.")
    parser.add_argument("--feature", default="sift", choices=["sift", "orb"])
    parser.add_argument("--image-dir", default=str(IMAGE_DIR))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    image_dir = Path(args.image_dir)
    image_map = build_image_map(image_dir)
    items = sorted(image_map.items())
    if args.max_images:
        items = items[: args.max_images]
    if not items:
        raise SystemExit(f"No images found in {image_dir}. Run scripts/download_oxford.py --images.")

    out_dir = Path(args.out_dir) if args.out_dir else FEATURE_DIR / args.feature
    for image_id, image_path in tqdm(items, desc="extracting features"):
        out_path = feature_path(out_dir, image_id)
        if out_path.exists() and not args.force:
            continue
        feature_set = extract_image_features(
            image_path=image_path,
            image_id=image_id,
            feature=args.feature,
            max_features=args.max_features,
        )
        save_features(feature_set, out_path)

    print(f"features saved in {out_dir}")


if __name__ == "__main__":
    main()
