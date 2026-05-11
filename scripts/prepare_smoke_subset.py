#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from src.oxford_io import build_image_map, load_ground_truth
from src.paths import GT_DIR, IMAGE_DIR, OXFORD_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small image subset for smoke tests.")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--out", default=str(OXFORD_DIR / "smoke_images"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path(args.out)
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_map = build_image_map(IMAGE_DIR)
    gt = load_ground_truth(GT_DIR)
    selected_ids = []

    for query in gt.queries:
        if query.query_image in image_map and query.query_image not in selected_ids:
            selected_ids.append(query.query_image)

    for image_id in sorted(image_map):
        if len(selected_ids) >= args.count:
            break
        if image_id not in selected_ids:
            selected_ids.append(image_id)

    for image_id in selected_ids:
        src = image_map[image_id]
        dst = out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"created smoke subset: {out_dir}")
    print(f"images: {len(selected_ids)}")


if __name__ == "__main__":
    main()

