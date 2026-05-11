#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import _bootstrap  # noqa: F401
import numpy as np
from tqdm import tqdm

from src.features import load_features
from src.paths import FEATURE_DIR, PROCESSED_DIR, ensure_dirs
from src.vocabulary import save_vocabulary, train_vocabulary


def collect_descriptors(
    feature_dir,
    max_descriptors: int,
    seed: int,
    max_files: int | None = None,
) -> np.ndarray:
    paths = sorted(feature_dir.glob("*.npz"))
    if max_files:
        paths = paths[:max_files]
    if not paths:
        raise SystemExit(f"No feature files found in {feature_dir}. Run extract_features first.")

    rng = random.Random(seed)
    rng.shuffle(paths)
    chunks = []
    total = 0
    for path in tqdm(paths, desc="sampling descriptors"):
        descriptors = load_features(path).descriptors
        if len(descriptors) == 0:
            continue
        remaining = max_descriptors - total
        if remaining <= 0:
            break
        if len(descriptors) > remaining:
            idx = np.random.default_rng(seed + total).choice(
                len(descriptors), size=remaining, replace=False
            )
            descriptors = descriptors[idx]
        chunks.append(descriptors.astype(np.float32, copy=False))
        total += len(descriptors)

    if not chunks:
        raise SystemExit("No descriptors sampled.")
    return np.vstack(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a visual vocabulary.")
    parser.add_argument("--feature", default="sift", help="feature directory name, e.g. sift")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-descriptors", type=int, default=500000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    feature_dir = FEATURE_DIR / args.feature
    descriptors = collect_descriptors(
        feature_dir,
        max_descriptors=args.max_descriptors,
        seed=args.seed,
        max_files=args.max_files,
    )
    print(f"training K={args.vocab_size} on {len(descriptors)} descriptors")
    vocabulary = train_vocabulary(
        descriptors,
        vocab_size=args.vocab_size,
        feature=args.feature,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    out_path = PROCESSED_DIR / f"vocab_{args.feature}_{args.vocab_size}.joblib"
    save_vocabulary(vocabulary, out_path)
    print(f"saved vocabulary: {out_path}")


if __name__ == "__main__":
    main()
