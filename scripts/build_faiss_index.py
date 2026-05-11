#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from scipy import sparse
from tqdm import tqdm

from src.features import load_features
from src.index import ImageIndex, compute_tfidf, save_index, words_to_histogram
from src.paths import FEATURE_DIR, PROCESSED_DIR, ensure_dirs
from src.vocabulary import build_faiss_quantizer, faiss_search, load_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BoVW TF-IDF index with FAISS quantization. GPU is default."
    )
    parser.add_argument("--feature", default="sift", help="feature directory name, e.g. sift")
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--no-gpu", action="store_true", help="force FAISS CPU quantization")
    args = parser.parse_args()

    ensure_dirs()
    feature_dir = FEATURE_DIR / args.feature
    vocab_path = PROCESSED_DIR / f"vocab_{args.feature}_{args.vocab_size}.joblib"
    vocabulary = load_vocabulary(vocab_path)
    if vocabulary.backend != "faiss":
        raise SystemExit(
            f"{vocab_path} is a {vocabulary.backend} vocabulary. "
            "Use scripts/build_index.py for sklearn vocabularies."
        )

    feature_paths = sorted(feature_dir.glob("*.npz"))
    if args.max_files:
        feature_paths = feature_paths[: args.max_files]
    if not feature_paths:
        raise SystemExit(f"No feature files found in {feature_dir}.")

    image_ids: list[str] = []
    histograms = []
    word_assignments = {}
    feature_path_map = {}
    use_gpu = not args.no_gpu
    quantizer = build_faiss_quantizer(vocabulary.centroids, use_gpu=use_gpu)

    for path in tqdm(feature_paths, desc="FAISS quantizing images"):
        feature_set = load_features(path)
        words = faiss_search(quantizer, feature_set.descriptors, batch_size=args.batch_size)
        image_ids.append(feature_set.image_id)
        word_assignments[feature_set.image_id] = words
        feature_path_map[feature_set.image_id] = str(path)
        histograms.append(words_to_histogram(words, vocabulary.size))

    hist = sparse.vstack(histograms).tocsr()
    matrix, idf = compute_tfidf(hist)
    index = ImageIndex(
        image_ids=image_ids,
        vocab_size=vocabulary.size,
        idf=idf,
        matrix=matrix,
        word_assignments=word_assignments,
        feature_paths=feature_path_map,
    )
    out_path = PROCESSED_DIR / f"index_{args.feature}_{args.vocab_size}.joblib"
    save_index(index, out_path)
    print(f"saved FAISS-built index: {out_path}")


if __name__ == "__main__":
    main()
