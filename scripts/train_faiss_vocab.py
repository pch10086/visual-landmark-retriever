#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from scripts.train_vocab import collect_descriptors
from src.paths import FEATURE_DIR, PROCESSED_DIR, ensure_dirs
from src.vocabulary import make_faiss_vocabulary, save_vocabulary


def train_faiss_kmeans(
    descriptors: np.ndarray,
    vocab_size: int,
    use_gpu: bool,
    seed: int,
    iterations: int,
    nredo: int,
    verbose: bool,
):
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "FAISS is not installed. Install faiss-gpu on the server, or faiss-cpu for CPU fallback."
        ) from exc

    descriptors = np.ascontiguousarray(descriptors.astype(np.float32, copy=False))
    if len(descriptors) < vocab_size:
        raise SystemExit(
            f"Need at least vocab_size descriptors: got {len(descriptors)}, K={vocab_size}"
        )
    if use_gpu:
        if not hasattr(faiss, "get_num_gpus"):
            raise SystemExit("This FAISS build does not expose GPU support. Install faiss-gpu.")
        if faiss.get_num_gpus() <= 0:
            raise SystemExit("FAISS GPU was requested, but no GPU is visible to FAISS.")

    d = descriptors.shape[1]
    kmeans = faiss.Kmeans(
        d=d,
        k=vocab_size,
        niter=iterations,
        nredo=nredo,
        verbose=verbose,
        gpu=use_gpu,
        seed=seed,
        spherical=False,
    )
    kmeans.train(descriptors)
    return np.ascontiguousarray(kmeans.centroids.astype(np.float32, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a FAISS visual vocabulary. GPU is used by default."
    )
    parser.add_argument("--feature", default="sift", help="feature directory name, e.g. sift")
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--max-descriptors", type=int, default=2000000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument("--no-gpu", action="store_true", help="force FAISS CPU training")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    feature_dir = FEATURE_DIR / args.feature
    descriptors = collect_descriptors(
        feature_dir,
        max_descriptors=args.max_descriptors,
        seed=args.seed,
        max_files=args.max_files,
    )
    use_gpu = not args.no_gpu
    device = "GPU" if use_gpu else "CPU"
    print(
        f"training FAISS {device} K={args.vocab_size} on "
        f"{len(descriptors)} descriptors x {descriptors.shape[1]} dims"
    )
    centroids = train_faiss_kmeans(
        descriptors=descriptors,
        vocab_size=args.vocab_size,
        use_gpu=use_gpu,
        seed=args.seed,
        iterations=args.iterations,
        nredo=args.nredo,
        verbose=not args.quiet,
    )
    vocabulary = make_faiss_vocabulary(centroids, feature=args.feature)
    out_path = PROCESSED_DIR / f"vocab_{args.feature}_{args.vocab_size}.joblib"
    save_vocabulary(vocabulary, out_path)
    print(f"saved FAISS vocabulary: {out_path}")


if __name__ == "__main__":
    main()
