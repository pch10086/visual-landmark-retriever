#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from tqdm import tqdm

from src.official_index import build_official_index, save_official_index
from src.official_words import list_official_word_files
from src.paths import OFFICIAL_WORDS_DIR, PROCESSED_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TF-IDF index from official Oxford5k hesaff_sift 1M word files."
    )
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    word_files = list_official_word_files(OFFICIAL_WORDS_DIR)
    if args.max_files:
        word_files = word_files[: args.max_files]
    if not word_files:
        raise SystemExit(
            f"No official word files found in {OFFICIAL_WORDS_DIR}. "
            "Run scripts/download_oxford.py --official-words."
        )

    print(f"building official 1M index from {len(word_files)} files")
    index = build_official_index(list(tqdm(word_files, desc="official files")))
    out_path = PROCESSED_DIR / "official_index_1m.joblib"
    save_official_index(index, out_path)
    print(f"saved official index: {out_path}")


if __name__ == "__main__":
    main()

