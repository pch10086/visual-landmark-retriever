#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

from src.oxford_io import list_images
from src.paths import GT_DIR, IMAGE_DIR, RAW_DIR, ensure_dirs


BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings"
GT_URL = f"{BASE_URL}/gt_files_170407.tgz"
OFFICIAL_WORDS_URL = f"{BASE_URL}/word_oxc1_hesaff_sift_16M_1M.tgz"
IMAGE_URLS = [
    f"{BASE_URL}/oxbuild_images.tgz",
    "https://thor.robots.ox.ac.uk/oxbuildings/oxbuild_images.tgz",
]
EXPECTED_SIZES = {
    "gt_files_170407.tgz": 8174,
    "oxbuild_images.tgz": 1980280437,
    "word_oxc1_hesaff_sift_16M_1M.tgz": 304646902,
}


def expected_size(dest: Path) -> int | None:
    return EXPECTED_SIZES.get(dest.name)


def complete_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    expected = expected_size(path)
    return expected is None or path.stat().st_size == expected


def download(url: str, dest: Path, force: bool = False) -> None:
    if complete_file(dest) and not force:
        print(f"exists: {dest}")
        return

    if dest.exists() and not complete_file(dest):
        print(f"incomplete archive found, resuming if server supports it: {dest}")
    elif force and dest.exists():
        dest.unlink()

    print(f"downloading: {url}")
    print(f"        to: {dest}")
    subprocess.run(
        ["curl", "-L", "-C", "-", "--fail", "--retry", "3", "-o", str(dest), url],
        check=True,
    )
    if not complete_file(dest):
        expected = expected_size(dest)
        actual = dest.stat().st_size if dest.exists() else 0
        raise RuntimeError(f"Incomplete download: {dest} has {actual} bytes, expected {expected}")


def extract_tgz(archive: Path, dest: Path, strip_single_dir: bool = False) -> None:
    print(f"extracting: {archive} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        tar.extractall(dest)

    if strip_single_dir:
        subdirs = [path for path in dest.iterdir() if path.is_dir()]
        files = [path for path in dest.iterdir() if path.is_file()]
        if len(subdirs) == 1 and not files:
            inner = subdirs[0]
            for path in inner.iterdir():
                target = dest / path.name
                if target.exists():
                    continue
                path.replace(target)
            inner.rmdir()


def try_download_images(force: bool = False) -> bool:
    archive = RAW_DIR / "oxbuild_images.tgz"
    for url in IMAGE_URLS:
        try:
            download(url, archive, force=force)
            extract_tgz(archive, IMAGE_DIR, strip_single_dir=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"image download failed from {url}: {exc}")
    return False


def print_status() -> None:
    images = list_images(IMAGE_DIR)
    gt_files = sorted(GT_DIR.glob("*.txt"))
    print()
    print("Oxford5k data status")
    print(f"  ground truth txt files: {len(gt_files)} in {GT_DIR}")
    print(f"  images: {len(images)} in {IMAGE_DIR}")
    if len(images) < 5000:
        print()
        print("Image data is incomplete.")
        print("Run with --images to download the historical Oxford5k image archive,")
        print("or manually place jpg files under:")
        print(f"  {IMAGE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Oxford5k data.")
    parser.add_argument(
        "--images",
        action="store_true",
        help="also download the Oxford5k image archive (~1.98GB)",
    )
    parser.add_argument(
        "--official-words",
        action="store_true",
        help="download official 1M visual word ids and geometry (~305MB)",
    )
    parser.add_argument("--force", action="store_true", help="redownload archives")
    args = parser.parse_args()

    ensure_dirs()

    gt_archive = RAW_DIR / "gt_files_170407.tgz"
    download(GT_URL, gt_archive, force=args.force)
    extract_tgz(gt_archive, GT_DIR, strip_single_dir=True)

    if args.images:
        ok = try_download_images(force=args.force)
        if not ok:
            raise SystemExit("Could not download Oxford5k images from known URLs.")

    if args.official_words:
        from src.paths import OFFICIAL_WORDS_DIR

        words_archive = RAW_DIR / "word_oxc1_hesaff_sift_16M_1M.tgz"
        download(OFFICIAL_WORDS_URL, words_archive, force=args.force)
        extract_tgz(words_archive, OFFICIAL_WORDS_DIR, strip_single_dir=True)

    print_status()


if __name__ == "__main__":
    main()
