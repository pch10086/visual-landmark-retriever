#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.paths import METRICS_DIR, TABLES_DIR, ensure_dirs


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(name: str, payload: dict | None) -> str:
    if payload is None:
        return f"| {name} | - | - | - | not run |\n"
    return (
        f"| {name} | {payload.get('feature', '-')} | "
        f"{payload.get('vocab_size', '-')} | {payload.get('mAP', 0.0):.4f} | "
        f"{len(payload.get('queries', []))} queries |\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready markdown tables.")
    parser.add_argument("--feature", default="sift", help="feature directory name, e.g. sift")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--official", action="store_true", help="summarize official hesaff_sift results")
    args = parser.parse_args()

    ensure_dirs()
    if args.official:
        bow = load_json(METRICS_DIR / "official_bow_1m.json")
        spatial = load_json(METRICS_DIR / "official_spatial_1m.json")
        out_name = "results_official_1m.md"
    else:
        bow = load_json(METRICS_DIR / f"bow_{args.feature}_{args.vocab_size}.json")
        spatial = load_json(METRICS_DIR / f"spatial_{args.feature}_{args.vocab_size}.json")
        out_name = f"results_{args.feature}_{args.vocab_size}.md"

    text = "# Reproduction Results\n\n"
    text += "## Main Metrics\n\n"
    text += "| Method | Feature | Vocabulary size | mAP | Notes |\n"
    text += "|---|---:|---:|---:|---|\n"
    text += metric_summary("BoVW + TF-IDF", bow)
    text += metric_summary("BoVW + TF-IDF + spatial reranking", spatial)

    if bow and spatial:
        delta = float(spatial.get("mAP", 0.0)) - float(bow.get("mAP", 0.0))
        text += f"\nSpatial reranking delta: `{delta:+.4f}` mAP.\n"

    text += "\n## Notes For Report\n\n"
    text += "- Positive examples are Oxford `good` and `ok` labels.\n"
    text += "- `junk` images are ignored during AP computation.\n"
    text += "- This reproduction uses OpenCV local features and MiniBatchKMeans.\n"
    text += "- Vocabulary size is smaller than the paper's 1M-word setting by default.\n"

    out_path = TABLES_DIR / out_name
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"saved table: {out_path}")


if __name__ == "__main__":
    main()
