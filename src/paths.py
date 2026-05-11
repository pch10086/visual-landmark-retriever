from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OXFORD_DIR = DATA_DIR / "oxford5k"
IMAGE_DIR = OXFORD_DIR / "images"
GT_DIR = OXFORD_DIR / "gt"
OFFICIAL_WORDS_DIR = OXFORD_DIR / "official_words"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = PROCESSED_DIR / "features"
OUTPUT_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
TABLES_DIR = OUTPUT_DIR / "tables"
VIS_DIR = OUTPUT_DIR / "visualizations"


def ensure_dirs() -> None:
    for path in [
        RAW_DIR,
        IMAGE_DIR,
        GT_DIR,
        OFFICIAL_WORDS_DIR,
        PROCESSED_DIR,
        FEATURE_DIR,
        METRICS_DIR,
        TABLES_DIR,
        VIS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

