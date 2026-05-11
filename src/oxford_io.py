from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class OxfordQuery:
    query_id: str
    landmark: str
    query_image: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class OxfordGroundTruth:
    queries: list[OxfordQuery]
    good: dict[str, set[str]]
    ok: dict[str, set[str]]
    junk: dict[str, set[str]]

    @property
    def positives(self) -> dict[str, set[str]]:
        return {qid: self.good[qid] | self.ok[qid] for qid in self.good}


def normalize_image_id(name: str) -> str:
    path = Path(name.strip())
    stem = path.stem
    if stem.startswith("oxc1_"):
        return stem
    return f"oxc1_{stem}"


def image_id_from_path(path: Path) -> str:
    return normalize_image_id(path.stem)


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_image_map(image_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in list_images(image_dir):
        mapping[image_id_from_path(path)] = path
    return mapping


def read_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    items: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.add(normalize_image_id(line))
    return items


def parse_query_file(path: Path) -> OxfordQuery:
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid query file: {path}")

    raw_image = parts[0]
    if raw_image.startswith("oxc1_"):
        query_image = raw_image
    else:
        query_image = f"oxc1_{raw_image}"
    query_image = Path(query_image).stem

    bbox = tuple(float(value) for value in parts[1:5])
    query_id = path.name.removesuffix("_query.txt")
    landmark = query_id.rsplit("_", 1)[0]
    return OxfordQuery(
        query_id=query_id,
        landmark=landmark,
        query_image=query_image,
        bbox=bbox,  # type: ignore[arg-type]
    )


def load_ground_truth(gt_dir: Path) -> OxfordGroundTruth:
    query_files = sorted(gt_dir.glob("*_query.txt"))
    if not query_files:
        raise FileNotFoundError(
            f"No Oxford query files found in {gt_dir}. Run scripts/download_oxford.py."
        )

    queries = [parse_query_file(path) for path in query_files]
    good: dict[str, set[str]] = {}
    ok: dict[str, set[str]] = {}
    junk: dict[str, set[str]] = {}

    for query in queries:
        base = gt_dir / query.query_id
        good[query.query_id] = read_list(base.with_name(f"{query.query_id}_good.txt"))
        ok[query.query_id] = read_list(base.with_name(f"{query.query_id}_ok.txt"))
        junk[query.query_id] = read_list(base.with_name(f"{query.query_id}_junk.txt"))

    return OxfordGroundTruth(queries=queries, good=good, ok=ok, junk=junk)


def iter_available_queries(
    gt: OxfordGroundTruth, image_ids: Iterable[str]
) -> list[OxfordQuery]:
    available = set(image_ids)
    return [query for query in gt.queries if query.query_image in available]

