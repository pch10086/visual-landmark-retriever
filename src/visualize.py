from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src.oxford_io import OxfordGroundTruth, OxfordQuery
from src.retrieve import RetrievalResult


def draw_query_box(image, bbox: tuple[float, float, float, float]):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    out = image.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 5)
    return out


def read_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def result_label(image_id: str, query_id: str, gt: OxfordGroundTruth) -> tuple[str, str]:
    if image_id in gt.good[query_id] or image_id in gt.ok[query_id]:
        return "positive", "green"
    if image_id in gt.junk[query_id]:
        return "junk", "orange"
    return "negative", "red"


def save_retrieval_grid(
    query: OxfordQuery,
    results: list[RetrievalResult],
    image_map: dict[str, Path],
    gt: OxfordGroundTruth,
    out_path: Path,
    top_n: int = 10,
) -> None:
    cols = min(top_n + 1, 6)
    rows = 1 + ((top_n + 1 - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax in axes_list:
        ax.axis("off")

    query_image = read_rgb(image_map[query.query_image])
    if query_image is not None:
        query_image = draw_query_box(query_image, query.bbox)
        axes_list[0].imshow(query_image)
    axes_list[0].set_title(f"query\n{query.query_id}", fontsize=9)

    for rank, result in enumerate(results[:top_n], start=1):
        ax = axes_list[rank]
        image_path = image_map.get(result.image_id)
        if image_path is None:
            continue
        image = read_rgb(image_path)
        if image is None:
            continue
        label, color = result_label(result.image_id, query.query_id, gt)
        ax.imshow(image)
        ax.set_title(
            f"#{rank} {label}\n{result.score:.3f}",
            fontsize=9,
            color=color,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

