from __future__ import annotations

from dataclasses import dataclass

from src.oxford_io import OxfordGroundTruth
from src.retrieve import RetrievalResult


@dataclass(frozen=True)
class QueryMetrics:
    query_id: str
    ap: float
    positives: int
    ranked: int


def average_precision(
    ranking: list[str],
    positives: set[str],
    junk: set[str] | None = None,
) -> float:
    if not positives:
        return 0.0

    ignored = junk or set()
    hits = 0
    precision_sum = 0.0
    evaluated_rank = 0

    for image_id in ranking:
        if image_id in ignored:
            continue
        evaluated_rank += 1
        if image_id in positives:
            hits += 1
            precision_sum += hits / evaluated_rank

    return precision_sum / len(positives)


def evaluate_query(
    query_id: str,
    results: list[RetrievalResult],
    gt: OxfordGroundTruth,
) -> QueryMetrics:
    positives = gt.good[query_id] | gt.ok[query_id]
    junk = gt.junk[query_id]
    ranking = [result.image_id for result in results]
    ap = average_precision(ranking, positives, junk)
    return QueryMetrics(
        query_id=query_id,
        ap=ap,
        positives=len(positives),
        ranked=len(ranking),
    )


def mean_average_precision(metrics: list[QueryMetrics]) -> float:
    if not metrics:
        return 0.0
    return sum(metric.ap for metric in metrics) / len(metrics)


def metrics_to_dict(metrics: list[QueryMetrics]) -> dict:
    return {
        "mAP": mean_average_precision(metrics),
        "queries": [
            {
                "query_id": metric.query_id,
                "ap": metric.ap,
                "positives": metric.positives,
                "ranked": metric.ranked,
            }
            for metric in metrics
        ],
    }

