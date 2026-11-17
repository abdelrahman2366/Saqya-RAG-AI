"""Metrics used by scripts/evaluate.py. Pure Python/NumPy, so they are unit-tested without any model.

Ranking metrics take, per question, the ranked list of "is this result relevant?" flags and the number of
relevant passages that exist. Probability metrics take predicted probabilities and 0/1 outcomes.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ------------------------------------------------------------------ ranking
def recall_at_k(hits: Sequence[bool], total_relevant: int, k: int) -> float:
    """Share of the relevant passages found in the top k (capped: a question with 30 answers can't fit in 5)."""
    if total_relevant <= 0:
        return 0.0
    return sum(hits[:k]) / min(total_relevant, k)


def hit_at_k(hits: Sequence[bool], k: int) -> float:
    """1 when at least one relevant passage is in the top k (the notebook's "retrieval relevant")."""
    return float(any(hits[:k]))


def precision_at_k(hits: Sequence[bool], k: int) -> float:
    return sum(hits[:k]) / k if k else 0.0


def mrr_at_k(hits: Sequence[bool], k: int) -> float:
    for rank, hit in enumerate(hits[:k], 1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: Sequence[bool], total_relevant: int, k: int) -> float:
    """Binary-relevance nDCG: discounted gain of the ranking over that of a perfect ranking."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(hits[:k], 1) if hit)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(total_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


def ranking_summary(rows: Sequence[tuple[Sequence[bool], int]]) -> dict[str, float]:
    """Mean metrics over questions; `rows` = (hits, total_relevant) per question."""
    if not rows:
        return {}
    def mean(values) -> float:
        return round(float(np.mean(list(values))), 4)
    return {
        "questions": len(rows),
        "hit@5": mean(hit_at_k(h, 5) for h, _ in rows),
        "hit@6": mean(hit_at_k(h, 6) for h, _ in rows),
        "recall@5": mean(recall_at_k(h, n, 5) for h, n in rows),
        "recall@10": mean(recall_at_k(h, n, 10) for h, n in rows),
        "precision@5": mean(precision_at_k(h, 5) for h, _ in rows),
        "mrr@10": mean(mrr_at_k(h, 10) for h, _ in rows),
        "ndcg@10": mean(ndcg_at_k(h, n, 10) for h, n in rows),
    }


# ------------------------------------------------------------------ probabilities (confidence)
def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    return round(float(np.mean((p - y) ** 2)), 4) if len(p) else float("nan")


def expected_calibration_error(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float:
    """ECE: the gap between stated confidence and observed accuracy, averaged over equal-width bins."""
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    if not len(p):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if inside.any():
            error += inside.mean() * abs(p[inside].mean() - y[inside].mean())
    return round(float(error), 4)


def auroc(scores: Sequence[float], outcomes: Sequence[int]) -> float:
    """Probability that a positive case scores higher than a negative one (ties count half)."""
    s, y = np.asarray(scores, dtype=float), np.asarray(outcomes, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return round(float(wins / (len(pos) * len(neg))), 4)


def risk_coverage(probabilities: Sequence[float], outcomes: Sequence[int], threshold: float) -> dict[str, float]:
    """What happens if only cases with confidence >= threshold are answered: coverage (share answered),
    selective accuracy (share of the answered that were right), and how many wrong ones were let through."""
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=int)
    answered = p >= threshold
    return {
        "threshold": round(float(threshold), 3),
        "coverage": round(float(answered.mean()), 4) if len(p) else float("nan"),
        "selective_accuracy": round(float(y[answered].mean()), 4) if answered.any() else float("nan"),
        "wrong_answered": int(((y == 0) & answered).sum()),
        "right_refused": int(((y == 1) & ~answered).sum()),
    }


def classification_report(predicted: Sequence[set], expected: Sequence[set]) -> dict[str, float]:
    """Micro precision / recall / F1 and exact-match accuracy for (multi)label predictions."""
    tp = sum(len(p & e) for p, e in zip(predicted, expected))
    fp = sum(len(p - e) for p, e in zip(predicted, expected))
    fn = sum(len(e - p) for p, e in zip(predicted, expected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(p == e for p, e in zip(predicted, expected)) / len(expected) if expected else 0.0
    return {"n": len(expected), "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "exact_match": round(exact, 4)}
