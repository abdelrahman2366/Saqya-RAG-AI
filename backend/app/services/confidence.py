"""How sure is SAQYA of an answer, and what should it do about it: answer, qualify, ask, or abstain?

**Signals** (all cheap, no model call), gathered from what the pipeline already computed:
- retrieval: best and mean cosine similarity of the passages, the margin over the relevance gate, how many
  of the question's content words the passages contain, whether a passage is about the crop that was
  named, the cross-encoder's top score when reranking is on;
- the question: is something missing that the answer depends on (query_understanding.py)?
- the answer: share of claims supported (claims.py), whether anything had to be retried or removed.

**Probability.** A logistic model turns the retrieval signals into P(the passages contain the answer). Its
weights are *fitted on the validation half of the evaluation questions* by `scripts/evaluate.py --calibrate`
and saved to data/confidence_calibration.json together with the two thresholds chosen on that same half
and the test-half metrics (Brier score, ECE, AUROC, coverage). Nothing here is a hand-picked number: without
that file there is no probability and no threshold, and the decision uses only structural facts (below).

**Decision**
- `abstain`: with calibration, P below `abstain_at` — the library search is treated as having found nothing
  (the other sources in beyond.py are then tried, exactly as when the similarity gate fails).
- `ask`: no source answered and the question lacks something an answer needs (usually the crop): the
  refusal becomes a short clarifying question.
- `qualify`: the answer stands but carries a one-line caution — with calibration when P is below `answer_at`;
  always when a claim is unsupported or contradicted.
- `answer`: otherwise.

Off by default (CONFIDENCE_CALIBRATION_ENABLED). With it off nothing in the pipeline changes.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from app.services import grounding
from app.services.glossary import crops_in

logger = logging.getLogger(__name__)

ANSWER, QUALIFY, ASK, ABSTAIN = "answer", "qualify", "ask", "abstain"

# the retrieval signals the logistic model is fitted on (order matters: it is the weight order in the file)
FEATURES = ("best_similarity", "mean_similarity", "gate_margin", "word_coverage", "crop_match", "rerank_top")

QUALIFY_NOTE = {
    "en": "I'm not fully sure this covers your case — please check it with your local extension officer before acting on it.",
    "ar": "لست متأكدة تمامًا أن هذا ينطبق على حالتك — يُفضّل مراجعة المرشد الزراعي في منطقتك قبل التنفيذ.",
}
# added under an answer to a "what is wrong with my plants?" question that names no crop: the answer can only be
# general, so the farmer is told what would let SAQYA answer properly
FOLLOW_UP = {
    "en": "To help you more precisely: which crop is it, what exactly do you see on the plant (yellowing, wilting, "
          "spots, and on which part), and how often do you irrigate?",
    "ar": "حتى أساعدك بدقة أكثر: ما هو المحصول، وماذا تلاحظ على النبات بالضبط (اصفرار أو ذبول أو بقع، وفي أي جزء)، "
          "وكل كم يومًا تروي؟",
}
ASK_TEXT = {
    "crop": {"en": "Which crop is this about? Tell me the crop and I'll look it up for you.",
             "ar": "عن أي محصول تسأل؟ اذكر لي اسم المحصول وسأبحث لك عن الإجابة."},
    "symptoms": {"en": "What exactly do you see on the plant (colour, spots, which part), and on which crop?",
                 "ar": "ماذا تلاحظ على النبات بالضبط (اللون، البقع، أي جزء)، وما هو المحصول؟"},
}


@dataclass
class Signals:
    best_similarity: float = 0.0
    mean_similarity: float = 0.0     # of the three closest passages
    gate_margin: float = 0.0         # best similarity minus the relevance gate
    word_coverage: float = 0.0       # share of the question's content words found in the passages
    crop_match: float = 0.5          # 1 a passage is about the named crop, 0 none is, 0.5 no crop named
    rerank_top: float = 0.5          # sigmoid of the best cross-encoder score (0.5 when reranking is off)
    passages: int = 0
    missing: list[str] = field(default_factory=list)
    supported_share: float | None = None
    worst_claim: str = ""
    retried: bool = False

    def vector(self) -> np.ndarray:
        return np.array([float(getattr(self, name)) for name in FEATURES], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Confidence:
    decision: str = ANSWER
    probability: float | None = None     # None: no calibration file, so no probability is claimed
    calibrated: bool = False
    reasons: list[str] = field(default_factory=list)
    signals: Signals = field(default_factory=Signals)

    def to_dict(self) -> dict:
        return {"decision": self.decision, "probability": self.probability, "calibrated": self.calibrated,
                "reasons": self.reasons, "signals": self.signals.to_dict()}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 30.0), -30.0)))


def retrieval_signals(queries: Sequence[str], chunks: Sequence, gate: float = 0.0,
                      missing: Sequence[str] = ()) -> Signals:
    """Signals available before any answer is written."""
    signals = Signals(passages=len(chunks), missing=list(missing))
    if not chunks:
        signals.gate_margin = -gate
        return signals
    scores = sorted((float(c.score) for c in chunks if c.score == c.score), reverse=True) or [0.0]
    signals.best_similarity = round(scores[0], 4)
    signals.mean_similarity = round(float(np.mean(scores[:3])), 4)
    signals.gate_margin = round(scores[0] - gate, 4)
    context: set[str] = set()
    for c in chunks:
        context |= grounding.content_words(c.text)
    coverage = 0.0
    for q in queries:   # the phrasing in the passages' language is the one whose words can be found
        words = grounding.content_words(q)
        if words:
            coverage = max(coverage, sum(w in context for w in words) / len(words))
    signals.word_coverage = round(coverage, 3)
    named = []
    for q in queries:
        named += [t for t in crops_in(q) if t not in named]
    if named:
        signals.crop_match = float(any(grounding.is_about(c, t) for c in chunks for t in named))
    reranked = [c.rerank_score for c in chunks if getattr(c, "rerank_score", None) is not None]
    if reranked:
        signals.rerank_top = round(_sigmoid(max(reranked)), 4)
    return signals


# ------------------------------------------------------------------ calibration
@dataclass
class Calibration:
    """Logistic model over FEATURES plus the thresholds chosen on validation data."""
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    answer_at: float | None = None     # below this the answer is qualified
    abstain_at: float | None = None    # below this the library's passages are not used
    info: dict = field(default_factory=dict)   # what it was fitted on, and the measured metrics

    def probability(self, signals: Signals) -> float:
        x = (signals.vector() - np.asarray(self.mean)) / np.asarray(self.scale)
        return round(_sigmoid(float(x @ np.asarray(self.coef)) + self.intercept), 4)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"features": list(FEATURES), **asdict(self)}, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibration | None":
        """The saved calibration, or None when there is none or it was fitted on other features."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if tuple(data.pop("features", ())) != FEATURES:
                logger.warning("%s was fitted on other signals; ignored (run scripts/evaluate.py --calibrate)", path)
                return None
            return cls(**data)
        except Exception as exc:  # noqa: BLE001 - a broken file means "not calibrated", never a crash
            logger.warning("Could not read %s (%s); confidence is uncalibrated", path, exc)
            return None


def fit_calibration(rows: Sequence[Signals], outcomes: Sequence[int], l2: float = 1.0, steps: int = 500) -> Calibration:
    """L2-regularised logistic regression (Newton's method) on standardised signals. Small data: the
    regularisation keeps the weights modest, and a feature that never varies gets weight 0."""
    x = np.stack([r.vector() for r in rows])
    y = np.asarray(outcomes, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    design = np.hstack([z, np.ones((len(z), 1))])
    w = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[-1, -1] = 0.0   # the intercept is not regularised
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ w, -30, 30)))
        gradient = design.T @ (p - y) + penalty @ w
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + penalty + np.eye(len(w)) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return Calibration(mean=mean.round(6).tolist(), scale=scale.round(6).tolist(), coef=w[:-1].round(6).tolist(),
                       intercept=round(float(w[-1]), 6))


def choose_thresholds(probabilities: Sequence[float], outcomes: Sequence[int],
                      target_accuracy: float = 0.9) -> tuple[float | None, float | None]:
    """(answer_at, abstain_at) from validation data.
    answer_at: the lowest threshold at which the cases at or above it are right at least `target_accuracy` of the
    time (None if no threshold reaches it). abstain_at: the threshold that best separates right from wrong
    (maximum of true-positive rate minus false-positive rate); None when there are no wrong cases to learn from."""
    p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=int)
    if not len(p) or y.min() == y.max():
        return None, None
    answer_at = None
    for t in sorted(set(p.round(4).tolist())):
        kept = p >= t
        if kept.sum() >= max(5, 0.2 * len(p)) and y[kept].mean() >= target_accuracy:
            answer_at = float(t)
            break
    best_j, abstain_at = -1.0, None
    for t in sorted(set(p.round(4).tolist())):
        kept = p >= t
        j = (kept & (y == 1)).sum() / (y == 1).sum() - (kept & (y == 0)).sum() / (y == 0).sum()
        if j > best_j:
            best_j, abstain_at = j, float(t)
    if answer_at is not None and abstain_at is not None:
        abstain_at = min(abstain_at, answer_at)
    return answer_at, abstain_at


# ------------------------------------------------------------------ decision
class ConfidenceEstimator:
    def __init__(self, calibration: Calibration | None = None) -> None:
        self.calibration = calibration

    @classmethod
    def from_file(cls, path: Path) -> "ConfidenceEstimator":
        calibration = Calibration.load(path)
        if calibration is None:
            logger.info("Confidence: no calibration at %s; decisions use claim checks and missing information only "
                        "(python scripts/evaluate.py --calibrate writes it)", path)
        return cls(calibration)

    def probability(self, signals: Signals) -> float | None:
        return self.calibration.probability(signals) if self.calibration is not None else None

    def before_answer(self, signals: Signals) -> Confidence:
        """Decision from the retrieval signals alone: abstain (do not use these passages) or go on."""
        p = self.probability(signals)
        out = Confidence(probability=p, calibrated=p is not None, signals=signals)
        if signals.passages and p is not None and self.calibration.abstain_at is not None \
                and p < self.calibration.abstain_at:
            out.decision = ABSTAIN
            out.reasons.append(f"P(passages hold the answer) {p:.2f} < {self.calibration.abstain_at:.2f}")
        return out

    def after_answer(self, signals: Signals, answered: bool, off_topic: bool = False) -> Confidence:
        """Final decision once the answer (or the refusal) is known."""
        p = self.probability(signals) if signals.passages else None
        out = Confidence(probability=p, calibrated=p is not None, signals=signals)
        if off_topic:
            out.decision = ABSTAIN
            out.reasons.append("not a farming question")
        elif not answered:
            if signals.missing:
                out.decision = ASK
                out.reasons.append("the question does not say: " + ", ".join(signals.missing))
            else:
                out.decision = ABSTAIN
                out.reasons.append("no source answered")
        else:
            if signals.worst_claim in ("unsupported", "contradicted"):
                out.decision = QUALIFY
                out.reasons.append(f"a statement is {signals.worst_claim} by the passages")
            if p is not None and self.calibration.answer_at is not None and p < self.calibration.answer_at:
                out.decision = QUALIFY
                out.reasons.append(f"P(passages hold the answer) {p:.2f} < {self.calibration.answer_at:.2f}")
        return out
