"""Claim-level grounding: is each statement of an answer supported by the passages it was written from?

grounding.py already judges an answer as a whole (citations, numbers, word overlap) and repairs or removes
what fails. This module adds a per-claim verdict, in three steps:

1. **Claims.** The answer is split into its statements (sentences and list items); refusals, safety lines,
   questions back to the farmer and connective fragments are not claims.
2. **Evidence.** For each claim, the passage it cites — or, without a citation, the passage sharing most of
   its content words — is its evidence.
3. **Verdict.**
   - `supported`: the claim's words are in the evidence and every number in it is in that same passage;
   - `partial`: most words are there, or its numbers are found only in another passage (right fact,
     wrong source);
   - `unsupported`: little overlap, or a number that is in no passage at all;
   - `contradicted`: only from the optional NLI model (words alone cannot tell "do not irrigate" from
     "irrigate");
   - `unverified`: the claim and its evidence are in different languages, so word overlap means nothing.
     Its numbers are still checked (digits survive translation); with the NLI model on it is judged.

Numbers are compared with grounding.py's normalisation (Arabic digits, "2,5" = "2.5", the reversed forms a
right-to-left PDF produces) and a quantity must be in the *same* passage as the rest of the claim, which is
what catches "30 kg" taken from the sunflower paragraph and attached to soybean. Units: a claim that gives a
quantity per feddan while its evidence speaks only per hectare is marked `partial` with the reason named.

The optional NLI model (CLAIM_VERIFICATION_NLI=true; default model MoritzLaurer/mDeBERTa-v3-base-xnli-
multilingual-nli-2mil7, multilingual incl. Arabic) is loaded lazily, never downloaded automatically, and can
only *add* evidence: entailment upgrades partial/unverified to supported, contradiction marks contradicted.
The lexical and numeric checks always run, so the LLM-free verdict never depends on one model.

Off by default (CLAIM_VERIFICATION_ENABLED). With it on, the verdicts are attached to the result and feed
confidence.py; the answer text itself is still repaired only by grounding.py's existing rules.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Sequence

from app.services import grounding
from app.utils.text import detect_language

logger = logging.getLogger(__name__)

SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED, UNVERIFIED = (
    "supported", "partial", "unsupported", "contradicted", "unverified")

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# thresholds on the share of a claim's content words found in its evidence passage. 0.6 is the value
# grounding.py already uses for a whole answer (SUPPORT_THRESHOLD); below 0.3 fewer than a third of the
# words are there. scripts/evaluate.py reports how these behave on perturbed passages.
SUPPORT_AT = grounding.SUPPORT_THRESHOLD
PARTIAL_AT = 0.3

_QUESTION_END = re.compile(r"[?؟]\s*$")
_MIN_WORDS = 3


@dataclass
class Claim:
    text: str
    verdict: str = UNVERIFIED
    evidence: int | None = None          # 1-based passage number, as in the answer's [n]
    overlap: float | None = None         # share of the claim's content words found in the evidence
    numbers: list[str] = field(default_factory=list)          # numbers the claim states
    missing_numbers: list[str] = field(default_factory=list)  # ... that are in no passage
    misplaced_numbers: list[str] = field(default_factory=list)  # ... that are only in another passage
    reason: str = ""
    nli: str = ""                        # entailment / neutral / contradiction, when the NLI model ran

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClaimReport:
    claims: list[Claim] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out = {v: 0 for v in (SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED, UNVERIFIED)}
        for c in self.claims:
            out[c.verdict] += 1
        return out

    @property
    def supported_share(self) -> float | None:
        """Supported claims (partial counts half) over the claims that could be judged; None if none could."""
        judged = [c for c in self.claims if c.verdict != UNVERIFIED]
        if not judged:
            return None
        score = sum(1.0 if c.verdict == SUPPORTED else 0.5 if c.verdict == PARTIAL else 0.0 for c in judged)
        return round(score / len(judged), 3)

    @property
    def worst(self) -> str:
        for verdict in (CONTRADICTED, UNSUPPORTED, PARTIAL, UNVERIFIED, SUPPORTED):
            if any(c.verdict == verdict for c in self.claims):
                return verdict
        return SUPPORTED

    def to_dict(self) -> dict:
        return {"counts": self.counts, "supported_share": self.supported_share,
                "claims": [c.to_dict() for c in self.claims]}


# ------------------------------------------------------------------ 1. claims
def extract_claims(answer: str) -> list[str]:
    """The statements of an answer that assert something (with their [n] markers, if any)."""
    for note in grounding.SAFETY_NOTE.values():
        answer = answer.replace(note, "")
    out = []
    for part in grounding.segments(answer):
        text = part.strip()
        if not text or text == "\n" or grounding.is_refusal(text) or _QUESTION_END.search(text):
            continue
        body = grounding.answer_body(text)
        if len(grounding.content_words(body)) < _MIN_WORDS and not grounding.numbers(body):
            continue   # "As follows:", "أولاً:" — nothing to verify
        out.append(text)
    return out


# ------------------------------------------------------------------ 2. evidence
def _overlap(words: set[str], passage_words: set[str]) -> float:
    return sum(w in passage_words for w in words) / len(words) if words else 0.0


def find_evidence(claim: str, chunks: Sequence, passage_words: list[set[str]]) -> tuple[int | None, float]:
    """(1-based passage number, word overlap) of the claim's evidence: the best of the passages it cites,
    else the best of all passages."""
    words = grounding.content_words(grounding.answer_body(claim))
    cited = [n for n in grounding.citations(claim, len(chunks))]
    pool = cited or range(1, len(chunks) + 1)
    best, best_score = None, -1.0
    for n in pool:
        score = _overlap(words, passage_words[n - 1])
        if score > best_score:
            best, best_score = n, score
    return best, max(best_score, 0.0)


# ------------------------------------------------------------------ 3. verdict
def verify_claim(claim: str, chunks: Sequence, passage_words: list[set[str]] | None = None,
                 passage_numbers: list[set[str]] | None = None, question: str = "") -> Claim:
    if passage_words is None:
        passage_words = [grounding.content_words(c.text) for c in chunks]
    if passage_numbers is None:
        passage_numbers = [grounding._passage_numbers(c.text) | grounding.numbers(getattr(c, "label", "")) for c in chunks]
    body = grounding.answer_body(claim)
    result = Claim(text=claim, numbers=sorted(grounding.numbers(body)))
    if not chunks:
        result.verdict, result.reason = UNSUPPORTED, "no passages"
        return result
    evidence, overlap = find_evidence(claim, chunks, passage_words)
    result.evidence = evidence
    passage = chunks[evidence - 1]
    everywhere = set().union(*passage_numbers) | grounding.numbers(question)
    here = passage_numbers[evidence - 1] | grounding.numbers(question)
    result.missing_numbers = [n for n in result.numbers if n not in everywhere]
    result.misplaced_numbers = [n for n in result.numbers if n in everywhere and n not in here]
    same_language = detect_language(body) == detect_language(passage.text)

    if result.missing_numbers:
        result.verdict = UNSUPPORTED
        result.reason = "number not in any passage: " + ", ".join(result.missing_numbers)
        result.overlap = round(overlap, 2) if same_language else None
        return result
    if not same_language:
        # words cannot be compared across languages; the numbers were (digits survive translation)
        result.verdict = PARTIAL if result.misplaced_numbers else UNVERIFIED
        result.reason = ("number only in another passage: " + ", ".join(result.misplaced_numbers)
                         if result.misplaced_numbers else "claim and evidence are in different languages")
        return result
    result.overlap = round(overlap, 2)
    if overlap < PARTIAL_AT:
        result.verdict, result.reason = UNSUPPORTED, "its words are not in the passages"
    elif result.misplaced_numbers:
        result.verdict = PARTIAL
        result.reason = "number only in another passage: " + ", ".join(result.misplaced_numbers)
    elif overlap < SUPPORT_AT:
        result.verdict, result.reason = PARTIAL, "only part of its words are in the evidence"
    else:
        asked, used = grounding.area_units(passage.text), grounding.area_units(body)
        if result.numbers and used and asked and not used & asked:
            result.verdict = PARTIAL
            result.reason = f"unit differs from the evidence ({', '.join(sorted(used))} vs {', '.join(sorted(asked))})"
        else:
            result.verdict = SUPPORTED
    return result


def verify(answer: str, chunks: Sequence, question: str = "", nli: "NLIVerifier | None" = None) -> ClaimReport:
    """Verdict for every claim of an answer against the passages it was generated from."""
    passage_words = [grounding.content_words(c.text) for c in chunks]
    passage_numbers = [grounding._passage_numbers(c.text) | grounding.numbers(getattr(c, "label", "")) for c in chunks]
    report = ClaimReport([verify_claim(text, chunks, passage_words, passage_numbers, question)
                          for text in extract_claims(answer)])
    if nli is not None and report.claims and chunks:
        try:
            nli.refine(report, chunks)
        except Exception as exc:  # noqa: BLE001 - the lexical verdicts stand
            logger.warning("NLI claim check failed (%s); keeping the lexical verdicts", exc)
    return report


# ------------------------------------------------------------------ optional NLI model
class NLIVerifier:
    """Multilingual natural-language-inference model: does the evidence entail, contradict or say nothing
    about the claim? Lazy, shared, never downloaded automatically (see reranker.py for the same rules)."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device: str = "auto", batch_size: int = 8,
                 max_length: int = 512, allow_download: bool = False, entail_at: float = 0.7,
                 contradict_at: float = 0.8, pipeline=None) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.allow_download = allow_download
        self.entail_at = entail_at          # probabilities above which the model's label is trusted; the
        self.contradict_at = contradict_at  # contradiction bar is higher because it overrides word evidence
        self._model = pipeline              # tests pass a callable: list[(premise, hypothesis)] -> list[dict]
        self._failed: str | None = None
        self._lock = threading.Lock()

    @property
    def failed(self) -> str | None:
        return self._failed

    def load(self):
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                from app.services.reranker import local_model_path, pick_device

                location = local_model_path(self.model_name) or self.model_name
                offline = not self.allow_download and location == self.model_name
                tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=offline)
                model = AutoModelForSequenceClassification.from_pretrained(location, local_files_only=offline)
                device = pick_device(self.device)
                model.to(device).eval()
                labels = {i: str(name).lower() for i, name in model.config.id2label.items()}

                def predict(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
                    out: list[dict[str, float]] = []
                    for start in range(0, len(pairs), self.batch_size):
                        batch = pairs[start:start + self.batch_size]
                        enc = tokenizer([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                                        max_length=self.max_length, padding=True, return_tensors="pt").to(device)
                        with torch.no_grad():
                            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().tolist()
                        out += [{labels[i]: p for i, p in enumerate(row)} for row in probs]
                    return out

                self._model = predict
                logger.info("NLI model %s loaded on %s", self.model_name, device)
            except Exception as exc:  # noqa: BLE001
                self._failed = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning("NLI model %s is not available (%s); claims are checked by words and numbers only. "
                               "To install it: python scripts/download_models.py nli", self.model_name, self._failed)
        return self._model

    def refine(self, report: ClaimReport, chunks: Sequence) -> None:
        """Let the model upgrade partial/unverified claims it finds entailed, and flag contradictions.
        A claim with a number that is in no passage stays unsupported whatever the model says."""
        predict = self.load()
        if predict is None:
            return
        todo = [c for c in report.claims if c.evidence and not c.missing_numbers]
        if not todo:
            return
        pairs = [(chunks[c.evidence - 1].text, grounding.answer_body(c.text).strip()) for c in todo]
        for claim, probs in zip(todo, predict(pairs)):
            entail = probs.get("entailment", 0.0)
            contra = probs.get("contradiction", 0.0)
            claim.nli = max(probs, key=probs.get) if probs else ""
            if contra >= self.contradict_at:
                claim.verdict, claim.reason = CONTRADICTED, f"the evidence says otherwise (NLI {contra:.2f})"
            elif entail >= self.entail_at and claim.verdict in (PARTIAL, UNVERIFIED) and not claim.misplaced_numbers:
                claim.verdict, claim.reason = SUPPORTED, f"entailed by the evidence (NLI {entail:.2f})"
