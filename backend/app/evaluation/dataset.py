"""The evaluation questions, the relevance rule and the fixed validation/test split.

Questions come from data/eval_questions.json (standard Arabic and English) and data/eval_questions_egyptian.json
(the same questions as an Egyptian farmer would ask them; `msa_id` names the twin). A passage is relevant when
it comes from the expected document and contains one of the expected phrases — the rule the notebooks use, and
one that does not depend on how the documents were chunked.

The split is by *question group*: an Egyptian question and its standard-Arabic twin always land in the same
half, so nothing tuned on the validation half has seen a paraphrase of a test question. It is derived from a
hash of the group id and a seed, so it is the same on every machine and needs no file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.services.grounding import contains_keyword
from app.utils.text import detect_language

DEFAULT_SEED = 13


@dataclass
class EvalQuestion:
    id: str
    question: str
    kind: str                  # in_domain · egyptian · out_of_scope
    category: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_source: str = ""
    answer_keywords: list[str] = field(default_factory=list)
    group: str = ""            # the id shared with paraphrases of the same question
    split: str = ""            # validation · test

    @property
    def language(self) -> str:
        return detect_language(self.question)

    @property
    def answerable(self) -> bool:
        return self.kind != "out_of_scope"

    def is_relevant(self, text: str, source: str) -> bool:
        if self.expected_source and self.expected_source.lower() not in (source or "").lower():
            return False
        return contains_keyword(text, self.expected_keywords)


def split_of(group: str, seed: int = DEFAULT_SEED, test_share: float = 0.5) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return "test" if int.from_bytes(digest[:4], "big") / 2**32 < test_share else "validation"


def load_questions(data_dir: Path, seed: int = DEFAULT_SEED, test_share: float = 0.5) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for name in ("eval_questions.json", "eval_questions_egyptian.json"):
        path = data_dir / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            group = row.get("msa_id") or row["id"]
            questions.append(EvalQuestion(
                id=row["id"], question=row["question"], kind=row.get("type", "in_domain"),
                category=row.get("category", ""), expected_keywords=row.get("expected_keywords", []),
                expected_source=row.get("expected_source", ""), answer_keywords=row.get("answer_keywords", []),
                group=group, split=split_of(group, seed, test_share),
            ))
    return questions


def subsets(questions: list[EvalQuestion]) -> dict[str, list[EvalQuestion]]:
    """The slices every experiment is reported on (answerable questions only)."""
    answerable = [q for q in questions if q.answerable]
    out = {
        "all": answerable,
        "standard Arabic": [q for q in answerable if q.kind == "in_domain" and q.language == "ar"],
        "Egyptian Arabic": [q for q in answerable if q.kind == "egyptian"],
        "English": [q for q in answerable if q.language == "en"],
        "cross-language": [q for q in answerable if q.category == "cross-language"],
    }
    return {name: rows for name, rows in out.items() if rows}
