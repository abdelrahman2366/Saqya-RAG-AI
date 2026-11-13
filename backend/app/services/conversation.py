"""Follow-up questions: "and how much water does it need?" after "when do I plant tomatoes?".

A follow-up names nothing by itself (it, its, this, ده, دي, كمان...) or is just a few words, so searching for
it alone finds the wrong passages. The chat app sends the farmer's last questions with every new one; when
the new one looks like a follow-up, the earlier question is searched together with it and shown to the LLM
as context. A question that names its own crop and has enough words stands on its own. Rules, no AI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.services.dialect import to_msa
from app.services.glossary import crops_in
from app.utils.text import normalize_for_match

# words that point back to something said before (Arabic on normalised text: أ/إ/آ -> ا, ة -> ه, ى -> ي)
_REFERS_EN = re.compile(r"\b(?:it|its|it's|they|them|their|theirs|this|that|these|those|there|he|she|him|her|"
                        r"also|too|else|same|then|instead|one)\b", re.IGNORECASE)
_REFERS_AR = re.compile(r"(?<![\w])(?:هو|هي|هم|هما|ده|دي|دا|دول|هذا|هذه|ذلك|تلك|هؤلاء|كمان|برضه|برضو|برده|ايضا|"
                        r"نفس|طيب|طب|بالنسبه|الكلام|عنه|عنها|عليه|عليها|فيه|فيها|منه|منها|له|لها)(?![\w])")
# "what about tomatoes?", "and in winter?", "وماذا عن الطماطم؟", "طب والقمح؟"
_CONTINUES = re.compile(r"^\s*(?:and|so|but|what about|how about|and what about|ok(?:ay)?|then)\b|"
                        r"^\s*(?:و(?:ماذا|ما|كيف|متى|كم|اين|لو)|طيب|طب|وبالنسبه|بالنسبه|يعني|طيب و)", re.IGNORECASE)
_QUESTION_FORM = re.compile(r"(?<![\w])(?:ما|من|ماذا) (?:هو|هي|هم)(?![\w])")
_WORD_RE = re.compile(r"[\w']+")
_SMALL = {"the", "a", "an", "is", "are", "do", "does", "i", "my", "of", "for", "in", "on", "to", "how", "what", "when",
          "which", "why", "where", "much", "many", "can", "should", "ما", "ماذا", "كيف", "متى", "كم", "هل", "من", "في",
          "على", "الى", "عن", "اين", "لماذا", "ايه", "ازاي", "امتى", "فين", "ليه", "اعمل", "افعل"}


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(normalize_for_match(text)) if w not in _SMALL and len(w) > 1]


@dataclass(frozen=True)
class FollowUp:
    previous: str     # the earlier question
    strong: bool      # clearly a follow-up (it / this / "what about…" / two words): search only with both together

    def search_texts(self, question: str) -> list[str]:
        """What to search: the two questions together, and — when it may stand on its own — the question alone too."""
        combined = f"{self.previous} {question}"
        return [combined] if self.strong else [question, combined]


def follow_up_context(question: str, history: Sequence[str] | None) -> FollowUp | None:
    """The earlier question this one follows up on, or None when it stands on its own."""
    previous = next((h.strip() for h in reversed(history or []) if h and h.strip()), None)
    if not previous or not question.strip():
        return None
    q = normalize_for_match(to_msa(question))
    words = _content_words(question)
    own_crops = crops_in(question)
    earlier_crops = crops_in(previous)
    q_refs = _QUESTION_FORM.sub(" ", q)              # "ما هو / ما هي" = "what is", not "it"
    refers = bool(_REFERS_EN.search(question) or _REFERS_AR.search(q_refs))
    continues = bool(_CONTINUES.search(question) or _CONTINUES.search(q))
    if continues and len(words) <= 6:
        return FollowUp(previous, True)              # "what about tomatoes?" / "وفي الشتاء؟"
    if own_crops:
        return None                                  # names its own crop: a question of its own
    if refers and len(words) <= 8:
        return FollowUp(previous, True)              # "how much water does it need?"
    if len(words) <= 2:
        return FollowUp(previous, True)              # "and the dose?" / "الكمية؟"
    if earlier_crops and len(words) <= 4:
        return FollowUp(previous, False)             # "بتحتاج مية قد إيه؟" after a question about a crop — or new
    return None
