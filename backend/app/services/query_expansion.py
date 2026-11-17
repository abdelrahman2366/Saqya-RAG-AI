"""Optional query expansion: extra phrasings of a question for the search, built without an LLM.

The farmer's own wording stays the main query (it keeps full weight and the keyword-search seat in
retrieval.Retriever.search); an expansion only adds candidates. Two kinds of phrasing are added:

1. **Bulletin vocabulary.** A question says "الفدان عايز كام كيلو بذرة؟" where the bulletin's heading says
   "معدل التقاوي". For each category query_understanding.py found, the words the bulletins' own headings use for
   it are added (taken from the library's section titles), together with the canonical name of a crop the
   farmer wrote in a short form ("قمح" -> "القمح").
2. **The other language's terms.** The glossary's fixed translation of every term found ("sugar beet",
   "planting date" -> "بنجر السكر ميعاد الزراعة"), so an English question can reach an Arabic bulletin (and the
   reverse) when the LLM translation is off, unavailable or wrong. Skipped when a translation is already
   among the search queries.

Safety rules, each covered by a test: nothing is added that the question does not imply (a crop appears in an
expansion only if the farmer named it; no pest, disease, dose or number is ever introduced), at most
`max_terms` words are added per phrasing and at most `max_variants` phrasings are returned, and any failure
returns no expansion — the search then runs exactly as before.

Off by default (QUERY_EXPANSION_ENABLED). scripts/evaluate.py measures it as experiments E5 and E6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.services.glossary import GLOSSARY, find_terms
from app.services.query_understanding import QueryUnderstanding, understand
from app.utils.text import normalize_for_match

logger = logging.getLogger(__name__)

# category -> the words the bulletins' headings use for it (seen in the library's own section titles)
BULLETIN_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "planting": {"ar": ("ميعاد الزراعة", "طرق الزراعة", "معدل التقاوي"), "en": ("planting", "sowing", "seed rate")},
    "varieties": {"ar": ("الأصناف", "الهجن"), "en": ("varieties", "hybrids")},
    "land preparation": {"ar": ("تجهيز الأرض للزراعة", "العزيق"), "en": ("land preparation",)},
    "irrigation": {"ar": ("الري",), "en": ("irrigation",)},
    "fertilization": {"ar": ("التسميد", "برنامج التسميد"), "en": ("fertilization", "fertilizer")},
    "pests": {"ar": ("الآفات الحشرية", "المكافحة"), "en": ("pest management", "insects")},
    "diseases": {"ar": ("الأمراض", "أعراض الإصابة", "المكافحة"), "en": ("disease management", "symptoms")},
    "weeds": {"ar": ("مكافحة الحشائش",), "en": ("weed management",)},
    "pesticide use": {"ar": ("المكافحة الكيماوية", "المبيدات"), "en": ("pesticide management",)},
    "harvest & storage": {"ar": ("الحصاد", "النضج", "التداول والتخزين"), "en": ("harvest", "storage", "post-harvest")},
    "soil": {"ar": ("الأرض المناسبة", "التربة"), "en": ("soil",)},
    "climate": {"ar": ("الظروف الجوية", "درجة الحرارة"), "en": ("climate", "temperature")},
    "protected cultivation": {"ar": ("الصوب الزراعية",), "en": ("greenhouse",)},
    "tree care": {"ar": ("التقليم", "التزهير", "الخف"), "en": ("pruning", "flowering")},
    "quality & disorders": {"ar": ("أسباب الظاهرة", "الجودة"), "en": ("quality",)},
    "sustainability": {"ar": ("الزراعة المستدامة",), "en": ("sustainable land management",)},
    "economics": {"ar": ("التسويق", "التصدير"), "en": ("marketing", "costs")},
}
# for these categories the right heading depends on what is asked: "when" -> the sowing-date section,
# "how much" -> the seed-rate section (otherwise "ميعاد الزراعة" pulls every crop's planting-date passage)
INTENT_TERMS: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("planting", "when"): {"ar": ("ميعاد الزراعة",), "en": ("planting date",)},
    ("planting", "how much"): {"ar": ("معدل التقاوي",), "en": ("seed rate",)},
    ("planting", "how to"): {"ar": ("طرق الزراعة",), "en": ("planting method",)},
    ("harvest & storage", "when"): {"ar": ("الحصاد", "النضج"), "en": ("harvest", "maturity")},
    ("fertilization", "how much"): {"ar": ("معدلات التسميد",), "en": ("fertilizer rate",)},
}
_CROP_TERMS = {t.en: t for t in GLOSSARY if t.kind == "crop"}


def bulletin_terms(category: str, intent: str, language: str) -> tuple[str, ...]:
    """The headings' words for a category, narrowed by the question's intent when that matters."""
    specific = INTENT_TERMS.get((category, intent))
    if specific:
        return specific.get(language, ())
    if (category, "when") in INTENT_TERMS or (category, "how much") in INTENT_TERMS:
        return BULLETIN_TERMS.get(category, {}).get(language, ())[:1]   # no intent match: the general heading only
    return BULLETIN_TERMS.get(category, {}).get(language, ())


@dataclass
class Expansion:
    variants: list[str] = field(default_factory=list)   # extra phrasings, best first (never the original)
    added: list[str] = field(default_factory=list)      # every word or phrase that was added, for the logs
    kinds: list[str] = field(default_factory=list)      # "vocabulary" / "other-language", one per variant

    def __bool__(self) -> bool:
        return bool(self.variants)


def _missing(names: list[str], text: str, limit: int) -> list[str]:
    """Names not already in the text (compared in normalised form, article ignored), at most `limit`."""
    seen = normalize_for_match(text)
    out: list[str] = []
    for name in names:
        form = normalize_for_match(name)
        bare = " ".join(w[2:] if w.startswith("ال") and len(w) > 4 else w for w in form.split())
        if form not in seen and bare not in seen and name not in out:
            out.append(name)
    return out[:limit]


def expand(question: str, understanding: QueryUnderstanding | None = None, *, max_variants: int = 2,
           max_terms: int = 4, other_language: bool = True) -> Expansion:
    """Extra phrasings for one question (already spelling-repaired and rewritten into standard Arabic).
    `other_language=False` when the search already has an LLM translation of the question."""
    try:
        return _expand(question, understanding or understand(question), max_variants, max_terms, other_language)
    except Exception as exc:  # noqa: BLE001 - an expansion must never break a search
        logger.warning("Query expansion failed (%s); searching with the question as it is", exc)
        return Expansion()


def _expand(question: str, u: QueryUnderstanding, max_variants: int, max_terms: int, other_language: bool) -> Expansion:
    out = Expansion()
    language = u.language
    crops = [_CROP_TERMS[c] for c in u.crops if c in _CROP_TERMS and "crop_from" not in u.conditions]

    # 1. the bulletins' own words for what was asked, and the crop's full name
    names = [c.ar if language == "ar" else c.en for c in crops]
    for category in u.categories[:1]:   # the main subject only: a second category's headings mostly add noise
        names += list(bulletin_terms(category, u.intent, language))
    extra = _missing(names, question, max_terms)
    if extra:
        out.variants.append(f"{question.rstrip(' ?؟.')} {' '.join(extra)}")
        out.added += extra
        out.kinds.append("vocabulary")

    # 2. the same terms in the other language (the glossary's fixed translations)
    if other_language:
        target = "en" if language == "ar" else "ar"
        terms = find_terms(question, language)
        terms += [c for c in crops if c not in terms]
        words = [t.en if target == "en" else t.ar for t in terms]
        for category in u.categories[:1]:
            words += list(bulletin_terms(category, u.intent, target)[:1])
        words = list(dict.fromkeys(words))[:max_terms]
        # only worth a search when it says *what* is asked, not just which crop: a bare crop name ("mango") ranks
        # the crop's general passages first
        if len(words) >= 2 and any(t.kind != "crop" for t in terms) or (crops and len(words) >= 2):
            out.variants.append(" ".join(words))
            out.added += words
            out.kinds.append("other-language")

    out.variants, out.kinds = out.variants[:max_variants], out.kinds[:max_variants]
    return out
