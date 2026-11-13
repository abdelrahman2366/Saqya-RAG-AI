"""Repairs a mis-heard or mistyped farming word in a question, before it is searched and answered.

Speech recognition sometimes splits or bends a word ("محاصيل" heard as "مع حصيل"), and typing does the same
("fertlizer", "الطماطمم"). The broken word then pollutes the search and shows up in the answer.

The repair is deliberately narrow, so a correct question is never changed:
- only a token that is not a real word (very rare in a large word-frequency list) is touched;
- it is replaced only by a farming word (the glossary's crops and practices, plus everyday farming words) that
  is spelled almost the same;
- two neighbouring tokens are joined only when the second is not a real word and together they are almost a
  farming word.
No AI is involved; without the `wordfreq` package nothing is changed.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache

logger = logging.getLogger(__name__)

_AR_WORDS = """محاصيل محصول زراعة الزراعة تسميد سماد أسمدة الري ري مبيد مبيدات آفة آفات تقاوي شتلات شتلة حصاد فدان
تربة التربة بذور بذرة مكافحة أمراض مرض حشرات حشرة الرش ملوحة خضروات خضر فاكهة أشجار شجرة ثمار أوراق جذور
نبات نباتات مزرعة فلاح مزارع عروة العروة الشتل التقليم التطعيم الحشائش كمبوست صوبة الصوب تنقيط بالتنقيط
ميعاد مواعيد معدل كمية إنتاج إنتاجية أصناف صنف الندوة البياض الصدأ الذبول المن العنكبوت دودة سوسة
ماشية دواجن أرانب نحل أعلاف علف برسيم يوريا نترات سوبر فوسفات بوتاسيوم نيتروجين فوسفور عضوي""".split()
_EN_WORDS = """crop crops farming farmer agriculture fertilizer fertilizers irrigation irrigate pesticide pesticides
harvest harvesting planting sowing seedling seedlings seeds soil salinity disease diseases pests weeds compost
manure nitrogen phosphorus potassium greenhouse drip yield variety varieties pruning grafting livestock poultry
feddan hectare tomatoes potatoes vegetables orchard nursery""".split()


def _norm(word: str) -> str:
    from app.utils.text import normalize_for_match

    return normalize_for_match(word)


@lru_cache(maxsize=1)
def _vocabulary() -> dict[str, dict[str, str]]:
    """{language: {normalised spelling: the word as it should be written}}"""
    from app.services.glossary import GLOSSARY

    ar, en = {}, {}
    for word in _AR_WORDS:
        ar.setdefault(_norm(word), word)
    for word in _EN_WORDS:
        en.setdefault(word.lower(), word)
    for term in GLOSSARY:
        for text in (term.ar, *getattr(term, "ar_alt", ())):
            for word in text.split():
                if len(word) >= 4:
                    ar.setdefault(_norm(word), word)
        for word in term.en.split():
            if len(word) >= 5:
                en.setdefault(word.lower(), word)
    for key, word in list(ar.items()):          # with the article too: الطماطم
        if not key.startswith("ال"):
            ar.setdefault("ال" + key, "ال" + word)
    return {"ar": ar, "en": en}


def _frequency(word: str, language: str) -> float | None:
    try:
        from wordfreq import zipf_frequency
    except Exception:  # noqa: BLE001 - not installed: no repair
        return None
    return zipf_frequency(word, language)


def _closest(word: str, language: str, minimum: float) -> str | None:
    key = _norm(word) if language == "ar" else word.lower()
    vocab = _vocabulary()[language]
    if key in vocab:
        return vocab[key]
    best, best_ratio = None, minimum
    for candidate, written in vocab.items():
        if abs(len(candidate) - len(key)) > 2 or candidate[0] != key[0]:
            continue
        ratio = SequenceMatcher(None, key, candidate).ratio()
        if ratio > best_ratio:
            best, best_ratio = written, ratio
    return best


_AR_LETTERS = "\u0621-\u0652\u0671-\u06D3"
_TOKEN_RE = re.compile(rf"[{_AR_LETTERS}]+|[A-Za-z]+|[^{_AR_LETTERS}A-Za-z]+")
_IS_AR = re.compile(rf"[{_AR_LETTERS}]")


def repair(text: str) -> str:
    """The question with mis-heard / mistyped farming words put right (unchanged when nothing is wrong)."""
    if not text or _frequency("test", "en") is None:
        return text
    tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        arabic = bool(_IS_AR.match(tok))
        latin = bool(re.match(r"[A-Za-z]", tok))
        if not (arabic or latin):
            out.append(tok)
            i += 1
            continue
        lang = "ar" if arabic else "en"
        known = _norm(tok) in _vocabulary()["ar"] if arabic else tok.lower() in _vocabulary()["en"]
        # two tokens that are really one word: "مع حصيل" -> "محاصيل"
        if arabic and not known and i + 2 < len(tokens) and tokens[i + 1].isspace() \
                and _IS_AR.match(tokens[i + 2]):
            nxt = tokens[i + 2]
            if _norm(nxt) not in _vocabulary()["ar"] and (_frequency(nxt, "ar") or 0) < 2.0 and len(tok) <= 4:
                joined = tok + nxt
                fixed = _closest(joined, "ar", 0.8) if len(joined) >= 5 else None
                if fixed:
                    logger.info("Spelling repair: %s %s -> %s", tok, nxt, fixed)
                    out.append(fixed)
                    i += 3
                    continue
        # one bent word: "fertlizer" -> "fertilizer", "الطماطمم" -> "الطماطم"
        if not known and len(tok) >= (4 if arabic else 5) and (_frequency(tok, lang) or 0) < 1.5:
            fixed = _closest(tok, lang, 0.84)
            if fixed and fixed != tok:
                logger.info("Spelling repair: %s -> %s", tok, fixed)
                tok = fixed if not (latin and tok[0].isupper()) else fixed.capitalize()
        out.append(tok)
        i += 1
    return "".join(out)
