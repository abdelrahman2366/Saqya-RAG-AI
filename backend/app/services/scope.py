"""Is a question about agriculture? Decides what happens when the documents don't cover a question.

A farming question the library can't answer ("give me a crop recommendation", "how do I raise rabbits?") goes
on to what SAQYA has learned, then to a web search, then to general knowledge. Anything else ("who won the
World Cup?", "a chocolate cake recipe") still gets the polite refusal.

A question with a clear farming word (farm, farming, agriculture, crops, livestock, fertilizer, irrigation…;
زراعة، فلاح، محصول، سماد…) is farming at once. Otherwise the local LLM decides (one short yes/no call): it tells
"tomato soup recipe" (cooking) from "the price of tomatoes" (farm business) better than any word list. When it
can't be reached, a word list decides: farming words in Arabic (Egyptian too) and English, and the crops and
practices of the glossary.
"""

from __future__ import annotations

import logging
import re

from app.services.dialect import to_msa
from app.services.glossary import find_terms
from app.utils.text import normalize_for_match

logger = logging.getLogger(__name__)

SCOPE_PROMPT = """Decide if a question is about agriculture.

Agriculture means: crops, vegetables, fruit trees, plants and gardening, soil, water and irrigation, fertilizers and \
compost, pests, weeds and plant diseases, pesticides, seeds and planting, harvest and storage, farm animals \
(livestock, poultry, rabbits, bees, fish farming), animal feed, farm machinery, farm business and markets, food \
production, land and climate as they affect farming.
Not agriculture: cooking and recipes, human health, sports, technology, politics, entertainment, general trivia, chat.

Question: {question}

Reply with one word: yes or no."""

# Farming words, for when the LLM can't be asked (on normalised text: أ/إ/آ -> ا, ة -> ه, ى -> ي).
_EN_WORDS = re.compile(
    r"\b(?:farm\w*|agricultur\w*|agronom\w*|horticultur\w*|crops?|harvest\w*|soils?|fertili[sz]\w*|manure|"
    r"compost\w*|irrigat\w*|seeds?|seedlings?|sow(?:ing|n)?|planting|orchards?|greenhouses?|pests?|pesticides?|"
    r"insecticides?|fungicides?|herbicides?|weeds?|livestock|cattle|cows?|buffalo(?:es)?|sheep|goats?|poultry|"
    r"chickens?|broilers?|hens?|bees?|beekeeping|beehives?|apicultur\w*|aquacultur\w*|fish ?farm\w*|tilapia|"
    r"rabbits?|feddans?|hectares?|tractors?|fodder|forage|silage|mulch\w*|prun\w*|graft\w*|nurser(?:y|ies)|"
    r"germinat\w*|agri\w*|vegetables?|fruit trees?|cereals?|grains?|yields?)\b", re.IGNORECASE)
_AR_STEMS = ("زراع", "زرع", "مزارع", "فلاح", "محصول", "محاصيل", "تربه", "سماد", "اسمده", "تسميد", "حصاد",
             "تقاوي", "بذور", "بذره", "شتل", "افه", "افات", "مبيد", "حشره", "حشرات", "حشائش", "ماشيه", "مواشي",
             "ابقار", "جاموس", "اغنام", "خراف", "ماعز", "دواجن", "فراخ", "دجاج", "نحل", "منحل", "استزراع", "اسماك",
             "صوبه", "صوب ", "بستان", "اشجار", "فاكهه", "خضر", "خضار", "فدان", "هكتار", "ارانب", "اعلاف", "علف",
             "غيط", "حقل", "نبات", "ثمار", "تقليم", "تطعيم", "مشتل", "كمبوست", "مزرعه", "ميكنه", "جرار", "ملوحه",
             "انبات", "حبوب", "ارض زراعيه", "الري", "ري ", "رش ")


# Words that make a question about farming whatever else it says ("give me more advice on farming"): no LLM needed.
# Crop names are not here on purpose: "tomato soup recipe" names a crop but is about cooking, so the LLM decides.
_EN_CLEAR = re.compile(r"\b(?:farm\w*|agricultur\w*|agronom\w*|horticultur\w*|crops?|livestock|poultry|beekeeping|"
                       r"aquacultur\w*|fertili[sz]\w*|irrigat\w*|pesticides?|harvest\w*|feddans?)\b", re.IGNORECASE)
_AR_CLEAR = ("زراع", "زرع", "فلاح", "محصول", "محاصيل", "مزرعه", "ماشيه", "مواشي", "دواجن", "سماد", "اسمده", "تسميد",
             "مبيد", "حصاد", "تقاوي", "فدان", "غيط")


def clearly_agricultural(question: str) -> bool:
    if _EN_CLEAR.search(question):
        return True
    text = f" {normalize_for_match(to_msa(question))} "
    return any(stem in text for stem in _AR_CLEAR)


def looks_agricultural(question: str) -> bool:
    """The word-list verdict: farming words or glossary crops/practices in the question."""
    if find_terms(question, "en") or find_terms(question, "ar"):
        return True
    if _EN_WORDS.search(question):
        return True
    text = f" {normalize_for_match(to_msa(question))} "
    return any(stem in text for stem in _AR_STEMS)


def parse_verdict(reply: str) -> bool | None:
    """'yes' / 'no' (or نعم / لا) at the start of the LLM's reply; None when it is neither."""
    word = re.sub(r"[^a-zA-Z؀-ۿ]+", " ", reply or "").strip().lower().split(" ")[0] if reply else ""
    if word in ("yes", "y", "نعم", "اجل", "أجل", "ايوه", "أيوه"):
        return True
    if word in ("no", "n", "لا", "كلا"):
        return False
    return None


def is_agricultural(question: str, generator=None) -> bool:
    """Is this about agriculture? Clear farming words decide at once; otherwise the LLM's yes/no when it can be
    asked, else the word list."""
    if clearly_agricultural(question):
        logger.info("Scope (farming words): %s -> agriculture", question[:80])
        return True
    ask = getattr(generator, "classify_scope", None)
    if ask is not None:
        try:
            verdict = parse_verdict(ask(SCOPE_PROMPT.format(question=question.strip()[:600])))
        except Exception as exc:  # noqa: BLE001 - Ollama down: the word list decides
            logger.warning("Scope check by the LLM failed (%s); using the word list", exc)
            verdict = None
        if verdict is not None:
            logger.info("Scope (LLM): %s -> %s", question[:80], "agriculture" if verdict else "not agriculture")
            return verdict
    verdict = looks_agricultural(question)
    logger.info("Scope (word list): %s -> %s", question[:80], "agriculture" if verdict else "not agriculture")
    return verdict
