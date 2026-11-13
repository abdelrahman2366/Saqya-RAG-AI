"""Fixed English <-> Arabic farming terms used to make query translation reliable.

A small local LLM translates general wording well but often gets crop names wrong
("sugar beet" -> "الكيوي"). Every term found in a question is (1) given to the LLM as the
term to use, and (2) added to the translated search query if the LLM still left it out.
Arabic terms are the ones used in Egyptian extension bulletins. Extend the list freely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.utils.text import normalize_for_match


@dataclass(frozen=True)
class Term:
    en: str        # canonical English term (added to Arabic -> English searches)
    en_re: str     # regex for English variants, matched as whole words, case-insensitive
    ar: str        # Arabic term as used in the bulletins
    ar_alt: tuple[str, ...] = ()  # other Arabic spellings to recognise in questions
    kind: str = "term"            # "crop" for crops (used to find what a document is about)
    ar_re: str = ""               # optional Arabic phrasing pattern (on normalised text), like en_re


def crop(en: str, en_re: str, ar: str, ar_alt: tuple[str, ...] = ()) -> Term:
    return Term(en, en_re, ar, ar_alt, kind="crop")


GLOSSARY: list[Term] = [
    # crops
    crop("wheat", r"wheat", "القمح"),
    crop("maize", r"maize|corn", "الذرة الشامية", ("الذرة",)),
    crop("sorghum", r"sorghum", "الذرة الرفيعة"),
    crop("cotton", r"cotton", "القطن"),
    crop("rice", r"rice|paddy", "الأرز", ("الارز", "الرز")),
    crop("sugar beet", r"sugar ?beets?", "بنجر السكر", ("البنجر",)),
    crop("sugarcane", r"sugar ?canes?", "قصب السكر", ("القصب",)),
    crop("faba bean", r"(?:faba|fava|broad) beans?", "الفول البلدي", ("الفول",)),
    crop("soybean", r"soy ?beans?|soya", "فول الصويا"),
    crop("sesame", r"sesame", "السمسم"),
    crop("sunflower", r"sunflowers?", "دوار الشمس", ("عباد الشمس", "زهرة الشمس")),
    crop("flax", r"flax|linseed", "الكتان"),
    crop("clover", r"clover|berseem", "البرسيم"),
    crop("potato", r"potato(?:es)?", "البطاطس"),
    crop("tomato", r"tomato(?:es)?", "الطماطم"),
    crop("onion", r"onions?", "البصل"),
    crop("pepper", r"peppers?|chill(?:i|ies)|capsicum", "الفلفل"),
    crop("eggplant", r"eggplants?|aubergines?|brinjals?", "الباذنجان"),
    crop("cucumber", r"cucumbers?", "الخيار"),
    crop("zucchini", r"zucchinis?|courgettes?|squash", "الكوسة"),
    crop("cabbage", r"cabbages?", "الكرنب"),
    crop("watermelon", r"watermelons?", "البطيخ"),
    crop("cantaloupe", r"cantaloupes?|musk ?melons?", "الكانتلوب"),
    crop("strawberry", r"strawberr(?:y|ies)", "الفراولة"),
    crop("citrus", r"citrus|oranges?|lemons?|mandarins?", "الموالح"),
    crop("mango", r"mangos|mangoes|mango", "المانجو"),
    crop("olive", r"olives?", "الزيتون"),
    crop("date palm", r"date palms?|palm trees?|(?<!planting )(?<!sowing )(?<!harvest )(?<!harvesting )dates",
         "نخيل البلح", ("النخيل", "البلح", "التمر")),
    crop("grape", r"grapes?|grapevines?|vineyards?", "العنب"),
    # practices
    Term("seed rate", r"seed(?:ing)? rates?|how much seed|kg of seed", "معدل التقاوي", ("التقاوي",)),
    Term("planting date", r"planting (?:date|time|season)|sowing (?:date|time)|when\b.*\b(?:plant(?:ed)?|sow(?:n)?)",
         "ميعاد الزراعة", ("موعد الزراعة",),
         ar_re=r"(?<!\w)(?:متي|امتي|وقت|موعد|ميعاد)(?!\w).*(?<!\w)(?:[وفبل]?ال)?(?:زراعه|يزرع|تزرع|ازرع|نزرع|زرع|لزراعه)(?!\w)"),
    Term("fertilizer", r"fertili[sz](?:er|ers|ation|ing)", "التسميد", ("سماد", "الأسمدة")),
    Term("irrigation", r"irrigat(?:e|ion|ing)|watering", "الري"),
    Term("nursery", r"nurser(?:y|ies)", "المشتل", ("مشاتل",)),
    Term("transplants", r"transplants?|seedlings?", "الشتلات"),
    Term("harvest", r"harvest(?:ing)?", "الحصاد"),
    Term("storage", r"storage|stor(?:e|ing) (?:the )?(?:grain|crop|harvest)", "التخزين"),
    Term("pruning", r"prun(?:e|ing)", "التقليم"),
    Term("pollination", r"pollinat(?:e|ion)", "التلقيح"),
    Term("alternate bearing", r"alternate bearing|biennial bearing", "المعاومة"),
    Term("soil testing", r"soil (?:test(?:ing)?|analysis|sample)", "تحليل التربة"),
    Term("organic manure", r"organic (?:manure|fertili[sz]er)|compost|farmyard manure", "السماد العضوي"),
    # pests, diseases, inputs
    Term("pests", r"pests?|insects?", "الآفات"),
    Term("aphids", r"aphids?", "المن"),
    Term("whitefly", r"white ?fl(?:y|ies)", "الذبابة البيضاء"),
    Term("Tuta absoluta", r"tuta(?: absoluta)?|tomato leaf ?miner", "التوتا أبسلوتا", ("التوتا",)),
    Term("late blight", r"late blight", "اللفحة المتأخرة"),
    Term("rust", r"rusts?", "الصدأ"),
    Term("weeds", r"weeds?", "الحشائش"),
    Term("pesticides", r"pesticides?|insecticides?|fungicides?", "المبيدات"),
    Term("resistance", r"resistance", "المقاومة"),
    Term("feddan", r"feddans?", "فدان"),
    # varieties, inputs and methods that bulletins name in a fixed way
    Term("single-cross hybrids", r"single[- ]cross(?: hybrids?)?", "الهجن الفردية", ("هجن فردية",)),
    Term("three-way hybrids", r"three[- ]way(?: cross)?(?: hybrids?)?", "الهجن الثلاثية", ("هجن ثلاثية",)),
    Term("hybrids", r"hybrids?", "الهجن", ("هجين",)),
    Term("seed drill", r"seed ?drills?|drill(?:ed|ing)? sowing", "السطارة", ("التسطير", "الزراعة الآلية")),
    Term("bread wheat", r"bread wheat", "قمح الخبز"),
    Term("durum wheat", r"durum wheat|pasta wheat", "قمح المكرونة"),
    Term("export", r"exports?|exporting", "التصدير"),
    Term("oil extraction", r"oil (?:extraction|production)|for oil", "استخراج الزيت", ("الزيت",)),
    Term("ammonium nitrate", r"ammonium nitrate", "نترات النشادر"),
    Term("ammonium sulfate", r"ammonium sulph?ate", "سلفات النشادر"),
    Term("potassium sulfate", r"potassium sulph?ate", "سلفات البوتاسيوم"),
    Term("superphosphate", r"(?:calcium )?superphosphate", "سوبر فوسفات", ("السوبر فوسفات",)),
    Term("urea", r"urea", "اليوريا"),
    Term("drip irrigation", r"drip irrigation|trickle irrigation", "الري بالتنقيط"),
    Term("salinity", r"salinity|saline|salt(?:y)? soils?", "الملوحة"),
    Term("seed treatment", r"seed (?:treatment|dressing)", "معاملة البذور", ("معاملة التقاوي",)),
    Term("conservation agriculture", r"conservation agriculture", "الزراعة الحافظة"),
    Term("land degradation", r"land degradation|degraded land", "تدهور الأراضي"),
    Term("personal protective equipment", r"personal protective equipment|\bppe\b|protective (?:clothing|equipment)",
         "معدات الوقاية الشخصية", ("الملابس الواقية",)),
    Term("thinning", r"thinning", "الخف"),
    Term("hoeing", r"hoeing|weeding", "العزيق"),
]

_COMPILED = [(t, re.compile(rf"\b(?:{t.en_re})\b", re.IGNORECASE)) for t in GLOSSARY]


def _ar_pattern(term: str) -> re.Pattern:
    """Whole-word Arabic pattern that allows the usual prefixes (و ف ب ك ل and the article ال).

    A short word written with the article must keep it: 'المن' (aphids) would otherwise match the
    preposition 'من', and 'الفول' (faba bean) would match the 'فول' of 'فول الصويا' (soybean).
    Words written without the article ('فول', 'فدان') accept it as an optional prefix.
    """
    article = "(?:[وفبك]?ال|لل)"
    parts = []
    for word in normalize_for_match(term).split():
        has_article = word.startswith("ال")
        core = word[2:] if has_article else word
        prefix = article if has_article and len(core) < 4 else f"(?:{article}|[وفبكل])?"
        parts.append(prefix + re.escape(core))
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)")


_AR_COMPILED = [(t, [_ar_pattern(a) for a in (t.ar, *t.ar_alt)]) for t in GLOSSARY]
# phrasings that name a term without its words: "متى يزرع ..." / "في أي وقت ... زراعة" = planting date
_AR_PHRASINGS = [(t, re.compile(t.ar_re)) for t in GLOSSARY if t.ar_re]


def _match_form(text: str) -> str:
    """Normalised Arabic with the article ال removed from each word, to check a term is present."""
    return re.sub(r"(?<!\w)ال", "", normalize_for_match(text))


def find_terms(question: str, source_lang: str) -> list[Term]:
    """Glossary terms mentioned in a question written in source_lang ('en' or 'ar')."""
    if source_lang == "en":
        return [t for t, rx in _COMPILED if rx.search(question)]
    q = normalize_for_match(question)
    # Longest names first, each match blanked out, so 'الذرة الرفيعة' (sorghum) is not also counted as
    # 'الذرة' (maize); the result keeps the glossary order.
    order = {id(t): i for i, (t, _) in enumerate(_AR_COMPILED)}
    rest, found = q, []
    for _, t, p in sorted(((len(p.pattern), t, p) for t, patterns in _AR_COMPILED for p in patterns),
                          key=lambda x: -x[0]):
        if p.search(rest):
            if t not in found:
                found.append(t)
            rest = p.sub(" ", rest)
    found.sort(key=lambda t: order[id(t)])
    found += [t for t, rx in _AR_PHRASINGS if t not in found and rx.search(q)]
    return found


def remove_terms(text: str, terms: list[Term], language: str) -> str:
    """The text with these terms' names taken out (to search for one item of a comparison)."""
    if language == "en":
        for t in terms:
            text = re.sub(rf"\b(?:{t.en_re})\b", " ", text, flags=re.IGNORECASE)
    else:
        text = normalize_for_match(text)
        for t, patterns in _AR_COMPILED:
            if t in terms:
                for p in patterns:
                    text = p.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def crops_in(text: str, language: str | None = None) -> list[Term]:
    """The crops a question or passage names (in either language when language is None)."""
    languages = [language] if language else ["ar", "en"]
    out: list[Term] = []
    for lang in languages:
        out += [t for t in find_terms(text, lang) if t.kind == "crop" and t not in out]
    return out


def synonyms(question: str, language: str) -> list[str]:
    """Other names of the glossary terms in a question, in the same language, for keyword search.

    A farmer may type 'عباد الشمس' where the bulletin (and its topic label) says 'دوار الشمس'. For a
    crop only its main name is added: that is the name in every topic label, and extra names
    ('التمر' for 'النخيل') would favour passages that merely repeat crop names.
    """
    q = normalize_for_match(question)
    extra = []
    for term in find_terms(question, language):
        if language == "ar":
            names = [term.ar] if term.kind == "crop" else [term.ar, *term.ar_alt]
        else:
            names = [term.en]
        extra += [n for n in names if normalize_for_match(n) not in q and n not in extra]
    return extra


def glossary_hint(terms: list[Term], target: str) -> str:
    if not terms:
        return ""
    pairs = "; ".join(f"{t.en} = {t.ar}" for t in terms)
    return f"Use these exact terms: {pairs}.\n"


def add_missing_terms(translation: str, terms: list[Term], target: str) -> str:
    """Append any glossary term the translation left out (in the target language)."""
    missing = []
    for t in terms:
        if target == "ar":
            if _match_form(t.ar) not in _match_form(translation):
                missing.append(t.ar)
        elif t.en.lower() not in translation.lower():
            missing.append(t.en)
    return " ".join([translation, *missing]).strip()


def count_terms(text: str, kind: str | None = None) -> dict[Term, int]:
    """How often each glossary term (optionally only one kind) occurs in a text, both languages.

    Longer names are counted first and blanked out, so 'الذرة الرفيعة' (sorghum) is not also
    counted as 'الذرة' (maize).
    """
    norm = normalize_for_match(text)
    jobs = []
    for (term, en_rx), (_, ar_rxs) in zip(_COMPILED, _AR_COMPILED):
        if kind is None or term.kind == kind:
            jobs += [(term, rx, False) for rx in ar_rxs] + [(term, en_rx, True)]
    jobs.sort(key=lambda job: len(job[1].pattern), reverse=True)
    counts: dict[Term, int] = {}
    for term, rx, english in jobs:
        target = text if english else norm
        n = len(rx.findall(target))
        if n:
            counts[term] = counts.get(term, 0) + n
            if english:
                text = rx.sub(" ", text)
            else:
                norm = rx.sub(" ", norm)
    return counts


def document_topic(texts: list[str], file_name: str = "", min_coverage: float = 0.25) -> Term | None:
    """The crop a document is about, or None for general documents (reports, manuals, handbooks).

    A crop bulletin names its crop in its file name or keeps mentioning it: the crop must appear in at
    least min_coverage of the document's chunks, and more often than all other crops together.
    """
    named = count_terms(re.sub(r"[_\-]+", " ", file_name), kind="crop")
    if len(named) == 1:
        return next(iter(named))
    per_chunk = [count_terms(t, kind="crop") for t in texts]
    totals: dict[Term, int] = {}
    for counts in per_chunk:
        for term, n in counts.items():
            totals[term] = totals.get(term, 0) + n
    if not totals:
        return None
    best = max(totals, key=totals.get)
    coverage = sum(best in counts for counts in per_chunk) / max(len(texts), 1)
    if coverage >= min_coverage and totals[best] > sum(totals.values()) - totals[best]:
        return best
    return None
