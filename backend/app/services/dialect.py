"""Egyptian colloquial Arabic -> the Modern Standard Arabic of the bulletins, for searching and prompting.

A farmer who speaks to the assistant says "إمتى أزرع الأوطة؟" or "أعمل إيه للدود اللي في الدرة؟". The
bulletins are written in standard Arabic ("متى تزرع الطماطم", "مكافحة الديدان في الذرة"), so neither the
keyword search nor the crop detection would find a match, and a small LLM reads standard Arabic better.

`to_msa()` rewrites the question word by word with a fixed list (question words, common verbs, crop names
farmers use in Egypt, and a few farming words). It only replaces whole words, keeps everything it doesn't
know, and leaves standard Arabic and English untouched, so it is safe to run on every Arabic question.
The farmer still sees their own words; the rewritten question is what is searched and answered.
Extend the lists freely.

`to_egyptian()` goes the other way for the spoken answer, when the farmer picks the Egyptian style in
Settings: the standard-Arabic answer is read aloud with Egyptian everyday words (كيف -> إزاي، هذا -> ده،
الذي -> اللي، يجب أن -> لازم، سوف تزرع -> هتزرع، لا يوجد -> مفيش...). Farming terms, numbers and units are
left exactly as written, so the spoken answer says the same thing as the written one.
"""

from __future__ import annotations

import re

_L = "ء-ي"  # Arabic letters (after the question has been typed or transcribed, no presentation forms)


def _rx(words: str) -> re.Pattern:
    """Whole-word pattern for '|'-separated spellings, after optional prefixes و ب ف ك ('and', 'with'...)."""
    return re.compile(rf"(?<![{_L}])(و?[بفك]?)(?:{words})(?![{_L}])")


# Order matters: phrases first, then single words.
_RULES: list[tuple[re.Pattern, str]] = [
    # ---- phrases
    (_rx("أعمل إيه|اعمل ايه|أعمل ايه|اعمل إيه|نعمل إيه|نعمل ايه|اعمل اية|أعمل اية"), r"\1ماذا أفعل"),
    (_rx("إيه هو|ايه هو|إيه هي|ايه هي"), r"\1ما هو"),
    (_rx("إيه اللي|ايه اللي|إيه الي|ايه الي"), r"\1ما الذي"),
    (_rx("إيه أحسن|ايه احسن|إيه أفضل|ايه افضل|ايه أحسن"), r"\1ما أفضل"),
    (_rx("قد إيه|قد ايه|أد إيه|اد ايه|أد ايه"), r"\1كم"),
    (_rx("عن إيه|عن ايه"), r"\1عن ماذا"),
    (_rx("ليه كده|ليه كدا"), r"\1لماذا"),
    (_rx("مش عارف|مش عارفة|مش عارفين"), r"\1لا أعرف"),
    (_rx("يعني إيه|يعني ايه"), r"\1ما معنى"),
    # ---- question words
    (_rx("إزاي|ازاي|إزّاي|ازّاي|إزاى|ازاى"), r"\1كيف"),
    (_rx("إمتى|امتى|إمتي|امتي"), r"\1متى"),
    (_rx("إيه|ايه|إيش|ايش"), r"\1ما"),
    (_rx("فين"), r"\1أين"),
    (_rx("منين"), r"\1من أين"),
    (_rx("ليه|ليش"), r"\1لماذا"),
    (_rx("كام"), r"\1كم"),
    (_rx("مين"), r"\1من"),
    (_rx("لحد"), r"\1حتى"),
    # ---- common words
    (_rx("عايز|عاوز|عايزة|عاوزة|عايزين|عاوزين|عايزه|عاوزه"), r"\1أريد"),
    (_rx("محتاج|محتاجة|محتاجين|محتاجه"), r"\1أحتاج"),
    (_rx("دلوقتي|دلوقت|دلوقتى"), r"\1الآن"),
    (_rx("كده|كدا|كدة"), r"\1هكذا"),
    (_rx("برضه|برضو|برده"), r"\1أيضا"),
    (_rx("عشان|علشان|عشانها|علشانها"), r"\1من أجل"),
    (_rx("بتاع|بتاعة|بتاعت|بتوع|بتاعه|بتاعها"), r"\1"),
    (_rx("اللي|إللي"), r"\1الذي"),
    (_rx("ده|دا"), r"\1هذا"),
    (_rx("دي|دى"), r"\1هذه"),
    (_rx("دول"), r"\1هؤلاء"),
    (_rx("مش|مُش"), r"\1لا"),
    (_rx("أوي|اوي|خالص"), r"\1جدا"),
    (_rx("اتنين|إتنين"), r"\1اثنين"),
    (_rx("تلاتة|تلاته|تلات"), r"\1ثلاثة"),
    (_rx("تقيل|تقيلة|تقيله"), r"\1ثقيل"),
    (_rx("الكويسة|الكويسه|الكويس"), r"\1الجيدة"),
    (_rx("كويسة|كويسه|كويس"), r"\1جيدة"),
    (_rx("جنب"), r"\1بجوار"),
    (_rx("راس"), r"\1رأس"),
    (_rx("بياخد|ياخد|بتاخد|تاخد"), r"\1يحتاج"),
    (_rx("بيعمل"), r"\1يسبب"),
    (_rx("بتعمل"), r"\1تسبب"),
    # ---- crops: the names farmers use in Egypt
    (_rx("الأوطة|الاوطة|الأوطه|الاوطه|القوطة|القوطه|أوطة|اوطة|قوطة"), r"\1الطماطم"),
    (_rx("الدرة|الدره"), r"\1الذرة"),
    (_rx("درة|دره"), r"\1ذرة"),
    (_rx("البتنجان|البدنجان|بتنجان|بدنجان"), r"\1الباذنجان"),
    (_rx("الكوسا|كوسا"), r"\1الكوسة"),
    (_rx("الشمام|شمام"), r"\1الكانتلوب"),
    (_rx("المانجا|مانجا|المنجة|المنجا|المانجة|المانجه|مانجة|مانجه"), r"\1المانجو"),
    (_rx("البرتقان|البرتقال|اليوسفي|اليوسفى|اليوسف أفندي|اليوسف افندي"), r"\1الموالح"),
    (_rx("النخل"), r"\1النخيل"),
    (_rx("الرز"), r"\1الأرز"),
    (_rx("عباد الشمس|عبّاد الشمس"), r"\1دوار الشمس"),
    (_rx("البطيخة|البطيخه"), r"\1البطيخ"),
    (_rx("الخضار"), r"\1الخضر"),
    (_rx("خضار"), r"\1خضر"),
    (_rx("الشجر"), r"\1الأشجار"),
    (_rx("شجر"), r"\1أشجار"),
    # ---- farming words
    (_rx("الغيط"), r"\1الحقل"),
    (_rx("غيط"), r"\1حقل"),
    (_rx("المية|الميه|المياة"), r"\1مياه الري"),
    (_rx("السباخ البلدي|السباخ|سباخ"), r"\1السماد البلدي"),
    (_rx("الكيماوي|الكيماوى"), r"\1السماد الكيماوي"),
    (_rx("الدود|دود"), r"\1الديدان"),
    (_rx("الحشيش"), r"\1الحشائش"),
    (_rx("الورق"), r"\1الأوراق"),
    (_rx("الزرعة|الزرعه"), r"\1النباتات"),
    (_rx("دبلان|دبلانة|بيدبل|بتدبل|بيدبلوا"), r"\1ذابل"),
    (_rx("بيصفر|بتصفر|بيصفّر|بتصفّر|مصفر|مصفرة|مصفّر"), r"\1اصفرار"),
    (_rx("بيعفن|بتعفن|معفن|معفنة"), r"\1متعفن"),
    (_rx("استوى|استوت|استوي|مستوي|مستوية|مستويه"), r"\1ناضج"),
    (_rx("بيستوي|بتستوي|يستوي|تستوي|بيستوى|بتستوى"), r"\1ينضج"),
    (_rx("بيفرقع|بتفرقع|بيتفرقع|بتتفرقع|مفرقع|مفرقعة"), r"\1يتشقق"),
    (_rx("الفريش|فريش"), r"\1الطازجة"),
    (_rx("أسقي|اسقي|نسقي"), r"\1أروي"),
    (_rx("الشكارة|الشيكارة|شكارة|شيكارة|شكاير|شكاكير"), r"\1شيكارة"),
]

_ARABIC_RE = re.compile(f"[{_L}]")
_SPACES_RE = re.compile(r"[ \t]{2,}")
# words that only occur in Egyptian colloquial (not in standard Arabic): used to report the dialect
_EGYPTIAN_MARKERS = _rx("إزاي|ازاي|إمتى|امتى|امتي|إيه|ايه|فين|كام|ليه|عايز|عاوز|دلوقتي|عشان|علشان|بتاع|اللي|مش|كده|"
                        "كويس|كويسة|ياخد|بياخد|لحد|استوى|بيستوي|بيفرقع|بتفرقع|"
                        "الأوطة|القوطة|الدرة|البتنجان|الكوسا|المانجا|المانجة|الغيط|السباخ|عباد الشمس")


def _with_lil(rules: list[tuple[re.Pattern, str]]) -> list[tuple[re.Pattern, str]]:
    """Add the 'لل' form of every word written with the article: ل + الدود is written للدود."""
    out = []
    for pattern, replacement in rules:
        out.append((pattern, replacement))
        alts = re.search(r"\(\?:(.*)\)\(\?!", pattern.pattern).group(1).split("|")
        target = replacement.replace("\\1", "")
        with_article = [a[2:] for a in alts if a.startswith("ال") and len(a) > 3]
        if with_article and target.startswith("ال"):
            cores = "|".join(with_article)
            out.append((re.compile(rf"(?<![{_L}])(و?)لل(?:{cores})(?![{_L}])"), r"\1لل" + target[2:]))
    return out


_RULES = _with_lil(_RULES)


def to_msa(text: str) -> str:
    """The question rewritten in standard Arabic (unchanged when there is nothing to rewrite)."""
    if not _ARABIC_RE.search(text):
        return text
    out = text
    for pattern, replacement in _RULES:
        out = pattern.sub(replacement, out)
    out = _SPACES_RE.sub(" ", out).strip()
    return out


def is_egyptian(text: str) -> bool:
    """Does the question use Egyptian colloquial words?"""
    return bool(_EGYPTIAN_MARKERS.search(text))


# ---------------------------------------------------------------- standard Arabic -> Egyptian (for speaking)
def _sx(words: str) -> re.Pattern:
    """Whole-word pattern for the spoken rewrite: optional و / ف before the word."""
    return re.compile(rf"(?<![{_L}])([وف]?)(?:{words})(?![{_L}])")


# Order matters: phrases first, then single words. Only words whose Egyptian form means exactly the same.
_SPOKEN_RULES: list[tuple[re.Pattern, str]] = [
    # ---- phrases
    (_sx("لا يوجد|لا توجد|ليس هناك|لا يوجد هناك"), r"\1مفيش"),
    (_sx("يجب عليك أن|يجب عليك ان|يجب أن|يجب ان|ينبغي أن|ينبغي ان|عليك أن|عليك ان|من الضروري أن|من الضروري ان"), r"\1لازم"),
    (_sx("يمكنك أن|يمكنك ان|يمكن أن|يمكن ان|من الممكن أن|من الممكن ان"), r"\1ممكن"),
    (_sx("بعد ذلك|بعد هذا"), r"\1بعد كده"),
    (_sx("قبل ذلك|قبل هذا"), r"\1قبل كده"),
    (_sx("لذلك|لذا|ولذلك"), r"عشان كده"),
    (_sx("من أجل|من اجل|لكي|كي"), r"\1عشان"),
    (_sx("ما هو|ما هي"), r"\1إيه هو"),
    (re.compile(rf"(?<![{_L}])سوف\s+([يتنأ][{_L}]+)"), r"ه\1"),
    # ---- single words
    (_sx("ولكن|لكن|لكنّ"), r"بس"),
    (_sx("يوجد|توجد"), r"\1فيه"),
    (_sx("ليس|ليست|لست|ليسوا"), r"\1مش"),
    (_sx("ماذا"), r"\1إيه"),
    (_sx("لماذا"), r"\1ليه"),
    (_sx("كيف"), r"\1إزاي"),
    (_sx("متى"), r"\1إمتى"),
    (_sx("أين|اين"), r"\1فين"),
    (_sx("هذا|ذلك"), r"\1ده"),
    (_sx("هذه|تلك"), r"\1دي"),
    (_sx("هؤلاء|أولئك"), r"\1دول"),
    (_sx("الذي|التي|الذين|اللذان|اللتان|اللاتي|اللواتي"), r"\1اللي"),
    (_sx("يجب|ينبغي|يلزم"), r"\1لازم"),
    (_sx("يمكن|يمكنك"), r"\1ممكن"),
    (_sx("الآن|الان"), r"\1دلوقتي"),
    (_sx("أيضا|أيضاً|ايضا|ايضاً|كذلك"), r"\1كمان"),
    (_sx("فقط"), r"\1بس"),
    (_sx("عندما|حينما|حين"), r"\1لما"),
    (_sx("إذا|اذا"), r"\1لو"),
    (_sx("مثل"), r"\1زي"),
    (_sx("كثيرا|كثيراً"), r"\1كتير"),
    (_sx("كثيرة"), r"\1كتيرة"),
    (_sx("كثير"), r"\1كتير"),
    (_sx("جدا|جداً"), r"\1أوي"),
    (_sx("لأن|لان|لأنّ"), r"\1عشان"),
    (_sx("ثم"), r"\1وبعدين"),
    (_sx("نعم"), r"\1أيوه"),
    (_sx("أريد|اريد"), r"\1عايز"),
    (_sx("نريد"), r"\1عايزين"),
    (_sx("هكذا"), r"\1كده"),
    (_sx("حسنا|حسناً"), r"\1تمام"),
]


def to_egyptian(text: str) -> str:
    """A standard-Arabic answer with Egyptian everyday words, for reading aloud in the Egyptian style.
    Farming terms, numbers and units are not touched; English text comes back unchanged."""
    if not _ARABIC_RE.search(text):
        return text
    out = text
    for pattern, replacement in _SPOKEN_RULES:
        out = pattern.sub(replacement, out)
    return _SPACES_RE.sub(" ", out).strip()
