"""What a farmer's question is about, as structured data: crop, kind of problem, intent, conditions, quantities.

No model is involved: everything is read from the words of the question with the project's own glossary
(glossary.py), dialect markers (dialect.py) and land parser (crop_calendar.py), plus the word lists below. That
keeps it fast (well under a millisecond), explainable and testable, and it means nothing here can "assume" a
diagnosis: a disease or pest is reported only when the farmer named it, and a crop only when it is in the text.

The domain categories are multi-label ("how do I fertilise and irrigate wheat?" is fertilisation + irrigation).
They are rules, not a trained classifier: the project has no labelled questions for these categories, and
labels are not invented to train one (see docs/ai-ml-engineering-report.md). The crop and crop-group outputs
are measured against the evaluation questions by scripts/evaluate.py.

The result is used by query_expansion.py (which words to add to the search), confidence.py (is anything
missing that the answer needs?) and returned by the API when QUERY_CLASSIFICATION_ENABLED=true.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.services.crop_calendar import land_in, month_in
from app.services.dialect import is_egyptian, to_msa
from app.services.glossary import find_terms
from app.utils.text import detect_language, normalize_for_match, to_ascii_digits

# crop -> group, for the crops the library has bulletins on (glossary.py's canonical English names)
CROP_GROUPS = {
    "field crops": {"wheat", "maize", "sorghum", "cotton", "rice", "sugar beet", "sugarcane", "faba bean", "soybean",
                    "sesame", "sunflower", "flax", "clover"},
    "vegetables": {"potato", "tomato", "onion", "pepper", "eggplant", "cucumber", "zucchini", "cabbage", "watermelon",
                   "cantaloupe", "strawberry"},
    "fruit trees": {"citrus", "mango", "olive", "date palm", "grape"},
}
GROUP_OF = {crop: group for group, crops in CROP_GROUPS.items() for crop in crops}

# glossary terms that name a specific pest or disease (reported only when the farmer wrote them)
_PEST_TERMS = {"aphids", "whitefly", "Tuta absoluta"}
_DISEASE_TERMS = {"late blight", "rust"}


_AR = "\u0621-\u064a"
_PROCLITICS = "(?:[وف]?(?:[بكل]?ال|لل|[بكل]))?"   # و ف ب ك ل and the article, as written attached to a word


def _ar(pattern: str) -> str:
    """An Arabic pattern in the form normalize_for_match() gives the question (أ -> ا, ة -> ه, ى -> ي)."""
    return normalize_for_match(pattern)


def _stems(en: str, ar: str) -> re.Pattern:
    """English whole words, or Arabic words that *start* with one of the stems (after و ف ب ك ل ال), so
    "تسميد", "والسماد" and "بالأسمدة" all match their stem, and "ري" does not match inside "يوريا"."""
    return re.compile(rf"(?i)\b(?:{en})\b|(?<![{_AR}]){_PROCLITICS}(?:{_ar(ar)})")


def _words(en: str, ar: str) -> re.Pattern:
    """English whole words, or whole Arabic words/phrases (an attached و or ف allowed)."""
    return re.compile(rf"(?i)\b(?:{en})\b|(?<![{_AR}])[وف]?(?:{_ar(ar)})(?![{_AR}])")


_END = f"(?![{_AR}])"   # for short stems that must be the whole word ("ري" but not "ريف")

# category -> words that signal it, English | Arabic stems (standard and Egyptian), matched on normalised text
CATEGORIES: dict[str, re.Pattern] = {
    "planting": _stems(
        r"plant(?:ing|ed)?|sow(?:ing|n)?|seed ?rates?|seeds?|seedlings?|transplant\w*|spacing|nurser\w+|germinat\w+|"
        r"cultivat\w+",
        r"زراع|ازرع|نزرع|تزرع|يزرع|زرع|تقاوي|بذر|بذور|شتل|مشتل|مشاتل|مسافات|انبات|ميعاد|موعد"),
    "varieties": _stems(r"variet(?:y|ies)|cultivars?|hybrids?|breeds?", r"صنف|اصناف|هجين|هجن|سلال"),
    "land preparation": _stems(r"plough\w*|plow\w*|till(?:age|ing)?|land prep\w*|levell?ing|bed prep\w*|ridg\w+",
                               r"حرث|تجهيز الارض|خدمه الارض|تسويه|تخطيط|مصاطب|عزيق"),
    "irrigation": _stems(r"irrigat\w+|watering|water requirement\w*|drip|sprinkler|drought|water stress",
                         rf"ري{_END}|ريه{_END}|ريات|ارو|اسقي|سقي|بسقي|تنقيط|رش محوري|بيفوت|رشاش|عطش|مياه|ميه{_END}|مقنن"),
    "fertilization": _stems(
        r"fertili[sz]\w+|manure|compost|nitrogen|phosph\w+|potassium|potash|urea|npk|nutrient\w*|deficien\w+|zinc|iron",
        r"سماد|سميد|تسميد|اسمد|ازوت|نيتروجين|فوسف|بوتاس|يوريا|نترات|سلفات|عناصر|كمبوست|سباخ|نقص|ناقص|زنك|حديد"),
    "pests": _stems(
        r"pests?|insects?|aphids?|white ?fl(?:y|ies)|worms?|borers?|mites?|nematod\w+|thrips|leaf ?miners?|tuta|"
        r"locusts?|rodents?|rats?",
        rf"افه{_END}|افات|حشر|دود|ذباب|عنكبوت|نيماتودا|تربس|توتا|سوس|فار{_END}|فيران|جراد|حفار"),
    "diseases": _stems(
        r"diseases?|blights?|rusts?|mildew|rot|wilt\w*|fung\w+|virus\w*|bacteri\w+|infect\w+|symptoms?|spots?|"
        r"yellowing|lesions?",
        rf"مرض|امراض|لفح|صدا{_END}|بياض|عفن|تعفن|ذبول|فطر|فيروس|بكتير|اصاب|اعراض|تبقع|بقع|اصفرار|يصفر|تصفر|تفحم"),
    "weeds": _stems(r"weeds?|herbicides?|weeding", r"حشايش|حشيش"),
    "pesticide use": _stems(r"pesticides?|insecticides?|fungicides?|spray\w*|dos(?:e|age)|pre-?harvest interval|ppe|"
                            r"protective (?:equipment|clothing)",
                            rf"مبيد|رش{_END}|ارش|جرعه|فتره الامان|مكافح|اكافح|اقاوم|وقايه"),
    "harvest & storage": _stems(
        r"harvest\w*|pick\w*|post-?harvest|stor(?:e|age|ing)|curing|drying|grading|pack\w+|shelf life|yields?|"
        r"ripe\w*|matur\w+",
        r"حصاد|حصد|احصد|جمع|جني|تقليع|تخزين|اخزن|تجفيف|تعبيه|فرز|تداول|انتاجيه|نضج|ناضج|استو|يستوي"),
    "soil": _stems(r"soils?|salin\w+|alkalin\w+|ph|drainage|soil test\w*|organic matter|arable land",
                   rf"تربه|ارض{_END}|ارضي{_END}|اراضي|ملوح|ملحي|قلوي|صرف{_END}|ماده عضويه"),
    "climate": _stems(r"weather|climate|temperature|frost|heat ?wave|rain\w*|wind\w*|humidity|season\w*",
                      r"طقس|مناخ|حراره|صقيع|مطر|امطار|رياح|رطوبه|موسم|عروه"),
    "protected cultivation": _stems(r"greenhouses?|tunnels?|plastic houses?|hydropon\w+|mulch\w*",
                                    r"صوب|انفاق|بيوت محميه|زراعه محميه|هيدروبون|ملش"),
    "tree care": _stems(r"prun\w+|graft\w+|pollinat\w+|thinning|alternate bearing|flowering|fruit set",
                        rf"تقليم|اقلم|تطعيم|تلقيح|خف{_END}|معاومه|تزهير|عقد{_END}"),
    "quality & disorders": _stems(r"crack\w*|split\w*|hollow\w*|disorders?|quality|oil content|protein|sugar content",
                                  r"تشقق|يتشقق|تتشقق|يفرقع|تفرقع|مجوف|اجوف|ظاهره|جوده|نسبه الزيت|بروتين|نسبه السكر"),
    "sustainability": _stems(r"conservation agriculture|biodiversity|land degradation|sustainab\w+|climate change|"
                             r"organic farming|agroecolog\w+",
                             r"زراعه حافظه|تنوع حيوي|تنوع بيولوجي|تدهور|مستدام|استدام|تغير المناخ|زراعه عضويه"),
    "livestock": _stems(r"livestock|cattle|cows?|buffalo\w*|sheep|goats?|poultry|chickens?|rabbits?|bees?|"
                        r"beekeeping|fish|tilapia|fodder|feed",
                        rf"ماشيه|مواشي|ابقار|بقر|جاموس|اغنام|غنم{_END}|ماعز|دواجن|فراخ|ارانب|نحل{_END}|سمك|اسماك|علف|اعلاف"),
    "economics": _stems(r"prices?|costs?|profit\w*|market\w*|export\w*|subsid\w+|loans?|budget|income",
                        rf"سعر|اسعار|تكلف|ربح{_END}|ارباح|سوق{_END}|تسويق|تصدير|دعم{_END}|قرض|دخل{_END}"),
}

# intent: first match wins, so the more specific patterns come first
INTENTS: list[tuple[str, re.Pattern]] = [
    ("compare", _words(r"compar\w+|versus|vs\.?|differences?|better than|which is better",
                       r"قارن|مقارنه|الفرق|ايهما|افضل من|احسن من")),
    ("diagnose", _words(r"what(?:'s| is) wrong|why (?:are|is|do|does)|symptoms?|turning|dying|what is causing|"
                        r"what causes|identify",
                        r"ما سبب|ما اسباب|ايه سبب|السبب ايه|سبب|اسباب|ليه|لماذا|اعراض|عنده ايه|مالها|ماله|بيموت|بتموت|"
                        r"يموت|تموت|تشخيص")),
    ("recommend", _words(r"recommend\w*|suggest\w*|what should i (?:plant|grow)|which crops?|best crops?|advice|tips",
                         r"اقترح|رشح|انصح|انصحني|نصايح|نصائح|ازرع ايه|ماذا ازرع|انسب محصول|افضل محصول")),
    ("when", _words(r"when|what time|which month|how often|how long|how many days",
                    r"متي|امتي|ميعاد|موعد|انسب ميعاد|كل كام|كل كم|كم يوم|كم يوما|كام يوم|مده")),
    ("how much", _words(r"how much|how many|what (?:rate|dose|amount|quantity)|rates?|dos(?:e|age)",
                        r"كم|كام|كميه|ما كميه|معدل|ما معدل|جرعه|قد ايه|ما نسبه")),
    ("how to", _words(r"how (?:do|can|should|to)|steps|method|way to|control|treat\w*|prevent\w*|manage",
                      r"كيف|ازاي|طريقه|خطوات|اعالج|اعالجه|علاج|مكافحه|اكافح|اقاوم|امنع|وقايه")),
    ("define", _words(r"what (?:is|are)|define|meaning of|explain", r"ما هو|ما هي|ايه هو|ايه هي|يعني ايه|تعريف|اشرح|اشرحلي")),
    ("list", _words(r"list|examples of|types of|kinds of", r"اذكر|عدد|انواع|امثله")),
]

# crops written in a way glossary.py deliberately skips (a three-letter name without its article could be another
# word in a passage; in a farmer's short question it is the crop) and the generic words for a crop group
_EXTRA_CROPS: list[tuple[str, re.Pattern]] = [
    ("wheat", _words(r"(?!)", r"قمح")), ("cotton", _words(r"(?!)", r"قطن")), ("onion", _words(r"(?!)", r"بصل")),
    ("grape", _words(r"(?!)", r"عنب")), ("rice", _words(r"(?!)", r"ارز|رز")),
    ("date palm", _words(r"(?!)", r"النخله|نخله|نخل|النخل")),
]
_GROUP_WORDS = {
    "vegetables": _stems(r"vegetables?", r"خضر|خضار|خضروات"),
    "fruit trees": _stems(r"fruit trees?|orchards?", r"فاكهه|فواكه|اشجار الفاكهه|بستان|بساتين"),
    "field crops": _stems(r"field crops?|cereals?|grains?", r"محاصيل حقليه|حبوب"),
}

# categories whose answer depends on the crop: a question about them that names none is missing something
_NEEDS_CROP = {"planting", "varieties", "irrigation", "fertilization", "pests", "diseases", "weeds", "pesticide use",
               "harvest & storage", "tree care"}

_UNITS = (r"kg|kilograms?|g|grams?|tons?|tonnes?|l|lit(?:er|re)s?|ml|cm|mm|m|meters?|metres?|ha|hectares?|feddans?|"
          r"acres?|days?|weeks?|months?|years?|%|°c|ppm|"
          + _ar("كجم|كيلو|كيلوجرام|جرام|جم|طن|لتر|مل|سم|مم|متر|هكتار|فدان|افدنه|قيراط|يوم|ايام|اسبوع|اسابيع|شهر|شهور|اشهر|"
                "سنه|سنوات|اردب|شيكاره|جوال"))
_QUANTITY_RE = re.compile(rf"(?i)(\d+(?:[.,]\d+)?)\s*(?:[-–]\s*(\d+(?:[.,]\d+)?)\s*)?({_UNITS})(?![a-zء-ي])")

_STAGES = {
    "seedling": _stems(r"seedlings?|germinat\w+|emergence|nursery", r"بادر|شتل|انبات|مشتل"),
    "vegetative": _stems(r"vegetative|tillering|growing", r"خضري|تفريع|نمو"),
    "flowering": _stems(r"flower\w+|bloom\w*|pollinat\w+", r"تزهير|ازهار|زهر|تلقيح"),
    "fruiting": _stems(r"fruit(?:ing| set)|grain fill\w*|boll\w*|ripening", r"عقد|اثمار|ثمار|طرد|نضج|لوز"),
    "harvest": _stems(r"harvest\w*|post-?harvest|storage", r"حصاد|جمع|تقليع|تخزين"),
}
_SYSTEMS = {
    "drip": _stems(r"drip|trickle", r"تنقيط"),
    "sprinkler": _stems(r"sprinkler|pivot", r"رش محوري|رشاش|بالرش"),
    "flood": _stems(r"flood|furrow|surface irrigation", r"غمر|ري سطحي|بالراحه"),
    "greenhouse": _stems(r"greenhouses?|tunnels?", r"صوب|انفاق"),
}


# symptoms a farmer describes -> a neutral name. Reported as observations; several causes (nutrients, water,
# disease, pests) can produce each one, so they never become a diagnosis here.
SYMPTOMS: dict[str, re.Pattern] = {
    "yellowing": _stems(r"yellow\w*|chlorosis|chlorotic", r"اصفر|يصفر|تصفر|بيصفر|بتصفر|صفار|اصفرار"),
    "wilting": _stems(r"wilt\w*|droop\w*", r"ذبول|ذابل|يذبل|بيذبل|بتذبل|دبلان|دبل"),
    "spots": _stems(r"spots?|lesions?|blotch\w*", r"بقع|تبقع|نقط|نقاط"),
    "rot": _stems(r"rot\w*|decay\w*", r"عفن|تعفن|بيعفن|بتعفن|معفن|متعفن"),
    "drying / browning": _stems(r"brown\w*|dry(?:ing)? (?:leaves|out)|scorch\w*|burn\w*", r"جفاف الاوراق|ناشف|بينشف|بتنشف|اسمرار|احتراق|محروق"),
    "leaf curl": _stems(r"curl\w*", r"تجعد|تجعيد|التفاف|ملفوف|مكرمش"),
    "fruit or flower drop": _stems(r"(?:fruit|flower|blossom) drop|dropping", r"تساقط|بيقع|بتقع|سقوط"),
    "cracking": _stems(r"crack\w*|split\w*", r"تشقق|يتشقق|يفرقع|بيفرقع|تفرقع|بتفرقع|شقوق"),
    "stunting": _stems(r"stunt\w*|not growing|slow growth", r"تقزم|قزم|مش بيكبر|ضعف النمو|ضعيف"),
    "dying plants": _stems(r"dying|dead plants?", r"بيموت|بتموت|يموت|تموت|موت النبات"),
}


@dataclass
class QueryUnderstanding:
    language: str                                   # "ar" or "en"
    dialect: str = ""                               # "egyptian" / "standard" for Arabic, "" for English
    crops: list[str] = field(default_factory=list)          # canonical English names, only crops named in the text
    crop_groups: list[str] = field(default_factory=list)    # field crops · vegetables · fruit trees
    categories: list[str] = field(default_factory=list)     # multi-label, strongest first
    category_scores: dict[str, float] = field(default_factory=dict)
    intent: str = "other"
    pests: list[str] = field(default_factory=list)          # named in the question, never inferred
    diseases: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)          # every glossary term found (English names)
    conditions: dict[str, str] = field(default_factory=dict)  # soil, region, place, month, stage, system
    quantities: list[dict] = field(default_factory=list)    # {"value", "to", "unit"} as written by the farmer
    symptoms: list[str] = field(default_factory=list)       # what the farmer sees, as said (not a diagnosis)
    missing: list[str] = field(default_factory=list)        # what an answer cannot do without (asked back if needed)
    would_help: list[str] = field(default_factory=list)     # details that would narrow the answer (never blocking)

    def to_dict(self) -> dict:
        return asdict(self)


def _score(pattern: re.Pattern, text: str) -> int:
    return len({m.group(0) for m in pattern.finditer(text)})


def classify(text: str, threshold: float = 0.34) -> tuple[list[str], dict[str, float]]:
    """Domain categories of a question: (labels strongest first, scores). A category's score is its number of
    distinct matched words over the strongest category's; labels are those at or above `threshold`."""
    counts = {name: _score(pattern, text) for name, pattern in CATEGORIES.items()}
    top = max(counts.values(), default=0)
    if top == 0:
        return [], {}
    scores = {name: round(n / top, 2) for name, n in counts.items() if n}
    labels = [name for name, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s >= threshold]
    return labels, scores


def understand(question: str, history: list[str] | None = None) -> QueryUnderstanding:
    """Structured reading of one question. `history` (the farmer's earlier questions) only fills in the crop of a
    follow-up that names none; it never adds categories or problems."""
    language = detect_language(question)
    # an Egyptian question is read after the app's own dialect rewrite ("الأوطة" -> "الطماطم"), as it is searched
    standard = to_msa(question) if language == "ar" else question
    text = normalize_for_match(to_ascii_digits(standard))
    found = find_terms(standard, language)
    if language == "ar":   # a mixed question ("ازرع tomato امتى؟") names its crop in English
        found += [t for t in find_terms(standard, "en") if t not in found]
    crops = [t.en for t in found if t.kind == "crop"]
    crops += [name for name, pattern in _EXTRA_CROPS if name not in crops and pattern.search(text)]
    inherited = False
    if not crops and history:
        for earlier in reversed(history[-3:]):
            earlier_crops = [t.en for t in find_terms(earlier, detect_language(earlier)) if t.kind == "crop"]
            if earlier_crops:
                crops, inherited = earlier_crops[:1], True
                break
    categories, scores = classify(text)
    intent = next((name for name, pattern in INTENTS if pattern.search(text)), "other")

    conditions: dict[str, str] = {}
    land = land_in(question)
    if land.soil:
        conditions["soil"] = land.soil
    if land.region:
        conditions["region"] = land.region
        conditions["place"] = land.place_en
    month = month_in(question)
    if month:
        conditions["month"] = str(month)
    for key, table in (("stage", _STAGES), ("system", _SYSTEMS)):
        for name, pattern in table.items():
            if pattern.search(text):
                conditions[key] = name
                break
    if inherited:
        conditions["crop_from"] = "earlier question"

    quantities = []
    for m in _QUANTITY_RE.finditer(text):
        item = {"value": m.group(1).replace(",", "."), "unit": m.group(3)}
        if m.group(2):
            item["to"] = m.group(2).replace(",", ".")
        quantities.append(item)

    symptoms = [name for name, pattern in SYMPTOMS.items() if pattern.search(text)]
    if symptoms and intent in ("other", "define"):
        intent = "diagnose"          # "ورق الطماطم بيصفر ومش عارف السبب": the farmer wants possible causes
    missing = []
    if not crops and (set(categories) & _NEEDS_CROP) and "livestock" not in categories:
        missing.append("crop")
    if intent == "diagnose" and not categories and not symptoms:
        missing.append("symptoms")   # "my plants are dying": nothing says what the farmer sees
    would_help = []
    if intent == "diagnose" or symptoms:
        if "stage" not in conditions:
            would_help.append("plant age or growth stage")
        if "irrigation" not in categories and "system" not in conditions:
            would_help.append("irrigation (how often, how much)")
        if symptoms and len(symptoms) < 2:
            would_help.append("where it shows (old or young leaves, fruit, roots) and how it spread")
        if "fertilization" not in categories:
            would_help.append("recent fertilizer or spraying")
    if (intent in ("when", "recommend") or "planting" in categories) and not conditions.get("region") and crops:
        would_help.append("region (Delta or Upper Egypt)")
    groups = list(dict.fromkeys(GROUP_OF[c] for c in crops if c in GROUP_OF))
    groups += [g for g, pattern in _GROUP_WORDS.items() if g not in groups and pattern.search(text)]

    return QueryUnderstanding(
        language=language,
        dialect=("egyptian" if is_egyptian(question) else "standard") if language == "ar" else "",
        crops=crops,
        crop_groups=groups,
        categories=categories,
        category_scores=scores,
        intent=intent,
        pests=[t.en for t in found if t.en in _PEST_TERMS],
        diseases=[t.en for t in found if t.en in _DISEASE_TERMS],
        terms=[t.en for t in found],
        conditions=conditions,
        quantities=quantities,
        symptoms=symptoms,
        missing=missing,
        would_help=would_help,
    )
