"""What can be planted in a given month in Egypt — answered straight from the library's bulletins.

"إيه أفضل محاصيل أزرعها في شهر نوفمبر؟" names no crop, so no single passage answers it and the question used to
fall through to a web search, which returned crops that are not grown in Egypt. But every crop bulletin in the
library states its sowing dates. They are collected here once (each line copied from the bulletin and page it
names), so the question gets a correct, complete, instant answer with the bulletins as sources — no LLM call.

A question is a calendar question when it asks what to plant, names a month (or "now" / "this month") and names
no crop. A question about one crop ("when do I plant wheat?") is answered from that crop's bulletin as usual.
When a bulletin is added to the library, add its sowing dates here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from app.services.glossary import crops_in
from app.services.retrieval import RetrievedChunk
from app.utils.text import normalize_for_match


@dataclass(frozen=True)
class Sowing:
    ar: str
    en: str
    months: tuple[int, ...]   # months in which it is sown / transplanted
    when_ar: str
    when_en: str
    source: str
    page: int


CALENDAR: tuple[Sowing, ...] = (
    Sowing("القمح", "Wheat", (11, 12), "من 15 إلى 30 نوفمبر، وقد تمتد الزراعة إلى 10 ديسمبر",
           "15–30 November; sowing may extend to 10 December", "NEW_wheat_new.pdf", 6),
    Sowing("الفول البلدي", "Faba bean", (10, 11),
           "النصف الثاني من أكتوبر في أسيوط وسوهاج وقنا وأسوان، والنصف الأول من نوفمبر في الوجه البحري",
           "second half of October in Assiut, Sohag, Qena and Aswan; first half of November in the Delta",
           "الفول-البلدي-نهائي_compressed.pdf", 2),
    Sowing("الكتان", "Flax", (11,), "النصف الأول من نوفمبر، على ألا يتجاوز الأسبوع الثالث منه",
           "first half of November, no later than its third week", "الكتان2020_compressed.pdf", 4),
    Sowing("بنجر السكر", "Sugar beet", (8, 9, 10, 11), "من أغسطس حتى منتصف نوفمبر", "August to mid-November",
           "1-نشرة-بنجر-السكر_compressed.pdf", 5),
    Sowing("البصل", "Onion", (8, 9, 10, 11),
           "من منتصف أغسطس إلى نهاية سبتمبر في الوجه القبلي، ومن أول أكتوبر حتى نهاية نوفمبر في الوجه البحري",
           "mid-August to end of September in Upper Egypt; 1 October to end of November in the Delta",
           "نشرة-البصل-2022.pdf", 2),
    Sowing("الطماطم", "Tomato", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
           "سبع عروات: الصيفية المبكرة من أواخر يناير حتى أوائل مارس، الصيفية العادية من أواخر مارس حتى أوائل مايو، "
           "الخريفية المبكرة من أواخر يونيه إلى أواخر يوليه، الخريفية العادية من أوائل أغسطس حتى منتصف سبتمبر، "
           "الشتوية من أواخر سبتمبر إلى أوائل أكتوبر، والمحيرة تحت الأقبية البلاستيكية من أواخر نوفمبر إلى أوائل ديسمبر",
           "seven seasons: early summer late January–early March, summer late March–early May, early autumn late "
           "June–late July, autumn early August–mid-September, winter late September–early October, and under "
           "plastic tunnels late November–early December", "NEW_tomato.pdf", 8),
    Sowing("البطاطس", "Potato", (8, 9, 10, 11, 12, 1, 2), "من منتصف أغسطس حتى منتصف فبراير (العروات الصيفية والنيلية والمحيرة)",
           "mid-August to mid-February (summer, Nili and winter seasons)", "البطاطس_compressed.pdf", 7),
    Sowing("البطيخ", "Watermelon", (11, 12, 1),
           "تحت الأنفاق من أواخر نوفمبر وخلال ديسمبر، وأنسب موعد من 15 ديسمبر حتى أول يناير",
           "under tunnels from late November through December; best from 15 December to 1 January", "البطيخ.pdf", 5),
    Sowing("الكانتلوب", "Cantaloupe", (11, 12, 1, 2, 3, 8, 9),
           "تحت الأنفاق البلاستيكية من منتصف نوفمبر حتى منتصف يناير، والعروة الصيفية من منتصف فبراير إلى منتصف مارس، "
           "والخريفية من منتصف أغسطس إلى بداية سبتمبر",
           "under plastic tunnels mid-November to mid-January; summer season mid-February to mid-March; autumn season "
           "mid-August to early September", "الكانتلوب.pdf", 3),
    Sowing("الكوسة", "Zucchini", (2, 3, 8, 9, 10, 11),
           "العروة الصيفية من أول فبراير حتى منتصف مارس، والنيلية من منتصف أغسطس حتى آخر سبتمبر، "
           "والشتوية من سبتمبر إلى نوفمبر وتحتاج إلى تدفئة",
           "summer season 1 February to mid-March; Nili season mid-August to end of September; winter season "
           "September to November (needs protection from cold)", "نشرة-الكوسة.pdf", 3),
    Sowing("الكرنب", "Cabbage", (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2),
           "الكرنب البلدي تزرع بذوره بالمشتل من مارس إلى يونيو؛ والأصناف الأخرى تزرع بالمشتل من يوليو إلى نوفمبر وتشتل من سبتمبر إلى فبراير",
           "baladi cabbage is sown in the nursery March to June; other varieties are sown in the nursery July to "
           "November and transplanted September to February", "الكرنب_compressed.pdf", 5),
    Sowing("الزيتون", "Olive", (11, 3), "غرس الشتلات في نوفمبر في مناطق الأمطار، وفي منتصف مارس في المناطق التي تروى",
           "seedlings are planted in November in rain-fed areas and in mid-March in irrigated areas",
           "شجرة-الزيتون-نشرة-10.pdf", 26),
    Sowing("قصب السكر", "Sugarcane", (9, 10, 1, 2, 3),
           "القصب الخريفي خلال سبتمبر وأكتوبر، والربيعي من أواخر يناير حتى أواخر فبراير بمصر العليا وخلال فبراير ومارس بمصر الوسطى",
           "autumn cane in September and October; spring cane late January to late February in Upper Egypt, "
           "February and March in Middle Egypt", "نشرة-االقصب-2023-copy_compressed.pdf", 5),
    Sowing("الفراولة", "Strawberry", (8, 9, 10),
           "من منتصف أغسطس في أراضي وسط الدلتا، ومن بداية سبتمبر حتى أول أكتوبر في الأراضي الرملية والأماكن الحارة",
           "from mid-August in the central Delta; early September to 1 October in sandy soils and hot areas",
           "نشرة-الفراوله-طبع_compressed.pdf", 17),
    Sowing("القطن", "Cotton", (3, 4), "من أول مارس إلى 15 مارس في الوجه القبلي، ومن منتصف مارس إلى منتصف أبريل في الوجه البحري",
           "1–15 March in Upper Egypt; mid-March to mid-April in the Delta", "نشرة-القطن2023_compressed-1.pdf", 38),
    Sowing("دوار الشمس", "Sunflower", (3, 4, 5, 6, 7),
           "من مارس حتى يوليو: الصيفية المبكرة في مارس وأبريل، الصيفية في مايو ويونيه، والنيلية في يوليو بمصر الوسطى والعليا",
           "March to July: early summer in March–April, summer in May–June, Nili in July in Middle and Upper Egypt",
           "نشرة-دوار-الشمس.pdf", 3),
    Sowing("السمسم", "Sesame", (4, 5), "من منتصف أبريل حتى نهاية مايو", "mid-April to end of May",
           "نشرة-السمسم-copy_compressed.pdf", 3),
    Sowing("الذرة الشامية", "Maize", (5, 6), "خلال شهر مايو، وحتى منتصف يونيو", "during May, up to mid-June", "NEW_maize.pdf", 3),
    Sowing("الذرة الرفيعة", "Sorghum", (5, 6), "من منتصف مايو حتى منتصف يونيو", "mid-May to mid-June",
           "الذرة-الرفيعة-نهائي_compressed.pdf", 9),
    Sowing("فول الصويا", "Soybean", (5,), "خلال شهر مايو، ولا تتأخر الزراعة عن الأسبوع الثالث منه",
           "during May, no later than its third week", "نشرة-فول-الصويا-طبع_compressed.pdf", 3),
    Sowing("الخيار", "Cucumber", (2, 3, 4, 5, 6, 7, 8, 9), "من منتصف فبراير حتى منتصف سبتمبر في خمس عروات",
           "mid-February to mid-September, in five seasons", "زراعة-وانتاج-الخيار.pdf", 4),
    Sowing("الفلفل", "Pepper", (4, 5), "العروة الرئيسية: تزرع البذور بالمشتل في فبراير وأوائل مارس، ويتم الشتل في أبريل ومايو",
           "main season: seed is sown in the nursery in February and early March and transplanted in April and May",
           "الفلفل-16_compressed.pdf", 15),
    Sowing("الباذنجان", "Eggplant", (3, 4, 5), "العروة الصيفية المبكرة تشتل خلال مارس، والصيفية المتأخرة في أبريل ومايو",
           "early summer season transplanted in March; late summer season in April and May",
           "الباذنجان-17_compressed_compressed_compressed.pdf", 5),
)

# Crops with several seasons: only the season(s) of the month asked about are given (to the farmer and to the LLM),
# so nobody has to work out which of seven tomato seasons is the November one. (months, Arabic, English)
WINDOWS: dict[str, tuple] = {
    "Tomato": (((1, 2, 3), "العروة الصيفية المبكرة، من أواخر يناير حتى أوائل مارس", "early summer season, late January to early March"),
               ((3, 4, 5), "العروة الصيفية العادية، من أواخر مارس حتى أوائل مايو", "summer season, late March to early May"),
               ((6, 7), "العروة الخريفية المبكرة، من أواخر يونيه إلى أواخر يوليه", "early autumn season, late June to late July"),
               ((8, 9), "العروة الخريفية العادية، من أوائل أغسطس حتى منتصف سبتمبر", "autumn season, early August to mid-September"),
               ((9, 10), "العروة الشتوية، من أواخر سبتمبر إلى أوائل أكتوبر", "winter season, late September to early October"),
               ((11, 12), "العروة المحيرة تحت الأقبية البلاستيكية، من أواخر نوفمبر إلى أوائل ديسمبر",
                "under plastic tunnels, late November to early December")),
    "Cantaloupe": (((11, 12, 1), "تحت الأنفاق البلاستيكية، من منتصف نوفمبر حتى منتصف يناير", "under plastic tunnels, mid-November to mid-January"),
                   ((2, 3), "العروة الصيفية، من منتصف فبراير إلى منتصف مارس", "summer season, mid-February to mid-March"),
                   ((8, 9), "العروة الخريفية، من منتصف أغسطس إلى بداية سبتمبر", "autumn season, mid-August to early September")),
    "Zucchini": (((2, 3), "العروة الصيفية، من أول فبراير حتى منتصف مارس", "summer season, 1 February to mid-March"),
                 ((8, 9), "العروة النيلية، من منتصف أغسطس حتى آخر سبتمبر", "Nili season, mid-August to end of September"),
                 ((9, 10, 11), "العروة الشتوية، من سبتمبر إلى نوفمبر وتحتاج إلى تدفئة", "winter season, September to November (needs protection from cold)")),
    "Cabbage": (((3, 4, 5, 6), "الكرنب البلدي: تزرع بذوره بالمشتل من مارس إلى يونيو", "baladi cabbage: sown in the nursery March to June"),
                ((7, 8, 9, 10, 11), "الأصناف الأخرى: تزرع بذورها بالمشتل من يوليو إلى نوفمبر", "other varieties: sown in the nursery July to November"),
                ((9, 10, 11, 12, 1, 2), "شتل الأصناف الأخرى في الحقل من سبتمبر إلى فبراير", "other varieties are transplanted to the field September to February")),
    "Olive": (((11,), "غرس الشتلات في نوفمبر في مناطق الأمطار", "seedlings are planted in November in rain-fed areas"),
              ((3,), "غرس الشتلات في منتصف مارس في المناطق التي تروى", "seedlings are planted in mid-March in irrigated areas")),
    "Sugarcane": (((9, 10), "القصب الخريفي، خلال سبتمبر وأكتوبر", "autumn cane, in September and October"),
                  ((1, 2), "القصب الربيعي بمصر العليا، من أواخر يناير حتى أواخر فبراير", "spring cane in Upper Egypt, late January to late February"),
                  ((2, 3), "القصب الربيعي بمصر الوسطى، خلال فبراير ومارس", "spring cane in Middle Egypt, February and March")),
    "Onion": (((8, 9), "في الوجه القبلي، من منتصف أغسطس إلى نهاية سبتمبر", "in Upper Egypt, mid-August to end of September"),
              ((10, 11), "في الوجه البحري، من أول أكتوبر حتى نهاية نوفمبر", "in the Delta, 1 October to end of November")),
    "Faba bean": (((10,), "النصف الثاني من أكتوبر في أسيوط وسوهاج وقنا وأسوان", "second half of October in Assiut, Sohag, Qena and Aswan"),
                  ((11,), "النصف الأول من نوفمبر في الوجه البحري", "first half of November in the Delta")),
    "Sunflower": (((3, 4), "العروة الصيفية المبكرة، خلال مارس وأبريل", "early summer season, March and April"),
                  ((5, 6), "العروة الصيفية، خلال مايو ويونيه", "summer season, May and June"),
                  ((7,), "العروة النيلية، خلال يوليو بمصر الوسطى والعليا", "Nili season, July in Middle and Upper Egypt")),
    "Strawberry": (((8,), "من منتصف أغسطس في أراضي وسط الدلتا", "from mid-August in the central Delta"),
                   ((9, 10), "من بداية سبتمبر حتى أول أكتوبر في الأراضي الرملية والأماكن الحارة", "early September to 1 October in sandy soils and hot areas")),
    "Cotton": (((3,), "من أول مارس إلى 15 مارس في الوجه القبلي، ومن منتصف مارس في الوجه البحري", "1–15 March in Upper Egypt; from mid-March in the Delta"),
               ((4,), "حتى منتصف أبريل في الوجه البحري", "until mid-April in the Delta")),
    "Eggplant": (((3,), "العروة الصيفية المبكرة، تشتل خلال مارس", "early summer season, transplanted in March"),
                 ((4, 5), "العروة الصيفية المتأخرة، تشتل في أبريل ومايو", "late summer season, transplanted in April and May")),
}

KINDS = {"vegetable": {"Onion", "Tomato", "Potato", "Watermelon", "Cantaloupe", "Zucchini", "Cabbage", "Cucumber", "Pepper",
                       "Eggplant", "Strawberry"},
         "fruit": {"Olive", "Strawberry", "Watermelon", "Cantaloupe"},
         "field": {"Wheat", "Faba bean", "Flax", "Sugar beet", "Cotton", "Sunflower", "Sesame", "Maize", "Sorghum", "Soybean",
                   "Sugarcane"}}
_KIND_RES = {"vegetable": re.compile(r"(?i)\bvegetables?\b|\bveg\b|خضار|خضروات|خضر(?!ا)"),
             "fruit": re.compile(r"(?i)\bfruits?\b|فاكهه|فواكه"),
             "field": re.compile(r"(?i)\bfield crops?\b|\bgrains?\b|\bcereals?\b|محاصيل حقليه|حبوب|محاصيل الحقل")}


def when(s: Sowing, month: int | None, k: int) -> str:
    """The crop's sowing dates for that month only (k: 0 Arabic, 1 English); all its dates when it has one season."""
    seasons = [w for w in WINDOWS.get(s.en, ()) if month in w[0]]
    if not seasons:
        return s.when_en if k else s.when_ar
    return ("; " if k else "؛ ").join(w[1 + k] for w in seasons)


def rows_for(month: int, *texts: str) -> list[Sowing]:
    """The crops sown in a month, shortest sowing window first; only vegetables / fruit / field crops when asked."""
    found = sorted((item for item in CALENDAR if month in item.months), key=lambda item: len(item.months))
    text = normalize_for_match(" ".join(t for t in texts if t))
    for kind, pattern in _KIND_RES.items():
        if pattern.search(text):
            return [item for item in found if item.en in KINDS[kind]] or found
    return found


@dataclass(frozen=True)
class Soil:
    suits: frozenset          # of "sandy", "clay", "calcareous"
    avoid: frozenset          # of "sandy", "clay", "calcareous", "saline"
    ar: str                   # what the bulletin says, in short
    en: str
    source: str
    page: int


def _soil(suits: str, avoid: str, ar: str, en: str, source: str, page: int) -> Soil:
    return Soil(frozenset(suits.split()), frozenset(avoid.split()), ar, en, source, page)


# What each bulletin says about soil (keyed by the crop's English name above). A crop that is not here has a
# bulletin that does not state a soil type: it is then listed separately, never guessed.
SOILS: dict[str, Soil] = {
    "Tomato": _soil("sandy clay calcareous", "", "تجود في جميع أنواع الأراضي من الرملية حتى الطينية الثقيلة، بشرط جودة الصرف وخلوها من أمراض الذبول والنيماتودا",
                    "does well in all soils from sandy to heavy clay, if well drained and free of wilt diseases and nematodes", "NEW_tomato.pdf", 5),
    "Pepper": _soil("sandy clay", "", "ينجح في جميع أنواع الأراضي من الرملية الخفيفة إلى الطينية بشرط جودة الصرف",
                    "succeeds in all soils from light sandy to clay, if well drained", "الفلفل-16_compressed.pdf", 7),
    "Zucchini": _soil("sandy clay calcareous", "saline", "تجود في جميع أنواع الأراضي (الجيرية والطينية والرملية) عدا الملحية ورديئة الصرف",
                      "does well in all soils (calcareous, clay and sandy) except saline and poorly drained ones", "نشرة-الكوسة.pdf", 2),
    "Faba bean": _soil("sandy clay calcareous", "saline", "يجود في معظم الأراضي بما فيها الأراضي الجديدة (الجيرية والرملية)، ماعدا المتأثرة بالملوحة والقلوية وسيئة الصرف",
                       "does well in most soils including new lands (calcareous and sandy), except saline, alkaline and poorly drained ones",
                       "الفول-البلدي-نهائي_compressed.pdf", 2),
    "Cantaloupe": _soil("sandy", "", "يجود في الأراضي الرملية الخالية من الأمراض الفطرية والنيماتودا",
                        "does well in sandy soils free of fungal diseases and nematodes", "الكانتلوب.pdf", 3),
    "Cucumber": _soil("sandy", "saline", "يجود في الأراضي الرملية الخالية من الأملاح، ويتأثر بشدة بملوحة التربة",
                      "does well in salt-free sandy soils; very sensitive to soil salinity", "زراعة-وانتاج-الخيار.pdf", 3),
    "Olive": _soil("sandy", "", "يجود في الأراضي الطميية الخفيفة الخصبة جيدة الصرف، وبدرجة مرضية في الأراضي الرملية مع العناية بالري والتسميد العضوي",
                   "does well in fertile, well-drained light loam, and acceptably in sandy soils with good irrigation and organic fertilizer",
                   "شجرة-الزيتون-نشرة-10.pdf", 23),
    "Onion": _soil("sandy clay", "", "يجود في التربة الصفراء والسوداء الخفيفة والرملية", "does well in yellow, light black and sandy soils",
                   "نشرة-البصل-2022.pdf", 12),
    "Strawberry": _soil("sandy", "clay calcareous", "تجود في الأراضي الطميية والرملية، ولا تنمو جيدا في الطينية الثقيلة ولا تفضل في الجيرية",
                        "does well in loam and sandy soils; grows poorly in heavy clay and is not advised in calcareous soils",
                        "نشرة-الفراوله-طبع_compressed.pdf", 16),
    "Cotton": _soil("clay", "sandy calcareous saline", "يجود في الأراضي الطينية السوداء والصفراء جيدة الصرف، مع تجنب الرملية والجيرية والملحية",
                    "does well in well-drained black and yellow clay soils; avoid sandy, calcareous and saline soils",
                    "نشرة-القطن2023_compressed-1.pdf", 22),
    "Wheat": _soil("sandy clay calcareous", "", "يزرع في الأراضي القديمة والجديدة، وله نشرة خاصة بزراعته في الأراضي الجديدة",
                   "grown in both old and new lands; it has its own bulletin for newly reclaimed lands",
                   "القمح-فى-الأراضى-الجديدة_compressed.pdf", 1),
    "Maize": _soil("sandy clay", "", "هجنه تجود زراعتها بالأراضي القديمة والجديدة", "its hybrids do well in both old and new lands", "NEW_maize.pdf", 4),
    "Soybean": _soil("sandy clay", "", "تجود زراعته بالأراضي القديمة والجديدة", "does well in both old and new lands",
                     "نشرة-فول-الصويا-طبع_compressed.pdf", 2),
    "Watermelon": _soil("sandy", "", "يمكن زراعته بالأراضي الجديدة تحت الري بالتنقيط في الأنفاق البلاستيكية",
                        "can be grown in new lands under drip irrigation in plastic tunnels", "البطيخ.pdf", 5),
}

SOIL_NAMES = {"sandy": ("الرملية", "sandy"), "clay": ("الطينية", "clay"), "calcareous": ("الجيرية", "calcareous"),
              "saline": ("الملحية", "saline")}
_SOIL_RES = {
    "saline": re.compile(r"(?i)\bsalin\w*|\bsalty\b|ملحي|مالح|ملوحه|املاح"),
    "calcareous": re.compile(r"(?i)\bcalcareous\b|\blime\w*|جيري"),
    "sandy": re.compile(r"(?i)\bsand\w*|\bdesert\b|\bnew(?:ly reclaimed)? lands?\b|\breclaimed\b|رملي|رمليه|رمل|صحراوي|صحراء|"
                        r"مستصلح|اراضي جديده|ارض جديده|الاراضي الجديده|الارض الجديده"),
    "clay": re.compile(r"(?i)\bclay\w*|\bheavy soil\b|\bold lands?\b|\bblack soil\b|طيني|طينيه|طين|سوداء|ثقيله|اراضي قديمه|ارض قديمه|"
                       r"الاراضي القديمه|الارض القديمه"),
}
# governorate -> (Arabic name, English name, region): "delta" = الوجه البحري, "upper" = مصر الوسطى والعليا (الوجه القبلي)
_GOVERNORATES = [
    ("المنيا", "Minya", "upper", r"المنيا|minya|minia"), ("بني سويف", "Beni Suef", "upper", r"بني سويف|beni ?suef"),
    ("الفيوم", "Fayoum", "upper", r"الفيوم|fayoum|faiyum"), ("أسيوط", "Assiut", "upper", r"اسيوط|assiut|asyut"),
    ("سوهاج", "Sohag", "upper", r"سوهاج|sohag"), ("قنا", "Qena", "upper", r"(?<!\w)قنا(?!\w)|qena"),
    ("الأقصر", "Luxor", "upper", r"الاقصر|luxor"), ("أسوان", "Aswan", "upper", r"اسوان|aswan"),
    ("الجيزة", "Giza", "upper", r"الجيزه|giza"), ("الوادي الجديد", "New Valley", "upper", r"الوادي الجديد|new valley"),
    ("البحيرة", "Beheira", "delta", r"البحيره|beheira|behera"), ("الغربية", "Gharbia", "delta", r"الغربيه|gharbia"),
    ("الدقهلية", "Dakahlia", "delta", r"الدقهليه|dakahlia"), ("الشرقية", "Sharqia", "delta", r"الشرقيه|sharqia|sharkia"),
    ("كفر الشيخ", "Kafr El-Sheikh", "delta", r"كفر الشيخ|kafr el.?sheikh"), ("المنوفية", "Monufia", "delta", r"المنوفيه|monufia|menoufia"),
    ("القليوبية", "Qalyubia", "delta", r"القليوبيه|qalyubia"), ("دمياط", "Damietta", "delta", r"دمياط|damietta"),
    ("الإسماعيلية", "Ismailia", "delta", r"الاسماعيليه|ismailia"), ("الإسكندرية", "Alexandria", "delta", r"الاسكندريه|alexandria"),
    ("النوبارية", "Nubaria", "delta", r"النوباريه|nubaria"),
    ("الوجه البحري", "the Delta", "delta", r"الوجه البحري|بحري|الدلتا|\bdelta\b|lower egypt"),
    ("الوجه القبلي", "Upper Egypt", "upper", r"الوجه القبلي|قبلي|الصعيد|upper egypt|middle egypt"),
]
_GOVERNORATE_RES = [(ar, en, region, re.compile(rf"(?i){pattern}")) for ar, en, region, pattern in _GOVERNORATES]


@dataclass
class Land:
    soil: str | None = None          # "sandy" / "clay" / "calcareous" / "saline"
    place_ar: str = ""
    place_en: str = ""
    region: str | None = None        # "delta" / "upper"

    def __bool__(self) -> bool:
        return bool(self.soil or self.region)


def land_in(question: str) -> Land:
    """The soil type and governorate a farmer mentions ("أرضي رملية في المنيا", "clay soil in Beheira")."""
    text = normalize_for_match(question)
    land = Land()
    for soil, pattern in _SOIL_RES.items():
        if pattern.search(text):
            land.soil = soil
            break
    for ar, en, region, pattern in _GOVERNORATE_RES:
        if pattern.search(text):
            land.place_ar, land.place_en, land.region = ar, en, region
            break
    return land


MONTHS_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
_MONTH_WORDS = {
    1: ("يناير", "كانون الثاني", "january", "jan"), 2: ("فبراير", "شباط", "february", "feb"),
    3: ("مارس", "اذار", "march"), 4: ("ابريل", "نيسان", "april", "apr"), 5: ("مايو", "ايار", "may"),
    6: ("يونيو", "يونيه", "حزيران", "june", "jun"), 7: ("يوليو", "يوليه", "تموز", "july", "jul"),
    8: ("اغسطس", "اب", "august", "aug"), 9: ("سبتمبر", "ايلول", "september", "sept", "sep"),
    10: ("اكتوبر", "تشرين الاول", "october", "oct"), 11: ("نوفمبر", "تشرين الثاني", "november", "nov"),
    12: ("ديسمبر", "كانون الاول", "december", "dec"),
}
_NOW_RE = re.compile(r"(?i)\b(?:now|this month|right now|these days|currently)\b|دلوقتي|الان|حاليا|الشهر ده|هذا الشهر|"
                     r"الشهر الحالي|الايام دي|هذه الايام")
_PLANT_RE = re.compile(r"(?i)\b(?:plant\w*|sow\w*|grow\w*|cultivat\w*)\b|[اأنتي]?زرع|زراع|ازرع|نزرع|تزرع|يزرع")
_WHAT_RE = re.compile(r"(?i)\b(?:crops?|what (?:can|should|to|do)|which|recommend\w*|best|suitable)\b|محاصيل|محصول|ايه|ماذا|ما هي|ما هو|"
                      r"افضل|انسب|مناسب|اقدر|ممكن|توصي|تنصح|اقتراح|ترشح")


def month_in(question: str, today: datetime | None = None) -> int | None:
    """The month a question names (Arabic, Levantine or English name), or the current one for "now"."""
    text = normalize_for_match(question)
    for number, names in _MONTH_WORDS.items():
        for name in names:
            if len(name) <= 3 and not name.isascii():   # "اب" (August) is also a common syllable: needs "شهر" before it
                if re.search(rf"شهر\s+{name}(?!\w)", text):
                    return number
            elif name == "may":
                if re.search(r"\b(?:in|of|during|month of)\s+may\b", text):
                    return number
            elif re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text):
                return number
    numbered = re.search(r"شهر\s*(\d{1,2})(?!\d)|\bmonth\s*(\d{1,2})\b", text)
    if numbered and 1 <= int(numbered.group(1) or numbered.group(2)) <= 12:
        return int(numbered.group(1) or numbered.group(2))
    if _NOW_RE.search(text):
        return (today or datetime.now()).month
    return None


_SUGGEST_RE = re.compile(r"(?i)\b(?:recommend\w*|suggest\w*|give me|i want|i need|looking for|ideas?)\b|اقترح|رشح|انصحني|"
                         r"اديني|اعطني|عايز|عاوز|اريد|ابغي|محتاج|دلني|قولي|قل لي")
_CROP_WORD_RE = re.compile(r"(?i)\b(?:crops?|plants?|vegetables?|something to (?:plant|grow))\b|محاصيل|محصول|نباتات|نبات|"
                           r"زرعات|زرعه|خضروات|خضار")
# another subject (fertilizer, pests, prices...): a normal question, not "what should I plant?"
_OTHER_TOPIC_RE = re.compile(r"(?i)\b(?:fertili\w*|pests?|disease\w*|irrigat\w*|price\w*|market\w*|harvest\w*|stor\w*|spray\w*|"
                             r"weed\w*|soil test)\b|سماد|اسمده|تسميد|مبيد|افات|افه|مرض|امراض|الري|سعر|اسعار|تسويق|حصاد|"
                             r"تخزين|رش|حشائش|مكافحه")


def is_calendar_question(question: str, today: datetime | None = None) -> int | None:
    """The month to answer for, when the question asks what to plant and names no crop; else None.
    With a month ("in November", "now") that month; a plain request for crops to plant ("اقترح علي محاصيل",
    "عايز نباتات أزرعها في أرضي", "recommend crops for my land") is answered for the current month."""
    if crops_in(question):
        return None
    text = normalize_for_match(question)
    asks_what_to_plant = bool(_PLANT_RE.search(text) and _WHAT_RE.search(text))
    asks_for_crops = bool(_SUGGEST_RE.search(text) and _CROP_WORD_RE.search(text))
    if not (asks_what_to_plant or asks_for_crops):
        return None
    month = month_in(question, today)
    if month is not None:
        return month
    # no month named: only a personal request ("what should I plant?", "عايز محاصيل أزرعها في أرضي"), not a
    # general question ("what crops are grown in Egypt?") or another subject (fertilizer, pests, prices...)
    personal = re.search(r"(?i)\b(?:i|my|me|we|our)\b|ارضي|ازرع|نزرع|اقدر|عندي|مزرعتي", text)
    if _OTHER_TOPIC_RE.search(text) or not (asks_for_crops or personal):
        return None
    return (today or datetime.now()).month


def names_month(question: str) -> bool:
    return month_in(question) is not None


def _passage(s: Sowing, month: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"calendar-{s.en.lower()}", text=f"{s.ar} — ميعاد الزراعة: {when(s, month, 0)}.",
                          source=s.source, page_start=s.page, page_end=s.page, score=1.0,
                          topic=f"{s.ar} / {s.en.lower()}", section="ميعاد الزراعة")


def _soil_passage(s: Sowing, soil: Soil) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"soil-{s.en.lower()}", text=f"{s.ar} — الأرض المناسبة: {soil.ar}.", source=soil.source,
                          page_start=soil.page, page_end=soil.page, score=1.0, topic=f"{s.ar} / {s.en.lower()}",
                          section="الأرض المناسبة")


def plan(question: str, history: list[str] | None = None, today: datetime | None = None):
    """(month, land, month was named) when this is a "what should I plant?" question — or the farmer's reply about
    their land to the one before ("أرضي رملية في المنيا") — else None."""
    land = land_in(question)
    month = is_calendar_question(question, today)
    if month is not None:
        return month, land, names_month(question)
    previous = history[-1] if history else None
    if land and previous and not crops_in(question) and len(question.split()) <= 14 \
            and not _OTHER_TOPIC_RE.search(normalize_for_match(question)):
        earlier = plan(previous, history[:-1], today)
        if earlier is not None:            # a reply to "what is your soil, and which governorate?"
            month, before, named = earlier
            return month, Land(land.soil or before.soil, land.place_ar or before.place_ar,
                               land.place_en or before.place_en, land.region or before.region), named
    return None


def answer(question: str, language: str, history: list[str] | None = None, today: datetime | None = None):
    """(answer text with [n] markers, passages) for a "what should I plant?" question, or None.
    With the farmer's soil type, only the crops whose bulletin says they suit it are recommended."""
    planned = plan(question, history, today)
    if planned is None:
        return None
    month, land, asked_month = planned
    rows = rows_for(month, question, (history or [""])[-1])
    if not rows:
        return None
    ar = language == "ar"
    k = 0 if ar else 1
    name = (MONTHS_AR if ar else MONTHS_EN)[month - 1]
    lines, chunks = [], []

    def cite(passage: RetrievedChunk) -> int:
        chunks.append(passage)
        return len(chunks)

    if land.soil:
        soil_name = SOIL_NAMES[land.soil][k]
        good = [s for s in rows if s.en in SOILS and land.soil in SOILS[s.en].suits]
        bad = [s for s in rows if s.en in SOILS and land.soil in SOILS[s.en].avoid]
        silent = [s for s in rows if s not in good and s not in bad]
        for s in good:
            n, m = cite(_passage(s, month)), cite(_soil_passage(s, SOILS[s.en]))
            lines.append(f"- {s.ar}: {when(s, month, 0)} [{n}]. {SOILS[s.en].ar} [{m}]." if ar else
                         f"- {s.en}: {when(s, month, 1)} [{n}]. Soil: {SOILS[s.en].en} [{m}].")
        where = (f" في {land.place_ar}" if ar else f" in {land.place_en}") if land.place_ar else ""
        if good:
            head = (f"لأرضك {soil_name}{where}، هذه محاصيل شهر {name} التي تذكر نشراتها أنها تناسب هذا النوع من الأراضي:" if ar else
                    f"For your {soil_name} soil{where}, these are the crops sown in {name} whose bulletins say they suit it:")
        else:
            head = (f"لا تذكر النشرات محصولا من محاصيل شهر {name} يناسب الأراضي {soil_name} تحديدا." if ar else
                    f"None of the bulletins names a crop sown in {name} as suited to {soil_name} soil.")
        extra = []
        if bad:
            items = []
            for s in bad:
                items.append(f"{s.ar if ar else s.en} [{cite(_soil_passage(s, SOILS[s.en]))}]")
            extra.append((f"لا تنصح النشرات بزراعة هذه في الأراضي {soil_name}: " if ar else
                          f"The bulletins advise against these in {soil_name} soil: ") + ("، " if ar else ", ").join(items) + ".")
        if silent:
            items = [f"{s.ar if ar else s.en} [{cite(_passage(s, month))}]" for s in silent]
            extra.append((f"تزرع أيضا في شهر {name}، لكن نشراتها لا تحدد نوع الأرض: " if ar else
                          f"Also sown in {name}, but their bulletins do not state a soil type: ")
                         + ("، " if ar else ", ").join(items) + ".")
    else:
        for s in rows:
            n = cite(_passage(s, month))
            lines.append(f"- {s.ar}: {when(s, month, 0)} [{n}]." if ar else f"- {s.en}: {when(s, month, 1)} [{n}].")
        extra = []
        if asked_month or land.region:
            head = (f"المحاصيل التي تزرع في مصر خلال شهر {name}، حسب نشرات الإرشاد الزراعي:" if ar else
                    f"Crops sown in Egypt in {name}, according to the extension bulletins:")
        else:
            head = (f"نحن الآن في شهر {name}، وهذه المحاصيل التي يمكنك زراعتها في مصر في هذا الوقت حسب نشرات الإرشاد الزراعي:" if ar else
                    f"It is {name} now, and these are the crops you can plant in Egypt at this time, according to the "
                    f"extension bulletins:")
    if land.region:
        region = {"delta": ("الوجه البحري", "the Delta (Lower Egypt)"), "upper": ("الوجه القبلي (مصر الوسطى والعليا)", "Upper and Middle Egypt")}[land.region]
        extra.append(f"{land.place_ar} من {region[0]}: حيث تختلف المواعيد بين الوجهين، اتبع مواعيد {region[0].split(' (')[0]}." if ar and land.place_ar != region[0].split(" (")[0]
                     else (f"اتبع المواعيد المذكورة لـ{region[0]}." if ar else
                           f"{land.place_en} is in {region[1]}: where dates differ by region, follow the ones given for it."))
    if land.soil and land.region:
        tail = ("اسألني عن أي محصول منها لمعرفة التقاوي والتسميد والري." if ar else
                "Ask me about any of these crops for seed rate, fertilizer and irrigation.")
    elif land.soil:
        tail = ("في أي محافظة أرضك؟ بعض المواعيد تختلف بين الوجه البحري والقبلي. ويمكنك أن تسألني عن أي محصول منها." if ar else
                "Which governorate is your land in? Some dates differ between the Delta and Upper Egypt. You can also ask "
                "me about any of these crops.")
    else:
        tail = ("لأرشح لك الأنسب منها: ما نوع أرضك (رملية أم طينية أم جيرية)" + ("؟ " if land.region else "، وفي أي محافظة؟ ") +
                "ويمكنك أن تسألني عن أي محصول منها لمعرفة التقاوي والتسميد والري." if ar else
                "To narrow it down: is your soil sandy, clay or calcareous" + ("? " if land.region else ", and which governorate are you in? ") +
                "You can also ask me about any of these crops for seed rate, fertilizer and irrigation.")
    body = "\n".join(lines)
    notes = "\n\n".join(extra)
    return "\n\n".join(part for part in (f"{head}\n{body}".strip(), notes, tail) if part), chunks


def context(question: str, language: str, history: list[str] | None = None, today: datetime | None = None):
    """(passages, note for the LLM) for a "what should I plant?" question, or None.

    The facts come from the bulletins (each crop sown that month: its dates, and what its bulletin says about soil);
    the LLM then writes the answer to the farmer's own question from them, so "suggest new crops", "what suits my
    sandy land?" and "which vegetables can I plant now?" each get their own answer. `answer()` above is the plain
    list used when the LLM's answer does not pass the checks."""
    planned = plan(question, history, today)
    if planned is None:
        return None
    month, land, _ = planned
    rows = rows_for(month, question, (history or [""])[-1])
    if not rows:
        return None
    ar = language == "ar"
    k = 0 if ar else 1
    name = (MONTHS_AR if ar else MONTHS_EN)[month - 1]

    def rank(s: Sowing) -> int:   # crops the bulletins say suit the farmer's soil first, advised-against last
        if not land.soil or s.en not in SOILS:
            return 1
        return 0 if land.soil in SOILS[s.en].suits else 2 if land.soil in SOILS[s.en].avoid else 1

    passages = []
    for s in sorted(rows, key=rank)[:10]:
        soil = SOILS.get(s.en)
        if ar:
            text = f"{s.ar} — ميعاد الزراعة: {when(s, month, 0)}."
            text += f" الأرض المناسبة: {soil.ar}." if soil else " النشرة لا تحدد نوع الأرض المناسبة."
            if land.soil and soil and land.soil in soil.avoid:
                text += f" لا ينصح بزراعته في الأراضي {SOIL_NAMES[land.soil][0]}."
        else:
            text = f"{s.en} — sowing date: {when(s, month, 1)}."
            text += f" Suitable soil: {soil.en}." if soil else " The bulletin does not state a suitable soil type."
            if land.soil and soil and land.soil in soil.avoid:
                text += f" Not advised in {SOIL_NAMES[land.soil][1]} soil."
        passages.append(RetrievedChunk(chunk_id=f"calendar-{s.en.lower()}", text=text, source=s.source, page_start=s.page,
                                       page_end=s.page, score=1.0, topic=f"{s.ar} / {s.en.lower()}",
                                       section="ميعاد الزراعة" if ar else "Sowing date"))
    region = {"delta": ("الوجه البحري", "the Delta (Lower Egypt)"), "upper": ("الوجه القبلي", "Upper Egypt")}.get(land.region or "")
    if ar:
        note = (f"الشهر المقصود: {name}. كل فقرة محصول يزرع في مصر في هذا الشهر، ومواعيدها كلها تناسب هذا الشهر: "
                f"لا تقل عن أي محصول منها إنه لا يناسب الشهر. اكتب بالعربية فقط، واذكر اسم كل محصول ترشحه.")
        if land.soil:
            note += f" أرض المزارع {SOIL_NAMES[land.soil][0]}: رشح ما تذكر الفقرات أنه يناسبها، ونبه إلى ما لا ينصح به فيها."
        if region:
            note += f" أرضه في {land.place_ar} ({region[0]}): اذكر المواعيد الخاصة بـ{region[0]} حيث تختلف."
        note += " أجب عن طلبه هو بالتحديد، واختر من الفقرات ما يناسبه مع ميعاد زراعة كل محصول."
    else:
        note = (f"The month in question: {name}. Each passage is a crop sown in Egypt in this month, and all their dates "
                f"fit this month: never say one of them does not suit the month. Name every crop you recommend.")
        if land.soil:
            note += (f" The farmer's soil is {SOIL_NAMES[land.soil][1]}: recommend what the passages say suits it, and warn "
                     f"about what is not advised in it.")
        if region:
            note += f" The land is in {land.place_en} ({region[1]}): give the dates for {region[1]} where they differ."
        note += " Answer this farmer's own request, choosing from the passages what fits it, with each crop's sowing date."
    if land.soil and region:
        follow = ""
    elif ar:
        follow = ("في أي محافظة أرضك؟ بعض المواعيد تختلف بين الوجه البحري والقبلي." if land.soil else
                  "ما نوع أرضك (رملية أم طينية أم جيرية)" + ("؟" if region else "، وفي أي محافظة؟") + " لأرشح لك الأنسب منها.")
    else:
        follow = ("Which governorate is your land in? Some dates differ between the Delta and Upper Egypt." if land.soil else
                  "Is your soil sandy, clay or calcareous" + ("?" if region else ", and which governorate are you in?") +
                  " Then I can narrow these down for you.")
    return passages, note, follow


_NEGATIVE_RE = re.compile(r"لا\s+[تي]ناسب\s+هذا\s+الشهر|غير\s+مناسب\w*\s+لهذا\s+الشهر|(?i:not\s+suit\w*\s+(?:for\s+)?this\s+month|"
                          r"does(?:n't| not)\s+suit\s+this\s+month)")


def sound(answer: str, passages: list, language: str) -> bool:
    """Is the LLM's suggestion usable? It must name at least two of the crops it was given (one if only one was),
    every list item must name one, and it must not claim a crop "does not suit this month" (they all do)."""
    names = [normalize_for_match(p.topic.split(" / ")[0 if language == "ar" else 1]) for p in passages]
    text = normalize_for_match(answer)
    if sum(n in text for n in names) < min(2, len(names)) or _NEGATIVE_RE.search(answer):
        return False
    items = [normalize_for_match(line) for line in answer.split("\n") if re.match(r"\s*(?:[-*•]|\d{1,2}[.)])\s+\S", line)]
    return all(any(n in line for n in names) for line in items)
