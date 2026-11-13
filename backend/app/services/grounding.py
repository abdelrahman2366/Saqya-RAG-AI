"""Is an answer supported by the passages it was written from? And small, safe repairs.

Used in two places with the same definitions:
- rag_service.RAGService checks every answer before the farmer sees it (and asks the LLM to try
  again once if something is wrong), and
- the notebook's evaluation (2.6) scores answers with grounding_check().

"Grounded" means: the answer cites at least one passage, every number in it appears in the
passages (or in the question), and, when the answer is in the same language as the passages it
cites, at least 60% of its content words appear in the passages.
"""

from __future__ import annotations

import re
from typing import Sequence

from app.services.glossary import Term, crops_in
from app.services.lexical import tokenize
from app.utils.text import detect_language, normalize_for_match, to_ascii_digits

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,،]\s*\d+)*)\]")
_NUMBER_RE = re.compile(r"\d+(?:[.,،٫]\d+)*")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-•*]\s*)?\d{1,2}[.)\-](?:\s|$)")
_LEADING_MARKER_RE = re.compile(r"^\s*(?:(?:[-•*]\s*)?\d{1,2}[.)\-]|[-•*])\s+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?؟])\s+")
SUPPORT_THRESHOLD = 0.6

# Fixed safety line the API adds to answers about pesticides (not a claim from the documents)
SAFETY_NOTE = {
    "en": "⚠️ Follow the product label and wear protective equipment when handling pesticides.",
    "ar": "⚠️ اتبع تعليمات الملصق على عبوة المبيد وارتدِ ملابس ومعدات الوقاية عند التعامل مع المبيدات.",
}

NO_ANSWER = {
    "en": "I don't have enough information in my documents to answer that.",
    "ar": "لا تتوفر لدي معلومات كافية في مستنداتي للإجابة على هذا السؤال.",
}
# Fragments that identify a refusal, compared after normalize_for_match()
_REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    normalize_for_match("لا تتوفر لدي معلومات كافية"),
    normalize_for_match("ليس لدي معلومات كافية"),
    normalize_for_match("لا توجد معلومات كافية"),
    normalize_for_match("لا تتوفر معلومات كافية"),
]
# Sentences that only send the farmer elsewhere (part of a refusal, not an answer)
_REFERRAL_MARKERS = [
    "extension officer", "extension agent", "extension service", "extension worker", "contact your",
    "contact a ", "consult",
    *(normalize_for_match(w) for w in ("المرشد", "الإرشاد", "اتصل", "الاتصال ب", "استشر", "extension")),
]
_CONNECTOR_RE = re.compile(
    r"^\s*(?:however|but|that said|still|nevertheless)\s*,?\s*|^\s*(?:ولكن|لكن|ومع ذلك|إلا أن|الا ان)\s*،?\s*",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ small helpers
_ARTICLE_RE = re.compile(r"(?<!\w)(?:[وفبك]?ال|لل)(?=\w\w)")


def _match_form(text: str) -> str:
    return _ARTICLE_RE.sub("", normalize_for_match(text))


def contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    """Does the text contain one of the keywords? Ignores case, diacritics, alef/taa-marbuta forms,
    Arabic-Indic digits and the Arabic article ('التجفيف' matches 'تجفيف')."""
    text = _match_form(text)
    return any(_match_form(k) in text for k in keywords)


def is_refusal(answer: str) -> bool:
    text = normalize_for_match(answer.replace("’", "'"))
    return any(marker in text for marker in _REFUSAL_MARKERS)


def citations(answer: str, num_chunks: int) -> list[int]:
    """Passage numbers cited in the answer, in order of first use (out-of-range numbers ignored)."""
    cited: list[int] = []
    for group in CITATION_RE.findall(answer):
        for part in re.split(r"[,،]", group):
            n = int(part)
            if 1 <= n <= num_chunks and n not in cited:
                cited.append(n)
    return cited


def _normal_number(raw: str) -> str:
    raw = raw.replace("،", ",").replace("٫", ".")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):  # 1,000 -> 1000
        raw = raw.replace(",", "")
    raw = raw.replace(",", ".")
    if raw.count(".") > 1:  # a date such as 15.11.2022: keep as written
        return raw
    whole, _, frac = raw.partition(".")
    whole, frac = whole.lstrip("0") or "0", frac.rstrip("0")
    return f"{whole}.{frac}" if frac else whole


def numbers(text: str) -> set[str]:
    """Numbers in a text, normalised: Arabic digits -> ASCII, '2,5' -> '2.5', '1,000' -> '1000'.

    A lone 0 is ignored: in these PDFs it is usually a broken degree sign ("15م0").
    """
    found = set()
    for raw in _NUMBER_RE.findall(to_ascii_digits(text)):
        n = _normal_number(raw)
        if n != "0":
            found.add(n)
    return found


def _passage_numbers(text: str) -> set[str]:
    """numbers() plus the forms a right-to-left PDF may have scrambled: '5.1' may be 1.5."""
    found = numbers(text)
    for n in list(found):
        if n.count(".") == 1:
            whole, frac = n.split(".")
            found |= {_normal_number(f"{frac}.{whole}"), whole, frac}
    return found


def answer_body(answer: str) -> str:
    """The answer without [n] markers and list numbering (those numbers are not claims)."""
    body = CITATION_RE.sub(" ", to_ascii_digits(answer))
    return _LIST_MARKER_RE.sub(" ", body)


def content_words(text: str) -> set[str]:
    """Stemmed content words (stop words and numbers removed), Arabic or English."""
    return {t for t in tokenize(text) if not t[0].isdigit() and len(t) >= 3}


def segments(answer: str) -> list[str]:
    """Sentences / bullet lines of an answer (newlines kept as their own items). A list marker ("5. ", "- ")
    stays with its sentence: on its own, "5." would look like a number the passages don't have."""
    out: list[str] = []
    for line in answer.split("\n"):
        marker = _LEADING_MARKER_RE.match(line)
        rest = line[marker.end():] if marker else line
        parts = [p for p in _SENTENCE_END_RE.split(rest) if p.strip()] or [""]
        if marker:
            parts[0] = marker.group(0) + parts[0]
        out.extend(parts)
        out.append("\n")
    return out[:-1]


def renumber_lists(answer: str) -> str:
    """1. 2. 4. 5. (a point was removed) -> 1. 2. 3. 4. Each numbered list counts from 1."""
    # a removed point leaves an empty line inside the list: close it up
    answer = re.sub(r"(?m)^(\s*(?:\d{1,2}[.)]|[-•*])\s.*)\n\s*\n(?=\s*(?:\d{1,2}[.)]|[-•*])\s)", r"\1\n", answer)
    lines, n, out = answer.split("\n"), 0, []
    for line in lines:
        m = re.match(r"^(\s*)(\d{1,2})([.)])(\s+)", line)
        if m:
            n += 1
            line = f"{m.group(1)}{n}{m.group(3)}{m.group(4)}{line[m.end():]}"
        elif line.strip():
            n = 0
        out.append(line)
    return "\n".join(out)


def join_segments(parts: list[str]) -> str:
    text = ""
    for p in parts:
        text += p if p == "\n" or not text or text.endswith("\n") else " " + p
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _context_numbers(chunks: Sequence, question: str) -> set[str]:
    allowed = numbers(question)
    for c in chunks:
        allowed |= _passage_numbers(c.text) | numbers(getattr(c, "label", ""))
    return allowed


def unsupported_numbers(answer: str, chunks: Sequence, question: str = "") -> list[str]:
    allowed = _context_numbers(chunks, question)
    return sorted(n for n in numbers(answer_body(answer)) if n not in allowed)


def support(answer: str, chunks: Sequence) -> float | None:
    """Share of the answer's content words found in the passages; None for a translated answer."""
    cited = citations(answer, len(chunks)) or range(1, len(chunks) + 1)
    language = detect_language(answer)
    if any(detect_language(chunks[n - 1].text) != language for n in cited):
        return None  # facts translated from another language: word overlap means nothing
    words = content_words(answer_body(answer))
    if not words:
        return None
    context = set()
    for c in chunks:
        context |= content_words(c.text)
    return round(sum(w in context for w in words) / len(words), 2)


def grounding_check(answer: str, chunks: Sequence, question: str = "") -> dict:
    """{'grounded', 'support', 'unsupported_numbers', 'citations'} for one answer."""
    for note in SAFETY_NOTE.values():
        answer = answer.replace(note, "").strip()
    if is_refusal(answer):
        return {"grounded": True, "support": 1.0, "unsupported_numbers": [], "citations": 0}
    if not chunks:
        return {"grounded": False, "support": 0.0, "unsupported_numbers": [], "citations": 0}
    cited = citations(answer, len(chunks))
    bad_numbers = unsupported_numbers(answer, chunks, question)
    share = support(answer, chunks) if cited else 0.0
    return {
        "grounded": bool(cited) and not bad_numbers and (share is None or share >= SUPPORT_THRESHOLD),
        "support": share,
        "unsupported_numbers": bad_numbers,
        "citations": len(cited),
    }


# ------------------------------------------------------------------ problems the LLM can fix
def is_echo(answer: str, question: str) -> bool:
    """The answer only repeats the question (a known failure of small models): no new words and
    no new numbers."""
    body = answer_body(answer)
    words = content_words(body)
    if not words or numbers(body) - numbers(question):
        return False
    return len(words - content_words(question)) <= 1


def has_content(answer: str) -> bool:
    """Something to tell the farmer: words or numbers ("60 kg/ha [3]" counts)."""
    body = answer_body(answer)
    return bool(content_words(body) or numbers(body))


def is_about(chunk, crop: Term) -> bool:
    """Is a passage about this crop? Its document's topic, or the passage names it (either language)."""
    topic = getattr(chunk, "topic", "") or ""
    return topic.split(" / ")[-1] == crop.en or crop in crops_in(chunk.text)


# Area units: Egyptian bulletins give amounts per feddan, Indian/FAO documents per hectare or acre
AREA_UNITS = {
    "feddan": re.compile(r"(?<![a-z])feddans?(?![a-z])|فدان|فدادين"),
    "hectare": re.compile(r"(?<![a-z])(?:hectares?|ha)(?![a-z])|هكتار"),
    "acre": re.compile(r"(?<![a-z])acres?(?![a-z])|ايكر"),
}
UNIT_NAMES = {"feddan": ("feddan", "فدان"), "hectare": ("hectare", "هكتار"), "acre": ("acre", "إيكر")}


def area_units(text: str) -> set[str]:
    """Area units a text names ('per feddan', 'للهكتار', ...)."""
    norm = normalize_for_match(text)
    return {unit for unit, rx in AREA_UNITS.items() if rx.search(norm)}


def unit_mismatch(answer: str, question: str) -> tuple[set[str], set[str]]:
    """(units the question asks for, other units the answer uses instead) — both empty if fine."""
    asked, used = area_units(question), area_units(answer_body(answer))
    if asked and used and not used & asked:
        return asked, used
    return set(), set()


# How bad each problem is (a retried answer is kept only if its total is lower)
SEVERITY = {"garbled": 6, "refusal": 6, "echo": 5, "language": 4, "crop": 4, "units": 3, "numbers": 3, "unsupported": 2,
            "uncited": 1}


# ------------------------------------------------------------------ garbled output
# A small model sometimes breaks down: Arabic glued to English ("للpassesات", "ي tolerant للجفاف"), a word
# stuttered ("و و و و"), its own instructions repeated ("اكتب الإجابة بالعربية"), or a list of near-identical
# made-up items. None of that may reach a farmer.
_AR = "؀-ۿ"
_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,،]\s*\d+)*\]")
# qwen is trained mostly on Chinese and sometimes slips into it mid-answer: never shown
_OTHER_SCRIPT_RE = re.compile("[\u0400-\u04FF\u0E00-\u0E7F\u3000-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uFF00-\uFFEF]")
_GLUED_RE = re.compile(rf"[{_AR}][A-Za-z]|[A-Za-z][{_AR}]")
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]+")
_STUTTER_RE = re.compile(r"(?<!\S)(\S{1,12})(?:\s+\1){2,}(?!\S)")
_OK_LATIN = {"ph", "npk", "ec", "ppm", "kg", "cm", "mm", "ml", "km", "ha", "gm", "fao", "icarda", "dna", "bt", "uv",
             "co2", "saqya", "em"}
_PROMPT_LEAK_RES = [
    re.compile(r"(?i)\b(?:write|answer|respond|reply)\b[^.\n]{0,40}\bin\s+(?:arabic|english)\b"),
    re.compile(r"(?i)\b(?:using|use)\s+only\s+the\s+passages\b|\bafter\s+every\s+sentence\b|\beven\s+if\s+the\s+passages\b"),
    re.compile(r"(?:اكتب|أكتب|اكتبي|أجب|اجب)\s+(?:ال)?(?:إجابة|اجابة|جواب|رد)?[^.\n]{0,30}(?:بالعربية|باللغة العربية|بالإنجليزية|باللغة الإنجليزية)"),
    re.compile(r"حتى لو كانت الفقرات|بعد كل جملة|باستخدام الفقرات فقط"),
]


def is_prompt_leak(sentence: str) -> bool:
    """The model repeating its instructions ("Write the whole answer in Arabic...")."""
    return any(rx.search(sentence) for rx in _PROMPT_LEAK_RES)


def is_garbled(sentence: str, language: str, passages_text: str = "") -> bool:
    """Is this sentence broken text rather than language? (see above)"""
    body = _CITATION_RE.sub(" ", sentence)
    if _OTHER_SCRIPT_RE.search(body) or _GLUED_RE.search(body) or _STUTTER_RE.search(body) or is_prompt_leak(body):
        return True
    if language == "ar":
        latin = _LATIN_WORD_RE.findall(body)
        arabic = re.findall(rf"[{_AR}]{{2,}}", body)
        if latin and not arabic and len(re.findall(r"\d", body)) < 2:
            return True          # a whole line in Latin letters inside an Arabic answer ("Mâche")
        lowered = passages_text.lower()
        for word in latin:
            w = word.lower()
            if w in _OK_LATIN or len(w) <= 2:
                continue
            # allowed: a name or scientific term (capitalised / upper case) that the passages themselves use
            if (word[0].isupper() or word.isupper()) and w in lowered:
                continue
            return True          # an ordinary English word in the middle of Arabic ("tolerant")
    return False


def degenerate_list(answer: str) -> bool:
    """A list whose items are nearly the same words again and again ("عشب الخس / عشب الخضرة / عشب الخضر")."""
    from difflib import SequenceMatcher

    items = [normalize_for_match(_CITATION_RE.sub("", _LEADING_MARKER_RE.sub("", line))).strip(" .،")
             for line in answer.split("\n") if line.strip()]
    items = [i for i in items if 0 < len(i.split()) <= 4]
    if len(items) < 3:
        return False
    close = sum(1 for a, b in zip(items, items[1:]) if SequenceMatcher(None, a, b).ratio() >= 0.6
                or a.split()[0] == b.split()[0])
    return close >= max(2, (len(items) - 1) * 0.6)


def garbled(answer: str, language: str, chunks: Sequence = ()) -> bool:
    language = detect_language(answer_body(answer)) or language   # judged in the language it is mostly written in
    text = "\n".join(getattr(c, "text", "") for c in chunks)
    return degenerate_list(answer) or any(p != "\n" and is_garbled(p, language, text) for p in segments(answer))


def drop_garbled(answer: str, language: str, chunks: Sequence = ()) -> str:
    """The answer without its broken sentences ("" when it is a degenerate list or nothing sound is left)."""
    if degenerate_list(answer):
        return ""
    language = detect_language(answer_body(answer)) or language
    text = "\n".join(getattr(c, "text", "") for c in chunks)
    kept = [p for p in segments(answer) if p == "\n" or not is_garbled(p, language, text)]
    out = renumber_lists(join_segments(kept)).strip()
    # an introduction left pointing at nothing ("...حسب العروة:" with every item gone)
    return out if has_content(out) and not re.fullmatch(r"[^\n]*[:：]\s*", out) else ""


def problems(answer: str, chunks: Sequence, question: str, language: str | None = None) -> list[str]:
    """Everything wrong with an answer, most serious first ([] = fine).

    refusal · echo (only repeats the question) · language (not in the question's language) ·
    crop (the question names a crop but no cited passage is about it) · units (asked per feddan, answered
    per hectare) · numbers (a number found in no passage) · unsupported (under 60% of its words are in the
    passages) · uncited (no [n] at all)
    """
    if is_refusal(answer):
        return ["refusal"]
    if is_echo(answer, question):
        return ["echo"]
    language = language or detect_language(question)
    found = []
    if detect_language(answer_body(answer)) != language:
        found.append("language")
    if garbled(answer, language, chunks):
        found.append("garbled")
    cited = citations(answer, len(chunks))
    asked = crops_in(question, language)
    if asked and cited and not any(is_about(chunks[n - 1], crop) for n in cited for crop in asked):
        found.append("crop")
    if unit_mismatch(answer, question)[0]:
        found.append("units")
    if unsupported_numbers(answer, chunks, question):
        found.append("numbers")
    share = support(answer, chunks) if cited else None
    if share is not None and share < SUPPORT_THRESHOLD:
        found.append("unsupported")
    if not cited:
        found.append("uncited")
    return found


def addresses(answer: str, questions: Sequence[str]) -> bool:
    """Does the answer talk about what was asked? It must share a content word with the question
    (or its translation) other than the crop's name: "the price of tomatoes" answered with tomato
    planting dates shares only "tomato".
    """
    crop_words: set[str] = set()
    for q in questions:
        for t in crops_in(q):
            crop_words |= content_words(" ".join([t.en, t.ar, *t.ar_alt]))
    asked = set().union(*(content_words(q) for q in questions)) - crop_words
    said = content_words(answer_body(answer)) - crop_words
    return not asked or bool(asked & said)


def severity(found: list[str]) -> int:
    return sum(SEVERITY[p] for p in found)


def diagnose(answer: str, chunks: Sequence, question: str) -> str | None:
    """The most serious problem of an answer, or None."""
    found = problems(answer, chunks, question)
    return found[0] if found else None


# ------------------------------------------------------------------ repairs that need no LLM
_CITE_AFTER_PUNCT_RE = re.compile(r"([.!?؟])\s*((?:\[\d+(?:\s*[,،]\s*\d+)*\]\s*)+)")


def citations_before_punctuation(answer: str) -> str:
    """'... November. [1]' -> '... November [1].' so a marker stays with its own sentence."""
    return _CITE_AFTER_PUNCT_RE.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", answer).strip()


def drop_invalid_citations(answer: str, num_chunks: int) -> str:
    """Remove [n] markers that point to passages that don't exist (e.g. [6] with 4 passages)."""
    def fix(match: re.Match) -> str:
        kept = [p.strip() for p in re.split(r"[,،]", match.group(1)) if 1 <= int(p) <= num_chunks]
        return f"[{', '.join(kept)}]" if kept else ""
    return re.sub(r"\s*\[\s*\]", "", CITATION_RE.sub(fix, answer)).strip()


def drop_echoed_question(answer: str, question: str) -> str:
    """Remove a first sentence that just repeats the question, when an answer follows it."""
    parts = segments(answer)
    first = next((i for i, p in enumerate(parts) if p.strip() and p != "\n"), None)
    if first is None:
        return answer
    rest = join_segments(parts[first + 1:])
    if is_echo(parts[first], question) and content_words(answer_body(rest)):
        return rest
    return answer


def add_missing_citations(answer: str, chunks: Sequence, min_overlap: float = 0.5) -> str:
    """Cite the best-matching passage after any sentence that states facts without a [n] marker.

    Only when the sentence clearly comes from one passage (half or more of its content words are in
    it); translated sentences are left alone.
    """
    passage_words = [content_words(c.text) for c in chunks]
    parts = segments(answer)
    for i, part in enumerate(parts):
        if part == "\n" or CITATION_RE.search(part):
            continue
        words = content_words(answer_body(part))
        if len(words) < 3:
            continue
        overlaps = [len(words & pw) / len(words) for pw in passage_words]
        best = max(range(len(chunks)), key=lambda k: overlaps[k])
        if overlaps[best] >= min_overlap:
            parts[i] = re.sub(r"([.!?؟:]?)\s*$", rf" [{best + 1}]\1", part.rstrip(), count=1)
    return join_segments(parts)


def strip_refusal(answer: str) -> str:
    """'I don't have enough information. However, according to [1], ...' -> the part after it.

    Returns '' if nothing but the refusal and a referral to the extension officer is left.
    """
    kept = []
    for part in segments(answer):
        text = normalize_for_match(part)
        if part != "\n" and (is_refusal(part) or any(m in text for m in _REFERRAL_MARKERS)):
            continue
        kept.append(part)
    rest = join_segments(kept)
    if kept and not CITATION_RE.sub("", rest).strip(" .،,\n"):
        return ""
    rest = _CONNECTOR_RE.sub("", rest, count=1).strip()
    return rest[:1].upper() + rest[1:]


def fix_wrong_citations(answer: str, chunks: Sequence) -> str:
    """A sentence copied from passage [3] but marked [1] gets [3]: the marker must point to the
    passage that actually says it (only when that is clear from the words)."""
    passage_words = [content_words(c.text) for c in chunks]
    parts = segments(answer)
    for i, part in enumerate(parts):
        cited = citations(part, len(chunks))
        words = content_words(answer_body(part))
        if not cited or len(words) < 3:
            continue
        overlaps = [len(words & pw) / len(words) for pw in passage_words]
        best = max(range(len(chunks)), key=lambda k: overlaps[k])
        if max(overlaps[n - 1] for n in cited) < 0.3 and overlaps[best] >= 0.5:
            parts[i] = CITATION_RE.sub("", part).rstrip()
            parts[i] = re.sub(r"\s*([.!?؟:]?)\s*$", rf" [{best + 1}]\1", parts[i], count=1)
    return join_segments(parts)


def drop_unsupported_sentences(answer: str, chunks: Sequence, question: str) -> str:
    """Remove sentences that state a number found in no passage, and, when the answer is in the
    language of the passages it cites, uncited sentences with few words from any passage."""
    allowed = _context_numbers(chunks, question)
    check_words = support(answer, chunks) is not None if citations(answer, len(chunks)) else False
    passage_words = [content_words(c.text) for c in chunks]
    kept = []
    for p in segments(answer):
        if p != "\n":
            if not numbers(answer_body(p)) <= allowed:
                continue
            words = content_words(answer_body(p))
            if check_words and not CITATION_RE.search(p) and len(words) >= 3 \
                    and max(len(words & pw) / len(words) for pw in passage_words) < 0.5:
                continue
        kept.append(p)
    return renumber_lists(join_segments(kept))


def renumber_citations(answer: str, mapping: dict[int, int]) -> str:
    """[1], [2] of an answer written from a subset of the passages -> their numbers in the full list."""
    def fix(match: re.Match) -> str:
        nums = [mapping.get(int(p), 0) for p in re.split(r"[,،]", match.group(1))]
        return f"[{', '.join(str(n) for n in nums if n)}]"
    return re.sub(r"\s*\[\s*\]", "", CITATION_RE.sub(fix, answer))


def tidy(answer: str, chunks: Sequence, question: str) -> str:
    """All the LLM-free repairs, in order. A refusal mixed with a real answer keeps the answer."""
    answer = citations_before_punctuation(drop_invalid_citations(answer, len(chunks)))
    if is_refusal(answer):
        rest = strip_refusal(answer)
        if rest and citations(rest, len(chunks)) and content_words(answer_body(rest)):
            answer = rest
    answer = drop_echoed_question(answer, question)
    if not is_refusal(answer):
        answer = add_missing_citations(fix_wrong_citations(answer, chunks), chunks)
    return answer


# ------------------------------------------------------------------ presentation
# Words that talk about the passages instead of farming ("according to passage 2", "وفقاً للفقرة [1]"): the farmer
# never sees the passages, so these are cut. The [n] markers themselves stay (the chat app hides them).
# (only words that name the passages: "sources", "texts" are left alone — "water sources", "nitrogen sources")
_REF_EN = r"(?:the\s+)?(?:provided\s+|given\s+|above\s+)?(?:passages?|documents?|bulletins?)(?:\s*\[\d+\])*(?:\s+\d+)?"
_META_RES = [
    re.compile(rf"(?i)\b(?:according\s+to|based\s+on|as\s+(?:stated|mentioned|described)\s+in|as\s+per)\s+{_REF_EN}\s*,?\s*"),
    re.compile(rf"(?i)\b{_REF_EN}\s+(?:says?|states?|mentions?|indicates?|notes?|explains?)\s+(?:that\s+)?"),
    re.compile(r"(?i)\b(?:in|from)\s+passage\s*\[?\d+\]?\s*,?\s*"),
]
_REF_AR = r"(?:ال)?(?:فقرة|فقرات|فقره|نص|نصوص|مستند|مستندات|وثيقة|وثائق|مصدر|مصادر|مرفق|مرفقات|نشرة|النشرة)(?:\s*\[\d+\])*(?:\s*(?:رقم\s*)?\d+)?"
_META_RES += [
    re.compile(rf"(?:وفقا|وفقاً|وفقًا|طبقا|طبقاً|حسب|بحسب|استنادا|استناداً|بناء|بناءً|بناءا)\s+(?:على\s+|إلى\s+|الى\s+|ل|لل)?{_REF_AR}\s*،?\s*"),
    re.compile(rf"(?:تشير|تذكر|تقول|توضح|يشير|يذكر|يقول|يوضح|أشارت|ذكرت)\s+{_REF_AR}\s*(?:إلى\s+|الى\s+)?(?:أن|ان|أنه|انه)?\s*"),
    re.compile(rf"(?:كما\s+)?(?:ورد|جاء|ذكر)\s+في\s+{_REF_AR}\s*،?\s*(?:أن|ان)?\s*"),
]
# A sentence only about what the passages do or don't say ("The passages do not mention the price.", "لا تذكر
# الفقرات موعد الحصاد."): nothing for the farmer, so it is dropped (an answer made only of these is a refusal)
_META_SENTENCE_RES = [
    re.compile(r"(?i)\b(?:the\s+)?(?:provided\s+|given\s+)?(?:passages?|documents?)\b[^.?!]*\b(?:do(?:es)?\s+not|don't|doesn't|did\s+not|"
               r"no\s+(?:information|mention|details?)|not\s+(?:mention|provide|say|include|specify|contain))"),
    re.compile(r"(?i)\b(?:no|not\s+any)\s+(?:information|details?)\s+(?:is|are|was|were)?\s*(?:given|provided|mentioned|available)"
               r"[^.?!]*\b(?:passages?|documents?)\b"),
    re.compile(r"(?:لا|لم)\s+(?:ت|ي)?(?:ذكر|تذكر|يذكر|توضح|يوضح|تتناول|يتناول|تحدد|يحدد|تشر|يشر|تتضمن|يتضمن|تحتوي|يحتوي)\w*\s+"
               r"(?:ال)?(?:فقرات|فقرة|نص|نصوص|مستند|مستندات|وثيقة|مصادر|المصادر)"),
    re.compile(r"(?:ال)?(?:فقرات|فقرة|نصوص|مستندات|وثائق)\s+(?:المقدمة\s+|المتاحة\s+)?(?:لا|لم)\s+\w+"),
]
_ANSWER_LABEL_RE = re.compile(r"^\s*(?:#{1,4}\s*)?(?:\*\*)?(?:answer|final answer|الإجابة|الاجابة|الجواب)\s*[:：]\s*(?:\*\*)?\s*",
                              re.IGNORECASE)
# The encyclopedia's record codes and source tags, if one slips through: AGR-0051, UC-TOM, ARC-TOM, UMN-VEG
_JUNK_CODE_RE = re.compile(r"\s*\(?\b(?:AGR-\d{3,5}|[A-Z]{2,4}-[A-Z]{3})\b\)?(?:\s*[|·,،]\s*(?=\b[A-Z]{2,4}-))?")


_ASK_WORDS = {"how", "what", "when", "where", "which", "why", "who", "does", "do", "did", "is", "are", "can", "should",
              "much", "many", "the", "and", "for", "with", "need", "needs", "about", "this", "that", "it", "its", "there",
              "ما", "ماذا", "كيف", "متى", "كم", "هل", "اين", "لماذا", "من", "في", "على", "الى", "عن", "او", "ثم", "هو", "هي",
              "هذا", "هذه", "ذلك", "تحتاج", "يحتاج", "احتاج", "اريد", "افعل", "اعمل", "يجب", "ممكن", "طيب", "بعد", "قبل"}


def _light_words(text: str) -> set[str]:
    """Words for a quick "same subject?" check: normalised, article and و/ب/ل/ف removed, English plural -s removed."""
    out = set()
    for w in re.findall(r"\w{2,}", normalize_for_match(text)):
        if w.startswith("و") and w[1:] in _ASK_WORDS:
            continue
        if w in _ASK_WORDS or w.isdigit():
            continue
        w = re.sub(r"^(?:و|ف|ب|ك)?(?:ال|لل)(?=\w\w)", "", w)
        w = re.sub(r"(?<=\w\w\w)s$", "", w) if w.isascii() else w
        if len(w) >= 2 and w not in _ASK_WORDS:
            out.add(w)
    return out


def mentions(question: str, chunks: Sequence) -> bool:
    """Do the passages speak about something the question asks, beyond the crop names? ("water" in "how much
    water does it need?"; nothing about "rabbits" in passages about cotton)."""
    asked = _light_words(re.sub(r"\[\d+\]", " ", question))
    for term in crops_in(question):
        asked -= _light_words(" ".join([term.en, term.ar, *term.ar_alt]))
    if not asked:
        return True
    found = set().union(*(_light_words(c.text) for c in chunks)) if chunks else set()
    return bool(asked & found)


def polish(answer: str, language: str | None = None) -> str:
    """The answer as a farmer should read it: no "Answer:" label, no talk about passages, no record codes, no sentence
    said twice, no empty bullets, tidy spacing and punctuation. Facts, numbers and [n] markers are not touched.
    Returns "" when nothing is left (only sentences about what the passages don't say): the caller then refuses."""
    if not answer or is_refusal(answer):
        return answer
    text = _ANSWER_LABEL_RE.sub("", answer.replace("\r", ""))
    lang = language or detect_language(text)
    text = join_segments([part for part in segments(text) if part == "\n" or not (
        is_prompt_leak(part) or _OTHER_SCRIPT_RE.search(part) or _GLUED_RE.search(part) or _STUTTER_RE.search(_CITATION_RE.sub(" ", part)))])
    text = join_segments([part for part in segments(text)
                          if part == "\n" or not any(rx.search(part) for rx in _META_SENTENCE_RES)])
    for rx in _META_RES:
        text = rx.sub("", text)
    text = _JUNK_CODE_RE.sub("", text)
    text = re.sub(r"[ \t]*\|[ \t]*(?=\n|$)", "", text)                  # a dangling "|" left by a removed code
    # a sentence or bullet said twice: keep the first
    seen: set[str] = set()
    kept: list[str] = []
    for part in segments(text):
        if part != "\n":
            key = re.sub(r"\W+", " ", normalize_for_match(answer_body(part))).strip()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
        kept.append(part)
    text = join_segments(kept)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"^(\s*)[*•▪●◦]\s+", r"\1- ", line)                # one bullet style
        if re.fullmatch(r"\s*(?:[-*•]|\d{1,2}[.)])\s*(?:\[\d+\]\s*)*[.،,]?\s*", line):   # an empty bullet
            continue
        line = re.sub(r"[ \t]{2,}", " ", line)
        line = re.sub(r"\s+([.,;:!?،؛؟])", r"\1", line)                  # no space before punctuation
        line = re.sub(r"([.,;:!?،؛؟])(?=[^\s\d\]\)\"'».,;:!?،؛؟-])", r"\1 ", line)  # one space after it
        line = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", line))
        line = re.sub(r"\.{2,}(?!\.)", ".", line)                        # ".." -> "."
        line = re.sub(r"(?<=\s)[.،,]\s*$", "", line).rstrip()
        lines.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if language == "ar" or detect_language(text) == "ar":
        text = re.sub(r"(?<=[؀-ۿ]),\s*(?=[؀-ۿ])", "، ", text)   # Arabic comma between Arabic words
        # PDF text read right to left can mirror brackets: "شهري)مارس وإبريل(" -> "شهري (مارس وإبريل)"
        text = re.sub(r"\s*\)([^()\n]{1,60})\(\s*", lambda m: f" ({m.group(1).strip()}) ", text)
        # a broken degree sign: "15مo", "15م0", "15oم" -> "15°م"
        text = re.sub(r"(\d)\s*م\s*[oO0°](?![\w\u0600-\u06FF])", r"\1°م", text)
        text = re.sub(r"(\d)\s*[oO°]\s*م(?![\u0600-\u06FF])", r"\1°م", text)
        text = re.sub(r"[ \t]{2,}", " ", re.sub(r"[ \t]+([.,;:!?،؛؟)])", r"\1", text))
    # a line that now starts with a lower-case letter (its opening words were cut): capitalise it
    text = re.sub(r"(?m)^(\s*(?:[-*]\s+|\d{1,2}[.)]\s+)?)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return renumber_lists(text) if has_content(text) else ""
