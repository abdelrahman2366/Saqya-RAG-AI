"""Read, clean and chunk one document the same way notebooks/rag_pipeline.ipynb treats the corpus.

Used for documents a farmer uploads in the chat app (see documents.py): an uploaded Arabic bulletin
must come out as readable as the indexed ones, so the reading, cleaning and chunking code below is the
pipeline's own, copied from the notebook (sections 2.1 and 2.2):

- PDFs: pdfminer, glyphs read right to left for Arabic, lines rebuilt from positions where a PDF stores
  one printed line as fragments; PDFs made by clean_rag_pdfs_no_ai.ipynb are read with their title /
  section / sub-section structure.
- Cleaning: NFKC, broken-glyph and wrong-glyph repairs, reversed pages, repeated headers / footers and
  page numbers removed, unreadable pages skipped.
- Chunking: 800 characters with 150 overlap, sentence-aware; section-aware with a heading path when the
  document has headings.

Word (.docx), Markdown and plain-text files are supported too (Word heading styles become sections).

The copied part is generated from the notebook by a small script; if you change those notebook cells,
copy the change here as well.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTAnno, LTChar, LTContainer, LTTextContainer, LTTextLine
from pypdf import PdfReader
from wordfreq import zipf_frequency

from app.utils.text import detect_language, normalize_for_match, strip_arabic_marks

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
OCR_THRESHOLD = 30        # characters; a page with less text is blank or scanned
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 50
DOCX_PAGE_CHARS = 3000    # Word files have no pages: every ~3000 characters counts as one "page"


# ================================================================ copied from rag_pipeline.ipynb
ARABIC_CHARS = "؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿"   # incl. presentation forms


_ARABIC_CHAR_RE = re.compile(f"[{ARABIC_CHARS}]")


_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


_LTR_RUN = r"[A-Za-z0-9٠-٩۰-۹][A-Za-z0-9٠-٩۰-۹.,٫%/:+\-]*"


_LTR_RUNS_RE = re.compile(f"{_LTR_RUN}(?:\\s+{_LTR_RUN})*")


_SWAP_BRACKETS = str.maketrans("()[]", ")(][")


def is_rtl(text: str) -> bool:
    return len(_ARABIC_CHAR_RE.findall(text)) > len(_LATIN_CHAR_RE.findall(text))


def fragment_text(line) -> str:
    """Text of one pdfminer line fragment in reading order.

    Many Arabic PDFs draw their glyphs left to right (visual order). Such fragments are read right to
    left *glyph by glyph*: a ligature glyph that stands for several letters (لأ, تح, جم ...) keeps its
    letters in order, and runs of digits / Latin words are turned back to left-to-right.
    """
    items = [o for o in line if isinstance(o, (LTChar, LTAnno))]
    text = "".join(o.get_text() for o in items)
    if not is_rtl(text):
        return _fix_arabic_runs(items) if _ARABIC_CHAR_RE.search(text) else text
    xs = [o.x0 for o in items if isinstance(o, LTChar)]
    if sum(b > a for a, b in zip(xs, xs[1:])) >= len(xs) / 2:  # glyphs stored left to right
        text = "".join(o.get_text() for o in reversed(items))
        text = _LTR_RUNS_RE.sub(lambda m: m.group(0)[::-1], text)
    return text


def _fix_arabic_runs(items) -> str:
    """English line containing Arabic words: if the Arabic glyphs are stored left to right, reverse each
    Arabic run glyph by glyph (keeps ligature glyphs whole)."""
    def is_ar(o):
        return isinstance(o, LTChar) and bool(_ARABIC_CHAR_RE.search(o.get_text()))
    ar_x = [o.x0 for o in items if is_ar(o)]
    if len(ar_x) < 2 or sum(b > a for a, b in zip(ar_x, ar_x[1:])) < (len(ar_x) - 1) / 2:
        return "".join(o.get_text() for o in items)
    out, run = [], []
    for o in items + [None]:
        if o is not None and (is_ar(o) or (run and o.get_text() == " ")):
            run.append(o.get_text())
            continue
        trailing = []
        while run and run[-1] == " ":
            trailing.append(run.pop())
        out.extend(reversed(run))
        out.extend(trailing)
        run = []
        if o is not None:
            out.append(o.get_text())
    return "".join(out)


def _text_lines(obj):
    if isinstance(obj, LTTextLine):
        yield obj
    elif isinstance(obj, LTContainer):
        for child in obj:
            yield from _text_lines(child)


def _fix_brackets(text: str) -> str:
    """Some PDFs store mirrored brackets: on lines where ')' comes before '(', swap them."""
    def violations(t):
        depth = bad = 0
        for ch in t:
            depth += (ch == "(") - (ch == ")")
            if depth < 0:
                bad, depth = bad + 1, 0
        return bad + depth
    lines = []
    for line in text.split("\n"):  # line by line: one page can mix both conventions
        swapped = line.translate(_SWAP_BRACKETS)
        lines.append(swapped if violations(swapped) < violations(line) else line)
    return "\n".join(lines)


def _block_text(layout) -> str:
    """pdfminer's own reading: text blocks in layout order (handles columns and tables well)."""
    blocks = []
    for element in layout:
        if isinstance(element, LTTextContainer):
            lines = [re.sub(r"\s+", " ", fragment_text(l)).strip() for l in _text_lines(element)]
            lines = [l for l in lines if l]
            if lines:
                blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _split_line_pairs(layout) -> tuple[int, int]:
    """Count touching fragments of the same printed line that pdfminer put in different blocks."""
    frags = []
    for b, element in enumerate(layout):
        if isinstance(element, LTTextContainer):
            for line in _text_lines(element):
                if line.get_text().strip():
                    frags.append((b, line))
    frags.sort(key=lambda f: (-round((f[1].y0 + f[1].y1) / 2), f[1].x0))
    pairs = 0
    rows = 1 if frags else 0
    for (b1, l1), (b2, l2) in zip(frags, frags[1:]):
        same_row = abs((l1.y0 + l1.y1) / 2 - (l2.y0 + l2.y1) / 2) <= 0.5 * min(l1.height, l2.height)
        rows += not same_row
        if same_row and b1 != b2 and l2.x0 - l1.x1 < 0.8 * max(l1.height, 1):
            pairs += 1
    return pairs, rows


def page_text_from_layout(layout) -> str:
    """Page text in reading order.

    Normally pdfminer's block reading is used. Some PDFs store one printed line as several touching
    fragments and pdfminer files them in different blocks, which scrambles sentences; on such pages
    the lines are rebuilt from fragment positions instead (right to left for Arabic), and real text
    columns, if any, are read one column at a time.
    """
    block_text = _block_text(layout)
    if not is_rtl(block_text):  # English pages: pdfminer's block reading
        return block_text
    pairs, rows = _split_line_pairs(layout)
    if pairs < max(3, 0.1 * rows):
        return _fix_brackets(block_text)
    frags = []
    for line in _text_lines(layout):
        raw = fragment_text(line).replace("\n", "")
        text = re.sub(r"\s+", " ", raw).strip()
        if text:
            sizes = [c.size for c in line if isinstance(c, LTChar)]
            frags.append({"x0": line.x0, "x1": line.x1, "y": (line.y0 + line.y1) / 2,
                          "h": max(line.height, 1.0), "size": sorted(sizes)[len(sizes) // 2] if sizes else line.height,
                          "text": text, "lead": raw[:1].isspace(), "trail": raw[-1:].isspace()})
    # sideways text (a title running down the margin) arrives as a stack of one-letter fragments: drop it
    def lone_letter(f):
        return len(f["text"]) == 1 and f["text"].isalpha()
    singles = Counter(round(f["x0"] / 3) for f in frags if lone_letter(f))
    frags = [f for f in frags if not (lone_letter(f) and singles[round(f["x0"] / 3)] >= 4)]
    if not frags:
        return ""
    rtl_page = is_rtl(" ".join(f["text"] for f in frags))

    # 1. fragments on the same baseline -> one visual line
    frags.sort(key=lambda f: -f["y"])
    rows = []
    for f in frags:
        if rows and abs(rows[-1]["y"] - f["y"]) <= 0.5 * min(rows[-1]["h"], f["h"]):
            rows[-1]["frags"].append(f)
        else:
            rows.append({"y": f["y"], "h": f["h"], "frags": [f]})

    # 2. within a line, split at wide gaps (columns / table cells) into segments
    for row in rows:
        row["frags"].sort(key=lambda f: f["x0"])
        segments, current = [], [row["frags"][0]]
        for a, b in zip(row["frags"], row["frags"][1:]):
            if b["x0"] - a["x1"] > max(0.8 * a["size"], 5):
                segments.append(current)
                current = [b]
            else:
                current.append(b)
        segments.append(current)
        row["segments"] = segments

    # 3. column gutters: an x position with wide text on both sides in many lines, and few lines crossing it
    width = max(f["x1"] for f in frags) - min(f["x0"] for f in frags)
    candidates = sorted({round((a[-1]["x1"] + b[0]["x0"]) / 2)
                         for r in rows for a, b in zip(r["segments"], r["segments"][1:])})
    gutters = []
    for x in candidates:
        crossing = sum(any(sg[0]["x0"] < x < sg[-1]["x1"] for sg in r["segments"]) for r in rows)
        column_rows = 0
        for r in rows:
            left = [sg for sg in r["segments"] if sg[-1]["x1"] <= x]
            right = [sg for sg in r["segments"] if sg[0]["x0"] >= x]
            if left and right:
                l, rr = max(left, key=lambda sg: sg[-1]["x1"]), min(right, key=lambda sg: sg[0]["x0"])
                column_rows += (l[-1]["x1"] - l[0]["x0"] > 0.2 * width) and (rr[-1]["x1"] - rr[0]["x0"] > 0.2 * width)
        if crossing <= 0.2 * len(rows) and column_rows >= max(4, 0.15 * len(rows)):
            if not gutters or x - gutters[-1] > 0.1 * width:
                gutters.append(x)

    def segment_text(seg) -> str:
        parts = sorted(seg, key=lambda f: -f["x0"]) if rtl_page else seg
        out = parts[0]["text"]
        for prev, f in zip(parts, parts[1:]):
            gap = (prev["x0"] - f["x1"]) if rtl_page else (f["x0"] - prev["x1"])
            space = prev["trail"] or f["lead"] or gap > 0.15 * f["size"]
            out += (" " if space else "") + f["text"]
        return out

    def column_of(seg) -> int:
        mid = (seg[0]["x0"] + seg[-1]["x1"]) / 2
        col = sum(mid > g for g in gutters)
        return (len(gutters) - col) if rtl_page else col

    # 4. emit in reading order; a line that crosses a gutter (title, full-width paragraph) ends a band
    out, band = [], []

    def flush():
        for col in sorted({c for c, _, _ in band}):
            out.extend(t for c, _, t in band if c == col)
            out.append("")
        band.clear()

    for row in rows:
        segs = row["segments"][::-1] if rtl_page else row["segments"]
        if gutters and not any(any(s[0]["x0"] < g < s[-1]["x1"] for g in gutters) for s in segs):
            for s in segs:
                band.append((column_of(s), row["y"], segment_text(s)))
        else:
            flush()
            out.append("   ".join(segment_text(s) for s in segs))
    flush()
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return _fix_brackets(text) if rtl_page else text


def pdfminer_pages(path: Path) -> list[str]:
    return [page_text_from_layout(layout) for layout in extract_pages(str(path), laparams=LAParams())]


def pypdf_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")  # many "protected" PDFs open with an empty password
    return [page.extract_text() or "" for page in reader.pages]


# ---------------------------------------------------------------- cleaned PDFs (structure-aware reading)
# PDFs rebuilt by notebooks/clean_rag_pdfs_no_ai.ipynb (Chromium output, one template) have reliable
# structure: the title, section and sub-section headings each use their own font size. Reading that
# structure lets every chunk stay inside one section and carry its heading path (see 2.2).
STRUCTURED_SIZES = [(21.0, 1), (16.5, 2), (13.5, 3), (12.4, 4)]  # font size (pt) -> heading level


CONTENTS_TITLES = {"المحتويات", "الفهرس", "contents", "table of contents"}


def is_structured_pdf(path: Path) -> bool:
    """True for PDFs written by the cleaning notebook (Chromium / Skia)."""
    try:
        meta = PdfReader(str(path)).metadata or {}
    except Exception:
        return False
    return "skia" in str(meta.get("/Producer", "")).lower() and "chrom" in str(meta.get("/Creator", "")).lower()


def _fix_number_marker(text: str) -> str:
    """'.2 ميعاد الزراعة' -> '2. ميعاد الزراعة', '-1 الزراعة اليدوية' -> '1- الزراعة اليدوية',
    and on Arabic lines '%97' -> '97%' (the % is read on the wrong side of a right-to-left number)."""
    text = re.sub(r"^\.(\d{1,3})\s+", r"\1. ", text)
    text = re.sub(r"^-(\d{1,3})\s+", r"\1- ", text)
    if is_rtl(text):
        text = re.sub(r"%\s?(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)", r"\1%", text)
    return text


def structured_pages(path: Path) -> list[str]:
    """Page texts of a cleaned PDF: '# title', '## section', '### sub-section' lines, one paragraph or
    list item per block (blank line between blocks). The contents list at the start is skipped."""
    pages, in_contents, pitches = [], False, []
    layouts = list(extract_pages(str(path), laparams=LAParams()))
    page_rows = []
    for layout in layouts:
        rows = []
        for line in _text_lines(layout):
            sizes = [round(c.size, 1) for c in line if isinstance(c, LTChar)]
            text = re.sub(r"\s+", " ", fragment_text(line)).strip()
            if not sizes or not text:
                continue
            size = Counter(sizes).most_common(1)[0][0]
            level = next((lvl for limit, lvl in STRUCTURED_SIZES if size >= limit), 0)
            rows.append({"top": line.y1, "level": level, "text": text})
        rows.sort(key=lambda r: -r["top"])
        pitches += [a["top"] - b["top"] for a, b in zip(rows, rows[1:]) if a["level"] == b["level"] == 0]
        page_rows.append(rows)
    # body line pitch inside a paragraph; a bigger step means a new paragraph or list item
    pitch = Counter(round(p) for p in pitches if p > 0).most_common(1)[0][0] if pitches else 0
    for rows in page_rows:
        blocks, prev = [], None
        for r in rows:
            text = _fix_number_marker(r["text"])
            if in_contents and (r["level"] or not re.match(r"^\s*(\d+[.\-]|…)", text)):
                in_contents = False           # contents entries are "N. title" lines
            if r["level"]:
                if unicodedata.normalize("NFKC", text).strip(" :").lower() in CONTENTS_TITLES:
                    in_contents = True
                    continue
                same = prev is not None and prev["level"] == r["level"] and prev["top"] - r["top"] < 2.2 * pitch
                if same and blocks:           # a heading wrapped over two lines
                    blocks[-1] += " " + text
                else:
                    blocks.append("#" * r["level"] + " " + text)
            elif in_contents:
                continue
            else:
                new_block = (prev is None or prev["level"] != 0 or not pitch
                             or prev["top"] - r["top"] > 1.1 * pitch)
                if new_block or not blocks:
                    blocks.append(text)
                else:
                    blocks[-1] += " " + text
            prev = r
        pages.append("\n\n".join(blocks))
    return pages


PERSIAN_TO_ARABIC = str.maketrans({"ی": "ي", "ک": "ك", "ھ": "ه", "ۀ": "ه", "ہ": "ه"})


PAGE_NUMBER_RE = re.compile(r"^\s*(page|صفحة|ص\.?)?\s*\d{1,4}(\s*(of|/|من)\s*\d{1,4})?\s*$", re.IGNORECASE)


# Frequent Arabic words: if mostly their mirror images appear, the page was extracted backwards
_AR_COMMON = {"في", "من", "على", "إلى", "الى", "عن", "أن", "التي", "الذي", "هذا", "هذه", "مع", "أو", "ذلك", "كما"}


_AR_COMMON_REVERSED = {w[::-1] for w in _AR_COMMON}


_ARABIC_WORD_RE = re.compile(r"[\u0621-\u064A]+")


def looks_reversed(text: str) -> bool:
    words = _ARABIC_WORD_RE.findall(text)
    normal = sum(w in _AR_COMMON for w in words) + sum(w.startswith("ال") and len(w) > 3 for w in words)
    backwards = sum(w in _AR_COMMON_REVERSED for w in words) + sum(w.endswith("لا") and len(w) > 3 for w in words)
    return backwards >= 5 and backwards > 2 * normal


def unreverse_line(line: str) -> str:
    """Flip a visually-ordered RTL line back to reading order, keeping LTR runs (numbers, Latin) intact."""
    if not _ARABIC_WORD_RE.search(line):
        return line
    return _LTR_RUNS_RE.sub(lambda m: m.group(0)[::-1], line[::-1])


def normalise_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


BROKEN_WORD_RE = re.compile(r"[\u0600-\u06ff\ufffd]*\ufffd[\u0600-\u06ff\ufffd]*")


def repair_word(match: re.Match) -> str:
    """An Arabic word with undecodable glyphs ("�", U+FFFD) -> the most likely real word.

    Some bulletins' fonts can't be decoded: elongation (kashida) glyphs come out as "��"
    (الأراض��ى), and ص / ض / ش / س drawn as two pieces come out as "�" + س or ص (م�سر = مصر,
    الن�سج = النضج, للإر�ساد = للإرشاد). The elongation marks are dropped; for the letters, every
    candidate is tried and the one that makes the most common real word (wordfreq) is kept.
    """
    word = re.sub(r"\ufffd{2,}", "", match.group(0))
    pieces = re.split(r"\ufffd[سصضش]", word)
    if 1 < len(pieces) <= 4:
        options = ["".join(p + l for p, l in zip(pieces, combo + ("",))).replace("\ufffd", "")
                   for combo in itertools.product("سصضش", repeat=len(pieces) - 1)]
        best = max(options, key=lambda o: zipf_frequency(o, "ar"))
        if zipf_frequency(best, "ar") > 0:
            return best
    return word.replace("\ufffd", "")


def repair_broken_glyphs(text: str) -> str:
    return BROKEN_WORD_RE.sub(repair_word, text) if "\ufffd" in text else text


_STRAY_RA_WORD_RE = re.compile(r"[\u0621-\u064A]*ر[\u0621-\u064A]*")


_stray_cache: dict[str, str] = {}


def repair_stray_ra(text: str) -> str:
    """Some fonts encode the stretching stroke (kashida) as the letter ر, so justified words gain one:
    الزراعيرة, المررجح, تتزايرد. A word that is not a real word but becomes a common one without one of its
    ر is repaired (الزراعية, المرجح, تتزايد); real words are never touched."""
    def fix(m: re.Match) -> str:
        word = m.group(0)
        if word not in _stray_cache:
            best = word
            if len(word) >= 4 and zipf_frequency(word, "ar") == 0:
                options = {word[:i] + word[i + 1:] for i, ch in enumerate(word) if ch == "ر"}
                good = max(options, key=lambda o: zipf_frequency(o, "ar"), default=word)
                if zipf_frequency(good, "ar") >= 2.0:
                    best = good
            _stray_cache[word] = best
        return _stray_cache[word]
    return _STRAY_RA_WORD_RE.sub(fix, text)


_SWAP_WORD_RE = re.compile(r"[\u0621-\u064A]{3,}")


def swapped_letter_fixes(texts: list[str]) -> dict[str, str]:
    """Word repairs for one document whose font wrote some letters with the wrong glyph.
    - 'اإستخدام', 'اأو' (an alef in front of إ / أ) -> 'إستخدام', 'أو': never a real word, fixed everywhere.
    - one bulletin draws ص, ش and ض with the glyph of س (المحسول, انتسار, اخسر). That is only repaired
      when it happens 20+ times in the document, so a rare real word elsewhere is never 'corrected'."""
    alef, swaps, words = {}, {}, Counter(w for t in texts for w in _SWAP_WORD_RE.findall(t))
    for w, n in words.items():
        if zipf_frequency(w, "ar") > 0:
            continue
        if w[:2] in ("اإ", "اأ") and zipf_frequency(w[1:], "ar") >= 2.0:
            alef[w] = w[1:]
        elif "س" in w:
            options = [w[:i] + new + w[i + 1:] for i, ch in enumerate(w) if ch == "س" for new in "صشض"]
            best = max(options, key=lambda o: zipf_frequency(o, "ar"))
            if zipf_frequency(best, "ar") >= 2.5:
                swaps[w] = best
    if sum(words[w] for w in swaps) < 20:
        swaps = {}
    return {**alef, **swaps}


def apply_word_fixes(text: str, fixes: dict[str, str]) -> str:
    return _SWAP_WORD_RE.sub(lambda m: fixes.get(m.group(0), m.group(0)), text) if fixes else text


def prepare_page(text: str) -> tuple[str, bool]:
    """Unicode + Arabic normalisation, done before anything compares lines.
    Returns (text, whether the page had to be un-reversed)."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")  # ligatures, Arabic forms, soft hyphens
    text = re.sub(r"\(cid:\d+\)\s?", "", text)                          # glyphs with no Unicode (bullets, ticks)
    text = repair_broken_glyphs(text)                                 # الأراض��ى, م�سر -> الأراضى, مصر
    text = repair_stray_ra(text)                                      # الزراعيرة -> الزراعية
    text = strip_arabic_marks(text)                                   # harakat + tatweel
    text = text.translate(PERSIAN_TO_ARABIC)                          # ی ک ھ -> ي ك ه
    reversed_page = looks_reversed(text)
    if reversed_page:
        text = "\n".join(unreverse_line(l) for l in text.splitlines())
    return text, reversed_page


def find_repeated_lines(page_texts: list[str], min_share: float = 0.5) -> set[str]:
    """Short lines that appear on at least `min_share` of a document's pages (headers/footers)."""
    if len(page_texts) < 4:
        return set()
    counts = Counter()
    for text in page_texts:
        # heading lines ("## التسميد") of the cleaned PDFs are structure, never page furniture
        counts.update({normalise_line(l) for l in text.splitlines()
                       if 0 < len(l.strip()) <= 80 and not l.startswith("#")})
    return {line for line, n in counts.items() if n / len(page_texts) >= min_share}


LEADER_RE = re.compile(r"(?:[ \t]*[.…_][ \t]?){5,}[ \t]*\d{0,4}")  # "..........  27" (contents dots + page)


def clean_page(text: str, repeated: set[str]) -> str:
    """`text` has already been through prepare_page()."""
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)                   # irriga-\ntion -> irrigation
    lines = text.splitlines()
    # A table of contents (5+ lines with dot leaders) is only titles and page numbers: drop those
    # lines, or every section title would match keyword searches. Elsewhere just remove the dots.
    toc = sum(bool(LEADER_RE.search(l)) for l in lines) >= 5
    kept = []
    for line in lines:
        if PAGE_NUMBER_RE.match(line) or normalise_line(line) in repeated or (toc and LEADER_RE.search(line)):
            continue
        line = LEADER_RE.sub(" ", line)
        kept.append(re.sub(r"[ \t]+", " ", line).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


MIN_PAGE_VALID_WORDS = 0.80   # pages with less than this share of real words are skipped


# Cleaned PDFs hold real text already; only a page that is mostly broken font is skipped (a page of
# scientific names, e.g. Cuscuta / Isoproturon, scores 0.7 and must stay)
MIN_PAGE_VALID_WORDS_CLEAN = 0.60


_AR_WORD_RE, _EN_WORD_RE = re.compile("[\u0621-\u064A]{2,}"), re.compile("[A-Za-z]{3,}")


_word_cache = {}


def valid_word_share(text: str) -> tuple[float, int]:
    words = [(w, "ar") for w in _AR_WORD_RE.findall(text)] + [(w.lower(), "en") for w in _EN_WORD_RE.findall(text)]
    for w in words:
        if w not in _word_cache:
            _word_cache[w] = zipf_frequency(*w) > 0
    return (sum(_word_cache[w] for w in words) / len(words) if words else 1.0), len(words)


BULLET_RE = re.compile(r"^([•▪●◦\-–*]|\d{1,2}[.)\-]|[a-zأ-ي][.)\-])\s+")


# Split after . ! ? ؟ when the next word does not start with a lower-case Latin letter
# (Arabic has no capitals, so "next char is uppercase" would never split Arabic text).
SENTENCE_RE = re.compile(r"(?<=[.!?؟])\s+(?=[^\sa-z])")


def split_into_units(text: str) -> list[str]:
    """Paragraphs -> units. Wrapped lines are joined; headings and list items stay separate."""
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        buffer = ""
        for line in (l.strip() for l in paragraph.split("\n")):
            if not line:
                continue
            last_line_short = bool(buffer) and len(buffer.rsplit("\n", 1)[-1]) < 60
            if buffer and (BULLET_RE.match(line) or last_line_short):
                units.append(buffer)
                buffer = line
            else:
                buffer = f"{buffer} {line}".strip()
        if buffer:
            units.append(buffer)
    return units


def split_sentences(unit: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(unit) if s.strip()]


def hard_split(sentence: str, size: int) -> list[str]:
    """Only for a single 'sentence' longer than a whole chunk (e.g. a flattened table)."""
    pieces, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return pieces + ([current] if current else [])


def document_sentences(pages: list[dict], size: int) -> list[dict]:
    out = []
    for p in pages:
        for unit in split_into_units(p["text"]):
            for i, sentence in enumerate(split_sentences(unit)):
                for j, piece in enumerate(hard_split(sentence, size) if len(sentence) > size else [sentence]):
                    out.append({"text": piece, "page": p["page"], "new_unit": i == 0 and j == 0})
    return out


def pack(sentences: list[dict], size: int, overlap: int) -> list[list[dict]]:
    chunks, current, length = [], [], 0
    for s in sentences:
        n = len(s["text"]) + 1
        if current and length + n > size:
            chunks.append(current)
            # carry over whole trailing sentences: at least `overlap` chars, at most half a chunk
            keep, kept = [], 0
            for prev in reversed(current):
                if kept >= overlap or kept + len(prev["text"]) + 1 > size // 2:
                    break
                keep.insert(0, prev)
                kept += len(prev["text"]) + 1
            current, length = keep, kept
            while current and length + n > size:  # make room for the new sentence
                length -= len(current.pop(0)["text"]) + 1
        current.append(s)
        length += n
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------- section-aware chunks (cleaned PDFs)
HEADING_RE = re.compile(r"^(#{1,4}) (.+)$")


def clean_heading(text: str) -> str:
    """'10. 5. التسميد :-' -> 'التسميد' (numbering and trailing colons are layout, not meaning)."""
    text = re.sub(r"\(\d+:[a-z]+\)\s*", "", text)   # '(0:dic)' left by some source PDFs
    for _ in range(2):
        text = re.sub(r"^\s*\d{1,3}\s*[.\-)]\s*", "", text)
    return text.strip(" :-–").strip()


def is_structured_doc(pages: list[dict]) -> bool:
    return any(HEADING_RE.match(line) for p in pages for line in p["text"].splitlines())


def structured_sentences(pages: list[dict], size: int) -> tuple[str, list[dict]]:
    """(document title, sentences) where every sentence knows its section path.
    A section is the document's highest heading level below the title (## in most documents)."""
    levels = [len(m.group(1)) for p in pages for line in p["text"].splitlines()
              if (m := HEADING_RE.match(line)) and len(m.group(1)) > 1]
    top = min(levels, default=2)
    title, path, section_no, out = "", {}, 0, []
    for p in pages:
        for block in re.split(r"\n\s*\n", p["text"]):
            block = " ".join(l.strip() for l in block.splitlines() if l.strip())
            if not block:
                continue
            m = HEADING_RE.match(block)
            if m:
                level, text = len(m.group(1)), clean_heading(m.group(2))
                if level == 1:
                    title = title or text
                    continue
                path = {k: v for k, v in path.items() if k < level}
                path[level] = text
                section_no += level == top
                # the heading stays in the text as its own line: it names what the next lines are about
                out.append({"text": text, "page": p["page"], "new_unit": True,
                            "section": (section_no, path.get(top, "")), "path": dict(path)})
                continue
            for i, sentence in enumerate(split_sentences(block)):
                pieces = hard_split(sentence, size) if len(sentence) > size else [sentence]
                for j, piece in enumerate(pieces):
                    out.append({"text": piece, "page": p["page"], "new_unit": i == 0 and j == 0,
                                "section": (section_no, path.get(top, "")), "path": dict(path)})
    return title, out


def chunk_header(title: str, path: dict) -> str:
    """'زراعة وانتاج زهرة الشمس › التسميد › ثانيا: التسميد في الأراضي الجديدة'"""
    parts = [title] + [path[k] for k in sorted(path)]
    seen, kept = set(), []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            kept.append(part)
    return " › ".join(kept)


def _shingles(text: str, n: int = 8) -> set[int]:
    words = re.findall(r"\w+", normalize_for_match(text))
    return {hash(" ".join(words[i:i + n])) for i in range(max(len(words) - n + 1, 1))}


# ================================================================ end of the copied part


class ExtractionError(ValueError):
    """The file can't be read as text (unsupported type, scanned PDF, damaged file...)."""


@dataclass
class ExtractedDocument:
    filename: str
    title: str
    pages: list[dict]                     # [{"page", "text", "lang"}] after cleaning
    raw_pages: int                        # pages in the file
    structured: bool                      # has '#' headings (cleaned PDF, Word headings, Markdown)
    language: str
    skipped_pages: int = 0                # blank, scanned or unreadable pages
    sections: list[str] = field(default_factory=list)   # top-level section headings, in order


# ---------------------------------------------------------------- reading
def docx_pages(path: Path) -> list[dict]:
    """Word file -> pseudo-pages. Title / Heading 1-3 styles become '#' / '##' / '###' / '####' lines."""
    from docx import Document  # python-docx

    levels = {"title": 1, "heading 1": 2, "heading 2": 3, "heading 3": 4}
    blocks: list[str] = []
    doc = Document(str(path))
    for p in doc.paragraphs:
        text = re.sub(r"\s+", " ", p.text).strip()
        if not text:
            continue
        style = (p.style.name if p.style is not None else "").lower()
        level = levels.get(style)
        blocks.append(f"{'#' * level} {text}" if level else text)
    for table in doc.tables:  # tables after the text: one row per line, cells separated by ' | '
        for row in table.rows:
            cells = [re.sub(r"\s+", " ", c.text).strip() for c in row.cells]
            cells = [c for i, c in enumerate(cells) if c and c not in cells[:i]]  # merged cells repeat
            if cells:
                blocks.append(" | ".join(cells))
    pages, current = [], ""
    for block in blocks:
        if current and len(current) + len(block) > DOCX_PAGE_CHARS and not block.startswith("#"):
            pages.append(current)
            current = ""
        current = f"{current}\n\n{block}" if current else block
    if current:
        pages.append(current)
    return [{"page": i, "text": t} for i, t in enumerate(pages, 1)]


def text_pages(text: str) -> list[dict]:
    """Plain text / Markdown -> pseudo-pages of about DOCX_PAGE_CHARS characters (split at blank lines)."""
    text = re.sub(r"(?m)^[ \t]*(#{1,4} \S.*)$", r"\n\1\n", text.replace("\r\n", "\n"))  # headings: own blocks
    pages, current = [], ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if current and len(current) + len(block) > DOCX_PAGE_CHARS:
            pages.append(current)
            current = ""
        current = f"{current}\n\n{block}" if current else block
    if current:
        pages.append(current)
    return [{"page": i, "text": t} for i, t in enumerate(pages, 1)]


def read_pages(path: Path) -> list[dict]:
    """[{'page': n, 'text': ...}] straight from the file (1-based page numbers)."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ExtractionError(f"Unsupported file type '{suffix}'. Upload a PDF, Word (.docx), .txt or .md file.")
    try:
        if suffix == ".pdf":
            if is_structured_pdf(path):  # a PDF made by clean_rag_pdfs_no_ai.ipynb: keep its headings
                return [{"page": i, "text": t} for i, t in enumerate(structured_pages(path), 1)]
            try:
                texts = pdfminer_pages(path)
            except Exception:  # damaged or unusual PDF: fall back to pypdf
                texts = pypdf_pages(path)
            return [{"page": i, "text": t} for i, t in enumerate(texts, 1)]
        if suffix == ".docx":
            return docx_pages(path)
        return text_pages(path.read_text(encoding="utf-8", errors="ignore"))
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Could not read the file: {exc}") from exc


def pdf_title(path: Path) -> str:
    try:
        title = str((PdfReader(str(path)).metadata or {}).get("/Title") or "").strip()
    except Exception:
        return ""
    # ignore generator defaults and file-name titles ("Microsoft Word - x.docx", "untitled", "AGRI_021")
    if not title or re.search(r"untitled|microsoft|\.\w{2,5}$|^[\w\-]+$", title, re.IGNORECASE):
        return ""
    return title[:150]


def clean_pages(raw: list[dict], structured: bool) -> tuple[list[dict], int]:
    """The pipeline's cleaning (section 2.2 of the notebook) for one document: (pages, skipped pages)."""
    min_valid = MIN_PAGE_VALID_WORDS_CLEAN if structured else MIN_PAGE_VALID_WORDS
    prepared = [prepare_page(p["text"]) for p in raw]
    fixes = swapped_letter_fixes([text for text, _ in prepared])
    if fixes:
        prepared = [(apply_word_fixes(text, fixes), was_reversed) for text, was_reversed in prepared]
    repeated = find_repeated_lines([text for text, _ in prepared])
    pages, skipped = [], 0
    for p, (prepared_text, _) in zip(raw, prepared):
        text = clean_page(prepared_text, repeated)
        if len(text) < OCR_THRESHOLD:
            skipped += 1
            continue
        share, n_words = valid_word_share(text)
        if n_words >= 40 and share < min_valid:
            skipped += 1
            continue
        pages.append({"page": p["page"], "text": text, "lang": detect_language(text)})
    return pages, skipped


def read_document(path: Path, filename: str | None = None) -> ExtractedDocument:
    """Read and clean one file. Raises ExtractionError when there is no usable text."""
    filename = filename or path.name
    raw = read_pages(path)
    structured = any(HEADING_RE.match(line) for p in raw for line in p["text"].splitlines())
    pages, skipped = clean_pages(raw, structured)
    text = "\n".join(p["text"] for p in pages)
    if len(text.strip()) < 200:
        if path.suffix.lower() == ".pdf" and raw:
            raise ExtractionError("This PDF has no readable text (it is probably a scan). Run OCR on it first, "
                                  "for example: ocrmypdf -l ara+eng input.pdf output.pdf")
        raise ExtractionError("The file has almost no readable text.")
    title = ""
    sections: list[str] = []
    if structured:
        title, sentences = structured_sentences(pages, CHUNK_SIZE)
        sections = list(dict.fromkeys(s["section"][1] for s in sentences if s["section"][1]))
    if not title and path.suffix.lower() == ".pdf":
        title = pdf_title(path)
    if not title:
        title = re.sub(r"[_\-]+", " ", Path(filename).stem).strip() or filename
    return ExtractedDocument(filename=filename, title=title, pages=pages, raw_pages=len(raw), structured=structured,
                             language=detect_language(text), skipped_pages=skipped, sections=sections[:40])


# ---------------------------------------------------------------- chunking
def chunk_document(doc: ExtractedDocument, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """The pipeline's chunker for one document: [{"id", "text", "metadata"}].

    Documents with headings are packed section by section and every chunk starts with its heading path
    ('title › section › sub-section'); other documents are one stream of sentences. Repeated sentences
    and near-duplicate chunks are dropped.
    """
    pages, source = doc.pages, doc.filename
    structured = doc.structured and is_structured_doc(pages)
    if structured:
        title, doc_sentences = structured_sentences(pages, size)
    else:
        title, doc_sentences = "", document_sentences(pages, size)
    title = doc.title or title
    seen, sentences = set(), []
    for s in doc_sentences:
        key = re.sub(r"\W+", " ", normalize_for_match(s["text"])).strip()
        if len(key) >= 40:
            if key in seen:
                continue
            seen.add(key)
        sentences.append(s)
    groups = [list(g) for _, g in itertools.groupby(sentences, key=lambda s: s.get("section"))] \
        if structured else [sentences]
    chunks, shingles_seen = [], set()
    for group in groups:
        if not group:
            continue
        header = chunk_header(title, group[0]["path"]) if structured else ""
        budget = max(size - len(header) - 1, size // 2)
        for packed in pack(group, budget, overlap):
            if structured:
                header = chunk_header(title, packed[0]["path"])
                if packed[0]["text"] == header.rsplit(" › ", 1)[-1]:
                    packed = packed[1:]
                if not packed:
                    continue
            body = ""
            for k, s in enumerate(packed):
                sep = "" if k == 0 else ("\n" if s["new_unit"] else " ")
                body += sep + s["text"]
            if len(body) < MIN_CHUNK_CHARS:
                continue
            sh = _shingles(body)
            if sh and len(sh & shingles_seen) / len(sh) >= 0.8:  # mostly a repeat of an earlier chunk
                continue
            shingles_seen |= sh
            idx = len(chunks)
            chunks.append({
                "id": f"chunk-{idx:04d}",
                "text": f"{header}\n{body}" if header else body,
                "metadata": {
                    "source": source,
                    "page_start": min(s["page"] for s in packed),
                    "page_end": max(s["page"] for s in packed),
                    "chunk_index": idx,
                    "lang": detect_language(body),
                    "title": title,
                    "section": packed[0]["section"][1] if structured else "",
                },
            })
    return chunks
