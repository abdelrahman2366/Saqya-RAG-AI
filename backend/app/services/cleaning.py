"""Clean an uploaded document the way notebooks/clean_rag_pdfs_no_ai.ipynb cleans the corpus — no AI.

A farmer's PDF is rebuilt from its glyphs (reversed Arabic, broken لا ligatures, presentation forms,
displaced numbers fixed), its useless parts are removed (cover and credits pages, ministry headers,
ISBN / copyright lines, contents pages, running headers and footers, page numbers, prefaces,
acknowledgments, references, back covers), and the rest is organised into a title, numbered sections,
paragraphs, lists and one line per table row. documents.py then:

- writes that clean version as a new PDF in the session's temporary folder (the notebook's look:
  title, summary, contents, DejaVu Sans), which the farmer can download, and
- indexes exactly that clean text — section by section, each passage knowing the page it came from in
  the original file — instead of the raw file.

Everything between the two "copied part" markers is the notebook's engine, generated from its cells by
a small script; if you change those notebook cells, copy the change here as well.
"""

from __future__ import annotations

import bisect
import difflib
import html
import importlib.util
import logging
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.utils.text import detect_language

logger = logging.getLogger(__name__)


# ================================================================ copied from the notebook (section 3)
AR_CHAR = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LTR_CHAR = re.compile(r"[A-Za-z0-9À-ɏ٠-٩۰-۹]")
DIGIT = re.compile(r"[0-9٠-٩۰-۹]")
HARAKA_ONLY = re.compile(r"^[ً-ٰٟۖ-ۭؐ-ؚ]+$")
NUM_ATTACH = set("%٪°‰+#$€£")
MIRROR = str.maketrans("()[]{}«»", ")(][}{»«")
PH_BASE = 0x100000                         # private-use placeholders for glyphs without a Unicode mapping


def is_ph(c):
    return ord(c) >= PH_BASE


class Glyph:
    __slots__ = ("x0", "x1", "y", "t", "size", "bold", "font")

    def __init__(self, x0, x1, y, t, size, bold, font):
        self.x0, self.x1, self.y, self.t, self.size, self.bold, self.font = x0, x1, y, t, size, bold, font


def page_glyphs(page, unknown=None):
    """Glyphs in drawing order. Ligature components (gid == -1) stay attached to their glyph,
    so a لا ligature keeps its logical letter order."""
    vis, inv = [], []
    for tr in page.get_texttrace():
        d = tr["dir"]
        if abs(d[1]) > 0.1 or d[0] < 0:            # rotated / vertical text
            continue
        target = inv if (tr.get("type") == 3 or tr.get("opacity", 1) == 0) else vis
        font, size = tr["font"], tr["size"]
        bold = bool(tr["flags"] & 16) or bool(re.search(r"bold|black|heavy|semibold|demi", font, re.I))
        for uc, gid, origin, bbox in tr["chars"]:
            ch = chr(uc) if uc > 0 else ""
            if (uc == 0xFFFD or uc <= 0) and gid >= 0 and unknown is not None:
                key = (font, gid)                  # unmapped glyph: placeholder, resolved per document
                if key not in unknown:
                    unknown[key] = chr(PH_BASE + len(unknown))
                ch = unknown[key]
            if gid == -1 and target:
                target[-1].t += ch
                target[-1].x1 = max(target[-1].x1, bbox[2])
                continue
            if not ch or HARAKA_ONLY.match(ch):      # diacritics: dropped (they extract as stray marks)
                continue
            target.append(Glyph(bbox[0], bbox[2], origin[1], ch, size, bold, font))
    out = vis if vis else inv                      # scanned PDFs with an invisible OCR layer
    seen, uniq = set(), []
    for g in out:                                  # same glyph drawn twice (fake bold / shadow)
        k = (g.t, round(g.x0 / max(g.size * 0.25, 0.5)), round(g.y / max(g.size * 0.25, 0.5)))
        if k in seen:
            continue
        seen.add(k); uniq.append(g)
    return uniq


def _cls(t):
    if DIGIT.search(t) and not re.search(r"[^\d\u0660-\u0669\u06F0-\u06F9.,٫٬]", t.strip() or "x"):
        return "L"
    if AR_CHAR.search(t):
        return "R"
    if LTR_CHAR.search(t):
        return "L"
    return "N"


def visual_to_logical(gl):
    """Glyphs sorted left->right on one line -> text in logical (reading) order."""
    seq = []
    for i, g in enumerate(gl):
        if seq and not seq[-1].t.isspace() and not g.t.isspace():
            if g.x0 - gl[i - 1].x1 > 0.22 * max(g.size, 1):          # missing space glyph
                seq.append(Glyph(g.x0, g.x0, g.y, " ", g.size, g.bold, g.font))
        seq.append(g)
    has_ar = any(AR_CHAR.search(g.t) for g in seq)
    cls = [("R" if has_ar else "L") if (g.t and is_ph(g.t[0])) else _cls(g.t) for g in seq]
    for i, g in enumerate(seq):                                      # % ° stick to their number
        s = g.t.strip()
        if cls[i] == "N" and s and s[0] in NUM_ATTACH:
            if (i and DIGIT.search(seq[i - 1].t)) or (i + 1 < len(seq) and DIGIT.search(seq[i + 1].t)):
                cls[i] = "L"
    n_r = cls.count("R")
    n_l = cls.count("L")
    if n_r == 0 or (n_r < 3 and n_l > 1.5 * n_r):
        return "".join(g.t for g in seq), False                     # left-to-right line
    rev = list(range(len(seq)))[::-1]                               # right-to-left line: reverse glyphs,
    out, i = [], 0                                                  # then restore numbers / Latin runs
    while i < len(rev):
        if cls[rev[i]] == "L":
            j, last = i, i
            while j < len(rev) and cls[rev[j]] != "R":
                if cls[rev[j]] == "L":
                    last = j
                j += 1
            out += rev[i:last + 1][::-1]
            i = last + 1
        else:
            out.append(rev[i])
            i += 1
    return "".join(seq[k].t for k in out), True


def normalize(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ـ", "")                                     # kashida
    s = s.translate(str.maketrans("یکۀ", "يكه"))                    # Persian letter forms
    s = re.sub(r"[ً-ٰٟ]", "", s)                      # diacritics
    s = re.sub(r"[​-‏‪-‮⁦-⁩﻿\x00-\x08\x0b-\x1f]", "", s)
    s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٪", "01234567890123456789.%"))
    s = re.sub(r"[ \t  - ]+", " ", s)
    return s.strip()


_RANGE = re.compile(r"(?<![\d.\-–])(\d+(?:\.\d+)?)\s*([-–])\s*(\d+(?:\.\d+)?)(?!\d|\.\d|\s*[-–]\s*\d)")


def fix_arabic_line(s):
    """Typing conventions of Arabic Word documents."""
    s = re.sub(r"(?<=\d),(?=\d{1,2}(?!\d))", ".", s)                 # 1,5 -> 1.5
    def swap(m):
        a, d, b = m.group(1), m.group(2), m.group(3)
        return f"{b}{d}{a}" if float(a) > float(b) else f"{a}{d}{b}"  # ranges are written high-low
    s = _RANGE.sub(swap, s)
    s = re.sub(r"%\s*(\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?)", r"\1%", s)  # %15 -> 15%
    s = re.sub(r"(?<![\d.])0{3,}(?![\d.])", "…", s)
    s = re.sub(r"(?<=[؀-ۿ)\s])\s*(?<![\d.])0\s*$", ".", s)    # "0" typed as a full stop
    s = re.sub(r"(?<=[؀-ۿ)])\s+0\s+(?=[؀-ۿ(])", ". ", s)
    if s in ("0", "00"):
        s = "."
    return s.strip()


def _family(font):
    f = font.split("+")[-1]
    f = re.split(r"[-,]", f)[0]
    return re.sub(r"(PSMT|PS|MT|Bold|Italic|Regular)+$", "", f, flags=re.I).lower()


def _mode(vals):
    return Counter(vals).most_common(1)[0][0] if vals else None


def glyph_lines(glyphs):
    """Cluster loose glyphs into lines (used for table cells)."""
    gl = sorted(glyphs, key=lambda g: g.y)
    lines, cur = [], []
    for g in gl:
        if cur and abs(g.y - cur[-1].y) > 0.45 * max(g.size, 1):
            lines.append(cur); cur = []
        cur.append(g)
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        ln.sort(key=lambda g: (g.x0 + g.x1) / 2)
        txt, rtl = visual_to_logical(ln)
        txt = normalize(txt)
        if rtl:
            txt = fix_arabic_line(txt)
        if txt:
            out.append((txt, rtl))
    return out


def page_lines(page, glyphs, exclude_rects=()):
    glyphs = sorted(glyphs, key=lambda g: g.y)
    ys = [g.y for g in glyphs]
    used = bytearray(len(glyphs))
    rd = page.get_text("rawdict")
    W, H = page.rect.width, page.rect.height
    blocks = []
    for b in rd["blocks"]:
        if b["type"] != 0:
            continue
        # MuPDF splits justified lines at wide gaps: merge pieces that share a baseline into one visual line
        rows = []
        for l in b["lines"]:
            r = pymupdf.Rect(l["bbox"])
            if any(r.intersects(e) and (r & e).get_area() > 0.5 * r.get_area() for e in exclude_rects):
                continue
            for i, row in enumerate(rows):
                h = min(r.height, row.height) or 1
                if min(r.y1, row.y1) - max(r.y0, row.y0) > 0.6 * h:
                    rows[i] = pymupdf.Rect(min(row.x0, r.x0), min(row.y0, r.y0), max(row.x1, r.x1), max(row.y1, r.y1))
                    break
            else:
                rows.append(r)
        lines = []
        for r in rows:
            x0, y0, x1, y1 = r
            lo, hi = bisect.bisect_left(ys, y0 - 1.5), bisect.bisect_right(ys, y1 + 1.5)
            mem = []
            for k in range(lo, hi):
                if used[k]:
                    continue
                g = glyphs[k]
                if x0 - 1.5 <= (g.x0 + g.x1) / 2 <= x1 + 1.5:
                    mem.append(g); used[k] = 1
            if not mem:
                continue
            mem.sort(key=lambda g: (g.x0 + g.x1) / 2)
            raw, rtl = visual_to_logical(mem)
            text = normalize(raw)
            if rtl:
                text = fix_arabic_line(text)
            if not text:
                continue
            vis = [g for g in mem if g.t.strip()]
            lines.append({
                "text": text, "rtl": rtl, "bbox": (x0, y0, x1, y1),
                "size": round(_mode([round(g.size * 2) / 2 for g in vis]) or 0, 1),
                "bold": sum(g.bold for g in vis) >= 0.6 * max(1, len(vis)),
                "family": _mode([_family(g.font) for g in vis]) or "",
                "W": W, "H": H,
            })
        if lines:
            blocks.append({"bbox": b["bbox"], "lines": lines})
    return blocks


AR_CANDIDATES = ["", "لا", "لأ", "لإ", "لآ"] + list("ابتثجحخدذرزسشصضطظعغفقكلمنهويةىأإآئؤء")
LAT_CANDIDATES = ["", "fi", "fl", "ff", "ffi", "ffl", "-", "'", "•"]
WORD = re.compile(r"[\w\U00100000-\U0010FFFD]+")


def resolve_placeholders(texts, placeholders):
    """Pick, for every unmapped glyph, the letter(s) that turn its words into words used elsewhere in the
    same document ("" for stretching/kashida glyphs, "لا" for unmapped lam-alef ligatures, ...)."""
    vocab, occ = Counter(), {ph: [] for ph in placeholders}
    for t in texts:
        for w in WORD.findall(t):
            phs = [c for c in w if is_ph(c)]
            if not phs:
                vocab[w] += 1
            else:
                for ph in set(phs):
                    if ph in occ and len(occ[ph]) < 400:
                        occ[ph].append(w)
    mapping = {}
    for ph, words in occ.items():
        if not words:
            mapping[ph] = ""; continue
        arabic = sum(1 for w in words if AR_CHAR.search(w)) >= len(words) / 2
        best, best_score = None, -1
        for c in (AR_CANDIDATES if arabic else LAT_CANDIDATES):
            score = 0
            for w in words:
                test = "".join(c if x == ph else ("" if is_ph(x) else x) for x in w)
                score += vocab.get(test, 0) > 0
            if score > best_score:
                best, best_score = c, score
        if best_score < max(2, 0.15 * len(words)):
            mid = sum(1 for w in words if w.find(ph) > 0 and w.find(ph) < len(w) - 1) / len(words)
            best = "" if (mid > 0.6 or not arabic) else "لا"
        mapping[ph] = best
    # words that still are not real words: try every candidate for that word alone
    overrides = {}
    for ph, words in occ.items():
        arabic = sum(1 for w in words if AR_CHAR.search(w)) >= max(1, len(words)) / 2
        for w in set(words):
            plain = "".join(mapping.get(x, "") if is_ph(x) else x for x in w)
            if vocab.get(plain, 0):
                continue
            scored = []
            for c in (AR_CANDIDATES if arabic else LAT_CANDIDATES):
                test = "".join(c if x == ph else (mapping.get(x, "") if is_ph(x) else x) for x in w)
                if vocab.get(test, 0) >= 2:
                    scored.append((vocab[test], test))
            if scored:
                overrides[w] = max(scored)[1]
    return mapping, overrides


def _ruled_segments(page):
    n = 0
    for path in page.get_drawings():
        for it in path["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.y - b.y) < 1 or abs(a.x - b.x) < 1:
                    n += 1
            elif it[0] == "re":
                n += 4
        if n >= 8:
            return True
    return False


def page_tables(page, glyphs):
    """Ruled tables -> list of (bbox, rows[list of cell texts], rtl)."""
    out = []
    try:
        if not _ruled_segments(page):
            return out
        found = page.find_tables()
    except Exception:
        return out
    if not found.tables:
        return out
    gs = sorted(glyphs, key=lambda g: g.y)
    ys = [g.y for g in gs]
    for tab in found.tables:
        rows = []
        for r in tab.rows:
            cells = []
            for c in r.cells:
                if c is None:
                    cells.append(None); continue
                x0, y0, x1, y1 = c
                lo, hi = bisect.bisect_left(ys, y0), bisect.bisect_right(ys, y1 + 4)
                inside = [g for g in gs[lo:hi] if x0 <= (g.x0 + g.x1) / 2 <= x1 and y0 <= g.y - 0.3 * g.size <= y1]
                lines = glyph_lines(inside)
                cells.append(" ".join(t for t, _ in lines))
            rows.append(cells)
        if len(rows) < 2 or max(len(r) for r in rows) < 2:
            continue
        texts = [c for r in rows for c in r if c]
        if not texts or sum(len(t) for t in texts) / len(texts) > 220:
            continue                                   # a page frame, not a data table
        rtl = sum(len(AR_CHAR.findall(t)) for t in texts) > sum(len(re.findall(r"[A-Za-z]", t)) for t in texts)
        if rtl:
            rows = [r[::-1] for r in rows]             # first column = rightmost column
        for ci in range(max(len(r) for r in rows)):   # merged cells: carry the value down
            prev = None
            for r in rows:
                if ci < len(r):
                    if r[ci] is None:
                        r[ci] = prev
                    else:
                        prev = r[ci]
        out.append((pymupdf.Rect(tab.bbox), rows, rtl))
    return out


def table_to_items(rows):
    rows = [[(c or "").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    header = rows[0]
    has_header = all(header) and all(len(h) <= 60 for h in header) and len(rows) > 1
    items = []
    for r in (rows[1:] if has_header else rows):
        vals = [v for v in r if v]
        if len(vals) == 1:
            items.append(vals[0]); continue
        if has_header:
            parts = [f"{h.rstrip(' :')}: {v}" for h, v in zip(header, r) if v]
        else:
            parts = vals
        line = parts[0]
        for prev, part in zip(parts, parts[1:]):
            line += (" " if prev.rstrip().endswith((":", "：")) else " — ") + part
        items.append(line)
    return items


GLOBAL_DROP = re.compile(
    r"(@[\w.-]+\.\w+|https?://|www\.|\bISBN\b|\bISSN\b|\bDOI\b|doi\.org|Corresponding\s+author|©|"
    r"All\s+rights\s+reserved|Copyright|\bReceived\s*:|\bAccepted\s*:|\bPublished\s+online|"
    r"\bTel\s*[:.]|\bFax\s*[:.]|P\.?\s?O\.?\s?Box|تطلب\s*بالمجان|مع\s*تحيات|رقم\s*الإيداع|الترقيم\s*الدولي|"
    r"حقوق\s*الطبع|جميع\s*الحقوق\s*محفوظة)", re.I)
FRONT_KW = re.compile(
    r"([اإ]دارة\s*المركزية|جمهورية\s*مصر|وزارة\s*الزراعة|مركز\s*البحوث|الإدارة\s*المركزية|الادارة\s*المركزية|قطاع\s*الإرشاد|"
    r"المادة\s*العلمية|^\s*إعداد|^\s*اعداد|المشرف\s*العام|رئيس\s*التحرير|مدير\s*التحرير|الإخراج\s*الفني|"
    r"الاخراج\s*الفني|التوزيع|تصميم|مراجعة|ترجمة|معهد\s*بحوث|طبعة|نشرة\s*رقم|^\s*\d{4}\s*/\s*\d{4}\s*$|"
    r"\bAuthors?\b|\bEdited\s+by\b|\bPrepared\s+by\b|\bEdition\b|\bPublisher|\bPrinted\b|University|"
    r"Institute|Department|Faculty|Ministry|Organization|\bCollege\b|Title\s*ID|\bPrice\b|M\.\s?Sc|Ph\.\s?D|B\.\s?Sc|"
    r"reproduced|permission|publishers?\b|trademarks?|Co-?authors?|professor|lecturer|\bassistant\b|"
    r"سكرتارية|التحرير|\.indd|نسخة|document\s+control|research\s+corpus|knowledge\s+base\s*·|بطاقة\s*الإصدار|حدود\s*الاستخدام)", re.I)
NAME_LINE = re.compile(r"^\s*(أ\s?\.?\s?د|ا\s?\.\s?د|د\s?\.|د\s?/|م\s?/|م\s?\.|أ\s?/|Prof\.?|Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Eng\.?)(\s|/|$)")
TOC_HEAD = re.compile(r"^\s*(فهرس|الفهرس|المحتويات|محتويات|Contents|Table\s+of\s+Contents|CONTENTS|Index)\s*:?\s*$", re.I)
TOC_LINE = re.compile(r"(\.{4,}|…{2,}|_{4,}|\s\d{1,3}\s*$|^\s*\d{1,3}\s+\S)")
PAGE_NUM = re.compile(r"^[\s\-–—|]*(page|صفحة)?\s*[\divxlcIVXLC]{1,6}(\s*(/|of|من)\s*\d+)?[\s\-–—|]*$", re.I)
HARD_DROP = re.compile(r"^\s*(\d+[.\-)]?\s*)?(references?|literature\s+cited|bibliography|works\s+cited|المراجع|مراجع|المصادر|المراجع\s+العلمية)\s*:?\s*$", re.I)
DROP_SECTION = re.compile(
    r"^\s*(\d+[.\-)]?\s*)?(references?|literature\s+cited|bibliography|works\s+cited|acknowledge?ments?|"
    r"foreword|preface|المراجع|مراجع|المصادر|المراجع\s+العلمية|شكر\s*و?\s*تقدير|تقديم|إهداء|اهداء|"
    r"contents|table\s+of\s+contents|المحتويات|الفهرس|فهرس|required\s+citation|research\s+and\s+writing\s+team|"
    r"writing\s+team|project\s+team|background\s+papers|additional\s+(external\s+contributions|fao\s+inputs)|"
    r"(fao\s+)?advisory\s+group|steering\s+committee|editorial\s+(board|team)|photo\s+credits|"
    r"فريق\s*(العمل|الإعداد|الصياغة|المشروع)|لجنة\s*التوجيه|هيئة\s*التحرير)\s*:?\s*$", re.I)
BULLET = re.compile(r"^\s*([•●▪■◆◇○◦►▶➢➤✓✔❖∙·\*]|[-–—](?=\s*\S))\s*")
NUMBERED = re.compile(r"^\s*(\(?\d{1,2}[\-.)]|\(\d{1,2}\)|[أابجدهوزحطيكلمن](?:ـ)?\s?[\-)]|[a-hA-H][.)])\s+")
TERMINAL = tuple(".:؟?!؛…")
LEAD_IN = {"الى", "إلى", "في", "فى", "على", "علي", "من", "عن", "مثل", "وهي", "وهى", "هي", "هى", "وهما", "كالتالي",
           "كالتالى", "التالي", "التالية", "الآتي", "الاتي", "الآتية", "يلي", "يلى", "منها", "أهمها", "اهمها"}


SYMS = re.compile(r"[\u00A0-\u00BF\u00C0-\u024F\u2000-\u22FF\u25A0-\u25FF<>\[\]{}^~|\\¬]")
AUTHOR_LINE = re.compile(r"(\b[A-Z]{1,2}[a-z]?\.\s?){2,}")


def is_gibberish(t, arabic_doc):
    """Text from symbol/decorative fonts without a Unicode map."""
    if AR_CHAR.search(t):
        return False
    n = len(t.strip()) or 1
    sym = len(SYMS.findall(t))
    if arabic_doc:
        return sym >= 0.25 * n or bool(re.fullmatch(r"\s*[A-Za-z]{1,3}\s*", t))
    return n > 5 and sym >= 0.4 * n


def _key(text):
    return re.sub(r"[\W\d_]+", "", text.lower())[:50]


def extract_document(path, use_tables=True, log=None):
    doc = pymupdf.open(path)
    n = doc.page_count
    pages, unknown = [], {}
    for pno in range(n):
        page = doc[pno]
        glyphs = page_glyphs(page, unknown)
        tables = page_tables(page, glyphs) if use_tables else []
        blocks = page_lines(page, glyphs, exclude_rects=[t[0] for t in tables])
        items = [{"type": "block", "y": b["bbox"][1], "lines": b["lines"]} for b in blocks]
        items += [{"type": "table", "y": t[0].y0, "rows": t[1], "rtl": t[2]} for t in tables]
        items.sort(key=lambda it: it["y"])
        # keep MuPDF's block order (handles columns) but slot tables in by position
        ordered, tabs = [], [it for it in items if it["type"] == "table"]
        for it in [it for it in items if it["type"] == "block"]:
            while tabs and tabs[0]["y"] <= it["y"]:
                ordered.append(tabs.pop(0))
            ordered.append(it)
        ordered += tabs
        pages.append({"pno": pno, "items": ordered, "W": page.rect.width, "H": page.rect.height})
    meta_title = (doc.metadata or {}).get("title", "") or ""
    doc.close()

    if unknown:                                    # resolve unmapped glyphs from the document's own vocabulary
        texts = [ln["text"] for pg in pages for it in pg["items"] if it["type"] == "block" for ln in it["lines"]]
        texts += [c for pg in pages for it in pg["items"] if it["type"] == "table" for r in it["rows"] for c in r if c]
        mapping, overrides = resolve_placeholders(texts, set(unknown.values()))
        trans = str.maketrans(mapping)

        def fix(t):
            if not t or not any(is_ph(c) for c in t):
                return t
            t = WORD.sub(lambda m: overrides.get(m.group(0), m.group(0)), t)
            return re.sub(r"\s+", " ", t.translate(trans)).strip()
        for pg in pages:
            for it in pg["items"]:
                if it["type"] == "block":
                    for ln in it["lines"]:
                        ln["text"] = fix(ln["text"])
                    it["lines"] = [ln for ln in it["lines"] if ln["text"]]
                else:
                    it["rows"] = [[fix(c) for c in r] for r in it["rows"]]

    # --- running headers / footers / page numbers
    zone_keys = Counter()
    for pg in pages:
        seen = set()
        for it in pg["items"]:
            if it["type"] != "block":
                continue
            for ln in it["lines"]:
                y0, y1 = ln["bbox"][1], ln["bbox"][3]
                if y1 < 0.12 * pg["H"] or y0 > 0.88 * pg["H"]:
                    k = _key(ln["text"])
                    if len(k) >= 3 and k not in seen:
                        seen.add(k); zone_keys[k] += 1
    min_rep = max(3, int(0.2 * n))
    frequent = [k for k, c in zone_keys.items() if c >= min_rep]

    def is_furniture(ln, H):
        y0, y1 = ln["bbox"][1], ln["bbox"][3]
        in_zone = y1 < 0.12 * H or y0 > 0.88 * H
        if (y1 < 0.16 * H or y0 > 0.84 * H) and PAGE_NUM.match(ln["text"]):
            return True
        if in_zone:
            k = _key(ln["text"])
            if k in frequent or any(len(k) >= 6 and difflib.SequenceMatcher(None, k, f).ratio() > 0.8 for f in frequent):
                return True
        return False

    all_text = " ".join(ln["text"] for pg in pages for it in pg["items"] if it["type"] == "block" for ln in it["lines"])
    arabic_doc = len(AR_CHAR.findall(all_text)) > len(re.findall(r"[A-Za-z]", all_text))
    for pg in pages:
        for it in pg["items"]:
            if it["type"] == "block":
                it["lines"] = [ln for ln in it["lines"] if not is_furniture(ln, pg["H"])
                               and not GLOBAL_DROP.search(ln["text"]) and not is_gibberish(ln["text"], arabic_doc)
                               and not re.search(r"(\.{6,}|…{3,}|_{6,})\s*\d{0,4}\s*$|^\s*\d{0,4}\s*(\.{6,}|…{3,})", ln["text"])]

    # --- title candidates from the first pages (largest text that is not front-matter boilerplate)
    def title_from(pg_list):
        cands = []
        for pg in pg_list:
            for it in pg["items"]:
                if it["type"] != "block":
                    continue
                for ln in it["lines"]:
                    t = ln["text"]
                    if (len(_key(t)) >= 3 and not FRONT_KW.search(t) and not NAME_LINE.match(t) and not TOC_HEAD.match(t)
                            and not re.fullmatch(r"[\d\s/\-.]+", t) and len(t) <= 120 and not AUTHOR_LINE.search(t)):
                        cands.append((ln["size"], pg["pno"], ln["bbox"][1], t))
        if not cands:
            return ""
        big = max(c[0] for c in cands)
        first = min(c[1] for c in cands if c[0] == big)
        parts, seen = [], set()
        for c in sorted(cands, key=lambda c: (c[1], c[2])):
            if c[1] == first and c[0] >= 0.6 * big and c[3] not in seen and c[0] > 0:
                seen.add(c[3]); parts.append(c[3])
        t = " ".join(parts[:4])
        if len(t) > 220:
            t = t[:220].rsplit(" ", 1)[0] + " …"
        return t

    # --- front matter pages (cover, credits, TOC) and back cover
    def page_stats(pg):
        lines = [ln for it in pg["items"] if it["type"] == "block" for ln in it["lines"]]
        chars = sum(len(ln["text"]) for ln in lines) + sum(len(c or "") for it in pg["items"] if it["type"] == "table" for r in it["rows"] for c in r)
        fm = sum(1 for ln in lines if FRONT_KW.search(ln["text"]) or NAME_LINE.match(ln["text"]))
        toc = sum(1 for ln in lines if TOC_LINE.search(ln["text"]))
        has_toc_head = any(TOC_HEAD.match(ln["text"]) for ln in lines[:6])
        return {"chars": chars, "lines": len(lines), "fm": fm / max(1, len(lines)),
                "toc": has_toc_head and toc >= 3 or (len(lines) >= 6 and toc / len(lines) > 0.6)}

    stats = [page_stats(pg) for pg in pages]
    drop = set()
    for i in range(min(n, 8)):                                        # leading run
        s = stats[i]
        if s["chars"] < 250 or s["fm"] >= 0.4 or s["toc"]:
            drop.add(i)
        else:
            break
    for i in range(min(n, 10)):                                       # credits / TOC pages further in
        s = stats[i]
        if s["fm"] >= 0.5 or s["toc"]:
            drop.add(i)
    for i in range(n - 1, max(n - 4, -1), -1):                         # back cover
        s = stats[i]
        if s["chars"] < 250 or s["fm"] >= 0.4:
            drop.add(i)
        else:
            break
    if len(drop) >= n:                                                # never drop everything
        drop = set()
    title = title_from(pages[:2] if 0 in drop else pages[:1])

    content = [pg for pg in pages if pg["pno"] not in drop]

    # research papers: on the first content page, drop author/affiliation lines above "Abstract"
    if content:
        pg = content[0]
        flat = [(it, ln) for it in pg["items"] if it["type"] == "block" for ln in it["lines"]]
        abs_i = next((k for k, (_, ln) in enumerate(flat) if re.match(r"^\s*(abstract|summary|الملخص|ملخص|المستخلص)\b", ln["text"], re.I)), None)
        if abs_i is not None and abs_i > 0:
            above = flat[:abs_i]
            big = max(ln["size"] for _, ln in above)
            keep_title = [ln["text"] for _, ln in above if ln["size"] >= big - 0.6]
            if keep_title and not title:
                title = " ".join(keep_title)
            for it, ln in above:
                it["lines"] = [x for x in it["lines"] if x is not ln]

    if content:                                    # author / affiliation lines on the first content page
        lim = 80 if arabic_doc else 160
        for it in content[0]["items"]:
            if it["type"] == "block":
                it["lines"] = [ln for ln in it["lines"] if not (len(ln["text"]) < lim and (
                    FRONT_KW.search(ln["text"]) or NAME_LINE.match(ln["text"]) or
                    (not arabic_doc and AUTHOR_LINE.search(ln["text"]) and len(re.findall(r"\b[a-z]{3,}\b", ln["text"])) <= 4)))]

    # --- body style
    style_chars = Counter()
    for pg in content:
        for it in pg["items"]:
            if it["type"] == "block":
                for ln in it["lines"]:
                    style_chars[(ln["family"], ln["size"], ln["bold"])] += len(ln["text"])
    body_style = style_chars.most_common(1)[0][0] if style_chars else ("", 0, False)
    body_size = body_style[1]

    # --- paragraphs
    # typical line pitch per font size (baseline-to-baseline distance inside running text)
    pitch = {}
    for pg in content:
        for it in pg["items"]:
            if it["type"] != "block":
                continue
            ls = it["lines"]
            for a, b in zip(ls, ls[1:]):
                if a["size"] == b["size"]:
                    d = b["bbox"][1] - a["bbox"][1]
                    if 0 < d < 4 * max(a["size"], 1):
                        pitch.setdefault(a["size"], []).append(d)
    pitch = {k: sorted(v)[len(v) // 2] for k, v in pitch.items() if len(v) >= 3}

    def head_like(t):
        return bool(re.search(r"(:-?|-:)\s*$", t)) and len(t.split()) <= 7

    paras = []   # dict(kind, text, style, rtl, lines, page)
    for pg in content:
        cur = None
        for it in pg["items"]:
            if it["type"] == "table":
                paras.append({"kind": "table", "rows": it["rows"], "rtl": it["rtl"], "page": pg["pno"]})
                cur = None
                continue
            for ln in it["lines"]:
                style = (ln["family"], ln["size"], ln["bold"])
                start = cur is None or cur.get("closed")
                if not start:
                    last = cur["lines"][-1]
                    lh = max(1.0, last["bbox"][3] - last["bbox"][1])
                    step = ln["bbox"][1] - last["bbox"][1]
                    normal = pitch.get(last["size"], 1.25 * lh)
                    width_max = max(l["bbox"][2] - l["bbox"][0] for l in cur["lines"])
                    last_w = last["bbox"][2] - last["bbox"][0]
                    ov = min(ln["bbox"][2], last["bbox"][2]) - max(ln["bbox"][0], last["bbox"][0])
                    if (style != cur["style"] or step > 1.35 * normal or step < 0.3 * lh
                            or BULLET.match(ln["text"]) or NUMBERED.match(ln["text"])
                            or (last["text"].endswith(TERMINAL) and (last_w < 0.85 * width_max or head_like(ln["text"])))
                            or ov < 0.3 * min(last_w, ln["bbox"][2] - ln["bbox"][0])):
                        start = True
                if ln["text"] in (".", "…") and cur is not None:
                    cur["lines"].append(ln)                     # a lone full stop belongs to the previous line
                    continue
                if start:
                    cur = {"kind": "p", "lines": [ln], "style": style, "rtl": ln["rtl"], "page": pg["pno"]}
                    paras.append(cur)
                else:
                    cur["lines"].append(ln)
                if len(cur["lines"]) == 1 and head_like(ln["text"]):
                    cur["closed"] = True                        # a "label:" line stands alone
    for p in paras:
        if p["kind"] != "p":
            continue
        txt = ""
        for ln in p["lines"]:
            t = ln["text"]
            if not txt:
                txt = t
            elif t in (".", "…"):
                txt = txt.rstrip(" .") + t
            elif not p["rtl"] and txt.endswith("-") and t[:1].islower():
                txt = txt[:-1] + t                                     # de-hyphenate English
            else:
                txt += " " + t
        p["text"] = re.sub(r"\s+", " ", txt).strip()

    # join paragraphs broken by page breaks
    merged = []
    for p in paras:
        if (merged and p["kind"] == "p" and merged[-1]["kind"] == "p" and p["page"] != merged[-1]["page"]
                and p["style"] == merged[-1]["style"] and not merged[-1]["text"].endswith(TERMINAL)
                and not BULLET.match(p["text"]) and not NUMBERED.match(p["text"])
                and len(merged[-1]["text"]) > 60):
            merged[-1]["text"] += " " + p["text"]
            merged[-1]["lines"] += p["lines"]
        else:
            merged.append(p)
    paras = [p for p in merged if p["kind"] == "table" or p["text"]]

    NOTE = re.compile(r"^\s*(ملحوظة|ملحوظه|ملاحظة|ملاحظات|تنبيه|هام|تحذير|note|notes|n\.?b\.?)\s*[:\-–]*\s*$", re.I)
    folded = []
    for p in paras:
        if folded and folded[-1].get("note") and p["kind"] == "p":
            p["text"] = folded.pop()["text"].strip(" :-–") + ": " + p["text"]
        if p["kind"] == "p" and NOTE.match(p["text"]):
            p["note"] = True
        folded.append(p)
    paras = folded

    # mirrored brackets (some producers store the visual glyph): fix per paragraph
    for p in paras:
        if p["kind"] == "p" and p["rtl"]:
            t = p["text"]
            o, c = t.find("("), t.find(")")
            if c != -1 and (o == -1 or c < o) and t.count("(") == t.count(")"):
                p["text"] = t.translate(MIRROR)

    # --- headings
    def words(t):
        return len(t.split())

    def heading_candidate(p):
        t = p["text"]
        if p["kind"] != "p" or len(p["lines"]) > 2 or len(t) > 90 or words(t) > 12:
            return None
        if not re.search(r"[A-Za-z؀-ۿ]", t) or t.endswith((".", "،", ",", ";", "؛")):
            return None
        letters = re.findall(r"[A-Za-z؀-ۿ]", t)
        if len(letters) < 3 or t[:1].islower() or t.lower() in ("the", "and", "of", "in", "for", "with", "to"):
            return None
        lastw = re.sub(r"[\s:\-–]+$", "", t).split()[-1:] or [""]
        if lastw[0] in LEAD_IN or (t.startswith("و") and words(t) > 3):
            return None                            # a lead-in sentence ("… تؤدى الى:"), not a heading
        fam, size, bold = p["style"]
        styled = size >= body_size + 0.8 or (bold and not body_style[2]) or (fam != body_style[0] and size >= body_size - 0.6)
        colon = bool(re.search(r"(:-?|-:|:)\s*$", t)) and words(t) <= 7 and len(p["lines"]) == 1
        last = re.sub(r"[\s:\-]+$", "", t).split()[-1:] or [""]
        if colon and (last[0] in LEAD_IN or (t.startswith("و") and words(t) > 3) or
                      re.search(r"(following|include|includes|are|is|as follows|such as)\s*:\s*$", t, re.I)):
            colon = False
        if BULLET.match(t):
            return "bullet-colon" if colon else None
        if NUMBERED.match(t) and not colon and not styled:
            return None
        if styled:
            return "style"
        if colon:
            return "colon"
        return None

    for p in paras:
        p["hc"] = heading_candidate(p) if p["kind"] == "p" else None
    total_p = sum(1 for p in paras if p["kind"] == "p")
    style_count = Counter(p["style"] for p in paras if p["hc"] == "style")
    hstyles = [s for s, c in style_count.items() if c <= 0.4 * total_p]
    hstyles.sort(key=lambda s: (-s[1], not s[2], -style_count[s]))
    sizes = []
    for s in hstyles:                                                  # one level per size bucket
        if not sizes or all(abs(s[1] - z) > 0.6 for z in sizes):
            sizes.append(s[1])
    def size_level(sz):
        for i, z in enumerate(sizes):
            if abs(sz - z) <= 0.6:
                return min(2 + i, 4)
        return 4
    deepest = min(2 + len(sizes), 4) if sizes else 3
    for p in paras:
        hc = p.get("hc")
        if hc == "style" and p["style"] in hstyles:
            p["kind"], p["level"] = "h", size_level(p["style"][1])
        elif hc == "colon":
            p["kind"], p["level"] = "h", deepest if sizes else 3
        elif hc == "bullet-colon":
            p["kind"], p["level"] = "h", 4
        if p["kind"] == "h":
            t = BULLET.sub("", p["text"])
            t = re.sub(r"^\s*[:\-–]+\s*|\s*[:\-–]+\s*$", "", t)
            p["text"] = t.strip() or p["text"]
    # promote so the top level used is h2
    levels = sorted({p["level"] for p in paras if p["kind"] == "h"})
    if levels and levels[0] > 2:
        shift = levels[0] - 2
        for p in paras:
            if p["kind"] == "h":
                p["level"] -= shift

    for _ in range(2):
        cnt = Counter(p["level"] for p in paras if p["kind"] == "h")
        nxt = min((l for l in cnt if l > 2), default=None)
        if cnt.get(2, 0) < 3 and nxt is not None and cnt[nxt] >= 4:
            for p in paras:
                if p["kind"] == "h" and p["level"] > 2:
                    p["level"] -= 1
    ORD = re.compile(r"^\s*(أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا|first|second|third|fourth|fifth)\b", re.I)
    base, base_style = 2, None
    for p in paras:
        if p["kind"] != "h":
            continue
        numbered = NUMBERED.match(p["text"]) or ORD.match(p["text"]) or re.match(r"^\s*\d{1,2}\s*[-.)]\s*\S", p["text"])
        if numbered and p["style"] == base_style and p["level"] <= base:
            p["level"] = min(4, base + 1)          # "1- …" under a heading of the same look = subsection
        elif not numbered:
            base, base_style = p["level"], p["style"]

    # --- drop prefaces / acknowledgments / references / leftover contents sections
    out, skip_level, skip_hard = [], None, False
    for p in paras:
        is_head = p["kind"] == "h" or (p["kind"] == "p" and len(p["text"]) <= 40 and DROP_SECTION.match(p["text"]))
        if is_head:
            lvl = p.get("level", 2)
            if skip_level is not None and (not skip_hard or lvl <= skip_level):
                skip_level = None                  # prefaces end at the next heading; references at the next section
            if DROP_SECTION.match(p["text"]):
                skip_level, skip_hard = lvl, bool(HARD_DROP.match(p["text"]))
                continue
        if skip_level is not None:
            continue
        out.append(p)
    paras = out

    # --- lists
    for p in paras:
        if p["kind"] == "p":
            m = BULLET.match(p["text"])
            if m:
                p["kind"], p["text"] = "li", p["text"][m.end():].strip()
            elif NUMBERED.match(p["text"]):
                p["kind"] = "li"
    paras = [p for p in paras if p["kind"] == "table" or p["text"]]

    # final heading levels: top level = 2, and enough main sections for a useful contents list
    lv = sorted({p["level"] for p in paras if p["kind"] == "h"})
    if lv and lv[0] > 2:
        for p in paras:
            if p["kind"] == "h":
                p["level"] -= lv[0] - 2
    cnt = Counter(p["level"] for p in paras if p["kind"] == "h")
    deeper = [l for l in sorted(cnt) if l > 2 and cnt[l] >= 3]
    if cnt.get(2, 0) < 3 and deeper:
        shift = deeper[0] - 2
        for p in paras:
            if p["kind"] == "h":
                p["level"] = max(2, p["level"] - shift)

    # title: cover text if its words really occur in the document, else the cleaned PDF title
    def title_ok(t):
        if not t or TOC_LINE.search(t) or re.search(r"\.{3,}|[A-Za-z][\u0600-\u06FF]|[\u0600-\u06FF][A-Za-z]", t):
            return False
        ws = [w.lower() for w in re.findall(r"\w{3,}", t) if not w.isdigit()]
        tw = set(ws)
        vocab = Counter()
        for p in paras:                            # words must occur in real text, not in copies of the title
            if p["kind"] == "table":
                continue
            pw = [w.lower() for w in re.findall(r"\w{3,}", p["text"])]
            if pw and sum(1 for w in pw if w in tw) >= 0.5 * len(pw):
                continue
            vocab.update(pw)
        return bool(ws) and sum(1 for w in ws if vocab.get(w)) >= 0.7 * len(ws)

    meta = re.sub(r"\.(indd|docx?|dot|xps|pdf|qxd|p65|rtf)\s*$", "", meta_title.strip(), flags=re.I)
    meta = re.sub(r"\bcopy\b|microsoft\s+word\s*-?|\brevised\b|\bfinal\b", " ", meta, flags=re.I)
    meta = re.sub(r"\s+", " ", meta).strip(" -_")
    meta_good = (4 <= len(meta) <= 150 and len(re.findall(r"[A-Za-z\u0600-\u06FF]{2,}", meta)) >= 2
                 and not re.search(r"untitled|normal|template|createspace|^\W*\d+\W*$", meta, re.I) and not TOC_HEAD.match(meta))
    if not title_ok(title):
        title = meta if meta_good else ""
    title = re.sub(r"\s+", " ", title).strip(" -–:")
    return {"title": title, "elements": paras, "pages": n, "dropped_pages": sorted(d + 1 for d in drop)}


def to_outline(docd, fallback_title):
    """Elements -> list of (kind, level, text) with numbered h2 sections and a summary."""
    els = docd["elements"]
    arabic = sum(len(AR_CHAR.findall(e.get("text", ""))) for e in els if e["kind"] != "table") > \
             sum(len(re.findall(r"[A-Za-z]", e.get("text", ""))) for e in els if e["kind"] != "table")
    title = docd["title"]
    if not title:                                  # cleaned file name
        title = re.sub(r"[_\-]+", " ", fallback_title)
        title = re.sub(r"\b(compressed|copy|new|final|ticket file\w*)\b|نهائي|طبع|\(\d+\)", " ", title, flags=re.I)
        title = re.sub(r"\s+", " ", title).strip() or fallback_title
    outl, n2 = [], 0
    tkey = re.sub(r"\W+", "", title.lower())
    while els and els[0]["kind"] in ("p", "h") and len(re.sub(r"\W+", "", els[0]["text"].lower())) > 5 and \
            re.sub(r"\W+", "", els[0]["text"].lower()) in tkey:
        els = els[1:]                              # the title repeated as the first line
    for e in els:
        if e["kind"] == "h":
            if e["level"] == 2:
                n2 += 1
                outl.append(("h2", 2, f"{n2}. {e['text']}"))
            else:
                outl.append((f"h{e['level']}", e["level"], e["text"]))
        elif e["kind"] == "table":
            for item in table_to_items(e["rows"]):
                outl.append(("li", 0, item))
        else:
            outl.append((e["kind"], 0, e["text"]))
    h2 = [t.split(". ", 1)[-1] for k, _, t in outl if k == "h2"]
    if h2:
        lead = "يتناول هذا المستند: " if arabic else "This document covers: "
        summary = lead + ("، " if arabic else "; ").join(h2[:15]) + ("." if len(h2) <= 15 else " …")
    else:
        summary = ""
    return {"title": title, "summary": summary, "outline": outl, "arabic": arabic}


# ---------------------------------------------------------------- PDF template (notebook section 3.7)
def _font_css() -> str:
    """DejaVu Sans embedded like the notebook does (matplotlib ships it); without matplotlib the browser's fonts."""
    try:
        import base64

        import matplotlib
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

        def uri(name):
            return "data:font/ttf;base64," + base64.b64encode((ttf / name).read_bytes()).decode()
        return ('@font-face{font-family:"DocFont";font-weight:400;src:url(' + uri("DejaVuSans.ttf") + ') format("truetype");}\n'
                '@font-face{font-family:"DocFont";font-weight:700;src:url(' + uri("DejaVuSans-Bold.ttf") + ') format("truetype");}\n')
    except Exception:  # noqa: BLE001
        return ""


FONT_CSS = _font_css()
CSS = """
@page { size: A4; margin: 18mm; }
body { font-family: "DocFont", "DejaVu Sans", "Segoe UI", Tahoma, Arial, sans-serif; font-size: 15px; line-height: 2.15;
       color: #111; margin: 0; font-feature-settings: "rlig" 0, "liga" 0, "clig" 0; }
h1 { font-size: 30px; text-align: center; margin: 40px 0 14px; padding-bottom: 18px; border-bottom: 3px solid #2e7d32; line-height: 1.5; }
.summary { background: #f1f8f1; border: 1px solid #c8e6c9; padding: 12px 16px; margin: 24px 0; }
h2 { background: #2e7d32; color: #fff; font-size: 24px; padding: 12px 18px; margin: 34px 0 14px; line-height: 1.6; break-after: avoid; }
h3 { color: #1b5e20; font-size: 19px; border-inline-start: 5px solid #2e7d32; padding-inline-start: 12px; margin: 26px 0 8px; line-height: 1.6; break-after: avoid; }
h4 { color: #333; font-size: 17px; margin: 20px 0 6px; line-height: 1.6; break-after: avoid; }
p { margin: 6px 0 10px; }
ul { margin: 4px 0 12px; padding-inline-start: 34px; }
li { margin: 0 0 6px; }
.toc li { margin-bottom: 2px; }
"""
def outline_to_html(o):
    e = lambda s: html.escape(s, quote=False)
    parts = [f"<h1>{e(o['title'])}</h1>"]
    if o["summary"]:
        parts.append(f'<div class="summary">{e(o["summary"])}</div>')
    h2 = [t for k, _, t in o["outline"] if k == "h2"]
    if h2:
        shown = h2[:60] + (["…"] if len(h2) > 60 else [])
        parts.append(f'<h3>{"المحتويات" if o["arabic"] else "Contents"}</h3><ul class="toc">' + "".join(f"<li>{e(t)}</li>" for t in shown) + "</ul>")
    in_list = False
    for k, _, t in o["outline"]:
        if k == "li" and not in_list:
            parts.append("<ul>"); in_list = True
        if k != "li" and in_list:
            parts.append("</ul>"); in_list = False
        parts.append(f"<li>{e(t)}</li>" if k == "li" else f"<p>{e(t)}</p>" if k == "p" else f"<{k}>{e(t)}</{k}>")
    if in_list: parts.append("</ul>")
    lang, d = ("ar", "rtl") if o["arabic"] else ("en", "ltr")
    return (f'<!doctype html><html lang="{lang}" dir="{d}"><head><meta charset="utf-8"><title>{e(o["title"])}</title>'
            f"<style>{FONT_CSS}{CSS}</style></head><body>\n" + "\n".join(parts) + "\n</body></html>")
def outline_to_markdown(o):
    out = [f"# {o['title']}", ""]
    if o["summary"]: out += [f"> {o['summary']}", ""]
    prev = None
    for k, lvl, t in o["outline"]:
        if prev == "li" and k != "li": out.append("")
        if k.startswith("h"): out += ["", "#" * lvl + " " + t, ""]
        elif k == "li": out.append("- " + t)
        else: out += [t, ""]
        prev = k
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


# ================================================================ end of the copied part


# ---------------------------------------------------------------- the cleaned document
@dataclass
class CleanedDocument:
    title: str
    summary: str
    arabic: bool
    outline: list[tuple]            # (kind, level, text, page): kind h2/h3/h4/p/li, page = 1-based page in the original
    pages: int                      # pages in the original file
    dropped_pages: list[int] = field(default_factory=list)   # covers, credits, contents... (1-based)

    @property
    def sections(self) -> list[str]:
        return [t.split(". ", 1)[-1] for k, _, t, _ in self.outline if k == "h2"]

    @property
    def chars(self) -> int:
        return sum(len(t) for _, _, t, _ in self.outline)

    def as_outline(self) -> dict:
        """The notebook's to_outline() shape, for outline_to_html() / outline_to_markdown()."""
        return {"title": self.title, "summary": self.summary, "arabic": self.arabic,
                "outline": [(k, lvl, t) for k, lvl, t, _ in self.outline]}


def clean_pdf(path: Path, filename: str | None = None) -> CleanedDocument:
    """The notebook's extract_document() + to_outline(), keeping the original page of every element."""
    fallback_title = Path(filename or path).stem
    docd = extract_document(str(path), use_tables=True)
    els = docd["elements"]
    if not docd["title"]:                           # an upload's file name is often meaningless ("135905761430.pdf")
        docd["title"] = _cover_title(path)
    arabic = sum(len(AR_CHAR.findall(e.get("text", ""))) for e in els if e["kind"] != "table") > \
        sum(len(re.findall(r"[A-Za-z]", e.get("text", ""))) for e in els if e["kind"] != "table")
    title = docd["title"]
    if not title:                                   # cleaned file name, as in to_outline()
        title = re.sub(r"[_\-]+", " ", fallback_title)
        title = re.sub(r"\b(compressed|copy|new|final|ticket file\w*)\b|نهائي|طبع|\(\d+\)", " ", title, flags=re.I)
        title = re.sub(r"\s+", " ", title).strip() or fallback_title
    tkey = re.sub(r"\W+", "", title.lower())
    while els and els[0]["kind"] in ("p", "h") and len(re.sub(r"\W+", "", els[0]["text"].lower())) > 5 and \
            re.sub(r"\W+", "", els[0]["text"].lower()) in tkey:
        els = els[1:]                               # the title repeated as the first line
    outline, n2 = [], 0
    for e in els:
        page = int(e.get("page", 0)) + 1
        if e["kind"] == "h":
            if e["level"] == 2:
                n2 += 1
                outline.append(("h2", 2, f"{n2}. {e['text']}", page))
            else:
                outline.append((f"h{e['level']}", e["level"], e["text"], page))
        elif e["kind"] == "table":
            outline += [("li", 0, item, page) for item in table_to_items(e["rows"])]
        else:
            outline.append((e["kind"], 0, e["text"], page))
    h2 = [t.split(". ", 1)[-1] for k, _, t, _ in outline if k == "h2"]
    summary = ""
    if h2:
        lead = "يتناول هذا المستند: " if arabic else "This document covers: "
        summary = lead + ("، " if arabic else "; ").join(h2[:15]) + ("." if len(h2) <= 15 else " …")
    return CleanedDocument(title=title, summary=summary, arabic=arabic, outline=outline, pages=docd["pages"],
                           dropped_pages=list(docd["dropped_pages"]))


def _cover_title(path: Path) -> str:
    """The largest text on the first two pages that reads like a title (not a ministry / author / ISBN line),
    joined when it wraps over two or three lines: used when the engine found no title it trusts."""
    candidates = []   # (size, page, order, text)
    try:
        with pymupdf.open(str(path)) as pdf:
            for pno in range(min(2, pdf.page_count)):
                page = pdf[pno]
                order = 0
                for block in page_lines(page, page_glyphs(page, {}), exclude_rects=[]):
                    for ln in block["lines"]:
                        order += 1
                        t = re.sub(r"\s+", " ", ln["text"]).strip(" -–:")
                        if not 1 <= len(t.split()) <= 15 or len(re.findall(r"[A-Za-z\u0600-\u06FF]", t)) < 4:
                            continue
                        if FRONT_KW.search(t) or GLOBAL_DROP.search(t) or NAME_LINE.match(t) or TOC_LINE.search(t):
                            continue
                        candidates.append((float(ln.get("size", 0)), pno, order, t))
    except Exception:  # noqa: BLE001
        return ""
    if not candidates:
        return ""
    size, pno, _, _ = max(candidates, key=lambda c: c[0])
    lines = [c for c in candidates if c[1] == pno and abs(c[0] - size) < 0.6]
    title = " ".join(t for *_, t in sorted(lines, key=lambda c: c[2])[:3])
    return title[:150] if len(title.split()) >= 2 else ""


def cleaned_pages(doc: CleanedDocument) -> list[dict]:
    """Pages for extraction.chunk_document(): '# title', '## section' lines, paragraphs and list items, each on
    the page of the original file it came from — so the passages cite the farmer's own page numbers."""
    pages: list[dict] = []
    current, lines = None, []
    for kind, level, text, page in doc.outline:
        page = max(page, current or page)          # elements are in reading order; never go back a page
        if current is not None and page != current and lines:
            pages.append({"page": current, "lines": lines})
            lines = []
        current = page
        lines.append(f"{'#' * level} {text}" if kind.startswith("h") else text)
    if lines:
        pages.append({"page": current, "lines": lines})
    if pages:
        pages[0]["lines"].insert(0, f"# {doc.title}")
    out = []
    for p in pages:
        text = "\n\n".join(p["lines"])
        out.append({"page": p["page"], "text": text, "lang": detect_language(text)})
    return out


def outline_from_pages(title: str, pages: list[dict]) -> dict:
    """A Word / Markdown / text upload (already read by extraction.py) in the notebook's outline shape, so it
    gets the same clean PDF."""
    items, n2, arabic_chars, latin_chars = [], 0, 0, 0
    for p in pages:
        for block in re.split(r"\n\s*\n", p["text"]):
            block = " ".join(line.strip() for line in block.splitlines() if line.strip())
            if not block:
                continue
            arabic_chars += len(AR_CHAR.findall(block))
            latin_chars += len(re.findall(r"[A-Za-z]", block))
            m = re.match(r"^(#{1,4})\s+(.*)$", block)
            if m:
                level = max(2, len(m.group(1)))
                if len(m.group(1)) == 1 and (not title or m.group(2).strip() == title.strip()):
                    title = title or m.group(2)
                    continue                         # the title itself
                if level == 2:
                    n2 += 1
                    items.append(("h2", 2, f"{n2}. {m.group(2)}"))
                else:
                    items.append((f"h{level}", level, m.group(2)))
            else:
                items.append(("p", 0, block))
    arabic = arabic_chars > latin_chars
    h2 = [t.split(". ", 1)[-1] for k, _, t in items if k == "h2"]
    summary = (("يتناول هذا المستند: " if arabic else "This document covers: ")
               + ("، " if arabic else "; ").join(h2[:15]) + ("." if len(h2) <= 15 else " …")) if h2 else ""
    return {"title": title, "summary": summary, "outline": items, "arabic": arabic}


# ---------------------------------------------------------------- the clean PDF
# Chromium via Playwright, as in the notebook, in its own process (a browser in the API's worker threads
# would be fragile). Needs: pip install playwright && playwright install chromium.
_RENDER_SCRIPT = r"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
src, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(Path(src).resolve().as_uri(), wait_until="load")
    page.evaluate("document.fonts.ready.then(() => true)")
    options = dict(path=out, format="A4", print_background=True, prefer_css_page_size=True)
    try:
        page.pdf(tagged=True, outline=True, **options)
    except TypeError:
        page.pdf(**options)
    browser.close()
"""


def render_pdf(outline: dict, out: Path, timeout: float = 180) -> str | None:
    """Write the clean PDF. Returns the renderer used ("chromium" or "mupdf"), or None if neither worked
    (the document is still indexed from the clean text; there is just no PDF to download)."""
    out = Path(out)
    page_html = outline_to_html(outline)
    if importlib.util.find_spec("playwright") is not None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "document.html"
            src.write_text(page_html, encoding="utf-8")
            try:
                r = subprocess.run([sys.executable, "-c", _RENDER_SCRIPT, str(src), str(out)], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=timeout)
                if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
                    return "chromium"
                logger.warning("Chromium could not render the clean PDF (run `playwright install chromium`): %s",
                               r.stderr.strip()[-400:])
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("Chromium could not render the clean PDF: %s", exc)
    try:   # PyMuPDF's own HTML layout: simpler, but needs nothing else
        light = re.sub(r"@font-face\{[^}]*\}", "", page_html)     # MuPDF's CSS parser can't take the embedded font
        story = pymupdf.Story(html=light, user_css="body{font-family:sans-serif;}")
        writer = pymupdf.DocumentWriter(str(out))
        mediabox = pymupdf.paper_rect("a4")
        where = mediabox + (51, 51, -51, -51)
        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device)
            writer.end_page()
        writer.close()
        if out.exists() and out.stat().st_size > 0:
            return "mupdf"
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyMuPDF could not render the clean PDF: %s", exc)
    out.unlink(missing_ok=True)
    return None
