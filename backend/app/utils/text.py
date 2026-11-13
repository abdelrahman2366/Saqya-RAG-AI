"""Arabic / English text helpers shared by the backend and the notebook (no dependencies)."""

from __future__ import annotations

import re

# Arabic letters, including the "presentation forms" (shaped glyphs) that PDFs often store
ARABIC_LETTER_RE = re.compile("[ء-يٱ-ۓۺ-ۼﭐ-﷿ﹰ-ﻼ]")
LATIN_LETTER_RE = re.compile("[A-Za-z]")

# Harakat (fatha, damma, kasra, sukun, shadda, tanween...), superscript alef, Quranic marks
ARABIC_DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"

# Arabic-Indic and Persian digits, and the Arabic decimal separator
_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩"
    "۰۱۲۳۴۵۶۷۸۹٫",
    "01234567890123456789.",
)

LANGUAGE_NAMES = {"ar": "Arabic", "en": "English"}


def detect_language(text: str) -> str:
    """'ar' if the text is mostly Arabic script, otherwise 'en'."""
    arabic = len(ARABIC_LETTER_RE.findall(text))
    latin = len(LATIN_LETTER_RE.findall(text))
    return "ar" if arabic > latin else "en"


def strip_arabic_marks(text: str) -> str:
    """Remove diacritics and tatweel (elongation). Keeps the letters themselves unchanged."""
    return ARABIC_DIACRITICS_RE.sub("", text).replace(TATWEEL, "")


def to_ascii_digits(text: str) -> str:
    """Arabic-Indic and Persian digits -> 0-9, Arabic decimal separator -> '.'."""
    return text.translate(_DIGITS)


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation for *comparing* text (keyword checks), never for display.

    Unifies alef forms, alef maqsura, taa marbuta and hamza carriers so that
    'الأسمدة' matches 'الاسمده', and converts digits to ASCII.
    """
    text = strip_arabic_marks(to_ascii_digits(text.lower()))
    text = re.sub("[آأإٱ]", "ا", text)  # آ أ إ ٱ -> ا
    text = (
        text.replace("ى", "ي")  # ى -> ي
        .replace("ة", "ه")  # ة -> ه
        .replace("ؤ", "و")  # ؤ -> و
        .replace("ئ", "ي")  # ئ -> ي
    )
    return text
