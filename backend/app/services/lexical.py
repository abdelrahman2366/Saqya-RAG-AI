"""Keyword search (BM25) over the chunks, for Arabic and English.

The embedding model matches meaning, but it can rank a passage low even when it contains the exact
name the farmer typed: a rare pest ("التوتا"), a variety, a fertiliser or a number. BM25 scores exact
word matches (after normalising and stemming), so it is strong exactly where embeddings are weak.
retrieval.Retriever fuses the two rankings with Reciprocal Rank Fusion.

Arabic words are normalised (alef forms, taa marbuta, digits...) and light-stemmed: the conjunction
و, the article ال (with ب ك ف ل) and common suffixes are removed, so "للمبيدات" and "المبيد" meet
(Larkey's "light10" stemmer, the usual choice for Arabic search). English words use the Snowball
(Porter 2) stemmer.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np
import snowballstemmer

from app.utils.text import normalize_for_match

_EN_STEMMER = snowballstemmer.stemmer("english")

EN_STOPWORDS = set("""a an the and or of to in on for with by at from is are was were be been being as it its
this that these those can could should would will shall may might must your you yours their they them we our
i me my he she his her into about than then also not no nor do does did done have has had having more most less
very such each other any all some only when where which what who whom whose how why there here over under after
before during between up down out off so if but because while until again further once both few own same too
just get gets got make makes made use used using per much many compare comparison difference versus vs
better""".split())

AR_STOPWORDS = {normalize_for_match(w) for w in """في من على الى إلى عن ان أن إن انه أنه التي الذي الذين اللذان
اللتان هذا هذه ذلك تلك هؤلاء مع او أو ام أم كما ثم قد لا ما ماذا لماذا متى كيف كم هل أي اي أين اين لم لن هو هي
هم هن انت أنت انا أنا نحن يتم تم يجب يمكن ممكن عند بعد قبل كل بين حتى اذا إذا لكن وهو وهي ذات ذو غير ايضا أيضا
كان كانت يكون تكون وكان وكانت عليه عليها فيه فيها منه منها به بها له لها لهم التى الى
قارن قارني مقارنة الفرق فرق أيهما ايهما أم يا باشمهندس امتى إمتى ازاي إزاي ايه إيه""".split()}

_TOKEN_RE = re.compile(r"[a-z]+|[ء-ي]+|\d+(?:\.\d+)?")
_AR_ARTICLES = ("وال", "بال", "كال", "فال", "لل", "ال")
_AR_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")


def arabic_light_stem(word: str) -> str:
    """Larkey's light10 stemmer, on text already passed through normalize_for_match (ة -> ه)."""
    if len(word) > 3 and word.startswith("و"):
        word = word[1:]
    for article in _AR_ARTICLES:
        if word.startswith(article) and len(word) - len(article) >= 2:
            word = word[len(article):]
            break
    for suffix in _AR_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            word = word[: -len(suffix)]
    return word


def words(text: str) -> list[str]:
    """Normalised words and numbers of a text, stop words removed (not stemmed yet)."""
    out = []
    # "\ufffd" = an undecodable PDF glyph, usually inside a word (الأراض��ى): drop it so the word survives
    for token in _TOKEN_RE.findall(normalize_for_match(text.replace("\ufffd", "")).replace(",", ".")):
        if token[0].isdigit():
            out.append(token.rstrip("0").rstrip(".") if "." in token else token)
        elif len(token) > 1 and token not in EN_STOPWORDS and token not in AR_STOPWORDS:
            out.append(token)
    return out


def stem(word: str) -> str:
    if word[0].isdigit():
        return word
    if word[0] <= "z":  # Latin
        return _EN_STEMMER.stemWord(word)
    return arabic_light_stem(word)


def tokenize(text: str) -> list[str]:
    """Search terms of a text: normalised, stop words removed, stemmed. Numbers are kept."""
    return [stem(w) for w in words(text)]


class BM25Index:
    """Okapi BM25 over a fixed list of texts (built once at startup, a few seconds for 10k chunks).

    Arabic attaches the prepositions ل ب ك ف to the next word ("لحوامل", "لقمح"). light10 can't
    strip them blindly ("بطاطس" starts with ب), so the index learns from the corpus itself: a word
    loses its first letter when the rest is a more frequent word ("لقمح" -> "قمح", but "بطاطس" stays).
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        raw = [words(t) for t in texts]
        self.vocab = Counter(w for doc in raw for w in doc)
        docs = [self.terms(doc) for doc in raw]
        self.size = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.size else 1.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                postings[term].append((i, tf))
        self.postings = {
            term: (np.array([i for i, _ in p], dtype=np.int64), np.array([tf for _, tf in p], dtype=np.float32))
            for term, p in postings.items()
        }
        self.idf = {
            term: math.log(1 + (self.size - len(p) + 0.5) / (len(p) + 0.5)) for term, p in postings.items()
        }

    def _strip_preposition(self, word: str) -> str:
        if len(word) >= 4 and word[0] in "لبكف" and not word.startswith(("لل", "ال")) \
                and self.vocab.get(word[1:], 0) > self.vocab.get(word, 0):
            return word[1:]
        return word

    def terms(self, text_or_words: str | list[str]) -> list[str]:
        ws = words(text_or_words) if isinstance(text_or_words, str) else text_or_words
        return [stem(self._strip_preposition(w)) for w in ws]

    def scores(self, query: str | Iterable[str]) -> np.ndarray:
        """BM25 score of every text for the query (a string, or pre-tokenised terms)."""
        terms = self.terms(query) if isinstance(query, str) else list(query)
        scores = np.zeros(self.size, dtype=np.float32)
        for term in set(terms):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[ids] / self.avg_len)
            scores[ids] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def top(self, query: str | Iterable[str], n: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """(index, score) of the n best texts with a score > 0, best first (only `allowed` ones if given)."""
        scores = self.scores(query)
        if allowed is not None:
            mask = np.zeros(self.size, dtype=bool)
            mask[allowed] = True
            scores[~mask] = 0.0
        n = min(n, int((scores > 0).sum()))
        if n == 0:
            return []
        best = np.argpartition(-scores, n - 1)[:n]
        best = best[np.argsort(-scores[best])]
        return [(int(i), float(scores[i])) for i in best]
