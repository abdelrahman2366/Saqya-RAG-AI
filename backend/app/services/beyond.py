"""Beyond the documents: what happens when the library can't answer a farming question.

RAGService always asks the library (the indexed bulletins) first. When it has no answer, and the question is
about agriculture (scope.py), the next sources are tried in order, each with the same answer checks:

1. Learned knowledge (learning.py): passages kept from earlier web answers and attached documents. A question
   asked before is answered at once, offline, citing the page or document it came from.
2. The web (web_search.py): a short search query is written for the question (Egypt and the season added when
   they matter), several engines are searched, the best pages are read, and their passages are ranked against
   the question with the library's own hybrid search. The answer cites the pages; the passages it cited are
   learned for next time (when the farmer allows it).
3. General knowledge: the LLM answers from what it knows, told not to invent doses or prices and to say what
   the answer depends on; the chat app marks it "general knowledge — not from a source".

A question that is not about agriculture still gets the polite refusal, whatever the farmer allows. What the
farmer allows comes with each question (Settings in the chat app): search the web, learn, general knowledge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.services import grounding
from app.services.retrieval import MemoryCollection, Retriever
from app.services.scope import is_agricultural
from app.services.web_search import WebSearcher, page_passages
from app.utils.text import detect_language

logger = logging.getLogger(__name__)

# A farming question that no source could answer (web unreachable, general knowledge off)
NOT_FOUND_ANSWER = {
    "en": "I couldn't find this in my documents or on the web. Try asking with the crop, the problem and your region, "
          "or ask your local agricultural extension officer.",
    "ar": "لم أجد إجابة لهذا في مستنداتي أو على الإنترنت. جرّب أن تسأل مع ذكر المحصول والمشكلة ومنطقتك، "
          "أو استشر المرشد الزراعي في منطقتك.",
}

# A question that isn't about farming
OFF_TOPIC_ANSWER = {
    "en": "I'm SAQYA, a farming assistant, so I can only help with farming: crops, soil, water and irrigation, "
          "fertilizers, pests and diseases, livestock and farm business. Ask me anything about those.",
    "ar": "أنا ساقية، مساعد زراعي، لذلك أستطيع المساعدة في الزراعة فقط: المحاصيل والتربة والري والتسميد والآفات "
          "والأمراض والثروة الحيوانية وإدارة المزرعة. اسألني عن أي شيء من ذلك.",
}

GENERAL_NOTE = {
    "en": "This answer is from general farming knowledge, not from a document or web page.",
    "ar": "هذه الإجابة من المعرفة الزراعية العامة، وليست من وثيقة أو صفحة على الإنترنت.",
}


def closest_passages(passages: list[dict], queries: list[str], language: str, keep: int = 40) -> list[dict]:
    """The passages worth embedding: those sharing most words with the question (and its translation), pages in the
    question's language first. Embedding every passage of five pages is what made web answers slow."""
    if len(passages) <= keep:
        return passages
    asked = set().union(*(grounding._light_words(q) for q in queries))
    def score(p: dict) -> tuple:
        same = detect_language(p["text"]) == language
        return (len(asked & grounding._light_words(p["text"])), same)
    ranked = sorted(range(len(passages)), key=lambda i: score(passages[i]), reverse=True)[:keep]
    return [passages[i] for i in sorted(ranked)]


def without_refusals(answer: str) -> str:
    """The answer without its "I don't have enough information" sentences (a general answer often opens with one
    and then gives the advice anyway). The advice to see the extension officer is kept."""
    if not grounding.is_refusal(answer):
        return answer
    kept = [part for part in grounding.segments(answer) if part == "\n" or not grounding.is_refusal(part)]
    rest = grounding.join_segments(kept).strip()
    rest = re.sub(r"^\s*(?:however|but|that said|still)\s*,?\s*|^\s*(?:ولكن|لكن|ومع ذلك)\s*،?\s*", "", rest,
                  flags=re.IGNORECASE)
    return rest[:1].upper() + rest[1:]


@dataclass
class SourcePolicy:
    """What the farmer allows for a question the documents don't cover."""
    web: bool = True       # search the web
    learn: bool = True     # keep what the web answer cited, for later questions
    general: bool = True   # answer from general knowledge when nothing else does


class Beyond:
    def __init__(self, embedder, query_prefix: str = "", passage_prefix: str = "", embedding_model: str = "",
                 web: WebSearcher | None = None, knowledge=None, general: bool = True,
                 web_min_similarity: float = 0.45, learned_min_similarity: float = 0.5, top_k: int = 6) -> None:
        self.embedder = embedder
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.embedding_model = embedding_model
        self.web = web
        self.knowledge = knowledge
        self.general = general
        self.web_min_similarity = web_min_similarity
        self.learned_min_similarity = learned_min_similarity
        self.top_k = top_k

    @property
    def web_available(self) -> bool:
        return self.web is not None and self.web.available

    def answer(self, rag, question: str, language: str, queries: list[str], follow: bool,
               earlier: str | None, policy: SourcePolicy):
        """A RAGResult from the next source that answers, a friendly refusal for a question that isn't about farming,
        or "not found" when nothing answered."""
        scope_text = f"{earlier}\n{question}" if earlier else question
        if not is_agricultural(scope_text, rag.generator):
            from app.services.rag_service import RAGResult

            return RAGResult(OFF_TOPIC_ANSWER[language], [], language, [], queries, origin="none", off_topic=True)
        tried = []

        # 1. what SAQYA learned before
        if self.knowledge is not None:
            chunks = self.knowledge.search(queries, self.top_k, self.learned_min_similarity)
            if chunks:
                result = rag.respond(question, chunks, language, queries, follow, earlier, origin="learned",
                                     gate=self.learned_min_similarity)
                if result.cited:
                    self.knowledge.used([c for _, c in result.cited])
                    return result
                tried.append("learned: passages found but no answer")
            else:
                tried.append("learned: nothing similar")

        # 2. the web
        if not policy.web:
            tried.append("web: off in Settings")
        elif not self.web_available:
            tried.append("web: unavailable (pip install ddgs trafilatura, WEB_SEARCH=true)")
        else:
            result = self._from_web(rag, question, language, queries, follow, earlier, tried)
            if result is not None:
                if policy.learn and self.knowledge is not None:
                    try:
                        result.learned = self.knowledge.add_web([c for _, c in result.cited], question)
                    except Exception as exc:  # noqa: BLE001 - learning must never break an answer
                        logger.warning("Could not learn from the web answer: %s", exc)
                return result

        # 3. general knowledge
        if policy.general and self.general:
            result = self._general(rag, question, language, queries, earlier, tried)
            if result is not None:
                return result
        else:
            tried.append("general knowledge: off")
        logger.info("Nothing answered beyond the documents: %s", "; ".join(tried))
        from app.services.rag_service import RAGResult

        return RAGResult(NOT_FOUND_ANSWER[language], [], language, [], queries, origin="none")

    # ------------------------------------------------------------ the web
    def web_queries(self, rag, question: str, language: str, queries: list[str], earlier: str | None) -> list[tuple[str, str]]:
        """(query, language) pairs: the LLM's short English query, the question, and its translation."""
        pairs: list[tuple[str, str]] = []
        main = queries[0] if queries else question
        if language == "ar":   # pages in the farmer's own language first: they are answered most faithfully
            pairs.append((f"{main[:280]} مصر" if "مصر" not in main else main[:300], "ar"))
        write = getattr(rag.generator, "web_query", None)
        written = write(question, earlier) if write else None
        if written:
            pairs.append((written, "en"))
        if language != "ar":
            pairs.append((main[:300], language))
        if len(queries) > 1 and not written:
            other = "en" if language == "ar" else "ar"
            pairs.append((queries[1][:300], other))
        seen, out = set(), []
        for q, lang in pairs:
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append((q, lang))
        return out[:3]

    def _from_web(self, rag, question, language, queries, follow, earlier, tried=None):
        tried = tried if tried is not None else []
        pairs = self.web_queries(rag, question, language, queries, earlier)
        logger.info("Web search for: %s", " | ".join(q for q, _ in pairs))
        try:
            pages = self.web.pages(pairs)
        except Exception as exc:  # noqa: BLE001 - no internet etc.
            logger.warning("Web search failed: %s", exc)
            tried.append(f"web: failed ({exc})")
            return None
        passages = page_passages(pages)
        if not passages:
            logger.info("Web: nothing readable found")
            tried.append("web: no results or no readable page (see 'Web search failed' above)")
            return None
        passages = closest_passages(passages, [*queries, *(q for q, _ in pairs)], language)
        texts = [p["text"] for p in passages]
        vectors = np.asarray(self.embedder.encode([self.passage_prefix + t for t in texts], batch_size=16,
                                                  normalize_embeddings=True), dtype=np.float32)
        collection = MemoryCollection("web", [p["id"] for p in passages], texts, [p["metadata"] for p in passages], vectors)
        retriever = Retriever(collection, self.embedder, {"embedding_model": self.embedding_model,
                                                          "query_prefix": self.query_prefix}, hybrid=True)
        search_texts = list(dict.fromkeys([*queries, *(q for q, _ in pairs)]))
        found = retriever.search(search_texts, self.top_k)
        if found.best_similarity < self.web_min_similarity:
            logger.info("Web: best passage %.2f below %.2f; not used", found.best_similarity, self.web_min_similarity)
            tried.append(f"web: {len(pages)} pages, best passage {found.best_similarity:.2f} < {self.web_min_similarity:.2f}")
            return None
        result = rag.respond(question, found.chunks, language, queries, follow, earlier, origin="web",
                             gate=self.web_min_similarity)
        if result.cited and "broken text removed" in (result.problem or ""):
            # the model could not write these pages up cleanly even when asked twice: better no web answer than a
            # patched one (general knowledge, written directly in the farmer's language, comes next)
            tried.append("web: the answer came out garbled twice; not used")
            return None
        if not result.cited:
            tried.append(f"web: {len(pages)} pages read, but the answer didn't pass the checks")
        return result if result.cited else None

    # ------------------------------------------------------------ general knowledge
    def _general(self, rag, question, language, queries, earlier, tried=None):
        from app.services.rag_service import RAGResult

        tried = tried if tried is not None else []
        write = getattr(rag.generator, "general_answer", None)
        if write is None:
            return None
        try:
            answer = write(question, language, earlier, today=datetime.now()).strip()
        except Exception as exc:  # noqa: BLE001 - Ollama down: the refusal stands
            logger.warning("General-knowledge answer failed: %s", exc)
            tried.append(f"general knowledge: failed ({exc})")
            return None
        answer = grounding.polish(without_refusals(answer), language)
        if answer and grounding.garbled(answer, detect_language(answer)):
            answer = grounding.drop_garbled(answer, detect_language(answer))   # "I don't know your soil, but…" keeps the tips
        if not grounding.has_content(answer) or grounding.is_echo(answer, question):
            tried.append("general knowledge: the model gave no usable answer")
            return None
        if detect_language(answer) != language:   # the model wrote in the other language: translate it
            translate = getattr(rag.generator, "translate_answer", None)
            translated = translate(answer, language) if translate else None
            if translated and detect_language(translated) == language:
                answer = translated
        language = detect_language(answer)   # the language it is really written in
        if rag._about_pesticides(question, answer, language):
            answer = f"{answer}\n\n{grounding.SAFETY_NOTE[language]}"
        logger.info("Answered from general knowledge")
        return RAGResult(answer, [], language, [], queries, answer, origin="general", note=GENERAL_NOTE[language])
