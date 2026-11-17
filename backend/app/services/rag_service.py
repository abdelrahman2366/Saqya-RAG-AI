"""The full pipeline for one question: dialect -> translate -> search -> refuse or generate -> check -> sources.

An Arabic question in Egyptian colloquial is first rewritten into standard Arabic (dialect.py), so a
spoken "إمتى أزرع الأوطة؟" is searched and answered like "متى أزرع الطماطم؟".

Every answer is checked before the farmer sees it (grounding.py):
1. LLM-free repairs: citations to passages that don't exist are removed, a repeated question is
   cut, "I don't know. However, according to [1] ..." keeps only the answer, a sentence marked with
   the wrong passage number gets the right one, and sentences copied from a passage without a [n]
   marker get one.
2. If problems remain (only the question repeated, wrong language, a passage about another crop,
   a number or statement not in the passages, nothing cited), the LLM gets its answer back with
   every problem named and tries once more; the retried answer is kept only if it is better.
   If the model refused although passages were found, it gets a second look at only the two
   passages that share most words with the question (small models miss an answer among six), and
   that answer is kept only if it passes the grounding check.
3. Last safety net: a sentence with a number found in no passage, or an uncited sentence whose words
   are not in the passages, is removed (a farmer must never get an invented dose); if nothing is
   left, the answer is the refusal.
4. The answer is in the language of the question: if the LLM still wrote it in the passages' language,
   it is translated, and the translation is kept only with the same [n] markers and no new number.

Follow-up questions ("how much water does it need?") are searched together with the farmer's earlier
question (conversation.py), which the LLM also sees as context.

When the library has no answer to a farming question, beyond.py tries what SAQYA learned before, then the web,
then general knowledge (each answer says where it came from: `origin`).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status

from app.core.tracing import tracer
from app.services import crop_calendar, grounding
from app.services.conversation import follow_up_context
from app.services.dialect import to_msa
from app.services.generation import FEEDBACK, NO_CONTEXT_ANSWER, OllamaGenerator, select_passages
from app.services.glossary import crops_in, find_terms
from app.services.spelling import repair
from app.services.retrieval import RetrievedChunk, Retriever
from app.utils.text import LANGUAGE_NAMES, detect_language

logger = logging.getLogger(__name__)

SAFETY_NOTE = grounding.SAFETY_NOTE


@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    language: str
    # for the notebook's evaluation and the logs (not sent by the API)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    first_answer: str = ""
    problem: str | None = None     # what was wrong with the first answer, if anything
    retried: bool = False
    cited: list[tuple[int, RetrievedChunk]] = field(default_factory=list)  # (n, passage) behind `sources`
    topic: str | None = None       # crop focus applied (full topic label), if any
    origin: str = "library"        # library · document · learned · web · general · none (refused)
    note: str = ""                 # shown under the answer (general knowledge: "not from a source")
    learned: int = 0               # passages of this answer kept for later questions
    off_topic: bool = False        # refused because it isn't about farming
    # optional layers, each None unless its flag is on (see core/config.py)
    understanding: dict | None = None   # query_understanding.py: crop, categories, intent, conditions, missing
    claims: dict | None = None          # claims.py: a verdict per statement of the answer
    confidence: dict | None = None      # confidence.py: decision (answer/qualify/ask/abstain), probability, signals


# Passages that only list a document's contents ("يتناول هذا المستند: مقدمة، ...") rarely answer anything
_CONTENTS_RE = re.compile(r"^(?:[^\n]*\n)?\s*(?:يتناول هذا المستند|This document covers)\s*:", re.IGNORECASE)


def focus_passages(chunks: list[RetrievedChunk], keep: int = 4, margin: float = 0.15) -> list[RetrievedChunk]:
    """The passages worth showing the LLM: the best `keep` always, the rest only when nearly as close to the
    question as the best one, and contents-only passages only when nothing else was found. Fewer, closer
    passages give a small model a clearer, more coherent answer."""
    if len(chunks) <= 1:
        return chunks
    useful = [c for c in chunks if not (_CONTENTS_RE.match(c.text) and len(c.text) < 900)] or chunks
    best = max(c.score for c in useful)
    return [c for i, c in enumerate(useful) if i < keep or c.score >= best - margin]


def _takes_weights(retriever) -> bool:
    import inspect

    try:
        return "weights" in inspect.signature(retriever.retrieve_many).parameters
    except (TypeError, ValueError, AttributeError):
        return False


class RAGService:
    origin = "library"   # where this service's answers come from (DocumentRAG: "document")

    def __init__(
        self,
        retriever: Retriever,
        generator: OllamaGenerator,
        top_k: int = 6,
        min_similarity: float | None = None,
        translate_queries: bool = True,
        self_check: bool = True,
        beyond=None,
        *,
        reranker=None,
        rerank_candidates: int = 30,
        expansion: bool = False,
        expansion_weight: float = 0.25,
        expansion_max_variants: int = 2,
        expansion_max_terms: int = 4,
        classify: bool = False,
        verify_claims: bool = False,
        nli=None,
        confidence=None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        # Optional layers. Every one is off unless asked for, and with all of them off this class behaves exactly as
        # it did before they existed (tests/test_pipeline.py and tests/test_ai_layers.py check that).
        self.reranker = reranker                      # reranker.Reranker: cross-encoder over the search's candidates
        self.rerank_candidates = max(int(rerank_candidates), top_k)
        self.expansion = expansion                    # query_expansion.expand: extra, lower-weight search phrasings
        self.expansion_weight = min(max(float(expansion_weight), 0.0), 0.99)   # always below the farmer's own wording
        self.expansion_max_variants = expansion_max_variants
        self.expansion_max_terms = expansion_max_terms
        self.classify = classify                      # return query_understanding in the result
        self.verify_claims = verify_claims            # claims.verify on every sourced answer
        self.nli = nli                                # claims.NLIVerifier, optional
        self.confidence = confidence                  # confidence.ConfidenceEstimator
        self.beyond = beyond   # beyond.Beyond: learned knowledge, the web, general knowledge (None: library only)
        self.top_k = top_k
        self.translate_queries = translate_queries
        self.self_check = self_check
        # None -> the threshold the notebook calibrated for this embedding model
        self.min_similarity = (
            min_similarity
            if min_similarity is not None
            else getattr(retriever, "default_min_similarity", 0.0)
        )

    def search_queries(self, question: str, language: str) -> list[str]:
        """The question, plus its translation into the other language.

        A crop or pest name rarely matches across languages ("sugar beet" vs "بنجر السكر"), so an
        English question can miss the Arabic bulletin that answers it. Searching with both versions
        lets each document be matched in its own language.
        """
        queries = [question]
        if self.translate_queries:
            target = "en" if language == "ar" else "ar"
            with tracer.span("translate_query", "llm", inputs={"question": tracer.text(question), "target": target}) as span:
                translated = self.generator.translate(question, target)
                span.set(outputs={"translation": tracer.text(translated)})
            if translated and translated != question:
                queries.append(translated)
        return queries

    def _retrieve(self, queries: list[str], **focus) -> list[RetrievedChunk]:
        """The passages for these search queries (DocumentRAG adds each top passage's continuation).

        With the optional layers on: query expansions are searched as extra, lower-weight phrasings; the search
        returns `rerank_candidates` passages and the cross-encoder keeps the best `top_k`; and a calibrated
        confidence below its abstain threshold means "nothing found here". Each layer falls back to the plain
        search on any failure."""
        search, weights = list(queries), None
        if self.expansion and queries:
            try:
                from app.services.query_expansion import expand

                with tracer.span("query_expansion", "tool", inputs={"query": tracer.text(queries[0])}) as span:
                    extra = expand(queries[0], max_variants=self.expansion_max_variants,
                                   max_terms=self.expansion_max_terms, other_language=len(queries) == 1)
                    span.set(outputs={"variants": [tracer.text(v) for v in extra.variants], "kinds": extra.kinds})
                added = [v for v in extra.variants if v not in search]
                if added and not _takes_weights(self.retriever):
                    added = []   # a retriever without weighted phrasings (e.g. a test double): search as before
                if added:
                    search += added
                    weights = [1.0] * len(queries) + [self.expansion_weight] * len(added)
                    logger.info("Query expansion (%s): %s", ", ".join(extra.kinds), " | ".join(a[:80] for a in added))
            except Exception as exc:  # noqa: BLE001 - search with the question as it is
                logger.warning("Query expansion skipped (%s)", exc)
                search, weights = list(queries), None
        reranking = self.reranker is not None and not self.reranker.failed
        wanted = self.rerank_candidates if reranking else self.top_k
        if weights:
            focus = {**focus, "weights": weights}
        with tracer.span("retrieve", "retriever",
                         inputs={"queries": [tracer.text(q) for q in search], "weights": weights, "n": wanted,
                                 "gate": self.min_similarity, "topic": focus.get("topic")}) as span:
            chunks = self.retriever.retrieve_many(search, wanted, self.min_similarity, **focus)
            span.set(outputs={"passages": tracer.passages(chunks[:self.top_k]), "found": len(chunks)})
        if reranking and len(chunks) > 1:
            with tracer.span("rerank", "retriever", inputs={"candidates": len(chunks)},
                             metadata={"model": self.reranker.model_name}) as span:
                chunks = self.reranker.rerank(queries, chunks, self.top_k)
                if self.reranker.failed:   # the model could not be loaded: the plain search, exactly as without it
                    chunks = self.retriever.retrieve_many(search, self.top_k, self.min_similarity, **focus)
                span.set(outputs={"passages": tracer.passages(chunks), "failed": self.reranker.failed},
                         scoring_ms=round(self.reranker.last_ms, 1))
        chunks = chunks[:self.top_k]
        if chunks and self.confidence is not None:
            from app.services.confidence import ABSTAIN, retrieval_signals

            verdict = self.confidence.before_answer(retrieval_signals(queries, chunks, self.min_similarity))
            if verdict.decision == ABSTAIN:
                logger.info("Confidence: %s; the library's passages are not used", "; ".join(verdict.reasons))
                return []
        return chunks

    def answer(self, question: str, topic: str | None = None, history: list[str] | None = None,
               policy=None) -> RAGResult:
        """`topic`: optional crop the farmer chose to focus on (only its documents are searched).
        `history`: the farmer's previous questions in this conversation, oldest first (for follow-ups).
        `policy`: beyond.SourcePolicy — what the farmer allows when the library has no answer."""
        with tracer.request(), tracer.span(
                "saqya.answer", "chain",
                inputs={"question": tracer.text(question), "topic": topic, "history_turns": len(history or [])},
                metadata={"service": self.origin, "llm": getattr(self.generator, "model", ""),
                          "embedding_model": getattr(self.retriever, "embedding_model", ""), "top_k": self.top_k,
                          "reranker": getattr(self.reranker, "model_name", None), "expansion": self.expansion}) as span:
            result = self._answer(question, topic, history, policy)
            if self.classify or self.verify_claims or self.confidence is not None:
                try:
                    self._annotate(result, question, history)
                except Exception as exc:  # noqa: BLE001 - the answer stands without its annotations
                    logger.warning("Could not annotate the answer (%s)", exc)
            span.set(outputs={"answer": tracer.text(result.answer), "origin": result.origin, "sources": result.sources,
                              "language": result.language, "problem": result.problem, "retried": result.retried,
                              "confidence": (result.confidence or {}).get("decision")})
        return result

    def _annotate(self, result: RAGResult, question: str, history: list[str] | None) -> None:
        """Attach the optional layers' findings to a finished result, and act on the confidence decision:
        a refusal that only lacks the crop becomes a clarifying question; a shaky answer gets a caution line."""
        from app.services import claims as claim_checks
        from app.services.confidence import ASK, ASK_TEXT, FOLLOW_UP, QUALIFY, QUALIFY_NOTE, retrieval_signals
        from app.services.query_understanding import understand

        understanding = understand(repair(question), history)
        if self.classify:
            result.understanding = understanding.to_dict()
        sourced = bool(result.cited) and result.origin in ("library", "document", "learned", "web")
        report = None
        if self.verify_claims and sourced and result.chunks:
            with tracer.span("claim_verification", "chain", inputs={"answer": tracer.text(result.answer)}) as span:
                report = claim_checks.verify(result.answer, result.chunks, question, self.nli)
                result.claims = report.to_dict()
                span.set(outputs={"counts": report.counts, "supported_share": report.supported_share})
            logger.info("Claims: %s", report.counts)
        if self.confidence is None:
            return
        own = result.origin in ("library", "document")   # the calibration was fitted on the library's search
        signals = retrieval_signals(result.queries or [question], result.chunks if own else [],
                                    self.min_similarity, understanding.missing)
        signals.retried = result.retried
        if report is not None:
            signals.supported_share, signals.worst_claim = report.supported_share, report.worst
        answered = bool(result.cited) or result.origin == "general"
        with tracer.span("confidence", "chain", inputs={"signals": signals.to_dict()}) as span:
            verdict = self.confidence.after_answer(signals, answered, result.off_topic)
            result.confidence = verdict.to_dict()
            span.set(outputs={"decision": verdict.decision, "probability": verdict.probability,
                              "reasons": verdict.reasons})
        language = result.language if result.language in ("ar", "en") else "en"
        if verdict.decision == ASK:
            result.answer = ASK_TEXT.get(understanding.missing[0], ASK_TEXT["crop"])[language]
        elif (answered and not result.off_topic and understanding.intent == "diagnose" and understanding.symptoms
              and "crop" in understanding.missing and FOLLOW_UP[language] not in result.answer):
            # a symptom described without the crop ("my plants are dying"): whatever was found can only be general, so
            # ask for what would make it specific. A general "why" question ("why is seed treated?") has no symptom.
            result.answer = f"{result.answer.rstrip()}\n\n{FOLLOW_UP[language]}"
            verdict.reasons.append("no crop named: asked for crop, symptoms and irrigation")
            result.confidence = verdict.to_dict()
        elif verdict.decision == QUALIFY and QUALIFY_NOTE[language] not in (result.note or ""):
            result.note = f"{result.note} {QUALIFY_NOTE[language]}".strip()
        logger.info("Confidence: %s%s", verdict.decision, f" ({'; '.join(verdict.reasons)})" if verdict.reasons else "")

    def _answer(self, question: str, topic: str | None = None, history: list[str] | None = None,
                policy=None) -> RAGResult:
        question = repair(question)   # a mis-heard or mistyped farming word ("مع حصيل" -> "محاصيل")
        language = detect_language(question)
        if language == "ar":
            # Egyptian colloquial ("إمتى أزرع الأوطة؟") -> the standard Arabic of the bulletins
            # ("متى أزرع الطماطم؟"): that is what is searched, translated and answered
            standard = to_msa(question)
            if standard != question:
                logger.info("Dialect rewrite: %s -> %s", question[:80], standard[:80])
                question = standard
        if self.origin == "library" and not topic:
            # "what should I plant?" names no crop, so no single passage answers it. The facts come from the bulletins'
            # own sowing dates and soil notes (crop_calendar.py); the LLM writes the answer to this farmer's question.
            planned = crop_calendar.context(question, language, history)
            if planned:
                passages, note, follow_up = planned
                written = None
                try:
                    written = self.respond(f"{question}\n({note})", passages, language, [question])
                except Exception as exc:  # noqa: BLE001 - Ollama down: the plain list below still answers
                    logger.warning("Crop suggestions: the LLM could not be asked (%s)", exc)
                # used only when it is sound: in the right language, naming the crops it was given, no list item
                # without a crop, no "does not suit this month" (every crop given does), nothing removed as broken text
                if written is not None and written.cited and written.language == language \
                        and "broken text removed" not in (written.problem or "") \
                        and crop_calendar.sound(written.answer, passages, language):
                    if follow_up:
                        written.answer = f"{written.answer}\n\n{follow_up}"
                    logger.info("Crop suggestions written from the calendar (%d crops offered)", len(passages))
                    return written
                text, listed = crop_calendar.answer(question, language, history)   # the checks failed: the plain list
                cited = list(enumerate(listed, 1))
                logger.info("Crop suggestions: plain list from the calendar (%d crops)", len(listed))
                return RAGResult(text, [f"[{n}] {c.label}" for n, c in cited], language, listed, [question], text,
                                 cited=cited, origin="library")
        follow = follow_up_context(question, history)
        label = None
        if topic and hasattr(self.retriever, "topic_label"):
            label = self.retriever.topic_label(topic)
        focus = {"topic": topic} if label else {}
        earlier = None
        if follow and not follow.strong:
            # Maybe a follow-up ("بتحتاج مية قد إيه؟" after a question about a crop), maybe a new subject ("how do I
            # raise rabbits?"). On its own first; with the earlier question only if it finds nothing alone and the
            # passages it then finds speak about what was asked (water, not just the earlier crop).
            queries = self.search_queries(question, language)
            chunks = self._retrieve(queries, **focus)
            if not chunks:
                previous = to_msa(follow.previous) if detect_language(follow.previous) == "ar" else follow.previous
                combined = self.search_queries(f"{previous} {question}", language)
                found = self._retrieve(combined, **focus)
                if found and grounding.mentions(question, found):
                    logger.info("Follow-up of: %s", previous[:80])
                    chunks, earlier, queries = found, previous, list(dict.fromkeys([*queries, *combined]))
                else:
                    follow = None
            else:
                follow = None
        elif follow:
            earlier = to_msa(follow.previous) if detect_language(follow.previous) == "ar" else follow.previous
            logger.info("Follow-up of: %s", earlier[:80])
            queries = []
            for text in follow.search_texts(question):
                text = text.replace(follow.previous, earlier)
                queries += [q for q in self.search_queries(text, language) if q not in queries]
            chunks = self._retrieve(queries, **focus)
        else:
            queries = self.search_queries(question, language)
            chunks = self._retrieve(queries, **focus)

        if chunks:
            result = self.respond(question, chunks, language, queries, bool(follow), earlier, label)
        else:
            # Nothing relevant in the documents: never let the LLM answer from them anyway
            logger.info("Best similarity below min_similarity=%.2f; nothing in the documents", self.min_similarity)
            result = RAGResult(NO_CONTEXT_ANSWER[language], [], language, [], queries, topic=label, origin="none")
        if result.cited and self.beyond is not None and not label and not crops_in(question):
            # A question that names no crop can match passages that are only loosely related. One short yes/no:
            # does the answer give what was asked? If not, it is not shown and the other sources are tried.
            judge = getattr(self.generator, "answers_question", None)
            if judge is not None and judge(question, grounding.answer_body(result.answer)) is False:
                logger.info("Answer was beside the question; trying the other sources: %s", question[:80])
                beside = result
                result = RAGResult(NO_CONTEXT_ANSWER[language], [], language, [], queries, topic=label, origin="none")
                from app.services.beyond import SourcePolicy

                with tracer.span("beyond_documents", "chain", inputs={"reason": "answer beside the question"}) as span:
                    extra = self.beyond.answer(self, question, language, queries, bool(follow), earlier, policy or SourcePolicy())
                    span.set(outputs={"origin": getattr(extra, "origin", None)})
                return extra if extra is not None and (extra.origin != "none" or extra.off_topic) else beside
        if result.cited or self.beyond is None or label:
            return result
        from app.services.beyond import SourcePolicy

        with tracer.span("beyond_documents", "chain", inputs={"reason": "nothing in the library"}) as span:
            extra = self.beyond.answer(self, question, language, queries, bool(follow), earlier, policy or SourcePolicy())
            span.set(outputs={"origin": getattr(extra, "origin", None)})
        return extra or result

    def respond(self, question: str, chunks: list[RetrievedChunk], language: str, queries: list[str],
                follow: bool = False, earlier: str | None = None, label: str | None = None,
                origin: str | None = None, gate: float | None = None) -> RAGResult:
        """Generate, check and cite an answer from these passages (the library's, a learned or a web search's)."""
        started = time.perf_counter()
        origin = origin or self.origin
        chunks = focus_passages(chunks)
        # the translation line shows the LLM the same question in the other language (not for a follow-up,
        # whose search text also holds the earlier question)
        translation = queries[1] if len(queries) > 1 and not follow else None
        with tracer.span("generate", "llm", inputs={"question": tracer.text(question), "passages": tracer.passages(chunks),
                                                    "language": language, "earlier": tracer.text(earlier)},
                         metadata={"model": getattr(self.generator, "model", ""), "origin": origin}) as span:
            first = self.generator.generate(question, chunks, language, translation=translation,
                                            **({"earlier": earlier} if earlier else {}))
            span.set(outputs={"answer": tracer.text(first)})
        with tracer.span("grounding_check", "chain", inputs={"answer": tracer.text(first)}) as span:
            answer, problem, retried = self._checked(question, chunks, language, first, queries, earlier, gate)
            answer = grounding.polish(self._in_language(answer, chunks, question, language), language)
            span.set(outputs={"answer": tracer.text(answer), "problem": problem, "retried": retried})

        # the language the answer is really in (a failed translation leaves it in the passages' language): the chat
        # app sets its direction and the voice from it, and the safety line is written in it
        written = language
        if grounding.is_refusal(answer) or not answer.strip():
            answer, cited, origin = NO_CONTEXT_ANSWER[language], [], "none"
        else:
            written = detect_language(grounding.answer_body(answer))
            cited = select_passages(answer, chunks)
            if self._about_pesticides(question, answer, language):
                answer = f"{answer}\n\n{SAFETY_NOTE[written]}"
        sources = [f"[{n}] {c.label}" for n, c in cited]
        logger.info(
            "Answered (%s) in %.1fs | lang=%s queries=%d chunks=%d best=%.3f problem=%s retried=%s cited=%d",
            origin, time.perf_counter() - started, written, len(queries), len(chunks),
            max(c.score for c in chunks), problem, retried, len(sources),
        )
        return RAGResult(answer, sources, written, chunks, queries, first, problem, retried, cited, label, origin)

    # ------------------------------------------------------------------ answer checks
    def _in_language(self, answer: str, chunks: list[RetrievedChunk], question: str, language: str) -> str:
        """The answer in the question's language: translated when the LLM wrote it in the passages' language."""
        if not answer.strip() or grounding.is_refusal(answer) \
                or detect_language(grounding.answer_body(answer)) == language:
            return answer
        translate = getattr(self.generator, "translate_answer", None)
        translated = translate(answer, language) if translate else None
        if not translated or detect_language(grounding.answer_body(translated)) != language:
            return answer
        n = len(chunks)
        if set(grounding.citations(translated, n)) != set(grounding.citations(answer, n)) \
                or grounding.unsupported_numbers(translated, chunks, question):
            logger.info("Answer translation rejected (markers or numbers changed)")
            return answer
        logger.info("Answer was in the wrong language: translated to %s", language)
        return translated

    def _checked(
        self, question: str, chunks: list[RetrievedChunk], language: str, first: str, queries: list[str],
        earlier: str | None = None, gate: float | None = None,
    ) -> tuple[str, str | None, bool]:
        """(final answer, problems found in the first answer, whether the LLM was asked again)."""
        translation = queries[1] if len(queries) > 1 and not earlier else None
        context = {"earlier": earlier} if earlier else {}
        answer = grounding.tidy(first, chunks, question)
        found = grounding.problems(answer, chunks, question, language)
        retried = False

        if found and self.self_check:
            retried = True
            if found == ["refusal"]:
                second = self._second_look(question, chunks, language, queries, earlier)
                if second and grounding.grounding_check(second, chunks, question)["grounded"] \
                        and not grounding.problems(second, chunks, question, language) \
                        and grounding.addresses(second, queries):
                    answer = second
            else:
                second = grounding.tidy(
                    self.generator.generate(question, chunks, language, previous=first,
                                            feedback=self._feedback(found, answer, chunks, question, language),
                                            translation=translation, **context),
                    chunks, question,
                )
                if second.strip() and grounding.severity(grounding.problems(second, chunks, question, language)) \
                        < grounding.severity(found):
                    answer = second

        if not grounding.is_refusal(answer):
            # broken text (two languages glued, stuttering, a made-up list) never reaches the farmer
            if grounding.garbled(answer, language, chunks):
                answer = grounding.drop_garbled(answer, language, chunks)
                found = [*found, "broken text removed"]
            # a farmer must never get a number or a statement that is not in the documents
            answer = grounding.drop_unsupported_sentences(answer, chunks, question) if answer else answer
            # nothing left, or still only the question repeated: say so honestly
            if not grounding.has_content(answer) or grounding.is_echo(answer, question):
                answer = grounding.NO_ANSWER[language]
            # The question got past the relevance gate only because it names a crop we have a bulletin
            # for ("the price of tomatoes", "a poem about wheat"): keep the answer only if it is about
            # what was asked, not just about the crop.
            elif max(c.score for c in chunks) < (self.min_similarity if gate is None else gate) \
                    and not grounding.addresses(answer, queries):
                answer = grounding.NO_ANSWER[language]
        return answer, ", ".join(found) or None, retried

    @staticmethod
    def _feedback(found: list[str], answer: str, chunks, question: str, language: str) -> str:
        """One retry message naming every problem found."""
        crops = crops_in(question, language)
        asked, used = grounding.unit_mismatch(answer, question)
        k = 1 if language == "ar" else 0
        values = {
            "unit": " / ".join(grounding.UNIT_NAMES[u][k] for u in sorted(asked)),
            "other": " / ".join(grounding.UNIT_NAMES[u][k] for u in sorted(used)),
            "numbers": ", ".join(grounding.unsupported_numbers(answer, chunks, question)),
            "no_answer": grounding.NO_ANSWER[language],
            "language": LANGUAGE_NAMES[language],
            "crop": " / ".join((t.ar if language == "ar" else t.en) for t in crops),
        }
        return "\n".join(FEEDBACK[p].format(**values) for p in found)

    def _second_look(self, question: str, chunks: list[RetrievedChunk], language: str, queries: list[str],
                     earlier: str | None = None) -> str | None:
        """Ask again with only the two passages that share most words with the question
        (for a comparison, the closest passage about each crop)."""
        asked = set().union(*(grounding.content_words(q) for q in queries))
        if not asked:
            return None
        def overlap(i: int) -> float:
            return len(asked & grounding.content_words(f"{chunks[i].topic}\n{chunks[i].text}")) / len(asked)
        ranked = sorted(range(len(chunks)), key=overlap, reverse=True)
        crops = crops_in(question, language)
        if len(crops) >= 2:  # a comparison: the closest passage about each crop
            best = []
            for crop in crops[:3]:
                own = [i for i in ranked if grounding.is_about(chunks[i], crop) and i not in best]
                best += own[:1]
            best = best or ranked[:2]
        else:
            best = ranked[:2]
        answer = self.generator.generate(
            question, [chunks[i] for i in best], language,
            translation=queries[1] if len(queries) > 1 and not earlier else None,
            **({"earlier": earlier} if earlier else {}),
        )
        answer = grounding.renumber_citations(answer, {k + 1: i + 1 for k, i in enumerate(best)})
        answer = grounding.tidy(answer, chunks, question)
        return None if grounding.is_refusal(answer) else answer

    @staticmethod
    def _about_pesticides(question: str, answer: str, language: str) -> bool:
        text = f"{question}\n{answer}"
        return any(t.en == "pesticides" for t in find_terms(text, language)) or any(
            w in text.lower() for w in ("pesticide", "insecticide", "fungicide", "herbicide", "مبيد")
        )


def get_rag_service(request: Request) -> RAGService:
    """FastAPI dependency: the service loaded once at startup (see main.py)."""
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "startup_error", None)
            or "The document index is not loaded yet.",
        )
    return rag
