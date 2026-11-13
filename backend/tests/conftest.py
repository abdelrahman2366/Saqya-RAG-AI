"""Test fixtures: a fake retriever and LLM so tests need neither the vector store nor Ollama."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.generation import LLMUnavailableError
from app.services.rag_service import RAGService
from app.services.retrieval import RetrievedChunk

MAIZE_CHUNK = RetrievedChunk(
    chunk_id="maize_guide_0003",
    text="Plant maize at the start of the rainy season when the soil is moist.",
    source="maize_guide.pdf",
    page_start=3,
    page_end=3,
    score=0.71,
)


class FakeRetriever:
    """Returns `chunks` for every query, or only for the queries listed in `by_query`."""

    def __init__(self, chunks, by_query=None):
        self.chunks = chunks
        self.by_query = by_query
        self.queries = []

    def retrieve(self, question, top_k=4, min_similarity=0.0):
        self.queries.append(question)
        pool = self.chunks if self.by_query is None else self.by_query.get(question, [])
        return [c for c in pool if c.score >= min_similarity][:top_k]

    def retrieve_many(self, queries, top_k=4, min_similarity=0.0, topic=None):
        self.last_topic = topic
        best = {}
        for q in queries:
            for c in self.retrieve(q, top_k, min_similarity):
                if topic and not c.topic.endswith(topic):
                    continue
                best.setdefault(c.chunk_id, c)
        return list(best.values())[:top_k]

    def topic_label(self, key):
        labels = {c.topic.split(" / ")[-1]: c.topic for c in self.chunks if c.topic}
        return labels.get((key or "").lower())

    def library(self):
        return {
            "num_documents": len({c.source for c in self.chunks}),
            "num_chunks": len(self.chunks),
            "languages": {"en": len(self.chunks)},
            "crops": [{"key": "maize", "ar": "الذرة الشامية", "en": "maize", "documents": 1, "chunks": 1}],
            "documents": [{"source": c.source, "title": "Maize guide", "topic": c.topic, "language": "en",
                           "chunks": 1, "pages": c.page_end} for c in self.chunks],
            "embedding_model": "fake-embedder",
            "built_at": "2026-11-12T00:00:00+00:00",
        }

    def count(self):
        return len(self.chunks)


class FakeGenerator:
    model = "fake-llm"

    def __init__(self, answer="Plant maize at the start of the rainy season [1].", fail=False, translations=None,
                 retry_answer=None, scope_reply="yes", web_query_text=None, general_text=None):
        self.answer = answer
        self.retry_answer = retry_answer
        self.fail = fail
        self.calls = 0
        self.translations = translations or {}
        self.scope_reply = scope_reply        # beyond the documents: "yes"/"no" to "is it about agriculture?"
        self.web_query_text = web_query_text  # the short web search query it writes
        self.general_text = general_text      # its general-knowledge answer
        self.scope_prompts = []
        self.general_calls = 0

    def translate(self, question, target):
        return self.translations.get(question)

    def status(self):
        return (not self.fail, not self.fail)

    def generate(self, question, chunks, language=None, previous=None, feedback=None, translation=None, earlier=None):
        self.calls += 1
        self.last_language = language
        self.last_earlier = earlier
        self.last_question = question
        self.last_feedback = feedback
        self.last_translation = translation
        self.last_chunks = list(chunks)
        if self.fail:
            raise LLMUnavailableError("Cannot reach Ollama")
        if self.calls > 1 and self.retry_answer is not None:  # a retry or a second look
            return self.retry_answer
        return self.answer

    def translate_answer(self, answer, target):
        self.translated_to = target
        return self.translations.get(answer)

    relevance_reply = None   # "yes"/"no" to "does this answer the question?" (None: not judged)
    judged: list = []

    def answers_question(self, question, answer):
        if self.relevance_reply is None:
            return None
        self.judged = [*self.judged, question]
        return self.relevance_reply == "yes"

    def classify_scope(self, prompt):
        self.scope_prompts.append(prompt)
        if self.scope_reply is None:
            raise LLMUnavailableError("Cannot reach Ollama")
        return self.scope_reply

    def web_query(self, question, earlier=None, today=None):
        return self.web_query_text

    def general_answer(self, question, language, earlier=None, today=None):
        self.general_calls += 1
        return self.general_text or ""

    def explain(self, chunks, language, title, focus=None):
        self.calls += 1
        self.last_explain = {"chunks": list(chunks), "language": language, "title": title, "focus": focus}
        if self.fail:
            raise LLMUnavailableError("Cannot reach Ollama")
        return self.answer


def make_client(rag):
    # TestClient without `with` does not run the lifespan, so the real
    # vector store / Ollama are never loaded; we inject our fake service.
    app.state.rag = rag
    app.state.startup_error = None if rag else "Vector store missing"
    return TestClient(app)


@pytest.fixture
def fake_generator():
    return FakeGenerator()


@pytest.fixture
def client(fake_generator):
    rag = RAGService(FakeRetriever([MAIZE_CHUNK]), fake_generator, top_k=4, min_similarity=0.3)
    yield make_client(rag)
    app.state.rag = None
