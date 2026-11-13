"""Features used by the chat UI: crop focus, cited passages, the library and answer feedback."""

import json

from app.core.config import get_settings
from app.main import app
from app.services.generation import NO_CONTEXT_ANSWER
from app.services.rag_service import RAGService
from app.services.retrieval import RetrievedChunk

from tests.conftest import MAIZE_CHUNK, FakeGenerator, FakeRetriever, make_client

SUNFLOWER = RetrievedChunk(
    chunk_id="sunflower_0007",
    text="زراعة وانتاج زهرة الشمس › معدل التقاوي\nفي حالة الزراعة اليدوية 4-5 كجم بذرة للفدان.",
    source="نشرة-دوار-الشمس.pdf",
    page_start=3,
    page_end=3,
    score=0.66,
    topic="دوار الشمس / sunflower",
    section="معدل التقاوي",
)


# ---------------------------------------------------------------- /query: new response fields
def test_query_returns_cited_passages_matching_sources(client):
    body = client.post("/query", json={"question": "When should I plant maize?"}).json()
    assert body["sources"] == ["[1] maize_guide.pdf, p. 3"]
    assert len(body["passages"]) == 1
    passage = body["passages"][0]
    assert passage["n"] == 1 and passage["source"] == "maize_guide.pdf" and passage["pages"] == "p. 3"
    assert "rainy season" in passage["text"]
    assert body["language"] == "en" and body["refused"] is False
    assert isinstance(body["latency_ms"], int)


def test_refusal_is_flagged_and_has_no_passages():
    low = RetrievedChunk(**{**MAIZE_CHUNK.__dict__, "score": 0.05})
    client = make_client(RAGService(FakeRetriever([low]), FakeGenerator(), min_similarity=0.3))
    body = client.post("/query", json={"question": "Who won the World Cup?"}).json()
    assert body["answer"] == NO_CONTEXT_ANSWER["en"]
    assert body["refused"] is True and body["passages"] == []
    app.state.rag = None


def test_passage_carries_section_and_topic():
    generator = FakeGenerator(answer="في الزراعة اليدوية 4-5 كجم بذرة للفدان [1].")
    client = make_client(RAGService(FakeRetriever([SUNFLOWER]), generator, min_similarity=0.3))
    body = client.post("/query", json={"question": "كم كيلو تقاوي لفدان دوار الشمس؟"}).json()
    passage = body["passages"][0]
    assert passage["section"] == "معدل التقاوي"
    assert passage["topic"] == "دوار الشمس / sunflower"
    assert body["language"] == "ar"
    app.state.rag = None


# ---------------------------------------------------------------- crop focus
def test_topic_focus_is_passed_to_the_retriever():
    retriever = FakeRetriever([MAIZE_CHUNK, SUNFLOWER])
    generator = FakeGenerator(answer="في الزراعة اليدوية 4-5 كجم بذرة للفدان [1].")
    client = make_client(RAGService(retriever, generator, min_similarity=0.3))
    body = client.post("/query", json={"question": "كم كيلو تقاوي للفدان؟", "topic": "sunflower"}).json()
    assert retriever.last_topic == "sunflower"
    assert body["topic"] == "دوار الشمس / sunflower"
    assert [c.source for c in generator.last_chunks] == ["نشرة-دوار-الشمس.pdf"]
    app.state.rag = None


def test_unknown_topic_is_ignored():
    retriever = FakeRetriever([MAIZE_CHUNK])
    client = make_client(RAGService(retriever, FakeGenerator(), min_similarity=0.3))
    body = client.post("/query", json={"question": "When should I plant maize?", "topic": "bananas"}).json()
    assert retriever.last_topic is None
    assert body["topic"] is None and body["sources"]
    app.state.rag = None


def test_topic_too_long_returns_422(client):
    response = client.post("/query", json={"question": "When should I plant maize?", "topic": "x" * 81})
    assert response.status_code == 422


# ---------------------------------------------------------------- /library
def test_library_lists_documents_and_crops(client):
    body = client.get("/library").json()
    assert body["num_documents"] == 1 and body["num_chunks"] == 1
    assert body["documents"][0]["source"] == "maize_guide.pdf"
    assert body["crops"][0]["key"] == "maize"


def test_library_returns_503_without_vector_store():
    client = make_client(None)
    assert client.get("/library").status_code == 503


# ---------------------------------------------------------------- /feedback
def test_feedback_is_appended_as_json_lines(client, tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "feedback_file", tmp_path / "feedback.jsonl")
    first = client.post("/feedback", json={"question": "When should I plant maize?",
                                           "answer": "At the start of the rainy season [1].", "rating": "up",
                                           "sources": ["[1] maize_guide.pdf, p. 3"]})
    second = client.post("/feedback", json={"question": "q?", "answer": "a", "rating": "down", "comment": "wrong page"})
    assert first.status_code == 200 and first.json() == {"status": "saved", "total": 1, "helpful": 1}
    assert second.json()["total"] == 2 and second.json()["helpful"] == 1
    lines = [json.loads(l) for l in (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["rating"] == "up" and lines[1]["comment"] == "wrong page" and "time" in lines[0]


def test_feedback_rejects_unknown_rating(client):
    response = client.post("/feedback", json={"question": "q?", "answer": "a", "rating": "meh"})
    assert response.status_code == 422


# ---------------------------------------------------------------- retriever helpers (no Chroma needed)
def test_topic_label_accepts_english_arabic_or_full_label():
    from app.services.retrieval import Retriever

    r = Retriever.__new__(Retriever)
    r.topics = {"sunflower": "دوار الشمس / sunflower"}
    assert r.topic_label("Sunflower") == "دوار الشمس / sunflower"
    assert r.topic_label("دوار الشمس") == "دوار الشمس / sunflower"
    assert r.topic_label("دوار الشمس / sunflower") == "دوار الشمس / sunflower"
    assert r.topic_label("rice") is None and r.topic_label(None) is None


def test_library_summary_from_chunk_metadata():
    from app.services.retrieval import Retriever

    r = Retriever.__new__(Retriever)
    r.embedding_model, r.config = "BAAI/bge-m3", {"built_at": "today", "document_titles": {}}
    r._metas = [
        {"source": "sun.pdf", "topic": "دوار الشمس / sunflower", "lang": "ar", "page_end": 4, "title": "زهرة الشمس"},
        {"source": "sun.pdf", "topic": "دوار الشمس / sunflower", "lang": "ar", "page_end": 9},
        {"source": "farm.pdf", "topic": "", "lang": "en", "page_end": 2},
    ]
    lib = r.library()
    assert lib["num_documents"] == 2 and lib["num_chunks"] == 3
    assert lib["languages"] == {"ar": 2, "en": 1}
    assert lib["crops"] == [{"key": "sunflower", "ar": "دوار الشمس", "en": "sunflower", "documents": 1, "chunks": 2}]
    sun = next(d for d in lib["documents"] if d["source"] == "sun.pdf")
    assert sun["pages"] == 9 and sun["title"] == "زهرة الشمس" and sun["chunks"] == 2
