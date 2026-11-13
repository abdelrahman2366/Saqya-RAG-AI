"""Voice (Egyptian dialect, transcription, speech) and uploaded documents. No Whisper, no internet, no Ollama:
the speech models, the embedding model and the LLM are replaced by small fakes."""

import hashlib
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.dialect import is_egyptian, to_msa
from app.services.documents import DocumentNotFoundError, DocumentService, MemoryCollection
from app.services.extraction import ExtractionError
from app.services.rag_service import RAGService
from app.services.speech import SpeechUnavailableError, TextToSpeech, Transcript, speech_text

from tests.conftest import MAIZE_CHUNK, FakeGenerator, FakeRetriever

# ---------------------------------------------------------------- Egyptian dialect


@pytest.mark.parametrize("spoken, standard", [
    ("إمتى أزرع الأوطة؟", "متى أزرع الطماطم؟"),
    ("أعمل إيه للدود اللي في الدرة؟", "ماذا أفعل للديدان الذي في الذرة؟"),
    ("كام شيكارة سماد للفدان؟", "كم شيكارة سماد للفدان؟"),
    ("ازاي أكافح المن في البتنجان", "كيف أكافح المن في الباذنجان"),
    ("إيه أحسن ميعاد لزراعة الدرة الرفيعة؟", "ما أفضل ميعاد لزراعة الذرة الرفيعة؟"),
])
def test_egyptian_questions_become_standard_arabic(spoken, standard):
    assert to_msa(spoken) == standard
    assert is_egyptian(spoken)


@pytest.mark.parametrize("text", ["ما هو أفضل ميعاد لزراعة القمح؟", "When should I plant wheat?", "الذين يزرعون الذرة"])
def test_standard_arabic_and_english_are_left_alone(text):
    assert to_msa(text) == text


def test_rag_service_searches_and_answers_the_standard_question():
    retriever, generator = FakeRetriever([MAIZE_CHUNK]), FakeGenerator()
    RAGService(retriever, generator, top_k=4, min_similarity=0.3, translate_queries=False).answer("إمتى أزرع الدرة؟")
    assert retriever.queries[0] == "متى أزرع الذرة؟"


# ---------------------------------------------------------------- speech helpers


def test_speech_text_drops_citations_markdown_and_emoji():
    text = "**Plant** in March [1].\n- Thin after 20 days [2][3].\n\n⚠️ Follow the label."
    assert speech_text(text) == "Plant in March. Thin after 20 days. Follow the label."


def test_voice_choice():
    tts = TextToSpeech()
    assert tts.voice_for("ar", "female") == "ar-EG-SalmaNeural"
    assert tts.voice_for("ar", "male") == "ar-EG-ShakirNeural"
    assert tts.voice_for("en", "nonsense") == "en-US-JennyNeural"
    assert tts.voice_for("ar", "ar-EG-ShakirNeural") == "ar-EG-ShakirNeural"


class FakeSTT:
    available = True
    model_name = "fake-whisper"

    def __init__(self, text="إمتى أزرع الأوطة؟", fail=False):
        self.text, self.fail = text, fail

    def transcribe(self, audio, language=None):
        if self.fail:
            raise SpeechUnavailableError("model download failed")
        self.last = (audio, language)
        return Transcript(text=self.text, language="ar", duration=2.5, seconds=0.4)


class FakeTTS:
    available = True

    def __init__(self, fail=False):
        self.fail = fail

    async def synthesize(self, text, language="ar", voice="female"):
        if self.fail:
            raise SpeechUnavailableError("no internet")
        self.last = (text, language, voice)
        return b"ID3fake-mp3"

    async def stream(self, text, language="ar", voice="female"):
        if self.fail:
            raise SpeechUnavailableError("no internet")
        if not speech_text(text):
            raise ValueError("Nothing to read.")
        self.last = (text, language, voice)
        for part in (b"ID3", b"fake-", b"mp3"):
            yield part


@pytest.fixture
def voice_client():
    app.state.rag = None
    app.state.startup_error = None
    app.state.stt, app.state.tts = FakeSTT(), FakeTTS()
    yield TestClient(app)
    app.state.stt = app.state.tts = None


def test_transcribe_returns_text_and_standard_arabic(voice_client):
    r = voice_client.post("/voice/transcribe", files={"audio": ("q.wav", b"RIFFfake", "audio/wav")}, data={"language": "ar"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "إمتى أزرع الأوطة؟"
    assert body["standard"] == "متى أزرع الطماطم؟"
    assert body["dialect"] is True and body["language"] == "ar"
    assert app.state.stt.last == (b"RIFFfake", "ar")


def test_transcribe_errors(voice_client):
    assert voice_client.post("/voice/transcribe", files={"audio": ("q.wav", b"", "audio/wav")}).status_code == 422
    app.state.stt = FakeSTT(fail=True)
    assert voice_client.post("/voice/transcribe", files={"audio": ("q.wav", b"x", "audio/wav")}).status_code == 503
    app.state.stt = None
    r = voice_client.post("/voice/transcribe", files={"audio": ("q.wav", b"x", "audio/wav")})
    assert r.status_code == 503 and "faster-whisper" in r.json()["detail"]


def test_speak_returns_mp3(voice_client):
    r = voice_client.post("/voice/speak", json={"text": "تزرع الطماطم في يناير [1].", "voice": "male"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    assert r.content == b"ID3fake-mp3"
    assert app.state.tts.last[1:] == ("ar", "male")


def test_speak_without_internet_is_503(voice_client):
    app.state.tts = FakeTTS(fail=True)
    assert voice_client.post("/voice/speak", json={"text": "hello"}).status_code == 503


def test_stream_speaks_live(voice_client):
    r = voice_client.get("/voice/stream", params={"text": "تزرع الطماطم في يناير [1].", "voice": "male"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    assert r.content == b"ID3fake-mp3"
    assert app.state.tts.last[1:] == ("ar", "male")
    assert "max-age" in r.headers["cache-control"]
    r = voice_client.get("/voice/stream", params={"text": "Plant in March.", "language": "en"})
    assert r.status_code == 200 and app.state.tts.last[1] == "en"


def test_stream_errors(voice_client):
    assert voice_client.get("/voice/stream", params={"text": "[1]"}).status_code == 422   # nothing to read
    assert voice_client.get("/voice/stream", params={"text": "x" * 1501}).status_code == 422
    app.state.tts = FakeTTS(fail=True)
    assert voice_client.get("/voice/stream", params={"text": "hello"}).status_code == 503
    app.state.tts = None
    assert voice_client.get("/voice/stream", params={"text": "hello"}).status_code == 503


def test_stream_with_an_engine_that_cannot_stream(voice_client):
    class WholeClip:
        available = True

        async def synthesize(self, text, language="ar", voice="female"):
            return b"ID3whole"

    app.state.tts = WholeClip()
    assert voice_client.get("/voice/stream", params={"text": "hello"}).content == b"ID3whole"


def test_health_reports_voice_features(voice_client):
    body = voice_client.get("/health").json()
    assert body["speech_to_text"] is True and body["text_to_speech"] is True
    assert body["documents"] is False and body["whisper_model"] == "fake-whisper"


# ---------------------------------------------------------------- uploaded documents


class HashEmbedder:
    """Bag-of-words vectors: texts sharing words are close. Enough to test search without a model."""

    dim = 512

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


GUIDE = """# Tomato Growing Guide

## Planting date
Tomatoes are planted in the summer season from January to February and in the autumn season from July to August.
Transplants should be four to six weeks old with five true leaves when moved to the field.

## Fertilizing
Add 20 cubic meters of farmyard manure per feddan when preparing the soil, with 200 kg of superphosphate.
Nitrogen is split into three doses after transplanting, the last one before flowering ends.

## Irrigation
Irrigate lightly after transplanting and then every 7 to 10 days depending on the soil and the weather.
Avoid flooding the field because wet soil spreads root rot and wilt diseases.
"""


@pytest.fixture
def doc_service(tmp_path):
    return DocumentService(HashEmbedder(), FakeGenerator(answer="Tomatoes are planted from January to February [1]."),
                           tmp_path / "uploads", min_similarity=0.1, translate_queries=False, embedding_model="hash",
                           make_pdf=False)


@pytest.fixture
def doc_client(doc_service):
    app.state.rag = None
    app.state.startup_error = None
    app.state.documents = doc_service
    yield TestClient(app)
    app.state.documents = None


def test_upload_reads_sections_and_topic(doc_client):
    r = doc_client.post("/documents", files={"file": ("tomato_guide.md", GUIDE.encode(), "text/markdown")})
    assert r.status_code == 201
    doc = r.json()
    assert doc["title"] == "Tomato Growing Guide"
    assert doc["sections"] == ["Planting date", "Fertilizing", "Irrigation"]
    assert doc["topic"] == "الطماطم / tomato"
    assert doc["chunks"] >= 3 and doc["language"] == "en"
    assert doc_client.get("/documents").json()["documents"][0]["id"] == doc["id"]


def test_question_about_the_document_is_answered_from_it(doc_client):
    doc_id = doc_client.post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")}).json()["id"]
    r = doc_client.post(f"/documents/{doc_id}/query", json={"question": "When are tomatoes planted?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False and body["passages"]
    assert body["passages"][0]["source"] == "g.md"
    assert body["passages"][0]["text"].startswith("Tomato Growing Guide › ")


def test_explain_whole_document_and_one_section(doc_client, doc_service):
    doc_id = doc_client.post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")}).json()["id"]
    r = doc_client.post(f"/documents/{doc_id}/explain", json={})
    assert r.status_code == 200 and r.json()["passages"]
    sampled = doc_service.generator.last_explain
    assert sampled["focus"] is None and sampled["title"] == "Tomato Growing Guide"
    assert {c.section for c in sampled["chunks"]} == {"Planting date", "Fertilizing", "Irrigation"}
    r = doc_client.post(f"/documents/{doc_id}/explain", json={"focus": "Irrigation", "language": "ar"})
    assert r.status_code == 200
    sampled = doc_service.generator.last_explain
    assert sampled["language"] == "ar" and {c.section for c in sampled["chunks"]} == {"Irrigation"}


@pytest.mark.parametrize("question, overview", [
    ("Explain this document", True),
    ("اشرح لي الملف ده", True),
    ("لخص المستند", True),
    ("What is this document about?", True),
    ("ما هي كمية السماد للفدان؟", False),
    ("Explain the fertilizer doses for tomatoes", False),
])
def test_overview_questions(question, overview):
    assert DocumentService.is_overview(question) is overview


def test_overview_question_is_routed_to_explain(doc_client, doc_service):
    doc_id = doc_client.post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")}).json()["id"]
    doc_client.post(f"/documents/{doc_id}/query", json={"question": "اشرح لي المستند ده"})
    assert doc_service.generator.last_explain["language"] == "ar"


def test_documents_are_temporary_and_per_session(doc_client, doc_service, tmp_path):
    a = doc_client.post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")},
                        data={"session": "session-aaaa"}).json()["id"]
    doc_client.post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")}, data={"session": "session-bbbb"})
    assert [d["id"] for d in doc_client.get("/documents", params={"session": "session-aaaa"}).json()["documents"]] == [a]
    assert doc_client.delete("/documents", params={"session": "session-aaaa"}).json() == {"deleted": 1}
    assert doc_client.get(f"/documents/{a}").status_code == 404
    assert len(doc_client.get("/documents", params={"session": "session-bbbb"}).json()["documents"]) == 1
    # nothing survives a restart: a new service knows no documents, and the folder is emptied
    fresh = DocumentService(HashEmbedder(), FakeGenerator(), tmp_path / "uploads", embedding_model="hash")
    assert fresh.list("session-bbbb") == []
    (tmp_path / "uploads" / "keep.txt").write_text("not ours")
    fresh.reset_storage()
    assert list((tmp_path / "uploads" / "sessions").iterdir()) == []
    assert (tmp_path / "uploads" / "keep.txt").exists()          # only its own sub-folder is ever emptied


def test_unused_documents_expire(doc_service):
    doc = doc_service.add(GUIDE.encode(), "g.md", "session-cccc")
    doc.last_used -= doc_service.ttl_seconds + 1
    assert doc_service.expire() == 1
    with pytest.raises(DocumentNotFoundError):
        doc_service.get(doc.id)


def test_documents_per_session_limit(doc_service):
    doc_service.per_session = 2
    ids = [doc_service.add(GUIDE.encode(), "g.md", "session-dddd").id for _ in range(3)]
    assert {d["id"] for d in doc_service.list("session-dddd")} == set(ids[1:])


def test_clean_pdf_of_an_attached_document(doc_client, doc_service):
    doc_service.make_pdf = True
    info = doc_client.post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")},
                           data={"session": "session-eeee"}).json()
    if not info["clean_pdf"]:
        pytest.skip("no PDF renderer here")
    r = doc_client.get(f"/documents/{info['id']}/clean.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF") and r.headers["content-type"] == "application/pdf"
    assert info["cleaning"]["method"] == "plain" and info["cleaning"]["renderer"] in ("chromium", "mupdf")
    doc_client.delete("/documents", params={"session": "session-eeee"})
    assert doc_client.get(f"/documents/{info['id']}/clean.pdf").status_code == 404


def test_follow_up_question_keeps_the_document_context(doc_client, doc_service):
    doc_id = doc_client.post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")}).json()["id"]
    r = doc_client.post(f"/documents/{doc_id}/query",
                        json={"question": "How often should it be watered?", "history": ["When are tomatoes planted?"]})
    assert r.status_code == 200
    assert doc_service.generator.last_earlier == "When are tomatoes planted?"


def test_top_passages_bring_their_continuation(doc_service):
    doc = doc_service.add(GUIDE.encode(), "g.md")
    first = doc.retrieved([0])
    expanded = doc.with_neighbours(first)
    same_section = doc.chunks[1]["metadata"].get("section") == doc.chunks[0]["metadata"].get("section")
    assert len(expanded) == (2 if same_section else 1)


def test_delete_and_missing_documents(doc_client):
    doc_id = doc_client.post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")}).json()["id"]
    assert doc_client.delete(f"/documents/{doc_id}").status_code == 204
    assert doc_client.get(f"/documents/{doc_id}").status_code == 404
    assert doc_client.post(f"/documents/{doc_id}/query", json={"question": "When to plant?"}).status_code == 404
    assert doc_client.get("/documents/../../etc").status_code == 404


def test_upload_errors(doc_client):
    assert doc_client.post("/documents", files={"file": ("photo.jpg", b"\xff\xd8", "image/jpeg")}).status_code == 415
    assert doc_client.post("/documents", files={"file": ("empty.txt", b"", "text/plain")}).status_code == 422
    assert doc_client.post("/documents", files={"file": ("tiny.txt", b"too short", "text/plain")}).status_code == 422


def test_upload_without_the_index_is_503():
    app.state.rag = app.state.documents = None
    app.state.startup_error = "Vector store missing"
    r = TestClient(app).post("/documents", files={"file": ("g.md", GUIDE.encode(), "text/markdown")})
    assert r.status_code == 503


def test_service_limits(doc_service):
    doc_service.max_chunks = 1
    with pytest.raises(ExtractionError, match="too long"):
        doc_service.add(GUIDE.encode(), "g.md")
    with pytest.raises(DocumentNotFoundError):
        doc_service.get("0123456789ab")


def test_memory_collection_matches_chroma_shape():
    emb = HashEmbedder().encode(["wheat planting", "tomato irrigation"])
    col = MemoryCollection("t", ["a", "b"], ["wheat planting", "tomato irrigation"],
                           [{"topic": "x"}, {"topic": "y"}], emb)
    res = col.query(query_embeddings=[emb[1]], n_results=2, include=["documents", "metadatas", "distances"])
    assert res["ids"][0][0] == "b" and abs(res["distances"][0][0]) < 1e-5
    assert col.query(query_embeddings=[emb[1]], n_results=2, where={"topic": "x"})["ids"] == [["a"]]
    assert col.get(ids=["a"], include=["embeddings"])["ids"] == ["a"]


def test_recommendations_come_from_the_documents_advice(doc_service):
    doc = doc_service.add(GUIDE.encode(), "tomato_guide.md", "session-r")
    doc_service.generator.answer = "- Add 20 cubic meters of farmyard manure per feddan [1].\n- Irrigate every 7 to 10 days [2]."
    result = doc_service.explain(doc.id, "recommendations", "en")
    assert doc_service.generator.last_explain["focus"] == "recommendations"
    assert result.origin == "document" and result.sources and "manure" in result.answer
