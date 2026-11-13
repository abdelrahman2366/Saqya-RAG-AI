"""Attached documents are cleaned like the corpus (cleaning.py), recordings are decoded without faster-whisper's
PyAV call, follow-up questions keep their context, and answers come back in the question's language."""

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.conversation import follow_up_context
from app.services.documents import prepare_document
from app.services.extraction import chunk_document
from app.services.rag_service import RAGService
from app.services.speech import decode_audio
from tests.conftest import MAIZE_CHUNK, FakeGenerator, FakeRetriever, make_client

DATA = Path(__file__).parent / "data"
pymupdf = pytest.importorskip("pymupdf")


# ---------------------------------------------------------------- cleaning an attached PDF
def test_messy_arabic_bulletin_is_cleaned():
    prepared = prepare_document(DATA / "messy_tomato_ar.pdf", "messy_tomato_ar.pdf")
    assert prepared.method == "cleaned"
    assert prepared.doc.title == "زراعة وإنتاج الطماطم في الأراضي الجديدة"        # from the cover, not the file name
    assert {1, 2, 3} <= set(prepared.dropped_pages)                              # cover, credits, contents
    assert ["مقدمة", "ميعاد الزراعة", "الري", "التسميد", "الآفات"] == prepared.doc.sections
    chunks = chunk_document(prepared.doc)
    text = "\n".join(c["text"] for c in chunks)
    for noise in ("ISBN", "رقم الإيداع", "وزارة الزراعة", "المراجع", "Springer", "نشرة فنية رقم 17", "........"):
        assert noise not in text
    assert "الكمية للفدان: 50 كجم" in text                                      # a table row, one line per row
    assert min(c["metadata"]["page_start"] for c in chunks) >= 4                # pages of the original file
    assert all(c["text"].startswith("زراعة وإنتاج الطماطم") for c in chunks)    # title › section headers


def test_short_last_page_with_content_is_kept():
    prepared = prepare_document(DATA / "messy_wheat_en.pdf", "messy_wheat_en.pdf")
    assert prepared.recovered_pages == [6]
    text = "\n".join(c["text"] for c in chunk_document(prepared.doc))
    assert "Irrigate every 5 to 7 days" in text and "60 kg of seed per feddan" in text
    assert "Copyright" not in text and "thank the staff" not in text
    assert prepared.doc.title == "Wheat Production in Newly Reclaimed Lands"


def test_text_files_are_read_plainly_with_an_outline_for_the_pdf(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Onion notes\n\n## Planting\n" + "Plant onion sets in October on ridges 60 cm apart. " * 8)
    prepared = prepare_document(path, "notes.md")
    assert prepared.method == "plain" and prepared.outline["outline"][0] == ("h2", 2, "1. Planting")


# ---------------------------------------------------------------- recordings
def _wav(rate=16000, channels=1, seconds=1.0):
    t = np.arange(int(rate * seconds)) / rate
    x = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    if channels == 2:
        x = np.repeat(x, 2)
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return b.getvalue()


def test_recordings_are_decoded_to_16k_mono():
    for data in (_wav(), _wav(44100, 2)):
        samples = decode_audio(data)
        assert samples.dtype == np.float32 and abs(samples.size - 16000) <= 1 and 0.2 < np.abs(samples).max() <= 0.31


def test_other_formats_go_through_pyav():
    av = pytest.importorskip("av")
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp3") as out:
        stream = out.add_stream("mp3", rate=16000, layout="mono")
        t = np.arange(16000) / 16000
        frame = av.AudioFrame.from_ndarray((0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)[None, :],
                                           format="s16", layout="mono")
        frame.sample_rate = 16000
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    samples = decode_audio(buf.getvalue())
    assert samples.size > 12000 and np.abs(samples).max() > 0.1


def test_whisper_gets_samples_not_a_file():
    from app.services.speech import SpeechToText

    class Model:
        def transcribe(self, audio, **kwargs):
            self.audio = audio

            class Info:
                language, duration = "en", 1.0
            return iter([type("S", (), {"text": " Hello"})()]), Info()

    stt = SpeechToText()
    stt._model = Model()
    result = stt.transcribe(_wav(), "en")
    assert isinstance(stt._model.audio, np.ndarray) and result.text == "Hello"


# ---------------------------------------------------------------- follow-ups
@pytest.mark.parametrize("question, history, previous", [
    ("How much water does it need?", ["When should I plant tomatoes?"], "When should I plant tomatoes?"),
    ("What about cucumbers?", ["When should I plant tomatoes?"], "When should I plant tomatoes?"),
    ("طب والخيار؟", ["إمتى أزرع الطماطم؟"], "إمتى أزرع الطماطم؟"),
    ("الكمية؟", ["إمتى أزرع الطماطم؟"], "إمتى أزرع الطماطم؟"),
    ("What is the seed rate for wheat?", ["When should I plant tomatoes?"], None),
    ("ما هو ميعاد زراعة القمح؟", ["إمتى أزرع الطماطم؟"], None),
    ("How much water does it need?", [], None),
])
def test_follow_up_detection(question, history, previous):
    found = follow_up_context(question, history)
    assert (found.previous if found else None) == previous


def test_follow_up_is_searched_with_the_earlier_question():
    retriever, generator = FakeRetriever([MAIZE_CHUNK]), FakeGenerator()
    rag = RAGService(retriever, generator, top_k=4, min_similarity=0.0, translate_queries=False)
    rag.answer("How much water does it need?", history=["When should maize be planted?"])
    assert retriever.queries == ["When should maize be planted? How much water does it need?"]
    assert generator.last_earlier == "When should maize be planted?"


def test_query_route_passes_the_history():
    generator = FakeGenerator()
    client = make_client(RAGService(FakeRetriever([MAIZE_CHUNK]), generator, top_k=4, min_similarity=0.0,
                                    translate_queries=False))
    r = client.post("/query", json={"question": "And how deep?", "history": ["When should maize be planted?"]})
    assert r.status_code == 200 and generator.last_earlier == "When should maize be planted?"


# ---------------------------------------------------------------- the answer's language
def test_answer_in_the_wrong_language_is_translated():
    english = "Plant maize at the start of the rainy season [1]."
    arabic = "ازرع الذرة في بداية موسم الأمطار [1]."
    generator = FakeGenerator(answer=english, translations={english: arabic})
    rag = RAGService(FakeRetriever([MAIZE_CHUNK]), generator, top_k=4, min_similarity=0.0, translate_queries=False,
                     self_check=False)
    result = rag.answer("متى أزرع الذرة؟")
    assert result.answer.startswith("ازرع الذرة") and result.language == "ar" and generator.translated_to == "ar"


def test_translation_that_changes_markers_or_numbers_is_rejected():
    english = "Plant maize at the start of the rainy season [1]."
    generator = FakeGenerator(answer=english, translations={english: "ازرع الذرة بعد 45 يوما [2]."})
    rag = RAGService(FakeRetriever([MAIZE_CHUNK]), generator, top_k=4, min_similarity=0.0, translate_queries=False,
                     self_check=False)
    assert rag.answer("متى أزرع الذرة؟").answer == english


def test_prompt_ends_with_the_language_in_that_language():
    from app.services.generation import build_messages

    ar = build_messages("متى أزرع الذرة؟", [MAIZE_CHUNK])[-1]["content"]
    en = build_messages("When should I plant maize?", [MAIZE_CHUNK])[-1]["content"]
    assert ar.rstrip().endswith("لا تستخدم أي لغة أخرى.") and en.rstrip().endswith("Do not use any other language.")
