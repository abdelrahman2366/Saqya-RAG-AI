from app.main import app
from app.services.generation import NO_CONTEXT_ANSWER, extract_citations, select_sources
from app.services.rag_service import RAGService

from tests.conftest import MAIZE_CHUNK, FakeGenerator, FakeRetriever, make_client


# ---------------------------------------------------------------- /health
def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_store_loaded"] is True
    assert body["num_chunks"] == 1


def test_health_degraded_when_pipeline_not_loaded():
    client = make_client(None)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["vector_store_loaded"] is False


# ---------------------------------------------------------------- /query happy path
def test_query_happy_path(client):
    response = client.post("/query", json={"question": "When should I plant maize?"})
    assert response.status_code == 200
    body = response.json()
    assert "rainy season" in body["answer"]
    assert body["sources"] == ["[1] maize_guide.pdf, p. 3"]


# ---------------------------------------------------------------- invalid input -> 422
def test_query_empty_question_returns_422(client):
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_missing_field_returns_422(client):
    response = client.post("/query", json={"text": "When should I plant maize?"})
    assert response.status_code == 422


def test_query_wrong_type_returns_422(client):
    response = client.post("/query", json={"question": 123})
    assert response.status_code == 422


# ---------------------------------------------------------------- grounding behaviour
def test_query_without_relevant_context_refuses_without_calling_llm():
    low_score_chunk = MAIZE_CHUNK.__class__(**{**MAIZE_CHUNK.__dict__, "score": 0.05})
    generator = FakeGenerator()
    client = make_client(RAGService(FakeRetriever([low_score_chunk]), generator, min_similarity=0.3))

    body = client.post("/query", json={"question": "Who won the World Cup?"}).json()

    assert body["answer"] == NO_CONTEXT_ANSWER["en"]
    assert body["sources"] == []
    assert generator.calls == 0  # the LLM never answers from its own knowledge
    app.state.rag = None


def test_query_returns_503_when_llm_is_down():
    client = make_client(RAGService(FakeRetriever([MAIZE_CHUNK]), FakeGenerator(fail=True)))
    response = client.post("/query", json={"question": "When should I plant maize?"})
    assert response.status_code == 503
    app.state.rag = None


def test_query_returns_503_when_pipeline_not_loaded():
    client = make_client(None)
    response = client.post("/query", json={"question": "When should I plant maize?"})
    assert response.status_code == 503


# ---------------------------------------------------------------- citation helpers
def test_extract_citations_ignores_out_of_range_numbers():
    assert extract_citations("Use 50 kg [2]. Water weekly [1, 2]. See [9].", num_chunks=3) == [2, 1]


def test_refusal_has_no_sources():
    answer = "I don't have enough information in my documents to answer that."
    assert select_sources(answer, [MAIZE_CHUNK]) == []


# ---------------------------------------------------------------- Arabic support
def test_arabic_question_with_no_context_gets_arabic_refusal():
    low_score_chunk = MAIZE_CHUNK.__class__(**{**MAIZE_CHUNK.__dict__, "score": 0.05})
    client = make_client(RAGService(FakeRetriever([low_score_chunk]), FakeGenerator(), min_similarity=0.3))

    body = client.post("/query", json={"question": "من فاز بكأس العالم ٢٠٢٢؟"}).json()

    assert body["answer"] == NO_CONTEXT_ANSWER["ar"]
    assert body["sources"] == []
    app.state.rag = None


def test_arabic_question_is_answered_in_arabic_with_sources():
    generator = FakeGenerator(answer="ازرع الذرة في بداية موسم الأمطار [1].")
    client = make_client(RAGService(FakeRetriever([MAIZE_CHUNK]), generator, min_similarity=0.3))

    body = client.post("/query", json={"question": "متى أزرع الذرة؟"}).json()

    assert generator.last_language == "ar"  # prompt asks the LLM to answer in Arabic
    assert body["sources"] == ["[1] maize_guide.pdf, p. 3"]
    app.state.rag = None


def test_arabic_refusal_from_llm_has_no_sources():
    answer = "لا تتوفر لدي معلومات كافية في مستنداتي للإجابة على هذا السؤال."
    assert select_sources(answer, [MAIZE_CHUNK]) == []


def test_arabic_comma_citations_are_parsed():
    assert extract_citations("استخدم ٥٠ كجم [2، 1].", num_chunks=3) == [2, 1]


def test_language_detection_and_prompt_language():
    from app.services.generation import build_messages
    from app.utils.text import detect_language

    assert detect_language("ما هي أعراض نقص النيتروجين؟") == "ar"
    assert detect_language("How do I store wheat?") == "en"
    messages = build_messages("ما هي أعراض نقص النيتروجين؟", [MAIZE_CHUNK])
    assert "Write the answer in Arabic" in messages[0]["content"]


def test_reasoning_tags_are_removed():
    from app.services.generation import clean_llm_output

    assert clean_llm_output("<think>let me see</think>\nPlant early [1].") == "Plant early [1]."


# ---------------------------------------------------------------- cross-language search
SUGAR_BEET_CHUNK = MAIZE_CHUNK.__class__(
    chunk_id="beet_0005", text="يزرع بنجر السكر إبتداء من أغسطس وحتى منتصف نوفمبر.",
    source="sugar_beet.pdf", page_start=5, page_end=5, score=0.82,
)


def test_english_question_finds_arabic_bulletin_through_translation():
    question = "When should sugar beet be planted?"
    retriever = FakeRetriever([], by_query={"متى يزرع بنجر السكر؟": [SUGAR_BEET_CHUNK]})
    generator = FakeGenerator(answer="Plant sugar beet from August to mid-November [1].",
                              translations={question: "متى يزرع بنجر السكر؟"})
    client = make_client(RAGService(retriever, generator, min_similarity=0.3))

    body = client.post("/query", json={"question": question}).json()

    assert retriever.queries == [question, "متى يزرع بنجر السكر؟"]  # searched in both languages
    assert body["sources"] == ["[1] sugar_beet.pdf, p. 5"]
    app.state.rag = None


def test_translation_can_be_switched_off():
    question = "When should sugar beet be planted?"
    retriever = FakeRetriever([], by_query={"متى يزرع بنجر السكر؟": [SUGAR_BEET_CHUNK]})
    generator = FakeGenerator(translations={question: "متى يزرع بنجر السكر؟"})
    client = make_client(RAGService(retriever, generator, min_similarity=0.3, translate_queries=False))

    body = client.post("/query", json={"question": question}).json()

    assert retriever.queries == [question]
    assert body["answer"] == NO_CONTEXT_ANSWER["en"]  # without translation the bulletin is not found
    app.state.rag = None


def test_failed_translation_falls_back_to_the_original_question():
    retriever = FakeRetriever([MAIZE_CHUNK])
    client = make_client(RAGService(retriever, FakeGenerator(), min_similarity=0.3))  # translate() -> None

    body = client.post("/query", json={"question": "When should I plant maize?"}).json()

    assert retriever.queries == ["When should I plant maize?"]
    assert body["sources"] == ["[1] maize_guide.pdf, p. 3"]
    app.state.rag = None


def test_clean_translation_rejects_wrong_language_and_chatter():
    from app.services.generation import clean_translation

    assert clean_translation('"متى يزرع بنجر السكر؟"\nNote: translated.', "ar") == "متى يزرع بنجر السكر؟"
    assert clean_translation("When is sugar beet planted?", "ar") is None
    assert clean_translation("", "en") is None


# ---------------------------------------------------------------- translation quality guards
def test_translation_with_chinese_characters_is_rejected():
    from app.services.generation import clean_translation

    assert clean_translation("متى应该 زراعة الكيوي في مصر؟", "ar") is None


def test_glossary_finds_terms_in_both_languages():
    from app.services.glossary import find_terms

    assert [t.en for t in find_terms("When should sugar beet be planted in Egypt?", "en")] == ["sugar beet", "planting date"]
    assert "faba bean" in [t.en for t in find_terms("كم كيلو تقاوي يحتاج فدان الفول البلدي؟", "ar")]


def test_wrong_crop_from_llm_is_corrected_by_the_glossary():
    from app.services.glossary import add_missing_terms, find_terms

    terms = find_terms("When should sugar beet be planted?", "en")
    fixed = add_missing_terms("متى يزرع الكيوي؟", terms, "ar")  # the LLM said kiwi
    assert "بنجر السكر" in fixed and "ميعاد الزراعة" in fixed


class _FakeOllamaReply:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


def test_translate_falls_back_to_glossary_terms_when_llm_output_is_unusable():
    from app.services.generation import OllamaGenerator

    gen = OllamaGenerator(host="http://localhost:1", model="fake")
    gen._client = type("C", (), {"chat": lambda self, **kw: _FakeOllamaReply("متى应该 زراعة الكيوي")})()
    assert gen.translate("When should sugar beet be planted?", "ar") == "بنجر السكر ميعاد الزراعة"


def test_translate_keeps_a_good_llm_translation():
    from app.services.generation import OllamaGenerator

    gen = OllamaGenerator(host="http://localhost:1", model="fake")
    gen._client = type("C", (), {"chat": lambda self, **kw: _FakeOllamaReply("متى يزرع بنجر السكر في مصر؟")})()
    out = gen.translate("When should sugar beet be planted in Egypt?", "ar")
    assert out.startswith("متى يزرع بنجر السكر في مصر؟")
