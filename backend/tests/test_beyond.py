"""Beyond the documents (scope check, web search, learned knowledge, general knowledge) and the voice styles.
No internet, no Ollama, no embedding model: the search engine, web pages, LLM and embedder are small fakes."""

import asyncio
import base64
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import speech
from app.services.beyond import Beyond, SourcePolicy
from app.services.dialect import to_egyptian
from app.services.documents import DocumentService
from app.services.learning import KnowledgeStore, good_passage
from app.services.rag_service import RAGService
from app.services.retrieval import RetrievedChunk
from app.services.scope import is_agricultural, looks_agricultural, parse_verdict
from app.services.speech import GeminiSpeech, SpeechUnavailableError, TextToSpeech, spoken_text
from app.services.web_search import (WebHit, WebPage, WebSearcher, html_to_text, is_blocked, is_trusted, page_passages,
                                     split_passages)

from tests.conftest import MAIZE_CHUNK, FakeGenerator, FakeRetriever
from tests.test_voice_documents import GUIDE, FakeTTS, HashEmbedder

RABBIT_PAGE = ("Rabbits are raised in clean, dry hutches with good ventilation. Each doe needs about 0.5 square metres "
               "of cage floor. Give rabbits fresh water at all times, and feed them pellets with green fodder such as "
               "clover. Breeding does can be mated at five to six months of age.")
OTHER_PAGE = ("Football is a team sport played between two teams of eleven players. The ball is moved by kicking and "
              "the team with the most goals at the end of the match wins the game in front of the crowd.")


# ---------------------------------------------------------------- Egyptian spoken style
@pytest.mark.parametrize("formal, egyptian", [
    ("يجب أن تروى الطماطم كل يومين، ولكن لا يوجد داعٍ للري في الشتاء.",
     "لازم تروى الطماطم كل يومين، بس مفيش داعٍ للري في الشتاء."),
    ("سوف تزرع القمح بعد ذلك في نوفمبر لأن الحرارة مناسبة.", "هتزرع القمح بعد كده في نوفمبر عشان الحرارة مناسبة."),
    ("كيف أعرف متى ينضج البطيخ؟ عندما يجف المحلاق.", "إزاي أعرف إمتى ينضج البطيخ؟ لما يجف المحلاق."),
    ("هذه الآفة التي تصيب الأوراق تكافح بالمصائد فقط.", "دي الآفة اللي تصيب الأوراق تكافح بالمصائد بس."),
])
def test_answers_are_spoken_with_egyptian_words(formal, egyptian):
    assert to_egyptian(formal) == egyptian


def test_numbers_units_and_english_are_not_touched():
    assert to_egyptian("يضاف 50 كجم سلفات بوتاسيوم للفدان.") == "يضاف 50 كجم سلفات بوتاسيوم للفدان."
    assert to_egyptian("Plant in March [1].") == "Plant in March [1]."
    assert spoken_text("يجب أن تروى [1].", "ar", "formal") == "يجب أن تروى."
    assert spoken_text("يجب أن تروى [1].", "ar", "egyptian") == "لازم تروى."
    assert spoken_text("Water it [1].", "en", "egyptian") == "Water it."


def test_voice_catalog():
    tts = TextToSpeech()
    assert tts.voice_for("ar", "hamed") == "ar-SA-HamedNeural"
    assert tts.voice_for("en", "hamed") == "en-US-GuyNeural"        # English answers: the same gender
    assert tts.voice_for("ar", "kore") == "ar-EG-SalmaNeural"       # a Gemini voice asked of Edge (fallback)
    gemini = GeminiSpeech("key")
    assert gemini.voice_for("charon") == "Charon" and gemini.voice_for("shakir") == "Charon"
    assert gemini.voice_for("female") == "Kore" and gemini.voice_for("nonsense") == "Kore"


def test_gemini_request_and_audio():
    gemini = GeminiSpeech("key")
    body = gemini.request_body("ازرع القمح في نوفمبر.", "ar", "orus", "egyptian")
    text = body["contents"][0]["parts"][0]["text"]
    assert "Egyptian Arabic" in text and text.endswith(": ازرع القمح في نوفمبر.")
    assert body["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Orus"
    assert "Modern Standard Arabic" in gemini.request_body("x", "ar", "kore", "formal")["contents"][0]["parts"][0]["text"]
    pcm = b"\x00\x01" * 2400
    wav = GeminiSpeech.audio_from({"candidates": [{"content": {"parts": [{"inlineData": {
        "mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(pcm).decode()}}]}}]})
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and wav.endswith(pcm)
    with pytest.raises(SpeechUnavailableError):
        GeminiSpeech.audio_from({"candidates": []})
    assert not GeminiSpeech(None).available and not GeminiSpeech("  ").available


class FakeHTTP:
    """Stands in for httpx.AsyncClient: counts the calls, answers with one clip (or an error)."""
    calls = 0
    status = 200

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        FakeHTTP.calls += 1
        FakeHTTP.last = (url, headers, json)
        data = base64.b64encode(b"\x00\x00" * 100).decode()

        class Response:
            status_code = FakeHTTP.status
            text = "quota exceeded"

            def json(self_inner):
                return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;rate=24000",
                                                                              "data": data}}]}}]}
        return Response()


def test_gemini_speaks_and_caches(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeHTTP)
    FakeHTTP.calls, FakeHTTP.status = 0, 200
    gemini = GeminiSpeech("secret", model="gemini-test")
    first = asyncio.run(gemini.synthesize("تروى الطماطم كل يومين.", "ar", "kore", "formal"))
    again = asyncio.run(gemini.synthesize("تروى الطماطم كل يومين.", "ar", "kore", "formal"))
    assert first == again and first[:4] == b"RIFF" and FakeHTTP.calls == 1   # listen again: from the cache
    url, headers, _ = FakeHTTP.last
    assert "gemini-test:generateContent" in url and headers["x-goog-api-key"] == "secret"
    FakeHTTP.status = 429
    with pytest.raises(SpeechUnavailableError, match="429"):
        asyncio.run(gemini.synthesize("جملة أخرى.", "ar", "kore", "formal"))


class FailingGemini:
    name, media_type, styles, available = "gemini", "audio/wav", True, True

    async def stream(self, text, language="ar", voice="female", style="egyptian"):
        raise SpeechUnavailableError("quota")
        yield b""


class WorkingGemini(FailingGemini):
    async def stream(self, text, language="ar", voice="female", style="egyptian"):
        self.last = (text, language, voice, style)
        yield b"RIFFfake-wav"


@pytest.fixture
def voice_api():
    app.state.rag = None
    app.state.startup_error = None
    app.state.tts, app.state.tts_gemini = FakeTTS(), None
    yield TestClient(app)
    app.state.tts = app.state.tts_gemini = None


def test_stream_styles(voice_api):
    voice_api.get("/voice/stream", params={"text": "يجب أن تروى الطماطم [1].", "style": "egyptian", "voice": "shakir"})
    assert app.state.tts.last == ("لازم تروى الطماطم.", "ar", "shakir")
    voice_api.get("/voice/stream", params={"text": "يجب أن تروى الطماطم [1].", "style": "formal"})
    assert app.state.tts.last[0] == "يجب أن تروى الطماطم."


def test_gemini_engine_and_fallback_to_edge(voice_api):
    app.state.tts_gemini = WorkingGemini()
    r = voice_api.get("/voice/stream", params={"text": "ازرع القمح.", "engine": "gemini", "voice": "kore", "style": "formal"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content == b"RIFFfake-wav"
    assert app.state.tts_gemini.last == ("ازرع القمح.", "ar", "kore", "formal")
    app.state.tts_gemini = FailingGemini()   # quota used up: the Edge voice takes over
    r = voice_api.get("/voice/stream", params={"text": "ازرع القمح.", "engine": "gemini"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and r.content == b"ID3fake-mp3"
    r = voice_api.post("/voice/speak", json={"text": "Plant wheat.", "engine": "gemini"})
    assert r.status_code == 200 and r.content == b"ID3fake-mp3"


# ---------------------------------------------------------------- scope
@pytest.mark.parametrize("reply, verdict", [("yes", True), ("Yes.", True), ("نعم", True), ("no", False),
                                            ("No, it is not.", False), ("لا", False), ("maybe", None), ("", None)])
def test_scope_verdicts(reply, verdict):
    assert parse_verdict(reply) is verdict


@pytest.mark.parametrize("question, farming", [
    ("give me a crop recommendation", True), ("How do I raise rabbits at home?", True),
    ("إزاي أربي أرانب في البيت؟", True), ("ما أفضل سماد للفول؟", True), ("أفضل وقت لرش المبيد", True),
    ("Who won the 2022 World Cup?", False), ("ما هي عاصمة فرنسا؟", False), ("Write me a poem about love", False),
])
def test_scope_word_list(question, farming):
    assert looks_agricultural(question) is farming


def test_scope_asks_the_llm_first_and_falls_back_to_the_word_list():
    assert is_agricultural("tomato soup recipe", FakeGenerator(scope_reply="no")) is False
    assert is_agricultural("How do I raise rabbits?", FakeGenerator(scope_reply=None)) is True   # Ollama down
    assert is_agricultural("Who won the World Cup?", FakeGenerator(scope_reply="I think")) is False


# ---------------------------------------------------------------- web pages
def test_page_text_and_passages():
    html = ("<html><head><title>T</title><script>var x=1;</script></head><body><nav>Home | About</nav>"
            f"<article><h1>Raising rabbits</h1><p>{RABBIT_PAGE}</p><p>{RABBIT_PAGE.upper()}</p></article>"
            "<footer>Copyright 2026. All rights reserved.</footer></body></html>")
    text = html_to_text(html, "https://example.org/rabbits")
    assert "hutches" in text and "var x" not in text
    passages = split_passages("\n\n".join([RABBIT_PAGE] * 5) + "\n\nSubscribe to our newsletter")
    assert passages and all(len(p) <= 900 for p in passages) and not any("newsletter" in p for p in passages)


def test_sites_are_filtered_and_trusted_ones_come_first():
    assert is_blocked("youtube.com") and is_blocked("m.facebook.com") and is_blocked("pinterest.co.uk")
    assert is_trusted("fao.org") and is_trusted("www.agri.cairo.edu.eg".removeprefix("www.")) and is_trusted("extension.umn.edu")
    assert not is_trusted("myfarmblog.com")
    rows = [{"href": "https://myfarmblog.com/rabbits", "title": "Blog", "body": "rabbits"},
            {"href": "https://www.youtube.com/watch?v=1", "title": "Video", "body": "rabbits"},
            {"href": "https://www.fao.org/rabbits", "title": "FAO", "body": "rabbits"},
            {"href": "https://myfarmblog.com/rabbits#top", "title": "Blog again", "body": "rabbits"}]
    searcher = WebSearcher(search_fn=lambda q, lang, n: rows)
    hits = searcher.ranked_hits([("rabbits", "en")])
    assert [h.url for h in hits] == ["https://www.fao.org/rabbits", "https://myfarmblog.com/rabbits"]


def test_pages_are_read_or_snippets_used():
    rows = [{"href": "https://a.org/1", "title": "A", "body": "A snippet about rabbit hutches, clean water and feeding clover to the does every day."},
            {"href": "https://b.org/2", "title": "B", "body": "short"}]
    reader = WebSearcher(search_fn=lambda q, lang, n: rows, fetch_fn=lambda url: RABBIT_PAGE if url.endswith("1") else "")
    pages = reader.pages([("rabbits", "en")])
    assert [p.url for p in pages] == ["https://a.org/1"] and pages[0].site == "a.org" and not pages[0].from_snippet
    blocked = WebSearcher(search_fn=lambda q, lang, n: rows, fetch_fn=lambda url: "")
    pages = blocked.pages([("rabbits", "en")])
    assert len(pages) == 1 and pages[0].from_snippet   # only the long snippet is usable
    meta = page_passages(pages)[0]["metadata"]
    assert meta["kind"] == "web" and meta["url"] == "https://a.org/1" and meta["source"] == "a.org"


# ---------------------------------------------------------------- learned knowledge
def store(tmp_path, **kwargs):
    return KnowledgeStore(tmp_path / "learned", HashEmbedder(), embedding_model="hash", min_similarity=0.2, **kwargs)


def web_chunk(text=RABBIT_PAGE, url="https://www.fao.org/rabbits", n=0):
    return RetrievedChunk(f"web-{n}", text, "fao.org", 0, 0, 0.8, url=url, title="Raising rabbits", kind="web")


def test_learning_keeps_good_new_passages_only(tmp_path):
    knowledge = store(tmp_path)
    assert knowledge.add_web([web_chunk()], "How do I raise rabbits?") == 1
    assert knowledge.add_web([web_chunk(n=1)], "again") == 0                       # the same text
    assert knowledge.add_web([web_chunk(RABBIT_PAGE + " ")], "again") == 0         # near-duplicate
    assert knowledge.add_web([web_chunk("Too short.", url="https://x.org")], "q") == 0
    not_web = RetrievedChunk("c1", OTHER_PAGE, "x.pdf", 1, 1, 0.9)
    assert knowledge.add_web([not_web], "q") == 0                                  # only web passages
    assert not good_passage("| -- | -- | -- | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |")
    found = knowledge.search(["how many square metres does a rabbit doe need in a cage"])
    assert found and found[0].kind == "learned" and found[0].url == "https://www.fao.org/rabbits"
    assert found[0].label == "Raising rabbits (fao.org)"
    assert knowledge.search(["football match goals"]) == []


def test_learning_survives_a_restart_and_can_forget(tmp_path):
    knowledge = store(tmp_path)
    knowledge.add_web([web_chunk()], "rabbits")
    knowledge.add_web([web_chunk(OTHER_PAGE, url="https://sport.example/1", n=2)], "football")
    again = store(tmp_path)
    assert again.count() == 2
    sources = again.sources()
    assert {s["url"] for s in sources} == {"https://www.fao.org/rabbits", "https://sport.example/1"}
    rabbits = next(s for s in sources if "fao" in s["url"])
    assert rabbits["kind"] == "web" and rabbits["passages"] == 1 and rabbits["question"] == "rabbits"
    assert again.forget(rabbits["id"]) == 1 and again.count() == 1
    assert store(tmp_path).count() == 1
    assert again.forget_all() == 1 and store(tmp_path).count() == 0


def test_learning_drops_the_least_used_when_full(tmp_path):
    knowledge = store(tmp_path, max_passages=2)
    texts = [RABBIT_PAGE, OTHER_PAGE, "Sugar beet is sown in autumn in rows sixty centimetres apart and thinned "
             "after four weeks to one plant every twenty centimetres along the row for the best root size."]
    knowledge.add_web([web_chunk(texts[0], "https://a.org", 0)], "q")
    knowledge.used([RetrievedChunk(knowledge._rows[0]["id"], "", "", 0, 0, 0)])
    knowledge.add_web([web_chunk(texts[1], "https://b.org", 1)], "q")
    knowledge.add_web([web_chunk(texts[2], "https://c.org", 2)], "q")
    assert knowledge.count() == 2 and {s["url"] for s in knowledge.sources()} == {"https://a.org", "https://c.org"}


def test_new_embedding_model_rebuilds_the_vectors(tmp_path):
    store(tmp_path).add_web([web_chunk()], "rabbits")
    renewed = KnowledgeStore(tmp_path / "learned", HashEmbedder(), embedding_model="another-model", min_similarity=0.2)
    for _ in range(100):
        if renewed.summary()["ready"]:
            break
        time.sleep(0.02)
    assert renewed.search(["Rabbits are raised in clean, dry hutches with good ventilation"])
    assert json.loads((tmp_path / "learned" / "meta.json").read_text())["embedding_model"] == "another-model"


def test_attached_agricultural_documents_are_learned(tmp_path):
    knowledge = store(tmp_path)
    docs = DocumentService(HashEmbedder(), FakeGenerator(scope_reply="no"), tmp_path / "up", min_similarity=0.1,
                           translate_queries=False, embedding_model="hash", make_pdf=False, knowledge=knowledge)
    kept = docs.add(GUIDE.encode(), "tomato_guide.md", "session-1", learn=True)
    assert kept.learned >= 3 and kept.summary()["learned"] == kept.learned     # a tomato guide: its crop was found
    source = knowledge.sources()[0]
    assert source["kind"] == "document" and source["title"] == "Tomato Growing Guide"
    assert docs.add(GUIDE.encode(), "copy.md", "session-1", learn=True).learned == 0   # already known
    other = "# Club news\n\n" + OTHER_PAGE + "\n\n" + OTHER_PAGE.replace("Football", "The league")
    assert docs.add(other.encode(), "club.md", "session-1", learn=True).learned == 0   # not about farming
    assert docs.add(GUIDE.replace("Tomato", "Pepper").encode(), "p.md", "session-1", learn=False).learned == 0
    docs.delete_session("session-1")
    assert knowledge.count() == kept.learned   # the session's copy is gone, what was learned stays


# ---------------------------------------------------------------- the tiers
def beyond_service(tmp_path, generator, pages=(), knowledge=None, general=True, library=()):
    fetched = {p.url: p.text for p in pages}
    web = WebSearcher(search_fn=lambda q, lang, n: [{"href": p.url, "title": p.title, "body": ""} for p in pages],
                      fetch_fn=lambda url: fetched.get(url, ""))
    web.searched = []
    original = web.pages

    def recording(pairs):
        web.searched.append(pairs)
        return original(pairs)
    web.pages = recording
    beyond = Beyond(HashEmbedder(), web=web, knowledge=knowledge, general=general, web_min_similarity=0.1,
                    learned_min_similarity=0.1)
    rag = RAGService(FakeRetriever(list(library)), generator, min_similarity=0.3, translate_queries=False, beyond=beyond)
    return rag, web


RABBIT_SITE = WebPage(url="https://www.fao.org/rabbits", title="Raising rabbits", text=RABBIT_PAGE)


def test_library_answers_first(tmp_path):
    gen = FakeGenerator()
    rag, web = beyond_service(tmp_path, gen, [RABBIT_SITE], library=[MAIZE_CHUNK])
    result = rag.answer("When should I plant maize?")
    assert result.origin == "library" and web.searched == [] and gen.scope_prompts == []


def test_not_about_farming_is_still_refused(tmp_path):
    gen = FakeGenerator(scope_reply="no", general_text="Argentina won.")
    rag, web = beyond_service(tmp_path, gen, [RABBIT_SITE])
    result = rag.answer("Who won the 2022 World Cup?")
    assert result.origin == "none" and not result.sources and web.searched == [] and gen.general_calls == 0


def test_web_answer_is_cited_and_learned_then_reused(tmp_path):
    knowledge = store(tmp_path)
    gen = FakeGenerator(answer="Give each doe about 0.5 square metres of cage floor [1].",
                        web_query_text="rabbit cage size per doe")
    rag, web = beyond_service(tmp_path, gen, [RABBIT_SITE], knowledge=knowledge)
    result = rag.answer("How much cage space does a rabbit doe need?")
    assert result.origin == "web" and result.sources == ["[1] Raising rabbits (fao.org)"]
    assert result.cited[0][1].url == "https://www.fao.org/rabbits" and result.learned == 1
    assert web.searched[0][0] == ("rabbit cage size per doe", "en")       # the LLM's short query first
    # asked again: answered from what was learned, without searching the web
    again = rag.answer("How much cage space does a rabbit doe need?")
    assert again.origin == "learned" and len(web.searched) == 1
    assert knowledge.summary()["passages"] == 1


def test_farmer_can_turn_off_learning_and_the_web(tmp_path):
    knowledge = store(tmp_path)
    gen = FakeGenerator(answer="Give each doe about 0.5 square metres of cage floor [1].",
                        general_text="Rabbits need clean, dry hutches. Confirm with your extension officer.")
    rag, web = beyond_service(tmp_path, gen, [RABBIT_SITE], knowledge=knowledge)
    result = rag.answer("How much cage space does a rabbit doe need?", policy=SourcePolicy(learn=False))
    assert result.origin == "web" and knowledge.count() == 0
    result = rag.answer("How much cage space does a rabbit doe need?", policy=SourcePolicy(web=False))
    assert result.origin == "general" and "not from a document" in result.note and not result.sources
    result = rag.answer("How much cage space does a rabbit doe need?", policy=SourcePolicy(web=False, general=False))
    assert result.origin == "none"


def test_irrelevant_web_pages_are_not_used(tmp_path):
    gen = FakeGenerator(answer="Football has eleven players [1].", general_text=None)
    sport = WebPage(url="https://sport.example/1", title="Football", text=OTHER_PAGE)
    rag, _ = beyond_service(tmp_path, gen, [sport])
    rag.beyond.web_min_similarity = 0.5
    assert rag.answer("How do I treat mites on rabbits?").origin == "none"


def test_follow_up_scope_uses_the_earlier_question(tmp_path):
    gen = FakeGenerator(general_text="Usually every two to three days in sandy soil; confirm locally.")
    rag, _ = beyond_service(tmp_path, gen, [], general=True)
    rag.answer("and how often?", history=["How should I water tomatoes in sandy soil?"])
    assert "How should I water tomatoes in sandy soil?" in gen.scope_prompts[-1]


# ---------------------------------------------------------------- API
@pytest.fixture
def beyond_api(tmp_path):
    knowledge = store(tmp_path)
    gen = FakeGenerator(answer="Give each doe about 0.5 square metres of cage floor [1].",
                        general_text="Rabbits need clean, dry hutches. Confirm with your extension officer.")
    rag, _ = beyond_service(tmp_path, gen, [RABBIT_SITE], knowledge=knowledge)
    app.state.rag, app.state.startup_error = rag, None
    yield TestClient(app)
    app.state.rag = None


def test_query_reports_where_the_answer_came_from(beyond_api):
    body = beyond_api.post("/query", json={"question": "How much cage space does a rabbit doe need?"}).json()
    assert body["origin"] == "web" and body["refused"] is False and body["learned"] == 1
    passage = body["passages"][0]
    assert passage["url"] == "https://www.fao.org/rabbits" and passage["kind"] == "web" and passage["pages"] == ""
    assert beyond_api.delete("/knowledge").json() == {"removed": 1}
    body = beyond_api.post("/query", json={"question": "كيف أحصن الأغنام ضد مرض تعفن الحافر؟",
                                           "web": False}).json()
    assert body["origin"] == "general" and body["refused"] is False and body["note"]
    health = beyond_api.get("/health").json()
    assert health["web_search"] and health["learning"] and health["learned_passages"] == 0 and health["general_knowledge"]


def test_knowledge_endpoints(beyond_api):
    beyond_api.post("/query", json={"question": "How much cage space does a rabbit doe need?"})
    knowledge = beyond_api.get("/knowledge").json()
    assert knowledge["enabled"] and knowledge["passages"] == 1 and knowledge["web_sources"] == 1
    source = knowledge["sources"][0]
    assert source["url"] == "https://www.fao.org/rabbits" and source["title"] == "Raising rabbits"
    assert beyond_api.delete(f"/knowledge/sources/{source['id']}").json() == {"removed": 1}
    assert beyond_api.delete(f"/knowledge/sources/{source['id']}").status_code == 404
    assert beyond_api.delete("/knowledge").json() == {"removed": 0}


# ---------------------------------------------------------------- broad farming questions
def test_clear_farming_words_need_no_llm():
    gen = FakeGenerator(scope_reply="no")   # a small model saying "no" to an obvious farming question
    assert is_agricultural("give me more advice on farming", gen) is True and gen.scope_prompts == []
    assert is_agricultural("أنا فلاح جديد، انصحني", gen) is True
    assert is_agricultural("tomato soup recipe", gen) is False   # a crop name alone: the LLM decides


def test_general_answer_that_opens_with_a_hedge_is_kept(tmp_path):
    gen = FakeGenerator(general_text="I don't have enough information about your farm. However, start with a soil "
                                     "test, choose crops for the season, and keep records of costs. Confirm with "
                                     "your local agricultural extension officer.")
    rag, _ = beyond_service(tmp_path, gen, [])
    result = rag.answer("give me more advice on farming")
    assert result.origin == "general" and result.answer.startswith("Start with a soil test")
    assert "extension officer" in result.answer and "enough information" not in result.answer


def test_farming_question_nothing_found_says_so(tmp_path):
    gen = FakeGenerator(general_text="I don't have enough information to answer.")
    rag, _ = beyond_service(tmp_path, gen, [])
    result = rag.answer("give me more advice on farming")
    assert result.origin == "none" and "or on the web" in result.answer


# ---------------------------------------------------------------- broken output, mis-heard words, the crop calendar
@pytest.mark.parametrize("broken", [
    "اختلف مواعيد زراعة الطماطم حسب العروة، وفقاً للpassesات:",
    "اختلفت مواعيد زراعة الطماطم حسب العروة المناسبة، وفقاً للمرفق و و و و من نشرة الطماطم 2025",
    "الشوفان هو نوع من الحبوب ي tolerant للجفاف والبرد.",
    "Mâche\nعشب الخس\nعشب الخضرة\nعشب الخضر\nعشب الخضروات",
    "اكتب الإجابة كاملة باللغة العربية، حتى لو كانت الفقرات بالإنجليزية.",
])
def test_broken_text_is_caught(broken):
    from app.services import grounding

    assert grounding.garbled(broken, "ar")


@pytest.mark.parametrize("sound", [
    "تزرع الطماطم في العروة الصيفية من أواخر يناير حتى أوائل مارس [1].",
    "- القمح\n- الفول البلدي\n- بنجر السكر\n- البصل",
    "درجة pH المناسبة من 6 إلى 7 [1].", "Sow wheat from 15 to 30 November [1].",
])
def test_sound_text_is_not_flagged(sound):
    from app.services import grounding

    assert not grounding.garbled(sound, "ar" if "ا" in sound else "en")


def test_a_broken_answer_is_asked_again_and_cleaned():
    bad = "تزرع الذرة حسب الpassages و و و و [1]."
    good = "تزرع الذرة في بداية موسم الأمطار [1]."
    arabic = RetrievedChunk("m", "تزرع الذرة في بداية موسم الأمطار.", "maize.pdf", 3, 3, 0.8)
    gen = FakeGenerator(answer=bad, retry_answer=good)
    rag = RAGService(FakeRetriever([arabic]), gen, min_similarity=0.3, translate_queries=False)
    result = rag.answer("متى تزرع الذرة؟")
    assert result.answer == good and "garbled" in result.problem
    never = FakeGenerator(answer=bad, retry_answer=bad)   # still broken: refused, never shown
    assert RAGService(FakeRetriever([arabic]), never, min_similarity=0.3, translate_queries=False) \
        .answer("متى تزرع الذرة؟").origin == "none"


@pytest.mark.parametrize("heard, meant", [
    ("إيه أفضل مع حصيل أقدر أزرعها في شهر نوفمبر؟", "إيه أفضل محاصيل أقدر أزرعها في شهر نوفمبر؟"),
    ("متى أزرع الطماطمم؟", "متى أزرع الطماطم؟"), ("how much fertlizer per feddan?", "how much fertilizer per feddan?"),
])
def test_misheard_and_mistyped_farming_words_are_repaired(heard, meant):
    from app.services.spelling import repair

    assert repair(heard) == meant


@pytest.mark.parametrize("fine", ["إمتى أزرع القمح مع الفول؟", "كام شيكارة سماد للفدان؟", "إزاي أكافح الدودة في الأوطة؟",
                                  "When should I plant wheat in Egypt?", "What is BLITECAST for tomato blight?"])
def test_correct_questions_are_never_changed(fine):
    from app.services.spelling import repair

    assert repair(fine) == fine


def test_what_to_plant_is_written_from_the_bulletins_facts(tmp_path):
    from app.services.grounding import NO_ANSWER

    # the LLM writes the answer to the farmer's own wording, from the bulletins' dates (no web search)
    gen = FakeGenerator(answer="يمكنك زراعة الكتان في النصف الأول من نوفمبر [1]. والقمح من 15 إلى 30 نوفمبر [2].")
    rag, web = beyond_service(tmp_path, gen, [RABBIT_SITE])
    result = rag.answer("إيه أفضل مع حصيل أقدر أزرعها في شهر نوفمبر؟")      # mis-heard, Egyptian, no crop named
    assert result.origin == "library" and web.searched == [] and result.answer.startswith("يمكنك زراعة الكتان")
    assert "نوفمبر" in gen.last_question and "محاصيل" in gen.last_question          # repaired, with the situation
    assert result.cited[0][1].source == "الكتان2020_compressed.pdf" and "ما نوع أرضك" in result.answer   # asked by the code
    # the LLM has nothing sound to say (or is down): the plain list from the bulletins still answers
    for broken in (FakeGenerator(answer=NO_ANSWER["ar"], retry_answer=NO_ANSWER["ar"]), FakeGenerator(fail=True)):
        rag, web = beyond_service(tmp_path, broken, [RABBIT_SITE])
        result = rag.answer("إيه أفضل محاصيل أقدر أزرعها في شهر نوفمبر؟")
        assert "القمح: من 15 إلى 30 نوفمبر" in result.answer and "الشوفان" not in result.answer and web.searched == []
        assert any(c.source == "NEW_wheat_new.pdf" and c.page_start == 6 for _, c in result.cited)
    from app.services.crop_calendar import is_calendar_question
    assert is_calendar_question("متى أزرع القمح في نوفمبر؟") is None      # one crop: its own bulletin answers
    assert is_calendar_question("May I ask about tomato diseases?") is None


def test_an_answer_beside_the_point_is_not_given(tmp_path):
    # a vague question found passages, but the answer written from them is not what was asked
    off = RetrievedChunk("p", "Plough old lands twice to a depth of 30 cm before sowing.", "soil.pdf", 4, 4, 0.7)
    gen = FakeGenerator(answer="Plough old lands twice to a depth of 30 cm [1].",
                        general_text="Try drought-tolerant crops such as barley; confirm with your extension officer.")
    gen.relevance_reply = "no"
    rag, _ = beyond_service(tmp_path, gen, [], library=[off])
    result = rag.answer("How can I make my farm more profitable?")
    assert result.origin == "general" and "barley" in result.answer and gen.judged
    gen.relevance_reply = "yes"
    assert rag.answer("How can I make my farm more profitable?").origin == "library"
    gen.relevance_reply, gen.judged = "no", []
    assert rag.answer("When should I plant maize?").origin == "library" and gen.judged == []   # names a crop: not judged


def test_gemini_rests_after_its_quota_is_used_up(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeHTTP)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(speech.asyncio, "sleep", lambda *_: real_sleep(0))
    FakeHTTP.calls, FakeHTTP.status = 0, 429
    gemini = GeminiSpeech("secret")
    assert gemini.available
    with pytest.raises(SpeechUnavailableError):
        asyncio.run(gemini.synthesize("جملة.", "ar", "kore", "formal"))
    assert not gemini.available            # the free voice (Edge) reads everything until the quota is back
    calls = FakeHTTP.calls
    FakeHTTP.status = 200
    gemini.rest_until = 0                  # the rest is over: Gemini is used again
    assert gemini.available and asyncio.run(gemini.synthesize("جملة.", "ar", "kore", "formal"))[:4] == b"RIFF"
    assert FakeHTTP.calls == calls + 1


@pytest.mark.parametrize("question", ["أديني محاصيل جديدة أقدر أزرعها في أرضي.", "اقترح علي محاصيل جديدة",
                                      "عايز نباتات جديده ازرعها في ارضي", "give me a crop recommendation",
                                      "what should I plant?"])
def test_asking_for_crops_gets_this_months_crops_and_a_question_back(question):
    from datetime import datetime
    from app.services import crop_calendar

    assert crop_calendar.is_calendar_question(question, datetime(2026, 11, 13)) == 11
    text, passages = crop_calendar.answer(question, "ar" if "ا" in question else "en")
    assert passages and ("ما نوع أرضك" in text or "is your soil sandy, clay or calcareous" in text)


@pytest.mark.parametrize("question", ["ما أفضل سماد لمحاصيل الخضر؟", "اقترح علي مبيد للمحاصيل", "عايز ازرع طماطم",
                                      "What crops are grown in Egypt?", "give me more advice on farming",
                                      "I want to know about plant diseases", "كيف أزرع؟"])
def test_other_questions_are_not_taken_for_crop_suggestions(question):
    from app.services import crop_calendar

    assert crop_calendar.is_calendar_question(question) is None


def test_the_farmers_land_narrows_the_crop_list():
    from datetime import datetime
    from app.services import crop_calendar

    november = datetime(2026, 11, 13)
    text, passages = crop_calendar.answer("أرضي رملية في المنيا", "ar", ["اقترح علي محاصيل جديدة"], november)
    assert text.startswith("لأرضك الرملية في المنيا") and "الكانتلوب" in text and "الوجه القبلي" in text
    assert "لا تحدد نوع الأرض: الكتان" in text                     # never guessed: listed apart
    assert any(c.section == "الأرض المناسبة" and c.source == "الكانتلوب.pdf" for c in passages)
    clay, _ = crop_calendar.answer("I have clay soil, what should I plant in March?", "en", None, november)
    assert "- Cotton: 1–15 March" in clay and "Which governorate" in clay
    sandy, _ = crop_calendar.answer("sandy soil, what should I plant in March?", "en", None, november)
    assert "advise against these in sandy soil: Cotton" in sandy
    # not a reply about the land: another crop's question, or another subject
    assert crop_calendar.answer("أرضي رملية", "ar", ["متى أزرع القمح؟"], november) is None
    assert crop_calendar.answer("التربة الرملية تحتاج سماد إيه؟", "ar", ["اقترح علي محاصيل"], november) is None


def test_a_suggestion_that_breaks_down_is_replaced_by_the_plain_list(tmp_path):
    """qwen slipping into Chinese, items without a crop, or "does not suit this month": the bulletins' list is shown."""
    from app.services import grounding

    chinese = ("- يمكن زراعته من 15 ديسمبر حتى أول يناير [1].\n- الكانتلوب لا تناسب هذا الشهر [2].\n"
               "ما نوع الأرض المناسبة لك在那里添加了阿拉伯语答案后，我注意到您希望保持简洁")
    assert grounding.garbled(chinese, "ar") and "在" not in grounding.polish(chinese, "ar")
    gen = FakeGenerator(answer=chinese, retry_answer=chinese)
    rag, _ = beyond_service(tmp_path, gen, [])
    result = rag.answer("اقترح علي محاصيل جديدة")
    assert "在" not in result.answer and "لا تناسب هذا الشهر" not in result.answer
    assert "القمح: من 15 إلى 30 نوفمبر" in result.answer


def test_only_the_months_own_season_is_given():
    from datetime import datetime
    from app.services import crop_calendar

    november = datetime(2026, 11, 13)
    text, _ = crop_calendar.answer("ماذا أزرع الآن؟", "ar", None, november)
    assert "الطماطم: العروة المحيرة تحت الأقبية البلاستيكية، من أواخر نوفمبر إلى أوائل ديسمبر" in text
    assert "الصيفية المبكرة" not in text and "قصب السكر" not in text
    vegetables, _ = crop_calendar.answer("عايز خضار أزرعه دلوقتي", "ar", None, november)
    assert "الكوسة" in vegetables and "القمح" not in vegetables and "الكتان" not in vegetables
