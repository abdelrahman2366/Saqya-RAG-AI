"""Answer checks (grounding.py), keyword search (lexical.py), hybrid retrieval and the glossary.

No model or Ollama needed: a tiny bag-of-words embedder stands in for the embedding model and
Chroma runs in memory.
"""

import re

import numpy as np
import pytest

from app.services import grounding
from app.services.generation import NO_CONTEXT_ANSWER
from app.services.rag_service import SAFETY_NOTE, RAGService
from app.services.retrieval import RetrievedChunk, Retriever

from tests.conftest import FakeGenerator, FakeRetriever


def chunk(i, text, source="bulletin.pdf", page=1, score=0.8, topic=""):
    return RetrievedChunk(f"c{i}", text, source, page, page, score, topic)


FABA = chunk(1, "معدل التقاوي: من 40-30كجم للفدان وتزرع الحبوب في جور على ريشتي الخط.", "faba.pdf", 7,
             topic="الفول البلدي / faba bean")
POTATO = chunk(2, "تجب إضافة 20 إلى 30 طنا من السماد العضوي الحيواني للهكتار.", "potato.pdf", 7,
               topic="البطاطس / potato")
WHEAT = chunk(3, "Sow wheat from 15 to 30 November in Lower Egypt.", "wheat.pdf", 6)
SOY = chunk(5, "يحتاج الفدان نحو 30 كجم تقاوي فول صويا معتمدة.", "soybean.pdf", 5, topic="فول الصويا / soybean")


def service(answer, retry_answer=None, chunks=(FABA,)):
    gen = FakeGenerator(answer=answer, retry_answer=retry_answer)
    return RAGService(FakeRetriever(list(chunks)), gen, min_similarity=0.3, translate_queries=False), gen


# ---------------------------------------------------------------- answer checks in the pipeline
def test_good_answer_is_not_retried():
    rag, gen = service("يحتاج الفدان من 30 إلى 40 كجم تقاوي [1].")
    result = rag.answer("كم كيلو تقاوي يحتاج فدان الفول البلدي؟")
    assert gen.calls == 1 and not result.retried
    assert result.sources == ["[1] faba.pdf, p. 7"]


def test_repeated_question_is_retried():
    rag, gen = service("كم كيلو تقاوي يحتاج فدان الفول البلدي؟ [1]", retry_answer="من 30 إلى 40 كجم للفدان [1].")
    result = rag.answer("كم كيلو تقاوي يحتاج فدان الفول البلدي؟")
    assert result.problem == "echo" and result.retried
    assert "40" in result.answer and gen.last_feedback


def test_refusal_gets_a_second_look_and_keeps_a_grounded_answer():
    rag, _ = service(grounding.NO_ANSWER["ar"], retry_answer="تضاف 20 إلى 30 طنا من السماد العضوي الحيواني للهكتار [1].",
                     chunks=(POTATO,))
    result = rag.answer("كم طنا من السماد العضوي يضاف لكل هكتار عند زراعة البطاطس؟")
    assert result.problem == "refusal"
    assert "20" in result.answer and result.sources == ["[1] potato.pdf, p. 7"]


def test_refusal_stays_a_refusal_when_the_second_answer_is_not_grounded():
    rag, _ = service(grounding.NO_ANSWER["ar"], retry_answer="تضاف 40 إلى 60 طنا للهكتار [1].", chunks=(POTATO,))
    result = rag.answer("كم طنا من السماد العضوي يضاف لكل هكتار عند زراعة البطاطس؟")
    assert result.answer == NO_CONTEXT_ANSWER["ar"] and result.sources == []


def test_invented_number_is_retried_then_removed():
    # first and second answers both invent "40 to 60": the sentence is dropped, nothing is left -> refusal
    rag, _ = service("تضاف 40 إلى 60 طنا من السماد العضوي للهكتار [1].",
                     retry_answer="تضاف 40 إلى 60 طنا من السماد العضوي للهكتار [1].", chunks=(POTATO,))
    result = rag.answer("كم طنا من السماد العضوي يضاف لكل هكتار عند زراعة البطاطس؟")
    assert result.problem == "numbers" and "40" not in result.answer
    assert result.answer == NO_CONTEXT_ANSWER["ar"]


def test_answer_still_echoing_after_the_retry_becomes_a_refusal():
    rag, _ = service("كم كيلو تقاوي يحتاج فدان الفول البلدي؟ [1]", retry_answer="كم كيلو تقاوي يحتاج فدان الفول البلدي [1]")
    result = rag.answer("كم كيلو تقاوي يحتاج فدان الفول البلدي؟")
    assert result.answer == NO_CONTEXT_ANSWER["ar"]


def test_uncited_answer_is_kept_with_all_passages_as_sources():
    # e.g. an English answer translated from an Arabic passage, without [n] even after the retry
    rag, _ = service("Use 30 to 40 kg of seed per feddan.", retry_answer="Use 30 to 40 kg of seed per feddan.")
    result = rag.answer("How much faba bean seed per feddan?")
    assert result.problem == "uncited" and "30 to 40 kg" in result.answer
    assert result.sources == ["[1] faba.pdf, p. 7"]


def test_invented_number_fixed_by_the_retry():
    rag, _ = service("Sow wheat from 1 to 10 November [1].", retry_answer="Sow wheat from 15 to 30 November [1].",
                     chunks=(WHEAT,))
    result = rag.answer("When should wheat be sown in Lower Egypt?")
    assert result.answer.startswith("Sow wheat from 15 to 30 November [1]")


def test_refusal_followed_by_a_real_answer_keeps_the_answer_without_retry():
    rag, gen = service("I don't have enough information in my documents to answer that. "
                       "However, according to [1], sow wheat from 15 to 30 November.", chunks=(WHEAT,))
    result = rag.answer("When should wheat be sown?")
    assert gen.calls == 1
    assert result.answer.startswith("According to [1], sow wheat")
    assert result.sources == ["[1] wheat.pdf, p. 6"]


def test_pesticide_answers_get_the_safety_note():
    pest = chunk(4, "Rotate pesticide groups to delay resistance.", "ipm.pdf", 3)
    rag, _ = service("Rotate pesticide groups to delay resistance [1].", chunks=(pest,))
    result = rag.answer("How do I stop pests becoming resistant to pesticides?")
    assert result.answer.endswith(SAFETY_NOTE["en"])
    assert grounding.grounding_check(result.answer, [pest], "")["grounded"]  # the note is not a claim


def test_self_check_can_be_switched_off():
    gen = FakeGenerator(answer="كم كيلو تقاوي يحتاج فدان الفول البلدي؟ [1]", retry_answer="من 30 إلى 40 كجم [1].")
    rag = RAGService(FakeRetriever([FABA]), gen, min_similarity=0.3, translate_queries=False, self_check=False)
    rag.answer("كم كيلو تقاوي يحتاج فدان الفول البلدي؟")
    assert gen.calls == 1


def test_answer_in_the_wrong_language_is_retried():
    rag, gen = service("10 كجم للفدان من الهجن الفردية [1].", retry_answer="Use 10 kg per feddan for single-cross hybrids [1].",
                       chunks=(chunk(6, "معدل التقاوى 10 كجم للفدان من الهجن الفردية", "maize.pdf", 8, topic="الذرة الشامية / maize"),))
    result = rag.answer("How many kg of seed per feddan for single-cross maize hybrids?")
    assert "language" in result.problem and "Write it in English" in gen.last_feedback
    assert result.answer.startswith("Use 10 kg per feddan")


def test_answer_from_a_passage_about_another_crop_is_retried():
    # faba-bean question answered from the soybean bulletin (both say "30 كجم")
    rag, gen = service("يحتاج الفدان نحو 30 كجم [2].", retry_answer="يحتاج الفدان من 30 إلى 40 كجم تقاوي [1].",
                       chunks=(FABA, SOY))
    result = rag.answer("كم كيلو تقاوي يحتاج فدان الفول البلدي؟")
    assert "crop" in result.problem and "الفول البلدي" in gen.last_feedback
    assert result.sources == ["[1] faba.pdf, p. 7"]


def test_second_look_uses_the_two_closest_passages_and_keeps_their_numbers():
    land = chunk(7, "During the past 40 years, the world has lost a third of its arable land, about 430 million hectares.",
                 "fao.pdf", 1)
    others = [chunk(10 + i, f"Unrelated passage number {i} about irrigation scheduling.", "x.pdf", i) for i in range(3)]
    rag, gen = service(grounding.NO_ANSWER["en"],
                       retry_answer="The world has lost a third of its arable land, about 430 million hectares [1].",
                       chunks=(*others, land))
    result = rag.answer("How much arable land has the world lost over the past 40 years?")
    assert len(gen.last_chunks) == 2 and gen.last_chunks[0].chunk_id == "c7"
    assert "430 million hectares [4]" in result.answer and result.sources == ["[4] fao.pdf, p. 1"]


def test_answer_per_hectare_to_a_per_feddan_question_is_retried():
    india = chunk(9, "Select about 75,000 two-budded setts per hectare from 6-8 months old nursery.", "farmerbook.pdf", 60)
    egypt = chunk(10, "كمية التقاوي: 4,5–6 طن حسب نظام الزراعة، حيث يحتاج الفدان إلي 4,5 طن في حالة الزراعة بصف ونصف",
                  "sugarcane.pdf", 8, topic="قصب السكر / sugarcane")
    rag, gen = service("Select about 75,000 two-budded setts per hectare [1].",
                       retry_answer="A feddan needs 4.5 to 6 tons of setts [2].", chunks=(india, egypt))
    result = rag.answer("How many tons of sugarcane setts are needed per feddan?")
    assert "units" in result.problem and "per feddan" in gen.last_feedback and "per hectare" in gen.last_feedback
    assert result.answer.startswith("A feddan needs 4.5 to 6 tons") and result.sources == ["[2] sugarcane.pdf, p. 8"]


def test_area_units_are_recognised_in_both_languages():
    assert grounding.area_units("كم طنا للهكتار؟") == {"hectare"}
    assert grounding.area_units("How much seed per feddan?") == {"feddan"}
    assert grounding.unit_mismatch("20 to 30 t per ha [1]", "How many tons per feddan?") == ({"feddan"}, {"hectare"})
    assert grounding.unit_mismatch("20 إلى 30 طنا للهكتار [1]", "كم طنا لكل هكتار؟") == (set(), set())


def test_comparison_second_look_uses_one_passage_per_crop():
    melon = chunk(11, "كمية التقاوي: يحتاج الفدان إلى نحو 150-200 جرام في الهجن", "watermelon.pdf", 8, topic="البطيخ / watermelon")
    tomato = chunk(12, "كمية التقاوي يحتاج الفدان عند الزراعة بالهجن إلى 50-35 جرام بذور", "tomato.pdf", 16,
                   topic="الطماطم / tomato")
    filler = [chunk(20 + i, f"Unrelated passage {i} about irrigation scheduling.", "x.pdf", i) for i in range(3)]
    rag, gen = service(grounding.NO_ANSWER["en"],
                       retry_answer="Watermelon hybrids need 150-200 g per feddan [1]; tomato hybrids 35-50 g [2].",
                       chunks=(*filler, tomato, melon))
    result = rag.answer("Which needs more seed per feddan, watermelon hybrids or tomato hybrids?")
    assert {c.chunk_id for c in gen.last_chunks} == {"c11", "c12"}
    assert "150-200 g" in result.answer and len(result.sources) == 2


def test_comparison_words_are_not_search_terms():
    from app.services.lexical import tokenize

    assert tokenize("قارن بين ميعاد زراعة القمح والكتان") == tokenize("ميعاد زراعة القمح والكتان")
    assert tokenize("Compare wheat versus flax") == tokenize("wheat flax")


def test_planting_dates_is_not_date_palm():
    from app.services.glossary import crops_in

    assert [t.en for t in crops_in("Compare the planting dates of wheat and flax", "en")] == ["wheat", "flax"]
    assert [t.en for t in crops_in("When are dates harvested?", "en")] == ["date palm"]


class _AnyScoreRetriever(FakeRetriever):
    """Returns its chunks whatever their score (like the crop rule letting a question through)."""

    def retrieve_many(self, queries, top_k=4, min_similarity=0.0):
        self.queries += list(queries)
        return self.chunks[:top_k]


def test_crop_named_off_topic_question_gets_a_refusal():
    tomato = chunk(13, "مساحة وانتاج الطماطم في جمهورية مصر العربية تبلغ نحو 450 ألف فدان", "tomato.pdf", 7,
                   score=0.40, topic="الطماطم / tomato")
    gen = FakeGenerator(answer="تبلغ مساحة الطماطم في مصر نحو 450 ألف فدان [1].")
    rag = RAGService(_AnyScoreRetriever([tomato]), gen, min_similarity=0.6, translate_queries=False)
    result = rag.answer("ما هو سعر الطماطم في السوق اليوم؟")
    assert result.answer == NO_CONTEXT_ANSWER["ar"]


def test_crop_named_farming_question_keeps_its_answer():
    sun = chunk(14, "معدل التقاوي: في حالة الزراعة اليدوية 4 – 5 كجم بذرة / فدان", "sunflower.pdf", 4,
                score=0.40, topic="دوار الشمس / sunflower")
    gen = FakeGenerator(answer="معدل التقاوي 4 – 5 كجم بذرة للفدان في الزراعة اليدوية [1].")
    rag = RAGService(_AnyScoreRetriever([sun]), gen, min_similarity=0.6, translate_queries=False)
    assert rag.answer("ما هو معدل تقاوي دوار الشمس للفدان؟").answer.startswith("معدل التقاوي 4 – 5 كجم")


def test_second_look_does_not_turn_a_refusal_into_an_off_topic_answer():
    tomato = chunk(15, "تزرع الطماطم في العروة الصيفية في يناير وفبراير", "tomato.pdf", 13, topic="الطماطم / tomato")
    rag, _ = service(grounding.NO_ANSWER["ar"], retry_answer="تزرع الطماطم في العروة الصيفية في يناير وفبراير [1].",
                     chunks=(tomato,))
    assert rag.answer("ما هو سعر الطماطم في السوق اليوم؟").answer == NO_CONTEXT_ANSWER["ar"]


def test_sorghum_is_not_also_maize():
    from app.services.glossary import crops_in

    assert [t.en for t in crops_in("كم كيلو تقاوي يحتاج فدان الذرة الرفيعة؟", "ar")] == ["sorghum"]
    assert [t.en for t in crops_in("متى أزرع الذرة الشامية؟", "ar")] == ["maize"]


def test_short_numeric_answer_is_kept():
    rag, _ = service("45 كجم [1]", chunks=(chunk(8, "معدل التقاوي 45 كجم من قمح الخبز للفدان", "wheat.pdf", 8),))
    assert rag.answer("ما معدل تقاوي قمح الخبز بالسطارة؟").answer.startswith("45 كجم [1]")


def test_uncited_sentence_not_in_the_passages_is_removed():
    rag, _ = service("Sow wheat from 15 to 30 November [1]. Governments and companies must cooperate on strategies.",
                     retry_answer="Sow wheat from 15 to 30 November [1]. Governments and companies must cooperate on strategies.",
                     chunks=(WHEAT,))
    result = rag.answer("When should wheat be sown in Lower Egypt?")
    assert result.answer == "Sow wheat from 15 to 30 November [1]."


def test_wrong_passage_number_is_corrected():
    fixed = grounding.tidy("Sow wheat from 15 to 30 November in Lower Egypt [1].", [FABA, WHEAT], "When is wheat sown?")
    assert fixed.endswith("[2].")


def test_translation_is_shown_to_the_llm():
    gen = FakeGenerator(answer="Sow wheat from 15 to 30 November [1].", translations={"When is wheat sown?": "متى يزرع القمح؟"})
    RAGService(FakeRetriever([WHEAT]), gen, min_similarity=0.3).answer("When is wheat sown?")
    assert gen.last_translation == "متى يزرع القمح؟"


# ---------------------------------------------------------------- grounding helpers
def test_numbers_are_normalised_and_rtl_ranges_accepted():
    assert grounding.numbers("٢٫٥ كجم و 2,5 و 1,000 طن") == {"2.5", "1000"}
    # the passage says "40-30" (right-to-left range); the answer "30 to 40" is supported
    assert grounding.unsupported_numbers("من 30 إلى 40 كجم [1]", [FABA]) == []
    assert grounding.unsupported_numbers("45 kg [1]", [FABA]) == ["45"]


def test_list_numbering_and_citations_are_not_numbers():
    answer = "1. Sow wheat in November [1].\n2. Use 45 kg [2]."
    assert grounding.numbers(grounding.answer_body(answer)) == {"45"}


def test_invalid_citation_removed_and_missing_citation_added():
    text = "Sow wheat from 15 to 30 November in Lower Egypt [6]."
    fixed = grounding.tidy(text, [WHEAT], "When is wheat sown?")
    assert "[6]" not in fixed and "[1]" in fixed


def test_citation_after_question_mark_stays_with_its_sentence():
    fixed = grounding.tidy("كم كجم من تقاوي يحتاج الفدان؟ [1] يحتاج الفدان من 30 إلى 40 كجم [1]", [FABA],
                           "كم كجم من تقاوي يحتاج الفدان؟")
    assert fixed == "يحتاج الفدان من 30 إلى 40 كجم [1]"


def test_grounding_check():
    assert grounding.grounding_check("Sow wheat from 15 to 30 November [1].", [WHEAT])["grounded"]
    assert not grounding.grounding_check("Sow wheat from 15 to 30 November.", [WHEAT])["grounded"]  # no citation
    assert not grounding.grounding_check("Sow wheat in March [1] using 99 kg.", [WHEAT])["grounded"]
    assert grounding.grounding_check(grounding.NO_ANSWER["en"], [WHEAT])["grounded"]


# ---------------------------------------------------------------- keyword search
def test_arabic_light_stemming_matches_word_forms():
    from app.services.lexical import tokenize

    assert tokenize("للمبيدات") == tokenize("المبيدات") == tokenize("والمبيدات")
    assert "تقاو" in tokenize("التقاوي") and "سن" not in tokenize("من في على")


def test_bm25_learns_prepositions_from_the_corpus():
    from app.services.lexical import BM25Index

    index = BM25Index(["زراعة القمح", "قمح الخبز", "البطاطس", "بطاطس مبكرة", "حوامل الصواني"])
    assert index.terms("لقمح لحوامل بطاطس") == ["قمح", "حوامل", "بطاطس"]


def test_bm25_ranks_the_exact_rare_term_first():
    from app.services.lexical import BM25Index

    texts = ["مكافحة الآفات في الطماطم بالرش", "مكافحة حشرة التوتا أبسلوتا بالفرمونات", "ري الطماطم بالتنقيط"]
    assert BM25Index(texts).top("كيف أكافح التوتا في الطماطم؟", 1)[0][0] == 1


# ---------------------------------------------------------------- hybrid retrieval (in-memory Chroma)
VOCAB = ["wheat", "sow", "november", "irrigation", "water", "maize", "pest", "resistance"]


class BagOfWordsEmbedder:
    """Stand-in for the embedding model: a normalised count vector over a tiny vocabulary.

    Words outside VOCAB are unknown to it (like a rare pest name to a real model). Queries and
    documents get separate constant components, so a query with no known word matches nothing.
    """

    def _vec(self, text, is_query):
        v = np.array([text.lower().count(w) for w in VOCAB] + [0.3 * is_query, 0.3 * (not is_query)], dtype=np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts, normalize_embeddings=True, **kwargs):  # what the Retriever calls (queries)
        return np.array([self._vec(t, True) for t in texts])

    def encode_documents(self, texts):
        return np.array([self._vec(t, False) for t in texts])


@pytest.fixture
def retriever():
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.EphemeralClient()
    name = f"test_{np.random.randint(1e9)}"
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"}, embedding_function=None)
    docs = {
        "w1": ("Sow wheat in November after irrigation.", {"source": "wheat.pdf", "page_start": 1, "page_end": 1}),
        "w2": ("Wheat needs water every 20 days.", {"source": "wheat.pdf", "page_start": 2, "page_end": 2}),
        "m1": ("Maize is sown in May.", {"source": "maize.pdf", "page_start": 1, "page_end": 1}),
        "t1": ("Tuta absoluta is controlled with pheromone traps.",
               {"source": "tomato.pdf", "page_start": 9, "page_end": 9, "topic": "الطماطم / tomato"}),
        "p1": ("Rotate pest control products to delay resistance.", {"source": "ipm.pdf", "page_start": 3, "page_end": 3}),
    }
    emb = BagOfWordsEmbedder()
    col.add(ids=list(docs), documents=[d[0] for d in docs.values()], metadatas=[d[1] for d in docs.values()],
            embeddings=emb.encode_documents([d[0] for d in docs.values()]).tolist())
    yield Retriever(col, emb, {"embedding_model": "bow", "min_similarity": 0.5})
    client.delete_collection(name)


def test_keyword_search_finds_a_name_the_embedder_does_not_know(retriever):
    question = "Tuta absoluta pest on wheat: how to control it?"
    dense_only = retriever.search([question], top_k=2, hybrid=False)
    hybrid = retriever.search([question], top_k=2)
    assert "t1" not in [c.chunk_id for c in dense_only.chunks]
    assert "t1" in [c.chunk_id for c in hybrid.chunks]
    assert hybrid.chunks[[c.chunk_id for c in hybrid.chunks].index("t1")].topic == "الطماطم / tomato"


def test_document_topic_is_searchable(retriever):
    # "tomato" only appears in the topic of the Tuta chunk, not in its text
    assert retriever.search(["tomato"], top_k=1).chunks[0].chunk_id == "t1"


def test_relevance_gate_uses_the_best_cosine_similarity(retriever):
    assert retriever.retrieve_many(["When do I sow wheat in November?"], 3, min_similarity=0.5)
    assert retriever.retrieve_many(["Who won the World Cup?"], 3, min_similarity=0.5) == []


def test_question_naming_a_covered_crop_passes_the_gate(retriever):
    # no word the embedder knows -> similarity 0, but there is a tomato document
    result = retriever.search(["How to protect tomato from frost?"], 3)
    assert result.best_similarity < 0.5 and result.named_crops == ["tomato"] and result.passes(0.5)
    assert retriever.retrieve_many(["How to protect tomato from frost?"], 3, min_similarity=0.5)


def test_chunks_found_only_by_keywords_get_a_real_similarity(retriever):
    result = retriever.search(["Tuta absoluta"], top_k=5)
    assert all(0.0 <= c.score <= 1.0 for c in result.chunks)


# ---------------------------------------------------------------- glossary
def test_glossary_tells_soybean_from_faba_bean():
    from app.services.glossary import find_terms

    soy = [t.en for t in find_terms("كم كجم من تقاوي فول الصويا يحتاج الفدان؟", "ar")]
    faba = [t.en for t in find_terms("كم كيلو تقاوي يحتاج فدان الفول البلدي؟", "ar")]
    assert "soybean" in soy and "faba bean" not in soy
    assert "faba bean" in faba and "soybean" not in faba
    assert [t.en for t in find_terms("من البحري", "ar")] == []  # 'من' is not aphids, 'البحري' not irrigation


def test_arabic_planting_time_phrasing_is_recognised():
    from app.services.glossary import find_terms, synonyms

    q = "في أي وقت من السنة يمكن زراعة النخيل؟"
    assert [t.en for t in find_terms(q, "ar")] == ["date palm", "planting date"]
    assert "موعد الزراعة" in synonyms(q, "ar")  # the palm bulletin says "موعد الزراعة فى اى وقت فى السنة"


def test_document_topic():
    from app.services.glossary import document_topic

    assert document_topic(["..."], "نشرة-دوار-الشمس.pdf").en == "sunflower"
    texts = ["زراعة البطاطس في الأراضي الرملية"] * 3 + ["الري"]
    assert document_topic(texts, "ticket_file.pdf").en == "potato"
    general = ["القمح", "الذرة الشامية", "القطن", "الأرز"]  # a handbook about many crops
    assert document_topic(general, "handbook.pdf") is None


# ---------------------------------------------------------------- numbered lists
def test_list_numbers_are_not_claims_and_lists_stay_in_order():
    from app.services import grounding

    passage = chunk(7, "التوصيات: جمع الثمار في الوقت المناسب. استخدام العبوات المناسبة. زيادة عدد المبردات. "
                       "زيادة انشاء مصانع لتصنيع الطماطم.", "tomato.pdf", 40)
    answer = ("1. جمع الثمار في الوقت المناسب [1].\n2. استخدام العبوات المناسبة [1].\n"
              "3. يضاف 75 كجم يوريا للفدان [1].\n4. زيادة عدد المبردات [1].\n5. زيادة انشاء مصانع لتصنيع الطماطم [1].")
    kept = grounding.drop_unsupported_sentences(answer, [passage], "ما التوصيات؟")
    assert kept == ("1. جمع الثمار في الوقت المناسب [1].\n2. استخدام العبوات المناسبة [1].\n"
                    "3. زيادة عدد المبردات [1].\n4. زيادة انشاء مصانع لتصنيع الطماطم [1].")   # the invented dose is gone
    assert grounding.unsupported_numbers(answer.replace("75 كجم يوريا", "العبوات"), [passage]) == []
    assert grounding.segments("5. جمع الثمار [1]. ثم التعبئة.")[:2] == ["5. جمع الثمار [1].", "ثم التعبئة."]


# ---------------------------------------------------------------- presentable answers
@pytest.mark.parametrize("raw, clean", [
    ("According to passage [2], wheat is sown from 15 to 30 November [2].", "Wheat is sown from 15 to 30 November [2]."),
    ("The passages state that maize needs water every 12 days [1]. Maize needs water every 12 days [1].",
     "Maize needs water every 12 days [1]."),
    ("وفقاً للفقرة [1]، تزرع الطماطم في يناير [1]. تشير الفقرة 2 إلى أن الري يكون كل يومين [2].",
     "تزرع الطماطم في يناير [1]. الري يكون كل يومين [2]."),
    ("- يضاف السماد قبل الزراعة [1]\n- \n* يروى الحقل بعد الزراعة [2].", "- يضاف السماد قبل الزراعة [1]\n- يروى الحقل بعد الزراعة [2]."),
    ("تطبق إرشادات إدارة الآفات UC-TOM لمنع انتشار الفيروسات [3].", "تطبق إرشادات إدارة الآفات لمنع انتشار الفيروسات [3]."),
    ("Plant in March ,irrigate weekly  [1] .", "Plant in March, irrigate weekly [1]."),
    ("COVID-19 disrupted markets in 2020 [1].", "COVID-19 disrupted markets in 2020 [1]."),
])
def test_answers_are_polished(raw, clean):
    from app.services.grounding import polish

    assert polish(raw) == clean


def test_refusals_and_numbers_are_never_changed_by_polishing():
    from app.services.grounding import NO_ANSWER, polish

    assert polish(NO_ANSWER["en"]) == NO_ANSWER["en"]
    assert polish("Add 50 kg of potassium sulfate per feddan at flowering [1].") == \
        "Add 50 kg of potassium sulfate per feddan at flowering [1]."


def test_record_codes_and_contents_lines_never_reach_the_llm():
    from app.services.generation import build_context, clean_passage

    encyclopedia = chunk(8, "من المحاصيل الباذنجانية واسعة الاستخدام.\nAGR-0051 | ARC-TOM · UC-TOM · FAO-56 :سجل المعرفة",
                         "Agricultural_RAG_200_Pages_Arabic.pdf", 12)
    contents = chunk(9, "زراعة وإنتاج بنجر السكر\nيتناول هذا المستند: مقدمة، إعداد الأرض...\nيزرع البنجر في سبتمبر.",
                     "beet.pdf", 1)
    context = build_context([encyclopedia, contents])
    assert "UC-TOM" not in context and "سجل المعرفة" not in context and "يتناول هذا المستند" not in context
    assert "يزرع البنجر في سبتمبر" in context and clean_passage("COVID-19 hit markets.") == "COVID-19 hit markets."


def test_only_close_passages_reach_the_llm():
    from app.services.rag_service import focus_passages

    close = [chunk(i, f"Passage {i} about wheat sowing.", score=0.70 - i * 0.01) for i in range(5)]
    far = chunk(9, "An unrelated passage.", score=0.40)
    contents = chunk(10, "Wheat bulletin\nThis document covers: sowing; irrigation.", score=0.72)
    kept = focus_passages([contents, *close, far])
    assert contents not in kept and far not in kept and len(kept) == 5
    assert focus_passages([contents]) == [contents]   # nothing else: kept


def test_off_topic_questions_get_a_friendly_answer(tmp_path):
    from tests.test_beyond import beyond_service

    gen = FakeGenerator(scope_reply="no")
    rag, _ = beyond_service(tmp_path, gen, [])
    result = rag.answer("Who won the 2022 World Cup?")
    assert result.off_topic and result.origin == "none" and "farming assistant" in result.answer


# ---------------------------------------------------------------- new subject vs follow-up
COTTON = chunk(11, "ميعاد زراعة القطن في الأراضي القديمة من أول مارس، مع الري كل 15 يوما.", "cotton.pdf", 7,
               topic="القطن / cotton")


class KeywordRetriever(FakeRetriever):
    """Finds a passage only when the query shares a word of 4+ letters with it."""

    def retrieve(self, question, top_k=4, min_similarity=0.0):
        self.queries.append(question)
        words = set(re.findall(r"\w{4,}", question.lower()))
        return [c for c in self.chunks if words & set(re.findall(r"\w{4,}", c.text.lower()))][:top_k]


def test_a_new_subject_is_not_answered_from_the_earlier_question():
    gen = FakeGenerator(answer="ميعاد زراعة القطن في الأراضي القديمة من أول مارس [1].")
    rag = RAGService(KeywordRetriever([COTTON]), gen, min_similarity=0.3, translate_queries=False)
    result = rag.answer("How do I raise rabbits at home?", history=["ما هو ميعاد زراعة القطن؟"])
    assert result.origin == "none" and gen.calls == 0          # not "answered" with cotton passages
    follow = rag.answer("كم مرة الري؟", history=["ما هو ميعاد زراعة القطن؟"])
    assert follow.origin == "library" and gen.last_earlier == "ما هو ميعاد زراعة القطن؟"


def test_the_reported_language_is_the_answers_own():
    gen = FakeGenerator(answer="ميعاد زراعة القطن في الأراضي القديمة من أول مارس [1].")   # no translation available
    rag = RAGService(FakeRetriever([COTTON]), gen, min_similarity=0.3, translate_queries=False)
    result = rag.answer("When is cotton planted in the old lands?")
    assert result.language == "ar"   # the voice and the page direction follow the text


@pytest.mark.parametrize("text", [
    "Choose water sources that do not contain salts [1].", "Get water from sources such as wells [1].",
    "Nitrogen sources suggest a split dose [1].", "The bulletin recommends that you sow early [2].",
    "يوصي المرشد الزراعي بالري كل أسبوع [1].", "Use the FAO-56 method to estimate water needs [1].",
])
def test_polishing_leaves_normal_farming_sentences_alone(text):
    from app.services.grounding import polish

    assert polish(text) == text
