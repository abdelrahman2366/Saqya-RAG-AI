"""Tests for the optional AI/ML layers: evaluation metrics and split, query understanding and expansion, weighted
search, reranking, claim verification, confidence, tracing, and the API fields. No model, Ollama or network:
models are replaced by small fakes."""

import json
import math
import re

import numpy as np
import pytest

from app.core.tracing import Tracer, tracer
from app.evaluation import metrics
from app.evaluation.dataset import EvalQuestion, load_questions, split_of
from app.services import claims, confidence
from app.services.query_expansion import expand
from app.services.query_understanding import understand
from app.services.rag_service import RAGService
from app.services.reranker import Reranker
from app.services.retrieval import MemoryCollection, RetrievedChunk, Retriever
from tests.conftest import MAIZE_CHUNK, FakeGenerator, FakeRetriever, make_client


# ------------------------------------------------------------------ metrics
def test_ranking_metrics_known_values():
    hits = [False, True, False, True, False]
    assert metrics.mrr_at_k(hits, 10) == 0.5
    assert metrics.recall_at_k(hits, 2, 5) == 1.0
    assert metrics.recall_at_k(hits, 4, 5) == 0.5
    assert metrics.precision_at_k(hits, 5) == 0.4
    ideal = 1 + 1 / math.log2(3)
    assert metrics.ndcg_at_k(hits, 2, 10) == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert metrics.ndcg_at_k([True, True], 2, 10) == pytest.approx(1.0)
    assert metrics.mrr_at_k([], 10) == 0.0 and metrics.recall_at_k([], 0, 5) == 0.0


def test_probability_metrics():
    assert metrics.brier_score([1, 0], [1, 0]) == 0.0
    assert metrics.brier_score([0.5, 0.5], [1, 0]) == 0.25
    assert metrics.expected_calibration_error([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0, abs=1e-9)
    assert metrics.expected_calibration_error([0.9] * 10, [0] * 10) == pytest.approx(0.9)
    assert metrics.auroc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert metrics.auroc([0.1, 0.9], [1, 0]) == 0.0
    rc = metrics.risk_coverage([0.9, 0.2, 0.7], [1, 0, 0], 0.5)
    assert rc["coverage"] == pytest.approx(2 / 3, abs=1e-3) and rc["wrong_answered"] == 1 and rc["right_refused"] == 0


def test_split_keeps_paraphrases_together_and_is_stable():
    assert split_of("q01", 13) == split_of("q01", 13)
    questions = load_questions(__import__("pathlib").Path(__file__).resolve().parents[2] / "data")
    by_group = {}
    for q in questions:
        by_group.setdefault(q.group, set()).add(q.split)
    assert all(len(s) == 1 for s in by_group.values())   # an Egyptian question and its MSA twin share a split
    assert {"validation", "test"} == {q.split for q in questions}


def test_relevance_rule_needs_document_and_phrase():
    q = EvalQuestion("x", "?", "in_domain", "c", expected_keywords=["30 كجم"], expected_source="الفول")
    assert q.is_relevant("يحتاج الفدان 30 كجم تقاوي", "الفول-البلدي.pdf")
    assert not q.is_relevant("يحتاج الفدان 30 كجم تقاوي", "القمح.pdf")
    assert not q.is_relevant("يحتاج الفدان تقاوي", "الفول-البلدي.pdf")


# ------------------------------------------------------------------ query understanding
def test_understanding_egyptian_symptom_question_is_not_a_diagnosis():
    u = understand("الطماطم عندي ورقها بيصفر ومش عارف السبب.")
    assert u.crops == ["tomato"] and u.dialect == "egyptian" and u.language == "ar"
    assert u.intent == "diagnose" and u.symptoms == ["yellowing"]
    assert u.diseases == [] and u.pests == []          # nothing named, nothing assumed
    assert "plant age or growth stage" in u.would_help and u.missing == []


def test_understanding_reads_crop_names_farmers_use_and_conditions():
    u = understand("إمتى أزرع الأوطة في أرض رملية في المنيا؟")
    assert u.crops == ["tomato"] and u.crop_groups == ["vegetables"] and u.intent == "when"
    assert u.conditions["soil"] == "sandy" and u.conditions["region"] == "upper"


def test_understanding_english_mixed_quantities_and_named_pest():
    u = understand("How do I control Tuta absoluta on tomato? I sprayed 50 ml per 20 L")
    assert u.language == "en" and "tomato" in u.crops and u.pests == ["Tuta absoluta"]
    assert {"value": "50", "unit": "ml"} in u.quantities and "pests" in u.categories
    mixed = understand("امتى ازرع tomato في الصعيد؟")
    assert "tomato" in mixed.crops and mixed.conditions.get("region") == "upper"


def test_understanding_missing_crop_and_empty_question():
    assert understand("الزرع عندي بيموت").missing == ["crop"]
    empty = understand("")
    assert empty.crops == [] and empty.categories == [] and empty.intent == "other"
    assert understand("Who won the 2022 FIFA World Cup?").categories == []


def test_understanding_follow_up_takes_crop_from_history_only_when_unnamed():
    u = understand("بتحتاج مية قد إيه؟", history=["إمتى أزرع القمح؟"])
    assert u.crops == ["wheat"] and u.conditions.get("crop_from") == "earlier question"


# ------------------------------------------------------------------ query expansion
_NUMBER = re.compile(r"\d")


@pytest.mark.parametrize("question", [
    "متى أزرع القمح في الوجه البحري؟", "ليه ورق الطماطم بيصفر؟", "How much seed does maize need per feddan?",
    "كيف أكافح التوتا في الطماطم؟", "ما هو أفضل ميعاد لزراعة السمسم؟",
])
def test_expansion_adds_no_new_crop_pest_or_number(question):
    asked = understand(question)
    out = expand(question)
    assert len(out.variants) <= 2 and question not in out.variants
    for variant in out.variants:
        found = understand(variant)
        assert set(found.crops) <= set(asked.crops) | set()   # never a crop the farmer did not name
        assert set(found.pests) <= set(asked.pests) and set(found.diseases) <= set(asked.diseases)
        assert len(_NUMBER.findall(variant)) <= len(_NUMBER.findall(question))


def test_expansion_limits_and_failures(monkeypatch):
    assert not expand("")
    assert not expand("Who won the world cup?")
    assert len(expand("متى أزرع القمح؟", max_variants=1).variants) == 1
    import app.services.query_expansion as qe

    monkeypatch.setattr(qe, "_expand", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert not expand("متى أزرع القمح؟")   # a failure means "no expansion", never an exception


# ------------------------------------------------------------------ weighted search (a real Retriever, toy embedder)
class WordEmbedder:
    """Bag-of-words vectors over a fixed vocabulary: enough for the search code to rank."""

    def __init__(self, vocab):
        self.vocab = {w: i for i, w in enumerate(vocab)}

    def encode(self, inputs, normalize_embeddings=True, **_):
        out = np.zeros((len(inputs), len(self.vocab)), dtype=np.float32)
        for r, text in enumerate(inputs):
            for w in re.findall(r"\w+", text.lower()):
                if w in self.vocab:
                    out[r, self.vocab[w]] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture()
def toy_retriever():
    texts = ["maize planting date in spring", "maize fertilizer nitrogen rate", "wheat planting date november",
             "tomato irrigation drip", "sesame harvest when pods turn yellow"]
    metas = [{"source": f"d{i}.pdf", "page_start": i + 1, "page_end": i + 1, "section": f"s{i}"} for i in range(5)]
    emb = WordEmbedder(sorted({w for t in texts for w in t.split()} | {"seed", "rate"}))
    vecs = emb.encode(texts)
    return Retriever(MemoryCollection("toy", [f"c{i}" for i in range(5)], texts, metas, vecs), emb, {}, hybrid=True)


def test_weights_default_is_the_old_search(toy_retriever):
    a = toy_retriever.search(["maize planting date"], top_k=3)
    b = toy_retriever.search(["maize planting date"], top_k=3, weights=[1.0])
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks] and a.best_similarity == b.best_similarity


def test_expansion_phrasing_cannot_open_the_relevance_gate(toy_retriever):
    plain = toy_retriever.search(["unrelated words"], top_k=3)
    expanded = toy_retriever.search(["unrelated words", "sesame harvest"], top_k=3, weights=[1.0, 0.25])
    assert expanded.best_similarity == plain.best_similarity == 0.0
    assert expanded.chunks and expanded.chunks[0].chunk_id == "c4"   # it may still add a passage


def test_search_keeps_metadata_and_handles_empty_queries(toy_retriever):
    found = toy_retriever.search(["maize fertilizer"], top_k=2).chunks
    assert found[0].source == "d1.pdf" and found[0].page_start == 2 and found[0].section == "s1"
    assert toy_retriever.search([""], top_k=3).chunks == [] and toy_retriever.search([], top_k=3).chunks == []
    with pytest.raises(ValueError):
        toy_retriever.search(["maize"], top_k=2, weights=[1.0, 0.5])


def test_rrf_settings_are_used(toy_retriever):
    toy_retriever.rrf_k, toy_retriever.candidates = 5, 2
    assert len(toy_retriever.dense_search(toy_retriever.embed_query("maize"))) == 2


# ------------------------------------------------------------------ reranker
def chunk(cid, text, score=0.7, **kw):
    return RetrievedChunk(cid, text, kw.get("source", f"{cid}.pdf"), kw.get("page", 1), kw.get("page", 1), score,
                          section=kw.get("section", ""))


class FakeCrossEncoder:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.calls.append((len(pairs), batch_size))
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return [float(len(set(q.split()) & set(p.split()))) for q, p in pairs]


def test_reranker_reorders_keeps_metadata_and_cosine():
    chunks = [chunk("a", "nothing here", 0.80, page=4, section="x"), chunk("b", "maize seed rate per feddan", 0.65, page=9)]
    model = FakeCrossEncoder()
    out = Reranker(model=model, batch_size=8).rerank(["maize seed rate"], chunks, top_k=2)
    assert [c.chunk_id for c in out] == ["b", "a"]
    assert out[0].page_start == 9 and out[0].score == 0.65 and out[0].rerank_score == 3.0   # cosine untouched
    assert model.calls == [(2, 8)]


def test_reranker_failure_keeps_search_order():
    chunks = [chunk("a", "x"), chunk("b", "maize")]
    assert [c.chunk_id for c in Reranker(model=FakeCrossEncoder(fail=True)).rerank(["maize"], chunks, 2)] == ["a", "b"]


def test_missing_reranker_model_is_reported_not_downloaded(monkeypatch):
    reranker = Reranker("org/does-not-exist-anywhere", allow_download=False)
    assert reranker.load() is None and reranker.failed
    assert reranker.scores([("q", "p")]) is None


def test_rag_with_failed_reranker_answers_like_without_it():
    gen = FakeGenerator()
    plain = RAGService(FakeRetriever([MAIZE_CHUNK]), gen, top_k=4, min_similarity=0.3, translate_queries=False)
    broken = Reranker("org/does-not-exist-anywhere")
    broken.load()
    rag = RAGService(FakeRetriever([MAIZE_CHUNK]), FakeGenerator(), top_k=4, min_similarity=0.3, translate_queries=False,
                     reranker=broken)
    assert rag.answer("When do I plant maize?").answer == plain.answer("When do I plant maize?").answer


def test_rag_reranks_candidates_into_top_k():
    other = chunk("other", "Maize storage in dry bins.", 0.75)
    rag = RAGService(FakeRetriever([other, MAIZE_CHUNK]), FakeGenerator(), top_k=1, min_similarity=0.3,
                     translate_queries=False, reranker=Reranker(model=FakeCrossEncoder()), rerank_candidates=5)
    result = rag.answer("When do I plant maize at the start of the rainy season?")
    assert [c.chunk_id for c in result.chunks] == ["maize_guide_0003"]


# ------------------------------------------------------------------ claims
PASSAGES = [chunk("p1", "يحتاج فدان فول الصويا 30 كجم تقاوي ويزرع في مايو.", page=5),
            chunk("p2", "معدل تقاوي دوار الشمس 4 كجم للفدان في الأراضي القديمة.", page=7)]


def test_claim_verdicts():
    report = claims.verify("يحتاج فدان فول الصويا 30 كجم تقاوي [1]. يحتاج فدان فول الصويا 45 كجم تقاوي [1]. "
                           "يحتاج فدان فول الصويا 4 كجم تقاوي [1].", PASSAGES)
    verdicts = [c.verdict for c in report.claims]
    assert verdicts == [claims.SUPPORTED, claims.UNSUPPORTED, claims.PARTIAL]
    assert report.claims[0].evidence == 1 and report.claims[1].missing_numbers == ["45"]
    assert report.claims[2].misplaced_numbers == ["4"]
    assert report.supported_share == pytest.approx(0.5) and report.worst == claims.UNSUPPORTED


def test_claims_skip_refusals_questions_and_safety_note_and_handle_no_passages():
    from app.services.grounding import SAFETY_NOTE

    text = f"I don't have enough information in my documents to answer that.\n\nWhich crop is it?\n\n{SAFETY_NOTE['en']}"
    assert claims.extract_claims(text) == []
    assert claims.verify("Maize needs 30 kg of seed per feddan.", []).claims[0].verdict == claims.UNSUPPORTED


def test_cross_language_claim_is_unverified_but_numbers_still_checked():
    ok = claims.verify("Soybean needs 30 kg of seed per feddan [1].", PASSAGES).claims[0]
    bad = claims.verify("Soybean needs 70 kg of seed per feddan [1].", PASSAGES).claims[0]
    assert ok.verdict == claims.UNVERIFIED and bad.verdict == claims.UNSUPPORTED


class FakeNLI:
    def __init__(self, label=None, fail=False):
        self.label, self.fail = label, fail

    def __call__(self, pairs):
        if self.fail:
            raise RuntimeError("model broke")
        return [{"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9} if self.label == "contradiction"
                else {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05} for _ in pairs]


def test_nli_can_flag_contradiction_upgrade_but_never_rescue_a_missing_number():
    negated = "لا يحتاج فدان فول الصويا 30 كجم تقاوي [1]."
    assert claims.verify(negated, PASSAGES).claims[0].verdict == claims.SUPPORTED          # words alone miss it
    nli = claims.NLIVerifier(pipeline=FakeNLI("contradiction"))
    assert claims.verify(negated, PASSAGES, nli=nli).claims[0].verdict == claims.CONTRADICTED
    entails = claims.NLIVerifier(pipeline=FakeNLI("entailment"))
    assert claims.verify("Soybean needs 30 kg of seed [1].", PASSAGES, nli=entails).claims[0].verdict == claims.SUPPORTED
    assert claims.verify("Soybean needs 70 kg of seed [1].", PASSAGES, nli=entails).claims[0].verdict == claims.UNSUPPORTED
    broken = claims.NLIVerifier(pipeline=FakeNLI(fail=True))
    assert claims.verify(negated, PASSAGES, nli=broken).claims[0].verdict == claims.SUPPORTED   # lexical verdict stands


# ------------------------------------------------------------------ confidence
def _signals(best, coverage):
    return confidence.Signals(best_similarity=best, mean_similarity=best - 0.05, gate_margin=best - 0.59,
                              word_coverage=coverage, crop_match=1.0, rerank_top=0.5, passages=6)


def test_calibration_fit_save_load_and_thresholds(tmp_path):
    rng = np.random.default_rng(0)
    rows, y = [], []
    for _ in range(80):
        good = rng.random() < 0.6
        rows.append(_signals(0.75 + rng.normal(0, 0.03) if good else 0.55 + rng.normal(0, 0.03), 0.8 if good else 0.3))
        y.append(int(good))
    model = confidence.fit_calibration(rows, y)
    p = [model.probability(r) for r in rows]
    assert metrics.auroc(p, y) > 0.95
    model.answer_at, model.abstain_at = confidence.choose_thresholds(p, y)
    assert model.abstain_at is not None and model.abstain_at <= (model.answer_at or 1)
    model.save(tmp_path / "cal.json")
    loaded = confidence.Calibration.load(tmp_path / "cal.json")
    assert loaded.probability(rows[0]) == model.probability(rows[0])
    data = json.loads((tmp_path / "cal.json").read_text())
    data["features"] = ["something", "else"]
    (tmp_path / "bad.json").write_text(json.dumps(data))
    assert confidence.Calibration.load(tmp_path / "bad.json") is None
    assert confidence.Calibration.load(tmp_path / "missing.json") is None
    assert confidence.choose_thresholds([0.5, 0.6], [1, 1]) == (None, None)   # nothing to learn from


def test_uncalibrated_confidence_claims_no_probability():
    est = confidence.ConfidenceEstimator(None)
    verdict = est.after_answer(_signals(0.7, 0.8), answered=True)
    assert verdict.decision == confidence.ANSWER and verdict.probability is None and not verdict.calibrated
    shaky = _signals(0.7, 0.8)
    shaky.worst_claim = "unsupported"
    assert est.after_answer(shaky, answered=True).decision == confidence.QUALIFY
    lacking = confidence.Signals(missing=["crop"])
    assert est.after_answer(lacking, answered=False).decision == confidence.ASK
    assert est.after_answer(confidence.Signals(), answered=False).decision == confidence.ABSTAIN


def test_rag_asks_for_the_crop_instead_of_refusing():
    rag = RAGService(FakeRetriever([]), FakeGenerator(), min_similarity=0.3, translate_queries=False,
                     confidence=confidence.ConfidenceEstimator(None), classify=True)
    result = rag.answer("الزرع عندي بيموت")
    assert result.confidence["decision"] == confidence.ASK and "المحصول" in result.answer
    assert result.understanding["missing"] == ["crop"]


def test_rag_abstains_when_calibrated_probability_is_too_low():
    cal = confidence.Calibration(mean=[0] * 6, scale=[1] * 6, coef=[0] * 6, intercept=-5.0, answer_at=0.5, abstain_at=0.2)
    rag = RAGService(FakeRetriever([MAIZE_CHUNK]), FakeGenerator(), min_similarity=0.3, translate_queries=False,
                     confidence=confidence.ConfidenceEstimator(cal))
    result = rag.answer("When do I plant maize?")
    assert not result.cited and result.confidence["decision"] == confidence.ABSTAIN


def test_layers_off_change_nothing():
    a = RAGService(FakeRetriever([MAIZE_CHUNK]), FakeGenerator(), min_similarity=0.3, translate_queries=False)
    result = a.answer("When do I plant maize?")
    assert result.understanding is None and result.claims is None and result.confidence is None
    assert result.cited and result.sources == ["[1] maize_guide.pdf, p. 3"]


def test_claims_layer_attaches_verdicts_and_keeps_citations():
    rag = RAGService(FakeRetriever([MAIZE_CHUNK]), FakeGenerator(), min_similarity=0.3, translate_queries=False,
                     verify_claims=True)
    result = rag.answer("When do I plant maize?")
    assert result.claims["counts"]["supported"] == 1 and result.sources == ["[1] maize_guide.pdf, p. 3"]


def test_expansion_on_with_a_retriever_that_has_no_weights_still_answers():
    rag = RAGService(FakeRetriever([MAIZE_CHUNK]), FakeGenerator(), min_similarity=0.3, translate_queries=False,
                     expansion=True)
    assert rag.answer("When do I plant maize?").cited


# ------------------------------------------------------------------ tracing
def test_tracing_off_is_a_timer_only():
    t = Tracer()
    with t.request() as timings:
        with t.span("stage", inputs={"q": "x"}) as span:
            span.set(outputs={"a": 1})
    assert timings and timings[0][0] == "stage" and span.run is None


def test_tracing_needs_key_and_survives_missing_package(monkeypatch):
    t = Tracer()
    assert not t.configure(enabled=True, api_key=None)
    import builtins

    real_import = builtins.__import__

    def no_langsmith(name, *args, **kwargs):
        if name.startswith("langsmith"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_langsmith)
    assert not t.configure(enabled=True, api_key="key")
    with t.span("x"):
        pass


def test_tracing_errors_never_break_the_pipeline():
    t = Tracer()
    t.enabled = True

    def broken_trace(**kwargs):
        raise ConnectionError("LangSmith unreachable")

    t._trace, t._context = broken_trace, lambda **k: __import__("contextlib").nullcontext()
    with t.span("answer") as span:
        value = 41 + 1
    assert value == 42 and t._failed and "unreachable" in t._failed
    assert not t.active   # switched off after the first tracing error
    with pytest.raises(ValueError):          # the pipeline's own errors still propagate
        with t.span("answer"):
            raise ValueError("real bug")


def test_tracing_privacy_settings():
    t = Tracer()
    t.passage_chars, t.log_text = 0, False
    item = t.passages([MAIZE_CHUNK])[0]
    assert "text" not in item and item["source"] == "maize_guide.pdf" and item["pages"] == "3-3"
    assert t.text("my farm in Minya") == "<16 characters>"


def test_token_usage_is_recorded_on_the_current_span():
    t = Tracer()
    sink = []
    t.usage_sink = sink
    with t.span("generate") as span:
        t.usage("qwen2.5:7b", 120, 30)
    assert sink == [{"model": "qwen2.5:7b", "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}]
    assert span.metadata["llm_calls"][0]["total_tokens"] == 150


# ------------------------------------------------------------------ API compatibility
def test_api_fields_are_additive():
    client = make_client(RAGService(FakeRetriever([MAIZE_CHUNK]), FakeGenerator(), min_similarity=0.3,
                                    translate_queries=False, classify=True))
    body = client.post("/query", json={"question": "When do I plant maize?"}).json()
    assert body["sources"] == ["[1] maize_guide.pdf, p. 3"] and body["answer"]
    assert body["understanding"]["crops"] == ["maize"] and body["claims"] is None and body["confidence"] is None
    health = client.get("/health").json()
    assert health["ai_layers"]["query_classification"] is True and health["ai_layers"]["reranker"] is False
    assert tracer.active is False


# ------------------------------------------------------------------ evaluation runner (keyword-only, no model)
def test_offline_runner_scores_rankings_with_the_relevance_rule():
    from app.evaluation import runner

    ids = ["a", "b", "c"]
    texts = ["sesame harvest when pods turn yellow", "wheat planting date november", "maize seed rate 10 kg"]
    metas = [{"source": "sesame.pdf"}, {"source": "wheat.pdf"}, {"source": "maize.pdf"}]
    keyword = runner.keyword_retriever(ids, texts, metas)
    corpus = runner.Corpus(ids, texts, metas)
    q = EvalQuestion("t1", "How much maize seed per feddan?", "in_domain", "field crops",
                     expected_keywords=["10 kg"], expected_source="maize", split="test")
    exps = {e.id: e for e in runner.build_experiments(None, keyword)}
    assert exps["E1"].rank is None and "embedding model" in exps["E1"].not_run   # reported, never estimated
    run = runner.run_experiment(exps["E2"], [q], corpus, runner.Queries())
    row = run["rows"][0]
    assert row["first_hit"] == 1 and row["relevant_total"] == 1 and run["failures"] == 0
    assert runner.summarize(run["rows"])["mrr@10"] == 1.0


def test_keyword_weight_zero_is_the_embedding_ranking_and_seat_can_be_turned_off(toy_retriever):
    dense = [c.chunk_id for c in toy_retriever.search(["maize planting date"], top_k=4, hybrid=False).chunks]
    toy_retriever.keyword_weight = 0.0
    assert [c.chunk_id for c in toy_retriever.search(["maize planting date"], top_k=4).chunks] == dense
    toy_retriever.keyword_weight, toy_retriever.keyword_seat = 0.5, False
    assert toy_retriever.search(["maize planting date"], top_k=4).chunks   # still a valid ranking


def test_diagnose_question_without_crop_gets_a_follow_up_question_but_keeps_the_answer():
    from tests.conftest import FakeGenerator as Gen

    gen = Gen()
    gen.general_text = "تحقق من الري والتصريف وافحص الجذور."
    rag = RAGService(FakeRetriever([]), gen, min_similarity=0.3, translate_queries=False,
                     confidence=confidence.ConfidenceEstimator(None), classify=True,
                     beyond=__import__("app.services.beyond", fromlist=["Beyond"]).Beyond(embedder=None, general=True))
    result = rag.answer("الزرع عندي بيموت")
    assert result.origin == "general"   # an answer was given: it stays, with the question after it
    assert result.answer.startswith("تحقق من الري") and result.answer.endswith(confidence.FOLLOW_UP["ar"])
    named = RAGService(FakeRetriever([MAIZE_CHUNK]), FakeGenerator(), min_similarity=0.3, translate_queries=False,
                       confidence=confidence.ConfidenceEstimator(None))
    assert confidence.FOLLOW_UP["en"] not in named.answer("When do I plant maize?").answer   # crop named: no follow-up
    general = RAGService(FakeRetriever([]), gen, min_similarity=0.3, translate_queries=False,
                         confidence=confidence.ConfidenceEstimator(None),
                         beyond=__import__("app.services.beyond", fromlist=["Beyond"]).Beyond(embedder=None, general=True))
    # a general "why" question describes no symptom: no follow-up even though no crop is named
    assert confidence.FOLLOW_UP["en"] not in general.answer("What is seed treatment and why is it used?").answer
