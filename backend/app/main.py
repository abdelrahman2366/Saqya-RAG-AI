"""FastAPI entry point: CORS, logging, and loading the RAG pipeline once at startup."""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
from app.api.routes.vision import router as vision_router
from app.api.routes.voice import router as voice_router
from app.core.config import get_settings
from app.services.beyond import Beyond
from app.services.documents import DocumentService
from app.services.generation import OllamaGenerator
from app.services.learning import KnowledgeStore
from app.services.rag_service import RAGService
from app.services.retrieval import Retriever
from app.services.speech import GeminiSpeech, SpeechToText, TextToSpeech
from app.services.web_search import WebSearcher
from app.core.tracing import tracer
from app.utils.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("app")


def optional_layers() -> dict:
    """The optional AI/ML layers the settings turn on (all off by default). Nothing here loads a model: the reranker
    and the NLI model load on the first question that needs them, and only from disk."""
    layers: dict = {}
    if settings.reranker_enabled:
        from app.services.reranker import Reranker

        layers["reranker"] = Reranker(settings.reranker_model, settings.reranker_device, settings.reranker_batch_size,
                                      settings.reranker_max_length, settings.reranker_allow_download,
                                      settings.reranker_trust_remote_code)
        layers["rerank_candidates"] = settings.reranker_candidates
    if settings.query_expansion_enabled:
        layers.update(expansion=True, expansion_weight=settings.query_expansion_weight,
                      expansion_max_variants=settings.query_expansion_max_variants,
                      expansion_max_terms=settings.query_expansion_max_terms)
    layers["classify"] = settings.query_classification_enabled
    if settings.claim_verification_enabled:
        layers["verify_claims"] = True
        if settings.claim_verification_nli:
            from app.services.claims import NLIVerifier

            layers["nli"] = NLIVerifier(settings.nli_model, allow_download=settings.nli_allow_download)
    if settings.confidence_calibration_enabled:
        from app.services.confidence import ConfidenceEstimator

        layers["confidence"] = ConfidenceEstimator.from_file(settings.confidence_calibration_path)
    on = [k for k in ("reranker", "expansion", "classify", "verify_claims", "nli", "confidence") if layers.get(k)]
    if on:
        logger.info("Optional AI layers on: %s", ", ".join(on))
    return layers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store, embedding model and Ollama client once (voice models load on first use)."""
    app.state.rag = None
    app.state.documents = None
    app.state.tts_gemini = None
    app.state.startup_error = None
    app.state.stt = SpeechToText(settings.whisper_model, settings.whisper_device, settings.whisper_compute_type)
    app.state.tts = TextToSpeech(settings.tts_rate, settings.tts_voices)
    app.state.tts_gemini = GeminiSpeech(settings.gemini_api_key, settings.gemini_tts_model,
                                        {"female": settings.gemini_tts_voice_female, "male": settings.gemini_tts_voice_male})
    if app.state.tts_gemini.available:
        logger.info("Gemini voice available (%s)", settings.gemini_tts_model)
    # plant-disease detection: nothing is loaded here (a model loads on the first photo)
    from app.services.vision import VisionService, load_detector

    app.state.vision = VisionService(load_detector(settings), settings.vision_min_confidence, settings.vision_top_k)
    tracer.configure(settings.langsmith_tracing_enabled, settings.langsmith_api_key, settings.langsmith_project,
                     settings.langsmith_endpoint, settings.langsmith_log_text, settings.langsmith_passage_chars)
    if not app.state.stt.available:
        logger.info("Voice questions off: pip install faster-whisper to turn them on")
    elif settings.whisper_preload:
        threading.Thread(target=app.state.stt.model, name="whisper-preload", daemon=True).start()
    try:
        retriever = None
        if settings.fine_tuned_embeddings_enabled:
            # a separate store embedded with the fine-tuned model (scripts/build_collection.py); the original store
            # is the fallback when it or the model is missing
            store = settings.fine_tuned_store_path
            try:
                retriever = Retriever.from_store(store, hybrid=settings.hybrid_search, rrf_k=settings.rrf_k,
                                                 candidates=settings.retrieval_candidates, bm25_k1=settings.bm25_k1,
                                                 bm25_b=settings.bm25_b, keyword_weight=settings.keyword_weight,
                                                 keyword_seat=settings.keyword_seat)
                logger.info("Fine-tuned embeddings in use: %s (%s)", retriever.embedding_model, store)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FINE_TUNED_EMBEDDINGS_ENABLED=true but %s could not be used (%s); using the original "
                               "vector store", store, exc)
        if retriever is None:
            retriever = Retriever.from_store(
                vector_store_dir=settings.vector_store_path,
                collection_name=settings.collection_name,
                embedding_model=settings.embedding_model,
                hybrid=settings.hybrid_search,
                rrf_k=settings.rrf_k,
                candidates=settings.retrieval_candidates,
                bm25_k1=settings.bm25_k1,
                bm25_b=settings.bm25_b,
                keyword_weight=settings.keyword_weight,
                keyword_seat=settings.keyword_seat,
            )
        generator = OllamaGenerator(
            host=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.temperature,
            timeout=settings.ollama_timeout,
            think=settings.ollama_think,
            num_ctx=settings.ollama_num_ctx,
        )
        app.state.rag = RAGService(
            retriever,
            generator,
            top_k=settings.top_k,
            min_similarity=settings.min_similarity,
            translate_queries=settings.query_translation,
            self_check=settings.self_check,
            **optional_layers(),
        )
        gate = app.state.rag.min_similarity
        # beyond the library: the web's pages were chosen by the search engine for a farming question, so a lower
        # bar than the library's calibrated gate fits; learned passages answer again at the same bar
        beyond_gate = (settings.web_min_similarity if settings.web_min_similarity is not None
                       else max(round(gate - 0.12, 3), 0.4))
        passage_prefix = retriever.config.get("passage_prefix", "")
        knowledge = None
        if settings.learning:
            knowledge = KnowledgeStore(
                settings.learned_path, retriever.embedder, query_prefix=retriever.query_prefix,
                passage_prefix=passage_prefix, embedding_model=retriever.embedding_model,
                max_passages=settings.learned_max_passages,
                min_similarity=settings.learned_min_similarity if settings.learned_min_similarity is not None else beyond_gate,
            )
        web = None
        if settings.web_search:
            web = WebSearcher(settings.web_search_backend, settings.web_max_pages, settings.web_timeout,
                              settings.trusted_domains, settings.blocked_domains)
            if not web.available:
                logger.info("Web search off: pip install ddgs trafilatura to turn it on")
        app.state.rag.beyond = Beyond(
            retriever.embedder, retriever.query_prefix, passage_prefix, retriever.embedding_model,
            web=web, knowledge=knowledge, general=settings.general_knowledge,
            web_min_similarity=beyond_gate,
            learned_min_similarity=knowledge.min_similarity if knowledge is not None else beyond_gate,
            top_k=settings.top_k,
        )
        app.state.documents = DocumentService(
            retriever.embedder,
            generator,
            settings.uploads_path,
            query_prefix=retriever.query_prefix,
            passage_prefix=retriever.config.get("passage_prefix", ""),
            top_k=settings.top_k,
            min_similarity=settings.document_min_similarity,
            translate_queries=settings.query_translation,
            self_check=settings.self_check,
            max_documents=settings.max_documents,
            max_chunks=settings.max_document_chunks,
            embedding_model=retriever.embedding_model,
            ttl_minutes=settings.document_ttl_minutes,
            per_session=settings.documents_per_session,
            make_pdf=settings.document_clean_pdf,
            knowledge=knowledge,
        )
        app.state.documents.reset_storage()   # attached documents never outlive the backend
        reachable, available = generator.status()
        if not reachable:
            logger.warning("Ollama not reachable at %s; /query will return 503", settings.ollama_host)
        elif not available:
            logger.warning("Model '%s' not pulled. Run: ollama pull %s", settings.ollama_model, settings.ollama_model)
        logger.info(
            "RAG pipeline ready (%d chunks, LLM=%s, min_similarity=%.2f, hybrid=%s, top_k=%d, web=%s, learned=%s, "
            "general=%s)",
            retriever.count(),
            settings.ollama_model,
            app.state.rag.min_similarity,
            settings.hybrid_search,
            settings.top_k,
            bool(web and web.available),
            knowledge.count() if knowledge is not None else "off",
            settings.general_knowledge,
        )
    except Exception as exc:  # keep the API up so /health can explain the problem
        app.state.startup_error = f"Failed to load the RAG pipeline: {exc}"
        logger.exception(app.state.startup_error)
    yield
    logger.info("Shutting down")
    if app.state.documents is not None:
        app.state.documents.reset_storage()


app = FastAPI(
    title=settings.app_name,
    description="Ask farming questions and get answers grounded in agricultural documents, with citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(voice_router)
app.include_router(documents_router)
app.include_router(vision_router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": f"{settings.app_name} is running. See /docs for the API."}
