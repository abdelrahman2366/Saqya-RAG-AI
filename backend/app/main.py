"""FastAPI entry point: CORS, logging, and loading the RAG pipeline once at startup."""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
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
from app.utils.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("app")


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
    if not app.state.stt.available:
        logger.info("Voice questions off: pip install faster-whisper to turn them on")
    elif settings.whisper_preload:
        threading.Thread(target=app.state.stt.model, name="whisper-preload", daemon=True).start()
    try:
        retriever = Retriever.from_store(
            vector_store_dir=settings.vector_store_path,
            collection_name=settings.collection_name,
            embedding_model=settings.embedding_model,
            hybrid=settings.hybrid_search,
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


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": f"{settings.app_name} is running. See /docs for the API."}
