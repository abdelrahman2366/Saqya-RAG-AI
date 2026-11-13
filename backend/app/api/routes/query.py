"""GET /health, POST /query, GET /library, POST /feedback and the learned knowledge (GET / DELETE /knowledge)
(voice: voice.py, attached documents: documents.py)."""

import json
import logging
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import get_settings
from app.schemas.query import (
    FeedbackRequest,
    FeedbackResponse,
    ForgetResponse,
    HealthResponse,
    KnowledgeResponse,
    LibraryResponse,
    Passage,
    QueryRequest,
    QueryResponse,
)
from app.services.beyond import SourcePolicy
from app.services.generation import LLMUnavailableError, clean_passage
from app.services.rag_service import RAGService, get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter()
_feedback_lock = threading.Lock()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    settings = get_settings()
    rag: RAGService | None = getattr(request.app.state, "rag", None)
    state = request.app.state
    beyond = getattr(getattr(state, "rag", None), "beyond", None)
    knowledge = getattr(beyond, "knowledge", None)
    engines = [getattr(e, "name", "edge") for e in (getattr(state, "tts_gemini", None), getattr(state, "tts", None))
               if e is not None and getattr(e, "available", False)]
    features = {
        "speech_to_text": bool(getattr(getattr(state, "stt", None), "available", False)),
        "text_to_speech": bool(getattr(getattr(state, "tts", None), "available", False)),
        "documents": getattr(state, "documents", None) is not None,
        "whisper_model": getattr(getattr(state, "stt", None), "model_name", None),
        "voice_engines": sorted(engines, key=lambda e: e != "edge"),
        "web_search": bool(beyond is not None and beyond.web_available),
        "learning": knowledge is not None,
        "learned_passages": knowledge.count() if knowledge is not None else 0,
        "general_knowledge": bool(beyond is not None and beyond.general),
    }

    if rag is None:
        return HealthResponse(
            status="degraded",
            vector_store_loaded=False,
            num_chunks=0,
            ollama_reachable=False,
            llm_model=settings.ollama_model,
            llm_model_available=False,
            detail=getattr(request.app.state, "startup_error", None),
            **features,
        )

    reachable, model_available = rag.generator.status()
    detail = None
    if not reachable:
        detail = f"Ollama is not reachable at {settings.ollama_host}."
    elif not model_available:
        detail = f"Model '{rag.generator.model}' not found. Run: ollama pull {rag.generator.model}"

    return HealthResponse(
        status="ok" if reachable and model_available else "degraded",
        vector_store_loaded=True,
        num_chunks=rag.retriever.count(),
        ollama_reachable=reachable,
        llm_model=rag.generator.model,
        llm_model_available=model_available,
        detail=detail,
        **features,
    )


# Plain `def` (not async): FastAPI runs it in a thread pool, so the blocking
# embedding + Ollama calls don't freeze the event loop.
@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(payload: QueryRequest, rag: RAGService = Depends(get_rag_service)) -> QueryResponse:
    logger.info("Question: %s%s", payload.question[:120], f" (focus: {payload.topic})" if payload.topic else "")
    started = time.perf_counter()
    try:
        history = [h[:500] for h in payload.history if h and h.strip()][-4:]
        kwargs = {"topic": payload.topic} if payload.topic else {}
        if history:
            kwargs["history"] = history
        if rag.beyond is not None:
            kwargs["policy"] = SourcePolicy(web=payload.web, learn=payload.learn, general=payload.general)
        result = rag.answer(payload.question, **kwargs)
    except LLMUnavailableError as exc:
        logger.error("LLM unavailable: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return to_response(result, started)


def _pages(c) -> str:
    if getattr(c, "url", "") or not c.page_start:
        return ""
    return f"p. {c.page_start}" if c.page_start == c.page_end else f"pp. {c.page_start}-{c.page_end}"


def to_response(result, started: float) -> QueryResponse:
    """RAGResult -> QueryResponse (also used for questions about an uploaded document)."""
    passages = [
        Passage(
            n=n,
            source=c.source,
            pages=_pages(c),
            page_start=c.page_start,
            page_end=c.page_end,
            section=getattr(c, "section", "") or "",
            topic=c.topic,
            score=min(max(c.score, 0.0), 1.0) if c.score == c.score else 0.0,  # NaN-safe
            text=clean_passage(c.text),
            url=getattr(c, "url", "") or "",
            title=getattr(c, "title", "") or "",
            kind=getattr(c, "kind", "") or "",
        )
        for n, c in getattr(result, "cited", [])
    ]
    origin = getattr(result, "origin", "library") if result.sources or getattr(result, "origin", "") == "general" else "none"
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        passages=passages,
        language=result.language if result.language in ("ar", "en") else "en",
        refused=origin == "none",
        topic=getattr(result, "topic", None),
        origin=origin if origin in ("library", "document", "learned", "web", "general", "none") else "library",
        note=getattr(result, "note", "") or "",
        off_topic=bool(getattr(result, "off_topic", False)),
        learned=getattr(result, "learned", 0) or 0,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


@router.get("/library", response_model=LibraryResponse, tags=["rag"])
def library(rag: RAGService = Depends(get_rag_service)) -> LibraryResponse:
    """What the assistant knows: indexed documents, their crops and languages."""
    retriever = rag.retriever
    if hasattr(retriever, "library"):
        return LibraryResponse(**retriever.library())
    return LibraryResponse(num_documents=0, num_chunks=retriever.count(), languages={}, crops=[], documents=[])


@router.post("/feedback", response_model=FeedbackResponse, tags=["system"])
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """Store a thumbs up / down on an answer (one JSON line per entry) to find weak spots later."""
    path = get_settings().feedback_path
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload.model_dump()}
    with _feedback_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        total = helpful = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rating = json.loads(line).get("rating")
            except ValueError:
                continue
            total += 1
            helpful += rating == "up"
    logger.info("Feedback saved (%s): %s", payload.rating, payload.question[:80])
    return FeedbackResponse(total=total, helpful=helpful)


# ---------------------------------------------------------------- learned knowledge
def _knowledge(request: Request):
    beyond = getattr(getattr(request.app.state, "rag", None), "beyond", None)
    knowledge = getattr(beyond, "knowledge", None)
    if knowledge is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Learning is off (LEARNING=false) or the backend isn't ready.")
    return knowledge


@router.get("/knowledge", response_model=KnowledgeResponse, tags=["knowledge"])
def knowledge(request: Request) -> KnowledgeResponse:
    """What SAQYA has learned from the web and attached documents, one entry per source."""
    beyond = getattr(getattr(request.app.state, "rag", None), "beyond", None)
    store = getattr(beyond, "knowledge", None)
    if store is None:
        return KnowledgeResponse(enabled=False)
    return KnowledgeResponse(enabled=True, **store.summary())


@router.delete("/knowledge/sources/{source_id}", response_model=ForgetResponse, tags=["knowledge"])
def forget_source(source_id: str, request: Request) -> ForgetResponse:
    """Forget everything learned from one web page or document."""
    removed = _knowledge(request).forget(source_id)
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nothing was learned from that source.")
    return ForgetResponse(removed=removed)


@router.delete("/knowledge", response_model=ForgetResponse, tags=["knowledge"])
def forget_all(request: Request) -> ForgetResponse:
    """Forget everything SAQYA has learned (the indexed library is not touched)."""
    return ForgetResponse(removed=_knowledge(request).forget_all())
