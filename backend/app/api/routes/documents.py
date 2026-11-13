"""Documents attached in the chat: cleaned, kept for the chat session, asked about or explained.

POST   /documents                   attach a PDF, Word (.docx), .txt or .md file (+ the chat `session`) -> DocumentInfo
GET    /documents?session=…         the session's documents, newest first
DELETE /documents?session=…         remove all of them (the chat was cleared or closed)
GET    /documents/{id}              one document
GET    /documents/{id}/clean.pdf    the clean PDF made from it (cover, contents, headers… removed)
DELETE /documents/{id}              remove it
POST   /documents/{id}/query        a question about it (answered only from it, with citations; `history` for follow-ups)
POST   /documents/{id}/explain      explain the whole document, or one section / topic

Documents are temporary: deleted with the session, after DOCUMENT_TTL_MINUTES unused, and when the backend stops.
"""

import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.api.routes.query import to_response
from app.core.config import get_settings
from app.schemas.query import DocumentInfo, DocumentList, DocumentQuestion, ExplainRequest, QueryResponse
from app.services.documents import DocumentNotFoundError, DocumentService
from app.services.extraction import ExtractionError
from app.services.generation import LLMUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _service(request: Request) -> DocumentService:
    service = getattr(request.app.state, "documents", None)
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            getattr(request.app.state, "startup_error", None)
                            or "Document upload is not available (the embedding model is not loaded).")
    return service


def _not_found(doc_id: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND,
                         f"Document '{doc_id}' not found. It was removed when the chat ended or after a long pause; "
                         "attach it again.")


# Plain `def` everywhere: reading, embedding and the LLM are blocking (FastAPI uses its thread pool)
@router.post("", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
def upload(request: Request, file: UploadFile = File(..., description="PDF, Word (.docx), .txt or .md"),
           session: str | None = Form(None, description="The chat session the document belongs to"),
           learn: bool = Form(False, description="Also keep its passages in SAQYA's learned knowledge "
                              "(agricultural documents only)")) -> DocumentInfo:
    service = _service(request)
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"The file is larger than {get_settings().max_upload_mb} MB.")
    if not data:
        raise HTTPException(422, "The file is empty.")
    try:
        doc = service.add(data, file.filename or "document", session, learn=learn)
    except ExtractionError as exc:
        code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE if "Unsupported file type" in str(exc) else 422
        raise HTTPException(code, str(exc))
    return DocumentInfo(**doc.summary())


@router.get("", response_model=DocumentList)
def list_documents(request: Request, session: str | None = Query(None)) -> DocumentList:
    return DocumentList(documents=[DocumentInfo(**d) for d in _service(request).list(session)])


@router.delete("")
def delete_session_documents(request: Request, session: str = Query(..., min_length=6, max_length=64)) -> dict:
    return {"deleted": _service(request).delete_session(session)}


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str, request: Request) -> DocumentInfo:
    try:
        return DocumentInfo(**_service(request).get(doc_id).summary())
    except DocumentNotFoundError:
        raise _not_found(doc_id)


@router.get("/{doc_id}/clean.pdf", response_class=FileResponse,
            responses={200: {"content": {"application/pdf": {}}, "description": "The clean PDF"}})
def clean_pdf(doc_id: str, request: Request) -> FileResponse:
    service = _service(request)
    try:
        doc = service.get(doc_id)
        path = service.clean_pdf(doc_id)
    except DocumentNotFoundError:
        raise _not_found(doc_id)
    stem = re.sub(r"[^\w\-. ]+", "_", Path(doc.filename).stem).strip() or "document"
    return FileResponse(path, media_type="application/pdf", filename=f"{stem} (clean).pdf")


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, request: Request) -> Response:
    try:
        _service(request).delete(doc_id)
    except DocumentNotFoundError:
        raise _not_found(doc_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{doc_id}/query", response_model=QueryResponse)
def ask_document(doc_id: str, payload: DocumentQuestion, request: Request) -> QueryResponse:
    service = _service(request)
    started = time.perf_counter()
    history = [h[:500] for h in payload.history if h and h.strip()][-4:]
    try:
        result = service.answer(doc_id, payload.question, history or None)
    except DocumentNotFoundError:
        raise _not_found(doc_id)
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return to_response(result, started)


@router.post("/{doc_id}/explain", response_model=QueryResponse)
def explain_document(doc_id: str, payload: ExplainRequest, request: Request) -> QueryResponse:
    service = _service(request)
    started = time.perf_counter()
    try:
        result = service.explain(doc_id, payload.focus, payload.language)
    except DocumentNotFoundError:
        raise _not_found(doc_id)
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return to_response(result, started)
