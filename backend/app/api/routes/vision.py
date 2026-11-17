"""Plant-disease detection from a photo (the computer-vision extension point; see docs/computer-vision-handoff.md).

GET  /vision/status     is a model configured and loadable?
POST /vision/diagnose   a photo (+ optional crop hint, language) -> top labels with probabilities, and — when the model
                        is confident it is a disease — treatment advice from the bulletins (the normal /query answer)

Off by default: with VISION_ENABLED=false, /vision/diagnose answers 503 with how to enable it.
"""

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.api.routes.query import to_response
from app.core.config import get_settings
from app.schemas.vision import DiagnosisResponse, VisionStatus
from app.services.vision import HEALTHY, NOTE, UNCERTAIN, BadImage, VisionService, VisionUnavailable, check_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision"])


def _service(request: Request) -> VisionService:
    service = getattr(request.app.state, "vision", None)
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Plant-disease detection is not set up.")
    return service


@router.get("/status", response_model=VisionStatus)
def vision_status(request: Request) -> VisionStatus:
    service = getattr(request.app.state, "vision", None)
    settings = get_settings()
    detector = getattr(service, "detector", None)
    return VisionStatus(enabled=settings.vision_enabled, model=getattr(detector, "name", None),
                        loaded=bool(getattr(detector, "_pipe", None)), failed=getattr(detector, "failed", None),
                        min_confidence=getattr(service, "min_confidence", settings.vision_min_confidence))


# plain `def`: the model and the RAG answer are blocking (FastAPI runs this in its thread pool)
@router.post("/diagnose", response_model=DiagnosisResponse)
def diagnose(request: Request, file: UploadFile = File(..., description="JPEG, PNG or WebP photo of the plant"),
             crop: str | None = Form(None, description="The crop, if the farmer knows it (improves the advice)"),
             language: str = Form("ar", pattern="^(ar|en)$"),
             advice: bool = Form(True, description="Also fetch treatment advice from the bulletins")) -> DiagnosisResponse:
    started = time.perf_counter()
    service = _service(request)
    settings = get_settings()
    data = file.file.read(int(settings.vision_max_mb * 1024 * 1024) + 1)
    try:
        check_image(data, file.content_type, settings.vision_max_mb)
    except BadImage as exc:
        too_big = "larger than" in str(exc)
        raise HTTPException(413 if too_big else (415 if "JPEG" in str(exc) else 400), str(exc)) from exc
    try:
        diagnosis = service.diagnose(data)
    except VisionUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    top = diagnosis.top
    confident = service.confident(diagnosis)
    message, answer = NOTE[language], None
    if top is None or not confident:
        message = UNCERTAIN[language]
    elif top.healthy:
        message = HEALTHY[language]
    elif advice:
        rag = getattr(request.app.state, "rag", None)
        if rag is not None:
            question = service.advice_question(top, language, crop)
            try:
                answer = to_response(rag.answer(question), started)
            except Exception as exc:  # noqa: BLE001 - the diagnosis stands without advice
                logger.warning("Advice for %s failed: %s", top.label, exc)
    logger.info("Photo diagnosis: %s (%.2f) confident=%s", top.label if top else None, top.confidence if top else 0, confident)
    return DiagnosisResponse(predictions=[p.to_dict() for p in diagnosis.predictions], top=top.to_dict() if top else None,
                             confident=confident, message=message, advice=answer, model=diagnosis.model,
                             latency_ms=round((time.perf_counter() - started) * 1000))
