"""POST /voice/transcribe (speech -> text, Egyptian Arabic or English), POST /voice/speak (text -> audio) and
GET /voice/stream (text -> audio streamed while it is made: the chat app speaks answers live with it).

Speaking takes an `engine` ("edge", the default, or "gemini" when GEMINI_API_KEY is set; Edge takes over
when Gemini fails) and, for Arabic, a `style`: "egyptian" (everyday words said the Egyptian way) or
"formal" (the standard-Arabic answer as written)."""

import logging
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.schemas.query import SpeakRequest, TranscriptResponse
from app.services.dialect import is_egyptian, to_msa
from app.services.spelling import repair
from app.services.speech import SpeechToText, SpeechUnavailableError, TextToSpeech, spoken_text
from app.utils.text import detect_language

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])

MAX_AUDIO_BYTES = 15 * 1024 * 1024  # ≈ 8 minutes of 16 kHz WAV


def _stt(request: Request) -> SpeechToText:
    stt = getattr(request.app.state, "stt", None)
    if stt is None or not stt.available:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Speech to text needs faster-whisper. Install it with: pip install faster-whisper")
    return stt


def _tts(request: Request) -> TextToSpeech:
    tts = getattr(request.app.state, "tts", None)
    if tts is None or not tts.available:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Text to speech needs edge-tts. Install it with: pip install edge-tts")
    return tts


def _engines(request: Request, engine: str | None) -> list:
    """The voice engines to try, in order: the one asked for, then Edge."""
    order = []
    gemini = getattr(request.app.state, "tts_gemini", None)
    if engine == "gemini" and gemini is not None and gemini.available:
        order.append(gemini)
    edge = getattr(request.app.state, "tts", None)
    if edge is not None and edge.available:
        order.append(edge)
    if not order:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Text to speech needs edge-tts. Install it with: pip install edge-tts")
    return order


def _chunks(engine, text: str, language: str, voice: str, style: str):
    """The engine's audio chunks for text already rewritten for the style (Gemini also gets the style)."""
    extra = {"style": style} if getattr(engine, "styles", False) else {}
    if hasattr(engine, "stream"):
        return engine.stream(text, language, voice, **extra)

    async def whole():   # a voice engine that can't stream: the whole clip at once
        yield await engine.synthesize(text, language, voice, **extra)
    return whole()


async def _first_audio(engines: list, text: str, language: str, voice: str, style: str):
    """(engine, first chunk, the rest) from the first engine that makes audio; errors as HTTP statuses."""
    spoken = spoken_text(text, language, style) or text
    error: Exception | None = None
    for engine in engines:
        chunks = _chunks(engine, spoken, language, voice, style)
        try:
            first = await anext(chunks)
            return engine, first, chunks
        except StopAsyncIteration:
            error = SpeechUnavailableError("The voice service returned no audio.")
        except SpeechUnavailableError as exc:
            error = exc
            logger.warning("Voice engine %s failed: %s", getattr(engine, "name", "?"), exc)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error or "No voice could be made."))


# Plain `def`: Whisper is blocking, FastAPI runs it in the thread pool
@router.post("/transcribe", response_model=TranscriptResponse)
def transcribe(request: Request, audio: UploadFile = File(..., description="A recording (WAV, MP3, M4A, OGG, WEBM)"),
               language: str | None = Form(None, description='"ar", "en", or empty to detect')) -> TranscriptResponse:
    stt = _stt(request)
    data = audio.file.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(422, "The recording is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "The recording is too long (limit ≈ 8 minutes).")
    lang = language if language in ("ar", "en") else None
    try:
        result = stt.transcribe(data, lang)
    except SpeechUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except Exception as exc:  # undecodable audio, ...
        logger.exception("Transcription failed")
        raise HTTPException(422, f"Could not understand the recording: {exc}")
    text = result.text
    language_out = result.language if result.language in ("ar", "en") else detect_language(text or "ا")
    text = repair(text)   # a mis-heard farming word ("مع حصيل" -> "محاصيل"), so the chat shows and asks the right one
    standard = to_msa(text) if language_out == "ar" else text
    logger.info("Transcribed %.1fs of audio in %.1fs (%s): %s", result.duration, result.seconds, language_out, text[:120])
    return TranscriptResponse(text=text, standard=standard, dialect=is_egyptian(text), language=language_out,
                              duration=result.duration, seconds=result.seconds)


@router.post("/speak", response_class=Response,
             responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}}, "description": "Audio of the text"}})
async def speak(payload: SpeakRequest, request: Request) -> Response:
    """The whole answer as one audio file: MP3 (Edge) or WAV (Gemini)."""
    engines = _engines(request, payload.engine)
    language = payload.language or detect_language(payload.text)
    engine, first, rest = await _first_audio(engines, payload.text, language, payload.voice, payload.style)
    audio = first + b"".join([chunk async for chunk in rest])
    return Response(content=audio, media_type=getattr(engine, "media_type", "audio/mpeg"))


@router.get("/stream", response_class=StreamingResponse,
            responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}},
                             "description": "Audio, streamed while it is made (MP3 from Edge, WAV from Gemini)"}})
async def stream(
    request: Request,
    text: str = Query(..., min_length=1, max_length=1500, description="What to say (one sentence or a few)"),
    language: Literal["ar", "en"] | None = Query(None, description="Detected from the text when omitted"),
    voice: str = Query("female", max_length=60, description='"female", "male", a voice id ("salma", "kore") or a voice name'),
    style: Literal["egyptian", "formal"] = Query("egyptian", description="Arabic answers: Egyptian everyday words, or formal standard Arabic"),
    engine: Literal["edge", "gemini"] = Query("edge", description="Voice engine (Edge takes over when Gemini fails)"),
) -> StreamingResponse:
    """Live speech. The audio starts arriving within a fraction of a second, so an `<audio>` element pointed
    at this URL starts talking at once. The chat app reads an answer sentence by sentence this way, loading
    the next sentence while the current one plays."""
    engines = _engines(request, engine)
    language = language or detect_language(text)
    # wait for the first audio, so a failure is still a proper error status (the app then uses the browser's voice)
    used, first, chunks = await _first_audio(engines, text, language, voice, style)

    async def body():
        yield first
        async for chunk in chunks:
            yield chunk

    # replaying a sentence comes from the browser's cache
    return StreamingResponse(body(), media_type=getattr(used, "media_type", "audio/mpeg"),
                             headers={"Cache-Control": "private, max-age=86400"})
