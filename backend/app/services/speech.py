"""Voice: speech to text (faster-whisper, local) and text to speech (Microsoft Edge neural voices).

Speech to text
    faster-whisper runs OpenAI's Whisper locally (CTranslate2, no PyTorch). The default model,
    large-v3-turbo, understands Egyptian colloquial Arabic well; "small" is ~3x faster on a weak CPU.
    The model is downloaded from Hugging Face on first use (large-v3-turbo ≈ 1.6 GB) and loaded once.
    An initial prompt written in Egyptian farming Arabic nudges Whisper towards the words farmers use
    (أوطة، شيكارة، الغيط، الرش...). The transcript is then rewritten into standard Arabic for the search
    by dialect.py, like a typed question.

Text to speech
    Two engines, picked in the chat app's Settings:
    - Edge (default, free, no key): Microsoft's neural voices through edge-tts — Egyptian (Salma, Shakir)
      and Saudi (Zariyah, Hamed) Arabic voices. MP3, streamed while it is made.
    - Gemini (optional, more natural): Google's Gemini TTS, which takes a style instruction, so the same voice
      speaks warm Egyptian Arabic or formal standard Arabic. Needs GEMINI_API_KEY (Google AI Studio; it has a
      free tier). WAV, one clip per sentence. When it fails, the Edge voice takes over.
    Both need internet; without it the chat app falls back to the browser's own speech synthesis.
    Arabic answers are spoken in one of two styles: "formal" reads the standard-Arabic answer as written,
    "egyptian" first rewrites its everyday words into Egyptian ones (dialect.to_egyptian). Citations [n],
    Markdown and emoji are removed before reading.

Both libraries are optional: the API reports them in /health and the voice endpoints answer 503 with
an install hint when they are missing.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import logging
import re
import threading
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Whisper reads the previous text as context: a few typical Egyptian farmer questions bias it towards
# farming words and colloquial spelling instead of guessing similar-sounding standard words.
INITIAL_PROMPTS = {
    "ar": "سؤال من فلاح مصري عن الزراعة: إمتى أزرع القمح؟ إزاي أكافح الدودة في الأوطة؟ "
          "كام شيكارة سماد للفدان؟ إيه أفضل المحاصيل اللي أزرعها الشهر ده؟ البرسيم والقطن والذرة والبطاطس "
          "والطماطم والمحاصيل والأسمدة والمبيدات والآفات والرش والري والتقاوي والشتلات في الغيط.",
    "en": "A farmer's question about crops: when to plant wheat, fertilizer per feddan, pests, irrigation.",
}

VOICES = {
    "ar": {"female": "ar-EG-SalmaNeural", "male": "ar-EG-ShakirNeural"},
    "en": {"female": "en-US-JennyNeural", "male": "en-US-GuyNeural"},
}

# The voices the chat app offers, per engine: id -> (voice name, gender). English answers use the engine's
# English voice of the same gender (Edge) or the same voice (Gemini speaks every language).
VOICE_CATALOG = {
    "edge": {
        "salma": ("ar-EG-SalmaNeural", "female"),     # Egyptian
        "shakir": ("ar-EG-ShakirNeural", "male"),     # Egyptian
        "zariyah": ("ar-SA-ZariyahNeural", "female"), # Saudi
        "hamed": ("ar-SA-HamedNeural", "male"),       # Saudi
    },
    "gemini": {
        "kore": ("Kore", "female"),
        "aoede": ("Aoede", "female"),
        "despina": ("Despina", "female"),
        "charon": ("Charon", "male"),
        "orus": ("Orus", "male"),
        "puck": ("Puck", "male"),
    },
}
STYLES = ("egyptian", "formal")


class SpeechUnavailableError(RuntimeError):
    """The voice library is not installed, or the model / service can't be used right now."""


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@dataclass
class Transcript:
    text: str
    language: str        # "ar" or "en"
    duration: float      # seconds of audio
    seconds: float       # processing time


SAMPLE_RATE = 16000   # what Whisper expects


def decode_audio(data: bytes, rate: int = SAMPLE_RATE) -> np.ndarray:
    """A recording -> mono float32 samples at `rate` Hz.

    WAV (what the chat app's microphone records) is read with the standard library. Other formats (mp3, m4a,
    ogg, webm) go through PyAV — called directly, because faster-whisper 1.2.1's own decoder passes
    `metadata_errors=` to av.open(), which PyAV 15 and later no longer accept ("open() got an unexpected
    keyword argument 'metadata_errors'").
    """
    try:
        with wave.open(io.BytesIO(data)) as w:
            channels, width, src_rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.readframes(w.getnframes())
        if width == 1:
            samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 3:
            raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
            ints = (raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16))
            samples = np.where(ints >= 1 << 23, ints - (1 << 24), ints).astype(np.float32) / float(1 << 23)
        elif width == 4:
            samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / float(1 << 31)
        else:
            raise wave.Error(f"unsupported sample width {width}")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if src_rate != rate and samples.size:
            n = max(1, round(samples.size * rate / src_rate))
            samples = np.interp(np.linspace(0, samples.size - 1, n), np.arange(samples.size), samples)
        return samples.astype(np.float32)
    except (wave.Error, EOFError):
        pass   # not a plain PCM WAV: let PyAV read it
    import av

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=rate)
    parts = []
    with av.open(io.BytesIO(data), mode="r") as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):   # flush
            parts.append(out.to_ndarray().reshape(-1))
    if not parts:
        raise ValueError("The recording has no sound.")
    return (np.concatenate(parts).astype(np.float32) / 32768.0)


class SpeechToText:
    """Lazy-loaded faster-whisper model (thread-safe)."""

    def __init__(self, model: str = "large-v3-turbo", device: str = "auto", compute_type: str = "default",
                 beam_size: int = 5) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return installed("faster_whisper")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self, device: str | None = None):
        if not self.available:
            raise SpeechUnavailableError(
                "Speech to text needs faster-whisper. Install it with: pip install faster-whisper")
        from faster_whisper import WhisperModel

        device = device or self.device
        compute = self.compute_type
        if compute == "default":
            compute = "int8" if device == "cpu" else "default"
        logger.info("Loading Whisper model %s (device=%s, compute=%s); the first time it is downloaded",
                    self.model_name, device, compute)
        started = time.perf_counter()
        try:
            model = WhisperModel(self.model_name, device=device, compute_type=compute)
        except Exception as exc:
            if device != "cpu":  # e.g. CUDA libraries missing: CPU always works
                logger.warning("Whisper on %s failed (%s); using the CPU", device, exc)
                return self._load("cpu")
            raise SpeechUnavailableError(f"Could not load the Whisper model '{self.model_name}': {exc}") from exc
        self.device = device
        logger.info("Whisper model ready in %.1fs", time.perf_counter() - started)
        return model

    def model(self):
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return self._model

    def transcribe(self, audio: bytes, language: str | None = None) -> Transcript:
        """`language`: "ar", "en" or None (detect; anything other than Arabic or English is read as Arabic)."""
        if not audio:
            raise ValueError("Empty recording.")
        started = time.perf_counter()
        try:
            text, lang, duration = self._run(audio, language)
        except SpeechUnavailableError:
            raise
        except RuntimeError as exc:
            # CUDA found but its libraries missing (cublas / cudnn) only fails at the first transcription
            if self.device != "cpu" and re.search(r"cuda|cublas|cudnn", str(exc), re.IGNORECASE):
                logger.warning("Whisper on GPU failed (%s); switching to the CPU", exc)
                with self._lock:
                    self._model = self._load("cpu")
                text, lang, duration = self._run(audio, language)
            else:
                raise
        return Transcript(text=text, language=lang, duration=round(duration, 2),
                          seconds=round(time.perf_counter() - started, 2))

    def _run(self, audio: bytes, language: str | None) -> tuple[str, str, float]:
        model = self.model()
        try:
            samples = decode_audio(audio)
        except Exception as exc:  # damaged file, unknown format
            raise ValueError(f"The recording could not be read: {exc}") from exc
        if samples.size < SAMPLE_RATE // 4:
            raise ValueError("The recording is too short.")

        def run(lang: str | None):
            segments, info = model.transcribe(
                samples, language=lang, beam_size=self.beam_size, vad_filter=True,
                initial_prompt=INITIAL_PROMPTS.get(lang or "ar"), condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip(), info

        text, info = run(language)
        lang = language or info.language
        if lang not in ("ar", "en"):  # short Egyptian clips are sometimes detected as another language
            text, info = run("ar")
            lang = "ar"
        # Whisper echoes its prompt on silence: never return the prompt itself as the question
        if text and text in INITIAL_PROMPTS.get(lang, ""):
            text = ""
        return text, lang, float(getattr(info, "duration", 0.0) or samples.size / SAMPLE_RATE)


# ---------------------------------------------------------------- text to speech
_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,،]\s*\d+)*\]")
_MARKDOWN_RE = re.compile(r"[*_`#>|]+")
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-•*]|\d{1,2}[.)])\s+")


def speech_text(text: str, limit: int = 2500) -> str:
    """What should be read aloud: no [n] citations, Markdown, bullets or emoji; at most `limit` characters."""
    text = _CITATION_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _EMOJI_RE.sub("", _MARKDOWN_RE.sub(" ", text))
    text = re.sub(r"\s*\n+\s*", ". ", text.strip())
    text = re.sub(r"\.\s*\.", ".", re.sub(r"[ \t]{2,}", " ", text)).strip()
    text = re.sub(r"^[.\s]+", "", re.sub(r"\s+([.،,؟?!])", r"\1", text))
    if len(text) > limit:
        cut = max(text.rfind(".", 0, limit), text.rfind("،", 0, limit), text.rfind("؟", 0, limit))
        text = text[: cut + 1 if cut > limit // 2 else limit]
    return text


def spoken_text(text: str, language: str, style: str = "egyptian") -> str:
    """What the voice reads: `speech_text`, and for Arabic in the Egyptian style its everyday words in Egyptian."""
    from app.services.dialect import to_egyptian

    spoken = speech_text(text)
    return to_egyptian(spoken) if language == "ar" and style == "egyptian" else spoken


class TextToSpeech:
    """Edge neural voices (MP3, streamed)."""

    name = "edge"
    media_type = "audio/mpeg"

    def __init__(self, rate: str = "-5%", voices: dict | None = None) -> None:
        self.rate = rate
        self.voices = voices or VOICES

    @property
    def available(self) -> bool:
        return installed("edge_tts")

    def voice_for(self, language: str, voice: str = "female") -> str:
        """`voice`: "female", "male", a catalog id ("salma", "hamed") or a full voice name ("ar-EG-ShakirNeural")."""
        if "-" in voice and voice.endswith("Neural"):
            return voice
        table = self.voices.get(language, self.voices["ar"])
        if voice in VOICE_CATALOG["edge"]:
            name, gender = VOICE_CATALOG["edge"][voice]
            return name if language == "ar" else table.get(gender, table["female"])
        if voice in VOICE_CATALOG["gemini"]:   # a Gemini voice asked of Edge (fallback): same gender
            return table.get(VOICE_CATALOG["gemini"][voice][1], table["female"])
        return table.get(voice, table["female"])

    async def stream(self, text: str, language: str = "ar", voice: str = "female"):
        """MP3 audio of the text, chunk by chunk as the voice service makes it (for live playback: the first
        words can be heard while the rest is still being made). Raises before the first chunk when no audio
        can be made; a connection lost later just ends the stream."""
        if not self.available:
            raise SpeechUnavailableError("Text to speech needs edge-tts. Install it with: pip install edge-tts")
        import edge_tts

        spoken = speech_text(text)
        if not spoken:
            raise ValueError("Nothing to read.")
        sent = False
        try:
            communicate = edge_tts.Communicate(spoken, self.voice_for(language, voice), rate=self.rate)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and chunk["data"]:
                    sent = True
                    yield chunk["data"]
        except Exception as exc:  # no internet, service refused...
            if not sent:
                raise SpeechUnavailableError(f"The online voice service could not be reached: {exc}") from exc
            logger.warning("Voice stream cut short: %s", exc)
            return
        if not sent:
            raise SpeechUnavailableError("The voice service returned no audio.")

    async def synthesize(self, text: str, language: str = "ar", voice: str = "female") -> bytes:
        """MP3 audio of the text, all at once."""
        return b"".join([chunk async for chunk in self.stream(text, language, voice)])


# ---------------------------------------------------------------- Gemini TTS (optional)
GEMINI_STYLE_PROMPTS = {
    ("ar", "egyptian"): "Say this in a warm, natural Egyptian Arabic accent from Cairo, like a friendly farming "
                        "advisor talking to a farmer, at a calm pace",
    ("ar", "formal"): "Say this in clear, formal Modern Standard Arabic (Fusha), like a calm and professional "
                      "news presenter, at a steady pace",
    ("en", "egyptian"): "Say this in a clear, friendly and natural voice, at a calm pace",
    ("en", "formal"): "Say this in a clear, friendly and natural voice, at a calm pace",
}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    """Gemini sends raw 16-bit mono PCM; browsers play it with a WAV header."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


class GeminiSpeech:
    """Google's Gemini TTS: one natural voice that follows a style instruction (Egyptian or formal Arabic).
    Not streamed: each sentence comes back as one WAV clip (≈1–3 s), and the chat app loads the next sentence
    while the current one plays. Clips are cached, so "listen again" costs nothing."""

    name = "gemini"
    media_type = "audio/wav"
    styles = True   # accepts a `style` argument

    def __init__(self, api_key: str | None, model: str = "gemini-3.1-flash-tts-preview",
                 voices: dict | None = None, timeout: float = 40.0, cache_size: int = 300) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.voices = voices or {"female": "Kore", "male": "Charon"}
        self.timeout = timeout
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    REST_SECONDS = 600   # after a quota error, the free Edge voice reads everything for this long

    @property
    def available(self) -> bool:
        """Has a key, and is not resting after its free quota ran out (then the Edge voice is used, so an answer
        is never read half by one voice and half by another)."""
        return bool(self.api_key) and installed("httpx") and time.time() >= getattr(self, "rest_until", 0)

    def voice_for(self, voice: str = "female") -> str:
        if voice in VOICE_CATALOG["gemini"]:
            return VOICE_CATALOG["gemini"][voice][0]
        if voice in VOICE_CATALOG["edge"]:        # an Edge voice id: the Gemini voice of the same gender
            return self.voices.get(VOICE_CATALOG["edge"][voice][1], self.voices["female"])
        if voice in self.voices:
            return self.voices[voice]
        names = {name for name, _ in VOICE_CATALOG["gemini"].values()}
        return voice if voice in names else self.voices["female"]

    def request_body(self, text: str, language: str, voice: str, style: str) -> dict:
        instruction = GEMINI_STYLE_PROMPTS.get((language, style), GEMINI_STYLE_PROMPTS[("en", "formal")])
        return {
            "contents": [{"parts": [{"text": f"{instruction}: {text}"}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice_for(voice)}}},
            },
        }

    @staticmethod
    def audio_from(response: dict) -> bytes:
        """The WAV clip in a generateContent response (raw PCM, rate from its mime type, e.g. audio/L16;rate=24000)."""
        for candidate in response.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data") or {}
                if inline.get("data"):
                    mime = inline.get("mimeType") or inline.get("mime_type") or ""
                    match = re.search(r"rate=(\d+)", mime)
                    pcm = base64.b64decode(inline["data"])
                    if pcm[:4] == b"RIFF":     # already a WAV file
                        return pcm
                    return pcm_to_wav(pcm, int(match.group(1)) if match else 24000)
        raise SpeechUnavailableError("The Gemini voice returned no audio.")

    async def stream(self, text: str, language: str = "ar", voice: str = "female", style: str = "egyptian"):
        """The clip for `text` (already in the words to speak), as one chunk."""
        if not self.available:
            raise SpeechUnavailableError("The Gemini voice is not available right now (no GEMINI_API_KEY, or its free "
                                         "quota is used up).")
        import httpx

        spoken = speech_text(text)
        if not spoken:
            raise ValueError("Nothing to read.")
        key = (self.model, self.voice_for(voice), language, style, spoken)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is None:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    for attempt in (1, 2):
                        response = await client.post(GEMINI_URL.format(model=self.model),
                                                     headers={"x-goog-api-key": self.api_key},
                                                     json=self.request_body(spoken, language, voice, style))
                        if response.status_code not in (429, 500, 503) or attempt == 2:
                            break
                        await asyncio.sleep(2.5)   # a brief rate limit or hiccup: one more try before the Edge voice
            except Exception as exc:  # no internet, timeout...
                raise SpeechUnavailableError(f"The Gemini voice could not be reached: {exc}") from exc
            if response.status_code == 429:
                self.rest_until = time.time() + self.REST_SECONDS
                logger.warning("Gemini voice: free quota used up; using the Edge voice for %d minutes", self.REST_SECONDS // 60)
                raise SpeechUnavailableError("The Gemini voice's free quota is used up for now (429); using the "
                                             "Microsoft voice until it resets.")
            if response.status_code != 200:
                detail = response.text[:300]
                raise SpeechUnavailableError(f"The Gemini voice refused the request ({response.status_code}): {detail}")
            cached = self.audio_from(response.json())
            with self._lock:
                self._cache[key] = cached
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        yield cached

    async def synthesize(self, text: str, language: str = "ar", voice: str = "female", style: str = "egyptian") -> bytes:
        return b"".join([chunk async for chunk in self.stream(text, language, voice, style)])
