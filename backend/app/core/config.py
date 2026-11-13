"""Application settings, loaded from environment variables / backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ folder (this file is backend/app/core/config.py)
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,  # "COLLECTION_NAME=" in .env means "use the default"
    )

    app_name: str = "SAQYA AI API"
    log_level: str = "INFO"

    # Vector store produced by notebooks/rag_pipeline.ipynb (section 2.7)
    vector_store_dir: Path = Path("data/vector_store")
    collection_name: str | None = None  # None -> read from rag_config.json
    embedding_model: str | None = None  # None -> read from rag_config.json

    # Retrieval
    top_k: int = 6
    # Questions whose best cosine similarity is below this are refused without calling the LLM.
    # None -> the value the notebook calibrated for the embedding model (rag_config.json).
    min_similarity: float | None = None
    # Keyword (BM25) search fused with the embedding search. false = embeddings only.
    hybrid_search: bool = True

    # Local LLM (Ollama). qwen2.5 handles Arabic and English.
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: float = 120.0
    temperature: float = 0.0
    # Context window in tokens. Ollama's default is too small for 6 Arabic passages + the rules.
    ollama_num_ctx: int = 8192
    # Only for "thinking" models such as qwen3: false = answer directly (faster)
    ollama_think: bool | None = None
    # Also search with the question translated into the other language (Arabic <-> English).
    # Lets an English question find an Arabic bulletin and vice versa; costs one short LLM call.
    query_translation: bool = True
    # Check every answer (citations, numbers, repeated question, refusal) and let the LLM retry once.
    self_check: bool = True

    # Thumbs up/down from the chat UI are appended here, one JSON object per line (relative to backend/)
    feedback_file: Path = Path("data/feedback.jsonl")

    # ---- Voice (optional: pip install faster-whisper edge-tts)
    # Speech to text, local. large-v3-turbo understands Egyptian Arabic well (≈1.6 GB, downloaded on
    # first use); "small" is ~3x faster on a slow CPU, "medium" is in between.
    whisper_model: str = "large-v3-turbo"
    # "auto" uses an NVIDIA GPU when CUDA is set up, otherwise the CPU; or "cpu" / "cuda"
    whisper_device: str = "auto"
    # "default" = int8 on the CPU (fast, small), float16 on the GPU
    whisper_compute_type: str = "default"
    # Load the Whisper model at startup instead of on the first voice question (the first one is then fast)
    whisper_preload: bool = False
    # Text to speech: Microsoft Edge neural voices (needs internet). Speed: "-10%" slower, "+10%" faster
    tts_rate: str = "-5%"
    tts_voice_ar_female: str = "ar-EG-SalmaNeural"
    tts_voice_ar_male: str = "ar-EG-ShakirNeural"
    tts_voice_en_female: str = "en-US-JennyNeural"
    tts_voice_en_male: str = "en-US-GuyNeural"
    # Optional, more natural voice: Google's Gemini TTS (Egyptian or formal Arabic on request). Get a key at
    # Google AI Studio (free tier); empty = only the Edge voices are offered.
    gemini_api_key: str | None = None
    gemini_tts_model: str = "gemini-3.1-flash-tts-preview"
    gemini_tts_voice_female: str = "Kore"
    gemini_tts_voice_male: str = "Charon"

    # ---- Beyond the documents: farming questions the library doesn't answer
    # Search the web (free metasearch through the ddgs package: Wikipedia, DuckDuckGo, Bing, Brave, ...; no key)
    web_search: bool = True
    web_search_backend: str = "auto"      # ddgs engines: "auto" or a list such as "wikipedia,duckduckgo,brave"
    web_max_pages: int = 4                # pages read per question
    web_timeout: float = 6.0              # seconds per page
    # A web passage must be at least this similar to the question to be used (None: the library's gate - 0.12,
    # at least 0.4: the search engine already chose the pages, and the question was checked to be about farming)
    web_min_similarity: float | None = None
    web_trusted_domains: str = ""         # comma-separated sites to rank first (FAO, ministries, universities... are built in)
    web_blocked_domains: str = ""         # comma-separated sites never used (social media, video and shops are built in)
    # Keep what the web answers cite and what attached agricultural documents say, for later questions
    learning: bool = True
    learned_dir: Path = Path("data/learned")
    learned_max_passages: int = 20000
    learned_min_similarity: float | None = None   # None: the same as the web's (what answered once answers again)
    # When no source answers a farming question, let the LLM answer from general knowledge (marked as such)
    general_knowledge: bool = True

    # ---- Documents attached in the chat (temporary: cleaned, kept for the chat session, then deleted)
    # Folder for the clean PDFs, one sub-folder per chat session; emptied at startup and shutdown.
    # Empty = <system temp folder>/saqya-uploads
    uploads_dir: Path | None = None
    max_upload_mb: int = 25
    max_documents: int = 20          # in all sessions together (the least recently used go first)
    documents_per_session: int = 5
    max_document_chunks: int = 2500  # ≈ 600 pages
    # A document nobody has asked about for this long is deleted (a chat closed without saying so)
    document_ttl_minutes: int = 180
    # Write a clean PDF of every attached document (needs Playwright's Chromium, or falls back to PyMuPDF)
    document_clean_pdf: bool = True
    # Relevance gate inside one uploaded document (lower than the library's: the farmer chose the document)
    document_min_similarity: float = 0.3

    # Comma-separated list of allowed frontend origins (Streamlit runs on 8501)
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def vector_store_path(self) -> Path:
        """Resolve relative paths against backend/ so the app works from any cwd."""
        path = self.vector_store_dir
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def feedback_path(self) -> Path:
        path = self.feedback_file
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def uploads_path(self) -> Path:
        import tempfile

        path = self.uploads_dir
        if not path or not str(path).strip():
            return Path(tempfile.gettempdir()) / "saqya-uploads"
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def learned_path(self) -> Path:
        path = self.learned_dir
        return path if path.is_absolute() else BACKEND_DIR / path

    @staticmethod
    def _domains(value: str) -> tuple[str, ...]:
        return tuple(d.strip().lower() for d in (value or "").split(",") if d.strip())

    @property
    def trusted_domains(self) -> tuple[str, ...]:
        return self._domains(self.web_trusted_domains)

    @property
    def blocked_domains(self) -> tuple[str, ...]:
        return self._domains(self.web_blocked_domains)

    @property
    def tts_voices(self) -> dict:
        return {"ar": {"female": self.tts_voice_ar_female, "male": self.tts_voice_ar_male},
                "en": {"female": self.tts_voice_en_female, "male": self.tts_voice_en_male}}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
