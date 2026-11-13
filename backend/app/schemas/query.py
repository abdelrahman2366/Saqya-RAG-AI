"""Request / response models for the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    # Strip spaces first so "   " counts as empty and is rejected with 422
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The farmer's question in plain language, in Arabic or English.",
        examples=["When is the best time to plant maize?", "ما هو أفضل موعد لزراعة القمح؟"],
    )
    topic: str | None = Field(
        default=None,
        max_length=80,
        description="Optional crop to focus on (its English key from GET /library, e.g. 'sunflower'). "
        "Only that crop's documents are searched. Unknown values are ignored.",
        examples=["sunflower"],
    )
    history: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="The farmer's previous questions in this conversation, oldest first. A follow-up such as "
        "'how much water does it need?' is then searched together with the question it follows.",
        examples=[["When should I plant tomatoes?"]],
    )
    # What may be used when the documents don't cover a farming question (Settings in the chat app)
    web: bool = Field(True, description="Search the web when the documents and what SAQYA learned don't answer")
    learn: bool = Field(True, description="Keep the web passages the answer cites, for later questions")
    general: bool = Field(True, description="Answer from general knowledge (marked) when no source answers")


class Passage(BaseModel):
    """A passage the answer cites; `n` matches the [n] marker in the answer."""

    n: int
    source: str
    pages: str = Field(..., description='"p. 3" or "pp. 3-4"')
    page_start: int
    page_end: int
    section: str = Field("", description="Section heading the passage comes from (cleaned documents)")
    topic: str = Field("", description='The crop the document is about, e.g. "دوار الشمس / sunflower"')
    score: float = Field(..., description="Cosine similarity to the question (higher = closer)")
    text: str
    url: str = Field("", description="The web page, for passages from the web or learned from it")
    title: str = Field("", description="The web page's or learned document's title")
    kind: str = Field("", description='"" (the library or the attached document), "web" or "learned"')


class QueryResponse(BaseModel):
    answer: str = Field(
        ..., description="Answer grounded in the retrieved documents, in the question's language."
    )
    sources: list[str] = Field(
        default_factory=list,
        description='Cited sources, e.g. "[1] maize_guide.pdf, p. 12". '
        "The number matches the [n] citation in the answer.",
    )
    passages: list[Passage] = Field(
        default_factory=list, description="The cited passages themselves, in the same order as `sources`."
    )
    language: Literal["ar", "en"] = "en"
    refused: bool = Field(False, description="True when no source could answer (or it isn't about farming).")
    topic: str | None = Field(None, description="The crop focus that was applied, if any.")
    origin: Literal["library", "document", "learned", "web", "general", "none"] = Field(
        "library", description="Where the answer comes from: the indexed library, the attached document, what SAQYA "
        "learned before, a web search, general knowledge (no source), or none (refused)")
    note: str = Field("", description="A note to show with the answer (general knowledge: not from a source)")
    off_topic: bool = Field(False, description="Refused because the question isn't about farming")
    learned: int = Field(0, description="Passages of this answer kept for later questions")
    latency_ms: int = 0


class HealthResponse(BaseModel):
    status: str  # "ok" or "degraded"
    vector_store_loaded: bool
    num_chunks: int
    ollama_reachable: bool
    llm_model: str
    llm_model_available: bool
    detail: str | None = None
    # optional features (the chat app shows or hides them)
    speech_to_text: bool = Field(False, description="POST /voice/transcribe works (faster-whisper installed)")
    text_to_speech: bool = Field(False, description="POST /voice/speak works (edge-tts installed; needs internet)")
    documents: bool = Field(False, description="Uploading your own document works")
    whisper_model: str | None = None
    voice_engines: list[str] = Field(default_factory=list, description='Voice engines that can speak: "edge", "gemini"')
    web_search: bool = Field(False, description="Questions the documents don't cover can be searched on the web")
    learning: bool = Field(False, description="What the web and attached documents teach is kept for later questions")
    learned_passages: int = Field(0, description="Passages learned so far")
    general_knowledge: bool = Field(False, description="The LLM may answer farming questions no source covers (marked)")


class LibraryDocument(BaseModel):
    source: str
    title: str = ""
    topic: str = ""
    language: str = ""
    chunks: int
    pages: int


class LibraryCrop(BaseModel):
    key: str = Field(..., description="English name, the value to send as `topic` in POST /query")
    ar: str
    en: str
    documents: int
    chunks: int


class LibraryResponse(BaseModel):
    num_documents: int
    num_chunks: int
    languages: dict[str, int]
    crops: list[LibraryCrop]
    documents: list[LibraryDocument]
    embedding_model: str = ""
    built_at: str = ""


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=1, max_length=1000)
    answer: str = Field(..., min_length=1, max_length=6000)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)
    sources: list[str] = Field(default_factory=list, max_length=20)
    topic: str | None = Field(default=None, max_length=80)


class FeedbackResponse(BaseModel):
    status: Literal["saved"] = "saved"
    total: int = Field(..., description="Feedback entries saved so far")
    helpful: int = Field(..., description="How many of them were 'up'")


# ---------------------------------------------------------------- voice
class TranscriptResponse(BaseModel):
    text: str = Field(..., description="What was said, as Whisper wrote it")
    standard: str = Field(..., description="The same question in standard Arabic (what is searched); = text for English")
    dialect: bool = Field(False, description="True when Egyptian colloquial words were found")
    language: Literal["ar", "en"]
    duration: float = Field(..., description="Seconds of audio")
    seconds: float = Field(..., description="Processing time")


class SpeakRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, max_length=6000)
    language: Literal["ar", "en"] | None = Field(None, description="Detected from the text when omitted")
    voice: str = Field("female", max_length=60, description='"female", "male", a voice id ("salma", "kore") or a '
                       'voice name like "ar-EG-ShakirNeural"')
    style: Literal["egyptian", "formal"] = Field("egyptian", description="Arabic: Egyptian everyday words, or formal standard Arabic")
    engine: Literal["edge", "gemini"] = Field("edge", description="Voice engine; Edge takes over when Gemini fails")


# ---------------------------------------------------------------- uploaded documents
class DocumentInfo(BaseModel):
    id: str
    filename: str
    title: str
    language: str
    pages: int
    skipped_pages: int = 0
    chunks: int
    structured: bool = False
    sections: list[str] = Field(default_factory=list, description="Top-level section headings, in order")
    topic: str = Field("", description='The crop the document is about, e.g. "الطماطم / tomato"')
    created_at: str
    clean_pdf: bool = Field(False, description="A clean PDF was made: GET /documents/{id}/clean.pdf")
    cleaning: dict = Field(default_factory=dict, description='How it was read: method ("cleaned" or "plain"), '
                           "dropped_pages (covers, credits, contents…), recovered_pages, renderer")
    learned: int = Field(0, description="Passages kept in SAQYA's learned knowledge (agricultural documents, "
                         "when learning is allowed)")


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class DocumentQuestion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["ما هي كمية السماد الموصى بها؟", "Explain this document"])
    history: list[str] = Field(default_factory=list, max_length=8,
                               description="Previous questions in this conversation, oldest first (for follow-ups)")


class ExplainRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    focus: str | None = Field(None, max_length=300,
                              description="A section heading (from `sections`) or a topic; omit for the whole document")
    language: Literal["ar", "en"] | None = Field(None, description="Answer language; default: the focus's or the document's")


# ---------------------------------------------------------------- learned knowledge
class LearnedSource(BaseModel):
    id: str
    kind: Literal["web", "document"] = "web"
    title: str = ""
    site: str = ""
    url: str = ""
    passages: int
    learned_at: str = ""
    question: str = Field("", description="The question that found it (web)")


class KnowledgeResponse(BaseModel):
    enabled: bool = Field(True, description="LEARNING is on")
    passages: int = 0
    web_sources: int = 0
    document_sources: int = 0
    ready: bool = True
    sources: list[LearnedSource] = Field(default_factory=list)


class ForgetResponse(BaseModel):
    removed: int = Field(..., description="Passages forgotten")
