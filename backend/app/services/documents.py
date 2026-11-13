"""Documents a farmer attaches in the chat: cleaned, kept for the session, then asked about or explained.

1. Cleaning (cleaning.py, the no-AI engine of notebooks/clean_rag_pdfs_no_ai.ipynb). A PDF is rebuilt
   from its glyphs and its useless parts are removed: cover, credits and contents pages, ministry headers,
   running headers and footers, page numbers, prefaces, acknowledgments and references. What is left is
   organised into a title, numbered sections, paragraphs, lists and one line per table row. Two checks
   keep it safe on any document: a short page the engine dropped at the start or end of the file comes
   back when it reads like content, and when the cleaned text is far shorter than the file's own text the
   pipeline's plain reader (extraction.py) is used instead. Word, Markdown and text files are read by
   extraction.py (their headings become sections).
2. A clean PDF of that text (the notebook's layout) is written to the session's temporary folder; the
   farmer can download it. The passages are made from exactly that clean text, section by section, each
   citing the page of the original file it came from.
3. Searching: the same hybrid search (BM25 + embeddings), dialect rewrite, Arabic <-> English query
   translation and answer checks as the library, over this one document only — plus each top passage's
   continuation in the same section (an answer often runs on into the next passage), and follow-up
   questions searched together with the farmer's earlier question.

Documents are temporary: they live in memory and in a temporary folder (one sub-folder per chat session),
are deleted with the session (the chat app asks for it, and documents unused for DOCUMENT_TTL_MINUTES are
removed anyway), and the whole folder is emptied when the backend starts and stops.

Learning: when the farmer allows it (Settings → "Learn from attached documents", on by default) and the document
is about agriculture (its crop was found, or the LLM says so from its title and opening), its passages are also
kept in SAQYA's learned knowledge (learning.py) for later questions in any chat. The clean PDF and the session's
copy are still deleted as above; Settings can forget what was learned.

"Explain this document" (or "لخص الملف" / "what is this document about?") can't be answered by searching
for the question's words, so it gets a sample of the whole document: its start, then the opening passage of
each section (evenly spread when there are many), and a prompt asking for what the document is about plus
its main points, each with its [n].
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from app.services import grounding
from app.services.dialect import to_msa
from app.services.extraction import (ExtractedDocument, ExtractionError, SUPPORTED_SUFFIXES, chunk_document,
                                     read_document)
from app.services.generation import NO_CONTEXT_ANSWER, select_passages
from app.services.glossary import document_topic
from app.services.rag_service import RAGResult, RAGService
from app.services.retrieval import MemoryCollection, RetrievedChunk, Retriever  # noqa: F401 (re-exported)
from app.utils.text import detect_language, normalize_for_match

logger = logging.getLogger(__name__)

EXPLAIN_PASSAGES = 8      # passages sampled for "explain the whole document" (≈ 2,000 tokens)
NEIGHBOURS = 3            # top passages whose continuation (next passage, same section) is added
MIN_COVERAGE = 0.35       # cleaned words / words in the file below this: the cleaning lost too much

# "Explain / summarise the document" in Arabic (standard and Egyptian) and English, on normalised text
_OVERVIEW_RE = re.compile(
    r"(?<!\w)(?:اشرح|اشرحلي|اشرحي|لخص|لخصلي|ملخص|تلخيص|فكره|موضوع|بيتكلم عن|يتكلم عن|يتحدث عن|عن ماذا|عن ايه|"
    r"explain|summar\w*|overview|main points|key points|what is (?:this|the) (?:document|file|pdf|paper) about)(?!\w)"
)
_DOCUMENT_WORDS_RE = re.compile(
    r"(?<!\w)(?:[وفب]?ال)?(?:مستند|ملف|وثيقه|نشره|كتيب|دليل|بحث|ورقه|document|file|pdf|paper|bulletin|this)(?!\w)"
)
# Cover-page words: an original (not cleaned) bulletin often starts with the ministry, institute and authors
_FRONT_MATTER_RE = re.compile(
    r"جمهورية|وزارة|مركز البحوث|معهد بحوث|الإدارة المركزية|الادارة المركزية|إعداد|اعداد|تأليف|نشرة رقم|رقم الإيداع|"
    r"ministry|republic|institute|prepared by|authors?:|isbn|copyright|all rights reserved", re.IGNORECASE)
_SESSION_RE = re.compile(r"[A-Za-z0-9_-]{6,64}")


class DocumentNotFoundError(KeyError):
    pass


# ---------------------------------------------------------------- an in-memory "collection"
# ---------------------------------------------------------------- reading and cleaning one file
@dataclass
class Prepared:
    doc: ExtractedDocument
    outline: dict | None                # the notebook's outline shape, for the clean PDF (None: no PDF)
    method: str                         # "cleaned" (the cleaning engine) or "plain" (extraction.py)
    dropped_pages: list[int] = field(default_factory=list)
    recovered_pages: list[int] = field(default_factory=list)


def _words(text: str) -> int:
    return len(re.findall(r"\w{2,}", text))


def _file_words(path: Path, skip: set[int]) -> int:
    """Words on the file's pages (PyMuPDF's plain text; reversed Arabic still counts its words)."""
    import pymupdf

    with pymupdf.open(str(path)) as pdf:
        return sum(_words(pdf[i].get_text()) for i in range(pdf.page_count) if i + 1 not in skip)


def _recover_short_pages(path: Path, cleaned) -> list[int]:
    """The engine drops short pages at the start and end of a file as covers / back covers. A last page with a
    few paragraphs is not always a back cover: put such a page back when it reads like content."""
    from app.services import cleaning as c
    import pymupdf

    recovered = []
    with pymupdf.open(str(path)) as pdf:
        for pno in list(cleaned.dropped_pages):
            if not 1 <= pno <= pdf.page_count:
                continue
            page = pdf[pno - 1]
            height = page.rect.height
            try:
                blocks = c.page_lines(page, c.page_glyphs(page, {}), exclude_rects=[])
            except Exception:  # noqa: BLE001 - an odd page: leave it dropped
                continue
            lines = [ln for b in blocks for ln in b["lines"]]
            if not lines:
                continue
            noise = sum(1 for ln in lines if c.FRONT_KW.search(ln["text"]) or c.NAME_LINE.match(ln["text"])
                        or c.GLOBAL_DROP.search(ln["text"]) or c.TOC_LINE.search(ln["text"]))
            body = [ln for ln in lines if 0.055 * height < ln["bbox"][1] and ln["bbox"][3] < 0.945 * height
                    and not c.PAGE_NUM.match(ln["text"]) and ln["text"].strip()]   # not the header / footer zones
            paragraphs, current, prev = [], [], None
            for ln in body:                  # lines -> paragraphs: break after a full stop, a gap or a heading
                text = ln["text"].strip()
                heading = len(text) < 50 and not re.search(r"[.!?؟،,:;]$", text)
                gap = prev is not None and ln["bbox"][1] - prev["bbox"][3] > 1.2 * max(1.0, prev["bbox"][3] - prev["bbox"][1])
                if current and (heading or gap or re.search(r"[.!?؟:]$", current[-1])):
                    paragraphs.append(" ".join(current))
                    current = []
                current.append(text)
                if heading:
                    paragraphs.append(" ".join(current))
                    current = []
                prev = ln
            if current:
                paragraphs.append(" ".join(current))
            kept = []
            for text in paragraphs:          # stop at a references / acknowledgments heading
                if c.DROP_SECTION.match(text) or c.HARD_DROP.match(text):
                    break
                kept.append(text)
            body = " ".join(kept)
            if _words(body) < 12 or noise / len(lines) >= 0.25 or not re.search(r"[.!?؟]", body):
                continue
            at = next((i for i, item in enumerate(cleaned.outline) if item[3] > pno), len(cleaned.outline))
            cleaned.outline[at:at] = [("p", 0, text, pno) for text in kept]
            cleaned.dropped_pages.remove(pno)
            recovered.append(pno)
    return recovered


def prepare_document(path: Path, filename: str) -> Prepared:
    """Read and clean one uploaded file. Raises ExtractionError when it has no usable text."""
    try:
        from app.services import cleaning
    except ImportError as exc:   # PyMuPDF missing: the plain reader, and no clean PDF
        logger.warning("Document cleaning needs PyMuPDF (pip install pymupdf): %s", exc)
        return Prepared(read_document(path, filename), None, "plain")

    if path.suffix.lower() == ".pdf":
        try:
            cleaned = cleaning.clean_pdf(path, filename)
            recovered = _recover_short_pages(path, cleaned)
            words = _words(" ".join(t for _, _, t, _ in cleaned.outline))
            coverage = words / max(1, _file_words(path, set(cleaned.dropped_pages)))
        except Exception as exc:  # noqa: BLE001 - a file the engine can't handle: use the plain reader
            logger.warning("Cleaning failed for %s (%s); reading it as it is", filename, exc)
            cleaned, coverage, recovered = None, 0.0, []
        if cleaned is not None and cleaned.chars >= 200 and coverage >= MIN_COVERAGE:
            pages = cleaning.cleaned_pages(cleaned)
            text = "\n".join(p["text"] for p in pages)
            doc = ExtractedDocument(filename=filename, title=cleaned.title, pages=pages, raw_pages=cleaned.pages,
                                    structured=True, language=detect_language(text),
                                    skipped_pages=len(cleaned.dropped_pages), sections=cleaned.sections[:40])
            logger.info("Cleaned %s: %d pages, dropped %s, recovered %s, %d sections, coverage %.2f", filename,
                        cleaned.pages, cleaned.dropped_pages, recovered, len(cleaned.sections), coverage)
            return Prepared(doc, cleaned.as_outline(), "cleaned", list(cleaned.dropped_pages), recovered)
        if cleaned is not None:
            logger.info("Cleaning kept too little of %s (coverage %.2f); reading it as it is", filename, coverage)
    doc = read_document(path, filename)
    return Prepared(doc, cleaning.outline_from_pages(doc.title, doc.pages), "plain")


# ---------------------------------------------------------------- one uploaded document
@dataclass
class UploadedDocument:
    id: str
    session: str
    filename: str
    title: str
    language: str
    pages: int
    skipped_pages: int
    structured: bool
    sections: list[str]
    topic: str
    created_at: str
    chunks: list[dict] = field(repr=False)
    embeddings: np.ndarray = field(repr=False)
    cleaning: dict = field(default_factory=dict)
    learned: int = 0                    # passages kept in the learned knowledge
    clean_pdf: Path | None = None
    last_used: float = field(default_factory=time.time)
    rag: RAGService | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {"id": self.id, "filename": self.filename, "title": self.title, "language": self.language,
                "pages": self.pages, "skipped_pages": self.skipped_pages, "chunks": len(self.chunks),
                "structured": self.structured, "sections": self.sections, "topic": self.topic,
                "created_at": self.created_at, "clean_pdf": bool(self.clean_pdf and self.clean_pdf.exists()),
                "cleaning": self.cleaning, "learned": self.learned}

    def retrieved(self, rows: Sequence[int], score: float = 1.0) -> list[RetrievedChunk]:
        out = []
        for i in rows:
            c, m = self.chunks[i], self.chunks[i]["metadata"]
            out.append(RetrievedChunk(chunk_id=c["id"], text=c["text"], source=m["source"],
                                      page_start=int(m["page_start"]), page_end=int(m["page_end"]), score=score,
                                      topic=m.get("topic", ""), section=m.get("section", "")))
        return out

    def with_neighbours(self, chunks: list[RetrievedChunk], extra: int = NEIGHBOURS) -> list[RetrievedChunk]:
        """Each of the top `extra` passages followed by the next passage of its section, if not already there."""
        index = {c["id"]: i for i, c in enumerate(self.chunks)}
        have = {c.chunk_id for c in chunks}
        out = []
        for rank, chunk in enumerate(chunks):
            out.append(chunk)
            i = index.get(chunk.chunk_id)
            if rank >= extra or i is None or i + 1 >= len(self.chunks):
                continue
            nxt = self.chunks[i + 1]
            if nxt["id"] in have or nxt["metadata"].get("section", "") != self.chunks[i]["metadata"].get("section", ""):
                continue
            out += self.retrieved([i + 1], score=round(chunk.score * 0.98, 4))
            have.add(nxt["id"])
        return out


class DocumentRAG(RAGService):
    """The library's RAGService over one document, adding each top passage's continuation."""

    origin = "document"

    def __init__(self, doc: UploadedDocument, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.doc = doc

    def _retrieve(self, queries: list[str], **focus) -> list[RetrievedChunk]:
        return self.doc.with_neighbours(super()._retrieve(queries, **focus))


# ---------------------------------------------------------------- the service
class DocumentService:
    def __init__(self, embedder, generator, storage_dir: Path | None = None, query_prefix: str = "",
                 passage_prefix: str = "", top_k: int = 6, min_similarity: float = 0.3, translate_queries: bool = True,
                 self_check: bool = True, max_documents: int = 20, max_chunks: int = 2500, embedding_model: str = "",
                 ttl_minutes: float = 180, per_session: int = 5, make_pdf: bool = True, knowledge=None) -> None:
        self.embedder = embedder
        self.generator = generator
        self.knowledge = knowledge   # learning.KnowledgeStore (None: attached documents are never learned)
        # always a dedicated sub-folder: emptying it can never touch anything else in UPLOADS_DIR
        self.storage_dir = (Path(storage_dir) if storage_dir else Path(tempfile.gettempdir()) / "saqya-uploads") / "sessions"
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.translate_queries = translate_queries
        self.self_check = self_check
        self.max_documents = max_documents
        self.max_chunks = max_chunks
        self.embedding_model = embedding_model
        self.ttl_seconds = ttl_minutes * 60
        self.per_session = per_session
        self.make_pdf = make_pdf
        self._docs: dict[str, UploadedDocument] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ the temporary folder
    def reset_storage(self) -> None:
        """Empty the temporary folder (at startup and shutdown: uploads never outlive the backend)."""
        with self._lock:
            self._docs.clear()
        shutil.rmtree(self.storage_dir, ignore_errors=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def session_key(session: str | None) -> str:
        return session if session and _SESSION_RE.fullmatch(session) else "default"

    def _folder(self, session: str, doc_id: str) -> Path:
        return self.storage_dir / session / doc_id

    # ------------------------------------------------------------ adding
    def add(self, data: bytes, filename: str, session: str | None = None, learn: bool = False) -> UploadedDocument:
        """Clean, chunk and embed an uploaded file. Raises ExtractionError if it has no usable text.
        `learn`: also keep its passages in the learned knowledge, if it is about agriculture."""
        self.expire()
        session = self.session_key(session)
        filename = Path(filename or "document.pdf").name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ExtractionError(f"Unsupported file type '{suffix or '?'}'. Upload a PDF, Word (.docx), .txt or .md file.")
        started = time.perf_counter()
        doc_id = uuid.uuid4().hex[:12]
        folder = self._folder(session, doc_id)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:   # the original file is never kept
                path = Path(tmp) / f"upload{suffix}"
                path.write_bytes(data)
                prepared = prepare_document(path, filename)
            chunks = chunk_document(prepared.doc)
            if not chunks:
                raise ExtractionError("No readable passages were found in the file.")
            if len(chunks) > self.max_chunks:
                raise ExtractionError(f"The document is too long ({len(chunks)} passages; the limit is "
                                      f"{self.max_chunks}). Upload the part you need.")
            renderer = None
            if self.make_pdf and prepared.outline:
                from app.services.cleaning import render_pdf

                renderer = render_pdf(prepared.outline, folder / "clean.pdf")
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        extracted = prepared.doc
        term = document_topic([c["text"] for c in chunks], filename)
        topic = f"{term.ar} / {term.en}" if term else ""
        for c in chunks:
            c["metadata"]["topic"] = topic
        vectors = self._embed([c["text"] for c in chunks])
        doc = UploadedDocument(
            id=doc_id, session=session, filename=filename, title=extracted.title, language=extracted.language,
            pages=extracted.raw_pages, skipped_pages=extracted.skipped_pages, structured=extracted.structured,
            sections=extracted.sections, topic=topic, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            chunks=chunks, embeddings=vectors, clean_pdf=(folder / "clean.pdf") if renderer else None,
            cleaning={"method": prepared.method, "dropped_pages": prepared.dropped_pages,
                      "recovered_pages": prepared.recovered_pages, "renderer": renderer},
        )
        with self._lock:
            self._docs[doc.id] = doc
        self._prune(session)
        if learn and self.knowledge is not None:
            doc.learned = self._learn(doc)
        logger.info("Uploaded %s (session %s): %d pages, %d passages, %s, pdf=%s, topic=%s, %.1fs", filename,
                    session[:8], doc.pages, len(chunks), prepared.method, renderer, topic or "-",
                    time.perf_counter() - started)
        return doc

    def _learn(self, doc: UploadedDocument) -> int:
        """Keep an agricultural document's passages for later questions; how many were kept."""
        from app.services.scope import is_agricultural

        sample = f"{doc.title}\n" + " ".join(c["text"] for c in doc.chunks[:2])[:900]
        try:
            if not (doc.topic or is_agricultural(sample, self.generator)):
                logger.info("Not learned: %s doesn't look agricultural", doc.filename)
                return 0
            return self.knowledge.add_document(doc)
        except Exception as exc:  # noqa: BLE001 - learning must never break an upload
            logger.warning("Could not learn from %s: %s", doc.filename, exc)
            return 0

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.embedder.encode([self.passage_prefix + t for t in texts], batch_size=16, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    # ------------------------------------------------------------ removing
    def _drop(self, doc: UploadedDocument) -> None:
        with self._lock:
            self._docs.pop(doc.id, None)
        shutil.rmtree(self._folder(doc.session, doc.id), ignore_errors=True)
        session_dir = self.storage_dir / doc.session
        if session_dir.exists() and not any(session_dir.iterdir()):
            session_dir.rmdir()

    def _prune(self, session: str) -> None:
        """At most `per_session` documents per session and `max_documents` in all (the oldest go first)."""
        with self._lock:
            docs = sorted(self._docs.values(), key=lambda d: d.last_used)
        mine = [d for d in docs if d.session == session]
        for old in mine[:max(0, len(mine) - self.per_session)]:
            self._drop(old)
        with self._lock:
            docs = sorted(self._docs.values(), key=lambda d: d.last_used)
        for old in docs[:max(0, len(docs) - self.max_documents)]:
            self._drop(old)

    def expire(self) -> int:
        """Remove documents nobody has used for DOCUMENT_TTL_MINUTES (a chat closed without saying so)."""
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            old = [d for d in self._docs.values() if d.last_used < cutoff]
        for doc in old:
            self._drop(doc)
        return len(old)

    def delete(self, doc_id: str) -> None:
        self._drop(self.get(doc_id, touch=False))

    def delete_session(self, session: str | None) -> int:
        """Remove every document of a chat session (it was cleared or closed)."""
        session = self.session_key(session)
        with self._lock:
            docs = [d for d in self._docs.values() if d.session == session]
        for doc in docs:
            self._drop(doc)
        shutil.rmtree(self.storage_dir / session, ignore_errors=True)
        return len(docs)

    # ------------------------------------------------------------ access
    def get(self, doc_id: str, touch: bool = True) -> UploadedDocument:
        with self._lock:
            doc = self._docs.get(doc_id or "")
        if doc is None:
            raise DocumentNotFoundError(doc_id)
        if touch:
            doc.last_used = time.time()
        return doc

    def list(self, session: str | None = None) -> list[dict]:
        self.expire()
        session = self.session_key(session)
        with self._lock:
            docs = [d for d in self._docs.values() if d.session == session]
        return [d.summary() for d in sorted(docs, key=lambda d: d.created_at, reverse=True)]

    def clean_pdf(self, doc_id: str) -> Path:
        doc = self.get(doc_id)
        if not doc.clean_pdf or not doc.clean_pdf.exists():
            raise DocumentNotFoundError(doc_id)
        return doc.clean_pdf

    def rag(self, doc: UploadedDocument) -> RAGService:
        """The usual RAGService, searching only this document (hybrid search over its passages)."""
        if doc.rag is None:
            collection = MemoryCollection(doc.id, [c["id"] for c in doc.chunks], [c["text"] for c in doc.chunks],
                                          [c["metadata"] for c in doc.chunks], doc.embeddings)
            config = {"embedding_model": self.embedding_model, "query_prefix": self.query_prefix,
                      "min_similarity": self.min_similarity}
            retriever = Retriever(collection, self.embedder, config, hybrid=True)
            doc.rag = DocumentRAG(doc, retriever, self.generator, top_k=self.top_k, min_similarity=self.min_similarity,
                                  translate_queries=self.translate_queries, self_check=self.self_check)
        return doc.rag

    # ------------------------------------------------------------ answering
    @staticmethod
    def is_overview(question: str) -> bool:
        """'Explain / summarise this document' rather than a question about one thing in it."""
        q = normalize_for_match(to_msa(question))
        if not _OVERVIEW_RE.search(q):
            return False
        rest = _DOCUMENT_WORDS_RE.sub(" ", _OVERVIEW_RE.sub(" ", q))
        rest = re.sub(r"(?<!\w)(?:لي|ليا|لو سمحت|من فضلك|هذا|هذه|ده|دي|كله|كلها|بالتفصيل|ببساطه|please|me|the|it|in|simple|words|of)(?!\w)", " ", rest)
        return len(re.findall(r"\w{3,}", rest)) <= 1

    def answer(self, doc_id: str, question: str, history: list[str] | None = None) -> RAGResult:
        doc = self.get(doc_id)
        if self.is_overview(question):
            return self.explain(doc_id, language=detect_language(question))
        return self.rag(doc).answer(question, history=history)

    def explain(self, doc_id: str, focus: str | None = None, language: str | None = None) -> RAGResult:
        """Explain the whole document, or the section / topic named in `focus`."""
        doc = self.get(doc_id)
        language = language or (detect_language(focus) if focus and focus != "recommendations" else doc.language)
        if language not in ("ar", "en"):
            language = "ar"
        chunks = self._sample(doc, focus)
        if not chunks:
            return RAGResult(NO_CONTEXT_ANSWER[language], [], language, origin="none")
        request = doc.title if not focus or focus == "recommendations" else focus
        raw = self.generator.explain(chunks, language, doc.title, focus)
        answer = grounding.tidy(raw, chunks, request)
        if not grounding.is_refusal(answer):
            answer = grounding.drop_unsupported_sentences(answer, chunks, request)
        if grounding.is_refusal(answer) or not grounding.has_content(answer):
            return RAGResult(NO_CONTEXT_ANSWER[language], [], language, chunks, [request], raw, origin="none")
        answer = grounding.polish(self.rag(doc)._in_language(answer, chunks, request, language), language)
        if not grounding.has_content(answer):
            return RAGResult(NO_CONTEXT_ANSWER[language], [], language, chunks, [request], raw, origin="none")
        cited = select_passages(answer, chunks)
        written = detect_language(grounding.answer_body(answer))   # what the text is really in (voice, direction)
        return RAGResult(answer, [f"[{n}] {c.label}" for n, c in cited], written, chunks, [request], raw, cited=cited,
                         origin="document" if cited else "none")

    def _sample(self, doc: UploadedDocument, focus: str | None) -> list[RetrievedChunk]:
        """Passages to explain: one section's, the closest to a topic, the ones with advice ("recommendations"), or a
        spread over the whole document."""
        n = len(doc.chunks)
        if focus == "recommendations":
            rag = self.rag(doc)
            queries = ["التوصيات والإرشادات والنصائح العملية للمزارع: ماذا يفعل ومتى وكم",
                       "practical recommendations and advice for the farmer: what to do, when and how much"]
            found = rag.retriever.search(queries, EXPLAIN_PASSAGES).chunks
            opening = doc.retrieved([0]) if n else []
            return (opening + [c for c in found if c.chunk_id != doc.chunks[0]["id"]])[:EXPLAIN_PASSAGES] or opening
        if focus:
            in_section = [i for i, c in enumerate(doc.chunks) if c["metadata"].get("section") == focus]
            if in_section:
                return doc.retrieved(_spread(in_section, EXPLAIN_PASSAGES))
            rag = self.rag(doc)
            return rag.retriever.search(rag.search_queries(to_msa(focus), detect_language(focus)), EXPLAIN_PASSAGES).chunks
        first = 1 if n > 1 and len(_FRONT_MATTER_RE.findall(doc.chunks[0]["text"])) >= 2 else 0  # skip a cover page
        starts, seen = [], set()
        for i, c in enumerate(doc.chunks):  # the opening passage of every section
            section = c["metadata"].get("section") or ""
            if section not in seen:
                seen.add(section)
                starts.append(i)
        starts = [i for i in starts if i >= first]
        rows = [first] + [i for i in _spread(starts, EXPLAIN_PASSAGES - 1) if i != first] if len(starts) > 2 \
            else _spread(list(range(first, n)), EXPLAIN_PASSAGES)
        return doc.retrieved(sorted(dict.fromkeys(rows))[:EXPLAIN_PASSAGES])


def _spread(rows: list[int], k: int) -> list[int]:
    """k items evenly spread over rows (all of them when there are fewer)."""
    if len(rows) <= k:
        return list(rows)
    return [rows[round(i * (len(rows) - 1) / (k - 1))] for i in range(k)] if k > 1 else rows[:1]
