"""SAQYA's learned knowledge: what the web and attached documents taught it, kept for later questions.

What is kept
- Web: only the passages an answer actually cited (after the answer passed the grounding checks), with their
  page, site, title, the date they were read and the question that found them. Never the generated answers
  themselves, so a mistake of the LLM can't become "knowledge".
- Attached documents: the passages of an agricultural document the farmer attached (the clean text, already
  embedded), when "Learn from attached documents" is on. The temporary clean PDF is still deleted with the
  session; only the passages stay.

Before a passage is kept: at least 12 words and mostly letters, not a copy of a passage already kept (same
text, or 96 % similar), and the store stays under LEARNED_MAX_PASSAGES (the least used, oldest go first).

How it is used: RAGService asks it after the library and before the web (beyond.py), with the same hybrid
search and embedding model; a question asked before is then answered at once, offline, citing the page or
document the passage came from. Settings in the chat app shows what was learned and can forget a source or
everything.

Stored in LEARNED_DIR (backend/data/learned/): knowledge.jsonl (one passage per line) and vectors.npy. When the
embedding model changes, the vectors are rebuilt in the background.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from app.services.retrieval import MemoryCollection, RetrievedChunk, Retriever
from app.utils.text import detect_language, normalize_for_match

logger = logging.getLogger(__name__)

MIN_WORDS = 12
MAX_CHARS = 3000
NEAR_DUPLICATE = 0.96


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text_key(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", normalize_for_match(text)).strip().encode()).hexdigest()


def source_id(kind: str, key: str) -> str:
    return hashlib.sha1(f"{kind}|{key}".encode()).hexdigest()[:12]


def good_passage(text: str) -> bool:
    """Long enough, not too long, mostly words (not a table of symbols or a menu)."""
    words = re.findall(r"\w{2,}", text)
    letters = sum(ch.isalpha() for ch in text)
    return len(words) >= MIN_WORDS and len(text) <= MAX_CHARS and letters >= 0.5 * len(text.replace(" ", ""))


class KnowledgeStore:
    """Learned passages on disk, searchable like the library."""

    def __init__(self, folder: Path | str, embedder, query_prefix: str = "", passage_prefix: str = "",
                 embedding_model: str = "", max_passages: int = 20000, min_similarity: float = 0.5) -> None:
        self.folder = Path(folder)
        self.embedder = embedder
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.embedding_model = embedding_model
        self.max_passages = max_passages
        self.min_similarity = min_similarity
        self._lock = threading.RLock()
        self._rows: list[dict] = []           # {"id", "text", "key", "metadata"}
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._retriever: Retriever | None = None
        self._ready = True                    # False while vectors are rebuilt for a new embedding model
        self._load()

    # ------------------------------------------------------------ disk
    @property
    def _jsonl(self) -> Path:
        return self.folder / "knowledge.jsonl"

    @property
    def _npy(self) -> Path:
        return self.folder / "vectors.npy"

    @property
    def _meta(self) -> Path:
        return self.folder / "meta.json"

    def _load(self) -> None:
        if not self._jsonl.exists():
            return
        try:
            rows = [json.loads(line) for line in self._jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            vectors = np.load(self._npy) if self._npy.exists() else np.zeros((0, 0), dtype=np.float32)
            meta = json.loads(self._meta.read_text(encoding="utf-8")) if self._meta.exists() else {}
        except Exception as exc:  # noqa: BLE001 - a damaged file must not stop the backend
            logger.error("Learned knowledge could not be read (%s); starting empty", exc)
            return
        self._rows = rows
        stale = (len(vectors) != len(rows) or (self.embedding_model and meta.get("embedding_model")
                                              and meta.get("embedding_model") != self.embedding_model))
        if stale and rows:
            logger.info("Learned knowledge: re-embedding %d passages for %s", len(rows), self.embedding_model)
            self._ready = False
            threading.Thread(target=self._reembed, name="learned-reembed", daemon=True).start()
        else:
            self._vectors = np.asarray(vectors, dtype=np.float32)
        logger.info("Learned knowledge loaded: %d passages from %d sources", len(rows), len(self.sources()))

    def _reembed(self) -> None:
        try:
            vectors = self._embed([r["text"] for r in self._rows])
            with self._lock:
                self._vectors = vectors
                self._retriever = None
                self._save()
        finally:
            self._ready = True

    def _save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self._jsonl.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self._rows), encoding="utf-8")
        os.replace(tmp, self._jsonl)
        tmp_npy = self.folder / "vectors.tmp.npy"
        np.save(tmp_npy, self._vectors)
        os.replace(tmp_npy, self._npy)
        self._meta.write_text(json.dumps({"embedding_model": self.embedding_model, "passages": len(self._rows),
                                          "updated_at": _now()}), encoding="utf-8")

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self.embedder.encode([self.passage_prefix + t for t in texts], batch_size=16, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    # ------------------------------------------------------------ adding
    def count(self) -> int:
        return len(self._rows)

    def add(self, items: Iterable[dict], vectors: np.ndarray | None = None) -> int:
        """Keep new passages. items: {"text", "metadata": {source, kind ("web"/"document"), title, url, ...}}.
        `vectors`: their embeddings when already computed (same model and passage prefix). Returns how many were kept."""
        items = list(items)
        if not items or not self._ready:
            return 0
        keep = [i for i, it in enumerate(items) if good_passage(it.get("text", ""))]
        with self._lock:
            known = {r["key"] for r in self._rows}
            fresh, keys = [], []
            for i in keep:
                key = text_key(items[i]["text"])
                if key not in known and key not in keys:
                    fresh.append(i)
                    keys.append(key)
            if not fresh:
                return 0
            vecs = (np.asarray(vectors, dtype=np.float32)[fresh] if vectors is not None
                    else self._embed([items[i]["text"] for i in fresh]))
            if self._vectors.size and vecs.size:   # near-duplicates of what is already known
                similar = (vecs @ self._vectors.T).max(axis=1) >= NEAR_DUPLICATE
            else:
                similar = np.zeros(len(fresh), dtype=bool)
            added_rows, added_vecs = [], []
            for j, i in enumerate(fresh):
                if similar[j] or any(float(vecs[j] @ v) >= NEAR_DUPLICATE for v in added_vecs):
                    continue
                meta = dict(items[i].get("metadata") or {})
                kind = meta.get("kind_of_source") or meta.get("kind") or "web"
                meta.update(kind="learned", learned_from=kind, learned_at=_now(), hits=0,
                            lang=meta.get("lang") or detect_language(items[i]["text"]))
                meta.setdefault("source_id", source_id(kind, meta.get("url") or meta.get("title") or meta.get("source", "")))
                added_rows.append({"id": f"learned-{keys[j]}", "text": items[i]["text"], "key": keys[j], "metadata": meta})
                added_vecs.append(vecs[j])
            if not added_rows:
                return 0
            self._rows += added_rows
            stacked = np.vstack(added_vecs).astype(np.float32)
            self._vectors = stacked if not self._vectors.size else np.vstack([self._vectors, stacked])
            self._prune()
            self._retriever = None
            self._save()
        logger.info("Learned %d new passages (now %d)", len(added_rows), len(self._rows))
        return len(added_rows)

    def add_web(self, chunks: Sequence[RetrievedChunk], question: str) -> int:
        """Keep the web passages an answer cited."""
        items = [{"text": c.text, "metadata": {"source": c.source, "url": c.url, "title": c.title or c.source,
                                               "kind": "web", "page_start": 0, "page_end": 0, "question": question[:300],
                                               "source_id": source_id("web", c.url)}}
                 for c in chunks if c.kind == "web" and c.url]
        return self.add(items)

    def add_document(self, doc) -> int:
        """Keep an attached document's passages (its vectors are reused: same model, same passage prefix)."""
        title = doc.title or doc.filename
        sid = source_id("document", f"{doc.filename}|{text_key(' '.join(c['text'] for c in doc.chunks[:3]))}")
        items = []
        for c in doc.chunks:
            m = c.get("metadata") or {}
            items.append({"text": c["text"], "metadata": {
                "source": doc.filename, "title": title, "kind": "document", "topic": m.get("topic") or doc.topic,
                "section": m.get("section", ""), "page_start": m.get("page_start", 0),
                "page_end": m.get("page_end", m.get("page_start", 0)), "source_id": sid}})
        vectors = getattr(doc, "embeddings", None)
        if vectors is not None and len(vectors) != len(items):
            vectors = None
        return self.add(items, vectors)

    def _prune(self) -> None:
        """Over the limit: drop the passages used least, oldest first."""
        extra = len(self._rows) - self.max_passages
        if extra <= 0:
            return
        order = sorted(range(len(self._rows)), key=lambda i: (self._rows[i]["metadata"].get("hits", 0),
                                                              self._rows[i]["metadata"].get("learned_at", "")))
        drop = set(order[:extra])
        self._rows = [r for i, r in enumerate(self._rows) if i not in drop]
        self._vectors = np.delete(self._vectors, sorted(drop), axis=0)

    # ------------------------------------------------------------ searching
    def _retriever_now(self) -> Retriever | None:
        with self._lock:
            if not self._rows or not self._ready or not self._vectors.size:
                return None
            if self._retriever is None:
                rows = MemoryCollection("learned", [r["id"] for r in self._rows], [r["text"] for r in self._rows],
                                        [r["metadata"] for r in self._rows], self._vectors)
                config = {"embedding_model": self.embedding_model, "query_prefix": self.query_prefix,
                          "min_similarity": self.min_similarity}
                self._retriever = Retriever(rows, self.embedder, config, hybrid=True)
            return self._retriever

    def search(self, queries: list[str], top_k: int = 6, min_similarity: float | None = None) -> list[RetrievedChunk]:
        """The learned passages for these queries, or [] when none is similar enough."""
        retriever = self._retriever_now()
        if retriever is None:
            return []
        gate = self.min_similarity if min_similarity is None else min_similarity
        result = retriever.search(queries, top_k)
        if result.best_similarity < gate:
            return []
        return result.chunks

    def used(self, chunks: Sequence[RetrievedChunk]) -> None:
        """Count a use of the cited passages (the most used are kept longest)."""
        ids = {c.chunk_id for c in chunks}
        if not ids:
            return
        with self._lock:
            for r in self._rows:
                if r["id"] in ids:
                    r["metadata"]["hits"] = int(r["metadata"].get("hits", 0)) + 1
            try:
                self._save()
            except OSError as exc:
                logger.warning("Could not save learned knowledge: %s", exc)

    # ------------------------------------------------------------ listing and forgetting
    def sources(self) -> list[dict]:
        """One entry per source (web page or document): what it is and how many passages came from it."""
        groups: dict[str, dict] = {}
        for r in self._rows:
            m = r["metadata"]
            sid = m.get("source_id") or source_id(m.get("learned_from", "web"), m.get("url") or m.get("source", ""))
            g = groups.setdefault(sid, {"id": sid, "kind": m.get("learned_from", "web"), "title": m.get("title") or m.get("source", ""),
                                        "site": m.get("source", ""), "url": m.get("url", ""), "passages": 0,
                                        "learned_at": m.get("learned_at", ""), "question": m.get("question", "")})
            g["passages"] += 1
            g["learned_at"] = max(g["learned_at"], m.get("learned_at", ""))
        return sorted(groups.values(), key=lambda g: g["learned_at"], reverse=True)

    def forget(self, sid: str) -> int:
        with self._lock:
            keep = [i for i, r in enumerate(self._rows) if r["metadata"].get("source_id") != sid]
            removed = len(self._rows) - len(keep)
            if removed:
                self._rows = [self._rows[i] for i in keep]
                self._vectors = self._vectors[keep] if len(keep) else np.zeros((0, 0), dtype=np.float32)
                self._retriever = None
                self._save()
        return removed

    def forget_all(self) -> int:
        with self._lock:
            removed = len(self._rows)
            self._rows, self._vectors, self._retriever = [], np.zeros((0, 0), dtype=np.float32), None
            self._save()
        return removed

    def summary(self) -> dict:
        sources = self.sources()
        return {"passages": len(self._rows), "sources": sources,
                "web_sources": sum(s["kind"] == "web" for s in sources),
                "document_sources": sum(s["kind"] == "document" for s in sources), "ready": self._ready}
