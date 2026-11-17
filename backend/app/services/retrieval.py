"""Load the persisted Chroma vector store and retrieve relevant chunks.

Hybrid search: every phrasing of the question (the original and its translation) is searched twice,
by meaning (embedding similarity in Chroma) and by exact words (BM25, see lexical.py). The ranked
lists are merged with Reciprocal Rank Fusion: a chunk scores sum(1 / (60 + rank)) over the lists it
appears in, so a passage that both searches rank high comes first. RRF only uses ranks, so the two
very different score scales never have to be compared.

Whether the question is about the documents at all is decided separately (the relevance gate): it
passes when the best cosine similarity reaches the threshold calibrated in the notebook, or when it
names a crop that one of the documents is about (a sunflower question is in scope when there is a
sunflower bulletin, even if its wording scores low). An off-topic question is refused without
calling the LLM, whatever the keyword search found.

The vector store and rag_config.json are produced by notebooks/rag_pipeline.ipynb (section 2.7).
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import chromadb
import numpy as np

from app.core.tracing import tracer
from app.services.glossary import crops_in, remove_terms, synonyms
from app.services.lexical import BM25Index
from app.utils.text import detect_language

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "rag_config.json"
RRF_K = 60        # standard Reciprocal Rank Fusion constant
CANDIDATES = 30   # chunks taken from each ranked list before fusing


class Embedder(Protocol):
    def encode(self, inputs, **kwargs): ...


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page_start: int
    page_end: int
    score: float  # cosine similarity to the question, higher = more relevant
    topic: str = ""  # the crop the document is about, e.g. "دوار الشمس / sunflower" ("" if general)
    section: str = ""  # section heading in a cleaned document ("" if unknown)
    url: str = ""     # a web page (the web search and what was learned from it); source is then its site
    title: str = ""   # the web page's or learned document's title
    kind: str = ""    # "" (the library or an attached document), "web" or "learned"
    rerank_score: float | None = None   # the cross-encoder's score when reranking is on (reranker.py)

    @property
    def label(self) -> str:
        """Human-readable citation, e.g. 'maize_guide.pdf, p. 12' or 'Growing maize (fao.org)'."""
        if self.url:
            return f"{self.title} ({self.source})" if self.title and self.title != self.source else self.source
        if self.page_start == self.page_end:
            return f"{self.source}, p. {self.page_start}"
        return f"{self.source}, pp. {self.page_start}-{self.page_end}"


@dataclass
class SearchResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    best_similarity: float = 0.0  # best cosine over all phrasings
    named_crops: list[str] = field(default_factory=list)  # crops in the question that have a document

    def passes(self, min_similarity: float) -> bool:
        """The relevance gate: similar enough, or about a crop the documents cover."""
        return self.best_similarity >= min_similarity or bool(self.named_crops)


class VectorStoreNotFoundError(RuntimeError):
    pass


def load_rag_config(vector_store_dir: Path) -> dict:
    config_path = vector_store_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise VectorStoreNotFoundError(
            f"{config_path} not found. Run notebooks/rag_pipeline.ipynb first "
            "(it writes the vector store and rag_config.json into backend/data/vector_store)."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


class ModelNotFoundError(FileNotFoundError):
    """A local embedding model folder named in rag_config.json / EMBEDDING_MODEL is missing."""


def resolve_model(model_name: str, search_dirs: list[Path] | None = None) -> str:
    """Where to load an embedding model from.

    A Hugging Face id ("BAAI/bge-m3") is returned as it is. A local folder — like "models/bge-m3-egyptian",
    the Egyptian-Arabic fine-tune written by notebooks/egyptian_finetune.ipynb — is looked up relative to the
    project folder, then the backend folder (or `search_dirs`), and returned as a full path.
    """
    from app.core.config import BACKEND_DIR

    path = Path(model_name).expanduser()
    if path.is_absolute():
        if not path.is_dir():
            raise ModelNotFoundError(f"Embedding model folder not found: {path}")
        return str(path)
    for base in search_dirs or [BACKEND_DIR.parent, BACKEND_DIR]:
        if (base / path).is_dir():
            return str((base / path).resolve())
    if path.parts and path.parts[0] in ("models", ".", ".."):
        # looks like a local folder: don't let Hugging Face try to download "models/..." and fail obscurely
        raise ModelNotFoundError(
            f"Embedding model folder '{model_name}' not found next to the project or the backend. It is made by "
            "notebooks/egyptian_finetune.ipynb (models/ is not committed to git). Copy that folder here, or run the "
            "notebook — or rebuild the vector store with rag_pipeline.ipynb (USE_EGYPTIAN_MODEL = False) to use "
            "BAAI/bge-m3. In Docker, mount it: -v ./models:/models:ro")
    return model_name


def load_embedder(model_name: str) -> Embedder:
    # Imported lazily: torch is heavy and tests replace the embedder anyway.
    from sentence_transformers import SentenceTransformer

    location = resolve_model(model_name)
    logger.info("Loading embedding model %s%s", model_name, f" ({location})" if location != model_name else "")
    return SentenceTransformer(location)


def keyword_text(text: str, metadata: dict) -> str:
    """What BM25 indexes for a chunk: its text plus the document's topic (e.g. the crop)."""
    topic = metadata.get("topic") or ""
    return f"{topic}\n{text}" if topic else text


def _chunk(chunk_id: str, text: str, meta: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=str(meta.get("source", "unknown")),
        page_start=int(meta.get("page_start", 0)),
        page_end=int(meta.get("page_end", meta.get("page_start", 0))),
        score=round(float(score), 4),
        topic=str(meta.get("topic") or ""),
        section=str(meta.get("section") or ""),
        url=str(meta.get("url") or ""),
        title=str(meta.get("title") or ""),
        kind=str(meta.get("kind") or ""),
    )


class MemoryCollection:
    """The part of the Chroma collection API that Retriever uses, over vectors held in memory
    (an attached document, the learned knowledge, the pages of one web search)."""

    def __init__(self, name: str, ids: list[str], texts: list[str], metas: list[dict], embeddings: np.ndarray) -> None:
        self.name = name
        self._ids = list(ids)
        self._texts = list(texts)
        self._metas = [dict(m) for m in metas]
        self._emb = np.asarray(embeddings, dtype=np.float32).reshape(len(self._ids), -1)
        self._pos = {cid: i for i, cid in enumerate(self._ids)}

    def count(self) -> int:
        return len(self._ids)

    def get(self, ids: Sequence[str] | None = None, include: Sequence[str] = ("documents", "metadatas")) -> dict:
        rows = range(len(self._ids)) if ids is None else [self._pos[i] for i in ids if i in self._pos]
        out: dict = {"ids": [self._ids[i] for i in rows]}
        if "documents" in include:
            out["documents"] = [self._texts[i] for i in rows]
        if "metadatas" in include:
            out["metadatas"] = [self._metas[i] for i in rows]
        if "embeddings" in include:
            out["embeddings"] = [self._emb[i] for i in rows]
        return out

    def query(self, query_embeddings, n_results: int = 10, include: Sequence[str] = (), where: dict | None = None) -> dict:
        q = np.asarray(query_embeddings[0], dtype=np.float32)
        rows = np.arange(len(self._ids))
        if where:
            rows = np.array([i for i in rows if all(self._metas[i].get(k) == v for k, v in where.items())], dtype=int)
        if rows.size == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        sims = self._emb[rows] @ q
        order = rows[np.argsort(-sims)][:n_results]
        dist = [float(1.0 - self._emb[i] @ q) for i in order]  # cosine distance, like Chroma
        return {"ids": [[self._ids[i] for i in order]], "documents": [[self._texts[i] for i in order]],
                "metadatas": [[self._metas[i] for i in order]], "distances": [dist]}


class Retriever:
    """Hybrid (embedding + BM25) search over one Chroma collection."""

    keyword_seats = "original"   # whose best keyword match keeps a seat: "original" wording or "all" phrasings
    crop_seats_from = 1          # questions naming this many crops (with documents) get per-crop seats
    single_crop_seats = 3        # seats for the passages of the one crop a question names

    def __init__(self, collection, embedder: Embedder, config: dict, hybrid: bool = True,
                 rrf_k: int = RRF_K, candidates: int = CANDIDATES, bm25_k1: float = 1.5, bm25_b: float = 0.75,
                 keyword_weight: float = 1.0, keyword_seat: bool = True) -> None:
        self.collection = collection
        # RRF constant and how many chunks each ranked list contributes before fusing (RRF_K / RETRIEVAL_CANDIDATES
        # in .env; scripts/evaluate.py --tune compares values on the validation questions)
        self.rrf_k = max(1, int(rrf_k))
        self.candidates = max(1, int(candidates))
        self.bm25_k1, self.bm25_b = float(bm25_k1), float(bm25_b)   # BM25 term-frequency saturation and length normalisation
        # How much the keyword ranking counts in the fusion next to the embedding ranking (1 = equal, 0 = embeddings
        # only), and whether the best keyword match always keeps a place in the results (KEYWORD_WEIGHT / KEYWORD_SEAT;
        # scripts/evaluate.py --tune compares them on the validation questions)
        self.keyword_weight = max(0.0, float(keyword_weight))
        self.keyword_seat = bool(keyword_seat)
        self.embedder = embedder
        self.config = config
        self.embedding_model = config.get("embedding_model", "")
        # e5-style models expect "query: " / "passage: " prefixes; saved by the notebook
        self.query_prefix = config.get("query_prefix", "")
        self.default_min_similarity = float(config.get("min_similarity", 0.0))
        self.hybrid = hybrid
        self.bm25: BM25Index | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        if hybrid:
            self._build_keyword_index()
        else:
            self._metas = [m or {} for m in self.collection.get(include=["metadatas"])["metadatas"]]
        # crops that have their own document ("دوار الشمس / sunflower" -> "sunflower"), and their chunks
        self.topics: dict[str, str] = {}
        topic_rows: dict[str, list[int]] = defaultdict(list)
        topic_langs: dict[str, Counter] = defaultdict(Counter)
        for i, m in enumerate(self._metas):
            if m.get("topic"):
                self.topics[m["topic"].split(" / ")[-1]] = m["topic"]
                topic_rows[m["topic"]].append(i)
                topic_langs[m["topic"]][m.get("lang", "")] += 1
        self.topic_crops = set(self.topics)
        self._topic_rows = {t: np.array(rows) for t, rows in topic_rows.items()}
        # the language most of a topic's chunks are in (most crop bulletins are Arabic)
        self._topic_lang = {t: c.most_common(1)[0][0] for t, c in topic_langs.items()}

    @classmethod
    def from_store(
        cls,
        vector_store_dir: Path,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        embedder: Embedder | None = None,
        hybrid: bool = True,
        rrf_k: int = RRF_K,
        candidates: int = CANDIDATES,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        keyword_weight: float = 1.0,
        keyword_seat: bool = True,
    ) -> "Retriever":
        """Open the store the notebook wrote (what the API does at startup)."""
        config = load_rag_config(vector_store_dir)
        if embedding_model and embedding_model != config["embedding_model"]:
            logger.warning(
                "EMBEDDING_MODEL=%s differs from the model the store was built with (%s). "
                "Retrieval quality will suffer.", embedding_model, config["embedding_model"],
            )
        client = chromadb.PersistentClient(path=str(vector_store_dir))
        collection = client.get_collection(collection_name or config["collection_name"], embedding_function=None)
        retriever = cls(collection, embedder or load_embedder(embedding_model or config["embedding_model"]),
                        config, hybrid=hybrid, rrf_k=rrf_k, candidates=candidates,
                        bm25_k1=bm25_k1, bm25_b=bm25_b, keyword_weight=keyword_weight, keyword_seat=keyword_seat)
        logger.info("Vector store loaded: collection=%s chunks=%d model=%s hybrid=%s",
                    collection.name, retriever.count(), retriever.embedding_model, hybrid)
        return retriever

    def _build_keyword_index(self) -> None:
        data = self.collection.get(include=["documents", "metadatas"])
        self._ids = list(data["ids"])
        self._texts = list(data["documents"])
        self._metas = [m or {} for m in data["metadatas"]]
        self.bm25 = BM25Index([keyword_text(t, m) for t, m in zip(self._texts, self._metas)], k1=self.bm25_k1, b=self.bm25_b)

    def count(self) -> int:
        return self.collection.count()

    def embed_query(self, question: str) -> np.ndarray:
        return np.asarray(
            self.embedder.encode([self.query_prefix + question], normalize_embeddings=True)[0], dtype=np.float32
        )

    # ------------------------------------------------------------------ the two searches
    def dense_search(self, query_vec: np.ndarray, n: int | None = None, topic: str | None = None) -> list[RetrievedChunk]:
        """Chunks ranked by cosine similarity to one query embedding (only one topic's, if given)."""
        n = max(1, min(n or self.candidates, self.count()))
        res = self.collection.query(
            query_embeddings=[query_vec.tolist()], n_results=n,
            include=["documents", "metadatas", "distances"],
            **({"where": {"topic": topic}} if topic else {}),
        )
        return [
            _chunk(cid, text, meta or {}, 1.0 - float(dist))  # Chroma cosine distance = 1 - similarity
            for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def keyword_search(self, query: str, n: int | None = None, topic: str | None = None) -> list[tuple[int, float]]:
        """(position, BM25 score) of the best keyword matches; glossary synonyms are added."""
        if self.bm25 is None or not query.strip():
            return []
        n = n or self.candidates
        extra = synonyms(query, detect_language(query))
        allowed = self._topic_rows.get(topic, np.array([], dtype=int)) if topic else None
        return self.bm25.top(" ".join([query, *extra]), n, allowed)

    # ------------------------------------------------------------------ public API
    def retrieve(self, question: str, top_k: int = 4, min_similarity: float = 0.0) -> list[RetrievedChunk]:
        """Embedding search only, for one phrasing (used by the notebook to compare methods)."""
        return [c for c in self.dense_search(self.embed_query(question), top_k) if c.score >= min_similarity]

    def search(self, queries: list[str], top_k: int = 4, hybrid: bool | None = None,
               fusion: str = "max", balance: bool = True, topic: str | None = None,
               weights: Sequence[float] | None = None) -> SearchResult:
        """Fuse the searches for all phrasings and return the top_k chunks (no relevance gate).

        For one phrasing, the embedding and keyword rankings are complementary evidence, so their
        RRF scores add up. The phrasings (original, translation) are alternatives, so a chunk keeps
        its best phrasing's score ("max"), as the old embedding-only merge kept the best similarity.

        RRF rewards chunks that both searches rank fairly high, so a passage that only the keyword
        search finds (a rare pest or variety name the embedding model doesn't know) can lose to
        vaguer passages. The best keyword match for the farmer's own wording therefore always keeps
        a seat in the top_k (inside one topic's documents: the best match for the phrasing in those
        documents' language).

        A question comparing crops ("watermelon or tomato?") is also searched once per crop, inside
        that crop's documents, and each crop gets seats for its best passages: one ranking for the whole
        question tends to fill all seats with one crop, or with general text matching the shared words
        ("hybrid seed"). `topic` limits a search to one topic's documents.

        `weights` (one per phrasing, default all 1) scale a phrasing's fused score: query expansions get less
        than 1, so they can add a passage but never outrank what the farmer's own wording found equally well.
        """
        weights = list(weights) if weights is not None else [1.0] * len(queries)
        if len(weights) != len(queries):
            raise ValueError("weights must have one value per query")
        hybrid = self.hybrid if hybrid is None else hybrid
        if self.keyword_weight <= 0:
            hybrid = False   # KEYWORD_WEIGHT=0: the embedding ranking alone
        if not queries or not any(q.strip() for q in queries):
            return SearchResult()
        fused: dict[str, float] = defaultdict(float)
        extra: dict[str, float] = defaultdict(float)   # scores from query expansions (weight < 1)
        found: dict[str, RetrievedChunk] = {}
        vecs = [self.embed_query(q) for q in queries]
        keyword_best: list[str] = []
        gate_scores: list[float] = []
        seat_from = 0  # whose best keyword match gets the seat: the farmer's own wording...
        if topic:      # ...or, inside one topic's documents, the phrasing in their language
            seat_from = next((i for i, q in enumerate(queries) if detect_language(q) == self._topic_lang.get(topic)), 0)

        dense_ms = keyword_ms = 0.0
        dense_top: list[list[str]] = []
        keyword_top: list[list[str]] = []
        for qi, (query, vec) in enumerate(zip(queries, vecs)):
            per_query: dict[str, float] = defaultdict(float)
            started = time.perf_counter()
            dense = self.dense_search(vec, topic=topic)
            dense_ms += (time.perf_counter() - started) * 1000
            dense_top.append([c.chunk_id for c in dense[:5]])
            for rank, chunk in enumerate(dense):
                per_query[chunk.chunk_id] += 1.0 / (self.rrf_k + rank + 1)
                if weights[qi] >= 1.0 and chunk.score == chunk.score:
                    gate_scores.append(chunk.score)
                if chunk.chunk_id not in found or chunk.score > found[chunk.chunk_id].score:
                    found[chunk.chunk_id] = chunk
            if hybrid:
                started = time.perf_counter()
                keyword = self.keyword_search(query, topic=topic)
                keyword_ms += (time.perf_counter() - started) * 1000
                keyword_top.append([self._ids[i] for i, _ in keyword[:5]])
                for rank, (i, _) in enumerate(keyword):
                    cid = self._ids[i]
                    if self.keyword_seat and rank == 0 and (qi == seat_from or (self.keyword_seats == "all" and topic is None)):
                        keyword_best.append(cid)
                    per_query[cid] += self.keyword_weight / (self.rrf_k + rank + 1)
                    found.setdefault(cid, _chunk(cid, self._texts[i], self._metas[i], float("nan")))
            target = fused if weights[qi] >= 1.0 else extra
            for cid, score in per_query.items():
                target[cid] = max(target[cid], score) if fusion == "max" else target[cid] + score
        # Query expansions (weight < 1) add evidence instead of competing: a passage's score is its best score for the
        # farmer's own wording plus `weight` x its best score for an expansion. A passage both find rises; one only an
        # expansion finds can enter, but needs a strong match; with no expansion nothing changes.
        if extra:
            w = max(w for w in weights if w < 1.0)
            for cid, score in extra.items():
                fused[cid] += w * score

        # the relevance gate judges the farmer's own wording (and its translation): a query expansion — a phrasing
        # with a weight below 1 — may add passages but must not open the gate
        best_similarity = max(gate_scores, default=0.0)
        named = []
        for query, weight in zip(queries, weights):
            if weight < 1.0:
                continue
            named += [t.en for t in crops_in(query, detect_language(query)) if t.en in self.topic_crops and t.en not in named]
        if hybrid:
            ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
            for cid in dict.fromkeys(keyword_best):
                if cid not in ranked and top_k > len(keyword_best):
                    keep = [c for c in ranked if c in keyword_best] + [c for c in ranked if c not in keyword_best]
                    ranked = sorted(keep[:top_k - 1] + [cid], key=fused.get, reverse=True)
        else:
            ranked = sorted(found, key=lambda cid: found[cid].score, reverse=True)[:top_k]
        chunks = [found[cid] for cid in ranked]
        if balance and topic is None and len(named) >= self.crop_seats_from:
            chunks = self._balanced(queries, top_k, hybrid, named, chunks)
        self._fill_missing_scores(chunks, vecs)
        span = tracer.current()
        if span is not None and topic is None:   # what each search found, for the trace of the retrieval stage
            span.set(dense_ms=round(dense_ms, 1), keyword_ms=round(keyword_ms, 1), dense_top5=dense_top,
                     keyword_top5=keyword_top, fused_top=[c.chunk_id for c in chunks], rrf_k=self.rrf_k,
                     keyword_weight=self.keyword_weight, keyword_seat=self.keyword_seat,
                     candidates=self.candidates, named_crops=named)
        return SearchResult(chunks=chunks, best_similarity=round(best_similarity, 4), named_crops=named)

    def _balanced(self, queries: list[str], top_k: int, hybrid: bool, named: list[str],
                  general: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Seats for each compared crop's best passages, then the rest of the general ranking."""
        if len(named) == 1:
            per_crop = min(top_k, self.single_crop_seats)
        else:  # compared crops share all seats but one, which the general ranking keeps
            per_crop = max(1, (top_k - 1) // len(named))
        seats: list[RetrievedChunk] = []
        for crop in named[:top_k]:
            focused = []
            for q in queries:
                lang = detect_language(q)
                others = [t for t in crops_in(q, lang) if t.en != crop]
                focused.append(remove_terms(q, others, lang))
            own = self.search(focused, top_k=per_crop, hybrid=hybrid, topic=self.topics[crop]).chunks
            taken = {c.chunk_id for c in seats}
            seats += [c for c in own if c.chunk_id not in taken][:per_crop]
        taken = {c.chunk_id for c in seats}
        return (seats + [c for c in general if c.chunk_id not in taken])[:top_k]

    def topic_label(self, key: str | None) -> str | None:
        """'sunflower' (or the full label, or the Arabic name) -> 'دوار الشمس / sunflower'; None if unknown."""
        if not key:
            return None
        key = key.strip().lower()
        for crop, label in self.topics.items():
            if key in (crop.lower(), label.lower(), label.split(" / ")[0]):
                return label
        return None

    def retrieve_many(self, queries: list[str], top_k: int = 4, min_similarity: float = 0.0,
                      topic: str | None = None, weights: Sequence[float] | None = None) -> list[RetrievedChunk]:
        """Top_k chunks for the question, or [] when nothing reaches min_similarity (off-topic).

        `topic` (a crop the farmer chose in the UI) limits the search to that crop's documents; choosing a
        crop counts like naming it, so the relevance gate lets the question through.
        """
        label = self.topic_label(topic)
        if label:
            result = self.search(queries, top_k, topic=label, weights=weights)
            result.named_crops = result.named_crops or [label.split(" / ")[-1]]
        else:
            result = self.search(queries, top_k, weights=weights)
        return result.chunks if result.passes(min_similarity) else []

    def library(self) -> dict:
        """What is indexed: documents (title, crop, language, chunks, pages) and crops, for GET /library."""
        docs: dict[str, dict] = {}
        for m in self._metas:
            source = str(m.get("source", "unknown"))
            d = docs.setdefault(source, {"source": source, "title": "", "topic": "", "langs": Counter(),
                                         "chunks": 0, "pages": 0})
            d["chunks"] += 1
            d["title"] = d["title"] or str(m.get("title") or "")
            d["topic"] = d["topic"] or str(m.get("topic") or "")
            d["langs"][m.get("lang", "")] += 1
            d["pages"] = max(d["pages"], int(m.get("page_end", m.get("page_start", 0)) or 0))
        titles = self.config.get("document_titles", {})
        documents = []
        for d in sorted(docs.values(), key=lambda d: (d["topic"] == "", d["topic"], d["source"])):
            documents.append({
                "source": d["source"], "title": d["title"] or titles.get(d["source"], ""), "topic": d["topic"],
                "language": d["langs"].most_common(1)[0][0] if d["langs"] else "",
                "chunks": d["chunks"], "pages": d["pages"],
            })
        crops: dict[str, dict] = {}
        for d in documents:
            if d["topic"]:
                ar, _, en = d["topic"].partition(" / ")
                c = crops.setdefault(en, {"key": en, "ar": ar, "en": en, "documents": 0, "chunks": 0})
                c["documents"] += 1
                c["chunks"] += d["chunks"]
        languages = Counter(str(m.get("lang", "")) for m in self._metas)
        return {
            "num_documents": len(documents),
            "num_chunks": len(self._metas),
            "languages": {k: v for k, v in languages.most_common() if k},
            "crops": sorted(crops.values(), key=lambda c: -c["chunks"]),
            "documents": documents,
            "embedding_model": self.embedding_model,
            "built_at": str(self.config.get("built_at", "")),
        }

    def _fill_missing_scores(self, chunks: list[RetrievedChunk], vecs: list[np.ndarray]) -> None:
        """Chunks found only by keywords get their cosine similarity from the stored embeddings."""
        missing = [c for c in chunks if c.score != c.score]  # NaN
        if not missing:
            return
        got = self.collection.get(ids=[c.chunk_id for c in missing], include=["embeddings"])
        emb = {cid: np.asarray(e, dtype=np.float32) for cid, e in zip(got["ids"], got["embeddings"])}
        for c in missing:
            e = emb.get(c.chunk_id)
            c.score = round(max(float(e @ v) for v in vecs), 4) if e is not None else 0.0
