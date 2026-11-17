"""Retrieval experiments E1–E7 on the evaluation questions, with the production search code.

Every experiment ranks the same questions over the same passages and is scored with the same relevance rule
(dataset.py), so the numbers are comparable. Experiments that need something this run does not have (an
embedding model, the reranker, a fine-tuned model) are reported as NOT RUN with the reason, never estimated.

| id | what it measures |
|----|------------------|
| E1 | dense (embedding) search only — the app's search with HYBRID_SEARCH=false |
| E2 | BM25 keyword search only (same tokenizer, stemmer and glossary synonyms as the app) |
| E3 | hybrid: dense + BM25 fused with RRF — the app's current search (baseline) |
| E4 | E3 + cross-encoder reranking of the candidates |
| E5 | E3 + query expansion (query_expansion.py) |
| E6 | E3 + expansion + reranking |
| E7 | fine-tuned embeddings (separate collection) + hybrid + reranking |
| E2x| BM25 + query expansion (runs without any model: shows what the expansion adds to keyword search) |

Search texts are built exactly as the app builds them: spelling repair, then the Egyptian -> standard-Arabic
rewrite (unless `raw=True`), then, when a translator is given, the LLM translation into the other language
(cached in the output folder, so every experiment searches with identical text).
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from app.evaluation.dataset import EvalQuestion
from app.evaluation.metrics import ranking_summary
from app.services.dialect import to_msa
from app.services.retrieval import MemoryCollection, Retriever
from app.services.spelling import repair
from app.utils.text import detect_language

logger = logging.getLogger(__name__)

DEPTH = 10   # rankings are scored down to rank 10 (Recall@10, MRR@10, nDCG@10)


# ------------------------------------------------------------------ the passages
@dataclass
class Corpus:
    ids: list[str]
    texts: list[str]
    metas: list[dict]
    _relevant: dict[str, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = {cid: i for i, cid in enumerate(self.ids)}

    def relevant(self, q: EvalQuestion) -> set[int]:
        """Positions of every passage that answers the question (computed once per question)."""
        if q.id not in self._relevant:
            self._relevant[q.id] = {i for i, (t, m) in enumerate(zip(self.texts, self.metas))
                                    if q.is_relevant(t, str(m.get("source", "")))}
        return self._relevant[q.id]


def corpus_from_retriever(retriever: Retriever) -> Corpus:
    if not retriever._ids:   # built without BM25: read the collection once
        data = retriever.collection.get(include=["documents", "metadatas"])
        retriever._ids, retriever._texts = list(data["ids"]), list(data["documents"])
        retriever._metas = [m or {} for m in data["metadatas"]]
    return Corpus(list(retriever._ids), list(retriever._texts), [m or {} for m in retriever._metas])


def load_chunk_file(path: Path) -> tuple[list[str], list[str], list[dict]]:
    """Passages exported by `scripts/evaluate.py --export-chunks` (JSONL: id, text, metadata) or a pickle with a
    'chunks' list of {text, metadata[, id]}."""
    if path.suffix == ".pkl":
        data = pickle.load(open(path, "rb"))
        rows = data["chunks"] if isinstance(data, dict) else data
        ids = [str(r.get("id") or f"chunk-{i}") for i, r in enumerate(rows)]
        return ids, [r["text"] for r in rows], [dict(r.get("metadata") or {}) for r in rows]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [str(r["id"]) for r in rows], [r["text"] for r in rows], [dict(r.get("metadata") or {}) for r in rows]


def keyword_retriever(ids: list[str], texts: list[str], metas: list[dict], **kwargs) -> Retriever:
    """A Retriever with the production BM25 index and no embeddings (for keyword-only experiments)."""
    collection = MemoryCollection("offline", ids, texts, metas, np.zeros((len(ids), 1), dtype=np.float32))
    return Retriever(collection, embedder=None, config={}, hybrid=True, **kwargs)


# ------------------------------------------------------------------ search texts
class Queries:
    """The search texts for a question, as the app builds them (cached translations for reproducibility)."""

    def __init__(self, translate: Callable[[str, str], str | None] | None = None, cache_file: Path | None = None,
                 raw: bool = False) -> None:
        self.translate = translate
        self.raw = raw
        self.cache_file = cache_file
        self.cache: dict[str, str | None] = {}
        if cache_file and cache_file.exists():
            self.cache = json.loads(cache_file.read_text(encoding="utf-8"))

    def __call__(self, question: str) -> list[str]:
        text = repair(question)
        language = detect_language(text)
        if language == "ar" and not self.raw:
            text = to_msa(text)
        queries = [text]
        if self.translate is not None:
            if text not in self.cache:
                try:
                    self.cache[text] = self.translate(text, "en" if language == "ar" else "ar")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Translation failed for %r: %s", text[:60], exc)
                    self.cache[text] = None
            translated = self.cache[text]
            if translated and translated != text:
                queries.append(translated)
        return queries

    def save(self) -> None:
        if self.cache_file and self.translate is not None:
            self.cache_file.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ the experiments
def _bm25_ranking(retriever: Retriever, queries: Sequence[str], k: int, weights: Sequence[float] | None = None) -> list[str]:
    """BM25 only: each phrasing ranked by the app's keyword search, phrasings fused like the app ("max" RRF)."""
    weights = list(weights) if weights is not None else [1.0] * len(queries)
    main: dict[str, float] = {}
    extra: dict[str, float] = {}
    for query, weight in zip(queries, weights):
        target = main if weight >= 1.0 else extra
        for rank, (i, _) in enumerate(retriever.keyword_search(query, n=max(k, retriever.candidates))):
            cid = retriever._ids[i]
            target[cid] = max(target.get(cid, 0.0), 1.0 / (retriever.rrf_k + rank + 1))
    w = max((x for x in weights if x < 1.0), default=0.0)
    fused = dict(main)
    for cid, score in extra.items():   # same rule as Retriever.search: expansions add evidence
        fused[cid] = fused.get(cid, 0.0) + w * score
    return sorted(fused, key=fused.get, reverse=True)[:k]


def _with_depth(expanded: tuple[list[str], list[float]]) -> tuple[list[str], int, list[float]]:
    queries, weights = expanded
    return queries, DEPTH, weights


@dataclass
class Experiment:
    id: str
    name: str
    rank: Callable[[list[str]], list[str]] | None      # search texts -> ranked passage ids (None: cannot run)
    config: dict = field(default_factory=dict)
    not_run: str = ""                                    # why it could not run
    expand: bool = False


def build_experiments(retriever: Retriever | None, keyword: Retriever, reranker=None, finetuned: Retriever | None = None,
                      candidates: int = 30, expansion_weight: float = 0.25, max_variants: int = 2,
                      max_terms: int = 4) -> list[Experiment]:
    from app.services.query_expansion import expand

    dense_ok = retriever is not None and retriever.embedder is not None
    rerank_ok = reranker is not None and dense_ok

    def with_expansion(queries: list[str]) -> tuple[list[str], list[float]]:
        extra = expand(queries[0], max_variants=max_variants, max_terms=max_terms, other_language=len(queries) == 1)
        added = [v for v in extra.variants if v not in queries]
        return queries + added, [1.0] * len(queries) + [expansion_weight] * len(added)

    def hybrid(r: Retriever, queries, k, weights=None):
        return [c.chunk_id for c in r.search(queries, top_k=k, weights=weights).chunks]

    def reranked(r: Retriever, queries, weights=None):
        found = r.search(queries, top_k=candidates, weights=weights).chunks
        return [c.chunk_id for c in reranker.rerank(queries[: len(queries) if weights is None else weights.count(1.0)],
                                                     found, DEPTH)]

    no_model = "no embedding model in this run (needs the vector store and bge-m3: run on the PC)"
    no_reranker = ("reranker not available: " + (reranker.failed or "not loaded") if reranker is not None
                   else "reranker not requested (--reranker)") if dense_ok else no_model
    exps = [
        Experiment("E1", "dense only", (lambda q: [c.chunk_id for c in retriever.search(q, top_k=DEPTH, hybrid=False).chunks])
                   if dense_ok else None, not_run="" if dense_ok else no_model),
        Experiment("E1x", "dense only + query expansion",
                   (lambda q: (lambda t, w: [c.chunk_id for c in retriever.search(t, top_k=DEPTH, hybrid=False, weights=w).chunks])(
                       *with_expansion(q))) if dense_ok else None, not_run="" if dense_ok else no_model, expand=True),
        Experiment("E2", "BM25 only", lambda q: _bm25_ranking(keyword, q, DEPTH)),
        Experiment("E2x", "BM25 + query expansion", lambda q: _bm25_ranking(keyword, *_with_depth(with_expansion(q))),
                   expand=True),
        Experiment("E3", "hybrid (dense + BM25, RRF) — current app", (lambda q: hybrid(retriever, q, DEPTH))
                   if dense_ok else None, not_run="" if dense_ok else no_model),
        Experiment("E4", "hybrid + reranker", (lambda q: reranked(retriever, q)) if rerank_ok else None,
                   not_run="" if rerank_ok else no_reranker),
        Experiment("E5", "hybrid + query expansion", (lambda q: hybrid(retriever, *_with_depth(with_expansion(q))))
                   if dense_ok else None,
                   not_run="" if dense_ok else no_model, expand=True),
        Experiment("E6", "hybrid + expansion + reranker", (lambda q: reranked(retriever, *with_expansion(q)))
                   if rerank_ok else None, not_run="" if rerank_ok else no_reranker, expand=True),
    ]
    if finetuned is not None and rerank_ok:
        exps.append(Experiment("E7", "fine-tuned embeddings + hybrid + reranker", lambda q: reranked(finetuned, q),
                               config={"embedding_model": finetuned.embedding_model}))
    else:
        exps.append(Experiment("E7", "fine-tuned embeddings + hybrid + reranker", None,
                               not_run="no fine-tuned model/collection (scripts/finetune_embeddings.py, "
                                       "scripts/build_collection.py)" if finetuned is None else no_reranker))
    return exps


def run_experiment(exp: Experiment, questions: list[EvalQuestion], corpus: Corpus, queries: Queries) -> dict:
    """Ranks every question and returns {"rows": per-question results, "latency_ms": [...], "failures": n}."""
    rows, latencies, failures = [], [], 0
    for q in questions:
        relevant = corpus.relevant(q)
        texts = queries(q.question)
        started = time.perf_counter()
        try:
            ranked = exp.rank(texts)
        except Exception as exc:  # noqa: BLE001 - one failing question must not end the run
            logger.warning("%s failed on %s: %s", exp.id, q.id, exc)
            ranked, failures = [], failures + 1
        latencies.append((time.perf_counter() - started) * 1000)
        hits = [corpus.position.get(cid, -1) in relevant for cid in ranked[:DEPTH]]
        rows.append({"id": q.id, "split": q.split, "kind": q.kind, "category": q.category, "language": q.language,
                     "question": q.question, "search": " | ".join(texts), "relevant_total": len(relevant),
                     "hits": hits, "first_hit": next((r for r, h in enumerate(hits, 1) if h), None),
                     "top_sources": [corpus.metas[corpus.position[c]].get("source", "") for c in ranked[:3]
                                     if c in corpus.position]})
    return {"rows": rows, "latency_ms": latencies, "failures": failures}


def summarize(rows: list[dict], subset_ids: set[str] | None = None) -> dict:
    picked = [r for r in rows if (subset_ids is None or r["id"] in subset_ids) and r["relevant_total"] > 0]
    return ranking_summary([(r["hits"], r["relevant_total"]) for r in picked])


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values)
    return {"mean_ms": round(float(arr.mean()), 1), "p50_ms": round(float(np.percentile(arr, 50)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1)}
