"""Compare chunking strategies on the same cleaned pages and the same evaluation questions.

Nothing here touches the vector store: every strategy is chunked in memory, indexed with the production BM25
(and, with --embedding-model, embedded and searched with the production hybrid search), and scored with the
same relevance rule as scripts/evaluate.py. A strategy worth adopting is then rebuilt into a *separate*
collection by rag_pipeline.ipynb / scripts/build_collection.py — never over the current one.

Strategies:
- production  — the app's chunker (extraction.chunk_document): section by section, 800 chars, 150 overlap,
                every chunk starts with its heading path. The baseline.
- sentences   — the same sentence packing without sections (what the store used before cleaned PDFs).
- fixed       — plain 800-character windows with 150 overlap (cuts through words and sentences).
- recursive   — split on paragraphs, then lines, then sentences, then words, merged up to 800 characters.
- parent-child— ~300-character children (sentence groups inside each production chunk, with its heading) are
                searched; the answer gets each child's *parent* production chunk. Scored on the parents.
- semantic    — consecutive sentences grouped until their embedding similarity drops (needs --embedding-model).

Larger chunks are more likely to contain an expected phrase, so the table also shows the characters the top 5
passages put in front of the LLM ("context chars@5"): compare quality at similar context cost.

    python scripts/compare_chunking.py                         # reads data/clean (slow: every PDF is parsed)
    python scripts/compare_chunking.py --pages-file pages.pkl  # cached pages
    python scripts/compare_chunking.py --embedding-model BAAI/bge-m3   # + hybrid search (embeds every strategy)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
PROJECT = BACKEND.parent

from app.evaluation import runner  # noqa: E402
from app.evaluation.dataset import DEFAULT_SEED, load_questions  # noqa: E402
from app.services import extraction  # noqa: E402
from app.services.retrieval import MemoryCollection, Retriever  # noqa: E402
from app.utils.text import detect_language  # noqa: E402

logger = logging.getLogger("chunking")
SIZE, OVERLAP, CHILD = extraction.CHUNK_SIZE, extraction.CHUNK_OVERLAP, 300


# ------------------------------------------------------------------ pages
def load_pages(args) -> dict[str, list[dict]]:
    """{source file: [{"page", "text"}]} — cleaned exactly as the pipeline cleans them."""
    if args.pages_file:
        data = pickle.load(open(args.pages_file, "rb"))
        rows = data["clean_pages"] if isinstance(data, dict) else data
        docs: dict[str, list[dict]] = {}
        for r in rows:
            docs.setdefault(r["source"], []).append({"page": r["page"], "text": r["text"], "lang": r.get("lang", "")})
        return docs
    docs = {}
    for path in sorted((PROJECT / "data" / "clean").glob("*.pdf")):
        try:
            docs[path.name] = extraction.read_document(path).pages
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipped %s: %s", path.name, exc)
    return docs


def topics_of(args) -> dict[str, str]:
    for path in (args.rag_config, BACKEND / "data" / "vector_store" / "rag_config.json"):
        if path and Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8")).get("document_topics", {})
    return {}


# ------------------------------------------------------------------ strategies
def _meta(source: str, pages: list[int], topics: dict, text: str, **extra) -> dict:
    return {"source": source, "page_start": min(pages), "page_end": max(pages), "topic": topics.get(source, ""),
            "lang": detect_language(text), **extra}


def production(docs, topics):
    out = []
    for source, pages in docs.items():
        doc = extraction.ExtractedDocument(filename=source, title="", pages=pages, raw_pages=len(pages),
                                           structured=extraction.is_structured_doc(pages),
                                           language=detect_language(" ".join(p["text"][:200] for p in pages[:5])))
        for c in extraction.chunk_document(doc, SIZE, OVERLAP):
            meta = dict(c["metadata"])
            meta["topic"] = topics.get(source, meta.get("topic", ""))
            out.append((c["text"], meta))
    return out


def sentences(docs, topics):
    out = []
    for source, pages in docs.items():
        for packed in extraction.pack(extraction.document_sentences(pages, SIZE), SIZE, OVERLAP):
            text = " ".join(s["text"] for s in packed)
            out.append((text, _meta(source, [s["page"] for s in packed], topics, text)))
    return out


def fixed(docs, topics):
    out = []
    for source, pages in docs.items():
        text = "\n".join(p["text"] for p in pages)
        starts = np.cumsum([0] + [len(p["text"]) + 1 for p in pages])
        for begin in range(0, max(len(text) - OVERLAP, 1), SIZE - OVERLAP):
            piece = text[begin:begin + SIZE]
            first = int(np.searchsorted(starts, begin, side="right")) - 1
            last = int(np.searchsorted(starts, begin + len(piece), side="right")) - 1
            out.append((piece, _meta(source, [pages[first]["page"], pages[min(last, len(pages) - 1)]["page"]], topics, piece)))
    return out


def _recursive_split(text: str, size: int, separators=("\n\n", "\n", ". ", "؟ ", " ")) -> list[str]:
    if len(text) <= size:
        return [text]
    for sep in separators:
        if sep in text:
            parts, out, current = text.split(sep), [], ""
            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= size:
                    current = candidate
                    continue
                if current:
                    out.append(current)
                current = part if len(part) <= size else ""
                if len(part) > size:
                    out += _recursive_split(part, size, separators[separators.index(sep) + 1:] or (" ",))
            if current:
                out.append(current)
            return out
    return [text[i:i + size] for i in range(0, len(text), size)]


def recursive(docs, topics):
    out = []
    for source, pages in docs.items():
        previous_tail = ""
        for p in pages:
            for piece in _recursive_split(p["text"], SIZE - OVERLAP):
                text = (previous_tail + " " + piece).strip() if previous_tail else piece
                out.append((text, _meta(source, [p["page"]], topics, text)))
                previous_tail = piece[-OVERLAP:]
    return out


def parent_child(parents):
    """Children (sentence groups of ~CHILD chars inside each parent, with the parent's heading line)."""
    children = []
    for pi, (text, meta) in enumerate(parents):
        head, _, body = text.partition("\n") if " › " in text.split("\n", 1)[0] else ("", "", text)
        units = extraction.document_sentences([{"page": meta.get("page_start", 0), "text": body}], CHILD)
        for packed in extraction.pack(units, CHILD, 0):
            piece = " ".join(s["text"] for s in packed)
            children.append(((head + "\n" + piece).strip(), {**meta, "parent": pi}))
    return children


def semantic(docs, topics, embedder, threshold_percentile: float = 20.0):
    """Sentences grouped while consecutive sentences stay similar (split at the lowest 20% similarities)."""
    out = []
    for source, pages in docs.items():
        sents = extraction.document_sentences(pages, SIZE)
        if not sents:
            continue
        vecs = np.asarray(embedder.encode([s["text"] for s in sents], batch_size=32, normalize_embeddings=True))
        sims = (vecs[:-1] * vecs[1:]).sum(axis=1) if len(sents) > 1 else np.array([])
        cut = np.percentile(sims, threshold_percentile) if len(sims) else 0.0
        group, length = [sents[0]], len(sents[0]["text"])
        for s, sim in zip(sents[1:], sims):
            if sim < cut or length + len(s["text"]) > SIZE:
                text = " ".join(x["text"] for x in group)
                out.append((text, _meta(source, [x["page"] for x in group], topics, text)))
                group, length = [], 0
            group.append(s)
            length += len(s["text"]) + 1
        text = " ".join(x["text"] for x in group)
        out.append((text, _meta(source, [x["page"] for x in group], topics, text)))
    return out


# ------------------------------------------------------------------ evaluation
def build_retriever(chunks, name, embedder=None):
    ids = [f"{name}:{i}" for i in range(len(chunks))]
    texts = [t for t, _ in chunks]
    metas = [m for _, m in chunks]
    if embedder is None:
        return ids, runner.keyword_retriever(ids, texts, metas)
    vecs = np.asarray(embedder.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    return ids, Retriever(MemoryCollection(name, ids, texts, metas, vecs), embedder, {}, hybrid=True)


def evaluate(name, chunks, questions, queries, embedder=None, parents=None) -> dict:
    started = time.perf_counter()
    ids, retriever = build_retriever(chunks, name, embedder)
    corpus = runner.Corpus(*(([f"parent:{i}" for i in range(len(parents))], [t for t, _ in parents], [m for _, m in parents])
                             if parents else (ids, [t for t, _ in chunks], [m for _, m in chunks])))

    def rank(qs: list[str]) -> list[str]:
        if embedder is not None:
            found = [c.chunk_id for c in retriever.search(qs, top_k=runner.DEPTH * (3 if parents else 1)).chunks]
        else:
            found = runner._bm25_ranking(retriever, qs, runner.DEPTH * (3 if parents else 1))
        if not parents:
            return found
        ranked = []   # children -> their parents, best child first, each parent once
        for cid in found:
            parent = f"parent:{chunks[int(cid.split(':')[1])][1]['parent']}"
            if parent not in ranked:
                ranked.append(parent)
        return ranked[: runner.DEPTH]

    run = runner.run_experiment(runner.Experiment(name, name, rank), questions, corpus, queries)
    out = {"strategy": name, "chunks": len(chunks), "mean_chars": round(float(np.mean([len(t) for t, _ in chunks])))}
    for split in ("validation", "test"):
        ids_ = {q.id for q in questions if q.split == split}
        out.update({f"{split} {k}": v for k, v in runner.summarize(run["rows"], ids_).items() if k != "questions"})
    # context cost: characters of the top-5 passages the LLM would read
    lens = {cid: len(t) for cid, t in zip(corpus.ids, corpus.texts)}
    cost = []
    for q in questions:
        cost.append(sum(lens.get(c, 0) for c in rank(queries(q.question))[:5]))
    out["context chars@5"] = round(float(np.mean(cost)))
    out["seconds"] = round(time.perf_counter() - started, 1)
    logger.info("%-13s %5d chunks  test recall@5 %.3f  MRR@10 %.3f  nDCG@10 %.3f", name, len(chunks),
                out.get("test recall@5", 0), out.get("test mrr@10", 0), out.get("test ndcg@10", 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages-file", type=Path)
    parser.add_argument("--rag-config", type=Path)
    parser.add_argument("--embedding-model", help="also embed every strategy and compare with the hybrid search")
    parser.add_argument("--strategies", nargs="*", default=["production", "sentences", "fixed", "recursive",
                                                            "parent-child", "semantic"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("app.services.spelling").setLevel(logging.WARNING)

    docs, topics = load_pages(args), topics_of(args)
    questions = [q for q in load_questions(PROJECT / "data", seed=args.seed) if q.answerable]
    queries = runner.Queries()
    embedder = None
    if args.embedding_model:
        from app.services.retrieval import load_embedder

        embedder = load_embedder(args.embedding_model)
    built = {"production": production(docs, topics)}
    makers = {"sentences": sentences, "fixed": fixed, "recursive": recursive}
    rows, notes = [], {}
    for name in args.strategies:
        if name in makers:
            built[name] = makers[name](docs, topics)
        if name == "semantic":
            if embedder is None:
                notes[name] = "NOT RUN: needs --embedding-model"
                continue
            built[name] = semantic(docs, topics, embedder)
        if name == "parent-child":
            children = parent_child(built["production"])
            rows.append(evaluate("children only", children, questions, queries, embedder))
            rows.append(evaluate("parent-child", children, questions, queries, embedder, parents=built["production"]))
            continue
        rows.append(evaluate(name, built[name], questions, queries, embedder))

    out = args.out or PROJECT / "evaluation" / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-chunking")
    out.mkdir(parents=True, exist_ok=True)
    result = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "documents": len(docs),
              "pages": sum(len(p) for p in docs.values()), "search": "hybrid" if embedder else "BM25 only",
              "embedding_model": args.embedding_model, "seed": args.seed, "size": SIZE, "overlap": OVERLAP,
              "child_size": CHILD, "results": rows, "not_run": notes}
    (out / "chunking.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["strategy", "chunks", "mean_chars", "context chars@5", "test hit@5", "test recall@5", "test recall@10",
            "test mrr@10", "test ndcg@10", "validation mrr@10"]
    lines = [f"# Chunking comparison ({result['search']})", "", f"{result['documents']} documents, {result['pages']} pages, "
             f"{sum(q.split == 'test' for q in questions)} test / {sum(q.split == 'validation' for q in questions)} validation questions", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    lines += ["", *(f"- {k}: {v}" for k, v in notes.items())]
    (out / "chunking.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results in %s", out)


if __name__ == "__main__":
    main()
