"""SAQYA evaluation: retrieval experiments E1–E7, parameter tuning, confidence calibration, end-to-end answers.

Run from the backend folder (the same place you start the API):

    python scripts/evaluate.py                          # E1–E6 on the real vector store (E7 when a fine-tuned store exists)
    python scripts/evaluate.py --reranker               # + E4/E6 (needs the reranker on disk: scripts/download_models.py)
    python scripts/evaluate.py --translate              # + the LLM question translation, as the app does (needs Ollama)
    python scripts/evaluate.py --tune                   # pick RRF / candidates / BM25 / expansion settings on VALIDATION
    python scripts/evaluate.py --calibrate              # fit the confidence model on validation, report it on test
    python scripts/evaluate.py --answers                # full answers with the LLM on the test questions (slow)
    python scripts/evaluate.py --chunks-file chunks.jsonl   # keyword-only experiments without any model
    python scripts/evaluate.py --export-chunks chunks.jsonl # write the store's passages for the line above

Questions: data/eval_questions.json + data/eval_questions_egyptian.json, split once into validation and test by
question group (an Egyptian question and its standard-Arabic twin share a group). Settings are chosen on
validation; test is reported and never used to choose anything. Results go to evaluation/results/<run>/:
results.json (everything, with models, versions, settings, dataset hash, seed, date, runtime), metrics.csv,
per_question.csv and report.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.evaluation import runner  # noqa: E402
from app.evaluation.dataset import DEFAULT_SEED, load_questions, subsets  # noqa: E402
from app.evaluation.metrics import auroc, brier_score, expected_calibration_error, risk_coverage  # noqa: E402

PROJECT = BACKEND.parent
logger = logging.getLogger("evaluate")


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("sentence_transformers", "torch", "chromadb", "numpy", "transformers"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def dataset_hash(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("eval_questions.json", "eval_questions_egyptian.json"):
        path = data_dir / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def peak_memory_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:  # noqa: BLE001
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:  # noqa: BLE001 - Windows without psutil
            return None


# ------------------------------------------------------------------ loading
def open_store(args, settings, store_dir: Path | None = None, model: str | None = None):
    from app.services.retrieval import Retriever

    started = time.perf_counter()
    retriever = Retriever.from_store(store_dir or settings.vector_store_path, None if store_dir else settings.collection_name,
                                     model or (None if store_dir else settings.embedding_model), hybrid=True,
                                     rrf_k=args.rrf_k or settings.rrf_k,
                                     candidates=args.retrieval_candidates or settings.retrieval_candidates,
                                     bm25_k1=settings.bm25_k1, bm25_b=settings.bm25_b,
                                     keyword_weight=settings.keyword_weight if args.keyword_weight is None else args.keyword_weight,
                                     keyword_seat=settings.keyword_seat if args.keyword_seat is None else args.keyword_seat == "on")
    return retriever, round(time.perf_counter() - started, 2)


def write_outputs(out: Path, results: dict, tables: list[dict], per_question: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if tables:
        with open(out / "metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for t in tables for k in t)))
            writer.writeheader()
            writer.writerows(tables)
    if per_question:
        with open(out / "per_question.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_question[0]))
            writer.writeheader()
            writer.writerows(per_question)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------ retrieval experiments
def run_retrieval(args, questions, corpus, experiments, queries) -> tuple[dict, list[dict], list[dict]]:
    by_split = {"validation": {q.id for q in questions if q.split == "validation"},
                "test": {q.id for q in questions if q.split == "test"}}
    named = subsets(questions)
    results, tables, per_question = {}, [], []
    for exp in experiments:
        if args.experiments and exp.id not in args.experiments:
            continue
        if exp.rank is None:
            results[exp.id] = {"name": exp.name, "status": "NOT RUN", "reason": exp.not_run}
            logger.info("%s %s: NOT RUN (%s)", exp.id, exp.name, exp.not_run)
            continue
        started = time.perf_counter()
        run = runner.run_experiment(exp, [q for q in questions if q.answerable], corpus, queries)
        seconds = round(time.perf_counter() - started, 1)
        summary = {}
        for split, ids in by_split.items():
            for subset, qs in named.items():
                metrics = runner.summarize(run["rows"], ids & {q.id for q in qs})
                if metrics:
                    summary[f"{split}/{subset}"] = metrics
                    tables.append({"experiment": exp.id, "name": exp.name, "split": split, "subset": subset, **metrics})
        results[exp.id] = {"name": exp.name, "status": "OK", "config": exp.config, "runtime_s": seconds,
                           "latency": runner.latency_summary(run["latency_ms"]), "failures": run["failures"],
                           "metrics": summary}
        for r in run["rows"]:
            per_question.append({"experiment": exp.id, **{k: v for k, v in r.items() if k != "hits"},
                                 "hit@5": int(any(r["hits"][:5])), "hits": "".join("1" if h else "0" for h in r["hits"])})
        test = summary.get("test/all", {})
        logger.info("%s %-42s test: recall@5 %.3f  recall@10 %.3f  MRR@10 %.3f  nDCG@10 %.3f  (%.1fs)", exp.id, exp.name,
                    test.get("recall@5", 0), test.get("recall@10", 0), test.get("mrr@10", 0), test.get("ndcg@10", 0), seconds)
    return results, tables, per_question


# ------------------------------------------------------------------ tuning (validation only)
def tune(args, questions, corpus, retriever, keyword, queries) -> dict:
    val = [q for q in questions if q.split == "validation" and q.answerable]
    out: dict = {"split": "validation", "questions": len(val), "grids": {}}

    def score(rank) -> dict:
        exp = runner.Experiment("tune", "tune", rank)
        return runner.summarize(runner.run_experiment(exp, val, corpus, queries)["rows"])

    rows = []
    original = (keyword.bm25.k1, keyword.bm25.b, keyword.rrf_k)
    for k1 in (0.9, 1.2, 1.5, 2.0):
        for b in (0.4, 0.6, 0.75, 0.9):
            keyword.bm25.k1, keyword.bm25.b = k1, b
            rows.append({"k1": k1, "b": b, **score(lambda q: runner._bm25_ranking(keyword, q, runner.DEPTH))})
    keyword.bm25.k1, keyword.bm25.b, keyword.rrf_k = original
    out["grids"]["bm25"] = rows
    if retriever is not None and retriever.embedder is not None:
        rows = []
        saved = (retriever.rrf_k, retriever.candidates)
        for k in (10, 30, 60, 100):
            for n in (20, 30, 50):
                retriever.rrf_k, retriever.candidates = k, n
                rows.append({"rrf_k": k, "candidates": n,
                             **score(lambda q: [c.chunk_id for c in retriever.search(q, top_k=runner.DEPTH).chunks])})
        retriever.rrf_k, retriever.candidates = saved
        out["grids"]["rrf"] = rows
        # how much the keyword ranking should count, whether its best match keeps a place, with and without expansion
        from app.services.query_expansion import expand

        def expanded(q):
            extra = [v for v in expand(q[0], other_language=len(q) == 1).variants if v not in q]
            return q + extra, [1.0] * len(q) + [args.expansion_weight] * len(extra)

        rows = []
        saved = (retriever.keyword_weight, retriever.keyword_seat)
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            for seat in ((False,) if weight == 0 else (True, False)):
                for with_expansion in (False, True):
                    retriever.keyword_weight, retriever.keyword_seat = weight, seat

                    def rank(q, with_expansion=with_expansion):
                        texts, w = expanded(q) if with_expansion else (q, None)
                        return [c.chunk_id for c in retriever.search(texts, top_k=runner.DEPTH, weights=w).chunks]
                    rows.append({"keyword_weight": weight, "keyword_seat": seat, "expansion": with_expansion, **score(rank)})
        retriever.keyword_weight, retriever.keyword_seat = saved
        out["grids"]["fusion"] = rows
    for name, grid in out["grids"].items():
        best = max(grid, key=lambda r: (r.get("mrr@10", 0), r.get("recall@10", 0)))
        out[f"best_{name}"] = best
        logger.info("Tuning %s on validation: best %s", name, {k: v for k, v in best.items() if k in ("k1", "b", "rrf_k", "candidates", "keyword_weight", "keyword_seat",
                                                                                  "expansion", "mrr@10", "recall@5", "recall@10")})
    return out


# ------------------------------------------------------------------ confidence calibration
def calibrate(args, settings, questions, corpus, retriever, queries, out_dir: Path) -> dict:
    from app.services.confidence import Calibration, choose_thresholds, fit_calibration, retrieval_signals
    from app.services.query_understanding import understand

    if retriever is None or retriever.embedder is None:
        return {"status": "NOT RUN", "reason": "needs the embedding model and vector store"}
    gate = retriever.default_min_similarity if settings.min_similarity is None else settings.min_similarity
    # the passages the app will really answer from: the same expansion and reranker settings as backend/.env
    reranker = None
    if settings.reranker_enabled:
        from app.services.reranker import Reranker

        reranker = Reranker(settings.reranker_model, settings.reranker_device, settings.reranker_batch_size,
                            settings.reranker_max_length, settings.reranker_allow_download, settings.reranker_trust_remote_code)
        if reranker.load() is None:
            reranker = None
    expand = None
    if settings.query_expansion_enabled:
        from app.services.query_expansion import expand
    rows = []
    for q in questions:
        texts = queries(q.question)
        search, weights = texts, None
        if expand is not None:
            extra = [v for v in expand(texts[0], max_variants=settings.query_expansion_max_variants,
                                       max_terms=settings.query_expansion_max_terms,
                                       other_language=len(texts) == 1).variants if v not in texts]
            if extra:
                search, weights = texts + extra, [1.0] * len(texts) + [settings.query_expansion_weight] * len(extra)
        result = retriever.search(search, top_k=settings.reranker_candidates if reranker else settings.top_k, weights=weights)
        chunks = reranker.rerank(texts, result.chunks, settings.top_k) if reranker else result.chunks
        chunks = chunks[: settings.top_k]
        relevant = corpus.relevant(q)
        outcome = int(q.answerable and any(corpus.position.get(c.chunk_id, -1) in relevant for c in chunks))
        signals = retrieval_signals(texts, chunks, gate, understand(q.question).missing)
        rows.append({"q": q, "signals": signals, "outcome": outcome, "baseline": int(result.passes(gate))})
    val = [r for r in rows if r["q"].split == "validation"]
    test = [r for r in rows if r["q"].split == "test"]
    model = fit_calibration([r["signals"] for r in val], [r["outcome"] for r in val])
    val_p = [model.probability(r["signals"]) for r in val]
    model.answer_at, model.abstain_at = choose_thresholds(val_p, [r["outcome"] for r in val], args.target_accuracy)
    test_p = [model.probability(r["signals"]) for r in test]
    y = [r["outcome"] for r in test]
    report = {
        "fitted_on": f"validation ({len(val)} questions, {sum(r['outcome'] for r in val)} with the answer retrieved)",
        "pipeline": {"reranker": reranker.model_name if reranker else None, "query_expansion": expand is not None,
                     "keyword_weight": retriever.keyword_weight, "keyword_seat": retriever.keyword_seat},
        "test_questions": len(test),
        "brier": brier_score(test_p, y), "ece": expected_calibration_error(test_p, y), "auroc": auroc(test_p, y),
        "answer_at": model.answer_at, "abstain_at": model.abstain_at,
        "at_abstain_threshold": risk_coverage(test_p, y, model.abstain_at) if model.abstain_at is not None else None,
        "at_answer_threshold": risk_coverage(test_p, y, model.answer_at) if model.answer_at is not None else None,
        # the app's current single rule: best cosine >= gate, or a crop with a bulletin is named
        "baseline_gate": {"gate": gate, **risk_coverage([r["baseline"] for r in test], y, 0.5),
                          "brier_of_0_1_rule": brier_score([r["baseline"] for r in test], y)},
        "baseline_cosine_auroc": auroc([r["signals"].best_similarity for r in test], y),
    }
    model.info = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "embedding_model": retriever.embedding_model,
                  "seed": args.seed, "dataset": dataset_hash(PROJECT / "data"), "test_metrics": report}
    target = settings.confidence_calibration_path
    if target.exists() and not args.overwrite_calibration:
        target = out_dir / "confidence_calibration.json"
        logger.info("A calibration file already exists; the new one is written to %s (use --overwrite-calibration)", target)
    model.save(target)
    report["saved_to"] = str(target)
    return {"status": "OK", **report}


# ------------------------------------------------------------------ end-to-end answers
def answers(args, settings, questions, corpus, retriever, out_dir: Path) -> dict:
    from app.core.tracing import tracer
    from app.services import claims as claim_checks
    from app.services import grounding
    from app.services.generation import OllamaGenerator
    from app.services.rag_service import RAGService

    generator = OllamaGenerator(settings.ollama_host, settings.ollama_model, settings.temperature, settings.ollama_timeout,
                                settings.ollama_think, settings.ollama_num_ctx)
    reachable, available = generator.status()
    if not (reachable and available):
        return {"status": "NOT RUN", "reason": f"Ollama model {settings.ollama_model} not reachable"}
    from app.main import optional_layers

    layers = optional_layers()   # the same optional layers backend/.env turns on for the app: compare runs before/after
    rag = RAGService(retriever, generator, top_k=settings.top_k, min_similarity=settings.min_similarity,
                     translate_queries=settings.query_translation, self_check=settings.self_check, **layers)
    picked = [q for q in questions if q.split == "test"][: args.answers_limit or None]
    rows, usage = [], []
    tracer.usage_sink = usage
    for q in picked:
        started = time.perf_counter()
        before = len(usage)
        try:
            result = rag.answer(q.question)
            failed = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Answer failed for %s: %s", q.id, exc)
            result, failed = None, True
        ms = (time.perf_counter() - started) * 1000
        row = {"id": q.id, "kind": q.kind, "language": q.language, "question": q.question, "latency_ms": round(ms),
               "failed": failed, "tokens": sum(u.get("total_tokens", 0) for u in usage[before:])}
        if result is not None:
            refused = not result.cited
            check = grounding.grounding_check(result.answer, result.chunks, q.question) if result.cited else {}
            report = claim_checks.verify(result.answer, result.chunks, q.question) if result.cited else None
            relevant = corpus.relevant(q)
            cited_ok = [corpus.position.get(c.chunk_id, -1) in relevant for _, c in result.cited]
            row.update({
                "answer": result.answer, "sources": " ; ".join(result.sources), "refused": refused,
                "correct": (refused if not q.answerable else
                            (not refused and grounding.contains_keyword(result.answer, q.answer_keywords or q.expected_keywords))),
                "grounded": check.get("grounded"),
                "citation_precision": round(sum(cited_ok) / len(cited_ok), 3) if cited_ok else None,
                "context_recall": int(any(corpus.position.get(c.chunk_id, -1) in relevant for c in result.chunks)) if q.answerable else None,
                "claims_supported_share": report.supported_share if report else None,
                "claims_unsupported": (report.counts["unsupported"] + report.counts["contradicted"]) if report else None,
                "claims_total": len(report.claims) if report else None,
            })
        rows.append(row)
        logger.info("%s %s %.1fs", q.id, "OK" if row.get("correct") else "--", ms / 1000)
    tracer.usage_sink = None

    def mean(key, rows_=rows):
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return round(sum(float(v) for v in vals) / len(vals), 3) if vals else None
    answerable = [r for r in rows if r["kind"] != "out_of_scope"]
    claims_total = sum(r.get("claims_total") or 0 for r in rows)
    summary = {
        "questions": len(rows), "failures": sum(r["failed"] for r in rows),
        "answer_correct (answerable, keyword rule)": mean("correct", answerable),
        "out_of_scope_refused": mean("correct", [r for r in rows if r["kind"] == "out_of_scope"]),
        "grounded (answered)": mean("grounded"), "citation_precision": mean("citation_precision"),
        "context_recall": mean("context_recall", answerable), "faithfulness (claims supported)": mean("claims_supported_share"),
        "unsupported_claim_rate": round(sum(r.get("claims_unsupported") or 0 for r in rows) / claims_total, 3) if claims_total else None,
        "latency_mean_ms": mean("latency_ms"), "tokens_mean": mean("tokens"),
    }
    with open(out_dir / "answers.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    # a sample for a person to check (every 3rd answered question), with empty columns to fill in
    with open(out_dir / "human_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "sources", "correct (y/n)", "faithful to sources (y/n)",
                         "safe to act on (y/n)", "notes"])
        for r in [r for r in rows if r.get("answer")][::3]:
            writer.writerow([r["id"], r["question"], r["answer"], r.get("sources", ""), "", "", "", ""])
    on = sorted(k for k in ("reranker", "expansion", "classify", "verify_claims", "nli", "confidence") if layers.get(k))
    return {"status": "OK", "llm": settings.ollama_model, "layers_on": on or ["none (original pipeline)"], "summary": summary}


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="*", help="only these (E1 E2 E2x E3 E4 E5 E6 E7)")
    parser.add_argument("--chunks-file", type=Path, help="keyword-only run on exported passages (no model needed)")
    parser.add_argument("--export-chunks", type=Path, help="write the store's passages to this JSONL file and stop")
    parser.add_argument("--reranker", nargs="?", const="", default=None, help="evaluate reranking (optional model name)")
    parser.add_argument("--allow-download", action="store_true", help="let the reranker download its model")
    parser.add_argument("--finetuned-store", type=Path, help="vector store built by scripts/build_collection.py (E7)")
    parser.add_argument("--translate", action="store_true", help="add the LLM translation to the search, as the app does")
    parser.add_argument("--raw", action="store_true", help="search Egyptian questions without the dialect rewrite")
    parser.add_argument("--tune", action="store_true", help="grid-search settings on the validation questions")
    parser.add_argument("--calibrate", action="store_true", help="fit the confidence model (validation) and test it")
    parser.add_argument("--target-accuracy", type=float, default=0.9, help="accuracy the 'answer' threshold must reach")
    parser.add_argument("--overwrite-calibration", action="store_true")
    parser.add_argument("--answers", action="store_true", help="end-to-end answers with the LLM on the test questions")
    parser.add_argument("--answers-limit", type=int, default=0)
    parser.add_argument("--keyword-weight", type=float, default=None, help="override KEYWORD_WEIGHT for this run")
    parser.add_argument("--keyword-seat", choices=["on", "off"], default=None, help="override KEYWORD_SEAT for this run")
    parser.add_argument("--rrf-k", type=int, default=0)
    parser.add_argument("--retrieval-candidates", type=int, default=0)
    parser.add_argument("--rerank-candidates", type=int, default=30)
    parser.add_argument("--expansion-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from app.core.config import get_settings

    settings = get_settings()
    started = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else "")
    out = args.out or PROJECT / "evaluation" / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions(PROJECT / "data", seed=args.seed)

    retriever, load_s = None, None
    if args.chunks_file:
        ids, texts, metas = runner.load_chunk_file(args.chunks_file)
        corpus_source = str(args.chunks_file)
        rag_config = {}
    else:
        retriever, load_s = open_store(args, settings)
        rag_config = retriever.config
        corpus_source = str(settings.vector_store_path)
        corpus = runner.corpus_from_retriever(retriever)
        ids, texts, metas = corpus.ids, corpus.texts, corpus.metas
        if args.export_chunks:
            with open(args.export_chunks, "w", encoding="utf-8") as f:
                for cid, text, meta in zip(ids, texts, metas):
                    f.write(json.dumps({"id": cid, "text": text, "metadata": meta}, ensure_ascii=False) + "\n")
            logger.info("Wrote %d passages to %s", len(ids), args.export_chunks)
            return
    corpus = runner.Corpus(ids, texts, metas)
    keyword = retriever if retriever is not None else runner.keyword_retriever(
        ids, texts, metas, rrf_k=args.rrf_k or settings.rrf_k, candidates=args.retrieval_candidates or settings.retrieval_candidates,
        bm25_k1=settings.bm25_k1, bm25_b=settings.bm25_b)

    # label audit: every answerable question must have at least one passage the relevance rule accepts
    answerable = [q for q in questions if q.answerable]
    unlabelled = [{"id": q.id, "question": q.question, "expected_source": q.expected_source}
                  for q in answerable if not corpus.relevant(q)]
    logger.info("%d questions (%d answerable, %d without any matching passage: excluded from ranking metrics)",
                len(questions), len(answerable), len(unlabelled))

    translate = None
    if args.translate:
        from app.services.generation import OllamaGenerator

        generator = OllamaGenerator(settings.ollama_host, settings.ollama_model, 0.0, settings.ollama_timeout,
                                    settings.ollama_think, settings.ollama_num_ctx)
        translate = generator.translate
    queries = runner.Queries(translate, out / "translations.json", raw=args.raw)

    reranker, reranker_load_s = None, None
    if args.reranker is not None:
        from app.services.reranker import Reranker

        reranker = Reranker(args.reranker or settings.reranker_model, settings.reranker_device, settings.reranker_batch_size,
                            settings.reranker_max_length, args.allow_download or settings.reranker_allow_download,
                            settings.reranker_trust_remote_code)
        t0 = time.perf_counter()
        reranker.load()
        reranker_load_s = round(time.perf_counter() - t0, 2)
    finetuned = None
    if args.finetuned_store:
        try:
            finetuned, _ = open_store(args, settings, args.finetuned_store,
                                      json.loads((args.finetuned_store / "rag_config.json").read_text(encoding="utf-8"))["embedding_model"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fine-tuned store could not be opened: %s", exc)

    experiments = runner.build_experiments(retriever, keyword, reranker, finetuned, candidates=args.rerank_candidates,
                                           expansion_weight=args.expansion_weight)
    retrieval, tables, per_question = run_retrieval(args, questions, corpus, experiments, queries)
    queries.save()

    results = {
        "run": run_id, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"files": ["data/eval_questions.json", "data/eval_questions_egyptian.json"],
                    "sha256_12": dataset_hash(PROJECT / "data"), "questions": len(questions),
                    "answerable": len(answerable), "validation": sum(q.split == "validation" for q in questions),
                    "test": sum(q.split == "test" for q in questions), "seed": args.seed,
                    "without_matching_passage": unlabelled},
        "corpus": {"source": corpus_source, "passages": len(ids), "built_at": rag_config.get("built_at"),
                   "chunking": rag_config.get("chunking")},
        "models": {"embedding": getattr(retriever, "embedding_model", None), "embedding_load_s": load_s,
                   "reranker": reranker.describe() if reranker else None, "reranker_load_s": reranker_load_s,
                   "translation_llm": settings.ollama_model if args.translate else None},
        "settings": {"rrf_k": keyword.rrf_k, "candidates": keyword.candidates, "bm25_k1": keyword.bm25.k1,
                     "keyword_weight": keyword.keyword_weight, "keyword_seat": keyword.keyword_seat,
                     "bm25_b": keyword.bm25.b, "rerank_candidates": args.rerank_candidates,
                     "expansion_weight": args.expansion_weight, "dialect_rewrite": not args.raw,
                     "translation": bool(args.translate), "depth": runner.DEPTH},
        "versions": versions(),
        "experiments": retrieval,
    }
    if args.tune:
        results["tuning"] = tune(args, questions, corpus, retriever, keyword, queries)
    if args.calibrate:
        results["calibration"] = calibrate(args, settings, questions, corpus, retriever, queries, out)
    if args.answers:
        results["answers"] = answers(args, settings, questions, corpus, retriever, out) if retriever else \
            {"status": "NOT RUN", "reason": "needs the vector store"}
    results["runtime_s"] = round(time.perf_counter() - started, 1)
    results["memory_rss_mb"] = peak_memory_mb()
    write_outputs(out, results, tables, per_question)

    # the readable report
    lines = [f"# SAQYA evaluation {run_id}", "",
             f"- Questions: {len(answerable)} answerable ({results['dataset']['validation']} validation / "
             f"{results['dataset']['test']} test, split seed {args.seed}); dataset {results['dataset']['sha256_12']}",
             f"- Passages: {len(ids)} ({corpus_source})", f"- Embedding model: {results['models']['embedding']}",
             f"- Search text: spelling repair{'' if args.raw else ' + dialect rewrite'}{' + LLM translation' if args.translate else ''}",
             f"- Questions with no passage matching their label (excluded): {len(unlabelled)}", "",
             "## Test split (settings were not chosen on these questions)", ""]
    cols = ["experiment", "name", "questions", "recall@5", "recall@10", "precision@5", "mrr@10", "ndcg@10", "hit@5"]
    for subset in ("all", "standard Arabic", "Egyptian Arabic", "English", "cross-language"):
        rows = [t for t in tables if t["split"] == "test" and t["subset"] == subset]
        if rows:
            lines += [f"### {subset}", "", markdown_table(rows, cols), ""]
    lines += ["## Validation split", "", markdown_table([t for t in tables if t["split"] == "validation" and t["subset"] == "all"], cols), ""]
    lat = [{"experiment": k, "status": v["status"], **(v.get("latency") or {}), "reason": v.get("reason", "")}
           for k, v in retrieval.items()]
    lines += ["## Latency per question and experiments not run", "",
              markdown_table(lat, ["experiment", "status", "mean_ms", "p50_ms", "p95_ms", "reason"]), ""]
    for key in ("tuning", "calibration", "answers"):
        if key in results:
            lines += [f"## {key.title()}", "", "```json", json.dumps(results[key], ensure_ascii=False, indent=1, default=str)[:6000], "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results in %s (%.0fs)", out, results["runtime_s"])


if __name__ == "__main__":
    main()
