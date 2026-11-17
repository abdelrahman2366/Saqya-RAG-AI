"""Training data for contrastive fine-tuning of the embedding model, built from the library itself.

Why a new set (the audit of notebooks/results/egyptian_training_pairs.jsonl is in the report this writes):
the existing pairs are Egyptian text -> standard-Arabic text (distillation targets), with no passages and no
negatives, half of them identical pairs, and 4 evaluation questions copied in. A retrieval model learns best
from (question, passage that answers it, similar passage that does not) triples. No hand-labelled questions
exist beyond the evaluation set (which must not be trained on), so questions are generated from the
bulletins' own section headings — a weak but honest signal: a heading says what its passages are about.

For every passage with a section heading (and a crop, from its document's topic):
- queries: the heading as a standard-Arabic request, an Egyptian one (with the names farmers use: الأوطة،
  الدرة، البتنجان...) and its dialect-rewritten form (what the app searches), or English for English documents;
- positive: the passage;
- hard negative: the best BM25 match for the query from *another* section or crop (similar words, wrong
  answer). Passages with the same heading and crop are never negatives (they may answer it too).

Leakage control (reported):
- passages that answer any evaluation question (validation or test) are never positives or negatives;
- generated queries too close to an evaluation question (word overlap >= 0.5) are dropped;
- the dev split (held out from training, for monitoring) is by section group, not by row.
Duplicates (same normalised query) are removed and each section contributes at most --per-section rows.

    python scripts/build_finetune_data.py                          # passages from the vector store
    python scripts/build_finetune_data.py --chunks-file chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
PROJECT = BACKEND.parent

from app.evaluation import runner  # noqa: E402
from app.evaluation.dataset import DEFAULT_SEED, load_questions  # noqa: E402
from app.services.dialect import to_msa  # noqa: E402
from app.services.extraction import clean_heading  # noqa: E402
from app.services.glossary import GLOSSARY  # noqa: E402
from app.services.lexical import words  # noqa: E402
from app.utils.text import detect_language, normalize_for_match  # noqa: E402

logger = logging.getLogger("finetune-data")

# what farmers in Egypt call the crops (as in notebooks/egyptian_finetune.ipynb)
EGYPTIAN_NAMES = {
    "tomato": ["الأوطة", "القوطة"], "maize": ["الدرة", "الدرة الشامي"], "sorghum": ["الدرة الرفيعة", "الدرة العويجة"],
    "rice": ["الرز"], "sugar beet": ["البنجر"], "sugarcane": ["القصب"], "faba bean": ["الفول"],
    "sunflower": ["عباد الشمس"], "eggplant": ["البتنجان", "البدنجان"], "zucchini": ["الكوسا"],
    "cantaloupe": ["الشمام"], "mango": ["المانجا", "المنجة"], "citrus": ["البرتقان", "اليوسفي"],
    "date palm": ["النخل", "البلح"], "clover": ["البرسيم"],
}
CROPS = {t.en: t for t in GLOSSARY if t.kind == "crop"}
MSA_TEMPLATES = ["{h} في {c}", "ما هي {h} في {c}؟", "أريد معلومات عن {h} في {c}"]
EGY_TEMPLATES = ["عايز أعرف {h} في {c}", "قولي على {h} بتاع {c}", "إيه هي {h} في {c}؟"]
EN_TEMPLATES = ["{h} for {c}", "What about {h} in {c}?", "Tell me about {h} in {c}"]
GENERIC_HEADINGS = {"مقدمة", "المقدمة", "introduction", "مقدمه", "الخلاصة", "references", "المراجع", "contents",
                    "conclusion", "abstract", "الملخص", "summary"}


def heading_of(text: str, meta: dict) -> str:
    first = text.split("\n", 1)[0]
    heading = str(meta.get("section") or "")
    if " › " in first:
        parts = [p.strip() for p in first.split(" › ")[1:] if p.strip()]
        heading = parts[-1] if parts else heading
    return re.sub(r"^[\W\d_]+", "", clean_heading(heading)).strip()   # ".73 فقد الرطوبة" -> "فقد الرطوبة"


def overlap(a: str, b: str) -> float:
    wa, wb = set(words(a)), set(words(b))
    return len(wa & wb) / max(len(wa | wb), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks-file", type=Path, help="exported passages (default: read the vector store)")
    parser.add_argument("--existing-pairs", type=Path, default=PROJECT / "notebooks" / "results" / "egyptian_training_pairs.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT / "data" / "finetune")
    parser.add_argument("--per-section", type=int, default=3)
    parser.add_argument("--dev-share", type=float, default=0.05)
    parser.add_argument("--max-leak-overlap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng = random.Random(args.seed)

    if args.chunks_file:
        ids, texts, metas = runner.load_chunk_file(args.chunks_file)
    else:
        from app.core.config import get_settings
        from app.services.retrieval import Retriever, load_rag_config
        import chromadb

        settings = get_settings()
        config = load_rag_config(settings.vector_store_path)
        collection = chromadb.PersistentClient(path=str(settings.vector_store_path)).get_collection(
            settings.collection_name or config["collection_name"], embedding_function=None)
        data = collection.get(include=["documents", "metadatas"])
        ids, texts, metas = list(data["ids"]), list(data["documents"]), [m or {} for m in data["metadatas"]]
    keyword = runner.keyword_retriever(ids, texts, metas)
    corpus = runner.Corpus(ids, texts, metas)
    questions = load_questions(PROJECT / "data", seed=args.seed)
    eval_texts = [q.question for q in questions]
    protected = set().union(*(corpus.relevant(q) for q in questions if q.answerable))
    logger.info("%d passages; %d answer an evaluation question (never used for training)", len(ids), len(protected))

    stats = Counter()
    rows, seen_queries = [], set()
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, (text, meta) in enumerate(zip(texts, metas)):
        heading = heading_of(text, meta)
        if len(heading) < 4 or normalize_for_match(heading) in GENERIC_HEADINGS or len(heading) > 80:
            stats["no usable heading"] += 1
            continue
        groups[(meta.get("source"), heading)].append(i)
    # a heading that is really a template ("1213. خلاصة قابلة للفهرسة" repeated through a generated encyclopedia)
    # says nothing about its passage: the same text heading many sections of one document
    template = Counter((src, re.sub(r"^[\W\d_]+", "", h)) for src, h in groups)
    for key in [k for k in groups if template[(k[0], re.sub(r"^[\W\d_]+", "", k[1]))] > 5]:
        stats["template heading"] += len(groups.pop(key))
    for (source, heading), members in groups.items():
        topic = str(metas[members[0]].get("topic") or "")
        crop = CROPS.get(topic.split(" / ")[-1]) if topic else None
        language = detect_language(texts[members[0]])
        added = 0
        for i in rng.sample(members, len(members)):
            if added >= args.per_section:
                break
            if i in protected:
                stats["positive answers an evaluation question"] += 1
                continue
            h = heading
            if language == "ar":
                crop_ar = crop.ar if crop else ""
                has_crop = crop is None or normalize_for_match(crop.ar.replace("ال", "")) in normalize_for_match(h)
                msa = (rng.choice(MSA_TEMPLATES).format(h=h, c=crop_ar) if not has_crop else h)
                names = EGYPTIAN_NAMES.get(crop.en if crop else "", []) or [crop_ar]
                egy = (rng.choice(EGY_TEMPLATES).format(h=h, c=rng.choice(names)) if not has_crop
                       else rng.choice(["عايز أعرف {h}", "قولي على {h}", "اشرحلي {h}"]).format(h=h))
                candidates = [("msa", msa), ("egyptian", egy), ("egyptian-rewritten", to_msa(egy))]
            else:
                c = crop.en if crop else ""
                candidates = [("english", rng.choice(EN_TEMPLATES).format(h=h, c=c) if c and c.lower() not in h.lower()
                               else f"Tell me about {h}")]
            for kind, query in candidates:
                key = normalize_for_match(re.sub(r"\W+", " ", query)).strip()
                if key in seen_queries:
                    stats["duplicate query"] += 1
                    continue
                if max((overlap(query, e) for e in eval_texts), default=0) >= args.max_leak_overlap:
                    stats["too close to an evaluation question"] += 1
                    continue
                negative = None
                for j, _ in keyword.bm25.top(query, 15):
                    if j == i or j in protected:
                        continue
                    same_heading = heading_of(texts[j], metas[j]) == heading
                    same_crop = (metas[j].get("topic") or "") == topic
                    if same_heading and (same_crop or metas[j].get("source") == source):
                        continue   # may answer the same question: not a negative
                    negative = j
                    break
                seen_queries.add(key)
                rows.append({"query": query, "positive": texts[i], "negative": texts[negative] if negative is not None else None,
                             "kind": kind, "language": detect_language(query), "group": f"{source}::{heading}",
                             "positive_id": ids[i], "negative_id": ids[negative] if negative is not None else None})
                stats[f"rows {kind}"] += 1
                stats["with hard negative" if negative is not None else "without hard negative"] += 1
            added += 1

    # dev split by section group (no section is in both)
    group_names = sorted({r["group"] for r in rows})
    rng.shuffle(group_names)
    dev_groups = set(group_names[: max(1, int(len(group_names) * args.dev_share))])
    train = [r for r in rows if r["group"] not in dev_groups]
    dev = [r for r in rows if r["group"] in dev_groups]

    # audit of the existing distillation pairs
    audit = None
    if args.existing_pairs.exists():
        pairs = [json.loads(l) for l in args.existing_pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
        eval_norm = {normalize_for_match(t) for t in eval_texts}
        audit = {"file": str(args.existing_pairs.relative_to(PROJECT)) if args.existing_pairs.is_relative_to(PROJECT) else str(args.existing_pairs),
                 "rows": len(pairs), "kinds": dict(Counter(p.get("kind") for p in pairs)),
                 "identical_pairs (student == teacher)": sum(p["student"] == p["teacher"] for p in pairs),
                 "exact_duplicate_rows": len(pairs) - len({(p["student"], p["teacher"]) for p in pairs}),
                 "evaluation_questions_copied_in": sum(normalize_for_match(p["student"]) in eval_norm
                                                       or normalize_for_match(p["teacher"]) in eval_norm for p in pairs),
                 "has_passages": False, "has_negatives": False,
                 "objective": "distillation (Egyptian text -> the base model's vector of its standard-Arabic twin)"}

    out = args.out
    if out.exists() and any(out.iterdir()):
        out = out / datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.info("%s already has data: writing to %s", args.out, out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("dev", dev)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()[:12]
    report = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": args.seed, "train_sha256_12": digest,
              "passages": len(ids), "section_groups": len(groups), "train_rows": len(train), "dev_rows": len(dev),
              "dev_groups": len(dev_groups), "languages": dict(Counter(r["language"] for r in rows)),
              "kinds": dict(Counter(r["kind"] for r in rows)), "stats": dict(stats),
              "protected_passages (answer an evaluation question)": len(protected),
              "existing_pairs_audit": audit,
              "limitations": ["queries are generated from headings (weak supervision), not asked by farmers",
                              "a passage under a broad heading may not answer the generated query",
                              "hard negatives are mined with BM25 and may occasionally be relevant (false negatives)"]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("train %d / dev %d rows -> %s", len(train), len(dev), out)
    logger.info("%s", json.dumps({k: v for k, v in report.items() if k in ("kinds", "stats")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
