"""Contrastive fine-tuning of the embedding model on data/finetune/ (scripts/build_finetune_data.py).

Compatibility with BGE-M3 as the app uses it: the app loads `BAAI/bge-m3` with sentence-transformers and uses
only its dense vector (XLM-RoBERTa -> CLS pooling -> L2 normalisation; the sparse and ColBERT heads of the
FlagEmbedding package are not used anywhere). This script trains exactly that SentenceTransformer forward
pass and saves it in the same format, so the app, the notebooks and build_collection.py load it unchanged.
The script checks the pooling/normalisation modules before training and stops if they differ.

Objective: MultipleNegativesRankingLoss — for each query, its passage must score higher than every other
passage in the batch and than its BM25-mined hard negative (cross-entropy over scaled cosine similarities,
scale 20). Written out here rather than through SentenceTransformerTrainer so it needs no extra packages
(`datasets`, `accelerate`) and runs the same loop the Egyptian notebook already ran on your GPU.

Memory: only the top --train-layers transformer layers learn (6 of 24 by default; the rest are frozen), the
sequence length is capped (--max-length 256) and fp16 autocast is used on a GPU. Larger effective batches
(more in-batch negatives) would need CachedMultipleNegativesRankingLoss; batch 16 + one hard negative fits
an 8 GB GPU.

Model selection uses only the *validation* evaluation questions (dense search, MRR@10, every language):
the model is saved only when it beats the original on validation by --min-gain and no language slice drops
by more than --max-drop. The test questions are not read by this script; compare on them afterwards with
scripts/build_collection.py + scripts/evaluate.py --finetuned-store (experiment E7).

The original model is never modified. Output: models/bge-m3-saqya/ (a new timestamped folder if it exists)
with training_info.json (settings, data hash, seed, device, losses, validation metrics before/after).

    python scripts/finetune_embeddings.py                         # GPU recommended (CPU: hours)
    python scripts/finetune_embeddings.py --epochs 2 --lr 1e-5 --train-layers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
PROJECT = BACKEND.parent

from app.evaluation import runner  # noqa: E402
from app.evaluation.dataset import DEFAULT_SEED, load_questions, subsets  # noqa: E402
from app.evaluation.metrics import ranking_summary  # noqa: E402

logger = logging.getLogger("finetune")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_architecture(model) -> None:
    """The app's usage: transformer -> CLS pooling -> normalise. Stop rather than train something else."""
    names = [type(m).__name__ for m in model]
    pooling = next((m for m in model if type(m).__name__ == "Pooling"), None)
    mode = None
    if pooling is not None:   # sentence-transformers >= 6: .pooling_mode; older: get_pooling_mode_str()
        mode = getattr(pooling, "pooling_mode", None) or getattr(pooling, "get_pooling_mode_str", lambda: None)()
    if names[0] != "Transformer" or mode != "cls" or "Normalize" not in names:
        raise SystemExit(f"Unexpected model structure {names} (pooling {mode}); expected Transformer -> CLS -> Normalize")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------ validation retrieval (model selection)
class Validation:
    """Dense search over a passage set with the evaluation's relevance rule, on the VALIDATION questions."""

    def __init__(self, ids, texts, metas, questions, sample: int, seed: int) -> None:
        corpus = runner.Corpus(ids, texts, metas)
        self.questions = [q for q in questions if q.answerable and q.split == "validation"]
        relevant = set().union(*(corpus.relevant(q) for q in self.questions))
        rest = [i for i in range(len(ids)) if i not in relevant]
        if sample and len(rest) > sample:   # all answer passages + a fixed random sample of the others
            rest = random.Random(seed).sample(rest, sample)
        keep = sorted(relevant | set(rest))
        self.corpus = runner.Corpus([ids[i] for i in keep], [texts[i] for i in keep], [metas[i] for i in keep])
        self.queries = runner.Queries()
        self.slices = {name: {q.id for q in qs} for name, qs in subsets(self.questions).items()}

    def __call__(self, model, batch_size: int) -> dict:
        vecs = model.encode(self.corpus.texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
        out_rows = []
        for q in self.questions:
            qv = model.encode(self.queries(q.question), normalize_embeddings=True, convert_to_numpy=True)
            scores = (vecs @ qv.T).max(axis=1)
            top = np.argsort(-scores)[: runner.DEPTH]
            relevant = self.corpus.relevant(q)
            out_rows.append((q.id, [int(i) in relevant for i in top], len(relevant)))
        return {name: ranking_summary([(h, n) for qid, h, n in out_rows if qid in ids])
                for name, ids in self.slices.items()}


# ------------------------------------------------------------------ training
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="BAAI/bge-m3")
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "finetune")
    parser.add_argument("--chunks-file", type=Path, help="passages for validation (default: the vector store)")
    parser.add_argument("--output", type=Path, default=PROJECT / "models" / "bge-m3-saqya")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--train-layers", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=0, help="use only this many training rows (quick tests)")
    parser.add_argument("--val-sample", type=int, default=3000, help="non-answer passages in the validation corpus (0: all)")
    parser.add_argument("--min-gain", type=float, default=0.01, help="validation MRR@10 gain required to save")
    parser.add_argument("--max-drop", type=float, default=0.01, help="largest allowed MRR@10 drop on any language slice")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-anyway", action="store_true", help="save even if validation did not improve (marked)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("app.services.spelling").setLevel(logging.WARNING)

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    from app.services.reranker import pick_device

    set_seed(args.seed)
    device = pick_device(args.device)
    train_rows = read_rows(args.data / "train.jsonl")
    dev_rows = read_rows(args.data / "dev.jsonl") if (args.data / "dev.jsonl").exists() else []
    if args.max_rows:
        train_rows = random.Random(args.seed).sample(train_rows, min(args.max_rows, len(train_rows)))
    data_hash = hashlib.sha256((args.data / "train.jsonl").read_bytes()).hexdigest()[:12]

    if args.chunks_file:
        ids, texts, metas = runner.load_chunk_file(args.chunks_file)
    else:
        import chromadb

        from app.core.config import get_settings
        from app.services.retrieval import load_rag_config

        settings = get_settings()
        cfg = load_rag_config(settings.vector_store_path)
        data = chromadb.PersistentClient(path=str(settings.vector_store_path)).get_collection(
            settings.collection_name or cfg["collection_name"], embedding_function=None).get(include=["documents", "metadatas"])
        ids, texts, metas = list(data["ids"]), list(data["documents"]), [m or {} for m in data["metadatas"]]
    validation = Validation(ids, texts, metas, load_questions(PROJECT / "data", seed=args.seed), args.val_sample, args.seed)

    model = SentenceTransformer(args.base_model, device=device)
    check_architecture(model)
    model.max_seq_length = args.max_length
    encode_batch = 32 if device == "cuda" else 8
    t0 = time.perf_counter()
    before = validation(model, encode_batch)
    logger.info("Validation before training (%.0fs): %s", time.perf_counter() - t0,
                {k: v.get("mrr@10") for k, v in before.items()})

    layers = model[0].auto_model.encoder.layer
    for p in model.parameters():
        p.requires_grad = False
    for layer in layers[-args.train_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warmup = int(total * args.warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(warmup, 1)) * max(0.0, (total - s) / max(total - warmup, 1)))
    prepare = getattr(model, "preprocess", None) or model.tokenize
    use_amp = device == "cuda"

    def embed(batch: list[str]):
        features = prepare(batch)
        features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in features.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            return F.normalize(model(features)["sentence_embedding"].float(), dim=-1)

    def batch_loss(rows: list[dict]):
        q = embed([r["query"] for r in rows])
        passages = [r["positive"] for r in rows] + [r["negative"] for r in rows if r.get("negative")]
        p = embed(passages)
        logits = args.scale * q @ p.T            # every query against every passage in the batch (+ hard negatives)
        labels = torch.arange(len(rows), device=device)
        return F.cross_entropy(logits, labels), (logits.argmax(dim=1) == labels).float().mean().item()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = [train_rows[i] for i in order[start:start + args.batch_size]]
            if len(rows) < 2:
                continue
            loss, _ = batch_loss(rows)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                logger.info("epoch %d step %d/%d loss %.4f", epoch + 1, step, total, float(np.mean(losses[-50:])))
        model.eval()
        with torch.no_grad():
            dev = [batch_loss(dev_rows[i:i + args.batch_size]) for i in range(0, len(dev_rows), args.batch_size)
                   if len(dev_rows[i:i + args.batch_size]) >= 2]
        after_epoch = validation(model, encode_batch)
        history.append({"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 4) if losses else None,
                        "dev_loss": round(float(np.mean([l.item() for l, _ in dev])), 4) if dev else None,
                        "dev_in_batch_accuracy": round(float(np.mean([a for _, a in dev])), 4) if dev else None,
                        "validation_mrr@10": {k: v.get("mrr@10") for k, v in after_epoch.items()}})
        logger.info("Epoch %d: %s", epoch + 1, history[-1])

    after = validation(model, encode_batch)
    gain = after["all"]["mrr@10"] - before["all"]["mrr@10"]
    drops = {k: round(before[k]["mrr@10"] - after[k]["mrr@10"], 4) for k in before
             if k in after and before[k]["mrr@10"] - after[k]["mrr@10"] > args.max_drop}
    better = gain >= args.min_gain and not drops
    info = {
        "base_model": args.base_model, "objective": "MultipleNegativesRankingLoss (in-batch + 1 BM25 hard negative)",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": device, "seed": args.seed,
        "data": str(args.data), "train_sha256_12": data_hash, "train_rows": len(train_rows), "dev_rows": len(dev_rows),
        "settings": {k: v for k, v in vars(args).items() if k not in ("data", "output", "chunks_file")},
        "trainable_parameters_m": round(sum(p.numel() for p in trainable) / 1e6, 1),
        "history": history, "validation_before": before, "validation_after": after,
        "validation_mrr_gain": round(gain, 4), "slices_that_dropped": drops,
        "kept": bool(better or args.save_anyway), "better_on_validation": bool(better),
        "test_split": "not used here: run scripts/build_collection.py, then scripts/evaluate.py --finetuned-store",
    }
    logger.info("Validation MRR@10 %.4f -> %.4f (gain %.4f); dropped slices: %s", before["all"]["mrr@10"],
                after["all"]["mrr@10"], gain, drops or "none")
    if not (better or args.save_anyway):
        report = PROJECT / "evaluation" / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-finetune-not-kept.json")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Not better on validation: nothing saved, the app keeps %s. Report: %s", args.base_model, report)
        return
    output = args.output
    if output.exists():
        output = output.with_name(f"{output.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    model.save(str(output))
    (output / "training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved to %s%s", output, "" if better else " (NOT better on validation: saved because of --save-anyway)")


if __name__ == "__main__":
    main()
