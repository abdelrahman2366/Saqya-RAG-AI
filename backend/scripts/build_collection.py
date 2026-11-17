"""Embed the library's passages with another model into a SEPARATE vector store (for a fine-tuned model).

The current store is opened read-only: its passages, ids and metadata (source, pages, section, topic) are
copied unchanged and only the vectors are recomputed, so citations stay identical. The new store goes to
backend/data/vector_store_finetuned/ (or --out) with its own rag_config.json naming the model, so vectors of
two models are never mixed in one collection. An existing output folder is never overwritten: a new
timestamped folder is used instead.

The relevance gate (min_similarity) is copied from the original store but marked as not recalibrated: a new
model spreads similarities differently. Run `scripts/evaluate.py --finetuned-store <dir> --calibrate` (and
compare E7 with E3) before switching the app over with FINE_TUNED_EMBEDDINGS_ENABLED=true.

    python scripts/build_collection.py --model models/bge-m3-saqya
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

logger = logging.getLogger("build-collection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="embedding model folder (e.g. models/bge-m3-saqya)")
    parser.add_argument("--source", type=Path, help="vector store to copy passages from (default: the app's)")
    parser.add_argument("--out", type=Path, default=BACKEND / "data" / "vector_store_finetuned")
    parser.add_argument("--collection", default="agri_docs_finetuned")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import chromadb
    import numpy as np

    from app.core.config import get_settings
    from app.services.retrieval import load_embedder, load_rag_config, resolve_model

    settings = get_settings()
    source = args.source or settings.vector_store_path
    config = load_rag_config(source)
    original = chromadb.PersistentClient(path=str(source)).get_collection(
        settings.collection_name or config["collection_name"], embedding_function=None)
    data = original.get(include=["documents", "metadatas"])
    logger.info("%d passages from %s (%s)", len(data["ids"]), source, config["collection_name"])

    out = args.out
    if out.exists() and any(out.iterdir()):
        out = out.with_name(f"{out.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        logger.info("%s already exists: writing to %s", args.out, out)
    out.mkdir(parents=True, exist_ok=True)
    embedder = load_embedder(args.model)
    started = time.perf_counter()
    vectors = np.asarray(embedder.encode(list(data["documents"]), batch_size=args.batch_size, normalize_embeddings=True,
                                         show_progress_bar=True), dtype=np.float32)
    client = chromadb.PersistentClient(path=str(out))
    collection = client.create_collection(args.collection, metadata={"hnsw:space": "cosine"}, embedding_function=None)
    for start in range(0, len(data["ids"]), 1000):
        end = start + 1000
        collection.add(ids=data["ids"][start:end], documents=data["documents"][start:end],
                       metadatas=[m or {} for m in data["metadatas"][start:end]], embeddings=vectors[start:end].tolist())
    new_config = {**config, "collection_name": args.collection, "embedding_model": args.model,
                  "embedding_dim": int(vectors.shape[1]), "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "built_from": {"store": str(source), "collection": config["collection_name"],
                                 "embedding_model": config["embedding_model"], "built_at": config.get("built_at")},
                  "min_similarity_recalibrated": False,
                  "min_similarity_note": "copied from the original store; recalibrate for this model before relying on it"}
    try:
        info = json.loads((Path(resolve_model(args.model)) / "training_info.json").read_text(encoding="utf-8"))
        new_config["training"] = {k: info.get(k) for k in ("base_model", "date", "train_sha256_12", "validation_mrr_gain",
                                                            "better_on_validation")}
    except Exception:  # noqa: BLE001 - not a model trained by finetune_embeddings.py
        pass
    (out / "rag_config.json").write_text(json.dumps(new_config, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d vectors (%d dims) to %s in %.0fs", len(vectors), vectors.shape[1], out, time.perf_counter() - started)


if __name__ == "__main__":
    main()
