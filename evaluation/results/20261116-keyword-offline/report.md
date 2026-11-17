# SAQYA evaluation 20261116-202020-offline-keyword

- Questions: 133 answerable (84 validation / 59 test, split seed 13); dataset 2d1ba045dd9d
- Passages: 8872 (local export of the library, same chunker; keyword-only run)
- Embedding model: None
- Search text: spelling repair + dialect rewrite
- Questions with no passage matching their label (excluded): 0

## Test split (settings were not chosen on these questions)

### all

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E2 | BM25 only | 52 | 0.706 | 0.762 | 0.219 | 0.683 | 0.668 | 0.788 |
| E2x | BM25 + query expansion | 52 | 0.702 | 0.782 | 0.215 | 0.691 | 0.672 | 0.788 |

### standard Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E2 | BM25 only | 22 | 0.827 | 0.868 | 0.218 | 0.823 | 0.808 | 0.864 |
| E2x | BM25 + query expansion | 22 | 0.827 | 0.868 | 0.218 | 0.823 | 0.808 | 0.864 |

### Egyptian Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E2 | BM25 only | 21 | 0.700 | 0.719 | 0.191 | 0.573 | 0.592 | 0.762 |
| E2x | BM25 + query expansion | 21 | 0.700 | 0.767 | 0.191 | 0.594 | 0.605 | 0.762 |

### English

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E2 | BM25 only | 9 | 0.424 | 0.606 | 0.289 | 0.599 | 0.504 | 0.667 |
| E2x | BM25 + query expansion | 9 | 0.402 | 0.606 | 0.267 | 0.594 | 0.499 | 0.667 |

### cross-language

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E2 | BM25 only | 4 | 0.062 | 0.062 | 0.050 | 0.062 | 0.042 | 0.250 |
| E2x | BM25 + query expansion | 4 | 0.062 | 0.062 | 0.050 | 0.050 | 0.038 | 0.250 |

## Validation split

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E2 | BM25 only | 81 | 0.725 | 0.780 | 0.217 | 0.564 | 0.616 | 0.765 |
| E2x | BM25 + query expansion | 81 | 0.719 | 0.792 | 0.217 | 0.565 | 0.619 | 0.765 |

## Latency per question and experiments not run

| experiment | status | mean_ms | p50_ms | p95_ms | reason |
|---|---|---|---|---|---|
| E1 | NOT RUN |  |  |  | no embedding model in this run (needs the vector store and bge-m3: run on the PC) |
| E2 | OK | 0.300 | 0.300 | 0.400 |  |
| E2x | OK | 1.700 | 1.700 | 2.300 |  |
| E3 | NOT RUN |  |  |  | no embedding model in this run (needs the vector store and bge-m3: run on the PC) |
| E4 | NOT RUN |  |  |  | no embedding model in this run (needs the vector store and bge-m3: run on the PC) |
| E5 | NOT RUN |  |  |  | no embedding model in this run (needs the vector store and bge-m3: run on the PC) |
| E6 | NOT RUN |  |  |  | no embedding model in this run (needs the vector store and bge-m3: run on the PC) |
| E7 | NOT RUN |  |  |  | no fine-tuned model/collection (scripts/finetune_embeddings.py, scripts/build_collection.py) |
