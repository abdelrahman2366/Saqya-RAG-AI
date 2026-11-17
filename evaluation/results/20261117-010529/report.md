# SAQYA evaluation 20261117-010529

- Questions: 133 answerable (84 validation / 59 test, split seed 13); dataset 2d1ba045dd9d
- Passages: 8872 (D:\agri-rag-assistant\backend\data\vector_store)
- Embedding model: BAAI/bge-m3
- Search text: spelling repair + dialect rewrite + LLM translation
- Questions with no passage matching their label (excluded): 0

## Test split (settings were not chosen on these questions)

### all

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 52 | 0.861 | 0.879 | 0.254 | 0.840 | 0.810 | 0.962 |
| E1x | dense only + query expansion | 52 | 0.857 | 0.865 | 0.250 | 0.838 | 0.797 | 0.962 |
| E2 | BM25 only | 52 | 0.680 | 0.760 | 0.196 | 0.668 | 0.640 | 0.788 |
| E2x | BM25 + query expansion | 52 | 0.702 | 0.787 | 0.215 | 0.686 | 0.671 | 0.788 |
| E3 | hybrid (dense + BM25, RRF) — current app | 52 | 0.797 | 0.893 | 0.246 | 0.792 | 0.783 | 0.865 |
| E5 | hybrid + query expansion | 52 | 0.849 | 0.904 | 0.258 | 0.822 | 0.805 | 0.923 |

### standard Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 22 | 0.904 | 0.932 | 0.218 | 0.947 | 0.897 | 1.000 |
| E1x | dense only + query expansion | 22 | 0.904 | 0.927 | 0.218 | 0.947 | 0.895 | 1.000 |
| E2 | BM25 only | 22 | 0.816 | 0.868 | 0.209 | 0.811 | 0.786 | 0.864 |
| E2x | BM25 + query expansion | 22 | 0.827 | 0.868 | 0.218 | 0.818 | 0.805 | 0.864 |
| E3 | hybrid (dense + BM25, RRF) — current app | 22 | 0.871 | 0.911 | 0.227 | 0.894 | 0.863 | 0.909 |
| E5 | hybrid + query expansion | 22 | 0.871 | 0.911 | 0.227 | 0.894 | 0.863 | 0.909 |

### Egyptian Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 21 | 0.862 | 0.857 | 0.219 | 0.786 | 0.776 | 0.952 |
| E1x | dense only + query expansion | 21 | 0.852 | 0.852 | 0.209 | 0.782 | 0.755 | 0.952 |
| E2 | BM25 only | 21 | 0.640 | 0.695 | 0.171 | 0.523 | 0.539 | 0.714 |
| E2x | BM25 + query expansion | 21 | 0.700 | 0.767 | 0.191 | 0.587 | 0.600 | 0.762 |
| E3 | hybrid (dense + BM25, RRF) — current app | 21 | 0.724 | 0.902 | 0.181 | 0.699 | 0.713 | 0.809 |
| E5 | hybrid + query expansion | 21 | 0.864 | 0.941 | 0.219 | 0.774 | 0.782 | 0.952 |

### English

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 9 | 0.752 | 0.800 | 0.422 | 0.704 | 0.680 | 0.889 |
| E1x | dense only + query expansion | 9 | 0.752 | 0.741 | 0.422 | 0.704 | 0.653 | 0.889 |
| E2 | BM25 only | 9 | 0.441 | 0.646 | 0.222 | 0.657 | 0.517 | 0.778 |
| E2x | BM25 + query expansion | 9 | 0.402 | 0.639 | 0.267 | 0.594 | 0.513 | 0.667 |
| E3 | hybrid (dense + BM25, RRF) — current app | 9 | 0.789 | 0.828 | 0.444 | 0.759 | 0.749 | 0.889 |
| E5 | hybrid + query expansion | 9 | 0.761 | 0.800 | 0.422 | 0.759 | 0.717 | 0.889 |

### cross-language

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 4 | 0.875 | 0.875 | 0.250 | 0.833 | 0.771 | 1.000 |
| E1x | dense only + query expansion | 4 | 0.875 | 0.875 | 0.250 | 0.833 | 0.771 | 1.000 |
| E2 | BM25 only | 4 | 0.312 | 0.938 | 0.100 | 0.271 | 0.445 | 0.500 |
| E2x | BM25 + query expansion | 4 | 0.062 | 0.438 | 0.050 | 0.092 | 0.187 | 0.250 |
| E3 | hybrid (dense + BM25, RRF) — current app | 4 | 0.938 | 0.938 | 0.300 | 0.833 | 0.826 | 1.000 |
| E5 | hybrid + query expansion | 4 | 0.875 | 0.875 | 0.250 | 0.833 | 0.784 | 1.000 |

## Validation split

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 81 | 0.840 | 0.848 | 0.242 | 0.839 | 0.794 | 0.926 |
| E1x | dense only + query expansion | 81 | 0.834 | 0.844 | 0.242 | 0.821 | 0.782 | 0.914 |
| E2 | BM25 only | 81 | 0.630 | 0.768 | 0.178 | 0.538 | 0.573 | 0.691 |
| E2x | BM25 + query expansion | 81 | 0.725 | 0.773 | 0.215 | 0.565 | 0.612 | 0.765 |
| E3 | hybrid (dense + BM25, RRF) — current app | 81 | 0.796 | 0.856 | 0.230 | 0.744 | 0.737 | 0.889 |
| E5 | hybrid + query expansion | 81 | 0.847 | 0.881 | 0.247 | 0.776 | 0.775 | 0.926 |

## Latency per question and experiments not run

| experiment | status | mean_ms | p50_ms | p95_ms | reason |
|---|---|---|---|---|---|
| E1 | OK | 248.000 | 238.900 | 392.600 |  |
| E1x | OK | 110.100 | 120.500 | 138.800 |  |
| E2 | OK | 0.700 | 0.600 | 1.000 |  |
| E2x | OK | 1.600 | 1.700 | 2.200 |  |
| E3 | OK | 81.000 | 87.100 | 101.500 |  |
| E4 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E5 | OK | 124.200 | 136.400 | 158.400 |  |
| E6 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E7 | NOT RUN |  |  |  | no fine-tuned model/collection (scripts/finetune_embeddings.py, scripts/build_collection.py) |

## Tuning

```json
{
 "split": "validation",
 "questions": 81,
 "grids": {
  "bm25": [
   {
    "k1": 0.9,
    "b": 0.4,
    "questions": 81,
    "hit@5": 0.7778,
    "hit@6": 0.7778,
    "recall@5": 0.6986,
    "recall@10": 0.7677,
    "precision@5": 0.1951,
    "mrr@10": 0.4952,
    "ndcg@10": 0.539
   },
   {
    "k1": 0.9,
    "b": 0.6,
    "questions": 81,
    "hit@5": 0.7654,
    "hit@6": 0.7654,
    "recall@5": 0.6862,
    "recall@10": 0.7739,
    "precision@5": 0.1926,
    "mrr@10": 0.5187,
    "ndcg@10": 0.5582
   },
   {
    "k1": 0.9,
    "b": 0.75,
    "questions": 81,
    "hit@5": 0.7778,
    "hit@6": 0.7901,
    "recall@5": 0.6924,
    "recall@10": 0.7739,
    "precision@5": 0.1951,
    "mrr@10": 0.5315,
    "ndcg@10": 0.5645
   },
   {
    "k1": 0.9,
    "b": 0.9,
    "questions": 81,
    "hit@5": 0.7407,
    "hit@6": 0.7654,
    "recall@5": 0.6698,
    "recall@10": 0.7677,
    "precision@5": 0.1877,
    "mrr@10": 0.5326,
    "ndcg@10": 0.5686
   },
   {
    "k1": 1.2,
    "b": 0.4,
    "questions": 81,
    "hit@5": 0.7778,
    "hit@6": 0.7778,
    "recall@5": 0.7016,
    "recall@10": 0.7677,
    "precision@5": 0.1975,
    "mrr@10": 0.5215,
    "ndcg@10": 0.5598
   },
   {
    "k1": 1.2,
    "b": 0.6,
    "questions": 81,
    "hit@5": 0.7531,
    "hit@6": 0.7531,
    "recall@5": 0.6739,
    "recall@10": 0.7646,
    "precision@5": 0.1901,
    "mrr@10": 0.5364,
    "ndcg@10": 0.5687
   },
   {
    "k1": 1.2,
    "b": 0.75,
    "questions": 81,
    "hit@5": 0.7284,
    "hit@6": 0.7531,
    "recall@5": 0.6574,
    "recall@10": 0.7646,
    "precision@5": 0.1852,
    "mrr@10": 0.5303,
    "ndcg@10": 0.5655
   },
   {
    "k1": 1.2,
    "b": 0.9,
    "questions": 81,
    "hit@5": 0.716,
    "hit@6": 0.7284,
    "recall@5": 0.642,
    "recall@10": 0.7615,
    "precision@5": 0.1802,
    "mrr@10": 0.5467,
    "ndcg@10": 0.5774
   },
   {
    "k1": 1.5,
    "b": 0.4,
    "questions": 81,
    "hit@5": 0.7654,
    "hit@6": 0.7654,
    "recall@5": 0.6893,
    "recall@10": 0.7646,
    "precision@5": 0.1951,
    "mrr@10": 0.5336,
    "ndcg@10": 0.5676
   },
   {
    "k1": 1.5,
    "b": 0.6,
    "questions": 81,
    "hit@5": 0.7407,
    "hit@6": 0.7407,
    "recall@5": 0.6615,
    "recall@10": 0.7584,
    "precision@5": 0.1877,
    "mrr@10": 0.5291,
    "ndcg@10": 0.5645
   },
   {
    "k1": 1.5,
    "b": 0.75,
    "questions": 81,
    "hit@5": 0.6914,
    "hit@6": 0.716,
    "recall@5": 0.6296,
    "recall@10": 0.7677,
    "precision@5": 0.1778,
    "mrr@10": 0.5377,
    "ndcg@10": 0.5726
   },
   {
    "k1": 1.5,
    "b": 0.9,
    "questions": 81,
    "hit@5": 0.6914,
    "hit@6": 0.7037,
    "recall@5": 0.6265,
    "recall@10": 0.7553,
    "precision@5": 0.1753,
    "mrr@10": 0.5511,
    "ndcg@10": 0.5767
   },
   {
    "k1": 2.0,
    "b": 0.4,
    "questions": 81,
    "hit@5": 0.7654,
    "hit@6": 0.7654,
    "recall@5": 0.6831,
    "recall@10": 0.7739,
    "precision@5": 0.1951,
    "mrr@10": 0.5349,
    "ndcg@10": 0.5708
   },
   {
    "k1": 2.0,
    "b": 0.6,
    "questions": 81,
    "hit@5": 0.7037,
    "hit@6": 0.716,
    "recall@5": 0.642,
    "recall@10": 0.7615,
    "precision@5": 0.1802,
    "mrr@10": 0.5259,
    "ndcg@10": 0.5639
   },
   {
    "k1": 2.0,
    "b": 0.75,
    "questions": 81,
    "hit@5": 0.6667,
    "hit@6": 0.679,
    "recall@5": 0.6049,
    "recall@10": 0.7615,
    "precision@5": 0.1728,
    "mrr@10": 0.5346,
    "ndcg@10": 0.5671
   },
   {
    "k1": 2.0,
    "b": 0.9,
    "questions": 81,
    "hit@5": 0.642,
    "hit@6": 0.6667,
    "recall@5": 0.5833,
    "recall@10": 0.7183,
    "precision@5": 0.1654,
    "mrr@10": 0.5407,
    "ndcg@10": 0.5597
   }
  ],
  "rrf": [
   {
    "rrf_k": 10,
    "candidates": 20,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8617,
    "precision@5": 0.2296,
    "mrr@10": 0.7437,
    "ndcg@10": 0.7387
   },
   {
    "rrf_k": 10,
    "candidates": 30,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8556,
    "precision@5": 0.2296,
    "mrr@10": 0.7437,
    "ndcg@10": 0.7365
   },
   {
    "rrf_k": 10,
    "candidates": 50,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8679,
    "precision@5": 0.2296,
    "mrr@10": 0.7449,
    "ndcg@10": 0.7401
   },
   {
    "rrf_k": 30,
    "candidates": 20,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8556,
    "precision@5": 0.2296,
    "mrr@10": 0.7435,
    "ndcg@10": 0.7366
   },
   {
    "rrf_k": 30,
    "candidates": 30,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8556,
    "precision@5": 0.2296,
    "mrr@10": 0.7435,
    "ndcg@10": 0.7366
   },
   {
    "rrf_k": 30,
    "candidates": 50,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8556,
    "precision@5": 0.2296,
    "mrr@10": 0.7435,
    "ndcg@10": 0.7366
   },
   {
    "rrf_k": 60,
    "candidates": 20,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8556,
    "precision@5": 0.2296,
    "mrr@10": 0.7437,
    "ndcg@10": 0.737
   },
   {
    "rrf_k": 60,
    "candidates": 30,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8556,
    "precision@5": 0.2296,
    "mrr@10": 0.7437,
    "ndcg@10": 0.737
   },
   {
    "rrf_k": 60,
    "candidates": 50,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8556,
    "precision@5": 0.2296,
    "mrr@10": 0.7437,
    "ndcg@10": 0.737
   },
   {
    "rrf_k": 100,
    "candidates": 20,
    "questions": 81,
    "hit@5": 0.8889,
    "hit@6": 0.9012,
    "recall@5": 0.7965,
    "recall@10": 0.8543,
    "precision@5": 0.2296,
    "mrr@10": 0.7437,
    "ndcg@10": 0.7361
   },
   {
    "rrf_k": 100,
  
```
