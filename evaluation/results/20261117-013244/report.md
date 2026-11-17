# SAQYA evaluation 20261117-013244

- Questions: 133 answerable (84 validation / 59 test, split seed 13); dataset 2d1ba045dd9d
- Passages: 8872 (D:\agri-rag-assistant\backend\data\vector_store)
- Embedding model: BAAI/bge-m3
- Search text: spelling repair + dialect rewrite
- Questions with no passage matching their label (excluded): 0

## Test split (settings were not chosen on these questions)

### all

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 52 | 0.879 | 0.902 | 0.258 | 0.875 | 0.847 | 0.981 |
| E1x | dense only + query expansion | 52 | 0.810 | 0.877 | 0.235 | 0.814 | 0.780 | 0.923 |
| E2 | BM25 only | 52 | 0.706 | 0.762 | 0.219 | 0.683 | 0.668 | 0.788 |
| E2x | BM25 + query expansion | 52 | 0.702 | 0.782 | 0.215 | 0.691 | 0.672 | 0.788 |
| E3 | hybrid (dense + BM25, RRF) — current app | 52 | 0.835 | 0.918 | 0.254 | 0.824 | 0.817 | 0.904 |
| E5 | hybrid + query expansion | 52 | 0.835 | 0.894 | 0.258 | 0.801 | 0.792 | 0.904 |

### standard Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 22 | 0.904 | 0.943 | 0.218 | 0.970 | 0.920 | 1.000 |
| E1x | dense only + query expansion | 22 | 0.904 | 0.923 | 0.218 | 0.947 | 0.892 | 1.000 |
| E2 | BM25 only | 22 | 0.827 | 0.868 | 0.218 | 0.823 | 0.808 | 0.864 |
| E2x | BM25 + query expansion | 22 | 0.827 | 0.868 | 0.218 | 0.823 | 0.808 | 0.864 |
| E3 | hybrid (dense + BM25, RRF) — current app | 22 | 0.916 | 0.923 | 0.236 | 0.873 | 0.860 | 0.955 |
| E5 | hybrid + query expansion | 22 | 0.927 | 0.923 | 0.245 | 0.873 | 0.857 | 0.955 |

### Egyptian Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 21 | 0.909 | 0.905 | 0.229 | 0.841 | 0.830 | 1.000 |
| E1x | dense only + query expansion | 21 | 0.757 | 0.900 | 0.191 | 0.761 | 0.747 | 0.857 |
| E2 | BM25 only | 21 | 0.700 | 0.719 | 0.191 | 0.573 | 0.592 | 0.762 |
| E2x | BM25 + query expansion | 21 | 0.700 | 0.767 | 0.191 | 0.594 | 0.605 | 0.762 |
| E3 | hybrid (dense + BM25, RRF) — current app | 21 | 0.841 | 0.976 | 0.219 | 0.812 | 0.831 | 0.905 |
| E5 | hybrid + query expansion | 21 | 0.817 | 0.952 | 0.209 | 0.784 | 0.797 | 0.905 |

### English

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 9 | 0.746 | 0.794 | 0.422 | 0.722 | 0.710 | 0.889 |
| E1x | dense only + query expansion | 9 | 0.702 | 0.713 | 0.378 | 0.611 | 0.585 | 0.889 |
| E2 | BM25 only | 9 | 0.424 | 0.606 | 0.289 | 0.599 | 0.504 | 0.667 |
| E2x | BM25 + query expansion | 9 | 0.402 | 0.606 | 0.267 | 0.594 | 0.499 | 0.667 |
| E3 | hybrid (dense + BM25, RRF) — current app | 9 | 0.622 | 0.772 | 0.378 | 0.733 | 0.681 | 0.778 |
| E5 | hybrid + query expansion | 9 | 0.650 | 0.689 | 0.400 | 0.667 | 0.622 | 0.778 |

### cross-language

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 4 | 0.812 | 0.812 | 0.200 | 0.875 | 0.812 | 1.000 |
| E1x | dense only + query expansion | 4 | 0.812 | 0.812 | 0.200 | 0.625 | 0.627 | 1.000 |
| E2 | BM25 only | 4 | 0.062 | 0.062 | 0.050 | 0.062 | 0.042 | 0.250 |
| E2x | BM25 + query expansion | 4 | 0.062 | 0.062 | 0.050 | 0.050 | 0.038 | 0.250 |
| E3 | hybrid (dense + BM25, RRF) — current app | 4 | 0.562 | 0.562 | 0.150 | 0.625 | 0.505 | 0.750 |
| E5 | hybrid + query expansion | 4 | 0.625 | 0.625 | 0.200 | 0.500 | 0.475 | 0.750 |

## Validation split

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 81 | 0.841 | 0.865 | 0.244 | 0.826 | 0.795 | 0.926 |
| E1x | dense only + query expansion | 81 | 0.796 | 0.842 | 0.227 | 0.780 | 0.758 | 0.876 |
| E2 | BM25 only | 81 | 0.725 | 0.780 | 0.217 | 0.564 | 0.616 | 0.765 |
| E2x | BM25 + query expansion | 81 | 0.719 | 0.792 | 0.217 | 0.565 | 0.619 | 0.765 |
| E3 | hybrid (dense + BM25, RRF) — current app | 81 | 0.827 | 0.842 | 0.242 | 0.744 | 0.739 | 0.889 |
| E5 | hybrid + query expansion | 81 | 0.840 | 0.863 | 0.244 | 0.761 | 0.759 | 0.914 |

## Latency per question and experiments not run

| experiment | status | mean_ms | p50_ms | p95_ms | reason |
|---|---|---|---|---|---|
| E1 | OK | 43.800 | 42.600 | 52.000 |  |
| E1x | OK | 106.600 | 123.800 | 144.900 |  |
| E2 | OK | 0.300 | 0.300 | 0.500 |  |
| E2x | OK | 1.600 | 1.600 | 2.100 |  |
| E3 | OK | 41.500 | 45.100 | 50.900 |  |
| E4 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E5 | OK | 110.800 | 129.000 | 150.600 |  |
| E6 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E7 | NOT RUN |  |  |  | no fine-tuned model/collection (scripts/finetune_embeddings.py, scripts/build_collection.py) |

## Answers

```json
{
 "status": "OK",
 "llm": "qwen2.5:7b",
 "layers_on": [
  "none (original pipeline)"
 ],
 "summary": {
  "questions": 59,
  "failures": 0,
  "answer_correct (answerable, keyword rule)": 0.769,
  "out_of_scope_refused": 1.0,
  "grounded (answered)": 1.0,
  "citation_precision": 0.759,
  "context_recall": 0.846,
  "faithfulness (claims supported)": 0.99,
  "unsupported_claim_rate": 0.0,
  "latency_mean_ms": 37289.119,
  "tokens_mean": 1647.339
 }
}
```
