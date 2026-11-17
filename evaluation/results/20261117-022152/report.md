# SAQYA evaluation 20261117-022152

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
| E3 | hybrid (dense + BM25, RRF) — current app | 52 | 0.886 | 0.912 | 0.265 | 0.853 | 0.835 | 0.962 |
| E5 | hybrid + query expansion | 52 | 0.847 | 0.886 | 0.258 | 0.838 | 0.815 | 0.923 |

### standard Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 22 | 0.904 | 0.943 | 0.218 | 0.970 | 0.920 | 1.000 |
| E1x | dense only + query expansion | 22 | 0.904 | 0.923 | 0.218 | 0.947 | 0.892 | 1.000 |
| E2 | BM25 only | 22 | 0.827 | 0.868 | 0.218 | 0.823 | 0.808 | 0.864 |
| E2x | BM25 + query expansion | 22 | 0.827 | 0.868 | 0.218 | 0.823 | 0.808 | 0.864 |
| E3 | hybrid (dense + BM25, RRF) — current app | 22 | 0.916 | 0.920 | 0.236 | 0.901 | 0.882 | 0.955 |
| E5 | hybrid + query expansion | 22 | 0.916 | 0.916 | 0.236 | 0.901 | 0.872 | 0.955 |

### Egyptian Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 21 | 0.909 | 0.905 | 0.229 | 0.841 | 0.830 | 1.000 |
| E1x | dense only + query expansion | 21 | 0.757 | 0.900 | 0.191 | 0.761 | 0.747 | 0.857 |
| E2 | BM25 only | 21 | 0.700 | 0.719 | 0.191 | 0.573 | 0.592 | 0.762 |
| E2x | BM25 + query expansion | 21 | 0.700 | 0.767 | 0.191 | 0.594 | 0.605 | 0.762 |
| E3 | hybrid (dense + BM25, RRF) — current app | 21 | 0.912 | 0.964 | 0.229 | 0.843 | 0.846 | 1.000 |
| E5 | hybrid + query expansion | 21 | 0.864 | 0.941 | 0.219 | 0.845 | 0.837 | 0.952 |

### English

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 9 | 0.746 | 0.794 | 0.422 | 0.722 | 0.710 | 0.889 |
| E1x | dense only + query expansion | 9 | 0.702 | 0.713 | 0.378 | 0.611 | 0.585 | 0.889 |
| E2 | BM25 only | 9 | 0.424 | 0.606 | 0.289 | 0.599 | 0.504 | 0.667 |
| E2x | BM25 + query expansion | 9 | 0.402 | 0.606 | 0.267 | 0.594 | 0.499 | 0.667 |
| E3 | hybrid (dense + BM25, RRF) — current app | 9 | 0.756 | 0.772 | 0.422 | 0.759 | 0.695 | 0.889 |
| E5 | hybrid + query expansion | 9 | 0.635 | 0.689 | 0.400 | 0.667 | 0.627 | 0.778 |

### cross-language

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 4 | 0.812 | 0.812 | 0.200 | 0.875 | 0.812 | 1.000 |
| E1x | dense only + query expansion | 4 | 0.812 | 0.812 | 0.200 | 0.625 | 0.627 | 1.000 |
| E2 | BM25 only | 4 | 0.062 | 0.062 | 0.050 | 0.062 | 0.042 | 0.250 |
| E2x | BM25 + query expansion | 4 | 0.062 | 0.062 | 0.050 | 0.050 | 0.038 | 0.250 |
| E3 | hybrid (dense + BM25, RRF) — current app | 4 | 0.812 | 0.812 | 0.200 | 0.708 | 0.630 | 1.000 |
| E5 | hybrid + query expansion | 4 | 0.625 | 0.625 | 0.200 | 0.500 | 0.475 | 0.750 |

## Validation split

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 81 | 0.841 | 0.865 | 0.244 | 0.826 | 0.795 | 0.926 |
| E1x | dense only + query expansion | 81 | 0.796 | 0.842 | 0.227 | 0.780 | 0.758 | 0.876 |
| E2 | BM25 only | 81 | 0.725 | 0.780 | 0.217 | 0.564 | 0.616 | 0.765 |
| E2x | BM25 + query expansion | 81 | 0.719 | 0.792 | 0.217 | 0.565 | 0.619 | 0.765 |
| E3 | hybrid (dense + BM25, RRF) — current app | 81 | 0.827 | 0.855 | 0.242 | 0.775 | 0.764 | 0.889 |
| E5 | hybrid + query expansion | 81 | 0.833 | 0.872 | 0.242 | 0.790 | 0.781 | 0.914 |

## Latency per question and experiments not run

| experiment | status | mean_ms | p50_ms | p95_ms | reason |
|---|---|---|---|---|---|
| E1 | OK | 50.900 | 43.400 | 55.700 |  |
| E1x | OK | 105.200 | 120.000 | 143.500 |  |
| E2 | OK | 0.400 | 0.300 | 0.800 |  |
| E2x | OK | 1.700 | 1.700 | 2.400 |  |
| E3 | OK | 40.200 | 43.500 | 53.200 |  |
| E4 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E5 | OK | 109.600 | 125.200 | 148.500 |  |
| E6 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E7 | NOT RUN |  |  |  | no fine-tuned model/collection (scripts/finetune_embeddings.py, scripts/build_collection.py) |

## Answers

```json
{
 "status": "OK",
 "llm": "qwen2.5:7b",
 "layers_on": [
  "classify",
  "expansion",
  "reranker",
  "verify_claims"
 ],
 "summary": {
  "questions": 59,
  "failures": 0,
  "answer_correct (answerable, keyword rule)": 0.827,
  "out_of_scope_refused": 1.0,
  "grounded (answered)": 1.0,
  "citation_precision": 0.704,
  "context_recall": 0.923,
  "faithfulness (claims supported)": 0.996,
  "unsupported_claim_rate": 0.0,
  "latency_mean_ms": 36613.424,
  "tokens_mean": 1814.542
 }
}
```
