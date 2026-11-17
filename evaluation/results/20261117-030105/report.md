# SAQYA evaluation 20261117-030105

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
| E3 | hybrid (dense + BM25, RRF) — current app | 52 | 0.839 | 0.916 | 0.254 | 0.820 | 0.808 | 0.923 |
| E5 | hybrid + query expansion | 52 | 0.862 | 0.906 | 0.258 | 0.849 | 0.828 | 0.942 |

### standard Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 22 | 0.904 | 0.932 | 0.218 | 0.947 | 0.897 | 1.000 |
| E1x | dense only + query expansion | 22 | 0.904 | 0.927 | 0.218 | 0.947 | 0.895 | 1.000 |
| E2 | BM25 only | 22 | 0.816 | 0.868 | 0.209 | 0.811 | 0.786 | 0.864 |
| E2x | BM25 + query expansion | 22 | 0.827 | 0.868 | 0.218 | 0.818 | 0.805 | 0.864 |
| E3 | hybrid (dense + BM25, RRF) — current app | 22 | 0.893 | 0.916 | 0.227 | 0.924 | 0.888 | 0.955 |
| E5 | hybrid + query expansion | 22 | 0.916 | 0.916 | 0.236 | 0.924 | 0.888 | 0.955 |

### Egyptian Arabic

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 21 | 0.862 | 0.857 | 0.219 | 0.786 | 0.776 | 0.952 |
| E1x | dense only + query expansion | 21 | 0.852 | 0.852 | 0.209 | 0.782 | 0.755 | 0.952 |
| E2 | BM25 only | 21 | 0.640 | 0.695 | 0.171 | 0.523 | 0.539 | 0.714 |
| E2x | BM25 + query expansion | 21 | 0.700 | 0.767 | 0.191 | 0.587 | 0.600 | 0.762 |
| E3 | hybrid (dense + BM25, RRF) — current app | 21 | 0.795 | 0.955 | 0.191 | 0.741 | 0.755 | 0.905 |
| E5 | hybrid + query expansion | 21 | 0.864 | 0.941 | 0.219 | 0.814 | 0.812 | 0.952 |

### English

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 9 | 0.752 | 0.800 | 0.422 | 0.704 | 0.680 | 0.889 |
| E1x | dense only + query expansion | 9 | 0.752 | 0.741 | 0.422 | 0.704 | 0.653 | 0.889 |
| E2 | BM25 only | 9 | 0.441 | 0.646 | 0.222 | 0.657 | 0.517 | 0.778 |
| E2x | BM25 + query expansion | 9 | 0.402 | 0.639 | 0.267 | 0.594 | 0.513 | 0.667 |
| E3 | hybrid (dense + BM25, RRF) — current app | 9 | 0.811 | 0.828 | 0.467 | 0.750 | 0.734 | 0.889 |
| E5 | hybrid + query expansion | 9 | 0.724 | 0.800 | 0.400 | 0.750 | 0.715 | 0.889 |

### cross-language

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 4 | 0.875 | 0.875 | 0.250 | 0.833 | 0.771 | 1.000 |
| E1x | dense only + query expansion | 4 | 0.875 | 0.875 | 0.250 | 0.833 | 0.771 | 1.000 |
| E2 | BM25 only | 4 | 0.312 | 0.938 | 0.100 | 0.271 | 0.445 | 0.500 |
| E2x | BM25 + query expansion | 4 | 0.062 | 0.438 | 0.050 | 0.092 | 0.187 | 0.250 |
| E3 | hybrid (dense + BM25, RRF) — current app | 4 | 0.938 | 0.938 | 0.300 | 0.812 | 0.809 | 1.000 |
| E5 | hybrid + query expansion | 4 | 0.875 | 0.875 | 0.250 | 0.812 | 0.767 | 1.000 |

## Validation split

| experiment | name | questions | recall@5 | recall@10 | precision@5 | mrr@10 | ndcg@10 | hit@5 |
|---|---|---|---|---|---|---|---|---|
| E1 | dense only | 81 | 0.840 | 0.848 | 0.242 | 0.839 | 0.794 | 0.926 |
| E1x | dense only + query expansion | 81 | 0.834 | 0.844 | 0.242 | 0.821 | 0.782 | 0.914 |
| E2 | BM25 only | 81 | 0.630 | 0.768 | 0.178 | 0.538 | 0.573 | 0.691 |
| E2x | BM25 + query expansion | 81 | 0.725 | 0.773 | 0.215 | 0.565 | 0.612 | 0.765 |
| E3 | hybrid (dense + BM25, RRF) — current app | 81 | 0.800 | 0.849 | 0.232 | 0.793 | 0.766 | 0.889 |
| E5 | hybrid + query expansion | 81 | 0.844 | 0.880 | 0.247 | 0.812 | 0.798 | 0.926 |

## Latency per question and experiments not run

| experiment | status | mean_ms | p50_ms | p95_ms | reason |
|---|---|---|---|---|---|
| E1 | OK | 261.700 | 247.800 | 400.100 |  |
| E1x | OK | 107.900 | 119.100 | 134.400 |  |
| E2 | OK | 0.700 | 0.600 | 1.000 |  |
| E2x | OK | 1.600 | 1.600 | 2.100 |  |
| E3 | OK | 78.200 | 84.300 | 93.300 |  |
| E4 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E5 | OK | 112.300 | 123.900 | 139.100 |  |
| E6 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E7 | NOT RUN |  |  |  | no fine-tuned model/collection (scripts/finetune_embeddings.py, scripts/build_collection.py) |

## Calibration

```json
{
 "status": "OK",
 "fitted_on": "validation (84 questions, 76 with the answer retrieved)",
 "pipeline": {
  "reranker": "BAAI/bge-reranker-v2-m3",
  "query_expansion": true,
  "keyword_weight": 0.5,
  "keyword_seat": false
 },
 "test_questions": 59,
 "brier": 0.0184,
 "ece": 0.0212,
 "auroc": 0.9902,
 "answer_at": 0.0135,
 "abstain_at": 0.0135,
 "at_abstain_threshold": {
  "threshold": 0.013,
  "coverage": 0.9492,
  "selective_accuracy": 0.9107,
  "wrong_answered": 5,
  "right_refused": 0
 },
 "at_answer_threshold": {
  "threshold": 0.013,
  "coverage": 0.9492,
  "selective_accuracy": 0.9107,
  "wrong_answered": 5,
  "right_refused": 0
 },
 "baseline_gate": {
  "gate": 0.591,
  "threshold": 0.5,
  "coverage": 0.8475,
  "selective_accuracy": 0.98,
  "wrong_answered": 1,
  "right_refused": 2,
  "brier_of_0_1_rule": 0.0508
 },
 "baseline_cosine_auroc": 0.973,
 "saved_to": "D:\\agri-rag-assistant\\backend\\data\\confidence_calibration.json"
}
```
