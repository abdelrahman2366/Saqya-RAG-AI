# SAQYA evaluation 20261117-005606

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
| E5 | hybrid + query expansion | 52 | 0.830 | 0.902 | 0.254 | 0.822 | 0.803 | 0.904 |

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
| E3 | hybrid (dense + BM25, RRF) — current app | 21 | 0.724 | 0.902 | 0.181 | 0.699 | 0.712 | 0.809 |
| E5 | hybrid + query expansion | 21 | 0.817 | 0.936 | 0.209 | 0.772 | 0.778 | 0.905 |

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
| E3 | hybrid (dense + BM25, RRF) — current app | 81 | 0.803 | 0.856 | 0.232 | 0.744 | 0.737 | 0.889 |
| E5 | hybrid + query expansion | 81 | 0.847 | 0.881 | 0.247 | 0.776 | 0.775 | 0.926 |

## Latency per question and experiments not run

| experiment | status | mean_ms | p50_ms | p95_ms | reason |
|---|---|---|---|---|---|
| E1 | OK | 370.700 | 349.300 | 633.300 |  |
| E1x | OK | 135.600 | 131.200 | 207.900 |  |
| E2 | OK | 0.700 | 0.700 | 1.100 |  |
| E2x | OK | 1.700 | 1.600 | 2.200 |  |
| E3 | OK | 80.000 | 86.300 | 96.100 |  |
| E4 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E5 | OK | 121.100 | 130.100 | 145.700 |  |
| E6 | NOT RUN |  |  |  | reranker not requested (--reranker) |
| E7 | NOT RUN |  |  |  | no fine-tuned model/collection (scripts/finetune_embeddings.py, scripts/build_collection.py) |
