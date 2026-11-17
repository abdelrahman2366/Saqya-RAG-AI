# Chunking comparison (BM25 only)

52 documents, 2826 pages, 52 test / 81 validation questions

| strategy | chunks | mean_chars | context chars@5 | test hit@5 | test recall@5 | test recall@10 | test mrr@10 | test ndcg@10 | validation mrr@10 |
|---|---|---|---|---|---|---|---|---|---|
| production | 9264 | 630 | 2909 | 0.788 | 0.721 | 0.764 | 0.679 | 0.669 | 0.555 |
| sentences | 7984 | 708 | 3486 | 0.846 | 0.758 | 0.810 | 0.697 | 0.691 | 0.523 |
| fixed | 6386 | 797 | 3983 | 0.885 | 0.805 | 0.831 | 0.735 | 0.725 | 0.524 |
| recursive | 9203 | 593 | 2927 | 0.846 | 0.761 | 0.806 | 0.704 | 0.691 | 0.459 |
| children only | 23803 | 302 | 1348 | 0.769 | 0.691 | 0.726 | 0.635 | 0.632 | 0.468 |
| parent-child | 23803 | 302 | 3148 | 0.827 | 0.759 | 0.782 | 0.673 | 0.680 | 0.584 |

- semantic: NOT RUN: needs --embedding-model