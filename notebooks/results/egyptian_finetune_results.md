# Egyptian Arabic fine-tune of bge-m3

Distillation, 11641 examples, 1456 steps, top 6 layers, lr 2e-05, cuda. Embeddings-only search over 9341 passages.

| test set | questions | hit@6 before | hit@6 after | MRR@10 before | MRR@10 after |
|---|---:|---:|---:|---:|---:|
| Egyptian, as typed | 55 | 80% | 80% | 0.663 | 0.665 |
| Egyptian, after the app's dialect rewrite | 55 | 96% | 89% | 0.800 | 0.718 |
| Standard Arabic (original questions) | 57 | 98% | 96% | 0.901 | 0.867 |
| English | 21 | 81% | 71% | 0.599 | 0.613 |
