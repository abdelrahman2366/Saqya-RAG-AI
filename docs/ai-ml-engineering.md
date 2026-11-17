# SAQYA AI — AI/ML engineering report

What was audited, what was built, what was measured, and what still has to be run. Every number in this file
was measured; anything not measured says **NOT EVALUATED** and why. Measured outputs live in
`evaluation/results/`.

**Where things stand:** the improvements are implemented and tested. Each new layer is **off by default**, so
the app answers exactly as before until a setting turns it on. The keyword-side experiments ran on the real
library (8,872 passages, 133 answerable evaluation questions). The experiments that need bge-m3, a reranker or
the LLM could not run in the build environment (no GPU, no access to Hugging Face). They are one command each
on the project PC (section H).

---

## A. Initial assessment

### Architecture (as found)

```
question ─► spelling repair ─► Egyptian→MSA rewrite ─► crop-calendar route ─► follow-up handling
        ─► LLM translation (AR↔EN) ─► hybrid search: bge-m3 (Chroma, cosine) + BM25 ─► RRF (k=60, 30 per list)
           + keyword seat + per-crop seats ─► relevance gate (cosine 0.591) ─► focus passages
        ─► qwen2.5 (Ollama) ─► grounding checks / retry / repair ─► citations
        ─► (nothing found) learned knowledge ─► web search ─► general knowledge
```

| component | state | notes |
|---|---|---|
| Ingestion / cleaning | solid | no-AI cleaning, structured clean PDFs (`clean_rag_pdfs_no_ai.ipynb`, `extraction.py`) |
| Chunking | solid | already section-aware: 8,491 of 8,872 chunks carry their heading path; 800 chars / 150 overlap |
| Dense retrieval | correct | BAAI/bge-m3 via sentence-transformers, normalised vectors, Chroma cosine space, topic filter |
| BM25 | good | Arabic light10 stemming + corpus-aware preposition stripping, Snowball for English, glossary synonyms |
| Hybrid / RRF | good | per-phrasing RRF, "max" over phrasings, a reserved slot for the best keyword match, per-crop slots |
| Egyptian Arabic | rule-based | `dialect.to_msa` rewrite does most of the work (see below) |
| Translation | LLM + glossary | cross-language retrieval depends on it (64% in the last notebook run) |
| Grounding | strong, answer-level | citations, unsupported numbers, garbled text, retry, sentence removal |
| Reranking | missing | |
| Query understanding / expansion | missing | |
| Claim-level verification | missing | numbers were checked against *all* passages, not the one cited |
| Confidence | single threshold | cosine ≥ 0.591 or a named crop; calibrated on the same questions it is judged on |
| Evaluation | partial | notebook hit@6 + MRR on one question set, no validation/test split |
| Observability | logs only | no tracing, no stage timings |

### Main weaknesses

1. **No held-out evaluation.** The gate and the notebook's numbers come from the same 88 + 55 questions.
2. **Egyptian Arabic depends on rules.** bge-m3 finds an answer in the top 6 for 80% of Egyptian questions as
   typed, versus 96% after the rewrite (`notebooks/results/egyptian_finetune_results.md`). Anything the word
   lists miss falls back to that 80%.
3. **Cross-language retrieval rides on the LLM translation.** It was 64% (7/11) in the last notebook run, and
   keyword search alone gets 6% on the test cross-language questions (table D2).
4. **Hallucination surface.** A dose copied from the wrong crop's paragraph passed the number check, because
   numbers were looked up in all passages.
5. **The earlier fine-tune** was distillation with no passages and no negatives. Half its pairs taught the
   model to leave text unchanged, and evaluation questions were copied in (audit in C). It was correctly not
   kept: Egyptian-as-typed MRR 0.663 → 0.665, while the rewritten Egyptian, MSA and English questions all got
   worse.

### Preserved, not rewritten

Retriever, BM25, RRF, dialect rewrite, grounding, crop calendar, beyond-the-documents tiers, API contracts,
the frontend and the production chunking.

---

## B. Implemented improvements

All new layers are optional, configured in `backend/.env` (see `.env.example`), and fall back to the
original pipeline on any failure. With every flag off, `RAGService` behaves exactly as before. This is
covered by the original 252 tests plus `test_layers_off_change_nothing`.

### B1. Evaluation framework (`app/evaluation/`, `scripts/evaluate.py`)
- Uses the existing labels: expected document + expected phrase, the notebooks' rule, which doesn't depend
  on chunking.
- Label audit: all 133 answerable questions have at least one matching passage (0 excluded).
- **Fixed validation/test split by question group.** An Egyptian question and its MSA twin always land in the
  same half: 84 validation / 59 test questions (81 / 52 answerable), seed 13. All tuning uses validation;
  test is reported only.
- Experiments E1–E7 plus E2x (BM25 + expansion). All of them call the production search code.
- Metrics: Recall@5/10, Precision@5, MRR@10, nDCG@10, hit@5/6, latency (mean/p50/p95), failures, memory.
  Results are broken down by standard Arabic / Egyptian / English / cross-language.
- Outputs: `results.json` (models, versions, settings, dataset hash, seed, date, runtime), `metrics.csv`,
  `per_question.csv`, `report.md`.
- `--tune`: grids for BM25 k1/b and RRF k / candidates, on validation only.
- `--calibrate`: fits the confidence model (B7).
- `--answers`: full LLM answers on the test questions, with these metrics:
  - correctness: the existing answer-keyword rule;
  - groundedness;
  - citation precision;
  - context recall;
  - faithfulness: share of claims supported;
  - unsupported-claim rate;
  - out-of-scope refusal;
  - tokens and latency.

  It also writes `human_review.csv` (every third answer, with empty columns for a person to fill in).
- RAGAS was considered and not used: it needs an LLM judge, usually a paid API, and the instructions ask not
  to rely only on LLM evaluation. The metrics above are deterministic.

### B2. Configurable fusion (`retrieval.py`, `config.py`)
- `RRF_K`, `RETRIEVAL_CANDIDATES`, `BM25_K1` and `BM25_B` are now settings. Defaults are unchanged
  (60 / 30 / 1.5 / 0.75).
- Empty queries return an empty result safely.
- `search(..., weights=)` lets lower-weight phrasings (expansions) add evidence without competing. A passage
  scores its best score for the farmer's own wording plus `weight × ` its best score for an expansion.
- An expansion can never open the relevance gate or create crop slots. This was a real design bug found by
  measurement: with plain multiplicative weights under "max" fusion, an expansion could never change the
  ranking at all.

### B3. Cross-encoder reranking (`app/services/reranker.py`)
- Runs after RRF: the search returns `RERANKER_CANDIDATES` (30) passages and the cross-encoder keeps `top_k`.
  Each passage is scored against the phrasing in its own language.
- Loaded lazily on first use, once, behind a lock. It is never downloaded unless asked
  (`scripts/download_models.py reranker` or `RERANKER_ALLOW_DOWNLOAD=true`).
- Runs on CPU or GPU (`auto`), with a configurable batch size and maximum length.
- Graceful failure: if the model is missing, runs out of memory or a batch fails, the search's own order is
  kept and after a failed load it isn't retried.
- Metadata, pages and cosine similarity are preserved. The gate and `focus_passages` stay calibrated on
  cosine, and `rerank_score` is added alongside it.
- Model choice: `BAAI/bge-reranker-v2-m3` by default (568M parameters, Apache-2.0, multilingual, same family
  as the embedding model). `jinaai/jina-reranker-v2-base-multilingual` is supported (278M, faster) but is
  **CC-BY-NC-4.0, non-commercial**, needs `trust_remote_code`, and its model card does not mention Arabic.
  Neither is installed automatically.

### B4. Query understanding (`app/services/query_understanding.py`)
- Rule-based structured extraction, about 0.5 ms per question, with no model:
  - language and dialect;
  - crops (glossary + farmers' names after the dialect rewrite + short forms such as قمح/قطن/نخلة) and crop
    group;
  - **multi-label categories** (17, using stems taken from the library's own section headings);
  - intent (compare / diagnose / recommend / when / how much / how to / define / list);
  - pests or diseases **only if named**;
  - **symptoms as observed** (yellowing, wilting, rot…, never a diagnosis);
  - conditions (soil, region, governorate, month, growth stage, irrigation system);
  - quantities with units;
  - `missing` (blocking, e.g. no crop) and `would_help` (non-blocking, e.g. growth stage, irrigation, where
    the symptom shows).
- Example: «الطماطم عندي ورقها بيصفر ومش عارف السبب» gives crop tomato, symptom yellowing, intent diagnose,
  Egyptian, no disease, would help: growth stage / irrigation / where it shows / recent fertiliser or spraying.
- An LLM extractor was not added. The rules are deterministic and testable, and qwen2.5:3b/7b JSON extraction
  would add a model call to every question.
- Returned by the API when `QUERY_CLASSIFICATION_ENABLED=true`. It feeds expansion and confidence.

### B5. Agricultural classification
- The categories above are multi-label rules, **not a trained classifier**.
- The project has no category labels, and labels were not invented. Heading-derived labels were considered,
  but a passage's heading is not the question's intent.
- Used for routing expansion vocabulary and for the "ask for the crop" decision. Kept optional.

### B6. Query expansion (`app/services/query_expansion.py`)
- No LLM. It adds the bulletins' own heading words for the question's main category, narrowed by intent:
  "when" adds ميعاد الزراعة, "how much" adds معدل التقاوي.
- It also adds the glossary's fixed translations of the terms found, when there is no LLM translation.
- Safety: it never introduces a crop, pest, disease or number the farmer didn't write (parametrised test).
  At most 2 phrasings and 4 added terms. Any failure means no expansion.
- The farmer's own wording keeps full weight. Expansions add evidence at weight 0.25, chosen on validation.
- Off by default (`QUERY_EXPANSION_ENABLED`). Measured in D2.

### B7. Claim-level grounding (`app/services/claims.py`)
- Pipeline: claims are extracted (sentences and list items; refusals, questions and the safety line skipped),
  then matched to the cited passage (or the best-overlapping one), then given a verdict:
  - **supported / partial / unsupported / contradicted / unverified**;
  - numbers must be in the **same** passage as the claim, otherwise partial ("right fact, wrong source");
  - a number in no passage means unsupported;
  - a per-feddan vs per-hectare unit mismatch means partial;
  - a cross-language claim is unverified by words, but its numbers are still checked.
- Optional multilingual NLI (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`: MIT, NLI training
  covers 27 languages including Arabic). It is lazy, never auto-downloaded, and can only add evidence:
  entailment upgrades a claim, strong contradiction marks it contradicted, and it can never rescue a missing
  number. The lexical and numeric checks always run, so verification never depends on one model.
- The existing `grounding.py` repairs still decide the answer text. Verdicts are attached to the result and
  feed confidence.

### B8. Confidence and abstention (`app/services/confidence.py`)
- Signals:
  - best and mean cosine;
  - margin over the gate;
  - share of the question's words found in the passages;
  - crop match;
  - top reranker score;
  - missing information;
  - claim verdicts;
  - retry.
- An L2-regularised logistic model is fitted **only on validation** by `evaluate.py --calibrate`, which also
  picks thresholds on validation: `answer_at` (selective accuracy ≥ target) and `abstain_at` (best
  right-vs-wrong separation). Test metrics are recorded in the calibration file.
- **No hand-picked thresholds.** Without the file there is no probability. Decisions then use only structural
  facts:
  - **ask**: nothing answered and the crop or symptoms are missing, so the refusal becomes a clarifying
    question;
  - **follow-up**: an answer was given (often general knowledge) to a question that describes a symptom but
    names no crop («الزرع عندي بيموت»). The answer stays and ends with one question asking for the crop, what
    is seen on the plant and the irrigation. It never names a disease. It never triggers on general questions
    ("why is seed treated?"): none of the 143 evaluation questions trigger it, so the measured metrics are
    unaffected, and the follow-up is a question, so the claim checks ignore it;
  - **qualify**: a claim is unsupported or contradicted, so a caution line is added under the answer;
  - **abstain**: nothing answered.

  With calibration, a probability below `abstain_at` treats the library as having found nothing, and the
  other sources are tried as before.

### B9. Embedding fine-tuning pipeline (`scripts/build_finetune_data.py`, `finetune_embeddings.py`, `build_collection.py`)
- New contrastive data, with (query, passage, hard negative) triples from the library's section headings.
- Training objective: MultipleNegativesRankingLoss (in-batch negatives + one BM25 hard negative). It trains
  exactly the SentenceTransformer forward pass the app uses for bge-m3 (XLM-R → CLS → normalise; the
  sparse/ColBERT heads are unused). The script checks this before training.
- Top-6-layer training, fp16 on GPU, seeds fixed.
- Model selection on **validation** questions only; saved only if better, to `models/bge-m3-saqya/`, never
  over the original.
- `build_collection.py` embeds the same passages, ids and metadata into a **separate** store
  (`backend/data/vector_store_finetuned/`).
- `FINE_TUNED_EMBEDDINGS_ENABLED=true` switches to it, falling back to the original store if anything is
  missing.

### B10. Chunking comparison (`scripts/compare_chunking.py`)
- Strategies: production (section-aware), sentence packing, fixed windows, recursive, parent-child (300-char
  children → production parents) and semantic (needs an embedder).
- All built from the same cleaned pages, scored with the same rule, with the context cost shown.
- Never touches the store.

### B12. Plant-disease detection extension point (`app/services/vision.py`, `app/api/routes/vision.py`)
- `POST /vision/diagnose` checks the photo, asks a `DiseaseDetector` for its top labels, parses them into crop /
  disease / healthy, and, only when the top label is a disease above `VISION_MIN_CONFIDENCE`, asks the RAG pipeline
  for cited treatment advice. Below the threshold it asks for a clearer photo. A prediction is never presented as
  a confirmed diagnosis.
- **No model is trained or shipped. NOT EVALUATED.** `HFImageClassifier` loads any Hugging Face
  image-classification model from `models/plant-disease/` lazily, from disk only. Off by default
  (`VISION_ENABLED`). The 0.7 threshold is a placeholder to be chosen on validation photos.
- Handoff: `docs/computer-vision-handoff.md`.

### B11. Observability (`app/core/tracing.py`)
- Covered in section F.

---

## C. Model details

| model | role | status in this work |
|---|---|---|
| BAAI/bge-m3 | embeddings (production) | used, unchanged; **not evaluated here** (not downloadable in the build environment) |
| qwen2.5:3b / 7b (Ollama) | answers, translation | used, unchanged; not evaluated here (no Ollama in the build environment) |
| BAAI/bge-reranker-v2-m3 | reranker (default) | integrated, configurable; **considered, not downloaded, NOT EVALUATED** |
| jinaai/jina-reranker-v2-base-multilingual | reranker (alternative) | supported by config; **considered only** (non-commercial licence) |
| MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | claim NLI | integrated, optional; **considered, not downloaded, NOT EVALUATED** |
| intfloat/multilingual-e5, Jina embeddings | alternative embeddings | **considered only**: comparing means embedding 8,872 passages per model; `build_collection.py --model X` + `evaluate.py --finetuned-store` does it without touching the production store |
| models/bge-m3-saqya | fine-tuned embeddings | **NOT TRAINED**. The pipeline is complete and was smoke-tested end to end on a tiny random 4-layer model (code path only, meaningless numbers, not saved in the project). Training needs the GPU PC. |

**Earlier fine-tune audit** (`notebooks/results/egyptian_training_pairs.jsonl`):
- 11,641 rows, all distillation pairs, no passages and no negatives.
- 5,952 rows (51%) are identical student = teacher pairs.
- 8 rows contain an evaluation question (leakage).
- Objective: match the base model's vector of the MSA twin.

**New training set** (`data/finetune/report.json`, seed 13):
- 8,774 rows: 2,599 MSA, 2,940 Egyptian, 2,316 Egyptian-rewritten, 919 English.
- 99.6% have a hard negative.
- 215 passages that answer an evaluation question were excluded.
- 35 queries too close to an evaluation question were dropped.
- 3,102 duplicate queries were removed; template headings were removed.
- The dev split is by section group.
- Limitation: queries are generated from headings (weak supervision), not asked by farmers.

**Hardware**
- Reranker: ~2.3 GB on disk. Fine on a GPU; a few seconds per question on CPU at 30 candidates.
- NLI: ~0.6 GB.
- Fine-tuning bge-m3: GPU with ≥ 8 GB (batch 16, length 256, top-6 layers, fp16).

---

## D. Evaluation results

### D1. What could be measured here

All on the **test split** (52 answerable questions; nothing was chosen on them), over the library's 8,872
passages, search text built as the app builds it (spelling repair + dialect rewrite; no LLM translation).

### D2. Retrieval experiments — measured on the project PC (runs 20261117-005606 … 011655)

bge-m3 + the real vector store; search text as the app builds it (spelling repair + dialect rewrite + LLM
translation with qwen2.5:7b). **Test split, 52 answerable questions** (nothing chosen on them). Validation MRR in
the last column (81 questions).

| experiment | fusion | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@10 | hit@5 | search latency | val MRR |
|---|---|---|---|---|---|---|---|---|---|
| E1 dense only | — | 0.861 | 0.879 | 0.254 | 0.840 | 0.810 | 0.962 | ~250 ms* | 0.839 |
| E1x dense + expansion | — | 0.857 | 0.865 | 0.250 | 0.838 | 0.797 | 0.962 | 110 ms | 0.821 |
| E2 BM25 only | — | 0.680 | 0.760 | 0.196 | 0.668 | 0.640 | 0.788 | 0.7 ms | 0.538 |
| E2x BM25 + expansion | — | 0.702 | 0.787 | 0.215 | 0.686 | 0.671 | 0.788 | 1.7 ms | 0.565 |
| **E3 hybrid — the original app** | weight 1, slot on | 0.797 | 0.893 | 0.246 | 0.792 | 0.783 | 0.865 | 80 ms | 0.744 |
| E5 hybrid + expansion | weight 1, slot on | 0.849 | 0.904 | 0.258 | 0.822 | 0.805 | 0.923 | 124 ms | 0.776 |
| E3 hybrid | weight 0.5, slot off | 0.839 | 0.916 | 0.254 | 0.820 | 0.808 | 0.923 | 79 ms | 0.793 |
| E5 hybrid + expansion | weight 0.5, slot off | 0.862 | 0.906 | 0.258 | 0.849 | 0.828 | 0.942 | 119 ms | 0.812 |
| E4 hybrid + reranker | weight 0.5, slot off | 0.914 | 0.928 | 0.281 | 0.901 | 0.891 | 0.962 | 890 ms (GPU) | 0.847 |
| **E6 hybrid + expansion + reranker** | weight 0.5, slot off | **0.934** | **0.947** | **0.285** | **0.920** | **0.910** | **0.981** | 1,027 ms (GPU) | 0.847 |
| E7 fine-tuned | | NOT EVALUATED | | | | | | | no fine-tuned model |

\* includes the model warm-up of the first experiment. Reranker: BAAI/bge-reranker-v2-m3 on CUDA, loaded in 33 s,
30 candidates.

- **The reranker is the largest gain**: test MRR 0.792 → 0.920 (+0.128), Recall@5 0.797 → 0.934, and a relevant
  passage is in the top 5 for 98% of questions instead of 87%. Validation agrees (0.744 → 0.847). The cost is about
  0.9 s per question on the GPU, small next to the ~37 s an answer takes.
- **Fusion tuning** (validation grid, 18 settings): without a reranker, embeddings alone ranked first by MRR
  (0.839). Keyword weight 0.25 + expansion was close (0.828) and better on Recall@5/10 and nDCG. Equal weight
  with the reserved slot was the worst (0.744). With a reranker, what matters is that the right passage is
  among the 30 candidates, which favours hybrid + expansion (the highest Recall@10). The measured configuration
  is weight 0.5, slot off, expansion on, reranker on (E6). Weight 0.25 with the reranker was not measured.
- RRF k and candidate count make no difference (validation MRR 0.744–0.745 across 12 settings), so they stay
  60 / 30.

**BM25 parameters** (validation, 16 settings): MRR@10 ranged 0.536–0.575, and the app's 1.5 / 0.75 scored
0.564. The best (1.5 / 0.9, 0.575) is within noise for 81 questions, so the defaults are kept.

### D3. Chunking (keyword search only)

| strategy | chunks | mean chars | context chars@5 | test Recall@5 | test MRR@10 | test nDCG@10 | validation MRR@10 |
|---|---|---|---|---|---|---|---|
| production (section-aware) | 9,264 | 630 | 2,909 | 0.721 | 0.679 | 0.669 | 0.555 |
| sentences (no sections) | 7,984 | 708 | 3,486 | 0.758 | 0.697 | 0.691 | 0.523 |
| fixed 800/150 | 6,386 | 797 | 3,983 | 0.805 | 0.735 | 0.725 | 0.524 |
| recursive | 9,203 | 593 | 2,927 | 0.761 | 0.704 | 0.691 | 0.459 |
| children only (300) | 23,803 | 302 | 1,348 | 0.691 | 0.635 | 0.632 | 0.468 |
| parent-child | 23,803 | 302→630 | 3,148 | 0.759 | 0.673 | 0.680 | **0.584** |

Validation and test disagree on the winner, and larger chunks contain an expected phrase more often (fixed
windows put 37% more text in front of the LLM). **Production chunking is not changed on this evidence.**
Parent-child is the best candidate: best on validation, and +3.8 points test Recall@5 at 8% more context.
It needs the hybrid run (`--embedding-model BAAI/bge-m3`). The rebuild here gives 9,264 production chunks
versus 8,872 in the store, because the notebook also drops chunks repeated across documents.

### D4. Claim verifier (controlled cases from real test passages)

| case | n | correct verdict |
|---|---|---|
| supported sentence | 43 | 100% |
| number changed to a value in no passage | 29 | 100% |
| sentence from another document | 28 | 96% |
| number taken from another passage | 21 | 90% (marked partial) |
| negated sentence | 43 | **0%** (words can't see "not"; this is what the NLI option is for) |

The false-alarm rate on supported sentences was 0%. Copied sentences are easier than an LLM's paraphrases,
so real-answer faithfulness comes from `evaluate.py --answers` + `human_review.csv`.

### D5. Query understanding

Crop group extraction matched the question's category on 102 of 102 crop questions. The rules were developed
while looking at these questions, so this is **development accuracy, not an unbiased estimate**. A fresh
labelled set is needed for that.

### D6. Original vs improved system (test split, measured)

| metric | original (E3: weight 1, slot on) | improved (E6: weight 0.5, slot off, expansion, reranker) | difference |
|---|---|---|---|
| Recall@5 | 0.797 | 0.934 | +0.137 |
| Recall@10 | 0.893 | 0.947 | +0.054 |
| Precision@5 | 0.246 | 0.285 | +0.039 |
| MRR@10 | 0.792 | 0.920 | +0.128 |
| nDCG@10 | 0.783 | 0.910 | +0.127 |
| Retrieval latency per question | 80 ms | 1,027 ms (GPU) | +0.95 s |
| Answer correct / context recall | 0.769 / 0.846 | 0.827 / 0.923 | +0.058 / +0.077 (D7) |

### D7. Answer quality — before and after (59 test questions, qwen2.5:7b)

"Before" is the original pipeline (run 20261117-013244). "After" adds query expansion, keyword weight 0.5, no
keyword slot, the reranker, query classification and claim verification (run 20261117-022152).

| metric | before | after | difference |
|---|---|---|---|
| answer correct (answerable; expected phrase in the answer) | 0.769 | **0.827** | +0.058 (6 fixed, 3 broken) |
| context recall (a relevant passage among those given to the LLM) | 0.846 | **0.923** | +0.077 |
| faithfulness (claims supported, claims.py) | 0.990 | 0.996 | +0.006 |
| unsupported-claim rate | 0.00 | 0.00 | — |
| grounded | 1.00 | 1.00 | — |
| out-of-scope questions refused | 1.00 | 1.00 | — |
| citation precision (cited passages that match the label) | 0.759 | 0.704 | −0.055 (see below) |
| answers whose citations are **all** off-label | 5 | 3 | −2 |
| mean sources cited per answer | 1.62 | 1.84 | +0.22 |
| mean latency per answer | 37.3 s | 36.6 s | −0.7 s |
| mean tokens per answer | 1,647 | 1,815 | +10% |
| failures | 0 | 0 | — |

The citation-precision dip comes from answers citing more supporting passages. The label rule only counts a
passage that contains the expected phrase, so extra supporting citations score as "off-label". Answers whose
citations are all off-label (the harmful case) went down from 5 to 3. Broken cases worth a human look: q06
(clover before wheat), q70 (blossom end rot: describes the symptom, not the cause), e46 (mango harvest in Aswan:
no month given).

### D8. Confidence calibration (run 20261117-030105)

Fitted on validation (84 questions; 76 with the answer retrieved), using the same pipeline as the app (expansion
+ reranker + keyword weight 0.5). Measured on the 59 test questions:

| metric | calibrated model | current single rule (cosine ≥ 0.591 or named crop) |
|---|---|---|
| AUROC (separating retrievable from not) | **0.990** | 0.973 (cosine alone) |
| Brier score | **0.018** | 0.051 (as a 0/1 rule) |
| ECE | **0.021** | — |
| coverage / selective accuracy at its threshold | 0.949 / 0.911 | 0.847 / 0.980 |

The probabilities are well calibrated. The *thresholds* are weak, though: validation has only 8 failures to
learn from, and 90% of validation questions already succeed, so both thresholds landed at the minimum (0.0135).
In practice the layer abstains only on the very lowest cases. The existing 0.591 gate stays in force and keeps
doing most of the refusing. Switching it on adds the ask-back for a missing crop, cautions on unsupported
claims, and a probability in the API. Useful thresholds need more unanswerable questions in the evaluation set
(I.2).

---

## E. Testing

- Baseline before changes: **252 passed**.
- After: **307 passed, 0 failed** (`cd backend && python -m pytest -q`). That's the 252 original tests,
  unmodified, plus 44 new ones in `tests/test_ai_layers.py` and 11 in `tests/test_vision.py`.
- The new tests cover:
  - metrics and the split;
  - Arabic, Egyptian, English and mixed questions;
  - empty queries;
  - no relevant passages;
  - expansion safety and failure;
  - weighted search and the gate;
  - metadata preservation;
  - reranker reordering, failure and missing model;
  - claim verdicts with or without NLI, and NLI failure;
  - calibration fit/save/load and wrong-feature files;
  - ask / qualify / abstain;
  - layers off = unchanged;
  - tracing disabled, unavailable and failing;
  - privacy settings;
  - token usage;
  - API fields additive.
- Extra checks run outside pytest:
  - every evaluation path (E1–E7, tuning, calibration) on a small test store with a tiny random model;
  - the training loop end to end;
  - `build_collection.py`;
  - the real `langsmith` client pointed at an unreachable endpoint: the answer returned in 0.12 s and the
    upload failed quietly in the background.
- Not run here: anything needing bge-m3, a reranker, an NLI model, Ollama or a GPU.

---

## F. Observability (LangSmith)

- **Integrated** via `langsmith`'s `trace` context manager behind a thin wrapper (`app/core/tracing.py`). No
  LangChain, and no pipeline rewrite.
- **Stages traced:**
  - `saqya.answer` (root: question, topic, models, settings, final answer, origin, sources);
  - `translate_query` (LLM);
  - `query_expansion`;
  - `retrieve` (queries, weights, gate; dense and BM25 top-5 ids and their times, fused ranking, RRF
    settings);
  - `rerank`;
  - `generate` (LLM input passages, answer, model, **prompt/completion tokens and duration** from Ollama);
  - `grounding_check` (problems found, retry);
  - `claim_verification`;
  - `confidence`;
  - `beyond_documents`.

  Errors are recorded on the span and re-raised unchanged.
- **Stage timings** are logged for every request even with tracing off:
  `Stages: translate_query 410ms | retrieve 230ms | generate 3100ms | …`.
- **Enable:** `pip install langsmith`, then in `backend/.env`: `LANGSMITH_TRACING_ENABLED=true`,
  `LANGSMITH_API_KEY=…`, optionally `LANGSMITH_PROJECT=saqya-ai` and `LANGSMITH_ENDPOINT` (EU region).
  **Disable:** `LANGSMITH_TRACING_ENABLED=false` (the default).
- **Privacy:**
  - passages are sent as source, pages, section, scores and the first `LANGSMITH_PASSAGE_CHARS` (200)
    characters (0 = none);
  - `LANGSMITH_LOG_TEXT=false` sends questions and answers as their length only;
  - every string is capped at 2,000 characters;
  - the key is never put in a span.

  Uploaded documents' passages follow the same rule. Questions do leave the machine when text logging is on:
  turn it off, or self-host Langfuse (the wrapper is the single place to change), if that is not acceptable.
- **Independence:** if the package or key is missing, or LangSmith errors or is unreachable, answers are
  unchanged. After the first tracing error, tracing switches off until restart.
- **LangSmith evaluation** was not duplicated. `scripts/evaluate.py` is the evaluation, and its JSON/CSV can be
  uploaded as a LangSmith dataset if wanted.
- **Knowledge graph / GraphRAG: not implemented.** The evaluation questions are single-hop (a date, a rate, a
  cause in one bulletin). Crop→document routing already exists (topic metadata + per-crop slots), and
  nothing measured points at multi-hop failures. Relations extracted from these PDFs could not be checked
  against evidence without manual review. Revisit if `--answers` shows multi-hop failures.

---

## G. Files

**Created**
- `backend/app/core/tracing.py`
- `backend/app/evaluation/{__init__,metrics,dataset,runner}.py`
- `backend/app/services/{query_understanding,query_expansion,reranker,claims,confidence}.py`
- `backend/scripts/{evaluate,compare_chunking,evaluate_claims,build_finetune_data,finetune_embeddings,build_collection,download_models,enable_ai_layers,package_vector_store}.py`
- `backend/app/services/vision.py`, `backend/app/api/routes/vision.py`, `backend/app/schemas/vision.py`,
  `backend/tests/test_vision.py`
- `ONBOARDING.md`, `docs/computer-vision-handoff.md`
- `backend/tests/test_ai_layers.py`
- `docs/ai-ml-engineering.md`
- `data/finetune/{train,dev}.jsonl` and `report.json`
- `evaluation/results/20261116-{keyword-offline,chunking-bm25,claims-lexical}/`

**Modified (additive, defaults unchanged)**
- `backend/app/services/retrieval.py`: configurable RRF/BM25, weights, tracing data, empty-query guard
- `backend/app/services/rag_service.py`: optional layers, spans
- `backend/app/services/generation.py`: token usage for traces
- `backend/app/core/config.py`: new settings
- `backend/app/main.py`: wiring, fine-tuned-store fallback
- `backend/app/schemas/query.py`: optional `understanding` / `claims` / `confidence`, health `ai_layers`
- `backend/app/api/routes/query.py`
- `backend/.env.example`
- `backend/requirements.txt` (optional packages, commented)
- `.gitignore`
- `README.md` (one section linking here)

**Preserved:** the vector store, all datasets, notebooks, existing results, the frontend and the
`_backup_before_update` folder. Originals of every modified file are copied to `_backup_before_ai_ml/` before
the update, and git shows every change.

**New model checkpoints:** none.

---

## H. How to run (Windows, from `D:\agri-rag-assistant\backend`)

```bat
:: optional packages
pip install langsmith psutil

:: the app, unchanged
uvicorn app.main:app --reload

:: 1. baseline + improvements on your machine (E1, E2, E2x, E3, E5); --translate uses Ollama like the app
python scripts\evaluate.py --translate
:: 2. reranking (E4, E6): download once (~2.3 GB), then evaluate
python scripts\download_models.py reranker
python scripts\evaluate.py --translate --reranker
:: 3. settings on validation; confidence calibration (writes backend\data\confidence_calibration.json)
python scripts\evaluate.py --tune
python scripts\evaluate.py --calibrate
:: 4. full answers with the LLM (correctness, faithfulness, citations, tokens, latency) + human_review.csv
python scripts\evaluate.py --answers --answers-limit 60
:: 5. chunking with the real embeddings (slow: embeds every strategy)
python scripts\compare_chunking.py --embedding-model BAAI/bge-m3
:: 6. claim verifier with NLI
python scripts\evaluate.py --export-chunks ..\evaluation\chunks.jsonl
python scripts\download_models.py nli
python scripts\evaluate_claims.py --chunks-file ..\evaluation\chunks.jsonl --nli
:: 7. fine-tuning (GPU), separate collection, then E7
python scripts\build_finetune_data.py
python scripts\finetune_embeddings.py
python scripts\build_collection.py --model models\bge-m3-saqya
python scripts\evaluate.py --translate --reranker --finetuned-store data\vector_store_finetuned
```

**Turning layers on** (in `backend\.env`, only after the numbers above justify it):
- `RERANKER_ENABLED`
- `QUERY_EXPANSION_ENABLED`
- `QUERY_CLASSIFICATION_ENABLED`
- `CLAIM_VERIFICATION_ENABLED` (+ `CLAIM_VERIFICATION_NLI`)
- `CONFIDENCE_CALIBRATION_ENABLED`
- `FINE_TUNED_EMBEDDINGS_ENABLED`
- `LANGSMITH_TRACING_ENABLED`

Restart the backend after a change. `GET /health` → `ai_layers` shows what is on.

---

## I. Remaining recommendations

1. **Run steps 1–4 above.** They produce the numbers this report can't: dense, hybrid, reranker, calibration
   and answer quality. Turn on the reranker only if E4 beats E3 on test.
2. **More labelled questions.** 52 test questions resolve differences of about ±0.05 MRR at best. The most
   valuable addition is 100–200 real farmer questions (Egyptian and English especially, plus cross-language)
   with an expected document and phrase, in the existing JSON format, tagged as hand-checked.
3. **Category labels** (a few hundred questions) would allow a trained multi-label classifier and an
   unbiased estimate for B4/B5.
4. **Fill in `human_review.csv`** after step 4. It's the ground truth for faithfulness.
5. **Fine-tuning**: run step 7. Keep it only if E7 beats E3 on test across every language slice.
6. **Parent-child chunking**: if the hybrid comparison confirms it, build it as a separate collection.
7. Stale HNSW segment folders in `backend\data\vector_store` (11 folders, about 300 MB) are left untouched.
   Clean up only after a backup.
