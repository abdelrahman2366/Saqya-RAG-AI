# SAQYA AI — getting started on a new machine

This guide gets a team member from an empty computer to a running SAQYA (backend + chat app) with the same
search quality as the project owner's machine, in about an hour (mostly downloads). Commands are for Windows;
macOS/Linux differences are noted.

What SAQYA is: a bilingual (Arabic / Egyptian Arabic / English) farming assistant. It answers from 52 agricultural
bulletins with citations, using hybrid search (bge-m3 embeddings + BM25 keywords), a cross-encoder reranker and
qwen2.5 through Ollama. When the bulletins have no answer it falls back to what it learned, then the web, then general
knowledge. It also has voice input/output and lets you attach your own documents. Architecture and full technical reference: `docs/technical-reference.md`.
AI/ML details and measured results: `docs/ai-ml-engineering.md`.

**Working on plant-disease detection from photos?** Do this guide first, then read
[`docs/computer-vision-handoff.md`](docs/computer-vision-handoff.md).

---

## 1. Install the tools

| tool | version | check |
|---|---|---|
| Python | 3.11 or 3.12 (not 3.13+, not 3.10) | `python --version` |
| Git | any recent | `git --version` |
| Ollama | latest, from ollama.com | `ollama --version` |
| Disk | ~15 GB free (models + vector store) | |
| GPU | optional. An NVIDIA GPU with 8 GB+ makes answers and the reranker much faster | `nvidia-smi` |

## 2. Get the code

```bat
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r backend/requirements.txt -r frontend/requirements.txt
```

- Use the URL from the repo's green **Code** button. You need to be added as a collaborator if the repo is
  private.
- On macOS/Linux, activate with `source .venv/bin/activate`.
- With an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org *after* the line above. Without one,
  the CPU build that pip installs works, just slower.

## 3. Get the models

```bat
ollama pull qwen2.5:7b
cd backend
python scripts\download_models.py reranker
cd ..
```

- `ollama pull qwen2.5:7b` is the answer model (~5 GB). On a weak PC use `qwen2.5:3b` instead.
- `download_models.py reranker` fetches the reranker (~2.3 GB) into `models/`.

The embedding model (BAAI/bge-m3, ~2.3 GB) downloads by itself the first time the backend starts.

## 4. Get the vector store (the indexed bulletins)

It is not in git: it is too large for GitHub (> 100 MB), and it is built from PDFs that are not in the repo.

**Option A — download it (recommended, 2 minutes).**
1. Download `saqya-vector-store.zip` from the repo's **Releases** page, or wherever the project owner shared it.
2. Unzip it **into the `backend` folder**. You should end up with `backend\data\vector_store\chroma.sqlite3`
   and `backend\data\vector_store\rag_config.json`, and possibly `backend\data\confidence_calibration.json`.

The owner makes this zip with `python scripts\package_vector_store.py` (from `backend`).

**Option B — rebuild it from the PDFs (1–2 hours).** Get the original PDFs from the project owner into
`data\raw\`, then follow `docs/technical-reference.md` → Setup → "3. Build the vector store (notebooks)".

## 5. Settings

```bat
cd backend
copy .env.example .env
python scripts\enable_ai_layers.py
cd ..\frontend
copy .env.example .env
```

- `copy .env.example .env` (in `backend`) creates your own settings file. It is never committed.
- `enable_ai_layers.py` switches on the settings the evaluation chose (reranker, query expansion, keyword
  weight…). If `backend\data\confidence_calibration.json` exists, use
  `python scripts\enable_ai_layers.py --confidence` instead.
- On macOS/Linux, use `cp` instead of `copy`.

Optional: for the Gemini voice, put **your own** key after `GEMINI_API_KEY=` in `backend\.env`. Without it the app
uses Microsoft Edge voices. Never commit `.env` files or paste keys into `.env.example`.

## 6. Run it

Terminal 1 (backend):
```bat
.venv\Scripts\activate
cd backend
uvicorn app.main:app --reload
```

Terminal 2 (chat app):
```bat
.venv\Scripts\activate
cd frontend
streamlit run app.py
```

- Open <http://localhost:8501> and ask: «إمتى أزرع القمح؟» or "When should maize be planted?". Expect a cited
  answer. The first question after a start takes ~30 s longer while the models load.
- <http://localhost:8000/health> should show `"status": "ok"`, `"num_chunks": 8872` and, under `ai_layers`,
  `reranker: true`.
- <http://localhost:8000/docs> lists every API route.

## 7. Check your setup

```bat
cd backend
python -m pytest -q
```

All tests should pass (307 at the time of writing). They use fake models, so they need no Ollama, GPU or
vector store.

To measure retrieval quality on your machine (needs the vector store and Ollama):
```bat
python scripts\evaluate.py --translate --reranker
```
Results go to `evaluation\results\`. Compare them with `docs\ai-ml-engineering.md`, section D.

---

## Working on the code

**Workflow**
1. Make a branch: `git checkout -b feature/<name>`.
2. Write the code and its tests, then run `python -m pytest -q`.
3. `git push -u origin feature/<name>` and open a pull request.

**House rules** (they keep the app stable):
- **New features are optional and off by default.** Each gets a setting in `backend/app/core/config.py` and
  `backend/.env.example`. With it off, the app must behave exactly as before.
- **Measure before switching something on.** Use `scripts/evaluate.py` for search and answers. Choose settings
  on the *validation* questions and report the *test* questions.
- **Never commit:** `.env` files, API keys, `models/`, `backend/data/vector_store/`, `data/raw/`, `data/clean/`
  (already in `.gitignore`).
- **Large models load lazily** (on first use, once) and never download on startup.
- Don't change existing tests just to make them pass; fix the cause.

**Where things are**

| path | what |
|---|---|
| `backend/app/main.py` | starts the API, loads the vector store, wires the services |
| `backend/app/core/config.py` | every setting (read from `backend/.env`) |
| `backend/app/api/routes/` | API endpoints: `query.py`, `voice.py`, `documents.py`, `vision.py` |
| `backend/app/services/rag_service.py` | the answer pipeline (search → LLM → checks → citations) |
| `backend/app/services/retrieval.py` | hybrid search (bge-m3 + BM25, RRF) |
| `backend/app/services/reranker.py`, `query_expansion.py`, `query_understanding.py`, `claims.py`, `confidence.py` | optional AI layers |
| `backend/app/services/vision.py` | plant-disease detection from photos (extension point, no model yet) |
| `backend/app/services/beyond.py`, `web_search.py`, `learning.py` | answers beyond the bulletins |
| `backend/scripts/` | evaluation, fine-tuning, model download, packaging |
| `backend/tests/` | tests (pytest) |
| `frontend/app.py`, `frontend/api_client.py` | Streamlit chat app and its API client |
| `notebooks/` | data cleaning, vector-store build, evaluation notebooks |
| `data/eval_questions*.json` | evaluation questions (validation/test split is automatic) |

**Troubleshooting**
- `/health` says the vector store is missing: check that the zip was unzipped into `backend` (step 4).
- `reranker: false`: run step 3's `download_models.py reranker`, then restart the backend.
- Answers are very slow: use `qwen2.5:3b`, or add `RERANKER_DEVICE=cpu` if the GPU is short of memory.
- More: `docs/technical-reference.md` → Troubleshooting.
