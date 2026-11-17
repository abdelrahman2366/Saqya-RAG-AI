🌾 SAQYA AI — ساقية

An Arabic & English RAG-Based Agricultural AI Assistant

SAQYA AI is an intelligent agricultural assistant designed to help farmers find reliable information about crops, soil, fertilizers, irrigation, pests, diseases, harvesting, and storage.

It uses Retrieval-Augmented Generation (RAG) to retrieve information from agricultural documents before generating answers. It supports Arabic, English, and Egyptian Arabic, with source citations showing the documents and pages used to generate each answer.

Key Features

* Bilingual AI Chat: Ask agricultural questions in Arabic or English and receive answers in the same language.
* Egyptian Arabic Support: Understands common Egyptian agricultural expressions and dialect.
* Hybrid Search: Combines semantic vector search with BM25 keyword retrieval.
* Source Citations: Displays the documents, pages, and passages supporting generated answers.
* Answer Verification: Checks generated answers for citation consistency, unsupported numbers, and other grounding issues.
* Voice Assistant: Ask questions using speech and listen to answers in Egyptian or formal Arabic.
* Document Upload: Upload PDF, DOCX, TXT, or Markdown files and ask questions about their contents.
* Web Search: Searches external sources when the agricultural knowledge base cannot answer a question.
* Continuous Learning: Retains relevant cited information from web searches and uploaded agricultural documents for future questions.
* Bilingual Interface: Supports English and Arabic with right-to-left layout.

⸻

👥 For the team

* New to the project? [ONBOARDING.md](ONBOARDING.md) gets SAQYA running on a new machine in about an hour.
* Working on plant-disease detection from photos? Start with [docs/computer-vision-handoff.md](docs/computer-vision-handoff.md).
* AI/ML design, evaluation and measured results: [docs/ai-ml-engineering.md](docs/ai-ml-engineering.md).
* Full technical reference (API, settings, grounding, troubleshooting): [docs/technical-reference.md](docs/technical-reference.md).

⸻

🛠️ Tech Stack

Component	Technology
Backend	FastAPI
Frontend	Streamlit
LLM	Ollama (Qwen 2.5)
Embeddings	BAAI/bge-m3
Vector Database	ChromaDB
Retrieval	Hybrid Search (Vector Search + BM25)
Speech Recognition	Faster-Whisper
Text-to-Speech	Edge TTS / Gemini TTS
PDF Processing	PDFMiner, PyMuPDF, PyPDF
Language	Python
Testing	Pytest

⸻

📋 Prerequisites

Before installing SAQYA, make sure you have:

* Python 3.11 or 3.12
* Git
* Ollama

Download:

* Python: https://www.python.org/downloads/
* Git: https://git-scm.com/downloads
* Ollama: https://ollama.com/

Install the default language model:

ollama pull qwen2.5:3b

Make sure Ollama is running before starting the backend.

⸻

🚀 Installation & Setup

1. Clone the Repository

git clone https://github.com/abdelrahman2366/Saqya-RAG-AI.git
cd Saqya-RAG-AI

2. Create a Virtual Environment

python -m venv .venv

Activate it:

Windows:

.venv\Scripts\activate

Linux / macOS:

source .venv/bin/activate

3. Install Dependencies

Install the project dependencies:

pip install -r requirements.txt -r backend/requirements.txt -r frontend/requirements.txt

For improved PDF processing when uploading documents, install Playwright Chromium:

python -m playwright install chromium

4. Prepare the Agricultural Knowledge Base

The vector database is generated through the project’s Jupyter notebooks.

Place your original agricultural PDF documents inside:

data/raw/

Run the notebooks in the following order:

1. notebooks/clean_rag_pdfs_no_ai.ipynb
    Cleans and restructures agricultural PDFs.
2. notebooks/data_check.ipynb
    Checks documents for duplicates, unreadable pages, and extraction issues.
3. notebooks/egyptian_finetune.ipynb (Optional)
    Fine-tunes the BGE-M3 embedding model to better understand Egyptian Arabic agricultural questions.
4. notebooks/rag_pipeline.ipynb
    Processes documents, extracts text, creates chunks, generates embeddings, builds the ChromaDB vector store, and runs evaluation.

Run the final notebook using Kernel → Restart & Run All.

The generated vector database is saved in:

backend/data/vector_store/

Team members can skip the notebooks: download saqya-vector-store.zip from the repository's Releases page and unzip
it into backend/ (see ONBOARDING.md).

Note: The original PDFs, cleaned PDFs, fine-tuned models, and generated vector database are not included in the Git repository. They must be prepared locally.

⸻

⚙️ Configuration

Backend Configuration

Navigate to the backend directory:

cd backend

Create the environment file:

Windows:

copy .env.example .env

Linux / macOS:

cp .env.example .env

The backend configuration is stored in backend/.env.

Important settings include:

Variable	Default	Description
OLLAMA_HOST	http://localhost:11434	Ollama server address
OLLAMA_MODEL	qwen2.5:3b	Language model
TOP_K	6	Number of retrieved passages
HYBRID_SEARCH	true	Enable hybrid retrieval
QUERY_TRANSLATION	true	Enable Arabic-English query translation
SELF_CHECK	true	Enable answer verification
WEB_SEARCH	true	Enable external web search
LEARNING	true	Enable knowledge retention
GENERAL_KNOWLEDGE	true	Allow general-knowledge fallback

Additional settings are available in backend/.env.example.

Frontend Configuration

Navigate to the frontend directory:

cd frontend

Create the environment file:

Windows:

copy .env.example .env

Linux / macOS:

cp .env.example .env

Configure the backend connection:

API_BASE_URL=http://localhost:8000
REQUEST_TIMEOUT=180

⸻

▶️ Running SAQYA

SAQYA consists of two applications: the FastAPI backend and the Streamlit frontend.

Both must be running.

1. Start the Backend

Open a terminal:

cd backend
uvicorn app.main:app --reload

The backend will run at:

http://localhost:8000

API documentation is available at:

http://localhost:8000/docs

You can check the backend health using:

http://localhost:8000/health

2. Start the Frontend

Open a second terminal and activate the same virtual environment.

Then run:

cd frontend
streamlit run app.py

Open the application in your browser:

http://localhost:8501

SAQYA is now ready to use.

⸻

💬 How to Use SAQYA

1. Agricultural Chat

Ask questions about agricultural topics in Arabic or English.

Examples:

* When should I plant tomatoes?
* What fertilizer is suitable for wheat?
* How can I control pests affecting maize?
* إمتى أزرع الطماطم؟
* إيه أفضل طريقة لري القمح؟

SAQYA retrieves relevant agricultural information and generates an answer with supporting citations.

2. Voice Assistant

1. Click the microphone button in the question box.
2. Allow microphone access.
3. Speak your question.
4. Submit the recording.
5. SAQYA transcribes your speech, processes the question, and generates a spoken answer.

Egyptian Arabic speech is supported.

The first voice request may take longer because the Whisper model needs to be downloaded.

3. Upload Documents

1. Click the + button in the question box.
2. Select a PDF, DOCX, TXT, or Markdown file.
3. Upload the document.
4. Ask questions about its contents.

You can also request a document summary or an explanation of a particular section.

Uploaded documents are processed and indexed temporarily for the current session.

4. Settings

The Settings page allows you to:

* Change the interface language.
* Configure voice preferences.
* Select spoken Arabic style.
* Choose voice settings.
* Download or delete conversations.
* Enable or disable web search and learning.
* Manage information SAQYA has learned.

5. Answer Sources and Feedback

Each answer provides information about its origin, such as:

* Uploaded documents.
* Agricultural knowledge base.
* Previously learned information.
* Web search.
* General knowledge.

You can inspect the supporting passages and citations and rate answers using the thumbs-up and thumbs-down buttons.

⸻

🧪 Running Tests

The backend includes a Pytest test suite.

To execute the tests:

cd backend
pytest -v

The tests use mocked retrieval and language-model components, so a running Ollama instance and an existing vector database are not required for the test suite.

⸻

📁 Project Structure

Saqya-RAG-AI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── utils/
│   │
│   ├── data/
│   │   └── vector_store/
│   │
│   ├── scripts/          (evaluation, fine-tuning, model download, packaging)
│   ├── tests/
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── eval_questions.json
│   └── eval_questions_egyptian.json
│
├── docs/
│   ├── ai-ml-engineering.md
│   ├── computer-vision-handoff.md
│   ├── technical-reference.md
│   └── screenshots/
│
├── evaluation/
│   └── results/          (evaluation reports: JSON, CSV, Markdown)
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── theme.py
│   ├── ui_text.py
│   ├── assets/
│   ├── .streamlit/
│   ├── .env.example
│   └── requirements.txt
│
├── notebooks/
│   ├── clean_rag_pdfs_no_ai.ipynb
│   ├── data_check.ipynb
│   ├── egyptian_finetune.ipynb
│   ├── rag_pipeline.ipynb
│   └── results/
│
├── ONBOARDING.md
├── requirements.txt
└── README.md

⸻

🧠 AI/ML Improvements and Evaluation

Measured on the project's evaluation questions (test split, never used to choose settings):

| | Original search | Improved search |
|---|---|---|
| Recall@5 | 0.797 | 0.934 |
| MRR@10 | 0.792 | 0.920 |
| Correct answers | 76.9% | 82.7% |
| Right passage given to the LLM | 84.6% | 92.3% |

* Cross-encoder reranking (BAAI/bge-reranker-v2-m3), query expansion and tuned hybrid-search weights.
* Query understanding (crop, intent, symptoms, missing information) and claim-level answer checks.
* Calibrated confidence: answers, adds a caution, asks back when the crop is missing, or abstains.
* Optional LangSmith tracing of every stage.
* A reproducible evaluation framework: python backend/scripts/evaluate.py

Every layer is optional and set in backend/.env; python backend/scripts/enable_ai_layers.py switches on the
evaluated settings. Details: docs/ai-ml-engineering.md.

⸻

🌱 Project Goal

SAQYA aims to make agricultural knowledge more accessible by combining document-based retrieval, multilingual AI, Egyptian Arabic understanding, and voice interaction into one assistant.

Its goal is to help farmers access relevant agricultural information through a simple conversational interface.

⸻

Built with Python, FastAPI, Streamlit, Ollama, and RAG.
