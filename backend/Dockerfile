FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Ollama runs on the host machine, not inside this container
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so startup needs no download.
# Must match the model the notebook used (see data/vector_store/rag_config.json).
# Built with the Egyptian fine-tune (models/bge-m3-egyptian)? That folder is mounted at run time instead
# (docker run -v "$(pwd)/models:/models:ro" ...), so build with --build-arg EMBEDDING_MODEL= to skip this download.
ARG EMBEDDING_MODEL=BAAI/bge-m3
RUN if [ -n "$EMBEDDING_MODEL" ]; then python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"; fi

# Optional: bake the Whisper model in too (else it downloads on the first voice question)
ARG WHISPER_MODEL=""
RUN if [ -n "$WHISPER_MODEL" ]; then python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"; fi

# Optional: Chromium for the clean PDFs of attached documents (--build-arg CLEAN_PDF_CHROMIUM=1);
# without it they are drawn with MuPDF
ARG CLEAN_PDF_CHROMIUM=""
RUN if [ -n "$CLEAN_PDF_CHROMIUM" ]; then python -m playwright install --with-deps chromium; fi

COPY app ./app
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
