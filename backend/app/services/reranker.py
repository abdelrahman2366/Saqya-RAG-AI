"""Optional cross-encoder reranking of the hybrid search's candidates.

The hybrid search (retrieval.py) ranks passages with two independent signals: an embedding of the question
against an embedding of each passage, and shared words. Neither reads the question and the passage *together*.
A cross-encoder does: it takes "question + passage" as one input and scores how well the passage answers the
question, which is slower (one model pass per passage) but more precise. So the search returns more
candidates than the answer needs (RERANKER_CANDIDATES, default 30) and the cross-encoder picks the best
`top_k`.

Models (both multilingual, both cover Arabic and English):
- BAAI/bge-reranker-v2-m3 (default): the same family as the embedding model, about 570M parameters / 2.3 GB,
  Apache-2.0. Comfortable on a GPU; a few seconds per question on a CPU.
- jinaai/jina-reranker-v2-base-multilingual: about half the size and faster, but licensed CC BY-NC 4.0
  (non-commercial) and needs RERANKER_TRUST_REMOTE_CODE=true. Use it only if that licence suits you.

Design rules:
- Off by default (RERANKER_ENABLED=false) and never validated automatically: scripts/evaluate.py measures it
  (experiments E4 and E6) so it is turned on because the numbers say so.
- The model is loaded on the first question that needs it, once, and shared (never at startup).
- Nothing is downloaded unless RERANKER_ALLOW_DOWNLOAD=true or you ran `python scripts/download_models.py
  reranker`: a first question must not silently pull 2 GB. A model that is not on disk means "no reranking".
- Any failure (model missing, out of memory, a bad batch) falls back to the search's own order, and the
  farmer still gets an answer. After a failed load the model is not tried again until the backend restarts.
- Passages keep everything they had (id, pages, section, source, cosine similarity); only their order changes.
  The cosine similarity is left alone on purpose: the relevance gate and focus_passages() are calibrated on it.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Sequence

from app.utils.text import detect_language

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - no torch: the load below reports it
        return "cpu"


def local_model_path(model_name: str) -> str | None:
    """A model folder next to the project (models/<name> or the given path), if there is one."""
    from app.core.config import BACKEND_DIR

    candidates = [Path(model_name).expanduser()]
    short = model_name.split("/")[-1]
    for base in (BACKEND_DIR.parent, BACKEND_DIR):
        candidates += [base / model_name, base / "models" / short]
    for path in candidates:
        if path.is_dir() and any(path.glob("config.json")):
            return str(path.resolve())
    return None


class Reranker:
    """Lazy, shared cross-encoder. `scores()` returns None when reranking is not possible right now."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto", batch_size: int = 16,
                 max_length: int = 512, allow_download: bool = False, trust_remote_code: bool = False,
                 model=None) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.allow_download = allow_download
        self.trust_remote_code = trust_remote_code
        self._model = model          # tests pass a ready model (anything with predict(pairs, batch_size=...))
        self._failed: str | None = None
        self._lock = threading.Lock()
        self.last_ms: float = 0.0    # time of the last scoring call, for the evaluation's latency column

    # ------------------------------------------------------------------ loading
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def failed(self) -> str | None:
        """Why the model could not be loaded (None: not tried yet, or loaded)."""
        return self._failed

    def load(self):
        """The cross-encoder, loaded once; None when it cannot be loaded (the reason is in `failed`)."""
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            started = time.perf_counter()
            try:
                from sentence_transformers import CrossEncoder

                location = local_model_path(self.model_name) or self.model_name
                device = pick_device(self.device)
                self._model = CrossEncoder(
                    location, device=device, max_length=self.max_length,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=not self.allow_download and location == self.model_name,
                )
                logger.info("Reranker %s loaded on %s in %.1fs", self.model_name, device, time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001 - missing model, no torch, out of memory: search without it
                self._failed = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning(
                    "Reranker %s is not available (%s). Answers use the search's own order. To install it: "
                    "python scripts/download_models.py reranker", self.model_name, self._failed)
        return self._model

    # ------------------------------------------------------------------ scoring
    def scores(self, pairs: Sequence[tuple[str, str]]) -> list[float] | None:
        """Relevance score of each (question, passage) pair (higher = better), or None when not possible."""
        if not pairs:
            return []
        model = self.load()
        if model is None:
            return None
        started = time.perf_counter()
        try:
            raw = model.predict([list(p) for p in pairs], batch_size=self.batch_size, show_progress_bar=False)
            out = [float(s) for s in raw]
        except Exception as exc:  # noqa: BLE001 - one bad batch must not lose the answer
            logger.warning("Reranking failed (%s); keeping the search's own order", exc)
            return None
        self.last_ms = (time.perf_counter() - started) * 1000
        return out

    def rerank(self, queries: Sequence[str], chunks: list, top_k: int) -> list:
        """The `top_k` best of `chunks` for the question, best first. `queries` are the search phrasings (the
        question, then its translation if any): each passage is scored against the phrasing in its own
        language when there is one, because a cross-encoder reads same-language pairs most reliably.
        On any failure the first `top_k` chunks are returned in the order they came."""
        if len(chunks) <= 1 or not queries:
            return chunks[:top_k]
        by_language: dict[str, str] = {}
        for q in queries:
            by_language.setdefault(detect_language(q), q)
        pairs = [(by_language.get(detect_language(c.text), queries[0]), c.text) for c in chunks]
        scored = self.scores(pairs)
        if scored is None or len(scored) != len(chunks):
            return chunks[:top_k]
        order = sorted(range(len(chunks)), key=lambda i: scored[i], reverse=True)[:top_k]
        out = []
        for i in order:
            chunk = chunks[i]
            try:
                chunk.rerank_score = round(scored[i], 4)   # kept beside the cosine similarity, never instead of it
            except Exception:  # noqa: BLE001 - a frozen chunk type: the order is what matters
                pass
            out.append(chunk)
        return out

    def describe(self) -> dict:
        return {"model": self.model_name, "loaded": self.loaded, "failed": self._failed,
                "device": pick_device(self.device) if self.loaded else self.device,
                "batch_size": self.batch_size, "max_length": self.max_length}
