"""Thin wrapper around the SAQYA AI backend API.

Errors carry a `kind` so the UI can show them in the farmer's language.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


class ConfigError(RuntimeError):
    """The frontend is not configured (API_BASE_URL missing)."""


class APIError(Exception):
    """A request failed.

    kind: "unreachable" | "timeout" | "invalid" | "not_found" | "unavailable" | "server"
    detail: technical detail from the backend (English), may be empty
    """

    def __init__(self, kind: str, detail: str = "", status_code: int | None = None) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.status_code = status_code


class AgriAPIClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        base_url = base_url or os.getenv("API_BASE_URL")
        if not base_url:
            raise ConfigError(
                "API_BASE_URL is not set. Copy frontend/.env.example to frontend/.env "
                "and set API_BASE_URL to your backend address."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout or os.getenv("REQUEST_TIMEOUT", "180"))
        self._session = requests.Session()

    # ------------------------------------------------------------------ public
    def health(self) -> dict:
        return self._request("GET", "/health", timeout=5)

    def ask(self, question: str, topic: str | None = None, history: list[str] | None = None,
            web: bool = True, learn: bool = True, general: bool = True) -> dict:
        """Returns {"answer", "sources", "passages", "language", "refused", "origin", "note", "learned", "latency_ms"}.

        `history`: the farmer's previous questions in this chat (oldest first), so a follow-up such as
        "how much water does it need?" is understood. `topic` (a crop key) limits the search to that crop.
        `web` / `learn` / `general`: what may be used when the documents don't cover a farming question
        (a web search, keeping what it cites, a general-knowledge answer).
        """
        payload: dict = {"question": question, "web": web, "learn": learn, "general": general}
        if topic:
            payload["topic"] = topic
        if history:
            payload["history"] = history[-4:]
        # a web search reads several pages before answering: allow more time than the library needs
        return self._request("POST", "/query", json=payload, timeout=max(self.timeout, 240) if web else None)

    # ------------------------------------------------------------------ learned knowledge
    def knowledge(self) -> dict:
        return self._request("GET", "/knowledge", timeout=10)

    def forget_source(self, source_id: str) -> int:
        return self._request("DELETE", f"/knowledge/sources/{source_id}", timeout=10).get("removed", 0)

    def forget_all(self) -> int:
        return self._request("DELETE", "/knowledge", timeout=10).get("removed", 0)

    def library(self) -> dict:
        """What is indexed: {"num_documents", "num_chunks", "languages", "crops", "documents", ...}."""
        return self._request("GET", "/library", timeout=15)

    def feedback(self, question: str, answer: str, rating: str, sources: list[str] | None = None,
                 topic: str | None = None, comment: str | None = None) -> dict:
        """rating: "up" or "down". Returns {"status": "saved", "total": int, "helpful": int}."""
        payload = {"question": question, "answer": answer[:6000], "rating": rating,
                   "sources": (sources or [])[:20], "topic": topic, "comment": comment}
        return self._request("POST", "/feedback", json=payload, timeout=10)

    # ------------------------------------------------------------------ voice
    def transcribe(self, audio: bytes, filename: str = "question.wav", language: str | None = None) -> dict:
        """Speech -> {"text", "standard", "dialect", "language", "duration", "seconds"}.
        `language`: "ar", "en" or None (detect). The first call can be slow: Whisper loads (and downloads) then."""
        files = {"audio": (filename, audio, "audio/wav")}
        data = {"language": language} if language in ("ar", "en") else {}
        return self._request("POST", "/voice/transcribe", files=files, data=data, timeout=max(self.timeout, 600))

    def speak(self, text: str, language: str | None = None, voice: str = "female") -> bytes:
        """Text -> MP3 audio (Egyptian voice for Arabic). Raises APIError("unavailable") without internet."""
        payload = {"text": text[:6000], "voice": voice}
        if language in ("ar", "en"):
            payload["language"] = language
        return self._request("POST", "/voice/speak", json=payload, timeout=60, raw=True)

    # ------------------------------------------------------------------ uploaded documents
    def upload_document(self, filename: str, data: bytes, mime: str | None = None, session: str | None = None,
                        learn: bool = False) -> dict:
        """Attach a PDF / .docx / .txt / .md file to the chat `session`: the backend cleans it, makes a clean PDF
        and indexes it (and, with `learn`, keeps an agricultural document's passages for later questions).
        Returns the document's info (id, title, sections, clean_pdf, cleaning, learned...)."""
        files = {"file": (filename, data, mime or "application/octet-stream")}
        form = {"session": session} if session else {}
        if learn:
            form["learn"] = "true"
        return self._request("POST", "/documents", files=files, data=form, timeout=max(self.timeout, 900))

    def documents(self, session: str | None = None) -> list[dict]:
        params = {"session": session} if session else {}
        return self._request("GET", "/documents", params=params, timeout=15).get("documents", [])

    def document(self, doc_id: str) -> dict:
        return self._request("GET", f"/documents/{doc_id}", timeout=15)

    def clean_pdf(self, doc_id: str) -> bytes:
        """The clean PDF the backend made from the document."""
        return self._request("GET", f"/documents/{doc_id}/clean.pdf", timeout=60, raw=True)

    def delete_document(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}", timeout=15, raw=True)

    def delete_session_documents(self, session: str) -> None:
        """Remove every document of a chat session (the chat was deleted or closed)."""
        self._request("DELETE", "/documents", params={"session": session}, timeout=10)

    def ask_document(self, doc_id: str, question: str, history: list[str] | None = None) -> dict:
        """A question about one attached document; same response as ask()."""
        payload: dict = {"question": question}
        if history:
            payload["history"] = history[-4:]
        return self._request("POST", f"/documents/{doc_id}/query", json=payload)

    def explain_document(self, doc_id: str, focus: str | None = None, language: str | None = None) -> dict:
        """Explain the whole document, or one section / topic (`focus`)."""
        payload = {"focus": focus, "language": language if language in ("ar", "en") else None}
        return self._request("POST", f"/documents/{doc_id}/explain", json=payload)

    # ------------------------------------------------------------------ internals
    def _request(self, method: str, path: str, timeout: float | None = None, raw: bool = False, **kwargs):
        url = f"{self.base_url}{path}"
        try:
            response = self._session.request(method, url, timeout=timeout or self.timeout, **kwargs)
        except requests.exceptions.ConnectionError:
            raise APIError("unreachable", self.base_url)
        except requests.exceptions.Timeout:
            raise APIError("timeout")
        except requests.exceptions.RequestException as exc:
            raise APIError("server", str(exc))

        if response.ok:
            return response.content if raw else response.json()

        detail = self._detail(response)
        kind = {404: "not_found", 413: "invalid", 415: "invalid", 422: "invalid",
                503: "unavailable"}.get(response.status_code, "server")
        raise APIError(kind, detail, response.status_code)

    @staticmethod
    def _detail(response: requests.Response) -> str:
        try:
            detail = response.json().get("detail", "")
        except ValueError:
            return ""
        return detail if isinstance(detail, str) else ""
