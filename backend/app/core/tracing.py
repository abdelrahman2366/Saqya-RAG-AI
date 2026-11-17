"""Optional tracing of every stage of an answer, to LangSmith, plus stage timings in the log.

Usage inside the pipeline:

    from app.core.tracing import tracer
    with tracer.span("retrieve", "retriever", inputs={"queries": queries}) as span:
        chunks = ...
        span.set(outputs={"passages": tracer.passages(chunks)})

What it guarantees:
- **Never required.** Disabled (the default), not installed, no API key, LangSmith down, or an error inside the
  LangSmith client: the span still runs the code inside it, measures its time and never raises. After a
  tracing error, tracing is switched off for the rest of the process (one warning in the log).
- **Cheap when off.** A disabled span is a perf_counter call and a context variable.
- **Privacy.** Passages are sent as source, page, scores and their first LANGSMITH_PASSAGE_CHARS characters
  (0: none of their text); LANGSMITH_LOG_TEXT=false sends questions and answers as their length only. Every
  string is cut at 2,000 characters. API keys are never put in a span (the client reads its key itself).
- **Timings.** Each finished request logs one line with the time of every stage ("Stages: retrieve 212ms
  | generate 3410ms | ..."), with or without LangSmith.

Why LangSmith rather than Langfuse or OpenTelemetry: the pipeline has no LangChain, so any of them needs
manual spans; LangSmith's `trace` context manager does exactly that, shows LLM runs with their prompts
and token counts, and its datasets can store the evaluation questions. Langfuse is the self-hosted
choice if questions must not leave your machine; the Span wrapper below is the single place to change.
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

MAX_CHARS = 2000


@dataclass
class Span:
    name: str
    run: Any = None                     # the LangSmith RunTree, when tracing is on
    outputs: dict | None = None
    metadata: dict = field(default_factory=dict)
    ms: float = 0.0

    def set(self, outputs: dict | None = None, **metadata) -> None:
        """Record what this stage produced (sent when the span closes)."""
        if outputs is not None:
            self.outputs = {**(self.outputs or {}), **outputs}
        self.metadata.update(metadata)


_stack: contextvars.ContextVar[tuple[Span, ...]] = contextvars.ContextVar("saqya_spans", default=())
_timings: contextvars.ContextVar[list | None] = contextvars.ContextVar("saqya_timings", default=None)


class Tracer:
    def __init__(self) -> None:
        self.enabled = False
        self.project = "saqya-ai"
        self.log_text = True
        self.passage_chars = 200
        self._client = None
        self._trace = None
        self._context = None
        self._failed: str | None = None
        self.usage_sink: list | None = None   # scripts/evaluate.py collects token counts here

    # ------------------------------------------------------------------ setup
    def configure(self, enabled: bool = False, api_key: str | None = None, project: str = "saqya-ai",
                  endpoint: str | None = None, log_text: bool = True, passage_chars: int = 200) -> bool:
        """Turn LangSmith on when asked for, installed and given a key; returns whether it is on."""
        self.project, self.log_text, self.passage_chars = project, log_text, max(0, int(passage_chars))
        self.enabled, self._failed = False, None
        if not enabled:
            return False
        if not api_key:
            logger.warning("LANGSMITH_TRACING_ENABLED=true but LANGSMITH_API_KEY is empty: tracing stays off")
            return False
        try:
            from langsmith import Client
            from langsmith.run_helpers import trace, tracing_context

            self._client = Client(api_key=api_key, **({"api_url": endpoint} if endpoint else {}))
            self._trace, self._context = trace, tracing_context
        except Exception as exc:  # noqa: BLE001 - not installed, bad endpoint...
            logger.warning("LangSmith tracing could not start (%s): pip install langsmith", exc)
            return False
        self.enabled = True
        logger.info("LangSmith tracing on (project %s)", project)
        return True

    @property
    def active(self) -> bool:
        return self.enabled and not self._failed

    def _fail(self, exc: Exception) -> None:
        if not self._failed:
            self._failed = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning("LangSmith tracing error (%s); tracing is off until the backend restarts", self._failed)

    # ------------------------------------------------------------------ privacy
    def text(self, value: str | None) -> str | None:
        if value is None:
            return None
        if not self.log_text:
            return f"<{len(value)} characters>"
        return value[:MAX_CHARS]

    def passages(self, chunks) -> list[dict]:
        """Passages as they may leave the machine: where they come from and how they scored."""
        out = []
        for c in chunks or []:
            item = {"id": getattr(c, "chunk_id", ""), "source": getattr(c, "source", ""),
                    "pages": f"{getattr(c, 'page_start', '')}-{getattr(c, 'page_end', '')}",
                    "section": getattr(c, "section", "")[:120], "similarity": getattr(c, "score", None)}
            if getattr(c, "rerank_score", None) is not None:
                item["rerank_score"] = c.rerank_score
            if self.passage_chars:
                item["text"] = (getattr(c, "text", "") or "")[: self.passage_chars]
            out.append(item)
        return out

    def _clean(self, value: Any, depth: int = 0) -> Any:
        if isinstance(value, str):
            return value[:MAX_CHARS]
        if isinstance(value, dict) and depth < 4:
            return {str(k): self._clean(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)) and depth < 4:
            return [self._clean(v, depth + 1) for v in list(value)[:50]]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)[:MAX_CHARS]

    # ------------------------------------------------------------------ spans
    @contextmanager
    def request(self) -> Iterator[list]:
        """Collect the stage timings of one request; logs them when it ends."""
        timings: list = []
        token = _timings.set(timings)
        try:
            yield timings
        finally:
            _timings.reset(token)
            if timings:
                logger.info("Stages: %s", " | ".join(f"{name} {ms:.0f}ms" for name, ms in timings))

    @contextmanager
    def span(self, name: str, run_type: str = "chain", inputs: dict | None = None,
             metadata: dict | None = None) -> Iterator[Span]:
        span = Span(name, metadata=dict(metadata or {}))
        parents = _stack.get()
        started = time.perf_counter()
        context = run_cm = None
        if self.active:
            try:
                if not parents:   # the root span switches tracing on for everything nested in it
                    context = self._context(enabled=True, project_name=self.project, client=self._client)
                    context.__enter__()
                run_cm = self._trace(name=name, run_type=run_type, inputs=self._clean(inputs or {}),
                                     project_name=self.project, metadata=self._clean(span.metadata),
                                     client=self._client)
                span.run = run_cm.__enter__()
            except Exception as exc:  # noqa: BLE001
                self._fail(exc)
                run_cm = None
        token = _stack.set((*parents, span))
        error: BaseException | None = None
        try:
            yield span
        except BaseException as exc:
            error = exc
            raise
        finally:
            _stack.reset(token)
            span.ms = (time.perf_counter() - started) * 1000
            timings = _timings.get()
            if timings is not None:
                timings.append((name, span.ms))
            if run_cm is not None:
                try:
                    if span.run is not None:
                        if span.metadata:
                            span.run.add_metadata(self._clean(span.metadata))
                        if error is None:
                            span.run.end(outputs=self._clean(span.outputs or {}))
                    run_cm.__exit__(type(error) if error else None, error, error.__traceback__ if error else None)
                except Exception as exc:  # noqa: BLE001
                    self._fail(exc)
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                except Exception as exc:  # noqa: BLE001
                    self._fail(exc)

    @staticmethod
    def current() -> Span | None:
        stack = _stack.get()
        return stack[-1] if stack else None

    def usage(self, model: str, prompt_tokens: int | None, completion_tokens: int | None, **extra) -> None:
        """Token counts of an LLM call, attached to the innermost open span."""
        usage = {k: v for k, v in {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}.items()
                 if isinstance(v, int)}
        if usage:
            usage["total_tokens"] = sum(usage.values())
        if self.usage_sink is not None:
            self.usage_sink.append({"model": model, **usage})
        span = self.current()
        if span is not None:
            span.metadata.setdefault("llm_calls", []).append({"model": model, **usage, **extra})


tracer = Tracer()
