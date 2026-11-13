"""Prompt building, citation handling and the call to the local Ollama LLM.

Bilingual: the question's language (Arabic or English) decides the answer's
language, whatever language the retrieved passages are in.

This module has no dependency on the app settings, so the notebook imports it
directly: the evaluation in the notebook runs the exact prompt the API serves.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Sequence

import httpx
import ollama

from app.services.glossary import add_missing_terms, find_terms, glossary_hint
from app.services.grounding import NO_ANSWER, citations, is_refusal
from app.services.retrieval import RetrievedChunk
from app.utils.text import LANGUAGE_NAMES, detect_language

logger = logging.getLogger(__name__)

# Returned without calling the LLM when retrieval finds nothing relevant, and in place of any
# refusal the LLM writes (so the farmer always gets the same clear message and advice)
NO_CONTEXT_ANSWER = {
    "en": f"{NO_ANSWER['en']} Try rephrasing your question with the crop or problem name, "
    "or ask your local agricultural extension officer.",
    "ar": f"{NO_ANSWER['ar']} حاول إعادة صياغة سؤالك مع ذكر اسم المحصول أو المشكلة، "
    "أو استشر المرشد الزراعي في منطقتك.",
}

# Short, numbered rules work better than long prose for small local models (3-7B).
SYSTEM_PROMPT_TEMPLATE = """You are SAQYA (ساقية), an agricultural advisor for farmers in Egypt. You answer the farmer's question using ONLY the numbered passages that come with it.

How to write:
1. Write the answer in {language}, even when the passages are in another language. Use clear, simple words a farmer understands.
2. Start with the direct answer. Do not repeat the question.
3. Use only what the passages say that answers THIS question. Leave out everything else, such as points meant for the ministry, researchers or another crop.
4. Copy numbers, quantities, doses, dates and units exactly as the passages write them. Never calculate, convert or guess a number.
5. After every sentence put the number of the passage it comes from in square brackets, like [1] or [2][3]. The farmer never sees these numbers, so never mention passages, documents or sources in your words (do not write "according to passage 2" or "the document says").
6. Write two to five short sentences. Use a bullet list (lines starting with "- ") only for steps or several separate items, and do not start every bullet with the same word.
7. If the question compares two or more things, first say what the passages say about each one, then compare them.
8. If none of the passages answers the question, reply with exactly: "{no_answer}"

Example:
Passages:
[1] Thinning is done 20 days after sowing, leaving one plant per hill.
[2] Irrigate every 12 days in summer.
Question: When should I thin the plants?
Answer: Thin the plants 20 days after sowing, leaving one plant in each hill [1]."""

USER_PROMPT_TEMPLATE = """Passages:

{context}

{earlier}Question: {question}
{translation}
Answer in {language} using only the passages, with [n] after every sentence.
{language_line}"""

# The last line of every prompt, written in the answer's own language: a small model follows the language of the
# passages unless told — in that language — what to write in
LANGUAGE_LINE = {
    "en": "Write the whole answer in English only, even if the passages are in Arabic. Do not use any other language.",
    "ar": "اكتب الإجابة كاملة باللغة العربية فقط، حتى لو كانت الفقرات بالإنجليزية. لا تستخدم أي لغة أخرى.",
}

# A follow-up question ("how much water does it need?"): the farmer's earlier question, so "it" can be understood
EARLIER_LINE = "Earlier question in this conversation (for context only): {earlier}\n"

# When the passages may be in the other language, the question is also shown translated, so a small
# model can match "seed drill" to "التسطير" (the same translation the search used)
TRANSLATION_LINE = "(The same question in {other}: {translation})\n"

# Sent back to the LLM (with its first answer) when the answer has a problem it can fix
FEEDBACK = {
    "garbled": "Your answer has broken text: words of two languages mixed or glued together, repeated words, or your "
               "instructions repeated. Write it again in clean, correct {language} only, in full sentences, using "
               "only what the passages say, with [n] after every sentence.",
    "echo": "You only repeated the question. Now answer it: give the facts from the passages that answer it, "
            "with [n] after every sentence.",
    "numbers": "These numbers are not in the passages: {numbers}. Rewrite the answer using only numbers "
               "written in the passages, copied exactly, with [n] after every sentence.",
    "uncited": "Rewrite the answer with the passage number, like [1], after every sentence.",
    "language": "Your answer is not in {language}. Write it in {language}, translating the facts from the "
                "passages faithfully, with [n] after every sentence.",
    "crop": "The question is about {crop}, but the passages you used are not about {crop}. Answer only from "
            "passages about {crop}, with [n] after every sentence. If none of them answers, reply with "
            "exactly: \"{no_answer}\"",
    "units": "The question asks for the amount per {unit}, but your answer gives it per {other}. Answer from a "
             "passage that gives the amount per {unit}, copying its numbers exactly; do not convert units. If no "
             "passage gives it per {unit}, say the documents give it per {other}, with [n].",
    "unsupported": "Some of your sentences say things that are not in the passages. Rewrite the answer using "
                   "only what the passages say, with [n] after every sentence.",
    "refusal": "Read the passages again carefully. If any passage answers the question, even partly, answer "
               "with what it says, with [n] after every sentence. If none of them does, reply with exactly: "
               "\"{no_answer}\"",
}


# Explaining a document the farmer uploaded (documents.py): the passages are a sample of the whole document
EXPLAIN_SYSTEM_PROMPT_TEMPLATE = """You are SAQYA (ساقية), an agricultural advisor for farmers in Egypt. A farmer attached a document and wants it explained simply. You get numbered passages from that document.

How to write:
1. Write in {language}, even when the passages are in another language. Use clear, simple words a farmer understands.
2. Start with one or two sentences saying what the document (or the requested part) is about.
3. Then give the main points that are useful to a farmer as 3 to 7 short bullet points (lines starting with "- "). Do not start every bullet with the same word.
4. Leave out the authors, the publisher, the publication details and recommendations meant for the government, the ministry or researchers.
5. Copy numbers, quantities, doses, dates and units exactly as the passages write them. Never calculate, convert or guess a number.
6. After every sentence and bullet put the number of the passage it comes from, like [1] or [2][3]. The farmer never sees these numbers, so never mention passages in your words."""

EXPLAIN_USER_TEMPLATE = """Passages from the document "{title}":

{context}

{request}
Write in {language}, with [n] after every sentence and bullet.
{language_line}"""

EXPLAIN_REQUESTS = {
    None: "Explain this document: what it is about and its main practical points for a farmer.",
    "recommendations": "List the practical recommendations this document gives a farmer: what to do, when and how much. "
                       "Start with one sentence on what the document is about, then the recommendations as bullet points.",
    "focus": "Explain this part of the document: {focus}. Say what it covers and its practical recommendations.",
}


def build_explain_messages(chunks: Sequence[RetrievedChunk], language: str, title: str,
                           focus: str | None = None) -> list[dict]:
    if focus in EXPLAIN_REQUESTS:
        request = EXPLAIN_REQUESTS[focus]
    else:
        request = EXPLAIN_REQUESTS["focus"].format(focus=focus) if focus else EXPLAIN_REQUESTS[None]
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT_TEMPLATE.format(language=LANGUAGE_NAMES[language])},
        {"role": "user", "content": EXPLAIN_USER_TEMPLATE.format(
            title=title, context=build_context(chunks), request=request, language=LANGUAGE_NAMES[language],
            language_line=LANGUAGE_LINE[language])},
    ]


# ---------------------------------------------------------------- beyond the documents (beyond.py)
# Does an answer address the question? (asked only for questions that name no crop)
RELEVANCE_PROMPT = """A farmer asked a question and got an answer. Does the answer give what the farmer asked for?
Say "no" if it talks about a different matter than the one asked (for example the farmer asks which crops to grow and the answer explains how to plough).
Say "yes" if it answers the question, even partly.

Question: {question}
Answer: {answer}

Reply with one word: yes or no."""

# A short web search query for a question the documents don't cover
WEB_QUERY_PROMPT = """Write one web search query that would find the answer to this farmer's question.
The farmer is in Egypt. Today is {today} ({season} in Egypt).
- At most 10 words, in English.
- Name the crop, animal or problem. Add "Egypt" when the answer depends on the place, the season or the market.
- Reply with the query only, no quotes.
{earlier}Question: {question}"""

SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}

# Last resort, when no document, learned passage or web page answers a farming question (marked in the chat app)
GENERAL_SYSTEM_PROMPT = """You are SAQYA (ساقية), an agricultural advisor for farmers in Egypt. No document or web page was found for this question, so you answer from general agricultural knowledge.

Rules:
1. Write the answer in {language}, in clear, simple words a farmer understands.
2. Start with the direct answer. Keep it practical: two to six sentences, or short bullet points (lines starting with "- ") for steps.
3. Never invent exact doses, rates, prices or product names. Give a range only when it is standard and widely known.
4. If the answer depends on the farmer's situation (soil, water, region, season, market), say what it depends on and give the common options in Egypt.
5. Never reply that you don't have enough information: give the most useful general advice you can. If more details would help (the crop, the land, the water, the region), ask for them in one short question at the end.
6. End with one short sentence advising to confirm with the local agricultural extension officer."""

GENERAL_USER_TEMPLATE = """Today is {today} ({season} in Egypt).
{earlier}Question: {question}
{language_line}"""


class LLMUnavailableError(RuntimeError):
    """Raised when Ollama cannot be reached or the model is missing."""


# ---------------------------------------------------------------- prompting
def build_system_prompt(language: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        language=LANGUAGE_NAMES[language], no_answer=NO_ANSWER[language]
    )


# Lines in the indexed text that only help the index, never a farmer: the encyclopedia's record and reference lines
# ("AGR-0051 | ARC-TOM · UC-TOM :سجل المعرفة", "المراجع: FAO-56"), and the contents line each document starts with
# ("يتناول هذا المستند: مقدمة، ...", "This document covers: ..."). Left in, a small model copies them into answers.
_JUNK_LINE_RE = re.compile(
    r"(?m)^[^\n]*(?:سجل المعرفة|المراجع\s*:|:\s*المراجع|^\s*يتناول هذا المستند|^\s*This document covers)[^\n]*\n?")
_CODES_ONLY_LINE_RE = re.compile(r"(?m)^[\sA-Z0-9\-|·:,;]{0,80}\b[A-Z]{2,5}-(?:[A-Z]{2,5}|\d{2,5})\b[\sA-Z0-9\-|·:,;]*$\n?")


def clean_passage(text: str) -> str:
    """A passage as the LLM (and the farmer, under "evidence") should see it: without record, reference and
    contents lines."""
    cleaned = _CODES_ONLY_LINE_RE.sub("", _JUNK_LINE_RE.sub("", text))
    return cleaned.strip() or text.strip()


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Numbered passages. The document's crop is named, since a passage often doesn't repeat it."""
    def header(i: int, chunk: RetrievedChunk) -> str:
        topic = getattr(chunk, "topic", "")
        return f"[{i}] (source: {chunk.label}" + (f"; about: {topic})" if topic else ")")
    return "\n\n".join(f"{header(i, c)}\n{clean_passage(c.text)}" for i, c in enumerate(chunks, 1))


def build_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    language: str | None = None,
    previous: str | None = None,
    feedback: str | None = None,
    translation: str | None = None,
    earlier: str | None = None,
) -> list[dict]:
    """Chat messages for one question. With previous + feedback, the LLM sees its first answer
    and what was wrong with it, and is asked to try again. `earlier`: the question this one follows up on."""
    language = language or detect_language(question)
    other = "en" if language == "ar" else "ar"
    translation_line = (
        TRANSLATION_LINE.format(other=LANGUAGE_NAMES[other], translation=translation) if translation else ""
    )
    messages = [
        {"role": "system", "content": build_system_prompt(language)},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                context=build_context(chunks),
                question=question,
                translation=translation_line,
                language=LANGUAGE_NAMES[language],
                earlier=EARLIER_LINE.format(earlier=earlier) if earlier else "",
                language_line=LANGUAGE_LINE[language],
            ),
        },
    ]
    if previous is not None and feedback:
        messages += [{"role": "assistant", "content": previous}, {"role": "user", "content": feedback}]
    return messages


def no_context_answer(question: str) -> str:
    return NO_CONTEXT_ANSWER[detect_language(question)]


# ---------------------------------------------------------------- citations
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_citations(answer: str, num_chunks: int) -> list[int]:
    """Return the passage numbers cited in the answer, in order of first use."""
    return citations(answer, num_chunks)


def select_passages(answer: str, chunks: Sequence[RetrievedChunk]) -> list[tuple[int, RetrievedChunk]]:
    """(n, passage) pairs the farmer is shown; n matches the answer's [n] citations."""
    if not chunks or is_refusal(answer):
        return []
    cited = citations(answer, len(chunks))
    numbers = cited or range(1, len(chunks) + 1)  # model forgot to cite: show all
    return [(n, chunks[n - 1]) for n in numbers]


def select_sources(answer: str, chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Sources to show the farmer. The [n] prefix matches the answer's citations."""
    return [f"[{n}] {c.label}" for n, c in select_passages(answer, chunks)]


def clean_llm_output(text: str) -> str:
    """Drop <think>…</think> blocks that some reasoning models put in the answer."""
    return _THINK_RE.sub("", text).strip()


TRANSLATE_PROMPT = """Translate this farmer's question into {language}, using {script} script only.
Keep crop, pest, disease and product names accurate, using the names farmers in Egypt use.
{hint}Reply with the translated question only: no quotes, notes or explanations.

Examples:
Question: When should sugar beet be planted? -> متى يزرع بنجر السكر؟
Question: ما هو معدل تقاوي القمح للفدان؟ -> What is the wheat seed rate per feddan?

Question: {question} ->"""

SCRIPTS = {"ar": "Arabic", "en": "Latin"}

# Last resort when an answer still comes back in the wrong language (see RAGService): translate it, keeping
# the [n] markers and every number exactly
ANSWER_TRANSLATE_PROMPT = """Translate this answer for a farmer into {language}, using {script} script only.
Rules:
- Keep every [n] marker exactly as it is, after the same sentence.
- Copy every number, quantity, date and unit exactly; do not convert or round anything.
- Use accurate crop, pest and product names.{hint}
- Translate only: add nothing, remove nothing, no notes.
Reply with the translated answer only.

Answer:
{answer}"""
# letters from scripts that should never appear in an Arabic or English query (Chinese, Japanese,
# Korean, Cyrillic...): small Qwen models sometimes slip into Chinese mid-sentence
_FOREIGN_SCRIPT_RE = re.compile("[Ѐ-ӿ฀-๿぀-ヿ㐀-䶿一-鿿가-힯]")


def clean_translation(text: str, target: str) -> str | None:
    """First line of the model's reply, or None if it is empty, mixes in another script,
    or is not in the target language."""
    line = next((l.strip().strip('"«»“”') for l in clean_llm_output(text).splitlines() if l.strip()), "")
    line = re.sub(r"^(?:question|translation|السؤال|الترجمة)\s*[:：]\s*", "", line, flags=re.IGNORECASE).lstrip("-> ")
    if not line or len(line) > 400 or _FOREIGN_SCRIPT_RE.search(line) or detect_language(line) != target:
        return None
    return line


# ---------------------------------------------------------------- the LLM
class OllamaGenerator:
    def __init__(
        self,
        host: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        think: bool | None = None,
        num_ctx: int = 8192,
    ) -> None:
        self.host = host
        self.model = model
        self.temperature = temperature
        self.think = think  # None = model default; False turns off "thinking" (qwen3)
        # Ollama's default context window (2048-4096 tokens, depending on the version) is too small
        # for six Arabic passages + the instructions: the start of the prompt (the rules!) would be
        # cut off silently. 8192 tokens fits comfortably.
        self.num_ctx = num_ctx
        self.keep_alive = "30m"   # keep the model in memory between questions (no reload wait)
        self._client = ollama.Client(host=host, timeout=timeout)
        self._status_client = ollama.Client(host=host, timeout=3)

    def status(self) -> tuple[bool, bool]:
        """(Ollama reachable, model pulled)."""
        try:
            names = {m.model for m in self._status_client.list().models}
        except Exception:  # connection refused, timeout, ...
            return False, False
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return True, wanted in names or self.model in names

    def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        language: str | None = None,
        previous: str | None = None,
        feedback: str | None = None,
        translation: str | None = None,
        earlier: str | None = None,
    ) -> str:
        return self._chat(build_messages(question, chunks, language, previous, feedback, translation, earlier))

    def explain(self, chunks: Sequence[RetrievedChunk], language: str, title: str, focus: str | None = None) -> str:
        """An explanation of an uploaded document (or of one part of it) from a sample of its passages."""
        return self._chat(build_explain_messages(chunks, language, title, focus))

    def _chat(self, messages: list[dict]) -> str:
        kwargs = {"think": self.think} if self.think is not None else {}
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature, "num_ctx": self.num_ctx, "seed": 7,
                         "repeat_penalty": 1.15, "num_predict": 700},
                keep_alive=self.keep_alive,
                **kwargs,
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMUnavailableError(
                    f"Model '{self.model}' is not available. Run: ollama pull {self.model}"
                ) from exc
            raise LLMUnavailableError(f"Ollama error: {exc.error}") from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?"
            ) from exc
        return clean_llm_output(response.message.content or "")

    # ------------------------------------------------------------ beyond the documents
    def classify_scope(self, prompt: str) -> str:
        """A one-word yes/no reply (scope.py: is the question about agriculture?)."""
        kwargs = {"think": self.think} if self.think is not None else {}
        try:
            response = self._client.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                         options={"temperature": 0.0, "num_predict": 4, "num_ctx": self.num_ctx},
                                         **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailableError(str(exc)) from exc
        return clean_llm_output(response.message.content or "")

    def answers_question(self, question: str, answer: str) -> bool | None:
        """Does this answer address what the farmer asked? True / False, or None when the model can't say.
        Used for questions that name no crop, where the search can return passages that are only loosely related
        ("new crops for my land" answered with how to plough)."""
        prompt = RELEVANCE_PROMPT.format(question=question.strip()[:500], answer=answer.strip()[:900])
        kwargs = {"think": self.think} if self.think is not None else {}
        try:
            response = self._client.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                         options={"temperature": 0.0, "num_predict": 4, "num_ctx": self.num_ctx, "seed": 7},
                                         keep_alive=self.keep_alive, **kwargs)
        except Exception as exc:  # noqa: BLE001 - can't ask: the answer is kept
            logger.warning("Relevance check failed: %s", exc)
            return None
        word = re.sub(r"[^a-zA-Z\u0600-\u06FF]+", " ", clean_llm_output(response.message.content or "")).strip().lower().split(" ")[0]
        return True if word in ("yes", "نعم") else False if word in ("no", "لا") else None

    def web_query(self, question: str, earlier: str | None = None, today=None) -> str | None:
        """A short English web search query for the question (None when the reply is unusable)."""
        today = today or datetime.now()
        prompt = WEB_QUERY_PROMPT.format(today=f"{today:%d %B %Y}", season=SEASONS[today.month], question=question,
                                         earlier=f"Earlier question (context): {earlier}\n" if earlier else "")
        kwargs = {"think": self.think} if self.think is not None else {}
        try:
            response = self._client.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                         options={"temperature": 0.0, "num_predict": 40, "num_ctx": self.num_ctx},
                                         **kwargs)
        except Exception as exc:  # noqa: BLE001 - the question itself is searched instead
            logger.warning("Web query writing failed: %s", exc)
            return None
        text = clean_llm_output(response.message.content or "").strip().splitlines()
        query = re.sub(r'^(query|search)\s*:\s*', "", text[0] if text else "", flags=re.I).strip(' "\'`')
        return query if 2 <= len(query.split()) <= 16 and not _FOREIGN_SCRIPT_RE.search(query) else None

    def general_answer(self, question: str, language: str, earlier: str | None = None, today=None) -> str:
        """An answer from the model's own farming knowledge (no passages), for the last tier."""
        today = today or datetime.now()
        messages = [
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT.format(language=LANGUAGE_NAMES[language])},
            {"role": "user", "content": GENERAL_USER_TEMPLATE.format(
                today=f"{today:%d %B %Y}", season=SEASONS[today.month], question=question,
                earlier=EARLIER_LINE.format(earlier=earlier) if earlier else "", language_line=LANGUAGE_LINE[language])},
        ]
        return self._chat(messages)

    def translate_answer(self, answer: str, target: str) -> str | None:
        """The answer translated into 'ar' or 'en' (None when the model's reply is unusable)."""
        source = "en" if target == "ar" else "ar"
        terms = find_terms(answer, source)
        hint = glossary_hint(terms, target).strip()
        prompt = ANSWER_TRANSLATE_PROMPT.format(language=LANGUAGE_NAMES[target], script=SCRIPTS[target],
                                                hint=f" {hint}" if hint else "", answer=answer)
        try:
            text = self._chat([{"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001 - Ollama down etc.: keep the answer as it is
            logger.warning("Answer translation failed: %s", exc)
            return None
        text = text.strip().strip('"')
        if not text or _FOREIGN_SCRIPT_RE.search(text) or detect_language(text) != target:
            return None
        return text

    def translate(self, question: str, target: str) -> str | None:
        """Translate a question into 'ar' or 'en' for searching.

        Crop and practice names come from a fixed glossary, so a wrong or missing name from the
        small LLM can't send the search to the wrong crop. Returns None when there is nothing
        usable, so a translation problem never blocks an answer (the original is still searched).
        """
        source = "en" if target == "ar" else "ar"
        terms = find_terms(question, source)
        kwargs = {"think": self.think} if self.think is not None else {}
        translated = None
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": TRANSLATE_PROMPT.format(
                        language=LANGUAGE_NAMES[target], script=SCRIPTS[target],
                        hint=glossary_hint(terms, target), question=question,
                    ),
                }],
                options={"temperature": 0.0, "num_predict": 120, "num_ctx": self.num_ctx},  # same num_ctx: no model reload
                **kwargs,
            )
            translated = clean_translation(response.message.content or "", target)
        except Exception as exc:  # Ollama down, model missing, timeout...
            logger.warning("Query translation failed: %s", exc)
        if translated is None:
            # unusable LLM output: still search with the glossary terms, if the question had any
            fallback = " ".join(t.ar if target == "ar" else t.en for t in terms)
            return fallback or None
        return add_missing_terms(translated, terms, target)
