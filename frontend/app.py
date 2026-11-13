"""SAQYA - ساقية (SAQYA AI) — bilingual (English / Arabic) Streamlit app for Egyptian farmers.

Two views:
- Chat: questions about the library of agricultural bulletins, typed or spoken (Egyptian Arabic works: the
  backend's Whisper model writes it down and the question is rewritten into standard Arabic for the search).
  Answers come in the language of the question and are spoken live with an Egyptian voice. The ＋ in the chat
  box attaches a PDF / Word / text file: the backend cleans it (cover, contents, headers… removed), keeps a clean
  PDF of it for this chat session, and every question is answered from that document until it is removed —
  follow-up questions included; what it doesn't cover is answered from the library, and said so.
- Settings: voice, and the conversation (download / delete).

Every answer shows its sources and the passages behind it, and can be rated. Look and feel follow DESIGN.md
(see theme.py); all text is in ui_text.py.

Run:  cd frontend && streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import uuid
import weakref
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

from api_client import AgriAPIClient, APIError, ConfigError
from theme import (BASE_CSS, BRAND_NAME, BRAND_NAME_AR, HERO_HTML, LIVE_VOICE_HTML, PLAYER_JS, PLAYER_VERSION, RTL_CSS,
                   WHEEL_SVG)
from ui_text import TEXT

ICON = Path(__file__).parent / "assets" / "saqya-icon.png"  # the water wheel, for the browser tab and the avatar
st.set_page_config(page_title=f"{BRAND_NAME} - {BRAND_NAME_AR} · Farming Q&A", page_icon=str(ICON) if ICON.exists() else "🌾",
                   layout="wide", initial_sidebar_state="collapsed")

DOC_TYPES = ["pdf", "docx", "txt", "md"]
BOT_AVATAR = str(ICON) if ICON.exists() else "🌾"
MAX_UPLOAD_MB = 25  # keep in step with the backend's MAX_UPLOAD_MB
ARABIC_RE = re.compile("[؀-ۿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile("[A-Za-z]")
CITATION_RE = re.compile(r"\[(\d{1,2})\]")
SOURCE_RE = re.compile(r"^(\[\d+\])\s+(.*?)(, pp?\. [\d\-]+)$")


def text_lang(text: str) -> str:
    return "ar" if len(ARABIC_RE.findall(text)) > len(LATIN_RE.findall(text)) else "en"


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def new_id() -> str:
    return uuid.uuid4().hex[:10]


# ---------------------------------------------------------------- session state
for key, default in {"messages": [], "ui_lang": "en", "view": "chat", "doc": None, "doc_pdf": None,
                     "context_start": 0, "read_aloud": "voice", "speech_lang": "auto",
                     "voice_engine": "gemini", "speech_style": "egyptian", "voice_name": "kore",   # Gemini when the server has it, else Edge
                     "use_web": True, "use_learn": True, "use_general": True}.items():
    st.session_state.setdefault(key, default)
st.session_state.setdefault("session_id", uuid.uuid4().hex)   # this chat, for its temporary documents on the backend

LANG = st.session_state.ui_lang or "en"
T = TEXT[LANG]
RTL = LANG == "ar"
st.markdown(BASE_CSS, unsafe_allow_html=True)
if RTL:
    st.markdown(RTL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- API client
@st.cache_resource
def get_client() -> AgriAPIClient:
    return AgriAPIClient()


try:
    client = get_client()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()
# The browser plays the spoken answer straight from the backend (GET /voice/stream). Set PUBLIC_API_URL in
# frontend/.env when the browser reaches the backend at another address than this app does (Docker, a server).
VOICE_STREAM_URL = (os.getenv("PUBLIC_API_URL") or client.base_url).rstrip("/") + "/voice/stream"


def _forget_session(base_url: str, session: str) -> None:
    """The chat is gone: delete its temporary documents on the backend (they also expire on their own)."""
    try:
        requests.delete(f"{base_url}/documents", params={"session": session}, timeout=5)
    except Exception:  # noqa: BLE001 - the backend may be down too
        pass


class _SessionGuard:
    """Kept in the session state: Streamlit drops it when the browser tab is closed, and the finalizer runs."""


if "session_guard" not in st.session_state:
    _guard = _SessionGuard()
    weakref.finalize(_guard, _forget_session, client.base_url, st.session_state.session_id)
    st.session_state.session_guard = _guard


@st.cache_data(ttl=30, show_spinner=False)
def _cached_health() -> dict:
    return client.health()


def backend_health() -> dict:
    """The backend's features (cached 30 s). A failure isn't cached: the next click asks again, so a backend that
    was still starting doesn't leave the voice and attachments switched off."""
    try:
        return _cached_health()
    except APIError as exc:
        return {"status": "unreachable", "kind": exc.kind, "detail": exc.detail}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_library() -> dict:
    return client.library()


def backend_library() -> dict | None:
    try:
        return _cached_library()
    except APIError:
        return None  # backend down, or an older backend without GET /library


def error_text(exc: APIError) -> str:
    message = T["errors"].get(exc.kind, T["errors"]["server"])
    return f"{message}\n\n`{exc.detail}`" if exc.kind in ("unavailable", "invalid") and exc.detail else message


health = backend_health()
library = backend_library()
CAN_LISTEN = bool(health.get("speech_to_text"))   # the server can transcribe speech
CAN_SPEAK = bool(health.get("text_to_speech"))    # the server can read answers aloud
CAN_UPLOAD = bool(health.get("documents"))        # the server can clean and index an attached document
CAN_WEB = bool(health.get("web_search"))          # questions the documents don't cover can be searched online
CAN_LEARN = bool(health.get("learning"))          # what the web and attached documents say is kept
CAN_GENERAL = bool(health.get("general_knowledge"))
ENGINES = health.get("voice_engines") or (["edge"] if CAN_SPEAK else [])
VOICES = {"edge": ["salma", "shakir", "zariyah", "hamed"], "gemini": ["kore", "aoede", "despina", "charon", "orus", "puck"]}


def voice_choice() -> tuple[str, str, str]:
    """(engine, voice, style) to speak with: the farmer's choice, kept to what the server can do."""
    engine = st.session_state.voice_engine if st.session_state.voice_engine in ENGINES else "edge"
    voice = st.session_state.voice_name if st.session_state.voice_name in VOICES[engine] else VOICES[engine][0]
    return engine, voice, st.session_state.speech_style


def find_message(msg_id: str) -> dict | None:
    return next((m for m in st.session_state.messages if m.get("id") == msg_id), None)


# ---------------------------------------------------------------- hero
def stat_values() -> list[tuple[str, str]]:
    n_docs = library["num_documents"] if library else None
    n_chunks = (library or {}).get("num_chunks") or health.get("num_chunks")
    n_crops = len((library or {}).get("crops", [])) if library else None
    n_langs = len((library or {}).get("languages", {})) if library else None
    fmt = lambda v: f"{v:,}" if isinstance(v, int) else "—"  # noqa: E731
    s = T["stats"]
    return [(fmt(n_docs), s["documents"]), (fmt(n_chunks), s["passages"]), (fmt(n_crops), s["crops"]),
            (fmt(n_langs), s["languages"])]


def arabic_safe(text: str) -> str:
    """Escape, and wrap Arabic words so tracked (letter-spaced) labels don't break their joined letters."""
    return re.sub(r"([؀-ۿ][؀-ۿ\s]*[؀-ۿ]|[؀-ۿ])",
                  r'<span class="ar" lang="ar">\1</span>', esc(text))


def render_hero() -> None:
    stats = "".join(f'<div><div class="num">{esc(v)}</div><div class="lab">{esc(label)}</div></div>'
                    for v, label in stat_values())
    page = (HERO_HTML.replace("__DIR__", "rtl" if RTL else "ltr")
            .replace("__EYEBROW__", arabic_safe(T["eyebrow"]) if not RTL else esc(T["eyebrow"]))
            .replace("__TITLE__", esc(T["hero_title"])).replace("__SUB__", esc(T["hero_sub"]))
            .replace("__STATS__", stats))
    st.iframe(page, height=470)


# ---------------------------------------------------------------- answers: rendering
def page_label(pages: str) -> str:
    return str(pages or "").replace("pp.", T["page"]).replace("p.", T["page"]).strip()


def source_chips(sources: list[str], passages: list[dict]) -> str:
    """One chip per document page or web page the answer used (no numbers: the answer text has none either).
    Web pages open in a new tab; a file name is isolated so an Arabic name doesn't scramble the chip."""
    chips, seen = [], set()
    for p in passages:
        if p.get("url"):
            key = p["url"]
            title = p.get("title") or p.get("source", "")
            site = p.get("source", "")
            label = f'<bdi dir="auto">{esc(title[:70])}</bdi>' + (
                f'<span class="site">{esc(site)}</span>' if site and site != title else "")
            chip = (f'<a class="source-chip" dir="ltr" href="{esc(key)}" target="_blank" rel="noopener noreferrer">'
                    f'{label}</a>')
        else:
            key = (p.get("source", ""), p.get("pages", ""))
            pages = page_label(p.get("pages", ""))
            chip = (f'<span class="source-chip" dir="ltr"><bdi dir="ltr">{esc(p.get("source", ""))}</bdi>'
                    + (f", {esc(pages)}" if pages else "") + "</span>")
        if key not in seen:
            seen.add(key)
            chips.append(chip)
    if not chips:   # an older backend without passages: the source strings, without their [n]
        for source in sources:
            m = SOURCE_RE.match(source)
            body = (f'<bdi dir="ltr">{esc(m.group(2))}</bdi>{esc(page_label(m.group(3)).replace(",", ", ", 1))}' if m
                    else f"<bdi>{esc(re.sub(r'^\[\d+\]\s*', '', source))}</bdi>")
            if body not in seen:
                seen.add(body)
                chips.append(f'<span class="source-chip" dir="ltr">{body}</span>')
    return "".join(chips)


LIST_LINE_RE = re.compile(r"(?m)^(\s*(?:\d{1,2}[.)]|[-*•])\s.*)\n(?=\s*\S)(?!\s*(?:\d{1,2}[.)]|[-*•])\s)")


SPACE_CITATION_RE = re.compile(r"\s*\[\d{1,2}(?:\s*[,،]\s*\d{1,2})*\]")


def plain_answer(answer: str) -> str:
    """The answer without its [n] markers (they link sentences to passages inside the backend; a farmer reads the
    sources under the answer instead)."""
    text = SPACE_CITATION_RE.sub("", answer or "")
    return re.sub(r"[ \t]+([.,;:!?،؛؟])", r"\1", text).strip()


def answer_html(answer: str) -> str:
    """Escape the model's text (it is shown as Markdown), without [n] markers. A line after a list item gets an
    empty line first, or Markdown would glue it onto that item."""
    answer = LIST_LINE_RE.sub(r"\1\n\n", plain_answer(answer))
    return html.escape(answer, quote=False)


def thinking_html(lang: str, doc: bool) -> str:
    """While the answer is being found: a few words that take turns ("Thinking…", "Contemplating…"), in a new order
    each time, beside three pulsing dots."""
    words = list(TEXT[lang]["thinking_doc_words" if doc else "thinking_words"])
    random.shuffle(words)
    words = words[:6]
    step = 1.6
    spans = "".join(f'<b style="animation-delay:{i * step:.1f}s;animation-duration:{len(words) * step:.1f}s">'
                    f"{esc(w)}</b>" for i, w in enumerate(words))
    return (f'<div class="aa-thinking" dir="{"rtl" if lang == "ar" else "ltr"}"><span class="aa-dots"><i></i><i></i>'
            f'<i></i></span><span class="aa-words">{spans}</span></div>')


def passage_html(p: dict) -> str:
    pages = page_label(p.get("pages", ""))
    if p.get("url"):   # a web page (or learned from one): its site, opening the page
        where = (f'<a href="{esc(p["url"])}" target="_blank" rel="noopener noreferrer"><bdi dir="auto">'
                 f'{esc((p.get("title") or p.get("source", ""))[:90])}</bdi></a>')
        meta = [where, f'<bdi dir="ltr" class="file">{esc(p.get("source", ""))}</bdi>']
    else:
        meta = [f'<bdi dir="ltr" class="file">{esc(p.get("source", ""))}</bdi>']
        if pages:
            meta.append(esc(pages))
    if p.get("section"):
        meta.append(f"<bdi>{esc(p['section'])}</bdi>")
    score = max(0.0, min(1.0, float(p.get("score") or 0)))
    meta.append(f"{T['similarity']} {score:.2f}")
    return (f'<div class="aa-passage"><div class="aa-passage-meta">{"<span>·</span>".join(meta)}</div>'
            f'<div class="aa-score"><span style="width:{score * 100:.0f}%"></span></div>'
            f'<div class="aa-passage-text" dir="auto">{esc(p.get("text", ""))}</div></div>')


SPOKEN_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,،]\s*\d+)*\]")
LIST_MARK_RE = re.compile(r"^\s*(?:#+|[-*•▪●◦]|\d{1,2}[.)\-])\s+", re.M)
SENTENCE_END_RE = re.compile(r"(?<=[.!?؟؛;])\s+")


def speech_sentences(text: str, limit: int = 260) -> list[str]:
    """The answer as short pieces to speak one after another: no [n] markers or Markdown, one piece per
    sentence or list item; long sentences are cut at a comma, tiny pieces joined to the next one."""
    text = LIST_MARK_RE.sub("", SPOKEN_CITATION_RE.sub("", text).replace("**", "").replace("__", ""))
    pieces = []
    for line in text.splitlines():
        for sentence in SENTENCE_END_RE.split(line.strip()):
            sentence = sentence.strip(" -–")
            while len(sentence) > limit:
                cut = max(sentence.rfind("،", 0, limit), sentence.rfind(",", 0, limit), sentence.rfind(" ", 0, limit))
                cut = cut if cut > limit // 3 else limit
                pieces.append(sentence[:cut + 1].strip())
                sentence = sentence[cut + 1:].strip()
            if re.search(r"\w", sentence):
                pieces.append(sentence)
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < 25:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def voice_chunks(sentences: list[str], first: int = 220, limit: int = 650) -> list[str]:
    """Sentences grouped into a few longer pieces for the Gemini voice. Its free tier allows only a few requests a
    minute, so one request per sentence used the allowance up within an answer or two and the rest was read by the
    Microsoft voice. A short first piece starts quickly; the rest comes in pieces of up to `limit` characters."""
    chunks: list[str] = []
    for sentence in sentences:
        room = first if len(chunks) <= 1 else limit
        if chunks and len(chunks[-1]) + len(sentence) + 1 <= room:
            chunks[-1] = f"{chunks[-1]} {sentence}" if chunks[-1][-1:] in ".!?؟:،" else f"{chunks[-1]}. {sentence}"
        else:
            chunks.append(sentence)
    return chunks


def live_voice(msg: dict) -> None:
    """Speak the answer live (see theme.PLAYER_JS): it starts by itself once, right after the answer arrives;
    the bar under the answer shows the sentence being said, with pause / stop / listen again."""
    sentences = speech_sentences(plain_answer(msg["content"]))
    if not sentences:
        return
    if voice_choice()[0] == "gemini":
        sentences = voice_chunks(sentences)
    config = {
        "version": PLAYER_VERSION, "player": PLAYER_JS, "token": msg["speak"], "autoplay": True,
        "sentences": sentences, "lang": msg.get("language") or text_lang(msg["content"]),
        "url": VOICE_STREAM_URL if CAN_SPEAK else "",   # no server voice: the browser's own voice
        "voice": voice_choice()[1], "engine": voice_choice()[0], "style": voice_choice()[2],
        "t": {k: T[f"voice_{k}"] for k in ("speaking", "loading", "paused", "blocked", "novoice", "browser",
                                          "again", "play", "pause", "stop")},
    }
    # the config goes into a <script>: JSON-encoded, with < > & escaped so it can never close the tag
    as_js = json.dumps(config, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    st.iframe(LIVE_VOICE_HTML.replace("__DIR__", "rtl" if RTL else "ltr").replace("__CONFIG__", as_js), height=54)


def listen(msg_id: str) -> None:
    msg = find_message(msg_id)
    if msg is not None:
        msg["speak"] = new_id()   # a new token: the player speaks it once


def send_feedback(msg_id: str) -> None:
    msg = find_message(msg_id)
    value = st.session_state.get(f"fb-{msg_id}")
    if msg is None or value is None or msg.get("feedback") == value:
        return
    comment = f"document: {msg['document']}" if msg.get("document") else None
    try:
        client.feedback(question=msg.get("question", ""), answer=msg["content"], rating="up" if value == 1 else "down",
                        sources=msg.get("sources"), comment=comment)
        msg["feedback"] = value
        st.toast(T["feedback_thanks"], icon="✅")
    except APIError:
        st.toast(T["feedback_failed"], icon="⚠️")


def render_body(msg: dict) -> None:
    lang = text_lang(msg["content"])
    st.markdown(f'<span class="lang-{lang}"></span><span class="role-{msg["role"]}"></span>', unsafe_allow_html=True)
    if msg.get("error"):
        st.error(msg["content"])
        return
    if msg["role"] == "user":
        if msg.get("voice"):
            st.markdown(f'<div class="aa-badges"><span class="aa-badge">{esc(T["badge_voice"])}</span></div>',
                        unsafe_allow_html=True)
        if msg.get("file"):
            st.markdown(f'<div class="aa-badges"><span class="aa-badge file">📎 <bdi dir="auto">{esc(msg["file"])}</bdi>'
                        '</span></div>', unsafe_allow_html=True)
        if msg["content"]:
            st.markdown(esc(msg["content"]))
        if msg.get("standard") and msg["standard"] != msg["content"]:
            st.markdown(f'<div class="aa-heard">{esc(T["understood_as"].format(q=msg["standard"]))}</div>',
                        unsafe_allow_html=True)
        return
    if msg.get("notice"):   # "I've read your document…" / "document removed", in the page's current language
        st.markdown(f'<div class="aa-badges"><span class="aa-badge">{esc(T["doc_label"])}</span></div>',
                    unsafe_allow_html=True)
        st.markdown(notice_text(msg))
        return

    origin = msg.get("origin") or ("none" if msg.get("refused") else "library")
    answer_t = TEXT[msg.get("language") or lang]
    badges = [f'<span class="aa-badge">{esc(T["badge_lang"].get(msg.get("language") or lang, ""))}</span>']
    if origin == "document":
        badges.append(f'<span class="aa-badge">{esc(T["badge_doc"])}</span>')
    elif origin == "library" and msg.get("fallback"):
        badges.append(f'<span class="aa-badge">{esc(T["badge_library"])}</span>')
    elif origin in ("web", "learned", "general"):
        badges.append(f'<span class="aa-badge {origin}">{esc(T["badge_" + origin])}</span>')
    if msg.get("off_topic"):
        badges.append(f'<span class="aa-badge alert">{esc(T["badge_off_topic"])}</span>')
    elif msg.get("refused"):
        badges.append(f'<span class="aa-badge alert">{esc(T["badge_refused"])}</span>')
    if msg.get("latency_ms"):
        badges.append(f'<span class="aa-badge"><bdi dir="ltr">{msg["latency_ms"] / 1000:.1f} s</bdi></span>')
    st.markdown(f'<div class="aa-badges">{"".join(badges)}</div>', unsafe_allow_html=True)
    hint = None
    if msg.get("fallback") and origin == "library":
        hint = answer_t["not_in_doc"]
    elif origin in ("web", "learned", "general"):
        hint = answer_t[f"{origin}_hint"]
    if hint:
        st.markdown(f'<div class="aa-hint">{esc(hint)}</div>', unsafe_allow_html=True)
    st.markdown(answer_html(msg["content"]), unsafe_allow_html=True)
    sources, passages = msg.get("sources", []), msg.get("passages", [])
    if sources:
        st.markdown(f'<div class="aa-sources-label">{esc(T["sources"])}</div><div>{source_chips(sources, passages)}</div>',
                    unsafe_allow_html=True)
    elif not msg.get("refused") and origin != "general":
        st.caption(T["no_sources"])
    if msg.get("learned"):
        st.markdown(f'<div class="aa-saved">✓ {esc(answer_t["learned_saved"])}</div>', unsafe_allow_html=True)
    if passages:
        n = len(passages)
        with st.expander(T["evidence"].format(n=n, s="" if n == 1 else "s")):
            st.markdown("".join(passage_html(p) for p in passages), unsafe_allow_html=True)

    # spoken answer, live: the server's Egyptian voice streamed sentence by sentence (the browser's voice offline)
    if msg.get("speak"):
        live_voice(msg)
    if msg.get("id"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if not msg.get("speak"):
                st.button(T["listen"], key=f"listen-{msg['id']}", icon=":material/volume_up:", type="tertiary",
                          on_click=listen, args=(msg["id"],))
            # a thumbs-down on a refusal is useful too: it marks a gap in the documents
            st.feedback("thumbs", key=f"fb-{msg['id']}", on_change=send_feedback, args=(msg["id"],))


def render_message(msg: dict, unanswered: bool = False) -> None:
    with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else BOT_AVATAR):
        render_body(msg)
        if unanswered:   # a new question was sent before this one's answer came: offer to ask it again
            st.button(T["ask_again"], key=f"again-{msg.get('id')}", icon=":material/refresh:", type="tertiary",
                      on_click=ask_suggestion, args=(msg["content"],))


def unanswered_ids(messages: list[dict]) -> set:
    """Typed or spoken questions with no answer after them (the page was used again while it was thinking)."""
    out = set()
    for i, m in enumerate(messages):
        if m["role"] == "user" and m.get("content") and m.get("id"):
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is None or nxt["role"] == "user":
                out.add(m["id"])
    return out


# ---------------------------------------------------------------- the attached document
def notice(kind: str, **data) -> None:
    """A note in the conversation; its text is written when drawn, so it follows the page language."""
    st.session_state.messages.append({"role": "assistant", "notice": kind, "data": data, "content": ""})


def notice_text(msg: dict) -> str:
    data = msg.get("data") or {}
    if msg["notice"] == "doc_ready":
        n = data.get("sections", 0)
        # the title is isolated (U+2068…U+2069) so an Arabic title doesn't reorder an English sentence
        return T["doc_ready"].format(title=f"\u2068{data.get('title', '')}\u2069", pages=data.get("pages", 0),
                                     sections=T["doc_ready_sections"].format(n=n) if n else "")
    return T.get(msg["notice"], "")


def attach(file) -> bool:
    """Send the file to the backend (cleaned, indexed, clean PDF made); it becomes the chat's document."""
    with st.spinner(T["uploading"].format(name=file.name)):
        try:
            info = client.upload_document(file.name, file.getvalue(), getattr(file, "type", None),
                                          st.session_state.session_id, learn=bool(st.session_state.use_learn))
        except APIError as exc:
            st.session_state.messages.append({"role": "assistant", "error": True,
                                              "content": T["upload_failed"].format(detail=exc.detail or error_text(exc))})
            return False
        pdf = None
        if info.get("clean_pdf"):
            try:
                pdf = client.clean_pdf(info["id"])
            except APIError:
                pdf = None
    old = st.session_state.doc
    if old:   # one document at a time: the new one replaces it
        try:
            client.delete_document(old["id"])
        except APIError:
            pass
    st.session_state.update(doc=info, doc_pdf=pdf)
    st.session_state.messages.append({"role": "user", "id": new_id(), "content": "", "file": file.name})
    notice("doc_ready", title=info.get("title") or info["filename"], pages=info.get("pages", 0),
           sections=len(info.get("sections") or []))
    st.session_state.context_start = len(st.session_state.messages)   # follow-ups start from the document
    return True


def remove_document() -> None:
    doc = st.session_state.doc
    if doc:
        try:
            client.delete_document(doc["id"])
        except APIError:
            pass
        notice("doc_removed")
    st.session_state.update(doc=None, doc_pdf=None, context_start=len(st.session_state.messages))


def doc_bar(doc: dict) -> None:
    """The attached document, pinned above the conversation: what it is, how it was cleaned, its clean PDF."""
    cleaning = doc.get("cleaning") or {}
    m = T["doc_meta"]
    chips = [m["pages"].format(n=doc.get("pages", 0))]
    if doc.get("sections"):
        chips.append(m["sections"].format(n=len(doc["sections"])))
    chips.append(m["chunks"].format(n=doc.get("chunks", 0)))
    dropped = len(cleaning.get("dropped_pages") or [])
    if cleaning.get("method") == "cleaned":
        chips.append(T["doc_cleaned"].format(n=dropped, s="" if dropped == 1 else "s") if dropped else T["doc_cleaned_none"])
    elif cleaning.get("method") == "plain":
        chips.append(T["doc_plain"])
    if doc.get("learned"):
        chips.append(T["doc_learned"].format(n=doc["learned"]))
    with st.container(key="docbar", horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(
            f'<div class="aa-docbar"><span class="aa-docbar-icon">📄</span><div class="aa-docbar-text">'
            f'<div class="aa-eyebrow small">{esc(T["doc_label"])} · {esc(T["doc_active"])}</div>'
            f'<div class="aa-docbar-title" dir="auto">{esc(doc.get("title") or doc.get("filename"))}</div>'
            f'<div class="aa-docbar-meta"><bdi dir="auto" class="file">{esc(doc.get("filename", ""))}</bdi>'
            + "".join(f"<span>·</span><span>{esc(c)}</span>" for c in chips) + "</div></div></div>",
            unsafe_allow_html=True)
        if st.session_state.doc_pdf:
            # browsers get only an ASCII file name from Streamlit (an Arabic one becomes "download")
            stem = re.sub(r"[^A-Za-z0-9_\- ]+", "", Path(doc.get("filename", "")).stem).strip(" -_")
            st.download_button(T["clean_pdf"], st.session_state.doc_pdf,
                               file_name=f"{stem} (clean).pdf" if stem else "saqya-clean-document.pdf",
                               mime="application/pdf", icon=":material/download:", key="doc-pdf")
        st.button(T["remove_doc"], icon=":material/close:", type="tertiary", on_click=remove_document, key="doc-remove")


def doc_suggestions(doc: dict) -> None:
    """Right after attaching: summarise, main recommendations, explain a section."""
    with st.container(key="docsuggest", horizontal=True, gap="small"):
        for i, q in enumerate(T["doc_suggest"]):   # a summary (the whole document explained), then its recommendations
            st.button(q, key=f"dsg-{i}", on_click=ask_suggestion, args=(q, True if i == 0 else "recommendations"),
                      icon=":material/auto_awesome:")
        for i, section in enumerate((doc.get("sections") or [])[:4]):
            st.button(T["explain_section"].format(section=section[:48]), key=f"dsec-{i}", on_click=ask_suggestion,
                      args=(T["explain_section_q"].format(section=section), section))


# ---------------------------------------------------------------- asking
def history() -> list[str]:
    """The farmer's earlier questions since the document context last changed (for follow-up questions)."""
    msgs = st.session_state.messages[st.session_state.context_start:]
    return [m["content"] for m in msgs if m["role"] == "user" and m.get("content") and not m.get("error")][-4:]


def transcribe(audio) -> dict | None:
    """Show a user bubble while Whisper listens; returns the transcript, or None (message added) on failure."""
    lang = st.session_state.speech_lang if st.session_state.speech_lang in ("ar", "en") else None
    slot = st.empty()  # a "listening" bubble, replaced by the real question once it is written down
    with slot.container(), st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown('<span class="role-user"></span>', unsafe_allow_html=True)
        with st.spinner(T["listening"]):
            try:
                result = client.transcribe(audio.getvalue(), getattr(audio, "name", "question.wav") or "question.wav", lang)
            except APIError as exc:
                result = {"error": T["voice_failed"].format(detail=exc.detail or error_text(exc))}
    slot.empty()
    if result.get("error") or not (result.get("text") or "").strip():
        st.session_state.messages.append({"role": "assistant", "content": result.get("error") or T["not_heard"],
                                          "error": True})
        return None
    return result


def ask_library(question: str, earlier: list[str]) -> dict:
    """The library — and, when it has no answer to a farming question, what SAQYA learned, the web and general
    knowledge, as allowed in Settings."""
    # the farmer's choices as they are: the backend itself skips what it can't do (no web search installed...),
    # so a health check that ran while the backend was still starting can't switch the web off
    return client.ask(question, history=earlier, web=bool(st.session_state.use_web),
                      learn=bool(st.session_state.use_learn), general=bool(st.session_state.use_general))


def respond(question: str, voice: dict | None = None, explain: str | bool | None = None) -> None:
    """Ask the backend and draw the exchange: from the attached document if there is one, else the library.
    explain: True (the whole document) or a section of it to explain."""
    store = st.session_state.messages
    doc = st.session_state.doc
    earlier = history()
    user = {"role": "user", "id": new_id(), "content": question}
    if voice:
        user.update(voice=True, standard=voice.get("standard"))
    store.append(user)
    render_message(user)
    lang = text_lang(question)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        thinking = st.empty()
        thinking.markdown(thinking_html(lang, bool(doc)), unsafe_allow_html=True)
        fallback = False
        try:
            if doc and explain:
                result = client.explain_document(doc["id"], None if explain is True else explain, LANG)
            elif doc:
                result = client.ask_document(doc["id"], question, earlier)
                if not result.get("sources"):   # not in the document: the library, then beyond it
                    library_result = ask_library(question, earlier)
                    # only an answer replaces the document's refusal: a question about the attached document is
                    # never "not about farming", even when it doesn't say so ("what are the main recommendations?")
                    if library_result.get("origin", "library" if library_result.get("sources") else "none") != "none":
                        result, fallback = library_result, True
            else:
                result = ask_library(question, earlier)
            reply = {
                "role": "assistant", "id": new_id(), "question": question, "content": result["answer"],
                "sources": result.get("sources", []), "passages": result.get("passages", []),
                "language": result.get("language"), "refused": result.get("refused", not result.get("sources")),
                "latency_ms": result.get("latency_ms"), "document": doc["filename"] if doc else None,
                "fallback": fallback, "origin": result.get("origin") or ("document" if doc and not fallback else None),
                "note": result.get("note", ""), "learned": result.get("learned", 0),
                "off_topic": result.get("off_topic", False),
            }
        except APIError as exc:
            if exc.kind == "not_found" and doc:
                st.session_state.update(doc=None, doc_pdf=None, context_start=len(store))
                reply = {"role": "assistant", "content": T["doc_missing"], "error": True}
            else:
                reply = {"role": "assistant", "content": error_text(exc), "error": True}
        thinking.empty()
        mode = st.session_state.read_aloud
        if not reply.get("error") and (mode == "always" or (mode == "voice" and voice)):
            reply["speak"] = new_id()   # spoken live as soon as it is drawn
        store.append(reply)
        render_body(reply)


def ask_suggestion(q: str, explain: str | bool | None = None) -> None:
    st.session_state.pending = {"text": q, "explain": explain}


def handle_input(incoming: dict | None) -> None:
    """A clicked suggestion, or what came from the chat box: a file (＋), a recording, typed text."""
    pending = st.session_state.pop("pending", None)
    if pending:
        respond(pending["text"], explain=pending.get("explain"))
        return
    if not incoming:
        return
    if incoming["files"]:
        attached = attach(incoming["files"][0])
        if attached and incoming["text"]:
            st.session_state.pending = {"text": incoming["text"], "explain": None}   # asked with the file: answer it
        if incoming["audio"] is None:
            st.rerun()   # show the document bar (or the error), then the question if there was one
        for message in st.session_state.messages[-2:]:   # the file and "I've read it", before the spoken question
            render_message(message)
    if incoming.get("audio") is not None:
        heard = transcribe(incoming["audio"])
        if heard is None:
            st.rerun()
        respond(heard["text"], voice=heard)
    elif incoming.get("text"):
        respond(incoming["text"])


# ---------------------------------------------------------------- the chat box (read first)
# The chat box is pinned to the bottom wherever it is called, so it is read before anything is drawn:
# something sent from the settings page switches back to the conversation before the view control is drawn.
doc = st.session_state.doc
placeholder = T["doc_placeholder"].format(title=(doc.get("title") or doc.get("filename", ""))[:60]) if doc else T["placeholder"]
value = st.chat_input(
    placeholder, accept_audio=CAN_LISTEN, audio_sample_rate=16000,
    accept_file=CAN_UPLOAD, file_type=DOC_TYPES if CAN_UPLOAD else None,
    max_upload_size=MAX_UPLOAD_MB if CAN_UPLOAD else None, submit_mode="disable",
)
incoming = None
if value:
    if isinstance(value, str):
        incoming = {"text": value.strip(), "audio": None, "files": []}
    else:
        incoming = {"text": (value.text or "").strip(), "audio": getattr(value, "audio", None),
                    "files": list(getattr(value, "files", []) or [])}
    if not (incoming["text"] or incoming["audio"] is not None or incoming["files"]):
        incoming = None
if incoming and st.session_state.view != "chat":
    st.session_state.view = "chat"

# ---------------------------------------------------------------- header: brand · views · language
brand_col, nav_col, lang_col = st.columns([4, 4, 2], vertical_alignment="center")
with brand_col:
    st.markdown(f'<div class="aa-brand"><span class="aa-mark">{WHEEL_SVG}</span><span class="aa-word">{BRAND_NAME}</span>'
                f'<span class="aa-word-sep">-</span><span class="aa-word-ar" lang="ar">{BRAND_NAME_AR}</span></div>',
                unsafe_allow_html=True)
with nav_col:
    st.segmented_control("view", ["chat", "settings"], key="view", format_func=lambda v: T["nav"][v],
                         label_visibility="collapsed", required=True)
with lang_col:
    st.segmented_control("language", ["en", "ar"], key="ui_lang", format_func=lambda v: {"en": "EN", "ar": "عربي"}[v],
                         label_visibility="collapsed", required=True)
view = st.session_state.view or "chat"


# ---------------------------------------------------------------- view: chat
def synced_select(label: str, options: list, state_key: str, fmt, **kwargs) -> None:
    """A selectbox keyed per UI language (a selectbox keeps its first labels otherwise); `state_key` holds the choice."""
    widget_key = f"{state_key}_{LANG}"
    if st.session_state.get(state_key) not in options:
        st.session_state[state_key] = options[0]
    st.session_state[widget_key] = st.session_state[state_key]
    st.selectbox(label, options, key=widget_key, format_func=fmt,
                 on_change=lambda: st.session_state.update({state_key: st.session_state[widget_key]}), **kwargs)


def synced_toggle(label: str, state_key: str, **kwargs) -> None:
    """A toggle whose choice outlives the settings page (Streamlit forgets a widget's own key once it isn't drawn)."""
    widget_key = f"{state_key}_toggle"
    st.session_state[widget_key] = bool(st.session_state.get(state_key, True))
    st.toggle(label, key=widget_key, on_change=lambda: st.session_state.update({state_key: st.session_state[widget_key]}),
              **kwargs)


if view == "chat":
    if health.get("status") == "unreachable":
        st.warning(T["server_down"], icon=":material/cloud_off:")
    doc = st.session_state.doc
    pending = st.session_state.get("pending")
    messages = st.session_state.messages
    if doc:
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        doc_bar(doc)
        asked_since = any(m["role"] == "user" and m.get("content") for m in messages[st.session_state.context_start:])
        if not asked_since and not incoming and not pending:
            doc_suggestions(doc)
    elif not messages and not incoming and not pending:
        render_hero()
        with st.container(key="suggest"):
            st.markdown(f'<div class="aa-eyebrow aa-section">{esc(T["suggest_label"])}</div>', unsafe_allow_html=True)
            cols = st.columns(4, gap="medium")
            for col, (cat, label) in zip(cols, T["categories"].items()):
                with col:
                    st.markdown(f'<div class="aa-eyebrow small">{esc(label)}</div>', unsafe_allow_html=True)
                    for i, q in enumerate(T["suggestions"][cat]):
                        st.button(q, key=f"sg-{cat}-{i}", on_click=ask_suggestion, args=(q,), width="stretch")
        if CAN_LISTEN or CAN_UPLOAD:
            st.markdown(f'<div class="aa-hint-line">{esc(T["input_hint"])}</div>', unsafe_allow_html=True)
    if messages or incoming or pending:
        if not doc:
            st.markdown(f'<div class="aa-eyebrow">{esc(T["chat_eyebrow"])}</div>'
                        f'<div class="aa-title" style="font-size:clamp(1.8rem,3.2vw,2.4rem)">{esc(T["chat_title"])}</div>',
                        unsafe_allow_html=True)
        waiting = unanswered_ids(messages)
        for message in messages:
            render_message(message, message.get("id") in waiting)
    handle_input(incoming)

# ---------------------------------------------------------------- view: settings
else:
    def transcript_markdown(messages: list[dict]) -> str:
        lines = [f"# {T['export_title']}", "", f"_{datetime.now():%Y-%m-%d %H:%M}_", ""]
        for m in messages:
            if m.get("error"):
                continue
            who = T["you"] if m["role"] == "user" else T["assistant"]
            text = notice_text(m) if m.get("notice") else (plain_answer(m["content"]) if m["role"] == "assistant"
                                                           else m["content"]) or (f"📎 {m['file']}" if m.get("file") else "")
            lines += [f"**{who}:**{' 🎙' if m.get('voice') else ''}", "", text, ""]
            if m.get("sources"):
                lines += [f"{T['sources']}:"] + [f"- {s}" for s in m["sources"]] + [""]
        return "\n".join(lines)

    def clear_conversation() -> None:
        try:
            client.delete_session_documents(st.session_state.session_id)
        except APIError:
            pass
        st.session_state.update(messages=[], doc=None, doc_pdf=None, context_start=0)
        st.toast(T["cleared"], icon="🗑️")

    st.markdown(f'<div class="aa-eyebrow" style="margin-top:1.2rem">{esc(T["settings_eyebrow"])}</div>'
                f'<div class="aa-title">{esc(T["settings_title"])}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
    voice_col, conv_col = st.columns(2, gap="medium")
    with voice_col, st.container(key="set-voice"):
        st.markdown(f'<div class="aa-set-title">{esc(T["voice"])}</div><p class="aa-set-sub">{esc(T["voice_sub"])}</p>',
                    unsafe_allow_html=True)
        synced_select(T["speech_lang"], ["auto", "ar", "en"], "speech_lang", lambda k: T["speech_lang_options"][k],
                      disabled=not CAN_LISTEN)
        synced_select(T["read_aloud"], ["voice", "always", "off"], "read_aloud", lambda k: T["read_aloud_options"][k])
        synced_select(T["speech_style"], ["egyptian", "formal"], "speech_style", lambda k: T["speech_style_options"][k])
        if len(ENGINES) > 1:
            synced_select(T["voice_engine"], ENGINES, "voice_engine", lambda k: T["voice_engine_options"].get(k, k))
        engine = voice_choice()[0]
        synced_select(T["voice_name"], VOICES[engine], "voice_name", lambda k: T["voice_options"].get(k, k),
                      disabled=not CAN_SPEAK)
        reachable = health.get("status") in ("ok", "degraded")
        notes = ([T["voice_missing_stt"]] if reachable and not CAN_LISTEN else []) + \
                ([T["voice_missing_tts"]] if reachable and not CAN_SPEAK else []) + \
                ([T["voice_engine_note"]] if reachable and CAN_SPEAK and "gemini" not in ENGINES else [])
        for note in notes:
            st.markdown(f'<div class="aa-voice-note">{esc(note)}</div>', unsafe_allow_html=True)
    with conv_col, st.container(key="set-conv"):
        current = [m for m in st.session_state.messages if not m.get("error")]
        n = len(current)
        count = T["conversation_count"].format(n=n, s="" if n == 1 else "s") if n else T["conversation_empty"]
        st.markdown(f'<div class="aa-set-title">{esc(T["conversation"])}</div><p class="aa-set-sub">'
                    f'{esc(T["conversation_sub"])}</p><p class="aa-set-count">{esc(count)}</p>', unsafe_allow_html=True)
        st.download_button(T["export"], transcript_markdown(current), file_name=f"saqya-{datetime.now():%Y%m%d-%H%M}.md",
                           mime="text/markdown", width="stretch", disabled=not current, icon=":material/download:")
        st.button(T["clear"], width="stretch", disabled=not st.session_state.messages and not st.session_state.doc,
                  icon=":material/delete:", on_click=clear_conversation)
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    sources_col, learned_col = st.columns(2, gap="medium")
    with sources_col, st.container(key="set-sources"):
        st.markdown(f'<div class="aa-set-title">{esc(T["sources_title"])}</div><p class="aa-set-sub">'
                    f'{esc(T["sources_sub"])}</p>', unsafe_allow_html=True)
        synced_toggle(T["use_web"], "use_web", disabled=not CAN_WEB)
        synced_toggle(T["use_learn"], "use_learn", disabled=not CAN_LEARN)
        synced_toggle(T["use_general"], "use_general", disabled=not CAN_GENERAL)
        reachable = health.get("status") in ("ok", "degraded")
        for note in ([T["web_off"]] if reachable and not CAN_WEB else []):
            st.markdown(f'<div class="aa-voice-note">{esc(note)}</div>', unsafe_allow_html=True)
    with learned_col, st.container(key="set-learned"):
        st.markdown(f'<div class="aa-set-title">{esc(T["learned_title"])}</div><p class="aa-set-sub">'
                    f'{esc(T["learned_sub"])}</p>', unsafe_allow_html=True)
        knowledge = None
        if CAN_LEARN:
            try:
                knowledge = client.knowledge()
            except APIError:
                knowledge = None
        if not CAN_LEARN or not knowledge or not knowledge.get("enabled"):
            st.markdown(f'<p class="aa-set-count">{esc(T["learned_off"] if reachable else T["learned_empty"])}</p>',
                        unsafe_allow_html=True)
        elif not knowledge.get("passages"):
            st.markdown(f'<p class="aa-set-count">{esc(T["learned_empty"])}</p>', unsafe_allow_html=True)
        else:
            sources = knowledge.get("sources", [])
            st.markdown(f'<p class="aa-set-count">{esc(T["learned_count"].format(n=knowledge["passages"], m=len(sources)))}</p>',
                        unsafe_allow_html=True)

            def forget(source_id: str | None) -> None:
                try:
                    client.forget_source(source_id) if source_id else client.forget_all()
                    st.toast(T["learned_forgotten"], icon="🧹")
                except APIError as exc:
                    st.toast(error_text(exc), icon="⚠️")

            for source in sources[:8]:
                kind = T["learned_web"] if source.get("kind") == "web" else T["learned_doc"]
                title = esc((source.get("title") or source.get("site") or "")[:80])
                what = (f'<a href="{esc(source["url"])}" target="_blank" rel="noopener noreferrer">{title}</a>'
                        if source.get("url") else title)
                row, button = st.columns([6, 1], vertical_alignment="center", gap="small")
                with row:
                    st.markdown(f'<div class="aa-learned-row"><span class="kind">{esc(kind)}</span>'
                                f'<span class="what" dir="auto">{what}</span><span class="n">{source.get("passages", 0)}</span></div>',
                                unsafe_allow_html=True)
                with button:
                    st.button(T["learned_forget"], key=f"forget-{source['id']}", type="tertiary", on_click=forget,
                              args=(source["id"],))
            if len(sources) > 8:
                st.caption(T["learned_more"].format(n=len(sources) - 8))
            st.button(T["learned_forget_all"], icon=":material/delete_sweep:", width="stretch", on_click=forget,
                      args=(None,), key="forget-all")
    st.markdown(f'<div class="aa-well"><b>{esc(T["about"])}</b><br>{esc(T["name_note"])}<br>{esc(T["disclaimer"])}</div>',
                unsafe_allow_html=True)
