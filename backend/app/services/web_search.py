"""Web search for farming questions the documents don't cover.

1. Search: `ddgs`, a free metasearch library (no API key) that asks several engines at once — Wikipedia,
   DuckDuckGo, Bing, Brave, Google, Mojeek, Yahoo... — in the question's language region.
2. Choose: social media, video and shop pages are skipped; pages from trusted farming sources (FAO, CGIAR,
   ministries and research centres, universities and extension services, Wikipedia) come first.
3. Read: the best pages are downloaded in parallel and their main text is extracted with trafilatura (menus,
   ads and footers removed); PDFs (many bulletins are PDFs) are read with PyMuPDF. When no page can be read,
   the search results' own snippets are used.
4. The text is cut into passages, and RAGService ranks them against the question with the same hybrid search
   and embedding model as the library (beyond.py), so only passages that really answer it are used and cited.

Nothing here calls an AI service; the pages are fetched from the backend's own internet connection.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36 SAQYA-farming-assistant")
MAX_HTML_BYTES = 3_000_000
MAX_PDF_BYTES = 15_000_000
MAX_PDF_PAGES = 40
PASSAGE_CHARS = 900       # longest passage cut from a page
MIN_PASSAGE_CHARS = 120   # shorter pieces are joined to their neighbours or dropped
MAX_PASSAGES_PER_PAGE = 24

TRUSTED = ("fao.org", "cgiar.org", "ifpri.org", "icarda.org", "cimmyt.org", "irri.org", "ifad.org", "wfp.org",
           "worldbank.org", "moa.gov.eg", "agr-egypt.gov.eg", "arc.sci.eg", "agri.gov.eg", "gov.eg", "edu.eg",
           "extension.org", "ucanr.edu", "extension.umn.edu", "extension.psu.edu", "rhs.org.uk", "cabi.org",
           "plantwise.org", "wikipedia.org", "britannica.com", "usda.gov", "nrcs.usda.gov", "ahdb.org.uk",
           "agric.wa.gov.au", "dpi.nsw.gov.au", "icar.org.in", "tnau.ac.in")
BLOCKED = ("youtube.com", "youtu.be", "facebook.com", "fb.com", "instagram.com", "tiktok.com", "twitter.com", "x.com",
           "pinterest.", "linkedin.com", "reddit.com", "quora.com", "amazon.", "aliexpress.", "ebay.", "alibaba.",
           "noon.com", "jumia.", "olx.", "telegram.", "t.me", "whatsapp.", "snapchat.", "soundcloud.com", "spotify.com")

_BOILERPLATE = re.compile(
    r"(?i)\b(cookies?|privacy policy|terms of (use|service)|all rights reserved|subscribe|sign (in|up)|log ?in|"
    r"newsletter|share this|advertis\w+|copyright|javascript)\b|جميع الحقوق|سياسة الخصوصية|اشترك|تسجيل الدخول")


@dataclass
class WebHit:
    url: str
    title: str
    snippet: str = ""


@dataclass
class WebPage:
    url: str
    title: str
    text: str
    site: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    from_snippet: bool = False

    def __post_init__(self) -> None:
        self.site = self.site or site_of(self.url)


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def site_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _matches(site: str, domains: tuple[str, ...]) -> bool:
    return any(site == d or site.endswith("." + d) or (d.endswith(".") and (site.startswith(d) or f".{d}" in site))
               for d in domains)


def is_trusted(site: str, extra: tuple[str, ...] = ()) -> bool:
    return _matches(site, TRUSTED + extra) or bool(re.search(r"\.(gov|edu)(\.[a-z]{2})?$|\.ac\.[a-z]{2}$", site))


def is_blocked(site: str, extra: tuple[str, ...] = ()) -> bool:
    return not site or _matches(site, BLOCKED + extra)


# ---------------------------------------------------------------- page text
def html_to_text(html: str, url: str = "") -> str:
    """The main text of a web page: trafilatura when installed, else the page without tags and scripts."""
    if installed("trafilatura"):
        try:
            import trafilatura

            text = trafilatura.extract(html, url=url or None, include_tables=True, include_comments=False,
                                       favor_recall=True, deduplicate=True)
            if text and len(text) > 200:
                return text
        except Exception as exc:  # noqa: BLE001 - odd HTML: the plain fallback
            logger.debug("trafilatura failed on %s: %s", url, exc)
    html = re.sub(r"(?is)<(script|style|noscript|nav|footer|header|form|aside)\b.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section|article)>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    import html as html_lib

    text = html_lib.unescape(text)
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if len(line) > 40 or re.search(r"[.!؟?:]$", line))


def pdf_to_text(data: bytes) -> str:
    """The text of a PDF found on the web (bulletins are often PDFs)."""
    import pymupdf

    with pymupdf.open(stream=io.BytesIO(data), filetype="pdf") as pdf:
        return "\n\n".join(pdf[i].get_text() for i in range(min(pdf.page_count, MAX_PDF_PAGES)))


def split_passages(text: str, limit: int = PASSAGE_CHARS) -> list[str]:
    """Paragraph-aware passages of at most `limit` characters; boilerplate lines are dropped."""
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|\n(?=[-•*\d]+[.)]?\s)", text)]
    paragraphs = [p for p in paragraphs if p and not (_BOILERPLATE.search(p) and len(p) < 250)]
    out: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > limit:   # a very long paragraph: cut at a sentence end
            cut = max(para.rfind(". ", 0, limit), para.rfind("؟ ", 0, limit), para.rfind("! ", 0, limit),
                      para.rfind("، ", 0, limit))
            cut = cut + 1 if cut > limit // 3 else limit
            if current:
                out.append(current)
                current = ""
            out.append(para[:cut].strip())
            para = para[cut:].strip()
        if len(current) + len(para) + 1 <= limit:
            current = f"{current} {para}".strip()
        else:
            if current:
                out.append(current)
            current = para
    if current:
        out.append(current)
    return [p for p in out if len(p) >= MIN_PASSAGE_CHARS][:MAX_PASSAGES_PER_PAGE]


# ---------------------------------------------------------------- the searcher
class WebSearcher:
    """Search the web, pick and read the best pages. `search_fn` / `fetch_fn` can be replaced (tests)."""

    def __init__(self, backend: str = "auto", max_pages: int = 5, timeout: float = 8.0,
                 trusted: tuple[str, ...] = (), blocked: tuple[str, ...] = (),
                 region_en: str = "us-en", region_ar: str = "xa-ar", search_fn=None, fetch_fn=None,
                 cache_minutes: float = 60) -> None:
        self.backend = backend or "auto"
        self.max_pages = max_pages
        self.timeout = timeout
        self.trusted = tuple(trusted)
        self.blocked = tuple(blocked)
        self.regions = {"en": region_en, "ar": region_ar}
        self._search_fn = search_fn
        self._fetch_fn = fetch_fn
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._cache_seconds = cache_minutes * 60
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._search_fn is not None or installed("ddgs")

    # ------------------------------------------------------------ searching
    def search(self, query: str, language: str = "en", max_results: int = 8) -> list[WebHit]:
        if self._search_fn is not None:
            rows = self._search_fn(query, language, max_results)
        else:
            from ddgs import DDGS

            rows = DDGS(timeout=int(self.timeout)).text(query, region=self.regions.get(language, "us-en"),
                                                        safesearch="moderate", max_results=max_results,
                                                        backend=self.backend)
        hits = []
        for row in rows or []:
            url = row.get("href") or row.get("url") or ""
            if url.startswith("http"):
                hits.append(WebHit(url=url, title=(row.get("title") or "").strip(), snippet=(row.get("body") or "").strip()))
        return hits

    def ranked_hits(self, queries: list[tuple[str, str]], max_results: int = 8) -> list[WebHit]:
        """Hits for all (query, language) pairs: blocked sites out, duplicates out, trusted sites first."""
        seen: set[str] = set()
        hits: list[WebHit] = []
        for query, language in queries:
            try:
                found = self.search(query, language, max_results)
            except Exception as exc:  # noqa: BLE001 - rate limit, no internet...
                logger.warning("Web search failed for %r: %s", query[:80], exc)
                continue
            for hit in found:
                key = re.sub(r"[#?].*$", "", hit.url).rstrip("/").lower()
                if key in seen or is_blocked(site_of(hit.url), self.blocked):
                    continue
                seen.add(key)
                hits.append(hit)
        # stable sort: trusted sources first, otherwise the engines' order (the first query's hits lead)
        return sorted(hits, key=lambda h: 0 if is_trusted(site_of(h.url), self.trusted) else 1)

    # ------------------------------------------------------------ reading pages
    def fetch(self, url: str) -> str:
        """The main text of one page or PDF ('' when it can't be read)."""
        with self._lock:
            cached = self._cache.get(url)
            if cached and time.time() - cached[0] < self._cache_seconds:
                return cached[1]
        text = self._fetch_fn(url) if self._fetch_fn is not None else self._download(url)
        text = text or ""
        with self._lock:
            self._cache[url] = (time.time(), text)
            while len(self._cache) > 200:
                self._cache.popitem(last=False)
        return text

    def _download(self, url: str) -> str:
        import httpx

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True,
                              headers={"User-Agent": USER_AGENT, "Accept-Language": "ar,en;q=0.8"}) as client:
                with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        return ""
                    kind = response.headers.get("content-type", "").lower()
                    limit = MAX_PDF_BYTES if "pdf" in kind or url.lower().endswith(".pdf") else MAX_HTML_BYTES
                    data = b""
                    for chunk in response.iter_bytes():
                        data += chunk
                        if len(data) > limit:
                            return ""
            if "pdf" in kind or data[:5] == b"%PDF-":
                return pdf_to_text(data)
            if "html" not in kind and "text" not in kind and kind:
                return ""
            charset = re.search(r"charset=([\w-]+)", kind)
            try:
                html = data.decode(charset.group(1) if charset else "utf-8", errors="replace")
            except LookupError:   # an unknown charset name
                html = data.decode("utf-8", errors="replace")
            return html_to_text(html, url)
        except Exception as exc:  # noqa: BLE001 - timeouts, TLS, odd servers
            logger.info("Could not read %s: %s", url, exc)
            return ""

    def pages(self, queries: list[tuple[str, str]]) -> list[WebPage]:
        """The best pages for these queries, read in parallel; the search snippets when no page can be read."""
        hits = self.ranked_hits(queries)
        if not hits:
            return []
        chosen = hits[: self.max_pages + 2]   # a couple extra: some pages refuse or have no text
        # (Wikipedia is trusted, so for an Arabic question ar.wikipedia from the first, Arabic query leads)
        pages: list[WebPage] = []
        pool = ThreadPoolExecutor(max_workers=min(6, len(chosen)), thread_name_prefix="web-fetch")
        futures = {pool.submit(self.fetch, h.url): h for h in chosen}
        try:
            for future in as_completed(futures, timeout=self.timeout + 4):
                hit = futures[future]
                try:
                    text = future.result()
                except Exception:  # noqa: BLE001
                    text = ""
                if text and len(text) >= MIN_PASSAGE_CHARS:
                    pages.append(WebPage(url=hit.url, title=hit.title or site_of(hit.url), text=text))
        except TimeoutError:
            logger.info("Some web pages took too long; using the ones that came back")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)   # a slow page never holds up the answer
        order = {h.url: i for i, h in enumerate(chosen)}
        pages.sort(key=lambda p: order.get(p.url, 99))
        pages = pages[: self.max_pages]
        if not pages:   # nothing could be read (blocked sites, no time): the snippets still say something
            pages = [WebPage(url=h.url, title=h.title or site_of(h.url), text=h.snippet, from_snippet=True)
                     for h in hits[: self.max_pages * 2] if len(h.snippet) >= 60]
        logger.info("Web: %d hits, %d pages read (%s)", len(hits), len(pages),
                    ", ".join(p.site for p in pages[:6]))
        return pages


def page_passages(pages: list[WebPage]) -> list[dict]:
    """Passages from the pages, in the shape the in-memory retriever indexes (id, text, metadata)."""
    out = []
    for page in pages:
        texts = [page.text] if page.from_snippet else split_passages(page.text)
        for i, text in enumerate(texts):
            pid = hashlib.sha1(f"{page.url}#{i}".encode()).hexdigest()[:16]
            out.append({"id": f"web-{pid}", "text": text,
                        "metadata": {"source": page.site, "url": page.url, "title": page.title[:160], "kind": "web",
                                     "page_start": 0, "page_end": 0, "fetched_at": page.fetched_at}})
    return out
