"""Publisher transcript scrapers.

Each scraper is a best-effort HTML parse of a known podcast's transcript page.
All are wrapped in try/except — any failure returns None so the transcribe
worker can fall through to Gemini audio transcription.
"""
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - safety net only
    BeautifulSoup = None


HTTP_TIMEOUT_SEC = 30
USER_AGENT = 'Charlie/1.0 (+https://charlie.tonydlee.com)'


def _fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT_SEC, headers={'User-Agent': USER_AGENT})
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _text_from_selector(html: str, selectors: list[str]) -> str | None:
    """Try each CSS selector in order; return joined text from first that hits."""
    if not html or BeautifulSoup is None:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    for sel in selectors:
        nodes = soup.select(sel)
        if not nodes:
            continue
        parts = [n.get_text(separator='\n', strip=True) for n in nodes]
        text = '\n\n'.join(p for p in parts if p)
        # Heuristic minimum length — transcripts are long.
        if text and len(text) > 500:
            return text
    return None


def scrape_invest_like_the_best(source_url: str) -> str | None:
    """Joincolossus hosts Invest Like the Best transcripts under the episode page."""
    try:
        html = _fetch(source_url)
        if not html:
            return None
        return _text_from_selector(html, [
            'div.transcript',
            'section.transcript',
            'article .transcript',
            'article',
            'main article',
        ])
    except Exception:
        return None


def scrape_odd_lots(source_url: str) -> str | None:
    """Bloomberg Odd Lots transcripts appear inline on the episode article page."""
    try:
        html = _fetch(source_url)
        if not html:
            return None
        return _text_from_selector(html, [
            'div.transcript',
            'section[data-component="transcript"]',
            'article div.body-copy',
            'article',
        ])
    except Exception:
        return None


def scrape_acquired(source_url: str) -> str | None:
    """acquired.fm transcripts are rendered inside the episode page main content."""
    try:
        html = _fetch(source_url)
        if not html:
            return None
        return _text_from_selector(html, [
            'div.transcript',
            'section.transcript',
            'div.episode-transcript',
            'main article',
            'article',
        ])
    except Exception:
        return None


def scrape_capital_allocators(source_url: str) -> str | None:
    """capitalallocators.com shows transcripts inside the post body."""
    try:
        html = _fetch(source_url)
        if not html:
            return None
        return _text_from_selector(html, [
            'div.transcript',
            'div.entry-content',
            'article .post-content',
            'article',
        ])
    except Exception:
        return None


_HOST_DISPATCH = (
    ('joincolossus.com',       scrape_invest_like_the_best),
    ('colossus.com',           scrape_invest_like_the_best),
    ('bloomberg.com',          scrape_odd_lots),
    ('acquired.fm',            scrape_acquired),
    ('capitalallocators.com',  scrape_capital_allocators),
)


def try_publisher_scrape(source_url: str | None, audio_url: str | None = None) -> tuple[str | None, str | None]:
    """Dispatch on the source_url hostname to a publisher scraper.

    Returns (transcript_text, 'publisher') or (None, None).
    """
    if not source_url:
        return None, None
    try:
        host = (urlparse(source_url).hostname or '').lower()
    except Exception:
        return None, None
    if not host:
        return None, None
    for needle, scraper in _HOST_DISPATCH:
        if needle in host:
            try:
                text = scraper(source_url)
            except Exception:
                text = None
            if text:
                return text, 'publisher'
            return None, None
    return None, None
