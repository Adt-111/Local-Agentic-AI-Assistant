"""
Modular "tool" fetchers, one per external information source. Every fetcher
returns a `FetchResult` — never raises — so a flaky network, a missing API
key, or a malformed response degrades to a clear, structured message instead
of crashing the GUI thread that called it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

try:
    import wikipedia
except ImportError:  # pragma: no cover
    wikipedia = None  # type: ignore

try:
    import arxiv
except ImportError:  # pragma: no cover
    arxiv = None  # type: ignore

try:
    import yt_dlp
except ImportError:  # pragma: no cover
    yt_dlp = None  # type: ignore


# Result schema

@dataclass
class FetchResult:
    """Uniform return type for every fetcher, so `rag_engine.py` / `ui.py` don't need per-source branching."""
    success: bool
    source: str
    query: str
    content: str
    raw_items: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def as_context_string(self) -> str:
        """Formats this result for injection into an LLM RAG prompt."""
        if not self.success:
            return f"[{self.source}] No information retrieved ({self.error})."
        return f"[{self.source} results for '{self.query}']\n{self.content}"


def _fallback(source: str, query: str, error: str) -> FetchResult:
    logger.warning("Fetcher '%s' failed for query %r: %s", source, query, error)
    return FetchResult(
        success=False,
        source=source,
        query=query,
        content=f"Sorry, I couldn't retrieve {source} results right now ({error}).",
        error=error,
    )


def _missing_key_result(source: str, query: str, env_var_name: str) -> FetchResult:
    return _fallback(
        source, query, f"missing API key — set the {env_var_name} environment variable to enable {source}"
    )


# Wikipedia

def fetch_wikipedia(query: str, sentences: int = config.WIKIPEDIA_SENTENCES) -> FetchResult:
    """
    Fetches a short Wikipedia summary for `query`. Handles disambiguation
    pages (picks the first suggested option) and missing pages gracefully.
    """
    if wikipedia is None:
        return _fallback("Wikipedia", query, "the 'wikipedia' package is not installed")

    try:
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True, redirect=True)
        page = wikipedia.page(query, auto_suggest=True, redirect=True)
        return FetchResult(
            success=True,
            source="Wikipedia",
            query=query,
            content=summary,
            raw_items=[{"title": page.title, "url": page.url}],
        )
    except wikipedia.exceptions.DisambiguationError as exc:
        try:
            first_option = exc.options[0]
            summary = wikipedia.summary(first_option, sentences=sentences, auto_suggest=False)
            return FetchResult(
                success=True,
                source="Wikipedia",
                query=query,
                content=f"(Showing results for '{first_option}', the closest match to an ambiguous query.)\n{summary}",
                raw_items=[{"title": first_option, "disambiguation_options": exc.options[:10]}],
            )
        except Exception as inner_exc:
            return _fallback("Wikipedia", query, f"ambiguous query with no usable option ({inner_exc})")
    except wikipedia.exceptions.PageError:
        return _fallback("Wikipedia", query, "no matching Wikipedia page found")
    except requests.exceptions.RequestException as exc:
        return _fallback("Wikipedia", query, f"network error ({exc})")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _fallback("Wikipedia", query, f"unexpected error ({exc})")


# arXiv

def fetch_arxiv(query: str, max_results: int = config.ARXIV_MAX_RESULTS) -> FetchResult:
    """Fetches recent arXiv papers matching `query`, formatted as a readable list."""
    if arxiv is None:
        return _fallback("arXiv", query, "the 'arxiv' package is not installed")

    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        client = arxiv.Client()
        results = list(client.results(search))

        if not results:
            return _fallback("arXiv", query, "no matching papers found")

        lines = []
        raw_items = []
        for i, paper in enumerate(results, start=1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            published = paper.published.strftime("%Y-%m-%d") if paper.published else "unknown date"
            lines.append(f"{i}. {paper.title} ({published})\n   Authors: {authors}\n   {paper.entry_id}")
            raw_items.append(
                {"title": paper.title, "authors": authors, "published": published, "url": paper.entry_id}
            )

        return FetchResult(success=True, source="arXiv", query=query, content="\n".join(lines), raw_items=raw_items)
    except requests.exceptions.RequestException as exc:
        return _fallback("arXiv", query, f"network error ({exc})")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _fallback("arXiv", query, f"unexpected error ({exc})")


# Weather (OpenWeatherMap)

def fetch_weather(location: str) -> FetchResult:
    """Fetches current weather conditions for `location` via the OpenWeatherMap API."""
    if not config.WEATHER_API_KEY:
        return _missing_key_result("Weather", location, "WEATHER_API_KEY")

    params = {"q": location, "appid": config.WEATHER_API_KEY, "units": "metric"}
    try:
        response = requests.get(config.WEATHER_API_BASE_URL, params=params, timeout=config.HTTP_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        description = data["weather"][0]["description"].capitalize()
        temp_c = data["main"]["temp"]
        feels_like_c = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        city_name = data.get("name", location)

        content = (
            f"{city_name}: {description}, {temp_c:.1f}°C (feels like {feels_like_c:.1f}°C), "
            f"humidity {humidity}%, wind {wind_speed} m/s."
        )
        return FetchResult(success=True, source="Weather", query=location, content=content, raw_items=[data])
    except requests.exceptions.Timeout:
        return _fallback("Weather", location, "request timed out")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 404:
            return _fallback("Weather", location, f"location '{location}' not found")
        return _fallback("Weather", location, f"HTTP error {status}")
    except requests.exceptions.RequestException as exc:
        return _fallback("Weather", location, f"network error ({exc})")
    except (KeyError, IndexError) as exc:
        return _fallback("Weather", location, f"unexpected response format ({exc})")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _fallback("Weather", location, f"unexpected error ({exc})")


# Wolfram Alpha

def fetch_wolfram(query: str) -> FetchResult:
    """Fetches a short factual/computational answer from the Wolfram Alpha "short answers" API."""
    if not config.WOLFRAM_APP_ID:
        return _missing_key_result("Wolfram Alpha", query, "WOLFRAM_APP_ID")

    params = {"appid": config.WOLFRAM_APP_ID, "i": query}
    try:
        response = requests.get(config.WOLFRAM_API_BASE_URL, params=params, timeout=config.HTTP_REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 501:
            return _fallback("Wolfram Alpha", query, "Wolfram Alpha could not interpret the query")
        response.raise_for_status()
        answer = response.text.strip()
        return FetchResult(success=True, source="Wolfram Alpha", query=query, content=answer, raw_items=[{"answer": answer}])
    except requests.exceptions.Timeout:
        return _fallback("Wolfram Alpha", query, "request timed out")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return _fallback("Wolfram Alpha", query, f"HTTP error {status}")
    except requests.exceptions.RequestException as exc:
        return _fallback("Wolfram Alpha", query, f"network error ({exc})")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _fallback("Wolfram Alpha", query, f"unexpected error ({exc})")


# News (NewsAPI)

def fetch_news(query: str, max_articles: int = config.NEWS_MAX_ARTICLES) -> FetchResult:
    """Fetches top headlines matching `query` via NewsAPI's /top-headlines endpoint."""
    if not config.NEWS_API_KEY:
        return _missing_key_result("News", query, "NEWS_API_KEY")

    params = {"q": query, "apiKey": config.NEWS_API_KEY, "pageSize": max_articles, "language": "en"}
    try:
        response = requests.get(config.NEWS_API_BASE_URL, params=params, timeout=config.HTTP_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            return _fallback("News", query, "no matching articles found")

        lines, raw_items = [], []
        for i, article in enumerate(articles[:max_articles], start=1):
            title = article.get("title", "Untitled")
            source_name = article.get("source", {}).get("name", "unknown source")
            url = article.get("url", "")
            lines.append(f"{i}. {title} ({source_name})\n   {url}")
            raw_items.append({"title": title, "source": source_name, "url": url})

        return FetchResult(success=True, source="News", query=query, content="\n".join(lines), raw_items=raw_items)
    except requests.exceptions.Timeout:
        return _fallback("News", query, "request timed out")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return _fallback("News", query, f"HTTP error {status}")
    except requests.exceptions.RequestException as exc:
        return _fallback("News", query, f"network error ({exc})")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _fallback("News", query, f"unexpected error ({exc})")


# YouTube (search via yt-dlp, no download)

def fetch_youtube(query: str, max_results: int = config.YOUTUBE_MAX_RESULTS) -> FetchResult:
    """
    Searches YouTube for `query` using yt-dlp's search extractor
    (`ytsearchN:`), without downloading any video — just metadata (title,
    uploader, URL, duration) for the top results.
    """
    if yt_dlp is None:
        return _fallback("YouTube", query, "the 'yt-dlp' package is not installed")

    search_spec = f"ytsearch{max_results}:{query}"
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_spec, download=False)

        entries = info.get("entries", []) if info else []
        if not entries:
            return _fallback("YouTube", query, "no matching videos found")

        lines, raw_items = [], []
        for i, entry in enumerate(entries, start=1):
            title = entry.get("title", "Untitled")
            uploader = entry.get("uploader") or entry.get("channel") or "unknown uploader"
            video_id = entry.get("id", "")
            url = entry.get("url") or (f"https://www.youtube.com/watch?v={video_id}" if video_id else "")
            duration = entry.get("duration")
            duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}" if duration else "unknown length"
            lines.append(f"{i}. {title} — {uploader} ({duration_str})\n   {url}")
            raw_items.append({"title": title, "uploader": uploader, "url": url, "duration": duration})

        return FetchResult(success=True, source="YouTube", query=query, content="\n".join(lines), raw_items=raw_items)
    except Exception as exc:
        # yt-dlp raises its own broad exception hierarchy (DownloadError, ExtractorError, ...);
        # a catch-all here is intentional and matches this module's "never raise" contract.
        return _fallback("YouTube", query, f"search failed ({exc})")


# Dispatch table (used by rag_engine.py / ui.py to call the right fetcher by intent)

FETCHER_DISPATCH = {
    "WIKIPEDIA": fetch_wikipedia,
    "ARXIV": fetch_arxiv,
    "NEWS": fetch_news,
    "WOLFRAM": fetch_wolfram,
    "WEATHER": fetch_weather,
    "YOUTUBE": fetch_youtube,
}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    print("=== Wikipedia ===")
    print(fetch_wikipedia("Alan Turing").as_context_string())

    print("\n=== arXiv ===")
    print(fetch_arxiv("transformer attention mechanism", max_results=3).as_context_string())

    print("\n=== Weather (no API key expected) ===")
    print(fetch_weather("London").as_context_string())

    print("\n=== Wolfram (no API key expected) ===")
    print(fetch_wolfram("population of France").as_context_string())

    print("\n=== News (no API key expected) ===")
    print(fetch_news("technology").as_context_string())

    print("\n=== YouTube ===")
    print(fetch_youtube("python tutorial for beginners", max_results=3).as_context_string())
