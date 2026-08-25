"""
Keyword-based intent classifier. Tokenizes the prompt with NLTK, strips
stopwords, scores it against a keyword list per domain, picks a fetcher.

Rule-based on purpose rather than a trained model -- routing decisions stay
predictable and you can see exactly why something got routed where.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


# NLTK resource bootstrap

def _ensure_nltk_resources() -> None:
    """
    Downloads the small NLTK resources this module needs, if not already
    present. Silent/quiet download; failures are logged but non-fatal since
    `IntentRouter` falls back to a regex tokenizer when NLTK data is
    unavailable (e.g. no network access on first run).
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception as exc:  # pragma: no cover - network-dependent
                logger.warning("Could not download NLTK resource '%s': %s", package, exc)


_ensure_nltk_resources()

try:
    _STOPWORDS = set(stopwords.words("english"))
except Exception:  # pragma: no cover - resource unavailable
    _STOPWORDS = set()

# WH-words are useful WIKIPEDIA/definitional signals ("who was...", "what is...")
# but NLTK's stopword list includes them; exclude them from filtering so they
# remain available for keyword scoring below.
_STOPWORDS -= {"who", "what", "when", "where", "why", "how"}


def _safe_tokenize(text: str) -> list[str]:
    """Tokenizes with NLTK's `word_tokenize`, falling back to a regex tokenizer if NLTK data is missing."""
    try:
        return [t.lower() for t in word_tokenize(text)]
    except Exception as exc:  # pragma: no cover - resource unavailable
        logger.warning("NLTK word_tokenize failed (%s); falling back to regex tokenizer.", exc)
        return re.findall(r"[a-zA-Z0-9']+", text.lower())


# Domains

class Intent(str, Enum):
    WIKIPEDIA = "WIKIPEDIA"
    ARXIV = "ARXIV"
    NEWS = "NEWS"
    WOLFRAM = "WOLFRAM"
    WEATHER = "WEATHER"
    YOUTUBE = "YOUTUBE"
    CODE = "CODE"
    GENERAL_CHAT = "GENERAL_CHAT"


# Keyword gazetteer per domain. Order matters as a tie-break: earlier domains
# win ties over later ones (see `IntentRouter.classify`), reflecting that a
# more specific tool (e.g. WOLFRAM for "calculate") should usually win over
# a more generic one when scores are equal.
_DOMAIN_KEYWORDS: dict[Intent, set[str]] = {
    Intent.WOLFRAM: {
        "calculate", "compute", "solve", "equation", "integral", "derivative",
        "math", "formula", "convert", "unit", "square", "root", "algebra",
        "arithmetic", "sum", "product", "factorial", "logarithm", "matrix",
    },
    Intent.WEATHER: {
        "weather", "temperature", "forecast", "rain", "raining", "snow",
        "snowing", "humidity", "wind", "sunny", "cloudy", "storm", "climate",
        "degrees", "hot", "cold", "umbrella",
    },
    Intent.ARXIV: {
        "arxiv", "paper", "papers", "preprint", "research", "publication",
        "study", "journal", "citation", "abstract", "manuscript",
    },
    Intent.NEWS: {
        "news", "headline", "headlines", "breaking", "today's", "current",
        "events", "happening", "latest", "article", "press",
    },
    Intent.YOUTUBE: {
        "youtube", "video", "videos", "watch", "clip", "tutorial video",
        "channel", "playlist",
    },
    Intent.CODE: {
        "code", "function", "debug", "bug", "error", "exception", "script",
        "program", "programming", "python", "javascript", "java", "sql",
        "algorithm", "compile", "syntax", "refactor", "class", "variable",
    },
    Intent.WIKIPEDIA: {
        "who", "what", "when", "where", "why", "define", "definition",
        "explain", "history", "biography", "wikipedia", "meaning", "wiki",
    },
}

# Regex triggers that strongly imply a domain regardless of keyword overlap
# (checked before falling back to keyword scoring).
_STRONG_PATTERNS: dict[Intent, list[re.Pattern]] = {
    Intent.WEATHER: [re.compile(r"\bweather (in|for|at)\b", re.IGNORECASE)],
    Intent.ARXIV: [re.compile(r"\barxiv[:\s]", re.IGNORECASE)],
    Intent.YOUTUBE: [re.compile(r"\byoutube\.com|\byoutu\.be\b", re.IGNORECASE)],
    Intent.WOLFRAM: [re.compile(r"\d+\s*[\+\-\*/\^]\s*\d+")],  # e.g. "12 * 7"
}


@dataclass
class IntentResult:
    """Result of classifying a single user prompt."""
    intent: Intent
    confidence: float                       # 0.0-1.0, proportion of scoring tokens matched
    matched_keywords: list[str] = field(default_factory=list)
    cleaned_query: str = ""                 # prompt with routing trigger words stripped, for fetchers


class IntentRouter:
    """
    classify() tokenizes with NLTK (regex fallback), strips stopwords, and
    scores overlap against each domain's keyword set. A few regex patterns
    (raw arithmetic, a YouTube URL) skip straight to their domain. Ties go
    to whichever domain was declared first; no match falls back to
    GENERAL_CHAT.
    """

    def __init__(self, domain_keywords: dict[Intent, set[str]] | None = None) -> None:
        self.domain_keywords = domain_keywords or _DOMAIN_KEYWORDS

    def classify(self, prompt: str) -> IntentResult:
        """
        Classifies one prompt. Returns an IntentResult with the winning
        intent, a confidence score, which keywords matched, and a cleaned
        query with the trigger words stripped out (useful for the fetcher).
        """
        if not prompt or not prompt.strip():
            return IntentResult(intent=Intent.GENERAL_CHAT, confidence=0.0)

        for intent, patterns in _STRONG_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(prompt):
                    logger.debug("Strong pattern match for %s on prompt: %r", intent, prompt)
                    return IntentResult(
                        intent=intent,
                        confidence=1.0,
                        matched_keywords=[pattern.pattern],
                        cleaned_query=prompt.strip(),
                    )

        tokens = _safe_tokenize(prompt)
        content_tokens = [t for t in tokens if t.isalpha() and t not in _STOPWORDS]

        if not content_tokens:
            return IntentResult(intent=Intent.GENERAL_CHAT, confidence=0.0, cleaned_query=prompt.strip())

        scores: dict[Intent, list[str]] = {intent: [] for intent in self.domain_keywords}
        for token in content_tokens:
            for intent, keywords in self.domain_keywords.items():
                if token in keywords:
                    scores[intent].append(token)

        best_intent = Intent.GENERAL_CHAT
        best_matches: list[str] = []
        best_count = 0
        # dict preserves insertion order in Python 3.7+, so this naturally
        # respects the gazetteer's declared tie-break priority.
        for intent, matches in scores.items():
            if len(matches) > best_count:
                best_intent = intent
                best_matches = matches
                best_count = len(matches)

        confidence = best_count / len(content_tokens) if content_tokens else 0.0
        cleaned_query = self._strip_matched_keywords(prompt, best_matches)

        result = IntentResult(
            intent=best_intent,
            confidence=round(confidence, 3),
            matched_keywords=best_matches,
            cleaned_query=cleaned_query,
        )
        logger.debug("Classified prompt %r -> %s (confidence=%.2f)", prompt, best_intent, confidence)
        return result

    @staticmethod
    def _strip_matched_keywords(prompt: str, matched_keywords: list[str]) -> str:
        """Removes matched trigger words from the prompt to produce a tighter search query for fetchers."""
        cleaned = prompt
        for kw in matched_keywords:
            cleaned = re.sub(rf"\b{re.escape(kw)}\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.,!")
        return cleaned or prompt.strip()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    router = IntentRouter()

    test_prompts = [
        "What's the weather in San Francisco today?",
        "Find me recent arxiv papers on transformer attention mechanisms",
        "What are today's top news headlines?",
        "Calculate the integral of x^2 from 0 to 10",
        "Show me a youtube video tutorial on Docker",
        "Debug this Python function that raises a KeyError",
        "Who was Marie Curie?",
        "Tell me a joke",
        "12 * 7",
    ]

    for p in test_prompts:
        result = router.classify(p)
        print(f"{p!r:65s} -> {result.intent.value:14s} conf={result.confidence:.2f} matched={result.matched_keywords}")
