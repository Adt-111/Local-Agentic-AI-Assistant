"""
Config lives here -- API keys come from env vars, never hardcoded, and a
missing key just makes that one fetcher report a clean error instead of
crashing anything.

    export WEATHER_API_KEY="your_openweathermap_key"
    export WOLFRAM_APP_ID="your_wolfram_alpha_appid"
    export NEWS_API_KEY="your_newsapi_key"
    python main.py
"""

from __future__ import annotations

import logging
import os

# Ollama (local LLM runtime)

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_REQUEST_TIMEOUT_SECONDS: float = float(os.environ.get("OLLAMA_TIMEOUT", "60"))
OLLAMA_TEMPERATURE: float = float(os.environ.get("OLLAMA_TEMPERATURE", "0.7"))
OLLAMA_MAX_HISTORY_TURNS: int = int(os.environ.get("OLLAMA_MAX_HISTORY_TURNS", "10"))

# External API keys (all optional — missing keys degrade gracefully)

WEATHER_API_KEY: str = os.environ.get("WEATHER_API_KEY", "")
WEATHER_API_BASE_URL: str = "https://api.openweathermap.org/data/2.5/weather"

WOLFRAM_APP_ID: str = os.environ.get("WOLFRAM_APP_ID", "")
WOLFRAM_API_BASE_URL: str = "https://api.wolframalpha.com/v1/result"

NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "")
NEWS_API_BASE_URL: str = "https://newsapi.org/v2/top-headlines"

# Fetcher behavior

HTTP_REQUEST_TIMEOUT_SECONDS: float = 10.0
WIKIPEDIA_SENTENCES: int = 5
ARXIV_MAX_RESULTS: int = 5
NEWS_MAX_ARTICLES: int = 5
YOUTUBE_MAX_RESULTS: int = 5

# RAG behavior

RAG_SYSTEM_PROMPT_TEMPLATE: str = (
    "You are a helpful, precise local AI assistant. Use the RETRIEVED CONTEXT "
    "below to answer the user's question when it is relevant. If the context "
    "does not contain the answer, say so plainly and answer from your own "
    "knowledge instead of inventing facts.\n\n"
    "RETRIEVED CONTEXT:\n{context}\n"
)
RAG_MAX_CONTEXT_CHARS: int = 6000  # truncate retrieved context to keep prompts bounded

# Audio

AUDIO_ENABLED: bool = os.environ.get("AUDIO_ENABLED", "true").lower() != "false"
AUDIO_ASSETS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "audio")
SOUND_ON_RESPONSE: str = os.path.join(AUDIO_ASSETS_DIR, "response_ready.wav")
SOUND_ON_ERROR: str = os.path.join(AUDIO_ASSETS_DIR, "error.wav")
SOUND_ON_SUBMIT: str = os.path.join(AUDIO_ASSETS_DIR, "submit.wav")

# GUI

APP_TITLE: str = "Local Agentic AI Assistant"
APP_GEOMETRY: str = "1000x720"
APPEARANCE_MODE: str = "dark"       # customtkinter: "dark" | "light" | "system"
COLOR_THEME: str = "blue"           # customtkinter built-in theme name

# Logging

LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging() -> None:
    """Configures root logging once at app startup. Safe to call multiple times."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # already configured (e.g. re-imported in tests)
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
