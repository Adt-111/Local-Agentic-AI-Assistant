# Local Agentic AI Assistant & Multi-Mode RAG Desktop App

A desktop chat application that runs open-source LLMs entirely on your own
machine via [Ollama](https://ollama.com) — no data ever leaves your
computer. An NLTK-based intent router automatically classifies each
question and dynamically switches between three modes:

- **Chat Mode** — plain conversation with the local LLM.
- **Web Search Mode** — routes the query to a live information source
  (Wikipedia, arXiv, weather, news, Wolfram Alpha, YouTube) and shows the
  raw result.
- **AI + RAG Mode** — fetches the same live information, then feeds it to
  the LLM as context so the final answer is grounded in current data
  instead of the model's static training knowledge.

## Project layout

```
local_rag_agent_app/
├── requirements.txt
├── config.py                 # API keys (env vars) and Ollama model configuration
├── main.py                    # application entry point
└── src/
    ├── intent_router.py        # NLTK-based keyword intent classifier
    ├── fetchers.py               # Wikipedia / arXiv / News / Weather / Wolfram / YouTube
    ├── rag_engine.py               # Ollama HTTP client + RAG prompt construction
    ├── audio_manager.py             # threaded pygame.mixer audio cues
    └── ui.py                         # customtkinter GUI, threading, Markdown rendering
```

## Setup

```bash
pip install -r requirements.txt
```

### 1. Install and run Ollama

Download [Ollama](https://ollama.com), then:

```bash
ollama serve
ollama pull llama3      # or any model you prefer
```

Set `OLLAMA_MODEL` if you pull a different model name:

```bash
export OLLAMA_MODEL="mistral"
```

### 2. API keys for live-data fetchers

This is an optional step
Every fetcher degrades gracefully with a clear message if its key is
missing — the app is fully usable without any of these. Set the ones you
want:

```bash
export WEATHER_API_KEY="your_openweathermap_key"
export WOLFRAM_APP_ID="your_wolframalpha_appid"
export NEWS_API_KEY="your_newsapi_key"
```

Wikipedia, arXiv, and YouTube search need no API key.

### 3. Audio cues

This is an optional step
Drop `.wav` files at `assets/audio/submit.wav`, `assets/audio/response_ready.wav`,
and `assets/audio/error.wav` for sound feedback, or set `AUDIO_ENABLED=false`
to disable audio entirely. Missing files or an unavailable audio device are
handled gracefully — the app runs fine without them.

## Usage

```bash
python main.py
```

Pick a mode at the top of the window, type a message, and press Enter or
click Send. Use **Clear Session** to reset the conversation history.

## How it works

- **`IntentRouter`** tokenizes the message with NLTK, strips stopwords
  (keeping WH-words like "who"/"what" since they're useful signals), and
  scores the remaining tokens against a keyword gazetteer per category
  (`WIKIPEDIA`, `ARXIV`, `NEWS`, `WOLFRAM`, `WEATHER`, `YOUTUBE`, `CODE`,
  `GENERAL_CHAT`). A few strong regex patterns (e.g. a raw arithmetic
  expression, a YouTube URL) short-circuit straight to their category.
- **`fetchers.py`** — every function returns a `FetchResult` and never
  raises; a failed API call, missing key, or network timeout becomes a
  clear structured message instead of crashing the app.
- **`OllamaRAGEngine`** talks to Ollama's local `/api/chat` REST endpoint
  with plain `requests` calls (no SDK dependency), tracks conversation
  history across turns, trims it to a configurable max length, and builds
  a RAG system message (`chat_with_context`) that injects fetched content
  ahead of the user's question for a single turn without permanently
  polluting the conversation history.
- **`ui.py`** — every network/LLM call runs on a background
  `threading.Thread`; results are delivered back to the Tkinter main
  thread via a queue polled with `.after(...)`, so the window never
  freezes. LLM Markdown output is converted to HTML (`markdown2`) and
  rendered with `tkinterweb`, falling back to a plain-text view if either
  package is unavailable.

## Testing individual stages

```bash
python -m src.intent_router
python -m src.fetchers
python -m src.rag_engine
python -m src.audio_manager
```

Each prints a self-contained smoke test — `fetchers.py` and `rag_engine.py`
work offline too, demonstrating their graceful-failure behavior when a
network/API/Ollama call can't succeed.

## Troubleshooting

- **"Could not connect to Ollama"** — make sure `ollama serve` is running
  and reachable at `OLLAMA_BASE_URL` (default `http://localhost:11434`).
- **No sound** — check `AUDIO_ENABLED`, that `.wav` files exist at the
  configured paths, and that your system has a working audio device.
- **A fetcher always fails** — check whether it needs an API key (see
  Setup step 2); the error message names the missing environment variable.
