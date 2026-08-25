"""
Entry point -- builds the window (src/ui.py) and starts the event loop.

    python main.py

You'll want Ollama running for the LLM parts to actually work:
    ollama serve
    ollama pull llama3

Without it the app still opens fine, LLM calls just return a clear
connection error instead of hanging. Web Search mode doesn't need Ollama
at all.
"""

from __future__ import annotations

import logging
import sys

import config

logger = logging.getLogger(__name__)


def main() -> int:
    config.configure_logging()

    try:
        import customtkinter  # noqa: F401
    except ImportError:
        logger.error(
            "customtkinter is not installed. Install project dependencies first:\n"
            "    pip install -r requirements.txt"
        )
        return 1

    try:
        from src.ui import create_app
    except Exception as exc:  # pragma: no cover - defensive guard for import-time failures
        logger.exception("Failed to import the UI module: %s", exc)
        return 1

    logger.info("Starting %s...", config.APP_TITLE)
    app = create_app()

    try:
        app.mainloop()
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("Interrupted by user, shutting down.")
    finally:
        try:
            app.audio_manager.shutdown()
        except Exception:  # pragma: no cover - best-effort cleanup on exit
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
