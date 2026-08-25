"""
Talks to Ollama's local HTTP API (localhost:11434 by default) -- plain
chat, RAG-augmented chat (folds fetched context into a system message),
and conversation history tracking.

Uses plain `requests` against Ollama's REST API instead of their SDK, so
there's no extra dependency beyond what's already needed elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)


# Schema

@dataclass
class ChatMessage:
    role: str      # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    """Uniform return type from `OllamaRAGEngine` calls — never raises on connection issues."""
    success: bool
    text: str
    model: str
    error: str | None = None
    context_used: str | None = None


# Engine

class OllamaRAGEngine:
    """
    Thin wrapper around Ollama's /api/chat, plus history tracking.

        engine = OllamaRAGEngine()
        engine.chat("What's the capital of France?")
        engine.chat_with_context("Summarize this", context)
        engine.clear_session()
    """

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
        temperature: float = config.OLLAMA_TEMPERATURE,
        timeout_seconds: float = config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
        max_history_turns: int = config.OLLAMA_MAX_HISTORY_TURNS,
        system_prompt: str = "You are a helpful, concise local AI assistant.",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_history_turns = max_history_turns
        self.default_system_prompt = system_prompt

        self._history: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]

    # ----------------------------------------------------------------- #
    # Session management
    # ----------------------------------------------------------------- #

    def clear_session(self, system_prompt: str | None = None) -> None:
        """Clears conversation history and starts a fresh session, optionally with a new system prompt."""
        prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        self._history = [ChatMessage(role="system", content=prompt)]
        logger.info("Session cleared. New system prompt: %r", prompt[:80])

    def get_history(self) -> list[ChatMessage]:
        """Returns a copy of the current conversation history."""
        return list(self._history)

    def _append_and_trim_history(self, user_message: ChatMessage, assistant_message: ChatMessage | None) -> None:
        self._history.append(user_message)
        if assistant_message is not None:
            self._history.append(assistant_message)

        # Keep the system message + the most recent `max_history_turns` (user, assistant) pairs,
        # so long sessions don't grow the prompt unboundedly.
        system_msgs = [m for m in self._history if m.role == "system"]
        turn_msgs = [m for m in self._history if m.role != "system"]
        max_messages = self.max_history_turns * 2
        if len(turn_msgs) > max_messages:
            turn_msgs = turn_msgs[-max_messages:]
        self._history = system_msgs + turn_msgs

    # ----------------------------------------------------------------- #
    # Prompt construction
    # ----------------------------------------------------------------- #

    def build_rag_system_message(self, context: str) -> str:
        """Builds a RAG-augmented system message combining the base persona with retrieved context."""
        truncated_context = context[: config.RAG_MAX_CONTEXT_CHARS]
        if len(context) > config.RAG_MAX_CONTEXT_CHARS:
            truncated_context += "\n...[context truncated]"
        return self.default_system_prompt + "\n\n" + config.RAG_SYSTEM_PROMPT_TEMPLATE.format(context=truncated_context)

    # ----------------------------------------------------------------- #
    # Core HTTP call
    # ----------------------------------------------------------------- #

    def _call_ollama_chat(self, messages: list[ChatMessage]) -> GenerationResult:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            text = data.get("message", {}).get("content", "").strip()
            if not text:
                return GenerationResult(
                    success=False, text="", model=self.model,
                    error="Ollama returned an empty response.",
                )
            return GenerationResult(success=True, text=text, model=self.model)

        except requests.exceptions.ConnectionError:
            error = (
                f"Could not connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running locally (`ollama serve`) and the model "
                f"'{self.model}' is pulled (`ollama pull {self.model}`)."
            )
            logger.error(error)
            return GenerationResult(success=False, text="", model=self.model, error=error)

        except requests.exceptions.Timeout:
            error = f"Ollama request timed out after {self.timeout_seconds}s."
            logger.error(error)
            return GenerationResult(success=False, text="", model=self.model, error=error)

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text[:300] if exc.response is not None else ""
            error = f"Ollama returned HTTP {status}: {body}"
            logger.error(error)
            return GenerationResult(success=False, text="", model=self.model, error=error)

        except (ValueError, KeyError) as exc:
            error = f"Could not parse Ollama response: {exc}"
            logger.error(error)
            return GenerationResult(success=False, text="", model=self.model, error=error)

        except Exception as exc:  # pragma: no cover - defensive catch-all
            error = f"Unexpected error calling Ollama: {exc}"
            logger.exception(error)
            return GenerationResult(success=False, text="", model=self.model, error=error)

    # ----------------------------------------------------------------- #
    # Public generation methods
    # ----------------------------------------------------------------- #

    def chat(self, user_prompt: str) -> GenerationResult:
        """
        Plain Chat Mode: sends the user's prompt plus running conversation
        history to Ollama, with no retrieved context injected.
        """
        user_message = ChatMessage(role="user", content=user_prompt)
        messages_to_send = self._history + [user_message]

        result = self._call_ollama_chat(messages_to_send)

        assistant_message = ChatMessage(role="assistant", content=result.text) if result.success else None
        self._append_and_trim_history(user_message, assistant_message)
        return result

    def chat_with_context(self, user_prompt: str, context: str) -> GenerationResult:
        """
        Builds a RAG system message from `context` and sends it ahead of
        the user's prompt. Only applies to this one call -- it's not saved
        into history, so context from one turn doesn't leak into later ones.
        """
        rag_system_message = ChatMessage(role="system", content=self.build_rag_system_message(context))
        user_message = ChatMessage(role="user", content=user_prompt)

        non_system_history = [m for m in self._history if m.role != "system"]
        messages_to_send = [rag_system_message] + non_system_history + [user_message]

        result = self._call_ollama_chat(messages_to_send)
        result.context_used = context

        assistant_message = ChatMessage(role="assistant", content=result.text) if result.success else None
        self._append_and_trim_history(user_message, assistant_message)
        return result

    # ----------------------------------------------------------------- #
    # Health check
    # ----------------------------------------------------------------- #

    def is_available(self) -> bool:
        """Quick connectivity check — pings Ollama's root endpoint without waiting for a full generation."""
        try:
            response = requests.get(self.base_url, timeout=3.0)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_local_models(self) -> list[str]:
        """Returns the list of model names currently pulled in the local Ollama instance."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except requests.exceptions.RequestException as exc:
            logger.warning("Could not list local Ollama models: %s", exc)
            return []


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    engine = OllamaRAGEngine()
    print(f"Ollama available at {engine.base_url}: {engine.is_available()}")
    print(f"Locally pulled models: {engine.list_local_models()}")

    print("\n=== Chat Mode (expected to fail gracefully if Ollama isn't running) ===")
    result = engine.chat("Hello! What can you help me with?")
    print(f"success={result.success}")
    print(result.text if result.success else result.error)

    print("\n=== AI+RAG Mode ===")
    fake_context = "[Wikipedia results for 'Alan Turing']\nAlan Turing was a British mathematician and computer scientist."
    result = engine.chat_with_context("Summarize who this person was in one sentence.", fake_context)
    print(f"success={result.success}")
    print(result.text if result.success else result.error)

    print(f"\nConversation history length after 2 turns: {len(engine.get_history())}")
    engine.clear_session()
    print(f"Conversation history length after clear_session(): {len(engine.get_history())}")
