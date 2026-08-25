"""
The desktop window itself -- mode selector (Chat/Search/RAG), a
conversation view that renders Markdown to HTML, and everything wired to
run on background threads so the UI doesn't freeze on a slow request.

Doesn't start the Tk main loop -- main.py does that.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import customtkinter as ctk

import config
from src.audio_manager import AudioManager
from src.fetchers import FETCHER_DISPATCH, FetchResult
from src.intent_router import Intent, IntentRouter
from src.rag_engine import GenerationResult, OllamaRAGEngine

logger = logging.getLogger(__name__)

try:
    import markdown2
    from tkinterweb import HtmlFrame
    _HTML_RENDERING_AVAILABLE = True
except ImportError:  # pragma: no cover
    markdown2 = None  # type: ignore
    HtmlFrame = None  # type: ignore
    _HTML_RENDERING_AVAILABLE = False


MODE_CHAT = "Chat"
MODE_SEARCH = "Web Search"
MODE_RAG = "AI + RAG"
MODES = [MODE_CHAT, MODE_SEARCH, MODE_RAG]

# Minimal CSS so tkinterweb's rendered HTML doesn't look like an unstyled web page.
_HTML_STYLE = """
<style>
  body { font-family: Helvetica, Arial, sans-serif; font-size: 14px; color: #1a1a1a;
         background-color: #ffffff; margin: 8px; }
  .turn { margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0; }
  .role-user { color: #1f6feb; font-weight: bold; }
  .role-assistant { color: #2f9e44; font-weight: bold; }
  .role-system { color: #888888; font-style: italic; }
  .meta { color: #888888; font-size: 12px; }
  code { background-color: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
  pre { background-color: #f0f0f0; padding: 8px; border-radius: 4px; overflow-x: auto; }
</style>
"""


class AsyncTaskRunner:
    """
    Runs a blocking callable on a background `threading.Thread` and safely
    delivers its result back to the Tkinter main thread via `.after(...)`,
    since Tkinter widgets must only be touched from the main thread.
    """

    def __init__(self, root: ctk.CTk) -> None:
        self._root = root

    def run(
        self,
        task: Callable[[], object],
        on_success: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        result_queue: "queue.Queue" = queue.Queue()

        def worker() -> None:
            try:
                result = task()
                result_queue.put(("success", result))
            except Exception as exc:  # pragma: no cover - defensive catch-all
                logger.exception("Background task failed: %s", exc)
                result_queue.put(("error", exc))

        def poll_queue() -> None:
            try:
                status, payload = result_queue.get_nowait()
            except queue.Empty:
                self._root.after(50, poll_queue)
                return
            if status == "success":
                on_success(payload)
            else:
                on_error(payload)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._root.after(50, poll_queue)


class ChatApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode(config.APPEARANCE_MODE)
        ctk.set_default_color_theme(config.COLOR_THEME)

        self.title(config.APP_TITLE)
        self.geometry(config.APP_GEOMETRY)
        self.minsize(760, 560)

        self.intent_router = IntentRouter()
        self.rag_engine = OllamaRAGEngine()
        self.audio_manager = AudioManager()
        self.task_runner = AsyncTaskRunner(self)

        self.mode_var = ctk.StringVar(value=MODE_CHAT)
        self._conversation_turns: list[tuple[str, str]] = []  # (role, content)
        self._is_busy = False

        self._build_layout()
        self._render_conversation()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # Layout construction
    # ------------------------------------------------------------------ #

    def _build_layout(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_top_bar()
        self._build_conversation_view()
        self._build_input_bar()
        self._build_status_bar()

    def _build_top_bar(self) -> None:
        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        top_frame.grid_columnconfigure(1, weight=1)

        mode_label = ctk.CTkLabel(top_frame, text="Mode:", font=ctk.CTkFont(weight="bold"))
        mode_label.grid(row=0, column=0, padx=(4, 8), pady=6, sticky="w")

        mode_selector = ctk.CTkSegmentedButton(
            top_frame, values=MODES, variable=self.mode_var, command=self._on_mode_change
        )
        mode_selector.grid(row=0, column=1, pady=6, sticky="w")

        clear_button = ctk.CTkButton(
            top_frame, text="Clear Session", width=130, command=self._on_clear_session
        )
        clear_button.grid(row=0, column=2, padx=6, pady=6, sticky="e")

    def _build_conversation_view(self) -> None:
        container = ctk.CTkFrame(self)
        container.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        if _HTML_RENDERING_AVAILABLE:
            self.html_frame = HtmlFrame(container, messages_enabled=False)
            self.html_frame.grid(row=0, column=0, sticky="nsew")
            self.text_view = None
        else:  # pragma: no cover - exercised only when tkinterweb/markdown2 are missing
            logger.warning(
                "tkinterweb/markdown2 not available — falling back to plain-text rendering. "
                "Install both for full Markdown-to-HTML display."
            )
            self.html_frame = None
            self.text_view = ctk.CTkTextbox(container, wrap="word", state="disabled")
            self.text_view.grid(row=0, column=0, sticky="nsew")

    def _build_input_bar(self) -> None:
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(
            input_frame, placeholder_text="Ask me anything... (Enter to send)"
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(4, 8), pady=8)
        self.input_entry.bind("<Return>", lambda _event: self._on_submit())

        self.send_button = ctk.CTkButton(input_frame, text="Send", width=90, command=self._on_submit)
        self.send_button.grid(row=0, column=1, padx=(0, 4), pady=8)

    def _build_status_bar(self) -> None:
        self.status_label = ctk.CTkLabel(
            self, text=self._ollama_status_text(), anchor="w",
            font=ctk.CTkFont(size=11), text_color="gray60",
        )
        self.status_label.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))

    def _ollama_status_text(self) -> str:
        available = self.rag_engine.is_available()
        status = "connected" if available else "not reachable"
        return f"Ollama ({self.rag_engine.model} @ {self.rag_engine.base_url}): {status}"

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_mode_change(self, _value: str) -> None:
        logger.info("Mode changed to: %s", self.mode_var.get())

    def _on_clear_session(self) -> None:
        self.rag_engine.clear_session()
        self._conversation_turns.clear()
        self._render_conversation()
        self.status_label.configure(text=self._ollama_status_text())
        logger.info("Session cleared by user.")

    def _on_close(self) -> None:
        self.audio_manager.shutdown()
        self.destroy()

    def _on_submit(self) -> None:
        if self._is_busy:
            return  # ignore double-submits while a request is in flight

        user_text = self.input_entry.get().strip()
        if not user_text:
            return

        self.input_entry.delete(0, "end")
        self.audio_manager.play_submit_cue()

        self._conversation_turns.append(("user", user_text))
        self._render_conversation()
        self._set_busy(True, status_text="Thinking...")

        mode = self.mode_var.get()
        self.task_runner.run(
            task=lambda: self._process_message(user_text, mode),
            on_success=self._on_response_success,
            on_error=self._on_response_error,
        )

    # ------------------------------------------------------------------ #
    # Background work (runs off the Tk main thread via AsyncTaskRunner)
    # ------------------------------------------------------------------ #

    def _process_message(self, user_text: str, mode: str) -> dict:
        """
        Executes the actual (network/LLM-bound) work for one turn. Runs on
        a background thread — must not touch any Tkinter widgets directly.
        """
        if mode == MODE_CHAT:
            result: GenerationResult = self.rag_engine.chat(user_text)
            return {"kind": "chat", "generation": result}

        intent_result = self.intent_router.classify(user_text)

        fetch_result: FetchResult | None = None
        if intent_result.intent != Intent.GENERAL_CHAT and intent_result.intent.value in FETCHER_DISPATCH:
            fetcher_fn = FETCHER_DISPATCH[intent_result.intent.value]
            fetch_result = fetcher_fn(intent_result.cleaned_query or user_text)

        if mode == MODE_SEARCH:
            return {"kind": "search", "intent": intent_result, "fetch_result": fetch_result}

        # AI + RAG mode: fold fetched context (if any) into the LLM prompt.
        context = fetch_result.as_context_string() if fetch_result is not None else ""
        if context:
            generation = self.rag_engine.chat_with_context(user_text, context)
        else:
            generation = self.rag_engine.chat(user_text)
        return {
            "kind": "rag",
            "intent": intent_result,
            "fetch_result": fetch_result,
            "generation": generation,
        }

    # ------------------------------------------------------------------ #
    # Result handling (back on the Tk main thread)
    # ------------------------------------------------------------------ #

    def _on_response_success(self, payload: dict) -> None:
        kind = payload["kind"]

        if kind == "chat":
            generation: GenerationResult = payload["generation"]
            self._append_assistant_turn(generation)

        elif kind == "search":
            fetch_result: FetchResult | None = payload["fetch_result"]
            intent = payload["intent"]
            if fetch_result is None:
                content = (
                    f"_No specific data source matched this query (classified as "
                    f"`{intent.intent.value}`); try AI + RAG mode or Chat mode instead._"
                )
            else:
                content = fetch_result.as_context_string()
            self._conversation_turns.append(("assistant", content))

        else:  # kind == "rag"
            generation: GenerationResult = payload["generation"]
            self._append_assistant_turn(generation)

        self._render_conversation()
        self._set_busy(False, status_text=self._ollama_status_text())
        self.audio_manager.play_response_cue()

    def _on_response_error(self, exc: Exception) -> None:
        logger.exception("Error while processing message: %s", exc)
        self._conversation_turns.append(("assistant", f"An unexpected error occurred: {exc}"))
        self._render_conversation()
        self._set_busy(False, status_text=self._ollama_status_text())
        self.audio_manager.play_error_cue()

    def _append_assistant_turn(self, generation: GenerationResult) -> None:
        if generation.success:
            self._conversation_turns.append(("assistant", generation.text))
        else:
            self._conversation_turns.append(("assistant", f"{generation.error}"))

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _set_busy(self, busy: bool, status_text: str) -> None:
        self._is_busy = busy
        self.send_button.configure(state="disabled" if busy else "normal")
        self.input_entry.configure(state="disabled" if busy else "normal")
        self.status_label.configure(text=status_text)

    def _render_conversation(self) -> None:
        if _HTML_RENDERING_AVAILABLE:
            self._render_conversation_html()
        else:
            self._render_conversation_plaintext()

    def _render_conversation_html(self) -> None:
        blocks = [_HTML_STYLE]
        if not self._conversation_turns:
            blocks.append("<p class='meta'>No messages yet -- say hello!</p>")
        for role, content in self._conversation_turns:
            role_class = f"role-{role}"
            role_label = {"user": "You", "assistant": "Assistant", "system": "System"}.get(role, role)
            body_html = markdown2.markdown(
                content, extras=["fenced-code-blocks", "code-friendly", "break-on-newline"]
            )
            blocks.append(
                f"<div class='turn'><span class='{role_class}'>{role_label}:</span>{body_html}</div>"
            )
        full_html = "\n".join(blocks)
        try:
            self.html_frame.load_html(full_html)
        except Exception as exc:  # pragma: no cover - defensive guard for rendering edge cases
            logger.warning("HTML rendering failed, falling back to raw text: %s", exc)

    def _render_conversation_plaintext(self) -> None:
        self.text_view.configure(state="normal")
        self.text_view.delete("1.0", "end")
        if not self._conversation_turns:
            self.text_view.insert("end", "No messages yet -- say hello!\n")
        for role, content in self._conversation_turns:
            role_label = {"user": "You", "assistant": "Assistant", "system": "System"}.get(role, role)
            self.text_view.insert("end", f"{role_label}: {content}\n\n")
        self.text_view.configure(state="disabled")
        self.text_view.see("end")


def create_app() -> ChatApp:
    """Factory function: builds and returns the (not-yet-mainlooped) ChatApp window."""
    config.configure_logging()
    return ChatApp()
