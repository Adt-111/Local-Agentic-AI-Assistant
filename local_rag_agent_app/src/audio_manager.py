"""
Threaded, non-blocking audio cue playback using `pygame.mixer`. Designed to
never block or crash the Tkinter GUI thread: mixer initialization failures
(e.g. no audio device available, common in headless/CI/remote-desktop
environments) and missing sound files both degrade to no-ops with a logged
warning, rather than raising.
"""

from __future__ import annotations

import logging
import os
import threading

import config

logger = logging.getLogger(__name__)

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore


class AudioManager:
    """
    Manages short UI sound cues (submit / response-ready / error) played on
    a background thread so `pygame.mixer` I/O never stalls the GUI's main
    event loop. Safe to instantiate even when no audio device is present or
    `pygame` isn't installed — playback calls simply become no-ops.
    """

    def __init__(self, enabled: bool = config.AUDIO_ENABLED) -> None:
        self.enabled = enabled and pygame is not None
        self._mixer_ready = False
        self._lock = threading.Lock()

        if not self.enabled:
            if pygame is None:
                logger.info("AudioManager disabled: pygame is not installed.")
            else:
                logger.info("AudioManager disabled via configuration.")
            return

        self._init_mixer()

    def _init_mixer(self) -> None:
        """Attempts to initialize pygame's mixer. Failures (e.g. no audio device) are logged, not raised."""
        try:
            pygame.mixer.init()
            self._mixer_ready = True
            logger.info("pygame.mixer initialized successfully.")
        except Exception as exc:
            self._mixer_ready = False
            logger.warning("Could not initialize pygame.mixer (audio cues will be disabled): %s", exc)

    def _play_sync(self, sound_path: str) -> None:
        """Actually loads and plays a sound file. Runs on a background thread via `play_async`."""
        if not self._mixer_ready:
            return
        if not os.path.exists(sound_path):
            logger.warning("Audio cue file not found: %s", sound_path)
            return
        try:
            with self._lock:
                sound = pygame.mixer.Sound(sound_path)
                sound.play()
        except Exception as exc:
            logger.warning("Failed to play audio cue '%s': %s", sound_path, exc)

    def play_async(self, sound_path: str) -> None:
        """
        Plays a sound file on a background thread, returning immediately so
        the caller (typically the Tkinter main thread) never blocks on
        audio I/O.
        """
        if not self.enabled:
            return
        thread = threading.Thread(target=self._play_sync, args=(sound_path,), daemon=True)
        thread.start()

    def play_submit_cue(self) -> None:
        """Convenience wrapper: plays the 'message submitted' cue."""
        self.play_async(config.SOUND_ON_SUBMIT)

    def play_response_cue(self) -> None:
        """Convenience wrapper: plays the 'response ready' cue."""
        self.play_async(config.SOUND_ON_RESPONSE)

    def play_error_cue(self) -> None:
        """Convenience wrapper: plays the 'error occurred' cue."""
        self.play_async(config.SOUND_ON_ERROR)

    def stop_all(self) -> None:
        """Stops any currently playing sounds."""
        if not self._mixer_ready:
            return
        try:
            pygame.mixer.stop()
        except Exception as exc:
            logger.warning("Failed to stop audio playback: %s", exc)

    def shutdown(self) -> None:
        """Cleanly releases the mixer, e.g. on application exit."""
        if not self._mixer_ready:
            return
        try:
            pygame.mixer.quit()
            self._mixer_ready = False
            logger.info("pygame.mixer shut down.")
        except Exception as exc:
            logger.warning("Error shutting down pygame.mixer: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    import time

    logging.basicConfig(level=logging.INFO)

    manager = AudioManager()
    print(f"AudioManager enabled: {manager.enabled} | mixer ready: {manager._mixer_ready}")

    # These will log "file not found" warnings if the placeholder assets
    # under assets/audio/ haven't been supplied yet — that's expected and
    # demonstrates the graceful-degradation path, not a crash.
    manager.play_submit_cue()
    time.sleep(0.2)
    manager.play_response_cue()
    time.sleep(0.2)
    manager.play_error_cue()
    time.sleep(0.2)

    manager.shutdown()
    print("AudioManager smoke test completed without raising.")
