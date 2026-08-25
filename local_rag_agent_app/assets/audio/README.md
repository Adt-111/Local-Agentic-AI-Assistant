# Audio cues

Drop your `.wav` files here to enable sound feedback:

- `submit.wav` — played when a message is sent
- `response_ready.wav` — played when the assistant's reply is ready
- `error.wav` — played when something goes wrong

None of these are required — the app runs fine without any of them, and
missing files are handled gracefully (see `src/audio_manager.py`). You can
also disable audio entirely by setting `AUDIO_ENABLED=false`.
