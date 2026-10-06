# Contributing

Thanks for helping! Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/safwanidrees/ai-meeting-copilot.git
cd ai-meeting-copilot
uv sync
uv run python scripts/make_demo_audio.py   # macOS: creates the demo recording
```

## Before opening a pull request

```bash
uv run ruff check src tests scripts   # lint
uv run pytest -q                      # tests run offline: no API keys or audio devices needed
```

- Add or update a test for any behaviour you change. Tests use fakes for the VAD, Whisper and
  the LLM, so they run in a few seconds.
- Keep the audio thread non-blocking: it must never wait on the display, the network or a lock
  held during an LLM call (see `test_audio_thread_never_blocks_on_display`).
- Never commit `.env`, API keys, `sessions/` (real meeting transcripts) or personal documents.

## Reporting bugs

Include the command you ran, the startup output (device, model, Whisper size) and the error
line. **Remove API keys and any private transcript text first.**
