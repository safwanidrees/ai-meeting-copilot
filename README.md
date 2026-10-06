# AI Meeting Copilot

An open-source Python tool that listens to your computer's audio during meetings and
interviews, transcribes it in real time with Whisper, and streams a suggested answer
and the likely follow-up question to your terminal. It can answer from your own documents
(resume, notes, product docs).

![Demo: a mock interviewer asks two questions; the copilot transcribes them and streams answers grounded in the example notes](docs/demo.gif)

<sub>Real run, not a mock-up: an 18-second recorded interviewer, the example notes in
`examples/interview-prep`, and a free OpenRouter model. Qwen was busy, so OpenRouter fell back
to Nemotron automatically (`answered by …`). Recorded with [VHS](https://github.com/charmbracelet/vhs): `vhs docs/demo.tape`.</sub>

## Features

- **Real-time speech-to-text**: faster-whisper (CTranslate2, int8) runs locally on your machine.
- **Voice activity detection**: Silero VAD decides when the speaker has finished.
- **Any LLM**: GPT-4o, Claude, or hundreds of models (including free ones) via OpenRouter,
  chosen from a menu at startup. Answers stream token by token, with conversation memory.
- **Hybrid RAG over your documents**: BM25 keywords + Qdrant vectors, fused with Reciprocal
  Rank Fusion and optionally reranked by Cohere.
- **Auto or manual triggers**: answer whenever the speaker stops, or only when you press Enter.
- **Noise filtering**: high-pass filter, noise gate, VAD, and Whisper hallucination filters.
- **Resilient**: free-model fallbacks, readable errors, one-key retry, and retrieval that
  degrades instead of crashing when an API fails.
- **Session history**: printed when you quit and exported to Markdown + JSON.

## Quick start (5 minutes, no audio setup)

```bash
git clone https://github.com/safwanidrees/ai-meeting-copilot.git && cd ai-meeting-copilot
uv sync                        # installs Python 3.12 + dependencies
cp .env.example .env           # add one key: OPENROUTER_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY

# Create an 18-second mock interview recording (macOS text-to-speech):
uv run python scripts/make_demo_audio.py

# Replay it against the example notes:
uv run copilot -c examples/interview-prep --audio-file examples/demo-interview.wav
```

The demo recording is generated on your machine rather than shipped, because macOS voices are
licensed for personal use. On Linux/Windows, record any interview question yourself and pass it
with `--audio-file`.

Pick a model from the menu (a `:free` one is fine for this test). You'll see both questions
transcribed and answered, with `sources: ml-notes.md` showing which notes were used. The first
run downloads the Whisper model (~500 MB).

No API key yet? `uv run copilot --transcribe-only` gives live captions with no LLM.

## Setup

**Requirements:** macOS, Linux or Windows, [uv](https://docs.astral.sh/uv/), and one API key:

| Key | Gives you |
|---|---|
| `OPENROUTER_API_KEY` ([get one](https://openrouter.ai/keys)) | Claude, GPT and free models through one key, plus document embeddings |
| `OPENAI_API_KEY` | GPT-4o and embeddings |
| `ANTHROPIC_API_KEY` | Claude |
| `COHERE_API_KEY` (optional) | Reranking for better document search |

A Claude.ai Pro/Max subscription does **not** include API access; the API is billed separately.

### Capturing meeting audio (macOS)

A microphone only hears your room. To hear the *meeting* directly, route system audio into
the BlackHole virtual device:

1. `brew install blackhole-2ch`, then reload the audio system: `sudo killall coreaudiod` (or reboot).
2. Open **Audio MIDI Setup** → **+** → **Create Multi-Output Device**. Tick your
   speakers/headphones **and** BlackHole 2ch, set your speakers as **Primary Device**, and tick
   **Drift Correction** for BlackHole. This lets you hear the call while BlackHole copies it.
3. **System Settings → Sound → Output** → choose the Multi-Output Device. (macOS volume keys
   don't work with Multi-Output Devices; set the volume first or adjust it in the app.)
4. `uv run copilot --list-devices` should show `BlackHole 2ch [loopback]`.

The copilot picks the loopback device automatically. If none is found, it falls back to your
microphone and prints a warning.

- **Windows:** install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) and use `--device "CABLE Output"`.
- **Linux:** PulseAudio/PipeWire "Monitor of …" sources are detected automatically.

## Usage

```bash
uv run copilot                                    # model menu, auto mode, no documents
uv run copilot -c resume.pdf notes/               # answer from your own documents
uv run copilot --mode manual                      # only answer when you press Enter
uv run copilot --model qwen/qwen3.8-27b:free      # skip the menu with a specific model
uv run copilot --hotwords "Qdrant, LangChain"     # help Whisper spell your jargon
uv run copilot --transcribe-only                  # live captions, no LLM
uv run copilot --audio-file interview.mp3         # replay a recording (demo/debug)
uv run copilot --list-devices                     # show audio inputs
uv run copilot --help                             # every option
```

### Choosing a model

At startup a menu asks which model to use. The list depends on your key:

```
Choose a model (openrouter):
  1. qwen/qwen3.8-27b:free                      free · rate-limited, 50 requests/day
  ...
  6. anthropic/claude-sonnet-5.5                best answers · $2 / $10  ← default
  c. enter another model id
Model [Enter = anthropic/claude-sonnet-5.5]:
```

Press **Enter** to reuse your last choice, type a **number**, or **`c`** for any other model id.
Your choice is remembered per provider in `~/.config/ai-meeting-copilot/last_model.json`.
`--model <id>` or `--no-pick` skips the menu. Prices are per million input/output tokens and
may change; check [openrouter.ai/models](https://openrouter.ai/models).

### Keyboard commands while it runs

| Type | Effect |
|---|---|
| `m` + Enter | toggle AUTO ↔ MANUAL mode |
| Enter | (manual mode) answer everything heard since the last trigger |
| `c` + Enter | clear the manual-mode buffer |
| `r` + Enter | retry the last answer that failed (e.g. rate limit) |
| any text + Enter | ask that question directly (no audio needed) |
| `q` + Enter or Ctrl+C | end the session, print the history, save it to `sessions/` |

**Auto mode** answers whenever the speaker pauses for `--silence-ms` (900 ms by default) and
ignores very short utterances ("okay", "mm-hmm"). **Manual mode** suits long, multi-part
questions and busy meetings: it shows `[ heard 8.4s of speech — press Enter to answer ]` and
sends nothing until you press Enter.

## Using your own material

Pass files or folders with `-c`. Supported: `.pdf` (one source per page), `.docx`, `.md`,
`.txt`, `.csv`, `.json`, `.yaml`. Scanned (image-only) PDFs contain no text, so convert them first.

[`examples/interview-prep/`](examples/interview-prep) is a ready-made example: profile,
project write-up, ML viva notes, and prepared interview answers. Copy it, replace the
`EDIT ME` lines, and keep private material in `my-material/` (git-ignored).

Tips for better answers:
- **One topic per paragraph, under a clear heading**, so each chunk holds a complete idea.
- **Your own words, with real numbers**: problem → what you did → result.
- Each answer's timing line shows `sources: …`. If it's missing, nothing in your material matched.

### How retrieval works

1. Documents are split into ~800-character chunks with 150 characters of overlap.
2. **BM25** keyword index: always on, needs no API key. It folds plurals ("strengths" matches
   "strength") and still handles all-stopword questions like "Tell me about yourself".
3. **Vector index**: chunks are embedded (`text-embedding-3-small` via OpenAI or OpenRouter, or
   Cohere `embed-v4.0`) into **Qdrant**, embedded on disk at `~/.cache/ai-meeting-copilot/qdrant`
   (no server needed; set `QDRANT_URL` for a server or Qdrant Cloud). Cached by content hash,
   so unchanged documents aren't re-embedded.
4. For each question, the top 20 candidates from both indexes are fused with **Reciprocal Rank
   Fusion**, reranked by **Cohere `rerank-v3.5`** when `COHERE_API_KEY` is set, and the top 4
   are sent to the LLM.

If embeddings aren't available (no key, or $0 OpenRouter credit), it warns and uses BM25
alone. If an API fails mid-meeting, retrieval falls back to whatever still works.

## Free models (OpenRouter)

`:free` models cost nothing but are **shared and rate-limited**: everyone draws from the same
provider pool (429 errors mentioning `upstream_provider_shared_pool`), and accounts that have
bought less than $10 of credits get 20 requests/minute and **50 requests/day** across all
free models. To get the most out of them:

- When you pick a `:free` model, two other free models are added automatically as OpenRouter
  **fallbacks**. If one is busy, OpenRouter tries the next, and the timing line shows
  `answered by …`. Choose your own with `--fallback-models a,b`; putting a cheap paid model
  last (e.g. `openai/gpt-5.6-luna`) means you only pay when every free model is busy.
- Use `--mode manual` so requests are only sent when you press Enter.
- If an answer still fails, type `r` + Enter to retry it.

For important meetings, a paid model is far more reliable.

## How it works

```
 system audio ──► BlackHole ──► sounddevice ──► FramePipeline ─────► SpeechSegmenter
 (Zoom, Meet,      (virtual      (48 kHz        mono · 16 kHz ·      Silero VAD per
  Teams, ...)       device)       stereo)       100 Hz high-pass ·   32 ms frame; noise gate;
                                                512-sample frames    cuts utterances on silence
                                                                              │
                                                                              ▼ utterance
 terminal ◄── Display ◄──── LLM ◄──────── prompt ◄── HybridRetriever ◄── Whisper
 (streamed)   (labels     (OpenAI /      (system +    BM25 + Qdrant       (faster-whisper,
               coloured    Anthropic /    history +    → RRF fusion        hallucination
               live)       OpenRouter)    context)     → Cohere rerank     filter)
```

Three threads share a queue (`copilot.py`):

| Thread | Job |
|---|---|
| audio | reads frames, runs the VAD segmenter, emits utterances; never waits on the screen |
| input | reads your keyboard commands |
| worker | transcribes → retrieves → streams the answer, one job at a time |

### Long conversations

- **Listening never stops.** While an answer streams, new speech is still captured.
- **The current answer finishes first**, then everything said in the meantime is **merged into
  one next question**, so you get one answer instead of a burst.
- The LLM receives the **last 6 exchanges** as memory, so follow-ups like "and what about
  latency?" are understood.
- Monologues are cut every 45 s; overlapping speakers become one utterance (Whisper mostly
  writes down the louder voice).
- History is saved when you quit (`q` or Ctrl+C).

Known limits: no speaker labels, the model forgets anything older than 6 exchanges, and a
session that crashes before quitting isn't saved.

## Configuration

Every option can be set as a CLI flag (`uv run copilot --help`) or an environment variable in
`.env` (see [`.env.example`](.env.example)). Useful knobs:

| Setting | Default | Notes |
|---|---|---|
| `--model` | menu | Any model id from the chosen provider (OpenRouter ids look like `vendor/model`). |
| `--fallback-models` | auto for `:free` | OpenRouter only; comma-separated. |
| `--whisper-model` | `small.en` | `base.en` is about 3× faster; `medium.en` is more accurate. Use `large-v3` with `--language` for non-English. |
| `--silence-ms` | 900 | Raise it if auto mode cuts speakers off mid-question. |
| `--noise-gate` | -55 dBFS | Raise it (e.g. -45) in noisy rooms; lower it if quiet audio is ignored. |
| `--top-k` | 4 | Context chunks per question. |
| `--embeddings` | auto | `openai`, `openrouter`, `cohere` or `none`. |

## Troubleshooting

| Problem | Fix |
|---|---|
| Sits at `Listening...` | In **manual mode** this is normal: press Enter after the question. In auto mode, check the sound reaches the device; after 10 s of silence it prints `[ no sound reaching … ]`. |
| BlackHole not in `--list-devices` | Run `sudo killall coreaudiod` or reboot after installing. |
| `[ no sound reaching … ]` | Output isn't set to the Multi-Output Device, or the mic is too quiet. |
| `Rate-limited (… :free …)` | Free pool busy or 50/day used up. See [Free models](#free-models-openrouter); `r` + Enter retries. |
| `vector search disabled, using BM25 only` | No embedding key or no credit. Keyword search still works. |
| Jargon misspelled ("quadrant" for Qdrant) | `--hotwords "Qdrant, PyTorch"`, or a bigger `--whisper-model`. |
| macOS asks for microphone access | Allow it for your terminal app; macOS treats BlackHole as a microphone too. |

## Development

```bash
uv run ruff check src tests scripts   # lint
uv run pytest                          # 73 offline tests (fake VAD/LLM; real Qdrant in-memory)
```

To re-record the README demo after changing the output: `brew install vhs && vhs docs/demo.tape`
(needs an LLM key and the demo recording).

CI runs both on Ubuntu and macOS for every push and pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md).

```
src/ai_meeting_copilot/
├── cli.py            argument parsing and startup
├── config.py         Settings: defaults ← .env ← CLI flags
├── model_picker.py   startup model menu, remembers last choice
├── audio.py          device discovery, capture, resampling, high-pass, framing
├── vad.py            Silero VAD + SpeechSegmenter state machine
├── transcriber.py    faster-whisper + hallucination filters
├── llm.py            prompt, OpenAI / Anthropic / OpenRouter streaming, fallbacks, error messages
├── copilot.py        the orchestrator (threads, queue, trigger modes, retry)
├── display.py        terminal output, live label colouring
├── session.py        history, Markdown/JSON export
└── rag/
    ├── documents.py  loaders + chunking
    ├── bm25.py       keyword search (Lucene-style BM25, plural folding)
    ├── embeddings.py OpenAI / OpenRouter / Cohere embedders
    ├── vector_store.py Qdrant
    └── retriever.py  RRF fusion, Cohere rerank, HybridRetriever
examples/
└── interview-prep/   example context material
scripts/
└── make_demo_audio.py  generates examples/demo-interview.wav (macOS)
```

## Responsible use

Tell people when you are transcribing them; recording laws differ by country, and many
require everyone's consent. Don't use it where outside help is forbidden, such as exams or
assessments that disallow assistance. Transcripts are sent to your LLM provider, so check
your company's policy before using it in confidential meetings.

## License

MIT
