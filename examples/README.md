# Example: interview prep material

`interview-prep/` is a ready-to-run example of context material for the copilot:

| File | Contents |
|---|---|
| `about-me.md` | Your profile. **Replace every "EDIT ME" line** with your real details. |
| `project-ai-meeting-copilot.md` | How to explain this project in an interview (problem, architecture, challenges, results). |
| `ml-notes.md` | Machine-learning viva notes (generative vs discriminative, bias-variance, metrics, RAG, transformers). |
| `common-questions.md` | Prepared answers to common interview questions. |

`demo-interview.wav` is an 18-second mock interviewer asking two questions, so you can test without a
live call. Create it with `uv run python scripts/make_demo_audio.py` (macOS).

```bash
# Try it with the demo recording:
uv run copilot -c examples/interview-prep --audio-file examples/demo-interview.wav --mode auto

# Practise live (manual mode: answer out loud yourself, then press Enter to compare):
uv run copilot -c examples/interview-prep --mode manual --hotwords "Qdrant, Whisper, BM25"
```

Tips: one topic per paragraph under a clear heading, real numbers, your own words. Keep private
material (your real resume, company notes) in a folder outside the repo or in `my-material/`,
which is git-ignored.
