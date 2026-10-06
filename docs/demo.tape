# Terminal demo for the README. Regenerate with:  vhs docs/demo.tape
# Needs an LLM key in .env and examples/demo-interview.wav (scripts/make_demo_audio.py).
Output docs/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 820
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set Framerate 12

Hide
Type "export PS1='$ ' && clear"
Enter
Show

Type "uv run copilot -c examples/interview-prep --audio-file examples/demo-interview.wav \"
Enter
Type "    --model qwen/qwen3.8-27b:free --hotwords Qdrant --embeddings none --no-export"
Sleep 500ms
Enter

# model loading + 18 s recording replayed in real time + two streamed answers
Sleep 42s
