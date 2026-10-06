"""Command-line entry point: `uv run copilot`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="copilot",
        description="Listen to meeting audio, transcribe it live and stream suggested answers.",
    )
    p.add_argument("-c", "--context", nargs="+", type=Path, default=[], metavar="PATH",
                   help="files or folders to answer from (pdf, docx, md, txt, ...)")
    p.add_argument("--provider", choices=["openai", "anthropic", "openrouter"],
                   help="LLM provider (default: whichever API key is set)")
    p.add_argument("--no-pick", action="store_true", help="don't show the model menu at startup")
    p.add_argument("--model", help="LLM model (default: gpt-4o / claude-sonnet-5-5 / anthropic/claude-sonnet-5.5)")
    p.add_argument("--fallback-models", metavar="IDS",
                   help="OpenRouter: comma-separated models to try if the main one is rate-limited "
                        "(default for :free models: other free models)")
    p.add_argument("--mode", choices=["auto", "manual"], help="trigger mode at start (default: auto)")
    p.add_argument("--transcribe-only", action="store_true", help="live captions only, no LLM")

    audio = p.add_argument_group("audio")
    audio.add_argument("--device", help="input device name substring or index (default: auto-detect BlackHole etc.)")
    audio.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    audio.add_argument("--audio-file", type=Path, help="replay an audio file in real time instead of a live device")
    audio.add_argument("--silence-ms", type=int, help="silence that ends an utterance in auto mode (default: 900)")
    audio.add_argument("--noise-gate", type=float, help="ignore audio quieter than this dBFS (default: -55)")

    stt = p.add_argument_group("speech-to-text")
    stt.add_argument("--whisper-model", help="tiny.en | base.en | small.en | medium.en | large-v3 | ... (default: small.en)")
    stt.add_argument("--language", help="spoken language code, e.g. en, de (use a non-.en model for non-English)")
    stt.add_argument("--hotwords", help="names/terms Whisper should expect, e.g. 'Qdrant, LangChain, Kubernetes'")

    rag = p.add_argument_group("context retrieval")
    rag.add_argument("--embeddings", choices=["openai", "openrouter", "cohere", "none"], help="embedding provider (default: auto)")
    rag.add_argument("--no-rerank", action="store_true", help="skip Cohere reranking")
    rag.add_argument("--top-k", type=int, help="context chunks per question (default: 4)")

    out = p.add_argument_group("session")
    out.add_argument("--export-dir", type=Path, help="where session history is saved (default: ./sessions)")
    out.add_argument("--no-export", action="store_true", help="don't save session history")
    return p


def apply_args(settings: Settings, args: argparse.Namespace) -> Settings:
    overrides = {
        "provider": args.provider,
        "model": args.model,
        "mode": args.mode,
        "device": args.device,
        "audio_file": args.audio_file,
        "min_silence_ms": args.silence_ms,
        "noise_gate_dbfs": args.noise_gate,
        "whisper_model": args.whisper_model,
        "language": args.language,
        "hotwords": args.hotwords,
        "embedding_provider": args.embeddings,
        "top_k": args.top_k,
        "export_dir": args.export_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    if args.fallback_models:
        settings.fallback_models = [m.strip() for m in args.fallback_models.split(",") if m.strip()]
    if args.context:
        settings.context_paths = list(args.context)
    if args.transcribe_only:
        settings.transcribe_only = True
    if args.no_rerank:
        settings.rerank = False
    if args.no_export:
        settings.export = False
    return settings


def print_devices() -> None:
    from .audio import default_input_device, list_input_devices

    default = default_input_device()
    for d in list_input_devices():
        tags = []
        if d.is_loopback:
            tags.append("loopback")
        if default and d.index == default.index:
            tags.append("default")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        print(f"{d.index:>3}  {d.name}  ({d.channels} ch, {d.samplerate} Hz){suffix}")


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.list_devices:
        print_devices()
        return 0

    settings = apply_args(Settings.from_env(), args)

    from .display import Display
    from .session import Session

    display = Display()
    session = Session()

    # --- LLM (checked first: fail fast before loading models) -------------------
    llm = None
    if not settings.transcribe_only:
        provider = settings.resolve_provider()
        if provider is None:
            display.error("No LLM API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY or OPENROUTER_API_KEY (see .env.example),")
            display.error("or run with --transcribe-only for live captions.")
            return 2
        from .config import PROVIDER_KEYS

        key_name, key = PROVIDER_KEYS[provider]
        if key() is None:
            display.error(f"--provider {provider} needs {key_name} (see .env.example).")
            return 2
        from .llm import choose_fallbacks, create_llm, openrouter_models

        if settings.model is None and not args.no_pick and sys.stdin.isatty():
            from .model_picker import load_last, pick_model, save_last

            settings.model = pick_model(provider, load_last(provider))
            save_last(provider, settings.model)
            display.print()
        model = settings.resolve_model(provider)
        fallbacks: list[str] = []
        if provider == "openrouter" and (settings.fallback_models or model.endswith(":free")):
            fallbacks = choose_fallbacks(model, settings.fallback_models, openrouter_models())
        elif settings.fallback_models:
            display.warn("--fallback-models only works with OpenRouter; ignoring it.")
        llm = create_llm(provider, model, settings.max_answer_tokens, fallbacks)
        if fallbacks:
            display.print(f"Fallback models: {', '.join(fallbacks)}")

    # --- Context documents ------------------------------------------------------
    retriever = None
    if settings.context_paths:
        from .rag import build_retriever

        display.print("Indexing context files...")
        try:
            retriever = build_retriever(settings, display.print, display.warn)
        except (FileNotFoundError, ValueError) as exc:
            display.error(f"Could not load context: {exc}")
            return 2
    else:
        display.print("No context files provided — running without context.")
    display.print()
    display.print(f"Session ID: {session.id}")
    display.print(f"LLM: {llm.name}" if llm else "LLM: off (transcribe-only)")
    display.print()

    # --- Audio source -----------------------------------------------------------
    from .audio import DeviceAudioSource, FileAudioSource, choose_device, default_input_device, list_input_devices

    if settings.audio_file:
        source = FileAudioSource(settings.audio_file, settings.highpass_hz)
        display.print(f"Replaying {settings.audio_file} ({source.duration_s:.1f}s)")
    else:
        device = choose_device(list_input_devices(), settings.device)
        if device is None and settings.device:
            display.error(f"No input device matches {settings.device!r}. Run `copilot --list-devices`.")
            return 2
        if device is None:
            device = default_input_device()
            if device is None:
                display.error("No audio input device available.")
                return 2
            display.warn("No loopback device found, so you'll only hear what the microphone picks up.")
            display.warn("To capture meeting audio directly, install BlackHole: brew install blackhole-2ch (see README).")
            display.print(f"Using microphone: {device.name} (index {device.index})")
        else:
            label = "loopback" if device.is_loopback else "input"
            display.print(f"Using {label} device: {device.name} (index {device.index})")
        source = DeviceAudioSource(device, settings.highpass_hz)

    # --- Models -----------------------------------------------------------------
    from .transcriber import Transcriber
    from .vad import SileroVAD, SpeechSegmenter, VADConfig

    display.print()
    display.print("Loading VAD model...")
    segmenter = SpeechSegmenter(
        SileroVAD(),
        VADConfig(
            threshold=settings.vad_threshold,
            min_silence_ms=settings.min_silence_ms,
            min_speech_ms=settings.min_speech_ms,
            speech_pad_ms=settings.speech_pad_ms,
            max_segment_s=settings.max_segment_s,
            noise_gate_dbfs=settings.noise_gate_dbfs,
        ),
    )
    display.print("VAD model loaded.")
    display.print()
    display.print(f"Loading Whisper model ({settings.whisper_model})...")
    transcriber = Transcriber(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.language,
        beam_size=settings.beam_size,
        hotwords=settings.hotwords,
    )
    transcriber.warmup()
    display.print("Whisper model loaded.")
    display.print()

    from .copilot import Copilot

    copilot = Copilot(
        settings=settings,
        source=source,
        segmenter=segmenter,
        transcriber=transcriber,
        llm=llm,
        retriever=retriever,
        display=display,
        session=session,
    )
    copilot.run()

    display.history(session)
    if settings.export and session.exchanges:
        md, js = session.export(settings.export_dir)
        display.print()
        display.print(f"Saved: {md}  and  {js}", "green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
