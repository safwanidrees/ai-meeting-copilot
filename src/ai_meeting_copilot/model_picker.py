"""Interactive model menu shown at startup (skipped when --model / COPILOT_MODEL is set)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .config import DEFAULT_MODELS

# (model id, short note). OpenRouter prices are per million input / output tokens.
MODEL_CHOICES: dict[str, list[tuple[str, str]]] = {
    "openrouter": [
        ("qwen/qwen3.8-27b:free", "free · rate-limited, 50 requests/day"),
        ("google/gemma-4-31b-it:free", "free · rate-limited, 50 requests/day"),
        ("nvidia/nemotron-3-super-120b-a12b:free", "free · rate-limited, 50 requests/day"),
        ("openai/gpt-5.6-luna", "cheap · $0.20 / $1.20"),
        ("openai/gpt-4o", "solid all-rounder"),
        ("anthropic/claude-sonnet-5.5", "best answers · $2 / $10"),
    ],
    "openai": [
        ("gpt-4o", "solid all-rounder"),
        ("gpt-4o-mini", "faster and cheaper"),
    ],
    "anthropic": [
        ("claude-sonnet-5-5", "fast, strong answers"),
        ("claude-opus-5-5", "most capable, slower"),
        ("claude-haiku-4-5-20251001", "fastest, cheapest"),
    ],
}

STATE_FILE = Path.home() / ".config" / "ai-meeting-copilot" / "last_model.json"


def load_last(provider: str, path: Path = STATE_FILE) -> str | None:
    try:
        return json.loads(path.read_text()).get(provider)
    except (OSError, ValueError):
        return None


def save_last(provider: str, model: str, path: Path = STATE_FILE) -> None:
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    data[provider] = model
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except OSError:
        pass  # remembering the choice is a convenience, never a reason to fail


def pick_model(
    provider: str,
    default: str | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> str:
    """Show a numbered menu and return the chosen model id.

    Enter keeps the default (last used, else the provider default); "c" lets you type any id.
    """
    choices = list(MODEL_CHOICES.get(provider, []))
    default = default or DEFAULT_MODELS[provider]
    if default not in [m for m, _ in choices]:
        choices.append((default, "last used"))

    print_fn(f"Choose a model ({provider}):")
    for n, (model, note) in enumerate(choices, start=1):
        marker = "  ← default" if model == default else ""
        print_fn(f"  {n}. {model:42} {note}{marker}")
    print_fn("  c. enter another model id")

    while True:
        try:
            answer = input_fn(f"Model [Enter = {default}]: ").strip()
        except EOFError:
            return default
        if not answer:
            return default
        if answer.lower() == "c":
            custom = input_fn("Model id: ").strip()
            if custom:
                return custom
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1][0]
        if "/" in answer or "-" in answer:  # typed an id directly
            return answer
        print_fn(f"  Please type 1-{len(choices)}, c, or press Enter.")
