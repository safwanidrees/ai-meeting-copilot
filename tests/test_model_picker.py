from ai_meeting_copilot.model_picker import load_last, pick_model, save_last


def run(answers, provider="openrouter", default=None):
    feed = iter(answers)
    out = []
    model = pick_model(provider, default, input_fn=lambda _p: next(feed), print_fn=out.append)
    return model, "\n".join(out)


def test_enter_keeps_default():
    model, text = run([""])
    assert model == "anthropic/claude-sonnet-5.5"
    assert "anthropic/claude-sonnet-5.5" in text and "← default" in text


def test_number_selects_model():
    assert run(["1"])[0] == "qwen/qwen3.8-27b:free"


def test_custom_id_and_typed_id():
    assert run(["c", "meta/llama-5:free"])[0] == "meta/llama-5:free"
    assert run(["x/y"])[0] == "x/y"


def test_invalid_input_asks_again():
    model, text = run(["99", "2"])
    assert model == "google/gemma-4-31b-it:free" and "Please type" in text


def test_last_used_unknown_model_is_listed_as_default():
    model, text = run([""], default="some/other-model")
    assert model == "some/other-model" and "last used" in text


def test_eof_returns_default():
    def boom(_p):
        raise EOFError
    assert pick_model("anthropic", None, input_fn=boom, print_fn=lambda _: None) == "claude-sonnet-5-5"


def test_remembers_last_choice_per_provider(tmp_path):
    f = tmp_path / "last.json"
    assert load_last("openrouter", f) is None
    save_last("openrouter", "openai/gpt-4o", f)
    save_last("anthropic", "claude-opus-5-5", f)
    assert load_last("openrouter", f) == "openai/gpt-4o"
    assert load_last("anthropic", f) == "claude-opus-5-5"
