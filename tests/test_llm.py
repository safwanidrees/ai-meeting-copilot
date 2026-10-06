from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import openai

from ai_meeting_copilot.display import LabelStreamer
from ai_meeting_copilot.llm import (
    FREE_FALLBACKS,
    AnthropicChat,
    OpenAIChat,
    build_messages,
    choose_fallbacks,
    describe_error,
    parse_response,
)
from ai_meeting_copilot.rag import Chunk, RetrievedChunk


def test_build_messages_alternates_and_attaches_context_to_last_turn():
    ctx = [RetrievedChunk(Chunk(0, "resume.md", "Built RAG at Acme."), 0.9)]
    msgs = build_messages([("Hi, introduce yourself.", "ANSWER: I'm Sam.")], "What did you build?", ctx)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "CONTEXT" not in msgs[0]["content"]
    assert "[1] (resume.md)\nBuilt RAG at Acme." in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("THEY SAID: What did you build?")


def test_parse_response():
    assert parse_response("ANSWER: Yes, I can.\n\nFOLLOW-UP: How?") == ("Yes, I can.", "How?")
    assert parse_response("ANSWER: Only an answer.") == ("Only an answer.", "")
    assert parse_response("No labels at all") == ("No labels at all", "")


def run_streamer(tokens):
    s = LabelStreamer()
    out = [p for t in tokens for p in s.feed(t)] + s.finish()
    return out


def test_label_streamer_handles_labels_split_across_tokens():
    out = run_streamer(["ANS", "WER: Yes", " indeed.\n\nFOLL", "OW-UP", ": Why?"])
    assert "".join(p for p, _ in out) == "ANSWER: Yes indeed.\n\nFOLLOW-UP: Why?"
    assert [p for p, label in out if label] == ["ANSWER:", "FOLLOW-UP:"]


def test_label_streamer_releases_false_alarms():
    out = run_streamer(["The ANS", "WERS are F", "OLLOWED"])
    assert "".join(p for p, _ in out) == "The ANSWERS are FOLLOWED"
    assert not [p for p, label in out if label]


class FakeOpenAIClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        deltas = ["ANSWER:", " Hi", None]
        chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))]) for d in deltas]
        chunks.append(SimpleNamespace(choices=[]))  # usage-only chunk

        class Stream(list):
            closed = False

            def close(self):
                Stream.closed = True

        return Stream(chunks)


def test_openai_stream_request_shape():
    chat = OpenAIChat.__new__(OpenAIChat)
    chat._client, chat.model, chat.max_tokens, chat.fallback_models = FakeOpenAIClient(), "gpt-4o", 100, []
    assert "".join(chat.stream("SYS", [{"role": "user", "content": "q"}])) == "ANSWER: Hi"
    call = chat._client.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "SYS"}
    assert call["stream"] is True and call["model"] == "gpt-4o"
    assert "extra_body" not in call  # no OpenRouter-only fields sent to OpenAI


def test_anthropic_stream_request_shape():
    calls = []

    @contextmanager
    def stream(**kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(text_stream=iter(["ANSWER:", " Hello"]))

    chat = AnthropicChat.__new__(AnthropicChat)
    chat._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    chat.model, chat.max_tokens = "claude-sonnet-5-5", 100
    assert "".join(chat.stream("SYS", [{"role": "user", "content": "q"}])) == "ANSWER: Hello"
    assert calls[0]["system"] == "SYS" and calls[0]["messages"][0]["role"] == "user"


# --- OpenRouter fallbacks & error messages -------------------------------------------


def test_free_model_gets_other_free_fallbacks():
    available = set(FREE_FALLBACKS)
    out = choose_fallbacks("qwen/qwen3.8-27b:free", [], available)
    assert len(out) == 2 and "qwen/qwen3.8-27b:free" not in out
    assert all(m.endswith(":free") for m in out)


def test_fallbacks_skip_models_openrouter_no_longer_serves():
    assert choose_fallbacks("x/y:free", [], {"google/gemma-4-26b-a4b-it:free"}) == ["google/gemma-4-26b-a4b-it:free"]


def test_paid_model_gets_no_automatic_fallbacks_but_honours_user_list():
    assert choose_fallbacks("anthropic/claude-sonnet-5.5", [], None) == []
    assert choose_fallbacks("anthropic/claude-sonnet-5.5", ["openai/gpt-5.6-luna"], None) == ["openai/gpt-5.6-luna"]


def test_openrouter_request_carries_models_array():
    chat = OpenAIChat.__new__(OpenAIChat)
    chat._client, chat.model, chat.max_tokens = FakeOpenAIClient(), "a/b:free", 100
    chat.fallback_models = ["c/d:free"]
    "".join(chat.stream("SYS", [{"role": "user", "content": "q"}]))
    assert chat._client.calls[0]["extra_body"] == {"models": ["a/b:free", "c/d:free"]}


def _status_error(cls, status, body):
    response = httpx.Response(status, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"), json=body)
    return cls("Error code", response=response, body=body)


def test_describe_rate_limit_error_from_openrouter():
    body = {"message": "Provider returned error", "code": 429, "metadata": {
        "raw": "qwen/qwen3.8-27b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key",
        "provider_name": "ModelRun"}}
    msg = describe_error(_status_error(openai.RateLimitError, 429, body))
    assert msg.startswith("Rate-limited (qwen/qwen3.8-27b:free is temporarily rate-limited upstream)")
    assert "{" not in msg  # no JSON dump


def test_describe_other_errors():
    assert "401" in describe_error(_status_error(openai.AuthenticationError, 401, {"message": "bad key"}))
    assert describe_error(ValueError("boom")) == "ValueError: boom"
