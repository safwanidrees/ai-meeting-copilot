import pytest

from ai_meeting_copilot.config import Settings
from ai_meeting_copilot.llm import create_llm
from ai_meeting_copilot.rag.embeddings import OpenAIEmbedder

KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY", "CO_API_KEY", "OPENROUTER_BASE_URL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def test_openrouter_key_alone_selects_openrouter_for_llm_and_embeddings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    s = Settings()
    assert s.resolve_provider() == "openrouter"
    assert s.resolve_model("openrouter") == "anthropic/claude-sonnet-5.5"
    assert s.resolve_embedding_provider() == "openrouter"


def test_direct_keys_take_priority(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert Settings().resolve_provider() == "anthropic"
    assert Settings(provider="openrouter").resolve_provider() == "openrouter"


def test_no_keys():
    assert Settings().resolve_provider() is None
    assert Settings().resolve_embedding_provider() is None


def test_openrouter_client_points_at_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = create_llm("openrouter", "openai/gpt-4o", 100)
    assert str(llm._client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert llm._client.api_key == "sk-or-test"
    assert llm.name == "OpenRouter openai/gpt-4o"


def test_embedder_label_changes_cache_key():
    a = OpenAIEmbedder(api_key="x")
    b = OpenAIEmbedder("openai/text-embedding-3-small", base_url="https://openrouter.ai/api/v1", api_key="x", label="openrouter")
    assert a.name != b.name
