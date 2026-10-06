"""Text -> vector embedding providers (OpenAI or Cohere)."""

from __future__ import annotations

from typing import Protocol

from ..config import cohere_api_key


class Embedder(Protocol):
    name: str  # stable identifier, part of the vector-index cache key

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """OpenAI embeddings, or OpenRouter's compatible endpoint via `base_url`."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 128,
        base_url: str | None = None,
        api_key: str | None = None,
        label: str = "openai",
    ):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.name = f"{label}:{model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            resp = self._client.embeddings.create(
                model=self.model,
                input=texts[start : start + self.batch_size],
                encoding_format="float",  # the SDK defaults to base64, which compatible APIs may not support
            )
            vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class CohereEmbedder:
    """Cohere distinguishes documents from queries at embed time (asymmetric search)."""

    def __init__(self, model: str = "embed-v4.0", batch_size: int = 96):
        import cohere

        self._client = cohere.ClientV2(api_key=cohere_api_key())
        self.model = model
        self.batch_size = batch_size  # Cohere's per-request limit
        self.name = f"cohere:{model}"

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            resp = self._client.embed(
                model=self.model,
                input_type=input_type,
                texts=texts[start : start + self.batch_size],
                embedding_types=["float"],
            )
            vectors.extend(resp.embeddings.float_)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "search_document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "search_query")[0]
