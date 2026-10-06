"""Hybrid retrieval: BM25 + vectors, fused with RRF, optionally reranked by Cohere."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..config import Settings, cohere_api_key, openrouter_api_key, openrouter_base_url
from .bm25 import BM25Index
from .documents import Chunk, chunk_documents, load_documents
from .embeddings import CohereEmbedder, Embedder, OpenAIEmbedder
from .vector_store import QdrantStore, collection_name, connect


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]: ...


class CohereReranker:
    """A cross-encoder reads query and passage *together*, so it ranks far more precisely
    than either BM25 or embeddings, but it is too slow to run over every chunk. Hence:
    cheap retrieval for ~20 candidates, then rerank those."""

    def __init__(self, model: str = "rerank-v3.5"):
        import cohere

        self._client = cohere.ClientV2(api_key=cohere_api_key())
        self.model = model

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        resp = self._client.rerank(model=self.model, query=query, documents=documents, top_n=top_n)
        return [(r.index, float(r.relevance_score)) for r in resp.results]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists by summing 1 / (k + rank).

    BM25 scores and cosine similarities live on different scales, so we fuse
    *positions* instead of scores. k=60 is the constant from the original RRF paper;
    it damps the advantage of being #1 versus #2 in a single list.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Index,
        vector_store: QdrantStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        top_k: int = 4,
        candidates: int = 20,
        on_warning: Callable[[str], None] | None = None,
    ):
        self.chunks = chunks
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self.candidates = max(candidates, top_k)
        self._warn = on_warning or (lambda _msg: None)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        rankings = []
        keyword = self.bm25.search(query, self.candidates)
        if keyword:
            rankings.append([i for i, _ in keyword])

        # Network failures mid-meeting must degrade the answer, never kill it.
        if self.vector_store is not None and self.embedder is not None:
            try:
                semantic = self.vector_store.search(self.embedder.embed_query(query), self.candidates)
                rankings.append([i for i, _ in semantic])
            except Exception as exc:
                self._warn(f"vector search failed, using keyword search only ({exc})")

        fused = reciprocal_rank_fusion(rankings)[: self.candidates]
        if not fused:
            return []

        if self.reranker is not None:
            try:
                order = self.reranker.rerank(query, [self.chunks[i].text for i, _ in fused], self.top_k)
                return [RetrievedChunk(self.chunks[fused[pos][0]], score) for pos, score in order]
            except Exception as exc:
                self._warn(f"rerank failed, using fused ranking ({exc})")

        return [RetrievedChunk(self.chunks[i], score) for i, score in fused[: self.top_k]]


def build_retriever(settings: Settings, log: Callable[[str], None], warn: Callable[[str], None]) -> HybridRetriever | None:
    docs = load_documents(settings.context_paths)
    if not docs:
        warn("Context files contained no readable text — running without context.")
        return None
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    files = len({d.source.split("#")[0] for d in docs})
    log(f"Loaded {files} file(s) -> {len(chunks)} chunks")

    bm25 = BM25Index(chunks)
    log("Keyword index: BM25 ready")

    embedder: Embedder | None = None
    store: QdrantStore | None = None
    provider = settings.resolve_embedding_provider()
    if provider is None:
        warn("No OpenAI/OpenRouter/Cohere key for embeddings — vector search disabled, using BM25 only.")
    else:
        try:
            if provider == "openai":
                embedder = OpenAIEmbedder(settings.openai_embedding_model)
            elif provider == "openrouter":
                embedder = OpenAIEmbedder(
                    settings.openrouter_embedding_model,
                    base_url=openrouter_base_url(),
                    api_key=openrouter_api_key(),
                    label="openrouter",
                )
            else:
                embedder = CohereEmbedder(settings.cohere_embedding_model)
            client = connect(settings.qdrant_url, settings.qdrant_api_key, None if settings.qdrant_url else settings.qdrant_path)
            store = QdrantStore(client, collection_name(chunks, embedder.name))
            reused = store.index(chunks, embedder)
            log(f"Vector index: Qdrant + {embedder.name} ({'cached' if reused else 'built'})")
        except Exception as exc:
            warn(f"Vector index unavailable ({exc}) — using BM25 only.")
            embedder, store = None, None

    reranker: Reranker | None = None
    if settings.rerank and cohere_api_key():
        reranker = CohereReranker(settings.rerank_model)
        log(f"Reranker: Cohere {settings.rerank_model}")
    elif settings.rerank:
        log("Reranker: off (set COHERE_API_KEY to enable)")

    return HybridRetriever(
        chunks,
        bm25,
        vector_store=store,
        embedder=embedder,
        reranker=reranker,
        top_k=settings.top_k,
        candidates=settings.retrieval_candidates,
        on_warning=warn,
    )
