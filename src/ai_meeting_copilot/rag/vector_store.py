"""Semantic search over chunk embeddings, stored in Qdrant.

Runs embedded (on-disk, no server) by default; set QDRANT_URL to use a Qdrant
server or Qdrant Cloud instead. Collections are named after a hash of the content
and embedding model, so re-running with the same documents skips re-embedding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .documents import Chunk
from .embeddings import Embedder


def connect(url: str | None = None, api_key: str | None = None, path: Path | None = None) -> QdrantClient:
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if path is None:
        return QdrantClient(":memory:")
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def collection_name(chunks: list[Chunk], embedder_name: str) -> str:
    digest = hashlib.sha256(embedder_name.encode())
    for chunk in chunks:
        digest.update(b"\0" + chunk.source.encode() + b"\0" + chunk.text.encode())
    return f"copilot_{digest.hexdigest()[:16]}"


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    def index(self, chunks: list[Chunk], embedder: Embedder, batch_size: int = 256) -> bool:
        """Embed and upload chunks. Returns True if an existing identical index was reused."""
        if self.client.collection_exists(self.collection):
            if self.client.count(self.collection, exact=True).count == len(chunks):
                return True
            self.client.delete_collection(self.collection)  # partial upload from a crashed run

        vectors = embedder.embed_documents([c.text for c in chunks])
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=c.id, vector=v, payload={"source": c.source, "text": c.text})
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points=points[start : start + batch_size])
        return False

    def search(self, vector: list[float], limit: int = 20) -> list[tuple[int, float]]:
        """Return (chunk_id, cosine similarity), best first."""
        result = self.client.query_points(self.collection, query=vector, limit=limit, with_payload=False)
        return [(int(p.id), float(p.score)) for p in result.points]
