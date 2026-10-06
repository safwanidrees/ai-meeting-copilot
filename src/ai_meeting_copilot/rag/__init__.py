"""Hybrid retrieval over your own documents (resume, notes, product docs, ...).

    documents --> chunks --+--> BM25 keyword index ----+
                           |                           +--> Reciprocal Rank Fusion --> Cohere rerank --> top-k
                           +--> embeddings -> Qdrant --+
"""

from .documents import Chunk, Document, chunk_documents, chunk_text, load_documents
from .retriever import HybridRetriever, RetrievedChunk, build_retriever, reciprocal_rank_fusion

__all__ = [
    "Chunk",
    "Document",
    "HybridRetriever",
    "RetrievedChunk",
    "build_retriever",
    "chunk_documents",
    "chunk_text",
    "load_documents",
    "reciprocal_rank_fusion",
]
