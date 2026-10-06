import threading

import pytest

from ai_meeting_copilot.rag import Chunk, HybridRetriever, chunk_text, load_documents, reciprocal_rank_fusion
from ai_meeting_copilot.rag.bm25 import BM25Index, tokenize
from ai_meeting_copilot.rag.vector_store import QdrantStore, collection_name, connect


def test_chunk_text_respects_size_and_overlaps():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(60)]
    chunks = chunk_text(" ".join(sentences), chunk_size=200, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        last_sentence = a.split(". ")[-1]
        assert last_sentence.rstrip(".") in b  # tail of previous chunk carried over


def test_chunk_text_hard_splits_giant_tokens():
    chunks = chunk_text("x" * 1000, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks) and "".join(chunks).count("x") >= 1000


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hi", chunk_size=100, overlap=100)


def test_load_documents_reads_folders(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes\n\nI led the migration to Kubernetes.")
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")  # unsupported types in folders are skipped
    docs = load_documents([tmp_path])
    assert [d.source for d in docs] == ["notes.md"]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the Node.js event loop?") == ["node.js", "event", "loop"]


CHUNKS = [
    Chunk(0, "resume.md", "Built a RAG pipeline with Qdrant and BM25 at Acme, cutting support tickets by 30%."),
    Chunk(1, "resume.md", "Led a team of four engineers migrating services to Kubernetes."),
    Chunk(2, "notes.md", "Salary expectations: open to discuss, market rate for senior engineers."),
    Chunk(3, "notes.md", "Hobbies include hiking and chess."),
]


def test_bm25_finds_exact_terms():
    hits = BM25Index(CHUNKS).search("Tell me about Kubernetes")
    assert hits[0][0] == 1


def test_plurals_match_singulars():
    chunks = [Chunk(0, "qa.md", "My greatest strength is verifying."), Chunk(1, "n.md", "Gradient descent.")]
    assert BM25Index(chunks).search("What are your strengths?")[0][0] == 0
    assert tokenize("queries embeddings class bias") == ["query", "embedding", "class", "bia"]


def test_all_stopword_question_still_finds_its_answer():
    chunks = [
        Chunk(0, "qa.md", "Tell me about yourself. I am an AI engineer who builds real-time systems."),
        Chunk(1, "qa.md", "What is your greatest strength? I verify instead of assuming."),
        Chunk(2, "notes.md", "Gradient descent minimises a loss function."),
    ]
    hits = BM25Index(chunks).search("Tell me about yourself.")
    assert hits[0][0] == 0


def test_bm25_matches_in_tiny_corpus():
    # Regression: Okapi IDF is negative when a term is in >half the docs (e.g. one resume chunk).
    one = [Chunk(0, "resume.md", "Built a retrieval augmented generation assistant.")]
    hits = BM25Index(one).search("How would you use retrieval augmented generation?")
    assert hits and hits[0][0] == 0 and hits[0][1] > 0


def test_bm25_prefers_rarer_and_repeated_terms():
    chunks = [Chunk(0, "a", "python python python data"), Chunk(1, "b", "python data"), Chunk(2, "c", "rust data")]
    hits = BM25Index(chunks).search("python")
    assert [i for i, _ in hits] == [0, 1]  # chunk 2 has no match; more mentions rank higher


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert [i for i, _ in fused][:2] == [1, 3]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


class FakeEmbedder:
    """Bag-of-keywords vectors: enough to make semantic search testable offline."""

    name = "fake:v1"
    VOCAB = ["rag", "qdrant", "kubernetes", "team", "salary", "pay", "compensation", "hiking"]
    SYNONYMS = {"pay": "salary", "compensation": "salary"}

    def _vec(self, text):
        words = [self.SYNONYMS.get(w, w) for w in tokenize(text)]
        v = [float(words.count(term)) for term in self.VOCAB]
        return v if any(v) else [1e-3] * len(self.VOCAB)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def make_store(tmp_path=None):
    client = connect(path=tmp_path)
    store = QdrantStore(client, collection_name(CHUNKS, FakeEmbedder.name))
    reused = store.index(CHUNKS, FakeEmbedder())
    return store, reused


def test_qdrant_index_is_cached_on_disk(tmp_path):
    store, reused = make_store(tmp_path)
    assert not reused
    store.client.close()
    store2, reused2 = make_store(tmp_path)
    assert reused2


def test_hybrid_semantic_match_beats_missing_keyword():
    store, _ = make_store()
    retriever = HybridRetriever(CHUNKS, BM25Index(CHUNKS), store, FakeEmbedder(), top_k=2)
    # "pay" never appears in the docs; only the vector side can find the salary chunk.
    results = retriever.retrieve("What compensation are you expecting?")
    assert results[0].chunk.id == 2


def test_query_from_another_thread_works():
    store, _ = make_store()
    retriever = HybridRetriever(CHUNKS, BM25Index(CHUNKS), store, FakeEmbedder(), top_k=1)
    out = []
    t = threading.Thread(target=lambda: out.append(retriever.retrieve("qdrant rag")))
    t.start(); t.join()
    assert out[0][0].chunk.id == 0


class ReverseReranker:
    def rerank(self, query, documents, top_n):
        return [(i, 1.0 - i * 0.1) for i in reversed(range(len(documents)))][:top_n]


class BrokenReranker:
    def rerank(self, query, documents, top_n):
        raise ConnectionError("network down")


class BrokenEmbedder(FakeEmbedder):
    def embed_query(self, text):
        raise TimeoutError("embedding API timeout")


def test_reranker_reorders_candidates():
    retriever = HybridRetriever(CHUNKS, BM25Index(CHUNKS), reranker=ReverseReranker(), top_k=4)
    plain = HybridRetriever(CHUNKS, BM25Index(CHUNKS), top_k=4)
    q = "rag qdrant kubernetes team engineers"
    assert [r.chunk.id for r in retriever.retrieve(q)] == list(reversed([r.chunk.id for r in plain.retrieve(q)]))


def test_failures_degrade_instead_of_crashing():
    store, _ = make_store()
    warnings = []
    retriever = HybridRetriever(
        CHUNKS, BM25Index(CHUNKS), store, BrokenEmbedder(), BrokenReranker(), top_k=1, on_warning=warnings.append
    )
    results = retriever.retrieve("Kubernetes migration")
    assert results[0].chunk.id == 1  # BM25 still answered
    assert len(warnings) == 2
