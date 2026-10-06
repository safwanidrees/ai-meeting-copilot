"""Keyword search with BM25.

Embeddings are good at meaning ("compensation" ~ "salary") but weak at exact
tokens: product names, error codes, acronyms, people's names. BM25 is the
opposite, which is why hybrid search uses both.

Implemented here (≈30 lines) rather than via `rank_bm25`, because the classic Okapi
IDF, log((N - n + 0.5) / (n + 0.5)), goes NEGATIVE when a term appears in more than
half the chunks. With a small context set (one resume = a handful of chunks) that
turns exact matches into negative scores. We use Lucene's IDF, which is always > 0:

    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score(d, q) = Σ_t  idf(t) · tf·(k1 + 1) / (tf + k1·(1 - b + b·|d| / avgdl))
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .documents import Chunk

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your yours yourself yourselves
    tell explain describe please also""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.][a-z0-9]+)*")  # keeps dotted terms like "node.js" or "3.12" whole


def _singular(token: str) -> str:
    """Crude plural folding ("strengths" -> "strength", "queries" -> "query").

    Applied identically to documents and queries, so an odd result such as
    "bias" -> "bia" still matches itself; it only needs to be consistent.
    """
    if not token.isalpha():  # leave "node.js", "gpt4s", "3.12" alone
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_singular(t) for t in tokens]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1  # how quickly repeated terms stop adding score
        self.b = b  # how much long chunks are penalised
        # Index every word (stopwords included) so all-stopword questions can still match.
        # Normal queries never contain stopwords, so they only nudge document lengths.
        self._tf = [Counter(tokenize(c.text, keep_stopwords=True)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avgdl = (sum(self._len) / len(chunks)) if chunks else 0.0
        df = Counter(term for tf in self._tf for term in tf)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - df_t + 0.5) / (df_t + 0.5)) for t, df_t in df.items()}

    def score(self, query_terms: list[str], i: int) -> float:
        tf, length = self._tf[i], self._len[i]
        norm = self.k1 * (1 - self.b + self.b * length / self._avgdl) if self._avgdl else self.k1
        total = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f:
                total += self._idf[term] * f * (self.k1 + 1) / (f + norm)
        return total

    def search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """Return (chunk_id, score) for chunks sharing at least one query term, best first."""
        # "Tell me about yourself" is made only of stopwords; rather than return nothing,
        # search with them, and IDF will still favour the rarer ones.
        terms = tokenize(query) or tokenize(query, keep_stopwords=True)
        if not terms:
            return []
        scored = [(i, self.score(terms, i)) for i in range(len(self.chunks))]
        hits = [(i, s) for i, s in scored if s > 0]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:limit]
