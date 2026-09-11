"""
Lightweight retriever over the (customer_text -> brand_text) grounding
corpus.

Uses TF-IDF + cosine similarity rather than a neural embedding model on
purpose: it's fully offline (no model download / embedding API cost),
fast enough for a few thousand pairs, and good enough to find lexically
similar historical complaints ("app crashes on login" ~ "app keeps
crashing when I sign in"). Swapping in a sentence-embedding model later
is a one-file change (see decision log) if recall turns out to matter
more than we assumed.
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data_loader import ResolvedPair


@dataclass
class GroundingHit:
    pair: ResolvedPair
    similarity: float


class GroundingRetriever:
    def __init__(self, pairs: list[ResolvedPair]):
        self.pairs = pairs
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, max_features=20000
        )
        corpus = [p.customer_text for p in pairs] or [""]
        # Normalize once; repeated cosine_similarity calls otherwise
        # renormalize the entire corpus for every incoming message.
        self._matrix = normalize(self._vectorizer.fit_transform(corpus), norm="l2")

    def query(self, text: str, top_k: int = 3) -> list[GroundingHit]:
        if not self.pairs:
            return []
        q_vec = normalize(self._vectorizer.transform([text]), norm="l2")
        sims = (q_vec @ self._matrix.T).toarray()[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [GroundingHit(pair=self.pairs[i], similarity=float(sims[i])) for i in ranked_idx]
