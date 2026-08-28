"""
embeddings.py
-------------
Pluggable embedding backend.

Why this design:
- The JD/eval environment for this project has no outbound network access,
  so calling a hosted embedding API (OpenAI/Voyage) or downloading a
  sentence-transformers model isn't possible here.
- TfidfEmbedder (scikit-learn) is a fully local, dependency-light backend
  that works today and gives a real, testable retrieval baseline.
- SentenceTransformerEmbedder is provided as a drop-in swap for when this
  runs somewhere with internet/model access (better semantic recall,
  handles paraphrasing / synonyms that TF-IDF misses).

Both implement the same tiny interface: fit(texts), embed(texts) -> np.ndarray
so retriever.py doesn't need to know which one is active.
"""

from __future__ import annotations
import numpy as np
from typing import List


class TfidfEmbedder:
    """Local, no-network embedding backend using TF-IDF + L2 normalization."""

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before embed().")
        mat = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class SentenceTransformerEmbedder:
    """
    Drop-in replacement for TfidfEmbedder using real sentence embeddings.
    Requires: pip install sentence-transformers  (needs network the first
    time, to download the model weights).
    Not used by default in this offline environment, but wired up so
    swapping backends is a one-line change in retriever.py.
    """

    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> None:
        # No fitting needed for a pretrained model; kept for interface parity.
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs)


def get_default_embedder():
    """Central place to switch backends later (e.g. based on an env var)."""
    import os
    backend = os.environ.get("EMBEDDING_BACKEND", "tfidf")
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return TfidfEmbedder()
