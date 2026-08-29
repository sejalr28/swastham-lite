

from __future__ import annotations
import re
import numpy as np
from typing import List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


_EXTRA_STOPWORDS = {"good", "general", "generally", "common", "commonly"}


def _simple_stem(word: str) -> str:
    """
    Minimal, dependency-free suffix stripper (no nltk/network needed).
    Not a real linguistic stemmer -- just enough to collapse common
    plural/verb-form mismatches (screens -> screen, worries -> worri,
    stopped -> stop) that were causing retrieval misses. See
    eval/README.md for the specific case that motivated this.
    """
    w = word.lower()
    if len(w) <= 4:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 6:
        return w[:-3]
    if w.endswith("ed") and len(w) > 5:
        return w[:-2]
    if w.endswith("es") and len(w) > 5:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


_EXTRA_STOPWORDS_STEMMED = {_simple_stem(w) for w in _EXTRA_STOPWORDS}


def _stemmed_tokenizer(text: str) -> List[str]:
    
    words = [w.lower() for w in _WORD_RE.findall(text)]
    
    words = [w for w in words if len(w) >= 2]
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    stemmed = [_simple_stem(w) for w in words]
    stemmed = [w for w in stemmed if w not in _EXTRA_STOPWORDS_STEMMED]
    return stemmed



class TfidfEmbedder:
    """Local, no-network embedding backend using TF-IDF + L2 normalization.

    Uses a lightweight custom tokenizer with basic suffix-stripping
    (see _simple_stem below) so that plural/verb-form mismatches like
    "screens" (in the knowledge base) vs. "screen" (in a user's query)
    still match. This was added after the eval harness (eval/run_eval.py)
    found a retrieval miss caused by exactly this mismatch -- see
    eval/README.md "Findings" for the full writeup.
    """

    name = "tfidf-stemmed"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000,
            tokenizer=_stemmed_tokenizer,
            token_pattern=None,  # required by sklearn when a custom tokenizer is passed
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
