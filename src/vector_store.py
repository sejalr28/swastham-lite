"""
vector_store.py
----------------
A minimal, dependency-light vector store: numpy array of embeddings +
a parallel list of chunk metadata, persisted to disk as .npy + .json.

This intentionally mirrors what a real vector DB (Chroma/Qdrant/pgvector)
gives you -- add(), search(by cosine similarity) -- so swapping in a real
vector DB later only touches this file, not retriever.py or rag.py.
"""

from __future__ import annotations
import json
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple

from chunking import Chunk


class SimpleVectorStore:
    def __init__(self):
        self.embeddings: np.ndarray | None = None
        self.chunks: List[Dict] = []

    def build(self, chunks: List[Chunk], embedder) -> None:
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        self.embeddings = embedder.embed(texts)
        self.chunks = [c.to_dict() for c in chunks]

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> List[Tuple[Dict, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        sims = self.embeddings @ query_vec  # cosine similarity (vectors are L2-normalized)
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self, index_dir: str) -> None:
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def save_embedder(self, index_dir: str, embedder) -> None:
        # TF-IDF vectorizer needs to be persisted too, so queries at
        # serve-time use the exact same vocabulary as index-time.
        with open(os.path.join(index_dir, "embedder.pkl"), "wb") as f:
            pickle.dump(embedder, f)

    def load_embedder(self, index_dir: str):
        with open(os.path.join(index_dir, "embedder.pkl"), "rb") as f:
            return pickle.load(f)
