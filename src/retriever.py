

from __future__ import annotations
import os
from typing import List, Dict

from vector_store import SimpleVectorStore

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "..", "data", "index")

MIN_SIMILARITY = 0.095 


class Retriever:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.store = SimpleVectorStore()
        self.store.load(index_dir)
        self.embedder = self.store.load_embedder(index_dir)

    def retrieve(self, query: str, top_k: int = 4, min_similarity: float = MIN_SIMILARITY) -> List[Dict]:
        query_vec = self.embedder.embed([query])[0]
        results = self.store.search(query_vec, top_k=top_k)
        filtered = [
            {**chunk, "score": score}
            for chunk, score in results
            if score >= min_similarity
        ]
        return filtered


if __name__ == "__main__":
    r = Retriever()
    test_queries = [
        "How does caffeine affect sleep?",
        "What temperature should my bedroom be?",
        "What's the capital of France?",  
    ]
    for q in test_queries:
        print("=" * 70)
        print("QUERY:", q)
        hits = r.retrieve(q)
        if not hits:
            print("  (no chunks above similarity threshold)")
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['title']} ({h['doc_id']}): {h['text'][:120]}...")
