"""
retriever.py
------------
Loads the saved index and answers: given a query, what are the top-k
most relevant chunks (with similarity scores + provenance metadata)?

Also applies a minimum-similarity threshold: if nothing clears the bar,
we return an empty list so the RAG layer can say "I don't know" instead
of grounding an answer in a weakly-related chunk.
"""

from __future__ import annotations
import os
from typing import List, Dict

from vector_store import SimpleVectorStore

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "..", "data", "index")

MIN_SIMILARITY = 0.095  # tuned via eval/run_eval.py; see eval/README.md "Findings"
# for the full history of how this number was derived (started at 0.08,
# revised to 0.09, then to 0.095 after adding stemming + domain-filler
# stopwords in embeddings.py closed most of the gap between the lowest
# legitimate correctness score (0.099) and the highest off-topic false
# positive (0.089). Re-run `python eval/run_eval.py` after any change to
# the knowledge base or embedder to confirm this margin still holds.


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
        "What's the capital of France?",  # should retrieve nothing relevant
    ]
    for q in test_queries:
        print("=" * 70)
        print("QUERY:", q)
        hits = r.retrieve(q)
        if not hits:
            print("  (no chunks above similarity threshold)")
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['title']} ({h['doc_id']}): {h['text'][:120]}...")
