

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retriever import Retriever  
RELEVANCE_CASES = [
    ("How does caffeine affect my sleep?", "sh-006"),
    ("What temperature is best for my bedroom?", "sh-004"),
    ("How much should I exercise before bed?", "sh-007"),
    ("How long should a nap be?", "sh-008"),
    ("I keep waking up at a different time on weekends", "sh-003"),
    ("What are the stages of sleep?", "sh-002"),
    ("When should I see a doctor about my sleep?", "sh-010"),
]

OFF_TOPIC_QUERIES = [
    "What's the capital of France?",
    "Recommend me a good stock to buy",
    "How do I fix a Python import error?",
]


def test_relevant_doc_in_top3():
    r = Retriever()
    failures = []
    for query, expected_doc in RELEVANCE_CASES:
        hits = r.retrieve(query, top_k=3, min_similarity=0.0)  
        doc_ids = [h["doc_id"] for h in hits]
        if not any(expected_doc in d for d in doc_ids):
            failures.append((query, expected_doc, doc_ids))
    assert not failures, f"Relevant doc missing from top-3: {failures}"


def test_off_topic_returns_nothing():
    r = Retriever()
    failures = []
    for query in OFF_TOPIC_QUERIES:
        hits = r.retrieve(query) 
        if hits:
            failures.append((query, [h["doc_id"] for h in hits]))
    assert not failures, f"Off-topic query unexpectedly retrieved chunks: {failures}"


def _run_manually():
    passed, failed = 0, 0
    for fn in [test_relevant_doc_in_top3, test_off_topic_returns_nothing]:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {fn.__name__} -> {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")


if __name__ == "__main__":
    _run_manually()
