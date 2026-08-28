"""
rag.py
------
The orchestration layer: safety check -> retrieve -> generate -> package
a response with citations and metadata about how it was produced.

This is what app.py (the FastAPI layer, week 2) will call directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

from retriever import Retriever
from llm import get_default_llm
import safety

DIAGNOSIS_REMINDER = (
    "\n\n(Note: this is general information, not a diagnosis. If you're concerned "
    "about a specific symptom or condition, please consult a healthcare provider.)"
)


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]
    flagged_crisis: bool = False
    flagged_diagnosis_seeking: bool = False


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = get_default_llm()

    def answer(self, query: str, top_k: int = 4) -> RAGResponse:
        safety_result = safety.check(query)

        if safety_result.is_crisis:
            return RAGResponse(
                answer=safety_result.fixed_response,
                sources=[],
                flagged_crisis=True,
            )

        chunks = self.retriever.retrieve(query, top_k=top_k)
        answer_text = self.llm.generate(query, chunks)

        if safety_result.is_diagnosis_seeking and chunks:
            answer_text += DIAGNOSIS_REMINDER

        sources = [
            {"doc_id": c["doc_id"], "title": c["title"], "score": round(c["score"], 3)}
            for c in chunks
        ]

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            flagged_diagnosis_seeking=safety_result.is_diagnosis_seeking,
        )


if __name__ == "__main__":
    pipeline = RAGPipeline()
    demo_queries = [
        "How does caffeine before bed affect my sleep?",
        "Do I have insomnia if I can't fall asleep for an hour?",
        "I've been having thoughts of ending my life and can't sleep",
        "What's the best stock to invest in?",
    ]
    for q in demo_queries:
        print("=" * 70)
        print("Q:", q)
        resp = pipeline.answer(q)
        print("A:", resp.answer)
        print("Sources:", resp.sources)
        print("Flags: crisis=%s diagnosis_seeking=%s" % (resp.flagged_crisis, resp.flagged_diagnosis_seeking))
