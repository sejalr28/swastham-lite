"""
llm.py
------
Pluggable LLM backend for answer generation.

- StubLLM: no API key / no network required. Produces a grounded answer
  by extracting and lightly stitching the retrieved chunk text. This is
  what runs by default in this offline sandbox and in unit tests, so the
  whole pipeline is verifiable without any external dependency.
- ClaudeLLM: wraps the real Anthropic API for actual deployment. Requires
  `pip install anthropic` and an ANTHROPIC_API_KEY environment variable.

rag.py only depends on the .generate(query, context_chunks) interface,
so switching backends is a one-line change (see get_default_llm()).
"""

from __future__ import annotations
import os
from typing import List, Dict


SYSTEM_PROMPT = """You are a general wellness information assistant for a health platform.
Rules you must follow strictly:
1. Answer ONLY using the provided context chunks. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say you don't have
   enough information rather than guessing.
3. Never diagnose a condition or recommend medication/supplements/dosages.
4. Keep the tone calm, factual, and non-alarmist.
5. Cite which source(s) your answer draws on using the provided doc_id values.
"""


class StubLLM:
    """Deterministic, no-network 'LLM' for local development and testing."""

    name = "stub-extractive"

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return ("I don't have enough information in my knowledge base to answer "
                    "that confidently. This assistant only covers general sleep "
                    "hygiene topics.")
        lines = [f"Based on what I have on this topic:"]
        for c in context_chunks:
            lines.append(f"- {c['text']} (source: {c['doc_id']} - {c['title']})")
        return "\n".join(lines)


class ClaudeLLM:
    """Real LLM backend using the Anthropic API. Needs network + API key."""

    name = "claude-sonnet-4-6"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return ("I don't have enough information in my knowledge base to answer "
                    "that confidently. This assistant only covers general sleep "
                    "hygiene topics.")

        context_block = "\n\n".join(
            f"[{c['doc_id']}] {c['title']}\n{c['text']}" for c in context_chunks
        )
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Answer using only the context above, and cite doc_ids used."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )


def get_default_llm():
    backend = os.environ.get("LLM_BACKEND", "stub")
    if backend == "claude":
        return ClaudeLLM()
    return StubLLM()
