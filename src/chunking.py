"""
chunking.py
-----------
Turns raw knowledge markdown files into small, metadata-tagged chunks
suitable for embedding and retrieval.

Design choices (documented for the eval/README):
- Chunk by paragraph first, then merge small paragraphs up to a target
  word count. This keeps chunks topically coherent (better grounding)
  instead of using a naive fixed-character sliding window.
- Every chunk carries provenance metadata (doc_id, title, topic,
  last_reviewed, chunk_index) so answers can cite a specific source
  and we can filter/inspect retrieval quality later.
"""

from __future__ import annotations
import re
import glob
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict


TARGET_WORDS_PER_CHUNK = 120
MIN_WORDS_PER_CHUNK = 40


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    topic: str
    last_reviewed: str
    chunk_index: int
    text: str

    def to_dict(self) -> Dict:
        return asdict(self)


def _parse_frontmatter(raw: str) -> tuple[Dict[str, str], str]:
    """Very small YAML-frontmatter parser (--- key: value --- pairs)."""
    meta: Dict[str, str] = {}
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            fm = raw[3:end].strip().splitlines()
            for line in fm:
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            raw = raw[end + 3:]
    return meta, raw.strip()


def _split_paragraphs(body: str) -> List[str]:
    # Drop markdown headings on their own line, keep paragraph text
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    cleaned = []
    for p in paras:
        if p.startswith("#"):
            continue
        p = re.sub(r"\s+", " ", p)
        cleaned.append(p)
    return cleaned


def _merge_paragraphs(paragraphs: List[str]) -> List[str]:
    """Greedily merge consecutive paragraphs until ~TARGET_WORDS_PER_CHUNK."""
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0

    for p in paragraphs:
        words = len(p.split())
        if buf and buf_words + words > TARGET_WORDS_PER_CHUNK and buf_words >= MIN_WORDS_PER_CHUNK:
            chunks.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(p)
        buf_words += words

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def load_and_chunk(knowledge_dir: str) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    paths = sorted(glob.glob(os.path.join(knowledge_dir, "*.md")))

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, body = _parse_frontmatter(raw)
        paragraphs = _split_paragraphs(body)
        merged = _merge_paragraphs(paragraphs)

        doc_id = meta.get("doc_id", os.path.basename(path))
        title = meta.get("title", os.path.basename(path))
        topic = meta.get("topic", "general")
        last_reviewed = meta.get("last_reviewed", "unknown")

        for i, text in enumerate(merged):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{i}",
                    doc_id=doc_id,
                    title=title,
                    topic=topic,
                    last_reviewed=last_reviewed,
                    chunk_index=i,
                    text=text,
                )
            )
    return all_chunks


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    kd = os.path.join(here, "..", "data", "knowledge")
    chunks = load_and_chunk(kd)
    print(f"Loaded {len(chunks)} chunks from {kd}")
    for c in chunks[:3]:
        print("-" * 60)
        print(c.chunk_id, "|", c.title)
        print(c.text[:200], "...")
