
import os
from chunking import load_and_chunk
from embeddings import get_default_embedder
from vector_store import SimpleVectorStore

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(HERE, "..", "data", "knowledge")
INDEX_DIR = os.path.join(HERE, "..", "data", "index")


def main():
    print(f"Loading & chunking docs from {KNOWLEDGE_DIR} ...")
    chunks = load_and_chunk(KNOWLEDGE_DIR)
    print(f"  -> {len(chunks)} chunks from "
          f"{len(set(c.doc_id for c in chunks))} documents")

    embedder = get_default_embedder()
    print(f"Embedding with backend: {embedder.name}")

    store = SimpleVectorStore()
    store.build(chunks, embedder)

    store.save(INDEX_DIR)
    store.save_embedder(INDEX_DIR, embedder)
    print(f"Saved index to {INDEX_DIR}")
    print(f"  embeddings shape: {store.embeddings.shape}")


if __name__ == "__main__":
    main()
