"""
retriever.py
------------
Query the ChromaDB vector store for relevant credit risk guidelines.

Uses SentenceTransformer directly for query embedding (query_embeddings instead of
query_texts) to avoid a hanging issue in ChromaDB 1.x's embedding function wrapper.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR   = "chroma_db"
COLLECTION_NAME = "credit_risk_guidelines"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 4

_client     = None
_collection = None
_st_model   = None


def _get_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        _collection = _client.get_collection(name=COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return top-k guideline chunks for the query."""
    try:
        model      = _get_model()
        collection = _get_collection()

        # Embed query ourselves — avoids chromadb 1.x embedding_function hang
        q_embedding = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=q_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {
                "text":     doc,
                "source":   meta.get("source", "unknown"),
                "distance": round(dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
    except Exception as e:
        print(f"[!] Retrieval error: {e}")
        return []


def build_context_string(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    if not chunks:
        return "No relevant guidelines found in the knowledge base."
    return "\n\n".join(
        f"[Guideline {i} — Source: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
