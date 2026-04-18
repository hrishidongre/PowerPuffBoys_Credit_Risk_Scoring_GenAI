"""
ingest.py
---------
Build the ChromaDB vector store from RAG_Docs/ (PDFs and text files).

Run ONCE from the project root:
    python ingest.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = "RAG_Docs"
CHROMA_DB_DIR      = "chroma_db"
COLLECTION_NAME    = "credit_risk_guidelines"
CHUNK_SIZE         = 500
CHUNK_OVERLAP      = 100
EMBED_MODEL        = "all-MiniLM-L6-v2"
# ──────────────────────────────────────────────────────────────────────────────


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 50]


def load_documents(directory: str) -> list[dict]:
    docs = []
    path = Path(directory)

    if not path.exists():
        print(f"[!] Directory '{directory}' not found. Add PDFs or .txt files there.")
        return docs

    for file in sorted(path.iterdir()):
        if file.suffix == ".txt":
            text = file.read_text(encoding="utf-8", errors="ignore")
            docs.append({"source": file.name, "text": text})

        elif file.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                docs.append({"source": file.name, "text": text})
                print(f"  [+] Loaded: {file.name} ({len(reader.pages)} pages)")
            except Exception as e:
                print(f"  [!] Could not read {file.name}: {e}")

    print(f"[✓] Loaded {len(docs)} document(s) from '{directory}'")
    return docs


def build_vector_store(docs: list[dict]) -> None:
    print(f"[i] Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    # Collect all chunks
    all_texts, all_ids, all_metadata = [], [], []
    chunk_id = 0
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            all_texts.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadata.append({"source": doc["source"]})
            chunk_id += 1

    print(f"[i] Embedding {chunk_id} chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=64).tolist()

    # Store in ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[i] Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        documents=all_texts,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )
    print(f"[✓] Ingested {chunk_id} chunks into ChromaDB at '{CHROMA_DB_DIR}'")


if __name__ == "__main__":
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    if docs:
        build_vector_store(docs)
        print("\n[✓] Done. Run: streamlit run agent_app.py")
    else:
        print("[!] No documents found.")
