#!/usr/bin/env python3
"""
NeuralAI — RAG Module
Embedding + retrieval for document Q&A.
"""
import os, hashlib, json
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from sentence_transformers import SentenceTransformer
import pypdf, docx

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Embedding model
_embed_model = None
_chroma = None

def get_embedder():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def get_chroma():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma

# ── Text Extraction ─────────────────────────────────────────────
def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    text = ""

    if ext == ".pdf":
        try:
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n\n"
        except Exception as e:
            return f"[PDF error: {e}]"

    elif ext in (".docx", ".doc"):
        try:
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
        except Exception as e:
            return f"[DOCX error: {e}]"

    elif ext == ".txt":
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()

    elif ext == ".md":
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()

    else:
        return f"[Unsupported file type: {ext}]"

    return text.strip()

# ── Chunking ──────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ── Index Document ─────────────────────────────────────────────
def index_document(filepath: str, collection_name: str = "documents") -> dict:
    filename = os.path.basename(filepath)
    file_id = hashlib.sha256(filename.encode()).hexdigest()[:16]

    # Extract & chunk
    text = extract_text(filepath)
    if not text:
        return {"chunks": 0, "error": "No text extracted"}

    chunks = chunk_text(text)
    if not chunks:
        return {"chunks": 0, "error": "No chunks generated"}

    # Embed
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    # Store in Chroma
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_idx": i} for i in range(len(chunks))]

    chroma = get_chroma()
    try:
        col = chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction()
        )
    except Exception:
        col = chroma.get_or_create_collection(name=collection_name)

    col.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    return {
        "filename": filename,
        "file_id": file_id,
        "chunks": len(chunks),
        "chars": len(text)
    }

# ── Query ──────────────────────────────────────────────────────
def query_documents(query: str, collection_name: str = "documents", top_k: int = 4) -> list[dict]:
    embedder = get_embedder()
    chroma = get_chroma()

    try:
        col = chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction()
        )
    except Exception:
        return []

    query_emb = embedder.encode([query], show_progress_bar=False).tolist()
    results = col.query(query_embeddings=query_emb, n_results=top_k)

    docs = []
    if results and results.get("documents"):
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            docs.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "chunk": meta.get("chunk_idx", 0) + 1
            })
    return docs
