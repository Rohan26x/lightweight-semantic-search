# src/phase1_ingest/embedder.py
"""
Embedder — GPU-accelerated sentence embeddings + ChromaDB persistence.

Model choice: sentence-transformers/all-MiniLM-L6-v2
  - 384 dimensions: small enough for fast cosine search, rich enough for
    semantic tasks. Benchmarks well on STS and retrieval tasks.
  - Fits comfortably in 4GB VRAM (RTX 3050) even at batch_size=256.
  - Pre-trained on 1B+ sentence pairs — robust for English newsgroup text.

Alternative considered: all-mpnet-base-v2 (768-dim, higher quality but
2x memory and 3x slower — overkill for this corpus size).
"""

import json
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import (
    DEVICE, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_PATH, DOC_IDS_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
)


def load_model() -> SentenceTransformer:
    """Load embedding model onto GPU."""
    print(f"[Embedder] Loading '{EMBEDDING_MODEL}' on {DEVICE.upper()}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    # Confirm model is on GPU
    device_name = next(model.parameters()).device
    print(f"[Embedder] Model loaded on: {device_name}")
    return model


def embed_corpus(
    docs: list[dict],
    model: SentenceTransformer,
) -> tuple[np.ndarray, list[str]]:
    """
    Embed all documents in batches on GPU.

    Args:
        docs  : list of cleaned dicts with 'text' and 'doc_id' fields
        model : loaded SentenceTransformer model

    Returns:
        embeddings : np.ndarray of shape (N, D) float32
        doc_ids    : list of doc_id strings in same order as embeddings
    """
    texts   = [d["text"]   for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    print(f"[Embedder] Embedding {len(texts):,} documents "
          f"(batch={EMBEDDING_BATCH_SIZE}, device={DEVICE.upper()})...")

    # SentenceTransformer handles batching + GPU transfer internally
    # convert_to_numpy=True returns float32 ndarray directly
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
        device=DEVICE,
    )

    print(f"[Embedder] Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings, doc_ids


def save_embeddings(embeddings: np.ndarray, doc_ids: list[str]) -> None:
    """Persist embeddings and doc_id index to disk."""
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(DOC_IDS_PATH, "w") as f:
        json.dump(doc_ids, f)
    print(f"[Embedder] Saved embeddings → {EMBEDDINGS_PATH}")
    print(f"[Embedder] Saved doc_ids    → {DOC_IDS_PATH}")


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load persisted embeddings from disk."""
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(DOC_IDS_PATH) as f:
        doc_ids = json.load(f)
    return embeddings, doc_ids


def store_in_chromadb(docs: list[dict], embeddings: np.ndarray) -> None:
    """
    Persist documents + embeddings in ChromaDB for filtered retrieval.

    ChromaDB is chosen over FAISS here because:
      - Native metadata filtering (filter by category, cluster_id)
      - On-disk persistence out of the box
      - Python-native, no C++ build required
      - Sufficient for ~20k docs (FAISS shines at 1M+)
    """
    import chromadb

    print(f"[ChromaDB] Connecting to {CHROMA_PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Drop and recreate collection for clean re-runs
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity index
    )

    # ChromaDB upserts in batches (max 5461 per call due to SQLite limits)
    CHROMA_BATCH = 2000
    total = len(docs)

    print(f"[ChromaDB] Inserting {total:,} documents in batches of {CHROMA_BATCH}...")
    for start in tqdm(range(0, total, CHROMA_BATCH), desc="ChromaDB insert"):
        end        = min(start + CHROMA_BATCH, total)
        batch_docs = docs[start:end]
        batch_emb  = embeddings[start:end]

        collection.add(
            ids        = [d["doc_id"]   for d in batch_docs],
            embeddings = batch_emb.tolist(),
            documents  = [d["text"]     for d in batch_docs],
            metadatas  = [{"category": d["category"]} for d in batch_docs],
        )

    print(f"[ChromaDB] ✅ Collection '{CHROMA_COLLECTION}' — {collection.count():,} docs stored.")