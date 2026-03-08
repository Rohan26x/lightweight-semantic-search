# scripts/run_phase1.py
"""
Phase 1 runner — execute end-to-end:
  1. Load raw documents from data/raw/20_newsgroups/
  2. Clean and filter corpus
  3. Save cleaned corpus to data/processed/cleaned_docs.jsonl
  4. Embed on GPU → save embeddings.npy + doc_ids.json
  5. Store in ChromaDB

Run from repo root:
    python scripts/run_phase1.py
"""

import json
import time
from pathlib import Path

from src.phase1_ingest.loader  import iter_raw_docs
from src.phase1_ingest.cleaner import clean_corpus
from src.phase1_ingest.embedder import (
    load_model, embed_corpus, save_embeddings, store_in_chromadb
)
from src.config import CLEANED_DOCS_PATH, PROCESSED_DIR


def main():
    t0 = time.time()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n── Step 1: Loading raw documents ──")
    raw_docs = list(iter_raw_docs())

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    print("\n── Step 2: Cleaning corpus ──")
    cleaned_docs = clean_corpus(raw_docs)

    # ── 3. Save cleaned corpus ───────────────────────────────────────────────
    print(f"\n── Step 3: Saving cleaned corpus → {CLEANED_DOCS_PATH} ──")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLEANED_DOCS_PATH, "w", encoding="utf-8") as f:
        for doc in cleaned_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[Runner] Saved {len(cleaned_docs):,} cleaned docs.")

    # ── 4. Embed on GPU ───────────────────────────────────────────────────────
    print("\n── Step 4: Embedding on GPU ──")
    model = load_model()
    embeddings, doc_ids = embed_corpus(cleaned_docs, model)
    save_embeddings(embeddings, doc_ids)

    # ── 5. Store in ChromaDB ──────────────────────────────────────────────────
    print("\n── Step 5: Storing in ChromaDB ──")
    store_in_chromadb(cleaned_docs, embeddings)

    elapsed = time.time() - t0
    print(f"\n✅ Phase 1 complete in {elapsed/60:.1f} minutes.")
    print(f"   Corpus size : {len(cleaned_docs):,} documents")
    print(f"   Embeddings  : {embeddings.shape}")


if __name__ == "__main__":
    main()