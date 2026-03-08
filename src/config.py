# src/config.py
"""
Central configuration — all phases import from here.
All paths, model choices, and hyperparameters live here.
Edit .env to override without touching code.
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DATA_DIR        = ROOT / "data" / "raw" / "20_newsgroups"
PROCESSED_DIR       = ROOT / "data" / "processed"
CLEANED_DOCS_PATH   = PROCESSED_DIR / "cleaned_docs.jsonl"
EMBEDDINGS_PATH     = PROCESSED_DIR / "embeddings.npy"
DOC_IDS_PATH        = PROCESSED_DIR / "doc_ids.json"
CHROMA_PERSIST_DIR  = str(ROOT / "vectordb" / "chroma_store")
MODELS_DIR          = ROOT / "models"
CLUSTER_MEMBERSHIPS = MODELS_DIR / "cluster_memberships.npy"
CLUSTER_METADATA    = MODELS_DIR / "cluster_metadata.json"

# Create dirs if missing
for _d in [PROCESSED_DIR, ROOT / "vectordb" / "chroma_store", MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Device ───────────────────────────────────────────────────────────────────
# Auto-detect GPU; override with DEVICE=cpu in .env if needed
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, fast, excellent semantic quality for English text
# Good fit here: low memory footprint suits laptop GPU (4GB VRAM on RTX 3050)
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM       = 384
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 256))  # tuned for 4GB VRAM

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION   = "newsgroups"

# ── Clustering ───────────────────────────────────────────────────────────────
N_CLUSTERS          = int(os.getenv("N_CLUSTERS", 20))

# ── Cache ────────────────────────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.85))