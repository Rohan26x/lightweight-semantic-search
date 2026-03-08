# scripts/setup_project.py
"""
Phase 0 - Project Scaffold Generator
Generates the full directory structure and placeholder files.
Run once from the repo root: python scripts/setup_project.py
"""

import os

# ── Directory tree ──────────────────────────────────────────────────────────
DIRS = [
    "data/raw",
    "data/processed",
    "vectordb/chroma_store",
    "models",
    "src/phase1_ingest",
    "src/phase2_clustering",
    "src/phase3_cache",
    "src/phase4_api/routers",
    "notebooks",
    "scripts",
    "tests",
]

# ── Files with starter content ───────────────────────────────────────────────
FILES = {
    # Package inits
    "src/__init__.py": "",
    "src/phase1_ingest/__init__.py": "",
    "src/phase2_clustering/__init__.py": "",
    "src/phase3_cache/__init__.py": "",
    "src/phase4_api/__init__.py": "",
    "src/phase4_api/routers/__init__.py": "",

    # Git / env hygiene
    ".gitignore": "\n".join([
        "__pycache__/", "*.pyc", ".env", "data/raw/", "data/processed/",
        "vectordb/", "models/", "*.npy", "*.jsonl", ".ipynb_checkpoints/",
        "venv/", ".venv/",
    ]),
    ".env.example": "\n".join([
        "# Copy to .env and fill values",
        "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2",
        "CHROMA_PERSIST_DIR=./vectordb/chroma_store",
        "CACHE_SIMILARITY_THRESHOLD=0.85",
        "N_CLUSTERS=20",
        "DEVICE=cuda",  # or cpu
    ]),

    # Central config (populated fully in Phase 1)
    "src/config.py": '''"""
Central configuration — all phases import from here.
Edit .env to override defaults without touching code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# Paths
RAW_DATA_DIR        = ROOT / "data" / "raw" / "20_newsgroups"
PROCESSED_DIR       = ROOT / "data" / "processed"
CLEANED_DOCS_PATH   = PROCESSED_DIR / "cleaned_docs.jsonl"
EMBEDDINGS_PATH     = PROCESSED_DIR / "embeddings.npy"
DOC_IDS_PATH        = PROCESSED_DIR / "doc_ids.json"
CHROMA_PERSIST_DIR  = ROOT / "vectordb" / "chroma_store"
MODELS_DIR          = ROOT / "models"
CLUSTER_MEMBERSHIPS = MODELS_DIR / "cluster_memberships.npy"
CLUSTER_METADATA    = MODELS_DIR / "cluster_metadata.json"

# Model
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE              = os.getenv("DEVICE", "cuda")   # falls back to cuda if available

# Clustering
N_CLUSTERS          = int(os.getenv("N_CLUSTERS", 20))

# Cache
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.85))
''',

    # requirements.txt
    "requirements.txt": "\n".join([
        "# Core ML & embeddings",
        "torch>=2.2.0",
        "sentence-transformers>=2.7.0",
        "transformers>=4.40.0",
        "",
        "# Vector DB",
        "chromadb>=0.5.0",
        "",
        "# Clustering",
        "scikit-learn>=1.4.0",
        "scikit-fuzzy>=0.4.2",
        "umap-learn>=0.5.6",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "",
        "# API",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.7.0",
        "",
        "# Utilities",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pandas>=2.2.0",
        "notebook>=7.0.0",
        "",
        "# Testing",
        "pytest>=8.0.0",
        "httpx>=0.27.0",
    ]),

    # README stub
    "README.md": "# Newsgroups Semantic Search\n\n> Generated scaffold — fill in details after each phase.\n",

    # Dockerfile stub (Bonus)
    "Dockerfile": "\n".join([
        "FROM python:3.11-slim",
        "WORKDIR /app",
        "COPY requirements.txt .",
        "RUN pip install --no-cache-dir -r requirements.txt",
        "COPY . .",
        'CMD ["uvicorn", "src.phase4_api.main:app", "--host", "0.0.0.0", "--port", "8000"]',
    ]),
}


def scaffold():
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  [DIR]  {d}/")

    for fpath, content in FILES.items():
        # Don't overwrite existing files
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                f.write(content)
            print(f"  [FILE] {fpath}")
        else:
            print(f"  [SKIP] {fpath} already exists")

    print("\n✅ Scaffold complete. Next: Phase 1 — Data Ingestion & Embeddings.")


if __name__ == "__main__":
    scaffold()