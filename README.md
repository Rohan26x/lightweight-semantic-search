# Lightweight Semantic Search — 20 Newsgroups

Semantic search system over ~16k newsgroup documents with GPU embeddings,
fuzzy clustering, a hand-rolled semantic cache, and a FastAPI service.

## Quick Start
```bash
# Install (GPU build required)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .

# Run pipeline once (persists to disk)
python scripts/run_phase1.py   # ~3 min  — embed + ChromaDB
python scripts/run_phase2.py   # ~12 min — cluster + visualize

# Start API
uvicorn src.phase4_api.main:app --host 0.0.0.0 --port 8000
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /query | Semantic search with cache |
| GET | /cache/stats | Hit/miss stats |
| DELETE | /cache | Flush cache |
| GET | /health | Service health |

Interactive docs: http://localhost:8000/docs

## POST /query
```json
// Request
{"query": "What are the best graphics cards for gaming?", "threshold": 0.85}

// Cache miss response
{"query": "...", "cache_hit": false, "matched_query": null,
 "similarity_score": null, "result": "...", "dominant_cluster": 3}

// Cache hit response
{"query": "...", "cache_hit": true, "matched_query": "...",
 "similarity_score": 0.9085, "result": "...", "dominant_cluster": 3}
```

## Design Decisions

**Embedding model:** `all-MiniLM-L6-v2` (384-dim). Fast, strong English
semantic quality, fits in 4GB VRAM at batch=256. allmpnet-base-v2 is
higher quality but 3x slower — overkill for 16k docs.

**Vector DB:** ChromaDB. Native metadata filtering, on-disk persistence,
Python-native. FAISS is faster at 1M+ docs but adds C++ complexity
with no benefit at this scale.

**Clustering:** Fuzzy C-Means, K=10. Hard clustering rejected — documents
genuinely span multiple topics (gun legislation belongs to politics AND
firearms). FCM gives a probability distribution per doc. K=10 chosen
by FPC elbow analysis (FPC=0.52 at K=10, drops sharply after).

**Semantic cache:** Cluster-scoped cosine similarity — only scans entries
in the query's dominant cluster bucket, giving O(N/K) vs O(N) lookup.
No Redis, no libraries. Pure Python dict of lists.

**Threshold theta:** The key tunable. theta=0.85 requires strong semantic
overlap. theta=0.70 catches hardware synonym paraphrases (GPU vs graphics
card). Per-request override supported via body.threshold field.

## Cluster Summary (K=10)

| Cluster | Size | Top Category | Purity |
|---------|------|--------------|--------|
| 0 | 2377 | soc.religion.christian | 35.6% |
| 1 | 2112 | comp.windows.x | 36.2% |
| 2 | 814 | rec.sport.hockey | 95.0% |
| 3 | 3117 | comp.sys.ibm.pc.hardware | 24.6% |
| 4 | 1612 | sci.space | 43.5% |
| 5 | 708 | rec.sport.baseball | 96.9% |
| 6 | 940 | talk.politics.mideast | 80.2% |
| 7 | 1742 | rec.motorcycles | 42.5% |
| 8 | 2117 | talk.politics.guns | 36.7% |
| 9 | 755 | sci.crypt | 85.3% |

## Docker
```bash
docker build -t semantic-search .
docker-compose up
```

## Project Structure
```
src/
  phase1_ingest/    loader, cleaner, embedder
  phase2_clustering fuzzy_cluster, cluster_eval, visualize
  phase3_cache/     semantic_cache, query_engine
  phase4_api/       FastAPI app, routers, schemas
  config.py         all paths and hyperparameters
scripts/
  run_phase1.py     ingestion pipeline
  run_phase2.py     clustering pipeline
  run_phase3_test.py cache behaviour test
```
