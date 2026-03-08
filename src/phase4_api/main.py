# src/phase4_api/main.py
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.phase4_api.routers.query import router as query_router
from src.phase4_api.routers.cache import router as cache_router
from src.phase4_api.schemas import HealthResponse
from src.phase3_cache.query_engine import QueryEngine
from src.config import CACHE_SIMILARITY_THRESHOLD, DEVICE


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Loading QueryEngine...")
    threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", CACHE_SIMILARITY_THRESHOLD))
    app.state.engine = QueryEngine(cache_threshold=threshold)
    print("[Startup] ✅ Service ready.")
    yield
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title       = "Newsgroups Semantic Search",
    description = (
        "Lightweight semantic search over the 20 Newsgroups corpus. "
        "Features GPU embeddings, fuzzy cluster-scoped caching, and ChromaDB retrieval."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.include_router(query_router, tags=["Search"])
app.include_router(cache_router, tags=["Cache"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    engine = request.app.state.engine
    return HealthResponse(
        status           = "ok",
        chroma_docs      = engine.chroma_collection.count(),
        cache_entries    = engine.cache.total_entries,
        embedding_device = DEVICE,
        n_clusters       = engine.n_clusters,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )