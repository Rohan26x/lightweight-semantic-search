from fastapi import APIRouter, Request
from src.phase4_api.schemas import CacheStatsResponse, CacheFlushResponse

router = APIRouter()

@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request):
    stats = request.app.state.engine.cache_stats()
    return CacheStatsResponse(**stats)

@router.delete("/cache", response_model=CacheFlushResponse)
async def flush_cache(request: Request):
    engine = request.app.state.engine
    entries_before = engine.cache.total_entries
    engine.flush_cache()
    return CacheFlushResponse(
        message="Cache flushed successfully.",
        entries_cleared=entries_before,
    )
