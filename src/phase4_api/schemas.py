from typing import Optional, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    query            : str
    cache_hit        : bool
    matched_query    : Optional[str]   = None
    similarity_score : Optional[float] = None
    result           : str
    dominant_cluster : int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count    : int
    miss_count   : int
    hit_rate     : float
    threshold    : float
    bucket_sizes : dict[str, Any]


class CacheFlushResponse(BaseModel):
    message        : str
    entries_cleared: int


class HealthResponse(BaseModel):
    status          : str
    chroma_docs     : int
    cache_entries   : int
    embedding_device: str
    n_clusters      : int
