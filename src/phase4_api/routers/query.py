from fastapi import APIRouter, Request, HTTPException
from src.phase4_api.schemas import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request):
    engine = request.app.state.engine
    original_threshold = None
    if body.threshold is not None:
        original_threshold = engine.cache.threshold
        engine.cache.threshold = body.threshold
    try:
        result = engine.query(body.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    finally:
        if original_threshold is not None:
            engine.cache.threshold = original_threshold
    return QueryResponse(**result)
