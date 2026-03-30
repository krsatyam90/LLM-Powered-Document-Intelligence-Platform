"""
src/api/routes/query.py
POST /api/v1/query       → blocking JSON response
GET  /api/v1/query/stream → SSE streaming tokens
"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.core.rag_pipeline import RAGPipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    pipeline = RAGPipeline(
        vector_store=request.app.state.vector_store,
        llm=request.app.state.llm,
    )
    docs = pipeline.retrieve(req.query, top_k=req.top_k)
    answer = pipeline.answer(req.query, top_k=req.top_k)
    return QueryResponse(answer=answer, sources=docs)


@router.get("/query/stream")
async def query_stream(query: str, top_k: int = 5, request: Request = None):
    """Stream tokens via Server-Sent Events."""
    pipeline = RAGPipeline(
        vector_store=request.app.state.vector_store,
        llm=request.app.state.llm,
    )

    async def event_generator():
        for token in pipeline.stream_answer(query, top_k=top_k):
            yield {"data": token}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
