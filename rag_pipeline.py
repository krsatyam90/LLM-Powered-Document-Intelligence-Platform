"""
src/core/rag_pipeline.py
Orchestrates retrieval → prompt-assembly → generation.
"""

from __future__ import annotations

from typing import AsyncIterator, List

import structlog

from src.core.llm import LLMEngine
from src.core.vector_store import VectorStore

log = structlog.get_logger(__name__)


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, llm: LLMEngine):
        self.vs = vector_store
        self.llm = llm

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        return self.vs.search(query, top_k=top_k)

    def answer(self, query: str, top_k: int = 5) -> str:
        docs = self.retrieve(query, top_k)
        chunks = [d["text"] for d in docs]
        response = self.llm.generate(query, chunks)
        log.info("rag_answer", query=query[:60], chunks_used=len(chunks))
        return response

    def stream_answer(self, query: str, top_k: int = 5):
        """Generator that yields tokens for SSE endpoint."""
        docs = self.retrieve(query, top_k)
        chunks = [d["text"] for d in docs]
        log.info("rag_stream_start", query=query[:60])
        yield from self.llm.stream(query, chunks)
