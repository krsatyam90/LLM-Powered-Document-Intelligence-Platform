"""
src/core/llm.py
INT4-quantised LLaMA-3 inference via llama-cpp-python.
Supports both streaming (SSE) and blocking generation.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator

import structlog
from llama_cpp import Llama

from src.utils.config import settings

log = structlog.get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are a precise Q&A assistant. Use ONLY the provided context
to answer the user's question. If the context does not contain the answer, say so.
Be concise, accurate, and cite the source chunk IDs when helpful."""


class LLMEngine:
    """
    Wraps llama-cpp-python for INT4 GGUF model inference.

    INT4 quantisation delivers <2 % accuracy drop vs FP16 while
    reducing memory 4× — enabling 3× EC2 cost reduction.
    """

    def __init__(self, model_path: str, n_gpu_layers: int, context_length: int):
        log.info("loading_model", path=model_path, n_gpu_layers=n_gpu_layers)
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,     # -1 = all layers on GPU
            n_ctx=context_length,
            n_batch=512,
            verbose=False,
        )
        log.info("model_ready")

    # ── Streaming ─────────────────────────────────────────────────────────

    def stream(self, query: str, context_chunks: list[str]) -> Iterator[str]:
        """Yield tokens one by one for SSE streaming."""
        prompt = self._build_prompt(query, context_chunks)
        for token in self.llm(
            prompt,
            max_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=0.9,
            stream=True,
            stop=["<|eot_id|>", "</s>"],
        ):
            yield token["choices"][0]["text"]

    # ── Blocking ──────────────────────────────────────────────────────────

    def generate(self, query: str, context_chunks: list[str]) -> str:
        """Return full response (used in tests and batch eval)."""
        prompt = self._build_prompt(query, context_chunks)
        out = self.llm(
            prompt,
            max_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            stop=["<|eot_id|>", "</s>"],
        )
        return out["choices"][0]["text"].strip()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(query: str, chunks: list[str]) -> str:
        context = "\n\n".join(
            f"[Chunk {i}] {c}" for i, c in enumerate(chunks)
        )
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{RAG_SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Context:\n{context}\n\nQuestion: {query}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
