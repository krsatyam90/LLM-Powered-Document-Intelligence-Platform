"""
RAG-LLaMA3  ·  main.py
FastAPI application factory — mounts routers, registers middleware,
wires up startup/shutdown lifecycle hooks.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.routes import query, feedback, health
from src.core.vector_store import VectorStore
from src.core.llm import LLMEngine
from src.utils.config import settings
from src.utils.logging import configure_logging

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("startup", env=settings.ENV)

    # Warm up vector store & model (loaded once, shared across workers)
    app.state.vector_store = VectorStore(
        index_path=settings.FAISS_INDEX_PATH,
        embedding_model=settings.EMBEDDING_MODEL,
    )
    app.state.llm = LLMEngine(
        model_path=settings.MODEL_PATH,
        n_gpu_layers=settings.N_GPU_LAYERS,
        context_length=settings.CONTEXT_LENGTH,
    )
    log.info("components_ready")
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG-LLaMA3 API",
        version="1.0.0",
        docs_url="/docs",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(query.router,  prefix="/api/v1", tags=["query"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()
