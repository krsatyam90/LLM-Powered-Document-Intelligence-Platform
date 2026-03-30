"""src/utils/config.py"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENV: str = "development"

    # Model
    MODEL_PATH: str = "models/llama-3-8b-instruct-q4_k_m.gguf"
    N_GPU_LAYERS: int = -1          # -1 = offload all
    CONTEXT_LENGTH: int = 4096
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.2

    # FAISS
    FAISS_INDEX_PATH: str = "data/faiss.index"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # AWS
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET: str = "rag-llama3-docs"

    # Database (feedback)
    DATABASE_URL: str = "sqlite:///./feedback.db"

    # API
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
