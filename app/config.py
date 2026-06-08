
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Qdrant
    QDRANT_URL: str = "http://qdrant.datn-nextgen-suggest.site"
    QDRANT_API_KEY: str = "datn-nextgen-suggest@"
    QDRANT_COLLECTION: str = "products"
    

    # Embedding Model
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    # Alternatives: "BAAI/bge-m3"

    # Reranker Model
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # Model cache
    HF_HOME: str = "./models"

    # Search defaults
    DEFAULT_TOP_K: int = 10
    DEFAULT_CANDIDATE_K: int = 50

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"

    # Redis Cache Settings
    REDIS_HOST: str = "14.225.206.145"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "Redis_Nextgen"
    REDIS_DB: int = 0
    REDIS_CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
