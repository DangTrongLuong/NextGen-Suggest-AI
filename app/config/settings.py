from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str 
    QDRANT_COLLECTION: str

    # LLM
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None

    # Embedding
    EMBEDDING_MODEL: str

    # Reranker
    RERANKER_MODEL: str

    # HuggingFace Cache
    HF_HOME: str = "./models"

    # Search
    DEFAULT_TOP_K: int = 10
    DEFAULT_CANDIDATE_K: int = 50

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_DB: int = 0
    REDIS_CACHE_TTL: int = 3600

    # MySQL
    MYSQL_HOST: str
    MYSQL_PORT: int 
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_DATABASE: str

    # Spring Boot Url
    SPRING_BOOT_URL: str

    # Search Analytics Config
    ANALYTICS_JWT_SECRET: str 
    ANALYTICS_DEFAULT_USERNAME: Optional[str] = None
    ANALYTICS_DEFAULT_EMAIL: Optional[str] = None
    ANALYTICS_DEFAULT_PASSWORD: Optional[str] = None
    ANALYTICS_ALLOWED_IPS: Optional[str] = None

    # RabbitMQ
    RABBITMQ_HOST: str
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USERNAME: str
    RABBITMQ_PASSWORD: str
    RABBITMQ_QUEUE: str 
    RABBITMQ_EXCHANGE: str 
    RABBITMQ_ROUTING_KEY: str 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings():
    return Settings()


settings = get_settings()
