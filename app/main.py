
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
# Optimize CPU thread count for torch model inference
torch.set_num_threads(8)
from contextlib import asynccontextmanager
import logging
import os

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(" Starting Smart Product Recommendation API...")
    logger.info(f"  Host: {settings.QDRANT_URL}")
    logger.info(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"  Reranker Model: {settings.RERANKER_MODEL}")
    
    # Test Redis Connection
    try:
        from app.core.redis import redis_client
        if redis_client and redis_client.ping():
            logger.info(" Successfully connected to Redis Cache.")
        else:
            logger.warning(" Could not connect to Redis Cache (ping failed).")
    except Exception as e:
        logger.error(f" Failed to connect to Redis Cache: {e}")
    
    yield

    logger.info(" Shutting down...")


# Tạo FastAPI app
app = FastAPI(
    title=" Smart Product Recommendation API",
    description="""
    ## Hệ Thống Gợi Ý Sản Phẩm Thông Minh
    
    Kết hợp **LLM Embeddings** và **Vector Search** để tìm kiếm ngữ nghĩa 
    và gợi ý sản phẩm trong thương mại điện tử.
    
    ### Tính Năng:
    -  **Semantic Search**: Tìm kiếm theo ngữ nghĩa, không chỉ keyword
    -  **Reranking**: Sắp xếp lại kết quả với cross-encoder
    -  **Similar Products**: Gợi ý sản phẩm tương tự
    -  **Evaluation**: Đánh giá chất lượng với Recall@K, NDCG@K, MRR
    
    ### Models:
    - Embedding: `intfloat/multilingual-e5-base` hoặc `BAAI/bge-m3`
    - Reranker: `BAAI/bge-reranker-base`
    - Vector DB: Qdrant
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ──────────────────────────────────────────────────────────
from app.api.routes import search, recommend, products, evaluation

app.include_router(
    search.router,
    prefix=settings.API_V1_PREFIX,
    tags=[" Search"]
)
app.include_router(
    recommend.router,
    prefix=settings.API_V1_PREFIX,
    tags=[" Recommendation"]
)
app.include_router(
    products.router,
    prefix=settings.API_V1_PREFIX,
    tags=[" Products"]
)
app.include_router(
    evaluation.router,
    prefix=settings.API_V1_PREFIX,
    tags=[" Evaluation"]
)

# ── Serve frontend static files ───────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Kiểm tra trạng thái hệ thống."""
    try:
        from app.services.qdrant_service import qdrant_service
        qdrant_info = qdrant_service.client.get_collections()
        qdrant_status = "healthy"
        collection_count = len(qdrant_info.collections)
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
        collection_count = 0

    return {
        "status": "healthy",
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "qdrant": {
            "url": settings.QDRANT_URL,
            "status": qdrant_status,
            "collections": collection_count
        }
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": " Smart Product Recommendation API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

# Reload comment: mounted frontend directory

