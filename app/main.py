from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
# Optimize CPU thread count for torch model inference
torch.set_num_threads(8)
from contextlib import asynccontextmanager
import logging
import os

from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from app.features.search_analytics.websocket.manager import ws_manager
    ws_manager.loop = asyncio.get_running_loop()

    # Ensure local models directory exists for HuggingFace cache
    models_dir = os.path.abspath(settings.HF_HOME)
    os.makedirs(models_dir, exist_ok=True)
    os.environ["HF_HOME"] = models_dir

    logger.info(" Starting Smart Product Recommendation API...")
    logger.info(f"  Host: {settings.QDRANT_URL}")
    logger.info(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"  Reranker Model: {settings.RERANKER_MODEL}")
    logger.info(f"  Models Cache Directory: {models_dir}")
    
    # Auto-check and create Search Analytics tables in MySQL if not exist
    try:
        from scripts.migrate_analytics import run_migration
        run_migration()
        logger.info(" Search Analytics DB auto-migration check completed.")
    except Exception as e:
        logger.error(f" Failed to run Search Analytics DB auto-migration: {e}")

    # Test Redis Connection
    try:
        from app.shared.redis.client import redis_client
        if redis_client and redis_client.ping():
            logger.info(" Successfully connected to Redis Cache.")
        else:
            logger.warning(" Could not connect to Redis Cache (ping failed).")
    except Exception as e:
        logger.error(f" Failed to connect to Redis Cache: {e}")
    
    # Check and create Qdrant collection if not exists
    try:
        from app.shared.qdrant.service import qdrant_service
        from app.features.search.services.embedding_service import embedding_service
        vector_size = embedding_service.vector_size
        qdrant_service.create_collection(vector_size=vector_size, recreate=False)
        logger.info(f" Qdrant collection '{qdrant_service.collection_name}' checked/created successfully.")
    except Exception as e:
        logger.error(f" Failed to check/create Qdrant collection: {e}")
    
    # Warm-up AI Embedding & Reranker models to eliminate Cold Start latency
    try:
        from app.features.search.services.rerank_service import rerank_service
        logger.info(" Pre-warming AI Embedding & Reranker models...")
        _dummy_vec = embedding_service.embed_query("sản phẩm mỏng nhẹ")
        _dummy_docs = [{"title": "Sản phẩm thử nghiệm", "category": "Điện thoại", "short_context": "Sản phẩm thử nghiệm"}]
        rerank_service.rerank("sản phẩm mỏng nhẹ", _dummy_docs, top_k=1)
        logger.info(" AI models pre-warmed and ready in RAM!")
    except Exception as e:
        logger.warning(f" AI Model warm-up skipped: {e}")

    # Start Search Analytics Queue Worker Thread
    try:
        from app.features.search_analytics.worker import start_analytics_worker
        start_analytics_worker()
    except Exception as e:
        logger.error(f" Failed to start Search Analytics worker: {e}")

    # Start RabbitMQ Consumer Thread
    try:
        from app.features.indexing.consumer import rabbitmq_consumer
        rabbitmq_consumer.start()
    except Exception as e:
        logger.error(f" Failed to start RabbitMQ consumer: {e}")

    yield

    # Stop RabbitMQ Consumer Thread
    try:
        from app.features.indexing.consumer import rabbitmq_consumer
        rabbitmq_consumer.stop()
    except Exception as e:
        logger.error(f" Failed to stop RabbitMQ consumer: {e}")

    # Stop Search Analytics Queue Worker Thread
    try:
        from app.features.search_analytics.worker import stop_analytics_worker
        stop_analytics_worker()
    except Exception as e:
        logger.error(f" Failed to stop Search Analytics worker: {e}")

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
from app.features.search.api import search
from app.features.recommendation.api import recommend
from app.features.products.api import products
from app.features.evaluation.api import evaluation
from app.features.chatbot.api import chat
from app.features.indexing.api import import_routes, indexing_routes

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
app.include_router(
    chat.router,
    prefix=settings.API_V1_PREFIX,
    tags=[" Chatbot"]
)
app.include_router(
    import_routes.router,
    prefix=settings.API_V1_PREFIX,
    tags=[" Excel Import"]
)
app.include_router(
    indexing_routes.router,
    prefix=settings.API_V1_PREFIX,
    tags=[" Indexing Sync"]
)

# ── AI Search Helpers Router ─────────────────────────────────────────────────
from app.features.search_popup.routers import router as search_popup_router
app.include_router(
    search_popup_router
)

# ── Search Analytics Portal Routers ──────────────────────────────────────────
from app.features.search_analytics.routers import auth as analytics_auth
from app.features.search_analytics.routers import search_analytics
from app.features.search_analytics.websocket import router as analytics_ws

app.include_router(
    analytics_auth.router,
    prefix="/analytics/auth",
    tags=["Search Analytics Auth"]
)
app.include_router(
    search_analytics.router,
    prefix="/analytics",
    tags=["Search Analytics"]
)
app.include_router(
    analytics_ws.router,
    tags=["Search Analytics Realtime"]
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
        from app.shared.qdrant.service import qdrant_service
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
