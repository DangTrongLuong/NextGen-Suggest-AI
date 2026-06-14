import redis
import logging
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Connection pool setup
try:
    redis_pool = redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True,
        socket_timeout=2.0,
        socket_connect_timeout=2.0,
        retry_on_timeout=True
    )
    redis_client = redis.Redis(connection_pool=redis_pool)
    logger.info("Redis connection pool initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Redis connection pool: {e}")
    redis_client = None
