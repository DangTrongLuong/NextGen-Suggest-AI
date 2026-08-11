import json
import time
import threading
import logging
from datetime import datetime
from app.features.search_analytics.repositories.connection_pool import get_db_connection
from app.features.search_analytics.websocket.manager import ws_manager
from app.shared.redis.service import redis_service

logger = logging.getLogger("analytics_worker")

# Flag to control the loop
_worker_running = False
_worker_thread = None

def save_search_log_to_db(log_data: dict) -> int:
    """Save search log data and its product rankings to MySQL database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Insert into search_logs
            query = """
            INSERT INTO search_logs (
                request_id, keyword, query_text, response_time_ms, latency_ms,
                total_results, results_count, session_id, searched_at, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            now = datetime.now()
            searched_at_str = log_data.get("searched_at")
            searched_at = datetime.fromisoformat(searched_at_str) if searched_at_str else now

            cursor.execute(query, (
                log_data.get("request_id"),
                log_data.get("keyword"),
                log_data.get("keyword"),  # query_text compatibility field
                int(log_data.get("response_time_ms", 0)),
                int(log_data.get("response_time_ms", 0)),  # latency_ms compatibility field
                int(log_data.get("total_results", 0)),
                int(log_data.get("total_results", 0)),  # results_count compatibility field
                log_data.get("session_id", "anonymous"),
                searched_at,
                now
            ))
            search_log_id = cursor.lastrowid

            # Batch insert into search_log_products
            products = log_data.get("products", [])
            if products:
                product_query = """
                INSERT INTO search_log_products (
                    search_log_id, product_id, product_name, ranking_position
                ) VALUES (%s, %s, %s, %s);
                """
                product_values = [
                    (search_log_id, p.get("product_id"), p.get("product_name"), i + 1)
                    for i, p in enumerate(products)
                ]
                cursor.executemany(product_query, product_values)

            conn.commit()
            logger.debug(f"Saved search log ID {search_log_id} to database.")
            return search_log_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save search log to DB: {e}")
        raise e
    finally:
        conn.close()

def invalidate_analytics_caches():
    """Clear Redis caches for overview, top-keywords, and top-products stats."""
    if redis_service.client:
        try:
            # Match pattern analytics:* or delete all specific keys
            keys = redis_service.client.keys("analytics:overview*") + \
                   redis_service.client.keys("analytics:top_keywords*") + \
                   redis_service.client.keys("analytics:top_products*")
            if keys:
                redis_service.client.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} analytics cache keys in Redis.")
        except Exception as e:
            logger.warning(f"Error invalidating Redis caches: {e}")

def analytics_worker_loop():
    """Blocking poll loop to consume events from Redis search queue."""
    global _worker_running
    import redis
    logger.info("Search analytics background worker loop started.")
    
    redis_was_online = True
    
    while _worker_running:
        if not redis_service.client:
            time.sleep(2)
            continue
            
        try:
            # BLPOP blocks until an element is pushed, or times out after 1 second
            result = redis_service.client.blpop("analytics:search_queue", timeout=1)
            
            if not redis_was_online:
                logger.info("Redis connection restored. Resuming normal analytics polling.")
                redis_was_online = True
                redis_service.is_online = True
                
            if not result:
                continue
                
            # blpop returns a tuple: (key, value)
            _, payload_str = result
            log_data = json.loads(payload_str)
            
            # 1. Save to MySQL DB
            try:
                save_search_log_to_db(log_data)
            except Exception as db_err:
                logger.error(f"Worker failed to write search log to database: {db_err}")
                
            # 2. Invalidate Cache
            invalidate_analytics_caches()
            
            # 3. Broadcast to WebSocket clients
            message = {
                "type": "NEW_SEARCH",
                "data": {
                    "request_id": log_data.get("request_id"),
                    "keyword": log_data.get("keyword"),
                    "response_time_ms": log_data.get("response_time_ms"),
                    "total_results": log_data.get("total_results"),
                    "session_id": log_data.get("session_id"),
                    "searched_at": log_data.get("searched_at"),
                    "products": log_data.get("products", [])[:5]
                }
            }
            try:
                ws_manager.broadcast_threadsafe(message)
            except Exception as ws_err:
                logger.error(f"Failed to trigger WS broadcast: {ws_err}")
                
        except Exception as e:
            error_msg = str(e)
            is_redis_conn_error = (
                "Timeout connecting to server" in error_msg 
                or "Connection refused" in error_msg 
                or "Error connecting" in error_msg 
                or isinstance(e, (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError))
            )
            
            if is_redis_conn_error:
                if redis_was_online:
                    logger.warning(f"Redis connection lost: {e}. Worker will retry every 30 seconds.")
                    redis_was_online = False
                    redis_service.is_online = False
                    redis_service.last_check = time.time()
                # Sleep in small steps to allow quick shutdown on join()
                for _ in range(30):
                    if not _worker_running:
                        break
                    time.sleep(1)
            else:
                logger.error(f"Error in analytics worker loop: {e}")
                time.sleep(5)  # Backoff on general failures

    logger.info("Search analytics background worker loop stopped.")

def start_analytics_worker():
    """Start background worker thread."""
    global _worker_running, _worker_thread
    if _worker_running:
        return
        
    _worker_running = True
    _worker_thread = threading.Thread(target=analytics_worker_loop, name="AnalyticsWorkerThread", daemon=True)
    _worker_thread.start()
    logger.info("Search analytics background worker thread spawned successfully.")

def stop_analytics_worker():
    """Stop background worker thread."""
    global _worker_running, _worker_thread
    _worker_running = False
    if _worker_thread:
        _worker_thread.join(timeout=3)
        _worker_thread = None
    logger.info("Search analytics background worker thread stopped.")
