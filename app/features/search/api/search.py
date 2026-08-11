"""API routes for search endpoints."""
import time
import uuid
import json
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Header, BackgroundTasks
from typing import Optional

from app.features.search.schemas.search_schemas import SearchRequest, SearchResponse
from app.features.products.schemas.product_schemas import ProductResult
from app.features.search.services.search_service import search_service
from app.config.settings import settings
from app.shared.redis.service import redis_service
from app.shared.utils.query_validator import is_malicious_or_invalid_query

logger = logging.getLogger("search_api")

def enqueue_search_log_task(
    request_id: str,
    keyword: str,
    response_time_ms: float,
    total_results: int,
    session_id: str,
    products: list
):
    if is_malicious_or_invalid_query(keyword):
        logger.info(f"Skipping enqueue of search log for malicious query: {keyword}")
        return

    payload = {
        "request_id": request_id,
        "keyword": keyword,
        "response_time_ms": response_time_ms,
        "total_results": total_results,
        "session_id": session_id,
        "searched_at": datetime.now().isoformat(),
        "products": [{"product_id": p.get("product_id"), "product_name": p.get("title")} for p in products]
    }
    
    # Try pushing to Redis list queue
    pushed = False
    if redis_service.is_available():
        try:
            redis_service.client.rpush("analytics:search_queue", json.dumps(payload))
            pushed = True
            logger.debug(f"Enqueued search log to Redis: {request_id}")
        except Exception as e:
            redis_service.mark_offline(e)
            logger.warning(f"Failed to queue search log in Redis: {e}")
            
    if not pushed:
        # Fallback to direct DB save
        logger.info(f"Fallback to direct DB save for search log: {request_id}")
        from app.features.search_analytics.worker import save_search_log_to_db, invalidate_analytics_caches
        try:
            save_search_log_to_db(payload)
            invalidate_analytics_caches()
            
            # Try to broadcast WebSocket update from fallback
            from app.features.search_analytics.websocket.manager import ws_manager
            message = {
                "type": "NEW_SEARCH",
                "data": {
                    "request_id": payload["request_id"],
                    "keyword": payload["keyword"],
                    "response_time_ms": payload["response_time_ms"],
                    "total_results": payload["total_results"],
                    "session_id": payload["session_id"],
                    "searched_at": payload["searched_at"],
                    "products": payload["products"][:5]
                }
            }
            try:
                ws_manager.broadcast_threadsafe(message)
            except Exception as ws_err:
                logger.error(f"Failed to broadcast from fallback: {ws_err}")
        except Exception as err:
            logger.error(f"Direct DB save fallback failed: {err}")

def check_search_rate_limit_and_length(session_id: Optional[str], query: str, client_ip: str = "127.0.0.1"):
    # 1. Length check: max 100 chars
    if len(query.strip()) > 100:
        raise HTTPException(
            status_code=400,
            detail="Từ khóa tìm kiếm không được vượt quá 100 ký tự."
        )

    # 2. Rate limit check: max 20 searches / 60 seconds per session_id / IP
    rate_key = f"analytics:rate_limit:search:{session_id or client_ip}"
    if redis_service.is_available() and redis_service.client:
        try:
            count = redis_service.client.incr(rate_key)
            if count == 1:
                redis_service.client.expire(rate_key, 60)
            if count > 20:
                logger.warning(f"Rate limit exceeded for session '{session_id}' (IP: {client_ip}): {count} requests in 60s")
                
                try:
                    analytics_repository.insert_audit_event_direct(
                        "SPAM_SEARCH_DETECTED",
                        session_id or "anonymous",
                        client_ip,
                        f"Spam search detected: {count} search requests in 60 seconds."
                    )
                    from app.features.search_analytics.websocket.manager import ws_manager
                    ws_manager.broadcast_threadsafe({
                        "type": "ALERT",
                        "data": {
                            "level": "WARNING",
                            "message": f"Phát hiện nghi vấn Spam Tìm Kiếm từ phiên {session_id} ({count} lượt/phút)."
                        }
                    })
                except Exception:
                    pass

                raise HTTPException(
                    status_code=429,
                    detail="Bạn đang thao tác tìm kiếm quá nhanh. Vui lòng đợi trong giây lát và thử lại!"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Rate limit check failed in Redis: {e}")


router = APIRouter()


@router.get("/search", response_model=SearchResponse, summary="Semantic Search")
async def semantic_search(
    background_tasks: BackgroundTasks,
    q: str = Query(..., min_length=1, max_length=500, description="Từ khóa tìm kiếm"),
    category: Optional[str] = Query(default=None, description="Lọc theo danh mục"),
    min_price: Optional[float] = Query(default=None, ge=0, description="Giá tối thiểu"),
    max_price: Optional[float] = Query(default=None, ge=0, description="Giá tối đa"),
    top_k: int = Query(default=10, ge=1, le=200, description="Số lượng kết quả trả về"),
    candidate_k: int = Query(default=50, ge=10, le=300, description="Số candidates cho reranking"),
    use_reranking: bool = Query(default=True, description="Bật/tắt reranking"),
    page: int = Query(default=0, ge=0, description="Page index"),
    size: int = Query(default=10, ge=1, le=100, description="Page size"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    x_client_start_time: Optional[str] = Header(default=None, alias="X-Client-Start-Time"),
):
    """
    Tìm kiếm sản phẩm theo ngữ nghĩa.

    - **q**: Câu tìm kiếm tiếng Việt (Required)
    - **category**: Lọc theo danh mục (Optional)
    - **min_price**: Lọc giá tối thiểu (Optional)
    - **max_price**: Lọc giá tối đa (Optional)
    - **top_k**: Số kết quả trả về (Optional)
    - **candidate_k**: Số ứng viên cho reranking (Optional)
    - **use_reranking**: Bật/tắt cross-encoder reranking (Optional)
    - **page**: Vị trí trang (Optional)
    - **size**: Số lượng mỗi trang (Optional)
    """
    check_search_rate_limit_and_length(x_session_id, q)
    req_id = str(uuid.uuid4())
    if is_malicious_or_invalid_query(q):
        logger.warning(f"Blocked malicious/invalid semantic search query: '{q}'")
        return SearchResponse(
            request_id=req_id,
            query=q,
            results=[],
            total=0,
            search_mode="blocked",
            latency_ms=0,
            embedding_model=settings.EMBEDDING_MODEL,
            reranker_model=None,
        )

    try:
        price_range = None
        if min_price is not None or max_price is not None:
            price_range = (min_price, max_price)

        from fastapi.concurrency import run_in_threadpool

        results, latency_ms = await run_in_threadpool(
            search_service.semantic_search,
            query=q,
            use_reranking=use_reranking,
            category_filter=category,
            price_range=price_range,
            page=page,
            size=size,
        )

        # Compute end-to-end client latency if client sent timestamp
        if x_client_start_time:
            try:
                client_start = float(x_client_start_time)
                now_ms = time.time() * 1000.0
                if now_ms > client_start:
                    latency_ms = round(now_ms - client_start, 2)
            except Exception:
                pass

        actual = len(results)
        if actual >= size:
            total_elements = (page + 2) * size   # At least one more page
        else:
            total_elements = page * size + actual  # Last page

        mode = "semantic+reranking" if use_reranking else "semantic"

        # Log search asynchronously
        session_id = x_session_id or "anonymous"
        background_tasks.add_task(
            enqueue_search_log_task,
            req_id,
            q,
            latency_ms,
            total_elements,
            session_id,
            results
        )

        return SearchResponse(
            request_id=req_id,
            query=q,
            results=[ProductResult(**r) for r in results],
            total=total_elements,
            search_mode=mode,
            latency_ms=latency_ms,
            embedding_model=settings.EMBEDDING_MODEL,
            reranker_model=settings.RERANKER_MODEL if use_reranking else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/search/keyword", response_model=SearchResponse, summary="Keyword Search (Baseline)")
async def keyword_search(
    background_tasks: BackgroundTasks,
    q: str = Query(..., min_length=1, description="Từ khóa tìm kiếm"),
    top_k: int = Query(default=10, ge=1, le=50),
    category: Optional[str] = Query(default=None),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
):
    """
    Tìm kiếm bằng keyword (baseline để so sánh với semantic search).
    """
    check_search_rate_limit_and_length(x_session_id, q)
    req_id = str(uuid.uuid4())
    if is_malicious_or_invalid_query(q):
        logger.warning(f"Blocked malicious/invalid keyword search query: '{q}'")
        return SearchResponse(
            request_id=req_id,
            query=q,
            results=[],
            total=0,
            search_mode="blocked",
            latency_ms=0,
            embedding_model="keyword-tfidf",
        )

    try:
        results, latency_ms = search_service.keyword_search(
            query=q,
            top_k=top_k,
            category_filter=category,
        )
        session_id = x_session_id or "anonymous"
        background_tasks.add_task(
            enqueue_search_log_task,
            req_id,
            q,
            latency_ms,
            len(results),
            session_id,
            results
        )

        return SearchResponse(
            request_id=req_id,
            query=q,
            results=[ProductResult(**r) for r in results],
            total=len(results),
            search_mode="keyword",
            latency_ms=latency_ms,
            embedding_model="keyword-tfidf",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/search/compare",
    summary="Compare keyword vs semantic vs reranking"
)
async def compare_methods(
    q: str = Query(..., description="Query để so sánh"),
    top_k: int = Query(default=10, ge=1, le=20),
):
    """
    So sánh cả 3 phương pháp cho cùng một query.
    Endpoint này dùng cho demo và evaluation.
    """
    try:
        kw_results, kw_latency = search_service.keyword_search(q, top_k)
        sem_results, sem_latency = search_service.semantic_search(
            q, top_k=top_k, use_reranking=False
        )
        rer_results, rer_latency = search_service.semantic_search(
            q, top_k=top_k, use_reranking=True
        )

        return {
            "query": q,
            "keyword": {
                "results": kw_results,
                "latency_ms": kw_latency,
                "mode": "keyword",
            },
            "semantic": {
                "results": sem_results,
                "latency_ms": sem_latency,
                "mode": "semantic",
            },
            "semantic_reranking": {
                "results": rer_results,
                "latency_ms": rer_latency,
                "mode": "semantic+reranking",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
