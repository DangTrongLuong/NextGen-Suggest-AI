"""API routes for search endpoints."""
import time
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.models.schemas import SearchRequest, SearchResponse, ProductResult
from app.services.search_service import search_service
from app.config import settings

router = APIRouter()


@router.get("/search", response_model=SearchResponse, summary="Semantic Search")
async def semantic_search(
    q: str = Query(..., min_length=1, max_length=500, description="Từ khóa tìm kiếm"),
    category: Optional[str] = Query(default=None, description="Lọc theo danh mục"),
    min_price: Optional[float] = Query(default=None, ge=0, description="Giá tối thiểu"),
    max_price: Optional[float] = Query(default=None, ge=0, description="Giá tối đa"),
    top_k: int = Query(default=10, ge=1, le=200, description="Số lượng kết quả trả về"),
    candidate_k: int = Query(default=50, ge=10, le=300, description="Số candidates cho reranking"),
    use_reranking: bool = Query(default=True, description="Bật/tắt reranking"),
    page: int = Query(default=0, ge=0, description="Page index"),
    size: int = Query(default=12, ge=1, le=100, description="Page size"),
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
    try:
        price_range = None
        if min_price is not None or max_price is not None:
            price_range = (min_price, max_price)

        results, latency_ms = search_service.semantic_search(
            query=q,
            use_reranking=use_reranking,
            category_filter=category,
            price_range=price_range,
            page=page,
            size=size,
        )

        # Dynamically compute total so the frontend knows whether a next page exists.
        # If we received exactly `size` items → there may be more (signal next page).
        # If fewer → this is the last page.
        actual = len(results)
        if actual >= size:
            total_elements = (page + 2) * size   # At least one more page
        else:
            total_elements = page * size + actual  # Last page

        mode = "semantic+reranking" if use_reranking else "semantic"

        return SearchResponse(
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
    q: str = Query(..., min_length=1, description="Từ khóa tìm kiếm"),
    top_k: int = Query(default=10, ge=1, le=50),
    category: Optional[str] = Query(default=None),
):
    """
    Tìm kiếm bằng keyword (baseline để so sánh với semantic search).
    """
    try:
        results, latency_ms = search_service.keyword_search(
            query=q,
            top_k=top_k,
            category_filter=category,
        )
        return SearchResponse(
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
