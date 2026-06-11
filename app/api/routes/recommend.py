"""API routes for recommendation endpoints."""
from fastapi import APIRouter, HTTPException, Path, Query

from app.models.schemas import (
    SimilarProductsResponse, ProductResult,
    HistoryRecommendRequest, RecommendResponse,
    HybridRecommendRequest,
)
from app.services.recommendation_service import recommendation_service

router = APIRouter()


@router.get(
    "/products/{product_id}/similar",
    response_model=SimilarProductsResponse,
    summary="Sản phẩm tương tự"
)
async def get_similar_products(
    product_id: str = Path(..., description="ID sản phẩm gốc"),
    top_k: int = Query(default=10, ge=1, le=50),
    same_category: bool = Query(default=False, description="Chỉ tìm trong cùng danh mục"),
):
    """
    Tìm sản phẩm tương tự dựa trên vector embedding.
    """
    try:
        results, source_title, latency_ms = recommendation_service.get_similar_products(
            product_id=product_id,
            top_k=top_k,
            same_category=same_category,
        )
        if not results and source_title == "Not found":
            raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")

        return SimilarProductsResponse(
            source_product_id=product_id,
            source_product_title=source_title,
            similar_products=[ProductResult(**r) for r in results],
            total=len(results),
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/recommend/from-history",
    response_model=RecommendResponse,
    summary="Gợi ý từ lịch sử xem"
)
async def recommend_from_history(request: HistoryRecommendRequest):
    """
    Gợi ý sản phẩm dựa trên lịch sử xem của user.
    Dùng Average Embedding của các sản phẩm đã xem.
    """
    try:
        results, latency_ms = recommendation_service.get_recommendations_from_history(
            viewed_product_ids=request.viewed_product_ids,
            top_k=request.top_k,
            exclude_viewed=request.exclude_viewed,
        )
        return RecommendResponse(
            recommendations=[ProductResult(**r) for r in results],
            total=len(results),
            strategy="history-based-average-embedding",
            latency_ms=latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/recommend/hybrid",
    response_model=RecommendResponse,
    summary="Gợi ý lai ghép đa tầng"
)
async def recommend_hybrid(request: HybridRecommendRequest):
    """
    Gợi ý sản phẩm lai ghép đa tầng (Popularity + Content + Behavior + Collaborative).
    """
    try:
        results, latency_ms = recommendation_service.get_hybrid_recommendations(
            viewed_product_ids=request.viewed_product_ids,
            wishlist_product_ids=request.wishlist_product_ids,
            purchased_product_ids=request.purchased_product_ids,
            popular_product_ids=request.popular_product_ids,
            top_k=request.top_k,
            exclude_viewed=request.exclude_viewed,
        )
        return RecommendResponse(
            recommendations=[ProductResult(**r) for r in results],
            total=len(results),
            strategy="hybrid-scoring",
            latency_ms=latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
