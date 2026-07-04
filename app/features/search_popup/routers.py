from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from app.features.search_popup.services import search_popup_service

router = APIRouter(prefix="/api/v1/ai", tags=["AI Search Helpers"])

class FastApiSuggestionItem(BaseModel):
    keyword: str
    type: str
    categoryId: Optional[int] = None
    score: float

class FastApiAutocompleteResponse(BaseModel):
    query: str
    suggestions: List[FastApiSuggestionItem]


@router.get(
    "/semantic-suggestions",
    response_model=FastApiAutocompleteResponse,
    summary="Get semantic and prefix suggestions for autocomplete"
)
async def get_semantic_suggestions(
    q: str = Query(..., min_length=1, description="Từ khóa đang nhập"),
    userId: Optional[str] = Query(default=None, description="User ID nếu có"),
    limit: int = Query(default=8, ge=1, le=50, description="Giới hạn số kết quả")
):
    try:
        results = search_popup_service.get_semantic_suggestions(
            query=q,
            user_id=userId,
            limit=limit
        )
        # Convert output to FastApiSuggestionItem model list
        items = [
            FastApiSuggestionItem(
                keyword=r["keyword"],
                type=r["type"],
                categoryId=r["categoryId"],
                score=r["score"]
            )
            for r in results
        ]
        return FastApiAutocompleteResponse(query=q, suggestions=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/related-queries",
    response_model=List[str],
    summary="Get related search keywords based on a product"
)
async def get_related_queries(
    productUuid: str = Query(..., description="UUID của sản phẩm vừa xem"),
    limit: int = Query(default=5, ge=1, le=20, description="Giới hạn số kết quả")
):
    try:
        results = search_popup_service.get_related_queries(
            product_uuid=productUuid,
            limit=limit
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
