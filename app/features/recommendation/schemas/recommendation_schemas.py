from pydantic import BaseModel, Field
from typing import List, Optional
from app.features.products.schemas.product_schemas import ProductResult


class SimilarProductsResponse(BaseModel):
    source_product_id: str
    source_product_title: str
    similar_products: List[ProductResult]
    total: int
    latency_ms: float


class HistoryRecommendRequest(BaseModel):
    viewed_product_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        example=["prod_001", "prod_005"],
        description="Danh sách product IDs đã xem"
    )
    top_k: int = Field(default=10, ge=1, le=50)
    exclude_viewed: bool = Field(default=True, description="Loại bỏ sản phẩm đã xem")


class HybridRecommendRequest(BaseModel):
    viewed_product_ids: Optional[List[str]] = []
    wishlist_product_ids: Optional[List[str]] = []
    purchased_product_ids: Optional[List[str]] = []
    popular_product_ids: Optional[List[str]] = []
    location_product_ids: Optional[List[str]] = []
    user_age: Optional[int] = None
    user_gender: Optional[str] = None
    user_occupation: Optional[str] = None
    user_income: Optional[float] = None
    user_marital_status: Optional[str] = None
    user_location: Optional[str] = None
    user_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    exclude_viewed: bool = Field(default=True)


class RecommendResponse(BaseModel):
    recommendations: List[ProductResult]
    total: int
    strategy: str = Field(..., description="'similar' | 'history-based' | 'popular'")
    latency_ms: float
