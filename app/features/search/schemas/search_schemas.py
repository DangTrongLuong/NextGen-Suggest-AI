from pydantic import BaseModel, Field
from typing import List, Optional
from app.features.products.schemas.product_schemas import ProductResult


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        example="tai nghe chống ồn cho văn phòng"
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Số lượng kết quả trả về")
    candidate_k: int = Field(default=50, ge=10, le=200, description="Số candidates cho reranking")
    use_reranking: bool = Field(default=True, description="Bật/tắt reranking")
    category: Optional[str] = Field(default=None, description="Lọc theo danh mục")
    min_price: Optional[float] = Field(default=None, ge=0, description="Giá tối thiểu")
    max_price: Optional[float] = Field(default=None, ge=0, description="Giá tối đa")


class SearchResponse(BaseModel):
    request_id: Optional[str] = None
    query: str
    results: List[ProductResult]
    total: int
    search_mode: str = Field(
        ...,
        description="'keyword' | 'semantic' | 'semantic+reranking'"
    )
    latency_ms: float
    embedding_model: str
    reranker_model: Optional[str] = None


class KeywordSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, example="tai nghe sony")
    top_k: int = Field(default=10, ge=1, le=50)
    category: Optional[str] = None
