from pydantic import BaseModel, Field
from typing import List, Optional


class ProductBase(BaseModel):
    product_id: str
    title: str
    description: Optional[str] = ""
    category: Optional[str] = ""
    tags: Optional[List[str]] = []
    price: Optional[float] = 0.0
    image_url: Optional[str] = ""


class ProductResult(ProductBase):
    """Kết quả tìm kiếm/gợi ý với score."""
    score: float = Field(..., description="Cosine similarity score từ ANN search (0-1)")
    reranker_score: Optional[float] = Field(None, description="Cross-encoder reranking score")
    rank: Optional[int] = Field(None, description="Vị trí trong kết quả")
