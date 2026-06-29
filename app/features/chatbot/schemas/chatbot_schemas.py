from pydantic import BaseModel, Field
from typing import List, Optional
from app.features.products.schemas.product_schemas import ProductBase


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str = Field(..., description="Nội dung tin nhắn")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Danh sách lịch sử hội thoại")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_profile: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str = Field(..., description="Nội dung phản hồi từ chatbot (Markdown)")
    suggested_products: List[ProductBase] = Field(default=[], description="Danh sách sản phẩm gợi ý kèm theo")
    intent: str = Field(default="product_information", description="Ý định câu hỏi")
    is_out_of_scope: bool = Field(default=False, description="Đánh dấu câu hỏi ngoài phạm vi")
