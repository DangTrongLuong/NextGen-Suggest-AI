from fastapi import APIRouter, HTTPException
from typing import List

from app.features.chatbot.schemas.chatbot_schemas import ChatRequest, ChatResponse
from app.features.products.schemas.product_schemas import ProductBase
from app.features.chatbot.services.chatbot_service import chatbot_service

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Trợ lý mua sắm AI Chatbot"
)
async def chat_assistant(request: ChatRequest):
    """
    Tư vấn mua sắm và tìm kiếm sản phẩm thông minh.
    """
    try:
        # Chuyển đổi dữ liệu sang list of dicts phù hợp với chatbot_service
        messages_dict = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        response_text, suggested_products, intent, is_out_of_scope = chatbot_service.chat_response(
            messages=messages_dict,
            user_id=request.user_id,
            session_id=request.session_id,
            user_profile=request.user_profile
        )
        
        # Hydrate suggested products to match ProductBase Schema
        hydrated_products = []
        for p in suggested_products:
            # Gán giá trị mặc định cho các trường còn thiếu nếu cần
            hydrated_products.append(ProductBase(
                product_id=str(p.get("product_id", "")),
                title=str(p.get("title", "")),
                description=str(p.get("description", "")),
                category=str(p.get("category", "")),
                tags=p.get("tags", []),
                price=float(p.get("price", 0.0)) if p.get("price") is not None else 0.0,
                image_url=str(p.get("image_url", ""))
            ))
            
        return ChatResponse(
            response=response_text,
            suggested_products=hydrated_products,
            intent=intent,
            is_out_of_scope=is_out_of_scope
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
