import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import urllib.request

from app.services.search_service import search_service
from app.services.qdrant_service import qdrant_service
from app.config import settings

logger = logging.getLogger(__name__)

class ChatbotService:

    def chat_response(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict]]:
        """
        Xử lý tin nhắn chat của người dùng.
        Hỗ trợ LLM (Gemini / OpenAI) nếu có API Key, nếu không sẽ dùng rule-based fallback.
        """
        if not messages:
            return "Xin chào! Tôi là trợ lý mua sắm AI của Nivora. Tôi có thể giúp gì cho bạn?", []

        last_message = messages[-1].get("content", "").strip()
        last_message_lower = last_message.lower()

        logger.info(f"Chatbot received message: '{last_message}'")

        # 1. Tìm kiếm sản phẩm liên quan bằng Semantic Search (RAG Context)
        products = []
        try:
            # Chỉ tìm kiếm khi tin nhắn có từ khóa liên quan đến mua sắm, sản phẩm hoặc so sánh
            is_shopping_query = any(w in last_message_lower for w in [
                "tìm", "mua", "bán", "ghế", "bàn", "tủ", "kệ", "giá", "sản phẩm", "so sánh",
                "khác biệt", "nhất", "giường", "sofa", "nội thất", "đẹp", "rẻ", "đắt", "triệu"
            ])
            if is_shopping_query:
                products, _ = search_service.semantic_search(query=last_message, top_k=3)
        except Exception as e:
            logger.error(f"Semantic search failed during chatbot retrieval: {e}")

        # 2. Xây dựng System Prompt cho LLM
        system_prompt = (
            "Bạn là Trợ lý Mua sắm AI của Nivora - hệ thống thương mại điện tử chuyên cung cấp đồ nội thất cao cấp (bàn, ghế, tủ, giường, kệ,...).\n"
            "Nhiệm vụ của bạn:\n"
            "1. Trả lời khách hàng một cách lịch sự, thân thiện, súc tích và chuyên nghiệp.\n"
            "2. Sử dụng định dạng Markdown (in đậm, danh sách, bảng biểu, xuống dòng) để câu trả lời dễ đọc.\n"
            "3. Khi so sánh sản phẩm, BẮT BUỘC sử dụng bảng Markdown (ví dụ: | Tiêu chí | Sản phẩm A | Sản phẩm B |) để khách hàng dễ so sánh giá và tính năng.\n"
            "4. KHÔNG tự bịa ra thông tin sản phẩm hoặc giá bán nếu không có trong ngữ cảnh. Hãy sử dụng danh sách sản phẩm thật được cung cấp bên dưới.\n"
            "5. Nếu khách hàng hỏi về sản phẩm không có sẵn, hãy tư vấn nhẹ nhàng và giới thiệu các sản phẩm tương tự đang có.\n"
        )

        if products:
            system_prompt += (
                "\nDưới đây là danh sách sản phẩm THẬT từ cơ sở dữ liệu của cửa hàng phù hợp với yêu cầu hiện tại. "
                "Hãy dựa vào thông tin chính xác này để tư vấn, mô tả và so sánh cho khách hàng:\n"
            )
            for idx, p in enumerate(products, 1):
                desc = p.get("description", "")
                if desc and len(desc) > 300:
                    desc = desc[:300] + "..."
                system_prompt += (
                    f"Sản phẩm {idx}:\n"
                    f"- SKU / ID: {p.get('product_id')}\n"
                    f"- Tên sản phẩm: {p.get('title')}\n"
                    f"- Giá: ${p.get('price')}\n"
                    f"- Danh mục: {p.get('category')}\n"
                    f"- Mô tả: {desc}\n\n"
                )

        # 3. Gửi yêu cầu đến LLM (Gemini hoặc OpenAI)
        llm_response = None
        if settings.GEMINI_API_KEY:
            logger.info("Using Gemini API for chatbot response")
            llm_response = self._call_gemini(system_prompt, messages)
        elif settings.OPENAI_API_KEY:
            logger.info("Using OpenAI API for chatbot response")
            llm_response = self._call_openai(system_prompt, messages)

        if llm_response:
            return llm_response, products

        # 4. Fallback Rule-Based nếu không có API Keys hoặc LLM lỗi
        logger.info("No LLM key configured or LLM failed. Falling back to rule-based logic.")
        
        # Phát hiện Intent so sánh sản phẩm
        if "so sánh" in last_message_lower or "khác biệt" in last_message_lower:
            return self._handle_comparison(last_message)

        # Mặc định: Trả về danh sách sản phẩm tìm được
        if products:
            response_content = (
                f"Dựa trên yêu cầu của bạn, tôi đã tìm thấy một số sản phẩm phù hợp tại Nivora:\n\n"
            )
            for idx, p in enumerate(products, 1):
                desc = p.get("description", "")
                if desc and len(desc) > 120:
                    desc = desc[:120] + "..."
                
                response_content += (
                    f"**{idx}. {p.get('title')}**\n"
                    f"- **Giá**: ${p.get('price')}\n"
                    f"- **Danh mục**: {p.get('category')}\n"
                    f"- **Mô tả**: {desc}\n\n"
                )
            
            response_content += "Bạn có muốn tôi tư vấn chi tiết hơn về sản phẩm nào ở trên không?"
            return response_content, products

        # Phản hồi mặc định nếu không có sản phẩm nào
        return (
            "Hiện tại tôi chưa tìm thấy sản phẩm nào khớp chính xác với mô tả của bạn. "
            "Bạn có thể mô tả chi tiết hơn về nhu cầu sử dụng, màu sắc, chất liệu hoặc khoảng giá mong muốn không? "
            "(Ví dụ: 'Tôi cần tìm áo thun màu đen cotton' hoặc 'bàn học gỗ dưới 2 triệu')."
        ), []

    def _call_gemini(self, system_prompt: str, messages: List[Dict[str, str]]) -> Optional[str]:
        api_key = settings.GEMINI_API_KEY
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        contents = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })
            
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except Exception as e:
            logger.error(f"Failed to call Gemini API: {e}", exc_info=True)
            return None

    def _call_openai(self, system_prompt: str, messages: List[Dict[str, str]]) -> Optional[str]:
        api_key = settings.OPENAI_API_KEY
        url = "https://api.openai.com/v1/chat/completions"
        
        openai_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = "assistant" if msg.get("role") == "assistant" else "user"
            openai_messages.append({"role": role, "content": msg.get("content", "")})
            
        payload = {
            "model": "gpt-4o-mini",
            "messages": openai_messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["choices"][0]["message"]["content"]
                return text
        except Exception as e:
            logger.error(f"Failed to call OpenAI API: {e}", exc_info=True)
            return None

    def _handle_comparison(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Xử lý ý định so sánh sản phẩm bằng cách lấy 2 sản phẩm tương ứng và tạo bảng so sánh Markdown.
        """
        # Thử tìm các sản phẩm bằng semantic search
        products, _ = search_service.semantic_search(query=query, top_k=2)

        if len(products) < 2:
            # Fallback scroll nếu không đủ sản phẩm so sánh
            scroll_results, _ = qdrant_service.client.scroll(
                collection_name=qdrant_service.collection_name,
                limit=2,
                with_payload=True
              )
            products = [
                {
                    "product_id": r.payload.get("product_id"),
                    "title": r.payload.get("title"),
                    "description": r.payload.get("description"),
                    "category": r.payload.get("category"),
                    "price": r.payload.get("price"),
                    "image_url": r.payload.get("image_url")
                }
                for r in scroll_results
            ]

        if len(products) >= 2:
            p1, p2 = products[0], products[1]
            comparison_md = (
                f"Dưới đây là bảng so sánh chi tiết giữa **{p1.get('title')}** và **{p2.get('title')}**:\n\n"
                f"| Tiêu chí | {p1.get('title')} | {p2.get('title')} |\n"
                f"| :--- | :--- | :--- |\n"
                f"| **Giá bán** | ${p1.get('price')} | ${p2.get('price')} |\n"
                f"| **Danh mục** | {p1.get('category')} | {p2.get('category')} |\n"
                f"| **Mô tả** | {p1.get('description')[:80]}... | {p2.get('description')[:80]}... |\n\n"
                f"Nhìn chung, **{p1.get('title')}** sẽ phù hợp nếu bạn ưu tiên mức giá và danh mục **{p1.get('category')}**, "
                f"trong khi **{p2.get('title')}** là lựa chọn tuyệt vời cho các tính năng khác. Bạn thích sản phẩm nào hơn?"
            )
            return comparison_md, [p1, p2]

        return "Tôi chưa tìm thấy đủ sản phẩm trong cơ sở dữ liệu để thực hiện so sánh kĩ lưỡng. Bạn vui lòng thử lại với tên sản phẩm cụ thể hơn.", []

chatbot_service = ChatbotService()
