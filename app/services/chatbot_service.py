import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import urllib.request
import re

from app.services.search_service import search_service
from app.services.qdrant_service import qdrant_service
from app.services.embedding_service import embedding_service
from app.services.rerank_service import rerank_service
from app.services.redis_service import redis_service
from app.config import settings

logger = logging.getLogger(__name__)

class ChatbotService:

    def chat_response(
        self,
        messages: List[Dict[str, str]],
        user_id: str = None,
        session_id: str = None,
        user_profile: Dict = None
    ) -> Tuple[str, List[Dict]]:
        """
        Xử lý tin nhắn chat của người dùng bằng luồng RAG động & Tool Calling.
        Sử dụng bộ nhớ đệm Redis cho lịch sử hội thoại.
        """
        if not messages:
            return "Xin chào! Tôi là trợ lý mua sắm AI của Nivora. Tôi có thể giúp gì cho bạn hôm nay?", []

        last_message = messages[-1].get("content", "").strip()
        logger.info(f"Chatbot received message: '{last_message}' from user={user_id}/session={session_id}")

        # 1. Quản lý hội thoại qua Redis Cache (Memory)
        mem_id = user_id if user_id else session_id
        chat_history = []
        if mem_id:
            cached_mem = redis_service.get_cached_chat_memory(mem_id)
            if cached_mem:
                chat_history = cached_mem

        # Tích hợp tin nhắn mới từ Spring Boot
        # Spring Boot chuyển qua lịch sử hiện tại, ta giới hạn 10 tin nhắn gần nhất để làm ngữ cảnh
        chat_history = messages[-10:]

        # 2. Phân tích ý định & trích lọc bộ lọc (LLM Query Understanding)
        parsed = self._understand_query(last_message)
        intent = parsed.get("intent", "chitchat")
        
        products = []
        # Quyết định gọi công cụ tìm kiếm (Tool Calling Decision)
        if intent in ["product_search", "comparison"]:
            products = self.search_products(
                query=last_message,
                category=parsed.get("category"),
                color=parsed.get("color"),
                min_price=parsed.get("min_price"),
                max_price=parsed.get("max_price")
            )

        # 3. Xây dựng ngữ cảnh Prompt (RAG Context Builder)
        system_prompt = (
            "Bạn là Trợ lý Mua sắm AI của Nivora - hệ thống thương mại điện tử chuyên cung cấp đồ nội thất cao cấp (bàn, ghế, tủ, giường, kệ,...).\n"
            "Nhiệm vụ của bạn:\n"
            "1. Trả lời khách hàng một cách lịch sự, thân thiện, súc tích và chuyên nghiệp bằng tiếng Việt.\n"
            "2. Sử dụng định dạng Markdown (in đậm, danh sách, bảng biểu, xuống dòng) để câu trả lời dễ đọc.\n"
            "3. Khi so sánh sản phẩm, BẮT BUỘC sử dụng bảng Markdown (ví dụ: | Tiêu chí | Sản phẩm A | Sản phẩm B |) để khách hàng dễ so sánh giá và tính năng.\n"
            "4. KHÔNG tự bịa ra thông tin sản phẩm hoặc giá bán nếu không có trong ngữ cảnh. Hãy sử dụng danh sách sản phẩm thật được cung cấp bên dưới.\n"
            "5. Nếu khách hàng hỏi về sản phẩm không có sẵn, hãy tư vấn nhẹ nhàng và giới thiệu các sản phẩm tương tự đang có.\n"
        )

        if user_profile:
            system_prompt += (
                f"\nThông tin hồ sơ khách hàng đang trò chuyện:\n"
                f"- Họ tên: {user_profile.get('name', 'Khách hàng')}\n"
                f"- Giới tính: {user_profile.get('gender', 'Chưa cập nhật')}\n"
                f"- Độ tuổi: {user_profile.get('age', 'Chưa cập nhật')}\n"
                f"- Nghề nghiệp: {user_profile.get('occupation', 'Chưa cập nhật')}\n"
                f"Hãy cá nhân hóa cách xưng hô (ví dụ: anh, chị, bạn) và đưa ra gợi ý phù hợp với thông tin này.\n"
            )

        if products:
            system_prompt += (
                "\nDưới đây là danh sách sản phẩm THẬT từ cơ sở dữ liệu phù hợp với yêu cầu hiện tại. "
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

        # 4. Gửi yêu cầu đến LLM
        llm_response = None
        if settings.GEMINI_API_KEY:
            logger.info("Calling Gemini for chatbot response")
            llm_response = self._call_gemini(system_prompt, chat_history)
        elif settings.OPENAI_API_KEY:
            logger.info("Calling OpenAI for chatbot response")
            llm_response = self._call_openai(system_prompt, chat_history)

        # 5. Phản hồi Fallback nếu không có API keys hoặc lỗi LLM
        if not llm_response:
            logger.info("No LLM key configured or LLM failed. Using rule-based fallback response.")
            if intent == "comparison" and len(products) >= 2:
                llm_response, _ = self._handle_comparison(last_message)
            elif products:
                llm_response = (
                    f"Dựa trên yêu cầu của bạn, tôi đã tìm thấy một số sản phẩm phù hợp tại Nivora:\n\n"
                )
                for idx, p in enumerate(products, 1):
                    desc = p.get("description", "")
                    if desc and len(desc) > 120:
                        desc = desc[:120] + "..."
                    llm_response += (
                        f"**{idx}. {p.get('title')}**\n"
                        f"- **Giá**: ${p.get('price')}\n"
                        f"- **Danh mục**: {p.get('category')}\n"
                        f"- **Mô tả**: {desc}\n\n"
                    )
                llm_response += "Bạn có muốn tôi tư vấn chi tiết hơn về sản phẩm nào ở trên không?"
            else:
                llm_response = (
                    "Hiện tại tôi chưa tìm thấy sản phẩm nào khớp chính xác với mô tả của bạn. "
                    "Bạn có thể mô tả chi tiết hơn về nhu cầu sử dụng, màu sắc hoặc khoảng giá mong muốn không? "
                    "(Ví dụ: 'Tôi cần tìm bàn học dưới 2 triệu')."
                )

        # 6. Lưu lịch sử mới cập nhật vào Redis
        if mem_id:
            updated_mem = list(chat_history)
            updated_mem.append({"role": "assistant", "content": llm_response})
            redis_service.set_cached_chat_memory(mem_id, updated_mem[-10:])

        return llm_response, products

    def search_products(
        self,
        query: str,
        category: Optional[str] = None,
        color: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[Dict]:
        """
        [Search Tool]
        Thực hiện tìm kiếm sản phẩm kết hợp Qdrant Filter và Semantic Rerank.
        """
        logger.info(f"Executing search_products tool: query='{query}', category='{category}', color='{color}', price=({min_price}, {max_price})")
        
        # 1. Nhận Vector Embedding của câu truy vấn (có cache)
        query_vector = embedding_service.embed_query(query)

        # 2. Thực hiện tìm kiếm với bộ lọc cứng
        results = qdrant_service.search(
            query_vector=query_vector,
            category_filter=category,
            price_range=(min_price, max_price),
            tag_filter=color,
            top_k=30
        )

        # 3. Fallback bộ lọc nếu kết quả rỗng
        if not results and color:
            logger.info("Qdrant filters too strict on tag (color). Retrying without color filter...")
            results = qdrant_service.search(
                query_vector=query_vector,
                category_filter=category,
                price_range=(min_price, max_price),
                top_k=30
            )

        if not results and category:
            logger.info("Qdrant filters too strict on category. Retrying with price filter only...")
            results = qdrant_service.search(
                query_vector=query_vector,
                price_range=(min_price, max_price),
                top_k=30
            )

        if not results:
            logger.info("Qdrant search returned 0 results. Falling back to pure semantic search...")
            results = qdrant_service.search(
                query_vector=query_vector,
                top_k=30
            )

        # 4. Reranking qua Cross-Encoder
        reranked = rerank_service.rerank(
            query=query,
            documents=results,
            top_k=10,
            document_key="search_document"
        )
        return reranked

    def _understand_query(self, query: str) -> Dict[str, Any]:
        """
        [Query Understanding Tool]
        Sử dụng LLM (hoặc Regex fallback) để parse thông tin từ câu hỏi người dùng thành JSON.
        """
        system_prompt = (
            "You are an expert query parser for Nivora ecommerce store.\n"
            "Analyze the user's Vietnamese request and extract search fields.\n"
            "Output ONLY a raw JSON object (no markdown, no ```json formatting). Valid fields:\n"
            "{\n"
            '  "intent": "product_search" (searching products) | "comparison" (comparing items) | "greeting" | "chitchat",\n'
            '  "category": leaf category name in Vietnamese (e.g. "ghế xoay", "bàn học", "sofa", etc. or null),\n'
            '  "color": color keyword in Vietnamese (e.g. "trắng", "đen", "xám", "nâu", etc. or null),\n'
            '  "min_price": minimum price limit in VND as float or null,\n'
            '  "max_price": maximum price limit in VND as float or null (convert "500k" -> 500000, "2 triệu" -> 2000000),\n'
            '  "purpose": usage environment/style (e.g. "văn phòng", "phòng ngủ", "phòng khách", "phỏng vấn" or null)\n'
            "}"
        )

        parsed = None
        messages = [{"role": "user", "content": query}]
        llm_json = None

        if settings.GEMINI_API_KEY:
            llm_json = self._call_gemini(system_prompt, messages)
        elif settings.OPENAI_API_KEY:
            llm_json = self._call_openai(system_prompt, messages)

        if llm_json:
            try:
                clean_json = llm_json.strip()
                if clean_json.startswith("```"):
                    clean_json = clean_json.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                logger.info(f"Query understood by LLM: {parsed}")
            except Exception as e:
                logger.warning(f"Failed to parse JSON query understanding from LLM: {e}. Output was: {llm_json}")

        if not parsed:
            # Fallback Rule-based parser
            logger.info("Using Rule-based parser fallback.")
            parsed = self._rule_based_parse(query)

        return parsed

    def _rule_based_parse(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        
        intent = "chitchat"
        if any(w in q for w in ["tìm", "mua", "bán", "ghế", "bàn", "tủ", "kệ", "giường", "sofa", "nội thất", "giá", "bao nhiêu", "đắt", "rẻ"]):
            intent = "product_search"
        if any(w in q for w in ["so sánh", "khác biệt", "khác nhau"]):
            intent = "comparison"
        if any(w in q for w in ["chào", "hello", "hi", "tư vấn"]):
            if intent != "product_search" and intent != "comparison":
                intent = "greeting"

        min_price = None
        max_price = None

        # Pattern: dưới X triệu
        m_trieu = re.search(r'(?:dưới|thấp hơn|<|tầm|duoi)\s*([\d\.,]+)\s*(?:triệu|tr|t)', q)
        if m_trieu:
            try:
                val = float(m_trieu.group(1).replace(',', '.'))
                max_price = int(val * 1000000)
            except ValueError:
                pass

        # Pattern: dưới X k
        m_k = re.search(r'(?:dưới|thấp hơn|<|tầm|duoi)\s*([\d\.,]+)\s*(?:k|nghìn|ngàn|trăm)', q)
        if m_k and not max_price:
            try:
                val = float(m_k.group(1).replace(',', '.'))
                if "trăm" in q:
                    max_price = int(val * 100000)
                else:
                    max_price = int(val * 1000)
            except ValueError:
                pass

        # Color extraction
        colors = ["trắng", "đen", "đỏ", "xanh", "vàng", "nâu", "xám", "hồng", "gỗ", "kem"]
        color = None
        for col in colors:
            if col in q:
                color = col
                break

        # Category extraction
        categories = ["áo sơ mi", "áo thun", "ghế xoay", "ghế văn phòng", "ghế gỗ", "ghế", "bàn học", "bàn làm việc", "bàn", "tủ quần áo", "tủ", "giường", "sofa", "kệ"]
        category = None
        for cat in categories:
            if cat in q:
                category = cat
                break

        purpose = None
        for purp in ["phòng ngủ", "phòng khách", "văn phòng", "làm việc", "học tập", "phỏng vấn"]:
            if purp in q:
                purpose = purp
                break

        return {
            "intent": intent,
            "category": category,
            "color": color,
            "min_price": min_price,
            "max_price": max_price,
            "purpose": purpose
        }

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
        products, _ = search_service.semantic_search(query=query, top_k=2)

        if len(products) < 2:
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
