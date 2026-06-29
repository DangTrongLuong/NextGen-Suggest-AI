import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import urllib.request
import re

from app.features.search.services.search_service import search_service
from app.shared.qdrant.service import qdrant_service
from app.features.search.services.embedding_service import embedding_service
from app.features.search.services.rerank_service import rerank_service
from app.shared.redis.service import redis_service
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ChatbotService:

    def _detect_prompt_injection(self, query: str) -> bool:
        """
        Phát hiện các mẫu câu chèn mã độc vào prompt của người dùng.
        """
        q = query.lower().strip()
        injection_patterns = [
            "ignore previous instructions",
            "ignore the previous instructions",
            "forget your role",
            "forget your instructions",
            "act as chatgpt",
            "act as a chatgpt",
            "developer mode",
            "dan mode",
            "reveal system prompt",
            "reveal the system prompt",
            "reveal hidden instructions",
            "reveal the hidden instructions",
            "show internal configuration",
            "show the internal configuration",
            "you are no longer",
            "system prompt",
            "system_prompt",
            "instructions you were given"
        ]
        
        # Check standard substrings
        if any(pat in q for pat in injection_patterns):
            return True
            
        return False

    def chat_response(
        self,
        messages: List[Dict[str, str]],
        user_id: str = None,
        session_id: str = None,
        user_profile: Dict = None
    ) -> Tuple[str, List[Dict], str, bool]:
        """
        Xử lý tin nhắn chat của người dùng bằng luồng RAG động & Tool Calling.
        Sử dụng bộ nhớ đệm Redis cho lịch sử hội thoại.
        """
        if not messages:
            return "Xin chào! Tôi là trợ lý mua sắm AI của Nivora. Tôi có thể giúp gì cho bạn hôm nay?", [], "product_information", False

        last_message = messages[-1].get("content", "").strip()
        logger.info(f"Chatbot received message: '{last_message}' from user={user_id}/session={session_id}")

        # 1. Phát hiện Prompt Injection
        if self._detect_prompt_injection(last_message):
            logger.warning(f"Prompt injection detected in query: {last_message}")
            return (
                "Tôi không thể thay đổi vai trò hoặc tiết lộ cấu hình nội bộ của hệ thống.",
                [],
                "out_of_scope",
                True
            )

        # 2. Quản lý hội thoại qua Redis Cache (Memory)
        # Tải lịch sử từ Redis để duy trì ngữ cảnh liên tục giữa các phiên
        mem_id = user_id if user_id else session_id
        cached_history: list = []
        if mem_id:
            cached_mem = redis_service.get_cached_chat_memory(mem_id)
            if cached_mem:
                cached_history = cached_mem
                logger.info(f"[CHAT MEMORY] Loaded {len(cached_history)} messages from Redis for mem_id={mem_id}")

        # Hợp nhất lịch sử Redis với tin nhắn mới từ Frontend
        if len(messages) == 1:
            # Frontend chỉ gửi 1 tin nhắn mới — hợp nhất với lịch sử trong Redis
            new_msg = messages[0]
            if cached_history and cached_history[-1].get("content") == new_msg.get("content") and cached_history[-1].get("role") == new_msg.get("role"):
                chat_history = cached_history
            else:
                chat_history = cached_history + messages
        else:
            # Frontend gửi toàn bộ lịch sử cuộc hội thoại hoạt động — dùng trực tiếp
            chat_history = messages

        # Chỉ giữ tối đa 12 tin nhắn cuối (6 lượt user-assistant = 6 cuộc hội thoại)
        chat_history = chat_history[-12:]
        logger.info(f"[CHAT MEMORY] Using {len(chat_history)} messages as context for LLM")

        # 3. Phân tích ý định & trích lọc bộ lọc (LLM Query Understanding)
        parsed = self._understand_query(chat_history)
        intent = parsed.get("intent", "product_information")
        is_out_of_scope = parsed.get("is_out_of_scope", False) or (intent == "out_of_scope")

        if is_out_of_scope:
            response_msg = (
                "Tôi là trợ lý tư vấn sản phẩm của hệ thống. Nhiệm vụ của tôi là hỗ trợ tìm kiếm, "
                "so sánh và đề xuất sản phẩm phù hợp với nhu cầu của bạn. Tôi không được thiết kế "
                "để trả lời các câu hỏi ngoài lĩnh vực sản phẩm."
            )
            if mem_id:
                updated_mem = list(chat_history)
                updated_mem.append({"role": "assistant", "content": response_msg})
                redis_service.set_cached_chat_memory(mem_id, updated_mem[-12:])
            return response_msg, [], "out_of_scope", True

        products = []
        # Quyết định gọi công cụ tìm kiếm
        if intent in ["search_product", "compare_product", "recommend_product"]:
            products = self.search_products(
                query=last_message,
                category=parsed.get("category"),
                color=parsed.get("color"),
                min_price=parsed.get("min_price"),
                max_price=parsed.get("max_price")
            )

        # 4. Xây dựng ngữ cảnh Prompt (RAG Context Builder)
        system_prompt = (
            "Bạn là Trợ lý Mua sắm AI của Nivora - hệ thống thương mại điện tử chuyên cung cấp đồ nội thất cao cấp (bàn, ghế, tủ, giường, kệ,...).\n"
            "Nhiệm vụ của bạn:\n"
            "1. Chỉ hỗ trợ tư vấn sản phẩm đồ nội thất của cửa hàng. Không trả lời các câu hỏi ngoài lĩnh vực sản phẩm.\n"
            "2. Trả lời khách hàng một cách lịch sự, thân thiện, súc tích và chuyên nghiệp bằng tiếng Việt.\n"
            "3. Sử dụng định dạng Markdown (in đậm, danh sách, bảng biểu, xuống dòng) để câu trả lời dễ đọc.\n"
            "4. Khi so sánh sản phẩm, BẮT BUỘC sử dụng bảng Markdown (ví dụ: | Tiêu chí | Sản phẩm A | Sản phẩm B |) để khách hàng dễ so sánh giá và tính năng.\n"
            "5. KHÔNG tự bịa ra thông tin sản phẩm hoặc giá bán nếu không có trong ngữ cảnh. Hãy sử dụng danh sách sản phẩm thật được cung cấp bên dưới.\n"
            "6. Nếu khách hàng hỏi về sản phẩm không có sẵn, hãy tư vấn nhẹ nhàng và giới thiệu các sản phẩm tương tự đang có.\n"
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

        # 5. Gửi yêu cầu đến LLM
        llm_response = None
        if settings.GEMINI_API_KEY:
            logger.info("Calling Gemini for chatbot response")
            llm_response = self._call_gemini(system_prompt, chat_history)
        elif settings.OPENAI_API_KEY:
            logger.info("Calling OpenAI for chatbot response")
            llm_response = self._call_openai(system_prompt, chat_history)

        # 6. Phản hồi Fallback nếu không có API keys hoặc lỗi LLM
        if not llm_response:
            logger.info("No LLM key configured or LLM failed. Using rule-based fallback response.")
            if intent == "compare_product" and len(products) >= 2:
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

        # 7. Lưu lịch sử mới cập nhật vào Redis
        if mem_id:
            updated_mem = list(chat_history)
            updated_mem.append({"role": "assistant", "content": llm_response})
            redis_service.set_cached_chat_memory(mem_id, updated_mem[-12:])

        return llm_response, products, intent, False

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
        
        try:
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
        except Exception as e:
            logger.error(f"Search products via Qdrant/Embedding failed: {e}. Falling back to MySQL keyword search.", exc_info=True)
            return self._fallback_mysql_search(query, category, color, min_price, max_price)

    def _fallback_mysql_search(
        self,
        query: str,
        category: Optional[str] = None,
        color: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[Dict]:
        from app.features.recommendation.services.recommendation_service import get_mysql_connection
        conn = None
        try:
            conn = get_mysql_connection()
            with conn.cursor() as cursor:
                conditions = ["p.deleted_at IS NULL", "p.is_active = 1"]
                params = []

                if query and query.strip():
                    keywords = query.strip().split()
                    kw_conditions = []
                    for kw in keywords:
                        kw_conditions.append("(p.name LIKE %s OR p.description LIKE %s OR p.brand LIKE %s)")
                        like_param = f"%{kw}%"
                        params.extend([like_param, like_param, like_param])
                    conditions.append(f"({' AND '.join(kw_conditions)})")

                if category:
                    conditions.append("c.name LIKE %s")
                    params.append(f"%{category}%")

                if color:
                    conditions.append("(p.name LIKE %s OR p.description LIKE %s)")
                    like_color = f"%{color}%"
                    params.extend([like_color, like_color])

                query_sql = f"""
                SELECT p.uuid as product_id, p.name as title, p.description, 
                       c.name as category, p.image_url, p.brand,
                       COALESCE((SELECT MIN(v.price) FROM product_variants v WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 0) as price
                FROM products p
                LEFT JOIN categories c ON p.category_id = c.id
                WHERE {" AND ".join(conditions)}
                LIMIT 30;
                """
                cursor.execute(query_sql, tuple(params))
                rows = cursor.fetchall()

                results = []
                for r in rows:
                    r["price"] = float(r["price"])
                    r["tags"] = [color] if color else []
                    r["score"] = 0.5
                    r["reason"] = "Gợi ý tương tự"
                    r["search_document"] = f"{r['title']} {r['description']}"

                    if min_price is not None and r["price"] < min_price:
                        continue
                    if max_price is not None and r["price"] > max_price:
                        continue

                    results.append(r)
                return results[:10]
        except Exception as e:
            logger.error(f"Fallback MySQL chatbot search failed: {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()

    def _understand_query(self, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        [Query Understanding Tool]
        Sử dụng LLM (hoặc Regex fallback) để parse thông tin từ câu hỏi người dùng trong ngữ cảnh cuộc hội thoại thành JSON.
        """
        system_prompt = (
            "You are an expert query parser and intent classifier for Nivora ecommerce store (specializing in furniture, chairs, tables, beds, cabinets, sofas, etc.).\n"
            "Analyze the user's latest request in the context of the conversation history, and output ONLY a raw JSON object (no markdown, no ```json formatting).\n"
            "A query is out of scope if it is NOT related to finding, comparing, recommending furniture products, store policies, or friendly greeting/chitchat related to the shop. Topics like coding, math, general knowledge, politics, religion, medical, law, history, etc. are OUT-OF-SCOPE.\n\n"
            "Instructions:\n"
            "1. Identify the intent of the LATEST user message.\n"
            "2. If the user refers to previous items (e.g. 'màu đỏ', 'dưới 2 triệu', 'mẫu thứ nhất'), look at the conversation history to resolve the context (e.g. if previous message was about 'ghế xoay', then category is 'ghế xoay').\n\n"
            "Valid JSON format:\n"
            "{\n"
            '  "intent": "search_product" (searching specific products) | "compare_product" (comparing items) | "recommend_product" (recommending based on space/needs) | "product_information" (asking about store hours, policies, or general product details) | "out_of_scope" (questions about general knowledge, coding, math, politics, religion, history, medical, law, or anything unrelated to the product/store domain),\n'
            '  "is_out_of_scope": true if the query is out of scope else false,\n'
            '  "category": leaf category name in Vietnamese (e.g. "ghế xoay", "bàn học", "sofa", etc. or null),\n'
            '  "color": color keyword in Vietnamese (e.g. "trắng", "đen", "xám", "nâu", etc. or null),\n'
            '  "min_price": minimum price limit in VND as float or null,\n'
            '  "max_price": maximum price limit in VND as float or null (convert "500k" -> 500000, "2 triệu" -> 2000000),\n'
            '  "purpose": usage environment/style (e.g. "văn phòng", "phòng ngủ", "phòng khách" or null)\n'
            "}"
        )

        parsed = None
        llm_json = None

        if settings.GEMINI_API_KEY:
            llm_json = self._call_gemini(system_prompt, chat_history)
        elif settings.OPENAI_API_KEY:
            llm_json = self._call_openai(system_prompt, chat_history)

        if llm_json:
            try:
                clean_json = llm_json.strip()
                if clean_json.startswith("```"):
                    clean_json = clean_json.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                logger.info(f"Query understood by LLM with context: {parsed}")
            except Exception as e:
                logger.warning(f"Failed to parse JSON query understanding from LLM: {e}. Output was: {llm_json}")

        if not parsed:
            # Fallback Rule-based parser
            last_message = chat_history[-1].get("content", "") if chat_history else ""
            logger.info("Using Rule-based parser fallback.")
            parsed = self._rule_based_parse(last_message)

        return parsed

    def _rule_based_parse(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        
        # Simple out of scope checks
        coding_keywords = ["viết code", "lập trình", "java", "python", "javascript", "c++", "html", "css", "sql"]
        math_keywords = ["giải bài toán", "phương trình", "toán học", "tích phân", "đạo hàm"]
        general_keywords = ["tổng thống", "thủ tướng", "chính trị", "tôn giáo", "y khoa", "bác sĩ", "luật pháp", "hiến pháp", "thủ đô", "lịch sử"]
        
        is_out_of_scope = False
        intent = "product_information"
        
        if any(k in q for k in coding_keywords) or any(k in q for k in math_keywords) or any(k in q for k in general_keywords) or re.search(r'\d+\s*[+\-*/=]\s*\d+', q):
            is_out_of_scope = True
            intent = "out_of_scope"
        else:
            if any(w in q for w in ["tìm", "mua", "bán", "ghế", "bàn", "tủ", "kệ", "giường", "sofa", "nội thất"]):
                intent = "search_product"
            elif any(w in q for w in ["so sánh", "khác biệt", "khác nhau"]):
                intent = "compare_product"
            elif any(w in q for w in ["gợi ý", "tư vấn", "khuyên dùng"]):
                intent = "recommend_product"
            elif any(w in q for w in ["chào", "hello", "hi"]):
                intent = "product_information"

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
            "is_out_of_scope": is_out_of_scope,
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
