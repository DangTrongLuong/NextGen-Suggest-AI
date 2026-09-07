import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import urllib.request
import re

from app.features.search.services.search_service import search_service, clean_conversational_query
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
            # Let the LLM handle the out of scope response if API keys are available,
            # so it can provide a friendly, context-aware reply.
            if not (settings.GEMINI_API_KEY or settings.OPENAI_API_KEY):
                response_msg = (
                    "Dạ, tôi là trợ lý ảo mua sắm tại Nivora. "
                    "Chủ đề bạn vừa hỏi nằm ngoài phạm vi hỗ trợ của tôi. "
                    "Bạn có cần tôi tư vấn về các sản phẩm như nội thất, sách, mỹ phẩm, thời trang, đồ gia dụng hay thiết bị công nghệ không ạ?"
                )
                if mem_id:
                    updated_mem = list(chat_history)
                    updated_mem.append({"role": "assistant", "content": response_msg})
                    redis_service.set_cached_chat_memory(mem_id, updated_mem[-12:])
                return response_msg, [], "out_of_scope", True

        products = []
        cleaned_query = clean_conversational_query(last_message)

        # Quyết định gọi công cụ tìm kiếm
        if intent in ["search_product", "compare_product", "recommend_product"]:
            products = self.search_products(
                query=cleaned_query,
                category=parsed.get("category"),
                color=parsed.get("color"),
                min_price=parsed.get("min_price"),
                max_price=parsed.get("max_price")
            )

            # If strict category/filter search returned 0 products, retry pure semantic search
            if not products or len(products) == 0:
                logger.info(f"Strict search returned 0 results for '{cleaned_query}'. Retrying pure vector semantic search...")
                try:
                    sem_prods, _ = search_service.semantic_search(cleaned_query, top_k=10)
                    if sem_prods:
                        products = self._filter_existing_products(sem_prods)
                except Exception as sem_err:
                    logger.warning(f"Fallback semantic search error: {sem_err}")

        # 4. Xây dựng ngữ cảnh Prompt linh hoạt theo Ý Định (Intent-driven Dynamic RAG Context Builder)
        base_role = (
            "Bạn là Trợ lý Mua sắm AI thông minh & tận tâm của Nivora - hệ thống thương mại điện tử đa ngành (Nội thất, Sách, Mỹ phẩm, Thời trang, Đồ gia dụng, và Công nghệ).\n"
            "Mục tiêu tối thượng của bạn là đóng vai CHUYÊN VIÊN TƯ VẤN MUA SẮM cá nhân hóa (Personal Shopping Advisor), mang lại trải nghiệm tư vấn chu đáo, thấu hiểu nhu cầu và khéo léo thay vì chỉ liệt kê danh sách khô cứng.\n"
            "Quy tắc phản hồi chung:\n"
            "- Phản hồi lịch sự, ấm áp, xưng 'tôi' hoặc 'Nivora' và gọi khách bằng 'bạn' (hoặc xưng hô theo hồ sơ nếu có).\n"
            "- Sử dụng định dạng Markdown (in đậm, danh sách bullet, bảng biểu) sinh động.\n"
            "- KHÔNG tự bịa ra sản phẩm hoặc giá bán không có trong danh sách được cung cấp bên dưới.\n"
            "- Thẻ sản phẩm UI chi tiết sẽ tự động hiển thị ở phía dưới khung chat, do đó trong lời nhắn của bạn, hãy tập trung vào LẬP LUẬN TƯ VẤN, NÊU LÝ DO PHÙ HỢP và GỢI Ý ĐỂ KHÁCH HÀNG DỄ RA QUYẾT ĐỊNH.\n\n"
        )

        if intent == "recommend_product":
            intent_instruction = (
                "HƯỚNG DẪN DÀNH CHO KỊCH BẢN TƯ VẤN & ĐỀ XUẤT (RECOMMENDATION):\n"
                "1. Hãy phân tích súc tích nhu cầu/tiêu chí của khách hàng (ví dụ: công năng, giá cả, phong cách, độ tiện dụng).\n"
                "2. Chọn ra 2-3 sản phẩm tiêu biểu nhất từ danh sách bên dưới để phân tích sâu:\n"
                "   - Phân tích điểm mạnh/lý do vì sao sản phẩm này lại hợp với nhu cầu đó.\n"
                "   - Phân loại rõ ràng (ví dụ: 'Lựa chọn tối ưu chi phí', 'Lựa chọn cao cấp/toàn diện').\n"
                "3. Đưa ra LỜI KHUYÊN CHỐT ngắn gọn (ví dụ: 'Nếu bạn ưu tiên X thì nên chọn A, nếu cần Y thì chọn B').\n"
                "4. Nhắc nhẹ khách hàng nhấp vào Thẻ sản phẩm phía dưới để xem ảnh và chọn mua.\n"
            )
        elif intent == "compare_product":
            intent_instruction = (
                "HƯỚNG DẪN DÀNH CHO KỊCH BẢN SO SÁNH SẢN PHẨM (COMPARISON):\n"
                "1. BẮT BUỘC sử dụng BẢNG MARKDOWN để so sánh trực quan các sản phẩm (Cột: Tiêu chí, Sản phẩm A, Sản phẩm B...).\n"
                "2. Phân tích ưu/nhược điểm và điểm khác biệt cốt lõi giữa các lựa chọn.\n"
                "3. Kết luận rõ ràng đối tượng/nhu cầu nào phù hợp với sản phẩm nào.\n"
            )
        elif intent == "search_product":
            intent_instruction = (
                "HƯỚNG DẪN DÀNH CHO KỊCH BẢN TÌM KIẾM SẢN PHẨM (SEARCH):\n"
                "1. Phản hồi súc tích, xác nhận đã tìm thấy các mẫu sản phẩm khớp yêu cầu.\n"
                "2. Tóm tắt 2-4 mẫu nổi bật kèm điểm nhấn chính (giá bán, đặc tính chính).\n"
                "3. Hướng dẫn khách hàng click vào thẻ sản phẩm bên dưới để xem hình ảnh và chi tiết.\n"
            )
        elif is_out_of_scope:
            intent_instruction = (
                "HƯỚNG DẪN DÀNH CHO KỊCH BẢN NGOÀI PHẠM VI (OUT OF SCOPE):\n"
                "Yêu cầu của khách hàng nằm ngoài các ngành hàng của cửa hàng (như viết code, giải toán, chính trị, y khoa...).\n"
                "Hãy phản hồi cực kỳ lịch sự, khéo léo giải thích cửa hàng chuyên cung cấp các sản phẩm thuộc các ngành hàng (Nội thất, Sách, Mỹ phẩm, Thời trang, Đồ gia dụng, Công nghệ) và nhiệt tình gợi ý hỗ trợ tìm kiếm sản phẩm mua sắm liên quan.\n"
            )
        else:
            intent_instruction = (
                "HƯỚNG DẪN DÀNH CHO KỊCH BẢN TRỢ GIÚP THÔNG TIN (PRODUCT INFORMATION / FAQ):\n"
                "Phản hồi rõ ràng, nhiệt tình giải đáp thắc mắc về chính sách, dịch vụ hoặc thông tin chung của cửa hàng.\n"
            )

        system_prompt = base_role + intent_instruction

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
                f"\nDANH SÁCH {len(products)} SẢN PHẨM THẬT TỪ CƠ SỞ DỮ LIỆU ĐƯỢC TÌM THẤY:\n"
            )
            for idx, p in enumerate(products, 1):
                desc = p.get("description", "")
                if desc and len(desc) > 250:
                    desc = desc[:250] + "..."
                system_prompt += (
                    f"Sản phẩm {idx}:\n"
                    f"- SKU / ID: {p.get('product_id')}\n"
                    f"- Tên sản phẩm: {p.get('title')}\n"
                    f"- Giá: {p.get('price'):,.0f} VNĐ\n"
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
                llm_response, _ = self._handle_comparison(cleaned_query)
            elif intent == "recommend_product" and products and len(products) > 0:
                p1 = products[0]
                p2 = products[1] if len(products) > 1 else None
                p1_price = f"{float(p1.get('price', 0)):,.0f} VNĐ" if p1.get("price") else "Liên hệ"
                llm_response = (
                    f"Chào bạn, dựa trên nhu cầu **{cleaned_query}** của bạn, Nivora xin đưa ra lời khuyên tư vấn như sau:\n\n"
                    f"🌟 **Lựa chọn nổi bật nhất**: **{p1.get('title')}**\n"
                    f"- **Mức giá**: {p1_price}\n"
                    f"- **Lý do đề xuất**: {p1.get('description', '')[:160]}...\n\n"
                )
                if p2:
                    p2_price = f"{float(p2.get('price', 0)):,.0f} VNĐ" if p2.get("price") else "Liên hệ"
                    llm_response += (
                        f"🔹 **Lựa chọn thay thế phù hợp**: **{p2.get('title')}**\n"
                        f"- **Mức giá**: {p2_price}\n"
                        f"- **Lý do đề xuất**: {p2.get('description', '')[:160]}...\n\n"
                    )
                llm_response += (
                    "💡 **Lời khuyên**: Bạn có thể dựa vào mức giá và tính năng để chọn sản phẩm ưng ý nhất. "
                    "Mời bạn nhấp trực tiếp vào các thẻ sản phẩm hiển thị ngay phía dưới để xem hình ảnh chi tiết và đặt hàng nhé!"
                )
            elif products and len(products) > 0:
                llm_response = (
                    f"Dưới đây là các sản phẩm **{cleaned_query}** nổi bật tại Nivora dành cho bạn:\n\n"
                )
                for idx, p in enumerate(products[:4], 1):
                    desc = p.get("description", "")
                    if desc and len(desc) > 100:
                        desc = desc[:100] + "..."
                    price_str = f"{float(p.get('price', 0)):,.0f} VNĐ" if p.get("price") else "Liên hệ"
                    llm_response += (
                        f"• **{p.get('title')}** - Mức giá: {price_str}\n  _{desc}_\n\n"
                    )
                llm_response += "Mời bạn tham khảo thông tin chi tiết và đặt hàng qua các thẻ sản phẩm hiển thị ngay ở phía dưới nhé!"
            else:
                # If products list was empty, retry pure semantic search
                try:
                    fallback_prods, _ = search_service.semantic_search(cleaned_query, top_k=10)
                    if fallback_prods:
                        products = self._filter_existing_products(fallback_prods)
                except Exception as ex:
                    logger.warning(f"Semantic search fallback failed: {ex}")

                if products and len(products) > 0:
                    llm_response = (
                        f"Tôi đã tìm thấy **{len(products)} sản phẩm {cleaned_query}** nổi bật tại cửa hàng. "
                        f"Mời bạn tham khảo danh sách sản phẩm chi tiết ở phía dưới:"
                    )
                else:
                    llm_response = (
                        f"Hiện tại tôi chưa tìm thấy sản phẩm nào khớp với từ khóa '{last_message}'. "
                        "Bạn có thể thử các từ khóa khác như 'áo thun', 'áo sơ mi', 'bàn học', 'ghế xoay', 'son môi' nhé!"
                    )

        # 6. Extract ALL products mentioned in AI response OR return all retrieved products
        mentioned_products = []
        if llm_response and products:
            import re
            norm_response = re.sub(r'[^a-z0-9àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', '', llm_response.lower())

            for p in products:
                title = p.get("title", "")
                product_id = p.get("product_id", "")

                # Direct ID match
                if product_id and product_id.lower() in llm_response.lower():
                    mentioned_products.append(p)
                    continue

                # Title / model match
                if title:
                    pipe_parts = [x.strip() for x in title.split("|")]
                    specific_title = f"{pipe_parts[0]} {pipe_parts[1]}" if len(pipe_parts) >= 2 else pipe_parts[0]
                    short_title = specific_title.split(" - ")[0].split(" (")[0].strip()
                    norm_title = re.sub(r'[^a-z0-9àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', '', short_title.lower())
                    model_name = pipe_parts[1] if len(pipe_parts) >= 2 else ""
                    norm_model = re.sub(r'[^a-z0-9àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', '', model_name.lower()) if model_name else ""

                    if (len(norm_title) > 3 and norm_title in norm_response) or (norm_model and len(norm_model) >= 3 and norm_model in norm_response):
                        mentioned_products.append(p)

        # Return all products retrieved by search if string matching produced fewer items
        if not mentioned_products and products:
            mentioned_products = products

        # 7. Lưu lịch sử mới cập nhật vào Redis
        if mem_id:
            updated_mem = list(chat_history)
            updated_mem.append({"role": "assistant", "content": llm_response})
            redis_service.set_cached_chat_memory(mem_id, updated_mem[-12:])

        return llm_response, mentioned_products, intent, False

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

            # 2. Thực hiện tìm kiếm với bộ lọc cứng (top_k=15 để tăng tốc reranking)
            results = qdrant_service.search(
                query_vector=query_vector,
                category_filter=category,
                price_range=(min_price, max_price),
                tag_filter=color,
                top_k=15
            )

            # 3. Fallback bộ lọc nếu kết quả rỗng
            if not results and color:
                logger.info("Qdrant filters too strict on tag (color). Retrying without color filter...")
                results = qdrant_service.search(
                    query_vector=query_vector,
                    category_filter=category,
                    price_range=(min_price, max_price),
                    top_k=15
                )

            if not results and category:
                logger.info("Qdrant filters too strict on category. Retrying with price filter only...")
                results = qdrant_service.search(
                    query_vector=query_vector,
                    price_range=(min_price, max_price),
                    top_k=15
                )

            if not results:
                logger.info("Qdrant search returned 0 results. Falling back to pure semantic search...")
                results = qdrant_service.search(
                    query_vector=query_vector,
                    top_k=15
                )

            # 4. Reranking qua Cross-Encoder (dùng short_context để tối ưu hiệu năng CPU)
            reranked = rerank_service.rerank(
                query=query,
                documents=results,
                top_k=10,
                document_key="short_context"
            )
            # Lọc chỉ giữ lại các sản phẩm thực tế đang tồn tại trong MySQL
            return self._filter_existing_products(reranked)
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
            # Clean stop words for MySQL search
            stop_words_pat = re.compile(r'\b(tìm|mua|bán|cho|tôi|cần|bạn|giúp|vui lòng|mình|shop|hãy|được không|giới thiệu|gợi ý)\b', re.IGNORECASE)
            clean_q = stop_words_pat.sub('', query).strip()
            if not clean_q:
                clean_q = query

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
            "You are an expert query parser and intent classifier for Nivora ecommerce store (specializing in multiple categories: furniture/nội thất, books/sách, cosmetics/mỹ phẩm, fashion/thời trang, household appliances/đồ gia dụng, and technology/công nghệ).\n"
            "Analyze the user's latest request in the context of the conversation history, and output ONLY a raw JSON object (no markdown, no ```json formatting).\n"
            "Classification guidelines:\n"
            "- A query is in-scope if it searches for, compares, or asks for recommendations regarding any product under our domains: furniture, books, cosmetics, clothing/fashion, household items, or electronics/laptops/tech.\n"
            "- Do not mark queries about books, cosmetics, clothes, home appliances, or tech as out of scope. These are fully supported.\n"
            "- Only classify as 'out_of_scope' if it is completely unrelated to purchasing or seeking advice about products in these categories (e.g., writing programming code, solving math formulas, discussing politics/religion, or medical queries).\n\n"
            "Instructions:\n"
            "1. Identify the intent of the LATEST user message.\n"
            "2. If the user refers to previous items (e.g. 'màu đỏ', 'dưới 2 triệu', 'mẫu thứ nhất'), look at the conversation history to resolve the context.\n\n"
            "Valid JSON format:\n"
            "{\n"
            '  "intent": "search_product" (searching specific products) | "compare_product" (comparing items) | "recommend_product" (recommending based on space/needs) | "product_information" (asking about store hours, policies, or general product details) | "out_of_scope" (questions about coding, math, politics, religion, medical, law, or anything unrelated to any of our store categories),\n'
            '  "is_out_of_scope": true if the query is out of scope else false,\n'
            '  "category": leaf category name in Vietnamese (e.g. "ghế xoay", "bàn học", "sách", "truyện tranh", "son môi", "áo khoác", "nồi cơm điện", "laptop", etc. or null),\n'
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
        general_keywords = ["tổng thống", "thủ tướng", "chính trị", "tôn giáo", "y khoa", "bác sĩ", "luật pháp", "hiến pháp", "thủ đô"] # Removed "lịch sử"
        
        is_out_of_scope = False
        intent = "product_information"
        
        if any(k in q for k in coding_keywords) or any(k in q for k in math_keywords) or any(k in q for k in general_keywords) or re.search(r'\d+\s*[+\-*/=]\s*\d+', q):
            is_out_of_scope = True
            intent = "out_of_scope"
        else:
            if any(w in q for w in [
                "tìm", "mua", "bán", "ghế", "bàn", "tủ", "kệ", "giường", "sofa", "nội thất", 
                "sách", "truyện", "tiểu thuyết", "mỹ phẩm", "son", "phấn", "kem", "serum",
                "quần", "áo", "váy", "đầm", "giày", "dép", "thời trang",
                "nồi", "chảo", "gia dụng", "máy hút bụi", "lò vi sóng",
                "điện thoại", "laptop", "máy tính", "tai nghe", "công nghệ"
            ]):
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
        categories = [
            "ghế xoay", "ghế văn phòng", "ghế gỗ", "ghế", "bàn học", "bàn làm việc", "bàn", 
            "tủ quần áo", "tủ", "giường", "sofa", "kệ sách", "tủ sách", "kệ",
            "áo sơ mi", "áo thun", "áo khoác", "quần jean", "quần tây", "váy", "giày", "dép",
            "sách", "truyện tranh", "tiểu thuyết",
            "son môi", "kem chống nắng", "sữa rửa mặt",
            "nồi cơm điện", "chảo chống dính", "lò vi sóng", "máy hút bụi",
            "điện thoại", "laptop", "máy tính", "tai nghe"
        ]
        category = None
        for cat in categories:
            if cat in q:
                category = cat
                break

        # Bridge "lịch sử" queries to "sách"
        if not category and ("sách" in q or "lịch sử" in q):
            category = "sách"

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
                "maxOutputTokens": 5000
            }
        }
        
        try:
            import httpx
            with httpx.Client(timeout=20.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    res_data = res.json()
                    text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    return text
                else:
                    logger.error(f"Gemini API returned status code {res.status_code}: {res.text}")
                    return None
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
            "max_tokens": 5000
        }
        
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            with httpx.Client(timeout=15.0) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    res_data = res.json()
                    text = res_data["choices"][0]["message"]["content"]
                    return text
                else:
                    logger.error(f"OpenAI API returned status code {res.status_code}: {res.text}")
                    return None
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
            products = []
            for r in scroll_results:
                p = r.payload.copy()
                p["product_id"] = p.get("product_id") or ""
                p["title"] = p.get("title") or ""
                p["description"] = p.get("description") or ""
                p["category"] = p.get("category") or ""
                p["price"] = p.get("price") or 0
                p["image_url"] = p.get("image_url") or ""
                products.append(p)

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

    def _filter_existing_products(self, products: List[Dict]) -> List[Dict]:
        """Lọc các sản phẩm thực sự tồn tại và đang hoạt động trong MySQL database."""
        if not products:
            return []
        
        uuids = [p.get("product_id") for p in products if p.get("product_id")]
        if not uuids:
            return []
            
        from app.features.recommendation.services.recommendation_service import get_mysql_connection
        conn = None
        existing_uuids = set()
        try:
            conn = get_mysql_connection()
            with conn.cursor() as cursor:
                format_strings = ','.join(['%s'] * len(uuids))
                query = f"SELECT uuid FROM products WHERE is_active = 1 AND deleted_at IS NULL AND uuid IN ({format_strings})"
                cursor.execute(query, tuple(uuids))
                rows = cursor.fetchall()
                for r in rows:
                    val = r.get("uuid") if isinstance(r, dict) else r[0]
                    if val:
                        existing_uuids.add(str(val).lower())
        except Exception as e:
            logger.error(f"Error filtering existing products from MySQL: {e}", exc_info=True)
            return products
        finally:
            if conn:
                conn.close()
                
        return [p for p in products if p.get("product_id") and p.get("product_id").lower() in existing_uuids]

chatbot_service = ChatbotService()
