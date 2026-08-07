"""
Search Service - Kết hợp Embedding + Qdrant + Reranking + Redis Cache.
"""
import time
import re
from typing import List, Dict, Optional, Tuple

from app.features.search.services.embedding_service import embedding_service
from app.shared.qdrant.service import qdrant_service
from app.features.search.services.rerank_service import rerank_service
from app.shared.redis.service import redis_service
import logging

logger = logging.getLogger(__name__)


# ── Domain Intents & Cross-Category Noise Filter Map ──────────────────────
DOMAIN_INTENTS = {
    "FASHION": {
        "keywords": [
            "áo", "quần", "quần áo", "đầm", "váy", "thời trang", "giày", "dép", 
            "nón", "mũ", "polo", "tây", "jean", "jeans", "hoodie", "jacket", 
            "sơ mi", "sơmi", "áo phông", "áo thun", "chân váy", "ví da", "thắt lưng"
        ],
        "category_keywords": ["thời trang", "áo", "quần", "giày", "trang phục", "fashion", "váy"],
        "noise_terms": [
            "đồ chơi", "búp bê", "mô hình", "móc", "kệ", "giá treo", "tủ", "tranh", "đèn", "sách", "túi đựng", "hộp đựng"
        ]
    },
    "BEAUTY": {
        "keywords": [
            "mỹ phẩm", "mĩ phẩm", "son", "kem dưỡng", "serum", "sữa rửa mặt", "trang điểm", 
            "skincare", "nước hoa", "phấn", "mặt nạ", "tẩy trang", "chăm sóc da", "son môi"
        ],
        "category_keywords": ["mỹ phẩm", "mĩ phẩm", "sức khỏe & sắc đẹp", "sắc đẹp", "chăm sóc cá nhân", "beauty", "trang điểm", "skincare"],
        "noise_terms": [
            "túi đựng", "hộp đựng", "kệ đựng", "tủ đựng", "giá để", "búp bê", "đồ chơi", "gương"
        ]
    },
    "BOOKS": {
        "keywords": [
            "sách", "truyện", "sách giáo khoa", "tiểu thuyết", "văn học", "truyện tranh", "vở", "tập", "sách thiếu nhi"
        ],
        "category_keywords": ["sách", "nhà sách", "văn phòng phẩm", "truyện", "books"],
        "noise_terms": [
            "đèn", "kệ", "giá để", "tủ", "bút", "bìa", "túi đựng", "kẹp", "đồ chơi"
        ]
    },
    "TOYS": {
        "keywords": [
            "đồ chơi", "búp bê", "mô hình", "lego", "gấu bông", "xe điều khiển", "gundam", "xếp hình"
        ],
        "category_keywords": ["đồ chơi", "mẹ & bé", "mẹ và bé", "toys"],
        "noise_terms": [
            "móc", "kệ", "tủ"
        ]
    },
    "ELECTRONICS": {
        "keywords": [
            "laptop", "máy tính", "điện thoại", "tivi", "tủ lạnh", "máy giặt", "tai nghe", "loa", "điện máy", "máy tính bảng", "máy ảnh"
        ],
        "category_keywords": ["điện tử", "công nghệ", "điện thoại", "máy tính", "điện máy", "thiết bị số"],
        "noise_terms": [
            "ốp", "bao da", "túi đựng", "miếng dán", "kệ đỡ", "giá đỡ", "mô hình", "đồ chơi", "sạc", "cáp"
        ]
    },
    "FURNITURE": {
        "keywords": [
            "ghế", "bàn", "tủ", "giường", "nệm", "sofa", "đèn", "kệ", "gương", "nội thất", "gia dụng"
        ],
        "category_keywords": ["nội thất", "gia dụng", "nhà cửa", "ghế", "bàn", "tủ", "furniture"],
        "noise_terms": [
            "mô hình", "đồ chơi", "sách"
        ]
    }
}


def detect_query_domains(query: str) -> List[str]:
    """Phát hiện các nhóm ngành hàng (domains) mà câu truy vấn hướng tới."""
    query_lower = query.lower().strip()
    detected = []
    
    for domain_name, info in DOMAIN_INTENTS.items():
        for kw in info["keywords"]:
            pattern = r'(?:\b|_)' + re.escape(kw) + r'(?:\b|_)'
            if re.search(pattern, query_lower) or kw in query_lower:
                detected.append(domain_name)
                break
    return detected


def extract_intent(query: str) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
    """
    Tự động bóc tách danh mục và khoảng giá từ câu truy vấn tiếng Việt.
    Ví dụ: "Tôi cần ghế văn phòng dưới 2 triệu"
    """
    query_lower = query.lower().strip()
    category = None
    price_range = None

    # 1. Bóc tách danh mục phổ biến
    category_map = {
        "ghế văn phòng": "Ghế văn phòng",
        "ghế xoay": "Ghế văn phòng",
        "bàn học": "Bàn học",
        "bàn làm việc": "Bàn làm việc",
        "áo polo": "Áo polo",
        "áo thun": "Áo thun",
        "quần tây": "Quần tây",
        "quần jean": "Quần jean",
        "phụ kiện": "Phụ kiện",
        "giày": "Giày"
    }

    for key, val in category_map.items():
        if key in query_lower:
            category = val
            break

    # 2. Bóc tách khoảng giá (ví dụ: "dưới 2 triệu", "từ 500k đến 1 triệu")
    under_million_match = re.search(r'dưới\s+(\d+[\d\.,]*)\s*triệu', query_lower)
    if under_million_match:
        try:
            val = float(under_million_match.group(1).replace(',', '.'))
            price_range = (0.0, val * 1000000.0)
        except ValueError:
            pass
    else:
        under_k_match = re.search(r'dưới\s+(\d+)\s*(?:k|nghìn|trăm)', query_lower)
        if under_k_match:
            try:
                val = float(under_k_match.group(1))
                multiplier = 1000.0 if 'k' in query_lower or 'nghìn' in query_lower else 100000.0
                price_range = (0.0, val * multiplier)
            except ValueError:
                pass
        else:
            under_num_match = re.search(r'dưới\s+(\d+)', query_lower)
            if under_num_match:
                try:
                    val = float(under_num_match.group(1))
                    if val < 50:  # Giả định triệu nếu là số nhỏ
                        val = val * 1000000.0
                    price_range = (0.0, val)
                except ValueError:
                    pass

    range_match = re.search(r'từ\s+(\d+[\d\.,]*)\s*triệu\s+đến\s+(\d+[\d\.,]*)\s*triệu', query_lower)
    if range_match:
        try:
            min_v = float(range_match.group(1).replace(',', '.')) * 1000000.0
            max_v = float(range_match.group(2).replace(',', '.')) * 1000000.0
            price_range = (min_v, max_v)
        except ValueError:
            pass

    return category, price_range


class SearchService:
    """Orchestrates the full search pipeline: embed → retrieve → rerank."""

    def semantic_search(
        self,
        query: str,
        top_k: int = 12,
        candidate_k: int = 50,
        use_reranking: bool = True,
        category_filter: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
        page: int = 0,
        size: int = 12,
    ) -> Tuple[List[Dict], float]:
        """
        Paginated semantic search pipeline.
        Retrieves 50 candidates, reranks ALL of them, caches the full sorted list in Redis,
        and returns the requested page slice from the cache. This optimizes subsequent
        "See More" requests to take less than 5ms.
        """
        t0 = time.perf_counter()
        import hashlib
        import json

        try:
            # Detect intents/filters
            extracted_cat, extracted_price = extract_intent(query)
            if category_filter is None and extracted_cat:
                category_filter = extracted_cat
            if price_range is None and extracted_price:
                price_range = extracted_price

            clean_query = query.strip()
            
            # Construct a unique cache key based on query and filters
            filter_data = {
                "query": clean_query,
                "category": category_filter,
                "price_range": price_range
            }
            filter_str = json.dumps(filter_data, sort_keys=True)
            filter_hash = hashlib.md5(filter_str.encode('utf-8')).hexdigest()
            results_cache_key = f"result:reranked:{filter_hash}:{str(use_reranking).lower()}"

            # Check cache for pre-reranked full results first
            cached_full_results = None
            try:
                if redis_service.is_available():
                    cached_data = redis_service.client.get(results_cache_key)
                    if cached_data:
                        cached_full_results = json.loads(cached_data)
                        logger.info(f"[REDIS] Full Reranked Results HIT for key: {results_cache_key}")
            except Exception as e:
                logger.warning(f"Error reading search results cache: {e}")

            if cached_full_results is not None:
                page_results = cached_full_results[page * size : (page + 1) * size]
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.info(
                    f"search (cache hit) | query='{query}' | page={page} | size={size} "
                    f"| total_cached={len(cached_full_results)} | results={len(page_results)} | {latency_ms}ms"
                )
                return page_results, latency_ms

            # Cache MISS -> Run full pipeline and populate cache
            query_vector = None
            try:
                if redis_service.is_available():
                    query_vector = redis_service.get_cached_vector(query)
            except Exception as e:
                logger.error(f"Error checking vector cache: {e}")

            if query_vector is None:
                query_vector = embedding_service.embed_query(query)
                try:
                    if redis_service.is_available() and query_vector:
                        redis_service.set_cached_vector(query, query_vector, ttl=7 * 24 * 3600)
                except Exception as e:
                    logger.error(f"Error setting vector cache: {e}")

            # Fetch 50 candidates from Qdrant
            candidates = qdrant_service.search(
                query_vector=query_vector,
                top_k=50,
                category_filter=category_filter,
                price_range=price_range,
            )

            if not candidates:
                return [], round((time.perf_counter() - t0) * 1000, 2)

            # Rerank ALL candidates (max 50)
            if use_reranking:
                results = rerank_service.rerank(
                    query=query,
                    documents=candidates,
                    top_k=len(candidates),
                    document_key="short_context",
                )
            else:
                results = candidates

            # Filter out low-relevance candidates & out-of-catalog false matches
            filtered_results = [r for r in results if self._is_relevant_search_match(query, r, use_reranking)]
            
            # Smart fallback: If domain filtering was too strict and returned 0 results, soften restriction
            if not filtered_results and results:
                logger.info(f"Domain filter yielded 0 results for '{query}'. Softening domain strictness for fallback.")
                filtered_results = [r for r in results if (r.get("reranker_score", 0) > 0.0 or r.get("score", 0) > 0.4)]

            results = filtered_results

            if not results:
                logger.info(f"All candidates for query '{query}' failed relevance intent check. Returning 0 results.")
                return [], round((time.perf_counter() - t0) * 1000, 2)

            # Add rank indices
            for i, r in enumerate(results, 1):
                r["rank"] = i

            # Cache full reranked results (TTL = 10 minutes = 600s)
            try:
                if redis_service.is_available():
                    redis_service.client.set(results_cache_key, json.dumps(results), ex=600)
                    logger.info(f"[REDIS] Full Reranked Results SAVED for key: {results_cache_key}")
            except Exception as e:
                logger.warning(f"Error caching full search results: {e}")

            # Slice current page
            page_results = results[page * size : (page + 1) * size]
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                f"search (cache miss) | query='{query}' | page={page} | size={size} "
                f"| qdrant_fetched={len(candidates)} | total_reranked={len(results)} "
                f"| results={len(page_results)} | rerank={use_reranking} | {latency_ms}ms"
            )
            return page_results, latency_ms

        except Exception as e:
            logger.error(f"Semantic search pipeline failed: {e}. Falling back to MySQL keyword search.", exc_info=True)
            try:
                fallback_results = self._fallback_mysql_search(query, category_filter, price_range, page, size)
                for i, r in enumerate(fallback_results, 1):
                    r["rank"] = i
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return fallback_results, latency_ms
            except Exception as ex:
                logger.error(f"MySQL search fallback failed: {ex}", exc_info=True)
                return [], round((time.perf_counter() - t0) * 1000, 2)

    def _fallback_mysql_search(
        self,
        query: str,
        category_filter: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
        page: int = 0,
        size: int = 12,
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

                if category_filter:
                    conditions.append("c.name LIKE %s")
                    params.append(f"%{category_filter}%")

                offset = page * size
                if price_range:
                    query_sql = f"""
                    SELECT p.uuid as product_id, p.name as title, p.description, 
                           c.name as category, p.image_url, p.brand,
                           COALESCE((SELECT MIN(v.price) FROM product_variants v WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 0) as price
                    FROM products p
                    LEFT JOIN categories c ON p.category_id = c.id
                    WHERE {" AND ".join(conditions)}
                    LIMIT 100;
                    """
                else:
                    query_sql = f"""
                    SELECT p.uuid as product_id, p.name as title, p.description, 
                           c.name as category, p.image_url, p.brand,
                           COALESCE((SELECT MIN(v.price) FROM product_variants v WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 0) as price
                    FROM products p
                    LEFT JOIN categories c ON p.category_id = c.id
                    WHERE {" AND ".join(conditions)}
                    LIMIT %s OFFSET %s;
                    """
                    params.extend([size, offset])

                cursor.execute(query_sql, tuple(params))
                rows = cursor.fetchall()

                results = []
                for r in rows:
                    r["price"] = float(r["price"])
                    r["tags"] = []
                    r["score"] = 0.5
                    r["reason"] = "Tìm kiếm từ khóa (Dự phòng)"
                    r["short_context"] = f"{r['title']} {r['description']}"

                    if price_range:
                        min_p, max_p = price_range
                        if min_p is not None and r["price"] < min_p:
                            continue
                        if max_p is not None and r["price"] > max_p:
                            continue
                    results.append(r)

                if price_range:
                    return results[offset : offset + size]
                return results

        except Exception as e:
            logger.error(f"Fallback MySQL search failed: {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()

    def _is_relevant_search_match(self, query: str, doc: Dict, use_reranking: bool = True) -> bool:
        """
        Evaluates whether candidate doc genuinely matches the user search query intent.
        Filters out out-of-catalog items or accessories when searching for main product types.
        Prevents cross-category mismatches (e.g. searching 'áo' returning 'đồ chơi' or 'móc treo áo',
        searching 'mỹ phẩm' returning 'túi đựng mỹ phẩm', searching 'sách' returning 'đèn đọc sách').
        """
        query_lower = query.strip().lower()
        query_words = set(query_lower.split())
        if not query_words:
            return False

        title = (doc.get("title") or "").lower()
        category = (doc.get("category") or "").lower()
        desc = (doc.get("description") or "").lower()
        full_text = f"{title} {category} {desc}"

        orig_score = doc.get("original_score", doc.get("score", 0.0))
        reranker_score = doc.get("reranker_score")

        # 1. Cross-Domain Noise Filtering (Cơ chế chống lệch ngành hàng)
        detected_domains = detect_query_domains(query_lower)
        for domain_name in detected_domains:
            domain_info = DOMAIN_INTENTS[domain_name]
            noise_terms = domain_info["noise_terms"]
            
            # Check noise terms in title or category that are NOT present in query
            for noise in noise_terms:
                if noise in query_lower:
                    continue  # Người dùng chủ động tìm (vd: "đèn đọc sách", "móc treo áo", "túi đựng mỹ phẩm") -> Giữ lại
                
                # Nếu từ gây nhiễu xuất hiện ở tiêu đề sản phẩm hoặc danh mục sản phẩm bị lệch hẳn sang ngành khác
                if noise in title or noise in category:
                    logger.debug(f"[DOMAIN FILTER] Filtered product '{title}' (Cat: '{category}') for query '{query}' due to noise term '{noise}' in domain {domain_name}")
                    return False

            # Nếu truy vấn thuộc ngành này (vd: Thời trang), nhưng sản phẩm thuộc hẳn ngành xung đột (vd: Đồ chơi, Nội thất)
            off_domain_cats = []
            if domain_name == "FASHION":
                off_domain_cats = ["đồ chơi", "nội thất", "gia dụng", "sách", "thiết bị điện"]
            elif domain_name == "BEAUTY":
                off_domain_cats = ["đồ chơi", "nội thất", "gia dụng", "thời trang", "phụ kiện túi"]
            elif domain_name == "BOOKS":
                off_domain_cats = ["đồ chơi", "nội thất", "gia dụng", "điện tử"]

            for off_cat in off_domain_cats:
                if off_cat in query_lower:
                    continue
                if off_cat in category:
                    logger.debug(f"[DOMAIN FILTER] Filtered product '{title}' (Cat: '{category}') for query '{query}' due to off-domain category '{off_cat}'")
                    return False

        # 2. Handle main target category vs accessory mismatch (e.g., query="laptop", item="Túi đựng laptop")
        if query_lower in ["laptop", "máy tính bảng", "điện thoại", "tủ lạnh", "máy giặt", "tivi", "xe máy"]:
            accessory_terms = ["túi", "bao da", "ốp", "kính cường lực", "sạc", "cáp", "chuột", "bàn phím", "tủ sấy", "tủ ngăn kéo", "tủ trưng bày", "tủ giày", "đồ chơi", "mô hình"]
            if any(acc in title for acc in accessory_terms):
                return False

        matched_words = {w for w in query_words if w in full_text}
        match_ratio = len(matched_words) / len(query_words)

        # If query words are completely present in title or category (ratio == 1.0)
        if match_ratio == 1.0:
            if use_reranking and reranker_score is not None:
                return reranker_score >= -0.5
            return orig_score >= 0.35

        # Multi-word query (e.g. "tủ lạnh", "máy giặt") where not all words are present:
        if use_reranking and reranker_score is not None:
            return reranker_score >= 0.30 and match_ratio >= 0.5
        
        return orig_score >= 0.50 and match_ratio >= 0.5

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
    ) -> Tuple[List[Dict], float]:
        """Simple keyword search for benchmarking."""
        t0 = time.perf_counter()
        results = qdrant_service.keyword_search(query, top_k=top_k)
        if category_filter:
            results = [r for r in results if r.get("category") == category_filter]
        for i, r in enumerate(results[:top_k], 1):
            r["rank"] = i
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return results[:top_k], latency_ms


search_service = SearchService()
