"""
Search Service - Kết hợp Embedding + Qdrant + Reranking + Redis Cache.
"""
import time
from typing import List, Dict, Optional, Tuple

from app.features.search.services.embedding_service import embedding_service
from app.shared.qdrant.service import qdrant_service
from app.features.search.services.rerank_service import rerank_service
from app.shared.redis.service import redis_service
import logging

logger = logging.getLogger(__name__)


def extract_intent(query: str) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
    """
    Tự động bóc tách danh mục và khoảng giá từ câu truy vấn tiếng Việt.
    Ví dụ: "Tôi cần ghế văn phòng dưới 2 triệu"
    """
    import re
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
    # Khớp "dưới X triệu" hoặc "dưới X.000.000"
    under_million_match = re.search(r'dưới\s+(\d+[\d\.,]*)\s*triệu', query_lower)
    if under_million_match:
        try:
            val = float(under_million_match.group(1).replace(',', '.'))
            price_range = (0.0, val * 1000000.0)
        except ValueError:
            pass
    else:
        # Khớp "dưới Xk" hoặc "dưới X trăm"
        under_k_match = re.search(r'dưới\s+(\d+)\s*(?:k|nghìn|trăm)', query_lower)
        if under_k_match:
            try:
                val = float(under_k_match.group(1))
                multiplier = 1000.0 if 'k' in query_lower or 'nghìn' in query_lower else 100000.0
                price_range = (0.0, val * multiplier)
            except ValueError:
                pass
        else:
            # Khớp số trực tiếp, ví dụ "dưới 500000"
            under_num_match = re.search(r'dưới\s+(\d+)', query_lower)
            if under_num_match:
                try:
                    val = float(under_num_match.group(1))
                    if val < 50:  # Giả định triệu nếu là số nhỏ
                        val = val * 1000000.0
                    price_range = (0.0, val)
                except ValueError:
                    pass

    # Khớp "từ X triệu đến Y triệu"
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
        Only retrieves (page+1)*size candidates from Qdrant, slices the current
        page window, then reranks ONLY those items — keeping latency minimal.

        Returns:
            (results, latency_ms)
        """
        t0 = time.perf_counter()

        try:
            # Tự động phát hiện Intent nếu chưa truyền filter cứng
            extracted_cat, extracted_price = extract_intent(query)
            if category_filter is None and extracted_cat:
                category_filter = extracted_cat
                logger.info(f"AI Search: Extracted category_filter='{category_filter}' from query='{query}'")
            if price_range is None and extracted_price:
                price_range = extracted_price
                logger.info(f"AI Search: Extracted price_range={price_range} from query='{query}'")

            # Only vector cache is used; result cache is skipped for paginated queries
            use_vector_cache = (category_filter is None) and (price_range is None)

            # Step 1 – Get Query Vector (Check Vector Cache first)
            query_vector = None
            if use_vector_cache:
                try:
                    query_vector = redis_service.get_cached_vector(query)
                except Exception as e:
                    logger.error(f"Error checking vector cache: {e}", exc_info=True)

            if query_vector is None:
                # Vector Cache MISS -> call Embedding Service
                logger.info(f"Vector Cache MISS for query='{query}'. Calling Embedding Service.")
                query_vector = embedding_service.embed_query(query)

                # Save to Vector Cache (TTL 7 days = 604800s)
                if use_vector_cache and query_vector:
                    try:
                        redis_service.set_cached_vector(query, query_vector, ttl=7 * 24 * 3600)
                    except Exception as e:
                        logger.error(f"Error setting vector cache: {e}", exc_info=True)
            else:
                logger.info(f"Vector Cache HIT for query='{query}'. Skipped Embedding Service.")

            # Step 2 – Retrieve only (page+1)*size candidates from Qdrant
            # This avoids fetching the entire dataset; we get just enough for this page.
            fetch_k = (page + 1) * size
            candidates = qdrant_service.search(
                query_vector=query_vector,
                top_k=fetch_k,
                category_filter=category_filter,
                price_range=price_range,
            )

            if not candidates:
                return [], round((time.perf_counter() - t0) * 1000, 2)

            # Step 3 – Slice to the current page window BEFORE reranking
            page_candidates = candidates[page * size: (page + 1) * size]

            if not page_candidates:
                return [], round((time.perf_counter() - t0) * 1000, 2)

            # Step 4 – Rerank ONLY the page slice (max `size` items → very fast)
            if use_reranking:
                results = rerank_service.rerank(
                    query=query,
                    documents=page_candidates,
                    top_k=len(page_candidates),
                    document_key="short_context",
                )
            else:
                results = page_candidates

            # Add rank field
            for i, r in enumerate(results, 1):
                r["rank"] = i

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                f"search | query='{query}' | page={page} | size={size} "
                f"| qdrant_fetched={len(candidates)} | page_slice={len(page_candidates)} "
                f"| results={len(results)} | rerank={use_reranking} | {latency_ms}ms"
            )
            return results, latency_ms

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
