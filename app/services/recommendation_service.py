"""
Recommendation Service - Similar products & history-based recommendations.
"""
import time
import numpy as np
from typing import List, Dict, Optional, Tuple

from app.services.qdrant_service import qdrant_service
import logging

logger = logging.getLogger(__name__)


class RecommendationService:

    def get_similar_products(
        self,
        product_id: str,
        top_k: int = 10,
        same_category: bool = False,
    ) -> Tuple[List[Dict], str, float]:
        """
        Tìm sản phẩm tương tự dựa trên vector của sản phẩm gốc.

        Returns:
            (results, source_title, latency_ms)
        """
        t0 = time.perf_counter()

        # Lấy thông tin sản phẩm gốc
        source = qdrant_service.get_product_by_id(product_id)
        if source is None:
            return [], "Not found", 0.0

        # Lấy vector của sản phẩm
        product_vector = qdrant_service.get_product_vector(product_id)
        if product_vector is None:
            return [], source.get("title", ""), 0.0

        category_filter = source.get("category") if same_category else None

        results = qdrant_service.search(
            query_vector=product_vector,
            top_k=top_k + 1,           # +1 vì sẽ bao gồm chính nó
            category_filter=category_filter,
            exclude_ids=[product_id],  # Loại bỏ chính sản phẩm đó
        )

        for i, r in enumerate(results[:top_k], 1):
            r["rank"] = i

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return results[:top_k], source.get("title", ""), latency_ms

    def get_recommendations_from_history(
        self,
        viewed_product_ids: List[str],
        top_k: int = 10,
        exclude_viewed: bool = True,
    ) -> Tuple[List[Dict], float]:
        """
        Gợi ý dựa trên lịch sử xem sản phẩm (Average Embedding approach).

        Tính average vector của các sản phẩm đã xem → tìm sản phẩm gần nhất.

        Returns:
            (results, latency_ms)
        """
        t0 = time.perf_counter()

        # Thu thập vectors
        vectors = []
        valid_ids = []
        for pid in viewed_product_ids:
            vec = qdrant_service.get_product_vector(pid)
            if vec is not None:
                vectors.append(vec)
                valid_ids.append(pid)

        if not vectors:
            logger.warning("No valid product vectors found for history")
            return [], 0.0

        # Average pooling → user preference vector
        avg_vector = np.mean(vectors, axis=0)
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm

        exclude = viewed_product_ids if exclude_viewed else None

        results = qdrant_service.search(
            query_vector=avg_vector.tolist(),
            top_k=top_k + len(viewed_product_ids),
            exclude_ids=exclude,
        )

        results = results[:top_k]
        for i, r in enumerate(results, 1):
            r["rank"] = i

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"history_rec | viewed={len(valid_ids)} products "
            f"| results={len(results)} | {latency_ms}ms"
        )
        return results, latency_ms


recommendation_service = RecommendationService()
