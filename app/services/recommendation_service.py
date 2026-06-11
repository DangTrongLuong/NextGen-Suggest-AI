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

    def get_hybrid_recommendations(
        self,
        viewed_product_ids: List[str] = None,
        wishlist_product_ids: List[str] = None,
        purchased_product_ids: List[str] = None,
        popular_product_ids: List[str] = None,
        top_k: int = 10,
        exclude_viewed: bool = True
    ) -> Tuple[List[Dict], float]:
        t0 = time.perf_counter()

        viewed = viewed_product_ids or []
        wishlist = wishlist_product_ids or []
        purchased = purchased_product_ids or []
        popular = popular_product_ids or []

        # 1. Collect vectors
        view_vectors = []
        for pid in viewed:
            vec = qdrant_service.get_product_vector(pid)
            if vec is not None:
                view_vectors.append(vec)
        
        wishlist_vectors = []
        for pid in wishlist:
            vec = qdrant_service.get_product_vector(pid)
            if vec is not None:
                wishlist_vectors.append(vec)

        purchase_vectors = []
        for pid in purchased:
            vec = qdrant_service.get_product_vector(pid)
            if vec is not None:
                purchase_vectors.append(vec)

        # 2. Calculate profile vector of the user
        profile_components = []
        if view_vectors:
            profile_components.append(0.4 * np.mean(view_vectors, axis=0))
        if wishlist_vectors:
            profile_components.append(0.3 * np.mean(wishlist_vectors, axis=0))
        if purchase_vectors:
            profile_components.append(0.3 * np.mean(purchase_vectors, axis=0))

        if not profile_components:
            # Fallback: no history, return popular or scroll products
            logger.info("Hybrid Rec: No user history found. Falling back to popular/scroll.")
            fallback_ids = popular if popular else []
            results = []
            for pid in fallback_ids[:top_k]:
                p = qdrant_service.get_product_by_id(pid)
                if p:
                    p["score"] = 1.0
                    results.append(p)
            
            if not results:
                scroll_results, _ = qdrant_service.client.scroll(
                    collection_name=qdrant_service.collection_name,
                    limit=top_k,
                    with_payload=True
                )
                for r in scroll_results:
                    p = r.payload
                    p["score"] = 0.5
                    results.append({
                        "product_id": p.get("product_id"),
                        "title": p.get("title"),
                        "description": p.get("description"),
                        "category": p.get("category"),
                        "tags": p.get("tags"),
                        "price": p.get("price"),
                        "image_url": p.get("image_url"),
                        "score": 0.5
                    })
            
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return results[:top_k], latency_ms

        profile_vector = np.sum(profile_components, axis=0)
        norm = np.linalg.norm(profile_vector)
        if norm > 0:
            profile_vector = profile_vector / norm

        # 3. Get candidates from Qdrant
        exclude = viewed + purchased if exclude_viewed else None
        candidates = qdrant_service.search(
            query_vector=profile_vector.tolist(),
            top_k=top_k * 4,
            exclude_ids=exclude
        )

        # 4. Calculate Hybrid Score for each candidate
        scored_candidates = []
        for cand in candidates:
            pid = cand.get("product_id")

            # Content score (0 to 1)
            content_score = cand.get("score", 0.5)

            # Popularity score
            popularity_score = 0.1
            if pid in popular:
                try:
                    idx = popular.index(pid)
                    popularity_score = max(0.2, 1.0 - (idx * 0.05))
                except ValueError:
                    popularity_score = 0.5

            # Behavior score: Max similarity to wishlist/purchases
            behavior_score = 0.1
            cand_vector = qdrant_service.get_product_vector(pid)
            if cand_vector is not None:
                cand_vec_np = np.array(cand_vector)
                similarities = []
                for ref_vec in wishlist_vectors + purchase_vectors:
                    ref_vec_np = np.array(ref_vec)
                    cos_sim = np.dot(cand_vec_np, ref_vec_np) / (np.linalg.norm(cand_vec_np) * np.linalg.norm(ref_vec_np) + 1e-9)
                    similarities.append(cos_sim)
                if similarities:
                    behavior_score = max(0.1, float(np.max(similarities)))

            # Collaborative score (shared categories / tags)
            collaborative_score = 0.1
            categories = []
            tags = set()
            for ref_id in viewed + wishlist + purchased:
                ref_p = qdrant_service.get_product_by_id(ref_id)
                if ref_p:
                    if ref_p.get("category"):
                        categories.append(ref_p.get("category"))
                    if ref_p.get("tags"):
                        tags.update(ref_p.get("tags"))
            
            if cand.get("category") in categories:
                collaborative_score += 0.4
            cand_tags = set(cand.get("tags") or [])
            shared_tags = cand_tags.intersection(tags)
            if shared_tags:
                collaborative_score += min(0.4, len(shared_tags) * 0.1)

            # final score calculation:
            hybrid_score = 0.25 * popularity_score + 0.25 * content_score + 0.25 * behavior_score + 0.25 * collaborative_score
            
            cand_copy = cand.copy()
            cand_copy["score"] = round(hybrid_score, 4)
            scored_candidates.append(cand_copy)

        # 5. Sort candidates
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        results = scored_candidates[:top_k]

        for i, r in enumerate(results, 1):
            r["rank"] = i

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"hybrid_rec | viewed={len(viewed)} | wishlist={len(wishlist)} | purchased={len(purchased)} | results={len(results)} | {latency_ms}ms")
        return results, latency_ms


recommendation_service = RecommendationService()
