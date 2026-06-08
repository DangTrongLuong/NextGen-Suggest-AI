"""
Search Service - Kết hợp Embedding + Qdrant + Reranking + Redis Cache.
"""
import time
from typing import List, Dict, Optional, Tuple

from app.services.embedding_service import embedding_service
from app.services.qdrant_service import qdrant_service
from app.services.rerank_service import rerank_service
from app.services.redis_service import redis_service
import logging

logger = logging.getLogger(__name__)


class SearchService:
    """Orchestrates the full search pipeline: embed → retrieve → rerank."""

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
        use_reranking: bool = True,
        category_filter: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[List[Dict], float]:
        """
        Full semantic search pipeline with dual-layer caching (Query Vector Cache & Search Result Cache).

        Returns:
            (results, latency_ms)
        """
        t0 = time.perf_counter()

        # Check Result Cache first (if applicable)
        use_cache = (category_filter is None) and (price_range is None)
        if use_cache:
            try:
                cached_results = redis_service.get_cached_result(query, use_reranking)
                if cached_results is not None:
                    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                    logger.info(
                        f"search | query='{query}' | result_cache=HIT | results={len(cached_results)} "
                        f"| rerank={use_reranking} | {latency_ms}ms"
                    )
                    return cached_results, latency_ms
            except Exception as e:
                logger.error(f"Error checking result cache: {e}", exc_info=True)

        # Step 1 – Get Query Vector (Check Vector Cache first)
        query_vector = None
        if use_cache:
            try:
                query_vector = redis_service.get_cached_vector(query)
            except Exception as e:
                logger.error(f"Error checking vector cache: {e}", exc_info=True)

        if query_vector is None:
            # Vector Cache MISS -> call Embedding Service
            logger.info(f"Vector Cache MISS for query='{query}'. Calling Embedding Service.")
            query_vector = embedding_service.embed_query(query)
            
            # Save to Vector Cache (TTL 7 days = 604800s)
            if use_cache and query_vector:
                try:
                    redis_service.set_cached_vector(query, query_vector, ttl=7 * 24 * 3600)
                except Exception as e:
                    logger.error(f"Error setting vector cache: {e}", exc_info=True)
        else:
            logger.info(f"Vector Cache HIT for query='{query}'. Skipped Embedding Service.")

        # Step 2 – Retrieve candidates from Qdrant
        candidates = qdrant_service.search(
            query_vector=query_vector,
            top_k=candidate_k if use_reranking else top_k,
            category_filter=category_filter,
            price_range=price_range,
        )

        if not candidates:
            return [], round((time.perf_counter() - t0) * 1000, 2)

        # Step 3 – Optional Reranking
        if use_reranking:
            results = rerank_service.rerank(
                query=query,
                documents=candidates,
                top_k=top_k,
                document_key="short_context",
            )
        else:
            results = candidates[:top_k]

        # Add rank field
        for i, r in enumerate(results, 1):
            r["rank"] = i

        # Save to Result Cache if applicable
        if use_cache and results:
            try:
                redis_service.set_cached_result(query, use_reranking, results)
            except Exception as e:
                logger.error(f"Error setting result cache: {e}", exc_info=True)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"search | query='{query}' | candidates={len(candidates)} "
            f"| results={len(results)} | rerank={use_reranking} | {latency_ms}ms"
        )
        return results, latency_ms

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
