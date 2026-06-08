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
