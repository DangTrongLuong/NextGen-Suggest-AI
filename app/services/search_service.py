"""
Search Service - Kết hợp Embedding + Qdrant + Reranking.
"""
import time
from typing import List, Dict, Optional, Tuple

from app.services.embedding_service import embedding_service
from app.services.qdrant_service import qdrant_service
from app.services.reranker_service import reranker_service
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
        Full semantic search pipeline.

        Returns:
            (results, latency_ms)
        """
        t0 = time.perf_counter()

        # Step 1 – Embed query
        query_vector = embedding_service.embed_query(query)

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
            results = reranker_service.rerank(
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
