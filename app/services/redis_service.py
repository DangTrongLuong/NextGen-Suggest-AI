import re
import json
import logging
import unicodedata
from typing import Optional, Any, List, Dict
from app.core.redis import redis_client
from app.config import settings

logger = logging.getLogger(__name__)

class RedisService:
    """Service to handle caching search queries with the exact same normalization as the Java version."""
    
    def __init__(self):
        self.client = redis_client
        self.default_ttl = settings.REDIS_CACHE_TTL

    def normalize_keyword(self, keyword: str) -> str:
        """
        Normalize keyword using the same logic as Java's normalizeKeyword:
        - Trim and lowercase.
        - Unicode NFD normalization.
        - Strip diacritical marks.
        - Convert 'đ'/'Đ' to 'd'.
        - Collapse multiple spaces to single.
        """
        if not keyword:
            return ""
        
        # 1. Trim and lower case
        normalized = keyword.strip().lower()
        
        # 2. Normalize to NFD form to separate base letters from diacritics
        normalized = unicodedata.normalize('NFD', normalized)
        
        # 3. Remove all diacritical marks (Unicode category 'Mn' = Nonspacing Mark)
        normalized = "".join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # 4. Manually convert 'đ' and 'Đ' to 'd'
        normalized = normalized.replace('đ', 'd').replace('Đ', 'd')
        
        # 5. Replace multiple whitespace characters with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized

    def build_vector_key(self, query: str) -> str:
        """Build cache key for query vector of format vector:{query}"""
        normalized_query = self.normalize_keyword(query)
        return f"vector:{normalized_query}"

    def build_result_key(self, query: str, use_reranking: bool) -> str:
        """Build cache key for search results of format result:{query}:{use_reranking}"""
        normalized_query = self.normalize_keyword(query)
        return f"result:{normalized_query}:{str(use_reranking).lower()}"

    def get_cached_vector(self, query: str) -> Optional[List[float]]:
        """Get cached query vector from Redis."""
        if not self.client:
            return None
        
        key = self.build_vector_key(query)
        try:
            cached_data = self.client.get(key)
            if cached_data:
                logger.info(f"[REDIS] Vector HIT for key: {key}")
                return json.loads(cached_data)
            logger.info(f"[REDIS] Vector MISS for key: {key}")
        except Exception as e:
            logger.warning(f"[REDIS] Error reading vector cache for key {key}: {e}")
        return None

    def set_cached_vector(self, query: str, vector: List[float], ttl: int = 7 * 24 * 3600) -> bool:
        """Store query vector in Redis."""
        if not self.client:
            return False
        
        key = self.build_vector_key(query)
        try:
            serialized_value = json.dumps(vector)
            self.client.set(key, serialized_value, ex=ttl)
            logger.info(f"[REDIS] Vector SAVE key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Error writing vector cache for key {key}: {e}")
            return False

    def get_cached_result(self, query: str, use_reranking: bool) -> Optional[List[Dict]]:
        """Get cached search results from Redis."""
        if not self.client:
            return None
        
        key = self.build_result_key(query, use_reranking)
        try:
            cached_data = self.client.get(key)
            if cached_data:
                logger.info(f"[REDIS] Result HIT for key: {key}")
                return json.loads(cached_data)
            logger.info(f"[REDIS] Result MISS for key: {key}")
        except Exception as e:
            logger.warning(f"[REDIS] Error reading result cache for key {key}: {e}")
        return None

    def set_cached_result(self, query: str, use_reranking: bool, results: List[Dict], ttl: Optional[int] = None) -> bool:
        """Store search results in Redis."""
        if not self.client:
            return False
        
        key = self.build_result_key(query, use_reranking)
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            serialized_value = json.dumps(results)
            self.client.set(key, serialized_value, ex=ttl)
            logger.info(f"[REDIS] Result SAVE key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Error writing result cache for key {key}: {e}")
            return False

    # Backwards compatibility methods
    def get_cached_search(self, query: str, use_reranking: bool) -> Optional[List[Dict]]:
        """Get cached search results (delegates to get_cached_result)."""
        return self.get_cached_result(query, use_reranking)

    def set_cached_search(self, query: str, use_reranking: bool, results: List[Dict], ttl: Optional[int] = None) -> bool:
        """Store search results (delegates to set_cached_result)."""
        return self.set_cached_result(query, use_reranking, results, ttl)

# Singleton instance
redis_service = RedisService()
