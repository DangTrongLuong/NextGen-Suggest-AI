import re
import json
import logging
import unicodedata
from typing import Optional, Any, List, Dict
from app.shared.redis.client import redis_client
from app.config.settings import settings

logger = logging.getLogger(__name__)

class RedisService:
    """Service to handle caching search queries with the exact same normalization as the Java version."""
    
    def __init__(self):
        self.client = redis_client
        self.default_ttl = settings.REDIS_CACHE_TTL
        self.is_online = True
        self.last_check = 0.0

    def is_available(self) -> bool:
        if not self.client:
            return False
        if not getattr(self, "is_online", True):
            import time
            if time.time() - getattr(self, "last_check", 0.0) < 30:
                return False
        return True

    def mark_offline(self, e: Exception):
        import time
        self.is_online = False
        self.last_check = time.time()
        logger.warning(f"[REDIS] Connection lost, bypassing cache for 30s. Error: {e}")

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
        if not self.is_available():
            return None
        
        key = self.build_vector_key(query)
        try:
            cached_data = self.client.get(key)
            self.is_online = True
            if cached_data:
                logger.info(f"[REDIS] Vector HIT for key: {key}")
                return json.loads(cached_data)
            logger.info(f"[REDIS] Vector MISS for key: {key}")
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error reading vector cache for key {key}: {e}")
        return None

    def set_cached_vector(self, query: str, vector: List[float], ttl: int = 7 * 24 * 3600) -> bool:
        if not self.is_available():
            return False
        
        key = self.build_vector_key(query)
        try:
            serialized_value = json.dumps(vector)
            self.client.set(key, serialized_value, ex=ttl)
            self.is_online = True
            logger.info(f"[REDIS] Vector SAVE key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error writing vector cache for key {key}: {e}")
            return False

    def get_cached_result(self, query: str, use_reranking: bool) -> Optional[List[Dict]]:
        if not self.is_available():
            return None
        
        key = self.build_result_key(query, use_reranking)
        try:
            cached_data = self.client.get(key)
            self.is_online = True
            if cached_data:
                logger.info(f"[REDIS] Result HIT for key: {key}")
                return json.loads(cached_data)
            logger.info(f"[REDIS] Result MISS for key: {key}")
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error reading result cache for key {key}: {e}")
        return None

    def set_cached_result(self, query: str, use_reranking: bool, results: List[Dict], ttl: Optional[int] = None) -> bool:
        if not self.is_available():
            return False
        
        key = self.build_result_key(query, use_reranking)
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            serialized_value = json.dumps(results)
            self.client.set(key, serialized_value, ex=ttl)
            self.is_online = True
            logger.info(f"[REDIS] Result SAVE key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error writing result cache for key {key}: {e}")
            return False

    # Backwards compatibility methods
    def get_cached_search(self, query: str, use_reranking: bool) -> Optional[List[Dict]]:
        """Get cached search results (delegates to get_cached_result)."""
        return self.get_cached_result(query, use_reranking)

    def set_cached_search(self, query: str, use_reranking: bool, results: List[Dict], ttl: Optional[int] = None) -> bool:
        """Store search results (delegates to set_cached_result)."""
        return self.set_cached_result(query, use_reranking, results, ttl)

    # ── User Recommendation Caching ───────────────────────────────────────────
    def build_recommendation_key(self, user_id: str) -> str:
        return f"recommend:user:{user_id}:homepage"

    def build_profile_key(self, user_id: str) -> str:
        return f"recommend:profile:user:{user_id}"

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        if not self.is_available() or not user_id:
            return None
        key = self.build_profile_key(user_id)
        try:
            cached = self.client.get(key)
            self.is_online = True
            if cached:
                logger.info(f"[REDIS] User Profile HIT for user: {user_id}")
                return json.loads(cached)
            logger.info(f"[REDIS] User Profile MISS for user: {user_id}")
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error reading user profile from Redis for user {user_id}: {e}")
        return None

    def set_user_profile(self, user_id: str, profile: Dict, ttl: int = 14 * 24 * 3600) -> bool:
        if not self.is_available() or not user_id:
            return False
        key = self.build_profile_key(user_id)
        try:
            self.client.set(key, json.dumps(profile), ex=ttl)
            self.is_online = True
            logger.info(f"[REDIS] User Profile SAVE for user: {user_id} (TTL: {ttl}s)")
            return True
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error writing user profile to Redis for user {user_id}: {e}")
            return False

    def invalidate_user_profile(self, user_id: str) -> bool:
        if not self.is_available() or not user_id:
            return False
        profile_key = self.build_profile_key(user_id)
        rec_key = self.build_recommendation_key(user_id)
        try:
            # Delete both profile cache and recommendation cache
            self.client.delete(profile_key)
            self.client.delete(rec_key)
            self.is_online = True
            logger.info(f"[REDIS] Invalidated user profile & recommendation cache for user: {user_id}")
            return True
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error invalidating cache for user {user_id}: {e}")
            return False


    def get_cached_recommendations(self, user_id: str) -> Optional[List[Dict]]:
        if not self.is_available() or not user_id:
            return None
        key = self.build_recommendation_key(user_id)
        try:
            cached = self.client.get(key)
            self.is_online = True
            if cached:
                logger.info(f"[REDIS] Recommendation HIT for user: {user_id}")
                return json.loads(cached)
            logger.info(f"[REDIS] Recommendation MISS for user: {user_id}")
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error reading recommendation cache for user {user_id}: {e}")
        return None

    def set_cached_recommendations(self, user_id: str, results: List[Dict], ttl: int = 1800) -> bool:
        if not self.is_available() or not user_id:
            return False
        key = self.build_recommendation_key(user_id)
        try:
            self.client.set(key, json.dumps(results), ex=ttl)
            self.is_online = True
            logger.info(f"[REDIS] Recommendation SAVE for user: {user_id} (TTL: {ttl}s)")
            return True
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error writing recommendation cache for user {user_id}: {e}")
            return False

    # ── Chat Memory Caching ───────────────────────────────────────────────────
    def build_chat_memory_key(self, user_id: str) -> str:
        return f"chat:memory:{user_id}"

    def get_cached_chat_memory(self, user_id: str) -> Optional[List[Dict]]:
        if not self.is_available() or not user_id:
            return None
        key = self.build_chat_memory_key(user_id)
        try:
            cached = self.client.get(key)
            self.is_online = True
            if cached:
                logger.info(f"[REDIS] Chat Memory HIT for user: {user_id}")
                return json.loads(cached)
            logger.info(f"[REDIS] Chat Memory MISS for user: {user_id}")
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error reading chat memory cache for user {user_id}: {e}")
        return None

    def set_cached_chat_memory(self, user_id: str, memory: List[Dict], ttl: int = 3600) -> bool:
        if not self.is_available() or not user_id:
            return False
        key = self.build_chat_memory_key(user_id)
        try:
            self.client.set(key, json.dumps(memory), ex=ttl)
            self.is_online = True
            logger.info(f"[REDIS] Chat Memory SAVE for user: {user_id} (TTL: {ttl}s)")
            return True
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error writing chat memory cache for user {user_id}: {e}")
            return False

    # ── Query Embedding Caching ────────────────────────────────────────────────
    def build_embedding_key(self, query: str) -> str:
        normalized = self.normalize_keyword(query)
        import hashlib
        query_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        return f"embedding:{query_hash}"

    def get_cached_embedding(self, query: str) -> Optional[List[float]]:
        if not self.is_available() or not query:
            return None
        key = self.build_embedding_key(query)
        try:
            cached = self.client.get(key)
            self.is_online = True
            if cached:
                logger.info(f"[REDIS] Embedding HIT for query: '{query}'")
                return json.loads(cached)
            logger.info(f"[REDIS] Embedding MISS for query: '{query}'")
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error reading embedding cache for query '{query}': {e}")
        return None

    def set_cached_embedding(self, query: str, embedding: List[float], ttl: int = 7 * 24 * 3600) -> bool:
        if not self.is_available() or not query:
            return False
        key = self.build_embedding_key(query)
        try:
            self.client.set(key, json.dumps(embedding), ex=ttl)
            self.is_online = True
            logger.info(f"[REDIS] Embedding SAVE for query: '{query}' (TTL: {ttl}s)")
            return True
        except Exception as e:
            error_str = str(e)
            if "Timeout" in error_str or "Connection" in error_str or "TimeoutError" in e.__class__.__name__:
                self.mark_offline(e)
            else:
                logger.warning(f"[REDIS] Error writing embedding cache for query '{query}': {e}")
            return False

# Singleton instance
redis_service = RedisService()
