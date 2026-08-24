"""
Reranker Service - High-Performance & Concurrency-Guarded Cross-Encoder Reranking.
Dùng BAAI/bge-reranker-base để score lại (query, document) pairs.
Được trang bị Concurrency Guard (Semaphore) + torch.no_grad() để bảo vệ CPU & RAM.
"""
from sentence_transformers import CrossEncoder
import torch
import threading
import gc
from typing import List, Dict
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RerankerService:
    """Singleton cross-encoder reranker service with Concurrency Protection."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model_name = settings.RERANKER_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Concurrency Limiter: Allow max 2 simultaneous AI Rerank tasks on CPU
        # All excess requests under heavy load will fallback to ultra-fast Qdrant Vector Search (< 15ms)
        self._semaphore = threading.BoundedSemaphore(value=2)

        # Cap PyTorch CPU threads to 1 per worker to prevent CPU thrashing & lock contention
        try:
            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception as e:
            logger.warning(f"Could not set PyTorch thread limit: {e}")

        logger.info(f" Loading reranker: {self.model_name}")
        logger.info(f"   Device: {self.device}")

        import os
        models_dir = os.path.abspath(settings.HF_HOME)
        os.makedirs(models_dir, exist_ok=True)
        os.environ["HF_HOME"] = models_dir

        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=512,
            cache_folder=models_dir,
            trust_remote_code=True
        )

        logger.info("    Reranker model ready with Concurrency Guard")
        self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
        document_key: str = "search_document"
    ) -> List[Dict]:
        """
        Rerank candidates using Cross-Encoder with CPU Concurrency Guard.
        If CPU is busy under load (> 2 concurrent tasks), falls back gracefully to vector similarity (< 1ms).
        """
        if not documents:
            return []

        # Try acquiring semaphore without blocking (or max 50ms wait) to keep latency low
        acquired = self._semaphore.acquire(blocking=True, timeout=0.05)
        if not acquired:
            logger.info(f"[CPU GUARD] AI Reranker busy. Fast fallback to Vector Similarity for query: '{query}'")
            return self._fast_vector_fallback(query, documents, top_k)

        try:
            pairs = []
            for doc in documents:
                if document_key == "short_context":
                    title = doc.get("title", "")
                    category = doc.get("category", "")
                    parts = [f"Sản phẩm: {title}"]
                    if category:
                        parts.append(f"Danh mục: {category}")
                    doc_text = " | ".join(parts)
                else:
                    doc_text = doc.get(document_key) or doc.get("title", "")
                
                doc_text = doc_text[:1000] if doc_text else ""
                pairs.append([query, doc_text])

            # Memory-efficient inference with torch.no_grad()
            with torch.no_grad():
                scores = self.model.predict(
                    pairs,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

            reranked_docs = []
            query_words = set(query.lower().split()) if query else set()

            for doc, score in zip(documents, scores):
                doc_copy = dict(doc)
                doc_copy["reranker_score"] = float(score)
                doc_copy["original_score"] = doc.get("score", 0.0)

                title_words = set(doc.get("title", "").lower().split())
                overlap_count = len(query_words.intersection(title_words))
                lexical_boost = 0.5 * overlap_count
                doc_copy["lexical_boost"] = lexical_boost

                doc_copy["final_score"] = float(score) + 2.0 * doc_copy["original_score"] + lexical_boost
                reranked_docs.append(doc_copy)

            reranked_docs.sort(key=lambda x: x["final_score"], reverse=True)
            return reranked_docs[:top_k]

        finally:
            self._semaphore.release()

    def _fast_vector_fallback(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Ultra-fast fallback (< 1ms) using Qdrant vector score + lexical title boost."""
        query_words = set(query.lower().split()) if query else set()
        fallback_docs = []

        for doc in documents:
            doc_copy = dict(doc)
            orig_score = doc.get("score", 0.5)
            doc_copy["original_score"] = orig_score
            doc_copy["reranker_score"] = orig_score

            title_words = set(doc.get("title", "").lower().split())
            overlap_count = len(query_words.intersection(title_words))
            lexical_boost = 0.3 * overlap_count

            doc_copy["final_score"] = orig_score + lexical_boost
            fallback_docs.append(doc_copy)

        fallback_docs.sort(key=lambda x: x["final_score"], reverse=True)
        return fallback_docs[:top_k]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single pair safely."""
        with torch.no_grad():
            score = self.model.predict([[query, document]])
        return float(score[0])


# ── Singleton Instance ────────────────────────────────────────────────────────
reranker_service = RerankerService()
rerank_service = reranker_service
