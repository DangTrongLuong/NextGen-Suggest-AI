"""
Reranker Service - Cross-encoder reranking.

Dùng BAAI/bge-reranker-base để score lại (query, document) pairs.
Cross-encoder chính xác hơn bi-encoder nhưng chậm hơn → chỉ dùng trên top candidates.
"""
from sentence_transformers import CrossEncoder
import torch
from typing import List, Dict
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RerankerService:
    """Singleton cross-encoder reranker service."""

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

        logger.info(f" Loading reranker: {self.model_name}")
        logger.info(f"   Device: {self.device}")

        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=512,
            cache_folder=settings.HF_HOME,
            trust_remote_code=True
        )

        logger.info("    Reranker model ready")
        self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
        document_key: str = "search_document"
    ) -> List[Dict]:
        """
        Rerank danh sách candidates bằng cross-encoder.

        Args:
            query: Câu query của user
            documents: List dicts từ Qdrant search (có 'search_document' hoặc 'title')
            top_k: Số kết quả sau reranking
            document_key: Key trong dict để lấy text cho reranker

        Returns:
            Danh sách đã sắp xếp lại theo reranker score (cao → thấp)

        Flow:
            (query, doc1_text) → score_1
            (query, doc2_text) → score_2
            ...
            Sort by score → top_k
        """
        if not documents:
            return []

        # Tạo (query, document_text) pairs
        pairs = []
        for doc in documents:
            if document_key == "short_context":
                title = doc.get("title", "")
                category = doc.get("category", "")

                # Clean representation (Title + Category) for fast CPU reranking
                parts = [f"Sản phẩm: {title}"]
                if category:
                    parts.append(f"Danh mục: {category}")
                
                doc_text = " | ".join(parts)
            else:
                doc_text = doc.get(document_key) or doc.get("title", "")
            
            # Giới hạn độ dài text (512 tokens)
            doc_text = doc_text[:1000] if doc_text else ""
            pairs.append([query, doc_text])

        # Score tất cả pairs cùng lúc (batch)
        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Gắn các điểm số và tính toán Hybrid Score
        reranked_docs = []
        query_words = set(query.lower().split()) if query else set()

        for doc, score in zip(documents, scores):
            doc_copy = dict(doc)
            doc_copy["reranker_score"] = float(score)
            doc_copy["original_score"] = doc.get("score", 0.0)

            # Tính toán Lexical Boost (Trùng khớp từ khóa trong tiêu đề)
            title_words = set(doc.get("title", "").lower().split())
            overlap_count = len(query_words.intersection(title_words))
            lexical_boost = 0.5 * overlap_count
            doc_copy["lexical_boost"] = lexical_boost

            # Công thức Hybrid Score: Cross-Encoder + (2.0 * Vector Similarity) + Lexical Boost
            doc_copy["final_score"] = float(score) + 2.0 * doc_copy["original_score"] + lexical_boost
            
            reranked_docs.append(doc_copy)

        # Sắp xếp theo final_score giảm dần
        reranked_docs.sort(key=lambda x: x["final_score"], reverse=True)

        return reranked_docs[:top_k]

    def score_pair(self, query: str, document: str) -> float:
        """Score một cặp (query, document) đơn lẻ."""
        score = self.model.predict([[query, document]])
        return float(score[0])


# ── Singleton Instance ────────────────────────────────────────────────────────
reranker_service = RerankerService()
rerank_service = reranker_service
