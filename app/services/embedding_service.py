
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List, Union
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:


    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model_name = settings.EMBEDDING_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f" Loading embedding model: {self.model_name}")
        logger.info(f"   Device: {self.device}")

        import os
        os.environ["HF_HOME"] = settings.HF_HOME

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=settings.HF_HOME
        )

        # Detect model type để áp dụng prefix đúng
        name_lower = self.model_name.lower()
        if "e5" in name_lower:
            self.model_type = "e5"
        else:
            self.model_type = "bge"

        logger.info(f"   Model type: {self.model_type}")
        logger.info(f"   Vector size: {self.vector_size}")
        logger.info(f"    Embedding model ready")

        self._initialized = True

    # ── Prefix Helpers ────────────────────────────────────────────────────────

    def _format_query(self, text: str) -> str:
        if self.model_type == "e5":
            return f"query: {text}"
        return text  

    def _format_passage(self, text: str) -> str:
        if self.model_type == "e5":
            if not text.startswith("passage: "):
                return f"passage: {text}"
        return text  

    # ── Public Methods ────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> List[float]:

        processed = self._format_query(query.strip())
        embedding = self.model.encode(
            processed,
            normalize_embeddings=True,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        processed = [self._format_query(q.strip()) for q in queries]
        embeddings = self.model.encode(
            processed,
            normalize_embeddings=True,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return embeddings

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:

        processed = [self._format_passage(t) for t in texts]
        embeddings = self.model.encode(
            processed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_tensor=False
        )
        return embeddings

    @property
    def vector_size(self) -> int:
        """Dimension của vector output."""
        return self.model.get_embedding_dimension()


# ── Singleton Instance ────────────────────────────────────────────────────────
embedding_service = EmbeddingService()
