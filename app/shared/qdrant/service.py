"""
Qdrant Service - Quản lý Vector Database.

Các chức năng:
  - Tạo/xóa collection
  - Upload vectors (upsert)
  - ANN search với optional filters
  - Lấy vector theo product_id
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PointStruct, PointIdsList,
    Filter, FieldCondition, MatchValue, MatchText, Range,
    PayloadSchemaType, UpdateStatus
)
from typing import List, Dict, Any, Optional, Tuple
import uuid
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)


def _product_id_to_uuid(product_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(product_id)))


class QdrantService:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info(f"  Connecting to Qdrant: {settings.QDRANT_URL}")

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False,
            https=True, 
            timeout=30 
        )

        self.collection_name = settings.QDRANT_COLLECTION
        logger.info(f"   Collection: {self.collection_name}")
        logger.info("    Qdrant connected")
        self._initialized = True

    # ── Collection Management ─────────────────────────────────────────────────

    def create_collection(self, vector_size: int, recreate: bool = False):

        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in existing:
            if recreate:
                logger.warning(f"  Recreating collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"   Collection '{self.collection_name}' already exists. Skipping.")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE 
            )
        )

        # Tạo payload indexes để filter nhanh
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="price",
            field_schema=PayloadSchemaType.FLOAT
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="product_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

        logger.info(
            f" Created collection '{self.collection_name}' "
            f"with vector_size={vector_size}, distance=COSINE"
        )

    def get_collection_info(self) -> Dict:
        """Lấy thông tin collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            vectors_cnt = getattr(info, "vectors_count", getattr(info, "indexed_vectors_count", 0))
            return {
                "name": self.collection_name,
                "vectors_count": vectors_cnt,
                "points_count": info.points_count,
                "status": str(info.status),
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance)
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Data Ingestion ────────────────────────────────────────────────────────

    def upsert_products(
        self,
        products: List[Dict],
        embeddings: List[List[float]],
        batch_size: int = 100
    ):

        if len(products) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(products)} products vs {len(embeddings)} embeddings"
            )

        points = []
        for product, embedding in zip(products, embeddings):
            pid = str(product.get("product_id", ""))
            point_uuid = _product_id_to_uuid(pid)

            # Construct exact payload structure requested by the user (23 fields only)
            payload = {
                "product_id": pid,
                "product_name": product.get("product_name") or product.get("title") or product.get("name") or "",
                "description": product.get("description") or "",
                "category_name": product.get("category_name") or product.get("category") or "",
                "category_slug": product.get("category_slug") or product.get("slug") or "",
                "category_url": product.get("category_url") or "",
                "category_image_url": product.get("category_image_url") or "",
                "brand": product.get("brand") or "",
                "price": float(product.get("price", 0)) if product.get("price") is not None else 0.0,
                "original_price": float(product.get("original_price", 0)) if product.get("original_price") is not None else 0.0,
                "currency": product.get("currency") or "VND",
                "stock": int(product.get("stock", 0)) if product.get("stock") is not None else 0,
                "rating": float(product.get("rating", 0)) if product.get("rating") is not None else 0.0,
                "reviews_count": int(product.get("reviews_count") or product.get("reviews") or 0),
                "thumbnail_url": product.get("thumbnail_url") or product.get("image_url") or "",
                "image_urls": product.get("image_urls") or [],
                "tags": product.get("tags") or [],
                "color": product.get("color") or "",
                "size": product.get("size") or "",
                "material": product.get("material") or "",
                "product_url": product.get("product_url") or "",
                "source": product.get("source") or "",
                "searchable_text": product.get("searchable_text") or product.get("search_document") or product.get("document") or "",
            }

            point = PointStruct(
                id=point_uuid,
                vector=embedding,
                payload=payload
            )
            points.append(point)

        # Upload theo batch để tránh timeout
        total_uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
            if result.status == UpdateStatus.COMPLETED:
                total_uploaded += len(batch)

        logger.info(f" Indexed {total_uploaded}/{len(points)} products into Qdrant")

    # ── Search ────────────────────────────────────────────────────────────────

    def _map_payload_to_compatibility(self, payload: Dict) -> Dict:
        if not payload:
            return {}
        res = payload.copy()
        res["id"] = payload.get("product_id")
        res["title"] = payload.get("product_name") or ""
        res["category"] = payload.get("category_name") or ""
        res["image_url"] = payload.get("thumbnail_url") or ""
        res["search_document"] = payload.get("searchable_text") or ""
        res["price"] = payload.get("price") or 0.0
        res["reviews"] = payload.get("reviews_count") or 0
        return res

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        top_k: int = 50,
        category_filter: Optional[str] = None,
        price_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        exclude_ids: Optional[List[str]] = None,
        tag_filter: Optional[str] = None,
        tags_filter: Optional[List[str]] = None,
        with_vectors: bool = False
    ) -> List[Dict]:

        # Build filter conditions
        conditions = []

        if category_filter:
            conditions.append(
                FieldCondition(
                    key="category_name",
                    match=MatchText(text=category_filter)
                )
            )

        if tag_filter:
            conditions.append(
                FieldCondition(
                    key="tags",
                    match=MatchValue(value=tag_filter)
                )
            )

        if tags_filter:
            for tag in tags_filter:
                conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchValue(value=tag)
                    )
                )

        if price_range:
            min_p, max_p = price_range
            range_args = {}
            if min_p is not None:
                range_args["gte"] = min_p
            if max_p is not None:
                range_args["lte"] = max_p
            if range_args:
                conditions.append(
                    FieldCondition(key="price", range=Range(**range_args))
                )

        query_filter = Filter(must=conditions) if conditions else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=with_vectors
        )
        results = response.points

        output = []
        for r in results:
            p = r.payload
            pid = p.get("product_id", "")

            # Loại trừ product IDs không mong muốn
            if exclude_ids and pid in exclude_ids:
                continue

            item = self._map_payload_to_compatibility(p)
            item["score"] = float(r.score)
            if with_vectors:
                item["vector"] = r.vector
            output.append(item)

        return output

    def delete_product(self, product_id: str):
        """Xóa point vector của một sản phẩm khỏi Qdrant."""
        point_uuid = _product_id_to_uuid(product_id)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[point_uuid]),
            wait=True
        )
        logger.info(f"Deleted product vector point '{point_uuid}' (MySQL UUID: {product_id}) from Qdrant")

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Lấy thông tin sản phẩm theo product_id."""
        point_uuid = _product_id_to_uuid(product_id)
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_uuid],
            with_payload=True,
            with_vectors=False
        )
        if results:
            p = results[0].payload
            return self._map_payload_to_compatibility(p)
        return None

    def get_product_vector(self, product_id: str) -> Optional[List[float]]:
        """Lấy embedding vector của một sản phẩm (dùng cho similar products)."""
        point_uuid = _product_id_to_uuid(product_id)
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_uuid],
            with_payload=False,
            with_vectors=True
        )
        if results and results[0].vector:
            return results[0].vector
        return None

    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:

        query_lower = query.lower()
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        scored = []
        for point in results:
            p = point.payload
            comp_p = self._map_payload_to_compatibility(p)
            title = comp_p.get("title", "").lower()
            desc = comp_p.get("description", "").lower()
            tags = " ".join(comp_p.get("tags", [])).lower()
            category = comp_p.get("category", "").lower()

            full_text = f"{title} {desc} {tags} {category}"

            # Đếm số lần query words xuất hiện
            query_words = query_lower.split()
            score = sum(full_text.count(w) for w in query_words)

            if score > 0:
                item = comp_p.copy()
                item["score"] = float(score)
                scored.append(item)

        # Sort by score, normalize
        scored.sort(key=lambda x: x["score"], reverse=True)
        max_score = scored[0]["score"] if scored else 1
        for item in scored:
            item["score"] = item["score"] / max_score

        return scored[:top_k]


# ── Singleton Instance ────────────────────────────────────────────────────────
qdrant_service = QdrantService()
