import pymysql
from fastapi import APIRouter, HTTPException
from typing import List
import logging

from app.features.indexing.schemas.indexing_schemas import IndexSyncRequest
from app.shared.qdrant.service import qdrant_service
from app.features.search.services.embedding_service import embedding_service
from app.config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/indexing/sync", summary="Đồng bộ vector Qdrant cho danh sách Product UUIDs")
async def sync_products_index(request: IndexSyncRequest):
    if not request.product_uuids:
        return {"message": "Danh sách UUID trống"}
    
    try:
        conn = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4"
        )
    except Exception as e:
        logger.error(f"Failed to connect to MySQL database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to MySQL: {str(e)}")

    try:
        with conn.cursor() as cursor:
            # Query matching rows
            format_strings = ','.join(['%s'] * len(request.product_uuids))
            query = f"""
            SELECT 
                p.uuid,
                p.name,
                p.description,
                p.short_description,
                p.brand,
                p.image_url,
                p.rating_avg,
                p.rating_count,
                p.views,
                p.slug,
                c.name as category_name,
                COALESCE(
                    (SELECT MIN(v.price) FROM product_variants v 
                     WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 
                    0
                ) as min_price
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            WHERE p.is_active = 1 AND p.deleted_at IS NULL AND p.uuid IN ({format_strings})
            GROUP BY p.uuid, p.name, p.description, p.short_description, p.brand, 
                     p.image_url, p.rating_avg, p.rating_count, p.views, p.slug, c.name, p.id
            """
            cursor.execute(query, tuple(request.product_uuids))
            rows = cursor.fetchall()

            if not rows:
                logger.info(f"No active products found in MySQL for UUIDs: {request.product_uuids}")
                return {"message": "Không tìm thấy sản phẩm hoạt động để index", "count": 0}

            products = []
            documents = []
            for row in rows:
                doc_parts = []
                if row.get("name"):
                    doc_parts.append(row["name"])
                if row.get("category_name"):
                    doc_parts.append(f"Category: {row['category_name']}")
                if row.get("brand"):
                    doc_parts.append(f"Brand: {row['brand']}")
                if row.get("short_description"):
                    doc_parts.append(row["short_description"])
                if row.get("description"):
                    doc_parts.append(row["description"][:300])

                document = " | ".join(doc_parts)
                
                product = {
                    "product_id": row["uuid"],  # UUID as product_id
                    "title": row["name"] or "",
                    "description": row.get("description") or "",
                    "category": row.get("category_name") or "",
                    "brand": row.get("brand") or "",
                    "image_url": row.get("image_url") or "",
                    "price": float(row.get("min_price") or 0),
                    "rating": float(row.get("rating_avg") or 0),
                    "reviews": int(row.get("rating_count") or 0),
                    "views": int(row.get("views") or 0),
                    "tags": [row["category_name"]] if row.get("category_name") else [],
                    "search_document": document, 
                }
                products.append(product)
                documents.append(document)

            embeddings = embedding_service.embed_documents(documents, batch_size=len(documents), show_progress=False)
            
            # Setup collection size verification
            vector_size = embedding_service.vector_size
            qdrant_service.create_collection(vector_size=vector_size, recreate=False)
            
            qdrant_service.upsert_products(
                products=products,
                embeddings=[e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings],
                batch_size=len(products)
            )
            
            logger.info(f"Successfully indexed {len(products)} products into Qdrant.")
            return {"message": f"Đã đồng bộ {len(products)} sản phẩm", "count": len(products)}
    except Exception as e:
        logger.error(f"Error syncing products: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.post("/indexing/delete", summary="Xóa point vector Qdrant cho danh sách Product UUIDs")
async def delete_products_index(request: IndexSyncRequest):
    if not request.product_uuids:
        return {"message": "Danh sách UUID trống"}
    
    try:
        for uuid_str in request.product_uuids:
            qdrant_service.delete_product(uuid_str)
        return {"message": f"Đã xóa {len(request.product_uuids)} point vector khỏi Qdrant"}
    except Exception as e:
        logger.error(f"Error deleting products index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
