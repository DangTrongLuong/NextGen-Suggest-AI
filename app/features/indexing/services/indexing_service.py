import pymysql
import logging
from typing import List, Dict, Any

from app.shared.qdrant.service import qdrant_service
from app.features.search.services.embedding_service import embedding_service
from app.config.settings import settings

logger = logging.getLogger(__name__)

class IndexingService:

    def sync_uuids_to_qdrant(self, product_uuids: List[str]) -> Dict[str, Any]:
        if not product_uuids:
            return {"message": "Danh sách UUID trống", "count": 0}

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
            raise Exception(f"Failed to connect to MySQL: {str(e)}")

        try:
            with conn.cursor() as cursor:
                format_strings = ','.join(['%s'] * len(product_uuids))
                query = f"""
                SELECT 
                    p.*,
                    c.name as category_name,
                    COALESCE(
                        (SELECT MIN(v.price) FROM product_variants v 
                         WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 
                        0
                    ) as min_price
                FROM products p
                LEFT JOIN categories c ON p.category_id = c.id
                WHERE p.is_active = 1 AND p.deleted_at IS NULL AND p.uuid IN ({format_strings})
                GROUP BY p.id, c.name
                """
                cursor.execute(query, tuple(product_uuids))
                rows = cursor.fetchall()

                if not rows:
                    logger.info(f"No active products found in MySQL for UUIDs: {product_uuids}")
                    return {"message": "Không tìm thấy sản phẩm hoạt động để index", "count": 0}

                import json
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
                    
                    product = {}
                    for k, v in row.items():
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            product[k] = v
                        elif hasattr(v, 'isoformat'):  # datetime
                            product[k] = v.isoformat()
                        elif isinstance(v, bytes):
                            product[k] = v.decode('utf-8', errors='ignore')
                        else:
                            try:
                                product[k] = float(v)
                            except:
                                product[k] = str(v)
                                
                        # Parse JSON fields if they are valid JSON strings
                        if isinstance(product[k], str):
                            stripped = product[k].strip()
                            if (stripped.startswith('{') and stripped.endswith('}')) or (stripped.startswith('[') and stripped.endswith(']')):
                                try:
                                    product[k] = json.loads(stripped)
                                except:
                                    pass

                    # If attributes is a dictionary, unpack its keys to the top-level product dictionary
                    if "attributes" in product and isinstance(product["attributes"], dict):
                        for attr_k, attr_v in product["attributes"].items():
                            if isinstance(attr_v, str):
                                stripped = attr_v.strip()
                                if (stripped.startswith('{') and stripped.endswith('}')) or (stripped.startswith('[') and stripped.endswith(']')):
                                    try:
                                        attr_v = json.loads(stripped)
                                    except:
                                        pass
                            product[attr_k] = attr_v

                    # Standard compatibility fields for Qdrant payload
                    product["product_id"] = row.get("uuid")
                    product["title"] = row.get("name") or ""
                    product["category"] = row.get("category_name") or ""
                    product["price"] = float(row.get("min_price") or 0)
                    product["rating"] = product.get("rating") or float(row.get("rating_avg") or 0)
                    product["reviews"] = int(row.get("rating_count") or 0)
                    product["tags"] = product.get("tags") or ([row["category_name"]] if row.get("category_name") else [])
                    product["search_document"] = document
                    
                    products.append(product)
                    documents.append(document)

                embeddings = embedding_service.embed_documents(documents, batch_size=len(documents), show_progress=False)
                
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
            raise e
        finally:
            conn.close()

    def delete_uuids_from_qdrant(self, product_uuids: List[str]) -> Dict[str, Any]:
        if not product_uuids:
            return {"message": "Danh sách UUID trống", "count": 0}
        
        try:
            for uuid_str in product_uuids:
                qdrant_service.delete_product(uuid_str)
            logger.info(f"Successfully deleted {len(product_uuids)} products from Qdrant.")
            return {"message": f"Đã xóa {len(product_uuids)} point vector khỏi Qdrant", "count": len(product_uuids)}
        except Exception as e:
            logger.error(f"Error deleting products index: {e}")
            raise e

indexing_service = IndexingService()
