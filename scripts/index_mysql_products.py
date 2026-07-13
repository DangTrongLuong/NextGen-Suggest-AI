
import sys
import os
import io
import json
import pymysql
from dotenv import load_dotenv

# Fix Unicode on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.shared.qdrant.service import qdrant_service
from app.features.search.services.embedding_service import embedding_service

def get_mysql_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "230605"),
        database=os.getenv("MYSQL_DATABASE", "ecommerce"),
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4"
    )

def extract_mysql_products():
    """Lấy toàn bộ products active từ MySQL."""
    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            query = """
            SELECT 
                p.*,
                c.name as category_name,
                COALESCE(
                    (SELECT MIN(v.price) FROM product_variants v 
                     WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 
                    0
                ) as min_price,
                GROUP_CONCAT(DISTINCT pv.sku SEPARATOR ', ') as skus
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            LEFT JOIN product_variants pv ON pv.product_id = p.id AND pv.deleted_at IS NULL AND pv.is_active = 1
            WHERE p.is_active = 1 AND p.deleted_at IS NULL
            GROUP BY p.id, c.name
            ORDER BY p.id
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            products = []
            for row in rows:
                # Build document text for embedding
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
                    # Truncate description
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
                        product[attr_k] = attr_v

                # Standard compatibility fields for Qdrant payload
                product["product_id"] = row.get("uuid")
                product["title"] = row.get("name") or ""
                product["category"] = row.get("category_name") or ""
                product["price"] = float(row.get("min_price") or 0)
                product["rating"] = float(row.get("rating_avg") or 0)
                product["reviews"] = int(row.get("rating_count") or 0)
                product["tags"] = [row["category_name"]] if row.get("category_name") else []
                product["document"] = document
                
                products.append(product)
            
            return products
    finally:
        conn.close()


def main(recreate: bool = False):
    print("=" * 60)
    print("[INFO] Extracting products from MySQL...")
    print("=" * 60)
    
    products = extract_mysql_products()
    print(f"[INFO] Found {len(products)} active products in MySQL")
    
    if not products:
        print("[WARN] No products to index!")
        return
    
    # Show sample
    p = products[0]
    print(f"[SAMPLE] product_id (MySQL UUID): {p['product_id']}")
    print(f"[SAMPLE] title: {p['title'][:60]}")
    print(f"[SAMPLE] category: {p['category']}")
    print(f"[SAMPLE] price: {p['price']}")
    
    # Create embeddings
    print(f"[INFO] Creating embeddings for {len(products)} products...")
    documents = [p["document"] for p in products]
    
    embeddings = []
    batch_size = 16
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_embs = embedding_service.embed_documents(batch, batch_size=len(batch), show_progress=False)
        embeddings.extend(batch_embs)
        print(f"  Embedded {min(i + batch_size, len(documents))}/{len(documents)}")
    
    print(f"[INFO] Embedding shape: ({len(embeddings)}, {len(embeddings[0])})")
    
    # Setup Qdrant collection
    vector_size = embedding_service.vector_size
    qdrant_service.create_collection(vector_size=vector_size, recreate=recreate)
    
    # Index to Qdrant
    print("[INFO] Uploading to Qdrant...")
    qdrant_service.upsert_products(
        products=products,
        embeddings=[e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings],
        batch_size=32
    )
    
    # Verify
    info = qdrant_service.get_collection_info()
    print(f"[INFO] Done! Collection info: {info}")
    
    # Sanity check
    print("[INFO] Quick sanity check - search top 3:")
    test_vec = embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])
    results = qdrant_service.search(test_vec, top_k=3)
    for r in results:
        title = r['title'][:60].encode('ascii', 'replace').decode('ascii')
        print(f"  [{r['score']:.4f}] {title} | UUID: {r['product_id']}")
    
    print(f"[DONE] Indexed {len(products)} MySQL products into Qdrant.")
    print("[NOTE] product_id in Qdrant is now MySQL UUID.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index MySQL products vào Qdrant với MySQL UUID")
    parser.add_argument("--recreate", action="store_true", help="Xóa và tạo lại Qdrant collection")
    args = parser.parse_args()
    main(recreate=args.recreate)
