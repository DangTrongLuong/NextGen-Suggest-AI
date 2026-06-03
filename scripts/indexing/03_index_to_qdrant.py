
import json
import numpy as np
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.qdrant_service import qdrant_service
from app.services.embedding_service import embedding_service


def main(
    products_path: str = "data/processed/products_with_documents.jsonl",
    embeddings_path: str = "data/processed/embeddings.npz",
    batch_size: int = 100,
    recreate_collection: bool = False,
):
    # ── Load products ────────────────────────────────────────────────────────
    print(f" Loading products: {products_path}")
    products = []
    with open(products_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    print(f"   Products loaded: {len(products)}")

    # ── Load embeddings ───────────────────────────────────────────────────────
    npz_path = embeddings_path if embeddings_path.endswith(".npz") else embeddings_path + ".npz"
    print(f" Loading embeddings: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    embed_ids = list(data["product_ids"])
    print(f"   Embeddings shape: {embeddings.shape}")

    # Align products với embeddings theo product_id
    id_to_product = {p["product_id"]: p for p in products}
    aligned_products = []
    aligned_embeddings = []
    missing = 0
    for i, pid in enumerate(embed_ids):
        if pid in id_to_product:
            aligned_products.append(id_to_product[pid])
            aligned_embeddings.append(embeddings[i].tolist())
        else:
            missing += 1

    if missing:
        print(f"    {missing} embeddings không có product metadata, bỏ qua")

    print(f"   Ready to index: {len(aligned_products)} products")

    # ── Create collection ─────────────────────────────────────────────────────
    vector_size = embedding_service.vector_size
    qdrant_service.create_collection(
        vector_size=vector_size,
        recreate=recreate_collection,
    )

    # ── Index ─────────────────────────────────────────────────────────────────
    print(f"\n Uploading to Qdrant (batch_size={batch_size})...")
    qdrant_service.upsert_products(
        products=aligned_products,
        embeddings=aligned_embeddings,
        batch_size=batch_size,
    )

    # ── Verify ────────────────────────────────────────────────────────────────
    info = qdrant_service.get_collection_info()
    print(f"\n Done! Collection info: {info}")

    # Quick sanity-check search
    print("\n Quick sanity-check search...")
    test_vec = aligned_embeddings[0]
    results = qdrant_service.search(test_vec, top_k=3)
    for r in results:
        print(f"   [{r['score']:.4f}] {r['title'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", default="data/processed/products_with_documents.jsonl")
    parser.add_argument("--embeddings", default="data/processed/embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--recreate", action="store_true",
                        help="Xóa và tạo lại collection")
    args = parser.parse_args()

    main(args.products, args.embeddings, args.batch_size, args.recreate)
