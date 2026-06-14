

import json
import numpy as np
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.features.search.services.embedding_service import embedding_service


def main(
    input_path: str = "data/processed/products_with_documents.jsonl",
    output_path: str = "data/processed/embeddings",
    batch_size: int = 64,
):
    print(f" Loading products from: {input_path}")
    products = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))

    print(f"   Total products : {len(products)}")
    print(f"   Embedding model: {embedding_service.model_name}")
    print(f"   Vector size    : {embedding_service.vector_size}")
    print(f"   Batch size     : {batch_size}")
    print(f"   Device         : {embedding_service.device}")

    documents = [p["search_document"] for p in products]
    product_ids = [p["product_id"] for p in products]

    print("\n Embedding... (this may take a while on CPU)")
    embeddings = embedding_service.embed_documents(documents, batch_size=batch_size)

    print(f"\n Embedding shape: {embeddings.shape}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings.astype(np.float32),
        product_ids=np.array(product_ids, dtype=str),
    )

    size_mb = Path(output_path + ".npz").stat().st_size / 1024 / 1024 if Path(output_path + ".npz").exists() else 0
    print(f" Saved embeddings → {output_path}.npz  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/products_with_documents.jsonl")
    parser.add_argument("--output", default="data/processed/embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    main(args.input, args.output, args.batch_size)
