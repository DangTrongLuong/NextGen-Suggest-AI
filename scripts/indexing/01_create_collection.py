"""
Script 01-indexing: Tạo Qdrant collection.

Usage:
    python scripts/indexing/01_create_collection.py [--recreate]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.shared.qdrant.service import qdrant_service
from app.features.search.services.embedding_service import embedding_service


def main(recreate: bool = False):
    print("  Creating Qdrant collection...")
    print(f"   Embedding model : {embedding_service.model_name}")
    print(f"   Vector size     : {embedding_service.vector_size}")
    print(f"   Collection name : {qdrant_service.collection_name}")

    qdrant_service.create_collection(
        vector_size=embedding_service.vector_size,
        recreate=recreate,
    )
    info = qdrant_service.get_collection_info()
    print(f"\n Collection info: {info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="Xóa và tạo lại collection nếu đã tồn tại")
    args = parser.parse_args()
    main(recreate=args.recreate)
