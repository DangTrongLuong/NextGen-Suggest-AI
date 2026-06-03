"""
Script 02: Tạo search document từ metadata sản phẩm.

Usage:
    python scripts/data_processing/02_format_documents.py [--model-type e5|bge]
"""
import json
import argparse
from pathlib import Path


def format_document(product: dict, model_type: str = "e5") -> str:
    """
    Ghép title + description + category + tags thành một search document.

    model_type:
        "e5"  → thêm "passage: " prefix (bắt buộc cho multilingual-e5-base)
        "bge" → không cần prefix (dùng cho BAAI/bge-m3)
    """
    title = product.get("title", "").strip()
    desc = product.get("description", "").strip()
    category = product.get("category", "").strip()
    tags = product.get("tags", [])

    tags_str = ", ".join(tags) if tags else ""

    # Bỏ description nếu quá ngắn (spam/noise)
    if len(desc) < 10:
        desc = ""

    parts = [title]
    if desc:
        parts.append(desc)
    if category:
        parts.append(f"Danh mục: {category}")
    if tags_str:
        parts.append(f"Tags: {tags_str}")

    content = ". ".join(parts)

    if model_type == "e5":
        return f"passage: {content}"
    return content          # bge-m3: no prefix needed


def format_all_products(
    input_path: str = "data/processed/products_cleaned.jsonl",
    output_path: str = "data/processed/products_with_documents.jsonl",
    model_type: str = "e5",
):
    print(f" Input : {input_path}")
    print(f" Model type: {model_type}")

    products = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))

    print(f"   Loaded {len(products)} products")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in products:
            p["search_document"] = format_document(p, model_type)
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Preview
    sample = products[0] if products else {}
    print(f"\n Sample document:\n   {sample.get('search_document', '')[:200]}")
    print(f"\n   Saved {len(products)} docs → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["e5", "bge"], default="e5")
    parser.add_argument("--input", default="data/processed/products_cleaned.jsonl")
    parser.add_argument("--output", default="data/processed/products_with_documents.jsonl")
    args = parser.parse_args()

    format_all_products(args.input, args.output, args.model_type)
