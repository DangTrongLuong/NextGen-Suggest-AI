"""
Script 01: Làm sạch và chuẩn hóa dữ liệu sản phẩm.

Usage:
    python scripts/data_processing/01_clean_data.py
"""
import pandas as pd
import json
import re
import unicodedata
from pathlib import Path


def remove_html_tags(text: str) -> str:
    """Xóa HTML tags khỏi text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", " ", text).strip()


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode, bỏ khoảng trắng thừa."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)  
    text = re.sub(r"\s+", " ", text)             
    return text.strip()


def parse_tags(raw: object) -> list:
    """Parse tags từ string hoặc list."""
    if isinstance(raw, list):
        return [normalize_text(t) for t in raw if t]
    if isinstance(raw, str):
        return [normalize_text(t) for t in raw.split(",") if t.strip()]
    return []


def convert_price_to_vnd(price_usd: float) -> int:


    if price_usd <= 0:
        return 0

    vnd = price_usd * 10000

    return int(round(vnd / 1000) * 1000)


def safe_float(value, default=0.0):


    try:
        if pd.isna(value):
            return default

        if hasattr(value, "year"):
            return default

        if isinstance(value, str):

            value = (
                value.replace(",", "")
                     .replace("$", "")
                     .replace("₫", "")
                     .replace("VNĐ", "")
                     .strip()
            )

            if value == "":
                return default

        value = float(value)

        if value < 0 or value > 10000:
            return default

        return value

    except:
        return default

def clean_products(input_path: str, output_path: str):
    print(f" Loading: {input_path}")

    # Hỗ trợ XLSX / CSV / JSONL
    if input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path)
        df = df.fillna("")

    elif input_path.endswith(".csv"):
        df = pd.read_csv(
        input_path,
        encoding="utf-8-sig"
    )

    elif input_path.endswith(".jsonl"):
        df = pd.read_json(
        input_path,
        lines=True
    )

    else:
        raise ValueError(
        f"Unsupported file format: {input_path}"
        )

    print(f"   Raw rows: {len(df)}")

    cleaned = []
    skipped = 0

    for i, row in df.iterrows():
        title = normalize_text(remove_html_tags(str(row.get("title", ""))))

        # Bỏ qua sản phẩm không có title
        if not title or len(title) < 3:
            skipped += 1
            continue
        
        raw_price = safe_float(row.get("price"))
        product = {
            "product_id": str(row.get("product_id", f"prod_{i:06d}")),
            "title": title,
            "description": normalize_text(remove_html_tags(str(row.get("description", "")))),
            "category": normalize_text(str(row.get("category", ""))),
            "tags": parse_tags(row.get("tags", "")),
            "price": convert_price_to_vnd(raw_price),
            "image_url": str(row.get("image_url", "") or ""),
        }
        cleaned.append(product)

    # Bỏ duplicate product_ids
    seen_ids = set()
    unique = []
    for p in cleaned:
        if p["product_id"] not in seen_ids:
            seen_ids.add(p["product_id"])
            unique.append(p)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in unique:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"    Cleaned: {len(unique)} products (skipped {skipped + len(cleaned) - len(unique)})")
    print(f"   Saved → {output_path}")


if __name__ == "__main__":
    clean_products(
        input_path="data/raw/products_vi.xlsx",
        output_path="data/processed/products_cleaned.jsonl",
    )
