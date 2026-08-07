import io
import re
import httpx
import pandas as pd
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

def find_excel_val(row_dict: Dict[str, Any], *target_keys: str) -> Any:
    for key, val in row_dict.items():
        key_str = str(key).lower().replace("_", "").replace(" ", "").strip()
        for tk in target_keys:
            tk_str = tk.lower().replace("_", "").replace(" ", "").strip()
            if key_str == tk_str:
                return val
    return None

EXCEL_FIELD_MAPPINGS = {
    # Product fields (Vietnamese & English)
    "tên sản phẩm": "name", "tên": "name", "sản phẩm": "name", "ten san pham": "name", "ten": "name",
    "product_name": "name", "product name": "name", "title": "name",
    
    "giá bán lẻ": "basePrice", "giá bán": "basePrice", "giá": "basePrice", "đơn giá": "basePrice", 
    "gia ban le": "basePrice", "gia ban": "basePrice", "gia": "basePrice", "don gia": "basePrice",
    "price": "basePrice", "base_price": "basePrice", "base price": "basePrice",
    
    "giá gốc": "comparePrice", "giá so sánh": "comparePrice", "giá niêm yết": "comparePrice", 
    "gia goc": "comparePrice", "gia so sanh": "comparePrice", "gia niem yet": "comparePrice",
    "original_price": "comparePrice", "original price": "comparePrice", "compare_price": "comparePrice", "compare price": "comparePrice",
    
    "giá vốn": "costPrice", "gia von": "costPrice", "cost_price": "costPrice", "cost price": "costPrice",
    
    "số lượng": "stock", "tồn kho": "stock", "tồn": "stock", "tồn kho thực tế": "stock", 
    "so luong": "stock", "ton kho": "stock", "ton": "stock", "ton kho thuc te": "stock",
    "stock": "stock", "quantity": "stock", "qty": "stock",
    
    "mã sản phẩm": "sku", "mã": "sku", "sku": "sku", "ma san pham": "sku", "ma": "sku",
    
    "mô tả": "description", "mô tả sản phẩm": "description", "mo ta": "description", "mo ta san pham": "description",
    "description": "description",
    
    "mô tả ngắn": "shortDescription", "mo ta ngan": "shortDescription", "short_description": "shortDescription", "short description": "shortDescription",
    
    "thương hiệu": "brand", "nhãn hiệu": "brand", "hãng": "brand", "thuong hieu": "brand", "nhan hieu": "brand", "hang": "brand",
    "brand": "brand", "manufacturer": "brand",
    
    "đường dẫn sản phẩm": "productUrl", "đường dẫn": "productUrl", "url sản phẩm": "productUrl", 
    "duong dan san pham": "productUrl", "duong dan": "productUrl", "url san pham": "productUrl",
    "product_url": "productUrl", "product url": "productUrl",
    
    "màu": "color", "màu sắc": "color", "mau": "color", "mau sac": "color",
    "color": "color", "colour": "color",
    
    "kích thước": "size", "kích cỡ": "size", "kich thuoc": "size", "kich co": "size",
    "size": "size",
    
    "ảnh đại diện": "imageUrl", "ảnh": "imageUrl", "hình ảnh": "imageUrl", 
    "anh dai dien": "imageUrl", "anh": "imageUrl", "hinh anh": "imageUrl",
    "thumbnail_url": "imageUrl", "thumbnail": "imageUrl", "image_url": "imageUrl", "image url": "imageUrl",
    
    "danh sách ảnh": "imageUrls", "các ảnh khác": "imageUrls", 
    "danh sach anh": "imageUrls", "cac anh khac": "imageUrls",
    "image_urls": "imageUrls", "image urls": "imageUrls", "images": "imageUrls",
    
    "danh mục": "categoryId", "danh mục cha": "parentId", "danh muc": "categoryId", "danh muc cha": "parentId",
    "category_id": "categoryId", "category id": "categoryId", "category_name": "categoryId", "category name": "categoryId",
    "parent_id": "parentId", "parent id": "parentId",
    "category_main": "categoryMain", "category main": "categoryMain", "danh mục chính": "categoryMain", "danh muc chinh": "categoryMain",
    
    # Category fields
    "tên danh mục": "name", "category_name": "name", "category name": "name",
    "slug": "slug", 
    "thứ tự hiển thị": "sortOrder", "thu tu hien thi": "sortOrder", "display_order": "sortOrder", "display order": "sortOrder", "sort_order": "sortOrder",
    "trạng thái": "isActive", "trang thai": "isActive", "status": "isActive", "active": "isActive"
}

class ExcelImportService:
    def __init__(self):
        import os
        models_dir = os.path.abspath(settings.HF_HOME)
        os.makedirs(models_dir, exist_ok=True)
        os.environ["HF_HOME"] = models_dir

        logger.info("Initializing ExcelImportService with sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=models_dir)

    def get_schema_fields(self, entity_type: str, auth_header: str) -> List[str]:
        """Fetch schema fields dynamically from Spring Boot."""
        url = f"{settings.SPRING_BOOT_URL}/api/v1/admin/import/schema/{entity_type}"
        headers = {}
        if auth_header:
            headers["Authorization"] = auth_header
        
        try:
            with httpx.Client() as client: 
                response = client.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    fields = data.get("data", {}).get("fields", [])
                    if fields:
                        logger.info(f"Successfully fetched dynamic schema for '{entity_type}': {fields}")
                        return fields
                logger.error(f"Failed to fetch schema. Status code: {response.status_code}. Using fallback.")
        except Exception as e:
            logger.error(f"Error calling schema API on Spring Boot: {e}. Using fallback.")
        
        # Fallback schema if Spring Boot is unreachable
        if entity_type == "product":
            return ["name", "categoryId", "shortDescription", "description", "brand", "sku", "stock", "basePrice", "comparePrice", "costPrice", "taxRate", "weight", "color", "size", "imageUrl", "attributes", "isActive", "isFeatured", "isNew", "isBestSeller", "seoTitle", "seoDescription", "imageUrls", "productUrl"]
        else:
            return ["name", "slug", "description", "imageUrl", "icon", "parentId", "sortOrder", "isActive", "seoTitle", "seoDescription"]

    def fuzzy_match_columns(self, excel_columns: List[str], target_fields: List[str]) -> Dict[str, str]:
        """
        Map Excel columns to DTO target fields uniquely.
        Priority: RapidFuzz score >= 80 first, then SentenceTransformer similarity fallback.
        """
        mapping = {}
        if not excel_columns or not target_fields:
            return mapping

        # Fields that should only match with high confidence (e.g. RapidFuzz score >= 80)
        # We exclude them from SentenceTransformer matching to avoid random semantic assignments.
        high_confidence_only_fields = {
            "isactive", "isfeatured", "isnew", "isbestseller", 
            "weight", "taxrate", "costprice", "attributes"
        }

        # Pre-compute all score pairs to enable greedy unique matching
        pairs = []
        for col in excel_columns:
            col_lower = col.lower().strip()
            translated_col = EXCEL_FIELD_MAPPINGS.get(col_lower, col_lower)
            col_clean = translated_col.replace("_", "").replace(" ", "").strip()
            
            for field in target_fields:
                field_clean = field.lower().replace("_", "").replace(" ", "").strip()
                
                score = fuzz.ratio(col_clean, field_clean)
                if score >= 80:
                    pairs.append((col, field, score, True))

        # SentenceTransformers for remaining unmatched columns
        unmatched_cols = [c for c in excel_columns if not any(p[0] == c for p in pairs)]
        if unmatched_cols:
            # Filter out high_confidence_only_fields from SentenceTransformers search space
            semantic_target_fields = [f for f in target_fields if f.lower() not in high_confidence_only_fields]
            
            if semantic_target_fields:
                # For SentenceTransformer, we also translate unmatched columns if they exist in aliases
                translated_unmatched = [EXCEL_FIELD_MAPPINGS.get(c.lower().strip(), c) for c in unmatched_cols]
                col_embs = self.model.encode(translated_unmatched, convert_to_tensor=True)
                field_embs = self.model.encode(semantic_target_fields, convert_to_tensor=True)
                cos_sim = util.cos_sim(col_embs, field_embs)
                
                for i, col in enumerate(unmatched_cols):
                    for j, field in enumerate(semantic_target_fields):
                        score = cos_sim[i][j].item() * 100
                        pairs.append((col, field, score, False))

        # Greedily assign from highest score down to avoid duplicates
        pairs.sort(key=lambda x: x[2], reverse=True)
        assigned_cols = set()
        assigned_fields = set()
        
        for col, field, score, is_fuzz in pairs:
            if col in assigned_cols or field in assigned_fields:
                continue
            # Raise SentenceTransformer threshold to 60 to prevent junk mappings (like material -> weight)
            threshold = 80 if is_fuzz else 60
            if score >= threshold:
                mapping[col] = field
                assigned_cols.add(col)
                assigned_fields.add(field)
                
        logger.info(f"Mapped Excel headers: {mapping}")
        return mapping

    def detect_cycle_and_sort_categories(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """
        Cycle detection using DFS.
        If no cycle, returns categories in Topological Order (parents before children).
        """
        def clean_id_str(val):
            if val is None:
                return None
            val_str = str(val).strip()
            if val_str.lower() in ('nan', 'none', ''):
                return None
            if val_str.endswith('.0'):
                return val_str[:-2]
            return val_str

        row_map = {clean_id_str(row.get('category_id')): row for row in rows if row.get('category_id') is not None}
        slug_map = {row.get('slug'): row for row in rows if row.get('slug')}
        name_map = {row.get('name'): row for row in rows if row.get('name')}

        visited = {} # identifier -> state: 0=unvisited, 1=visiting, 2=visited
        sorted_rows = []
        cycle_detected = [False]
        cycle_path = []

        def get_parent_row(row):
            p_id = clean_id_str(row.get('parent_id'))
            p_slug = row.get('parent_slug')
            p_name = row.get('parent_name')

            parent_row = None
            if p_id and p_id in row_map:
                parent_row = row_map[p_id]
            elif p_slug and p_slug in slug_map:
                parent_row = slug_map[p_slug]
            elif p_name and p_name in name_map:
                parent_row = name_map[p_name]

            if parent_row:
                # Avoid self-referencing loops
                curr_id = clean_id_str(row.get('category_id'))
                p_row_id = clean_id_str(parent_row.get('category_id'))

                if curr_id and p_row_id:
                    if curr_id == p_row_id:
                        return None
                else:
                    curr_slug = row.get('slug')
                    p_row_slug = parent_row.get('slug')
                    if curr_slug and p_row_slug and curr_slug == p_row_slug:
                        return None

                    curr_name = row.get('name')
                    p_row_name = parent_row.get('name')
                    if curr_name and p_row_name and curr_name == p_row_name:
                        return None

            return parent_row

        def get_key(row):
            c_id = clean_id_str(row.get('category_id'))
            slug = row.get('slug')
            name = row.get('name')
            return c_id or slug or name

        def dfs(row):
            key = get_key(row)
            if not key:
                return
            if visited.get(key, 0) == 1:
                cycle_detected[0] = True
                cycle_path.append(row.get('name') or key)
                return
            if visited.get(key, 0) == 2:
                return

            visited[key] = 1 # Visiting
            cycle_path.append(row.get('name') or key)

            parent = get_parent_row(row)
            if parent:
                dfs(parent)

            if cycle_detected[0]:
                return

            visited[key] = 2 # Visited
            cycle_path.pop()
            sorted_rows.append(row)

        for row in rows:
            key = get_key(row)
            if key and visited.get(key, 0) == 0:
                dfs(row)
                if cycle_detected[0]:
                    path_str = " -> ".join(cycle_path + [cycle_path[0]])
                    return [], f"Phát hiện vòng lặp phân cấp danh mục: {path_str}"

        # DFS already appends parent first because it traverses from child to parent, 
        # so parents are completed and appended before their children. No need to reverse.
        return sorted_rows, None

    def parse_products(self, file_content: bytes, auth_header: str) -> Dict[str, Any]:
        """Parse Excel product data and return preview payload."""
        df = pd.read_excel(io.BytesIO(file_content))
        df = df.where(pd.notnull(df), None)
        
        excel_cols = df.columns.tolist()
        target_fields = self.get_schema_fields("product", auth_header)
        col_mapping = self.fuzzy_match_columns(excel_cols, target_fields)
        
        valid_rows = []
        errors = []
        
        for idx, row in df.iterrows():
            row_index = idx + 2
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            
            mapped_data = {}
            for excel_col, val in row_dict.items():
                target_field = col_mapping.get(excel_col)
                if target_field:
                    mapped_data[target_field] = val

            # Preserve explicit fields for resolution
            mapped_data["product_id"] = find_excel_val(row_dict, "product_id", "productId", "id")
            mapped_data["category_id"] = find_excel_val(row_dict, "category_id", "categoryId")
            mapped_data["category_slug"] = find_excel_val(row_dict, "category_slug", "categorySlug")
            mapped_data["category_name"] = find_excel_val(row_dict, "category_name", "categoryName")
            mapped_data["thumbnail_url"] = find_excel_val(row_dict, "thumbnail_url", "imageUrl", "image_url", "ảnh đại diện", "anh dai dien", "ảnh", "anh")
            mapped_data["image_urls"] = find_excel_val(row_dict, "image_urls", "imageUrls", "images")
            mapped_data["color"] = find_excel_val(row_dict, "color", "colour", "màu", "mau")
            mapped_data["size"] = find_excel_val(row_dict, "size", "kích thước", "kich thuoc")
            mapped_data["product_url"] = find_excel_val(row_dict, "product_url", "productUrl")
            mapped_data["price"] = find_excel_val(row_dict, "price", "basePrice", "base_price", "giá bán", "gia ban", "giá", "gia")
            mapped_data["original_price"] = find_excel_val(row_dict, "original_price", "originalPrice", "comparePrice", "compare_price", "giá gốc", "gia goc")
            mapped_data["stock"] = find_excel_val(row_dict, "stock", "quantity", "qty", "số lượng", "so luong")
            mapped_data["sku"] = find_excel_val(row_dict, "sku", "mã sản phẩm", "ma san pham", "mã", "ma")
            mapped_data["brand"] = find_excel_val(row_dict, "brand", "thương hiệu", "thuong hieu")
            
            # Fields requested to be pushed to Qdrant
            mapped_data["category_url"] = find_excel_val(row_dict, "category_url", "categoryUrl")
            mapped_data["category_image_url"] = find_excel_val(row_dict, "category_image_url", "categoryImageUrl")
            mapped_data["currency"] = find_excel_val(row_dict, "currency", "đơn vị tiền tệ", "don vi tien te")
            mapped_data["reviews_count"] = find_excel_val(row_dict, "reviews_count", "reviewsCount", "reviews", "review_count", "reviewCount")
            mapped_data["material"] = find_excel_val(row_dict, "material", "chất liệu", "chat lieu")
            mapped_data["source"] = find_excel_val(row_dict, "source", "nguồn", "nguon")
            mapped_data["searchable_text"] = find_excel_val(row_dict, "searchable_text", "searchableText")
            mapped_data["rating"] = find_excel_val(row_dict, "rating", "ratingAvg", "rating_avg", "sao", "đánh giá")
            raw_tags = find_excel_val(row_dict, "tags", "tag")
            if isinstance(raw_tags, str):
                import re
                mapped_data["tags"] = [t.strip() for t in re.split(r'[,;|]', raw_tags) if t.strip()]
            elif isinstance(raw_tags, list):
                mapped_data["tags"] = raw_tags
            else:
                mapped_data["tags"] = [raw_tags] if raw_tags is not None else []

            # Extract fields for validation
            p_name = mapped_data.get("name") or find_excel_val(row_dict, "product_name", "productName")
            mapped_data["product_name"] = p_name
            p_price = mapped_data.get("basePrice") or mapped_data.get("price")
            p_stock = mapped_data.get("stock")
            
            row_errors = []
            if not p_name or str(p_name).strip() == "":
                row_errors.append("Tên sản phẩm không được rỗng")
                
            if p_price is not None:
                try:
                    price_val = float(p_price)
                    if price_val < 0:
                        row_errors.append("Giá sản phẩm không được nhỏ hơn 0")
                except ValueError:
                    row_errors.append("Giá sản phẩm phải là số")
            
            if p_stock is not None:
                try:
                    stock_val = int(p_stock)
                    if stock_val < 0:
                        row_errors.append("Tồn kho không được nhỏ hơn 0")
                except ValueError:
                    row_errors.append("Tồn kho phải là số nguyên")
                    
            if row_errors:
                errors.append({
                    "row": row_index,
                    "reason": ", ".join(row_errors)
                })
            else:
                mapped_data["rowIndex"] = row_index
                valid_rows.append(mapped_data)
                
        return {
            "totalRows": len(df),
            "validRows": len(valid_rows),
            "errorRows": len(errors),
            "columnMappings": col_mapping,
            "errors": errors,
            "rows": valid_rows
        }

    def parse_categories(self, file_content: bytes, auth_header: str) -> Dict[str, Any]:
        """Parse Excel category data, check cycles, sort, and return preview payload."""
        df = pd.read_excel(io.BytesIO(file_content))
        df = df.where(pd.notnull(df), None)
        
        excel_cols = df.columns.tolist()
        target_fields = self.get_schema_fields("category", auth_header)
        col_mapping = self.fuzzy_match_columns(excel_cols, target_fields)
        
        rows = []
        for idx, row in df.iterrows():
            row_index = idx + 2
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            
            mapped_data = {}
            for excel_col, val in row_dict.items():
                target_field = col_mapping.get(excel_col)
                if target_field:
                    mapped_data[target_field] = val
            
            mapped_data["category_id"] = find_excel_val(row_dict, "category_id", "categoryId", "id")
            mapped_data["category_main"] = find_excel_val(row_dict, "category_main", "categoryMain", "categoryMainId", "danh mục chính", "danh muc chinh")
            mapped_data["parent_id"] = find_excel_val(row_dict, "parent_id", "parentId", "parent")
            mapped_data["parent_slug"] = find_excel_val(row_dict, "parent_slug", "parentSlug")
            mapped_data["parent_name"] = find_excel_val(row_dict, "parent_name", "parentName")
            mapped_data["category_image_url"] = find_excel_val(row_dict, "category_image_url", "categoryImageUrl", "imageUrl", "image_url", "ảnh danh mục", "anh danh muc", "ảnh", "anh")
            mapped_data["display_order"] = find_excel_val(row_dict, "display_order", "displayOrder", "sort_order", "sortOrder", "thứ tự hiển thị", "thu tu hien thi")
            mapped_data["slug"] = mapped_data.get("slug") or find_excel_val(row_dict, "slug")
            mapped_data["name"] = mapped_data.get("name") or find_excel_val(row_dict, "name", "category_name", "categoryName")
            mapped_data["description"] = mapped_data.get("description") or find_excel_val(row_dict, "description", "mô tả", "mo ta")
            mapped_data["rowIndex"] = row_index
            rows.append(mapped_data)
            
        sorted_rows, cycle_error = self.detect_cycle_and_sort_categories(rows)
        if cycle_error:
            return {
                "totalRows": len(df),
                "validRows": 0,
                "errorRows": len(df),
                "columnMappings": col_mapping,
                "errors": [{"row": 0, "reason": cycle_error}],
                "rows": []
            }
            
        valid_rows = []
        errors = []
        
        for mapped_data in sorted_rows:
            row_index = mapped_data["rowIndex"]
            c_name = mapped_data.get("name")
            
            row_errors = []
            if not c_name or str(c_name).strip() == "":
                row_errors.append("Tên danh mục không được rỗng")
                
            if row_errors:
                errors.append({
                    "row": row_index,
                    "reason": ", ".join(row_errors)
                })
            else:
                valid_rows.append(mapped_data)
                
        return {
            "totalRows": len(df),
            "validRows": len(valid_rows),
            "errorRows": len(errors),
            "columnMappings": col_mapping,
            "errors": errors,
            "rows": valid_rows
        }

excel_import_service = ExcelImportService()
