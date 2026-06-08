# Hướng Dẫn Chạy Hệ Thống Gợi Ý Sản Phẩm Thông Minh (FastAPI Backend)

Tài liệu này hướng dẫn chi tiết cách chạy hệ thống gợi ý sản phẩm kết hợp **LLM Vector Embeddings** và **Qdrant Vector Database**.

---

## 1. Yêu Cầu Đầu Vào (Input)

### 1.1 Dữ liệu sản phẩm thô
Hệ thống yêu cầu tệp dữ liệu sản phẩm đầu vào được đặt tại đường dẫn:
`data/raw/products_vi.xlsx` (Hoặc các định dạng hỗ trợ khác như `.csv`, `.jsonl` tùy cấu hình trong script clean data).
*Nếu chưa có thư mục `data/raw`, hãy tạo nó ở thư mục gốc của backend.*

Tệp Excel (`products_vi.xlsx`) cần chứa các cột thông tin tối thiểu sau:
- `product_id`: ID duy nhất của sản phẩm.
- `title`: Tên sản phẩm.
- `description`: Mô tả chi tiết sản phẩm.
- `category`: Danh mục sản phẩm.
- `tags`: Các nhãn tìm kiếm liên quan (ngăn cách bằng dấu phẩy).
- `price`: Giá sản phẩm (đơn vị USD hoặc đã quy đổi).
- `image_url`: Link ảnh sản phẩm.

### 1.2 Qdrant Vector Database
- Mặc định hệ thống kết nối đến server Qdrant online qua cấu hình trong [config.py](app/config.py):
  `QDRANT_URL = "https://qdrant.datn-nextgen-suggest.site"`
- Nếu bạn muốn chạy **Qdrant Local**, hãy cài đặt **Docker** và khởi chạy container Qdrant bằng lệnh:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
  ```
  Sau đó, bạn cần tạo file `.env` tại thư mục gốc và khai báo:
  ```env
  QDRANT_URL=http://localhost:6333
  QDRANT_API_KEY=
  ```

---

## 2. Các Bước Chạy & Thứ Tự Chạy

Tất cả các lệnh dưới đây cần được thực hiện từ thư mục gốc của backend: `e:/WEB_BAN_HANG/ecommerce-backend-fastapi`.

### Bước 1: Khởi tạo Môi trường ảo & Cài đặt thư viện
Mở terminal tại thư mục gốc backend và chạy:
```powershell
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường ảo (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# (Hoặc nếu dùng Command Prompt / CMD: .\venv\Scripts\activate.bat)
# (Đối với Linux/MacOS: source venv/bin/activate)

# 3. Cài đặt các thư viện cần thiết
pip install fastapi uvicorn pydantic pydantic-settings qdrant-client sentence-transformers torch transformers pandas numpy python-dotenv openpyxl
```

### Bước 2: Làm sạch và chuẩn hóa dữ liệu
Chạy script [01_clean_data.py](scripts/data_processing/01_clean_data.py) để lọc HTML, chuẩn hóa text tiếng Việt và quy đổi giá:
```powershell
python scripts/data_processing/01_clean_data.py
```
* **Kết quả**: Sinh ra file `data/processed/products_cleaned.jsonl` chứa 2000 sản phẩm đã chuẩn hóa.

### Bước 3: Định dạng tài liệu tìm kiếm
Chạy script [02_format_documents.py](scripts/data_processing/02_format_documents.py) để chuẩn bị văn bản cho mô hình AI E5:
```powershell
python scripts/data_processing/02_format_documents.py --model-type e5
```
* **Kết quả**: Sinh ra file `data/processed/products_with_documents.jsonl` (Thêm tiền tố `passage: ` trước văn bản).

### Bước 4: Khởi tạo Collection trên Qdrant DB
Chạy script [01_create_collection.py](scripts/indexing/01_create_collection.py) để tạo bảng chứa vector trên cơ sở dữ liệu Qdrant:
```powershell
python scripts/indexing/01_create_collection.py --recreate
```

### Bước 5: Tính toán Vector Embedding bằng mô hình AI
Chạy script [02_embed_products.py](scripts/indexing/02_embed_products.py) để mã hóa toàn bộ văn bản sản phẩm thành các vector số (768 chiều):
```powershell
python scripts/indexing/02_embed_products.py --batch-size 64
```
* **Kết quả**: Sinh ra tệp vector `data/processed/embeddings.npz`.

### Bước 6: Upload Vector và Metadata lên Qdrant DB
Chạy script [03_index_to_qdrant.py](scripts/indexing/03_index_to_qdrant.py) để đồng bộ hóa vector và payload sản phẩm lên cơ sở dữ liệu:
```powershell
python scripts/indexing/03_index_to_qdrant.py
```

### Bước 7: Khởi chạy API Server (FastAPI)
Khi dữ liệu đã sẵn sàng trong Qdrant, khởi chạy API server bằng lệnh:
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
* **API Swagger UI**: Truy cập [http://localhost:8000/docs](http://localhost:8000/docs) để test các API `/search`, `/recommend`,...
* **Health Check**: Truy cập [http://localhost:8000/health](http://localhost:8000/health) để kiểm tra kết nối Qdrant.

---

## ⚠️ Lưu Ý Quan Trọng Tránh Lỗi Khởi Động Server

Trong file [main.py](app/main.py) có gọi trực tiếp các thuộc tính:
- `settings.QDRANT_HOST`
- `settings.QDRANT_PORT`

Tuy nhiên, trong file cấu hình gốc [config.py](app/config.py) hiện tại chỉ định nghĩa `QDRANT_URL`. Điều này sẽ gây lỗi `AttributeError` khi khởi động server. 

**Cách khắc phục:**
Hãy mở file [config.py](app/config.py) và khai báo thêm 2 thuộc tính trên trong lớp `Settings` để khớp cấu hình:
```python
class Settings(BaseSettings):
    # Qdrant
    QDRANT_URL: str = "https://qdrant.datn-nextgen-suggest.site"
    QDRANT_HOST: str = "qdrant.datn-nextgen-suggest.site"  # Thêm dòng này
    QDRANT_PORT: int = 443                                 # Thêm dòng này
    QDRANT_API_KEY: str = "datn-nextgen-suggest@"
    QDRANT_COLLECTION: str = "products"
```
*(Nếu bạn chuyển sang dùng Docker Qdrant local, hãy cập nhật `QDRANT_HOST = "localhost"` và `QDRANT_PORT = 6333`)*
