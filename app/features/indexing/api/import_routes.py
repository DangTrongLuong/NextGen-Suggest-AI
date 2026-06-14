from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from app.features.indexing.services.excel_import_service import excel_import_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/import/parse/products", summary="Phân tích cấu trúc file Excel Products")
async def parse_products_excel(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    try:
        content = await file.read()
        return excel_import_service.parse_products(content, authorization)
    except Exception as e:
        logger.error(f"Error parsing products Excel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/import/parse/categories", summary="Phân tích cấu trúc file Excel Categories")
async def parse_categories_excel(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    try:
        content = await file.read()
        return excel_import_service.parse_categories(content, authorization)
    except Exception as e:
        logger.error(f"Error parsing categories Excel: {e}")
        raise HTTPException(status_code=500, detail=str(e))
