from fastapi import APIRouter, HTTPException
from app.features.indexing.schemas.indexing_schemas import IndexSyncRequest
from app.features.indexing.services.indexing_service import indexing_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/indexing/sync", summary="Đồng bộ vector Qdrant cho danh sách Product UUIDs")
async def sync_products_index(request: IndexSyncRequest):
    if not request.product_uuids:
        return {"message": "Danh sách UUID trống"}
    try:
        return indexing_service.sync_uuids_to_qdrant(request.product_uuids)
    except Exception as e:
        logger.error(f"Error syncing products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/indexing/delete", summary="Xóa point vector Qdrant cho danh sách Product UUIDs")
async def delete_products_index(request: IndexSyncRequest):
    if not request.product_uuids:
        return {"message": "Danh sách UUID trống"}
    try:
        return indexing_service.delete_uuids_from_qdrant(request.product_uuids)
    except Exception as e:
        logger.error(f"Error deleting products: {e}")
        raise HTTPException(status_code=500, detail=str(e))
