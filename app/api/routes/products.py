"""API routes for products CRUD."""
from fastapi import APIRouter, HTTPException, Path, Query
from app.services.qdrant_service import qdrant_service

router = APIRouter()


@router.get("/products/{product_id}", summary="Lấy thông tin sản phẩm")
async def get_product(product_id: str = Path(...)):
    product = qdrant_service.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    return product


@router.get("/products", summary="Danh sách sản phẩm (scroll)")
async def list_products(limit: int = Query(default=20, ge=1, le=100)):
    """Lấy danh sách sản phẩm từ Qdrant (scroll)."""
    try:
        results, _ = qdrant_service.client.scroll(
            collection_name=qdrant_service.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        products = [
            {
                "product_id": r.payload.get("product_id", ""),
                "title": r.payload.get("title", ""),
                "category": r.payload.get("category", ""),
                "price": r.payload.get("price", 0),
                "image_url": r.payload.get("image_url", ""),
            }
            for r in results
        ]
        return {"products": products, "total": len(products)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Thống kê collection")
async def collection_stats():
    return qdrant_service.get_collection_info()
