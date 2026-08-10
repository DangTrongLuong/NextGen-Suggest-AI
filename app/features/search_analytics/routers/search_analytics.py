import logging
import io
import csv
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Query, Depends, Response, HTTPException, status
from fastapi.responses import StreamingResponse

from app.features.search_analytics.schemas.auth_schemas import StandardResponse
from app.features.search_analytics.schemas.analytics_schemas import (
    SearchOverviewData, SearchHistoryItem, TopKeywordItem, TopProductItem, 
    SystemStatusResponse, ClickTrackingRequest, SearchHistoryItem
)
from app.features.search_analytics.services.analytics_service import analytics_service
from app.features.search_analytics.repositories.analytics_repository import analytics_repository
from app.features.search_analytics.auth.auth_handler import get_current_analytics_user

logger = logging.getLogger("analytics_router")

router = APIRouter()

# Dependency helper to add SEO noindex headers
def set_noindex_header(response: Response):
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response

@router.get("/search/overview", response_model=StandardResponse, summary="Get dashboard overview KPIs")
async def get_overview(
    time_range: str = Query("24h", description="Time period filter: 24h, 7d, 30d"),
    current_user: dict = Depends(get_current_analytics_user),
    response: Response = Depends(set_noindex_header)
):
    """Retrieve search performance KPIs and calculate current system alerts."""
    try:
        overview_data = analytics_service.get_overview(time_range)
        alerts = analytics_service.check_alerts(overview_data)
        
        return StandardResponse(
            success=True,
            message="Overview stats retrieved.",
            data={
                "kpis": overview_data.get("kpis", {}),
                "timeline": overview_data.get("timeline", []),
                "alerts": alerts
            }
        )
    except Exception as e:
        logger.error(f"Error in overview router: {e}")
        raise HTTPException(status_code=500, detail="Lỗi tải dữ liệu tổng quan.")

@router.get("/search/history", response_model=StandardResponse, summary="Paginated search history logs")
async def get_history(
    q: Optional[str] = Query(None, description="Search keyword filter"),
    start: Optional[str] = Query(None, description="Start date ISO string"),
    end: Optional[str] = Query(None, description="End date ISO string"),
    zero_results: bool = Query(False, description="Filter zero results searches only"),
    page: int = Query(0, ge=0, description="Page index"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    sort_by: str = Query("searched_at", description="Sort by field"),
    sort_desc: bool = Query(True, description="Sort descending"),
    current_user: dict = Depends(get_current_analytics_user),
    response: Response = Depends(set_noindex_header)
):
    """Retrieve search query history with composite criteria, sorted and paginated on server-side."""
    try:
        logs, total = analytics_service.get_history(
            q=q,
            start_date_str=start,
            end_date_str=end,
            zero_results_only=zero_results,
            page=page,
            size=size,
            sort_by=sort_by,
            sort_desc=sort_desc
        )
        return StandardResponse(
            success=True,
            data={
                "content": logs,
                "total_elements": total,
                "page": page,
                "size": size,
                "total_pages": (total + size - 1) // size if total > 0 else 0
            }
        )
    except Exception as e:
        logger.error(f"Error in history router: {e}")
        raise HTTPException(status_code=500, detail="Lỗi tải danh sách lịch sử.")

@router.get("/search/detail/{log_id}", response_model=StandardResponse, summary="Get details of a search log entry")
async def get_detail(
    log_id: int,
    current_user: dict = Depends(get_current_analytics_user),
    response: Response = Depends(set_noindex_header)
):
    """Retrieve details of a single query log including products and clicks."""
    log = analytics_service.get_detail(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi lịch sử tìm kiếm.")
    return StandardResponse(success=True, data=log)

@router.get("/search/top-keywords", response_model=StandardResponse, summary="Get top keywords list")
async def get_top_keywords(
    time_range: str = Query("24h", description="Time period filter"),
    limit: int = Query(15, ge=1, le=100),
    current_user: dict = Depends(get_current_analytics_user),
    response: Response = Depends(set_noindex_header)
):
    """Get aggregated top keywords counts, latency, and click-through rates."""
    results = analytics_service.get_top_keywords(time_range, limit)
    return StandardResponse(success=True, data=results)

@router.get("/search/top-products", response_model=StandardResponse, summary="Get top products list")
async def get_top_products(
    time_range: str = Query("24h", description="Time period filter"),
    limit: int = Query(15, ge=1, le=100),
    current_user: dict = Depends(get_current_analytics_user),
    response: Response = Depends(set_noindex_header)
):
    """Get aggregated top product appearances, average rankings, and clicks."""
    results = analytics_service.get_top_products(time_range, limit)
    return StandardResponse(success=True, data=results)

@router.post("/search/click", response_model=StandardResponse, summary="Track a product click event")
async def track_click(
    body: ClickTrackingRequest,
    response: Response = Depends(set_noindex_header)
    # Public endpoint since it is called by storefront users clicking search results
):
    """API called from the store client to record user clicks on search results."""
    success = analytics_service.log_click(body.request_id, body.product_id, body.product_name)
    if success:
        return StandardResponse(success=True, message="Đã ghi nhận click.")
    raise HTTPException(status_code=400, detail="Không thể ghi nhận click. Vui lòng kiểm tra tham số.")

@router.get("/system-status", response_model=StandardResponse, summary="Query internal services status")
async def get_system_status(
    current_user: dict = Depends(get_current_analytics_user),
    response: Response = Depends(set_noindex_header)
):
    """Probes status of FastAPI, Spring Boot, Redis, MySQL, Qdrant, and RabbitMQ."""
    status_data = await analytics_service.get_system_health()
    return StandardResponse(success=True, data=status_data)

@router.get("/audit-logs", response_model=StandardResponse, summary="Paginated security audit logs")
async def get_audit_logs(
    page: int = Query(0, ge=0, description="Page index"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    current_user: dict = Depends(get_current_analytics_user),
    response: Response = Depends(set_noindex_header)
):
    """Retrieve paginated security audit logs for System Health view."""
    try:
        logs, total = analytics_repository.get_audit_logs(page=page, size=size)
        return StandardResponse(
            success=True,
            data={
                "content": logs,
                "total_elements": total,
                "page": page,
                "size": size,
                "total_pages": (total + size - 1) // size if total > 0 else 0
            }
        )
    except Exception as e:
        logger.error(f"Error in audit-logs router: {e}")
        raise HTTPException(status_code=500, detail="Lỗi tải nhật ký hoạt động.")

@router.get("/health", response_model=StandardResponse, summary="Simple health ping")
async def health(
    response: Response = Depends(set_noindex_header)
):
    """Public simple health check endpoint for monitoring status."""
    return StandardResponse(success=True, message="Analytics portal backend is online.")

# ── CSV Data Streaming Exporters ──

@router.get("/export/history", summary="Download search history CSV")
async def export_history(
    q: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_analytics_user)
):
    """Streams the entire search query history logs as a CSV file to prevent memory exhaustion."""
    iterator = analytics_service.generate_history_csv_iterator(q, start, end)
    filename = f"search_history_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    return StreamingResponse(
        iterator,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Robots-Tag": "noindex, nofollow"
        }
    )

@router.get("/export/top-keywords", summary="Download top keywords CSV")
async def export_top_keywords_csv(
    time_range: str = Query("24h"),
    current_user: dict = Depends(get_current_analytics_user)
):
    """Streams top keywords list as a CSV file."""
    def get_top_keywords_csv_stream(time_range: str):
        keywords = analytics_service.get_top_keywords(time_range, limit=5000)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Keyword", "Search Count", "Zero Result Count", "Avg Latency (ms)", "Click Count", "CTR (%)"])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)
        
        for kw in keywords:
            writer.writerow([
                kw["keyword"],
                kw["search_count"],
                kw["zero_result_count"],
                kw["avg_latency_ms"],
                kw["click_count"],
                kw["ctr"]
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    filename = f"top_keywords_{time_range}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    return StreamingResponse(
        get_top_keywords_csv_stream(time_range),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Robots-Tag": "noindex, nofollow"
        }
    )

@router.get("/export/top-products", summary="Download top products CSV")
async def export_top_products_csv(
    time_range: str = Query("24h"),
    current_user: dict = Depends(get_current_analytics_user)
):
    """Streams top products list as a CSV file."""
    def get_top_products_csv_stream(time_range: str):
        products = analytics_service.get_top_products(time_range, limit=5000)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Product ID", "Product Name", "Appearance Count", "Average Ranking", "Last Appeared", "Click Count"])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)
        
        for p in products:
            writer.writerow([
                p["product_id"],
                p["product_name"],
                p["appearance_count"],
                p["average_ranking"],
                p["last_appeared"].isoformat() if isinstance(p["last_appeared"], datetime) else p["last_appeared"],
                p["click_count"]
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    filename = f"top_products_{time_range}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    return StreamingResponse(
        get_top_products_csv_stream(time_range),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Robots-Tag": "noindex, nofollow"
        }
    )
