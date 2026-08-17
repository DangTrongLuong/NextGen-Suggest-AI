from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

class SearchOverviewData(BaseModel):
    total_searches: int
    unique_keywords: int
    average_response_time_ms: float
    zero_result_searches: int
    zero_result_rate: float  # Percentage (0 - 100)
    average_results_per_query: float
    ctr: float  # Click-Through Rate (0 - 100)
    search_to_click_rate: float  # Unique search click rate (0 - 100)
    searches_today: int
    searches_last_7_days: int
    searches_last_30_days: int

class SearchProductRank(BaseModel):
    product_id: str
    product_name: str
    ranking_position: int

class SearchClickItem(BaseModel):
    product_id: str
    product_name: str
    clicked_at: datetime

class SearchHistoryItem(BaseModel):
    id: int
    request_id: Optional[str]
    keyword: str
    response_time_ms: int
    total_results: int
    session_id: str
    searched_at: datetime
    products: List[SearchProductRank] = []
    clicks: List[SearchClickItem] = []

class TopKeywordItem(BaseModel):
    keyword: str
    search_count: int
    zero_result_count: int
    avg_latency_ms: float
    click_count: int
    ctr: float

class TopProductItem(BaseModel):
    product_id: str
    product_name: str
    appearance_count: int
    average_ranking: float
    last_appeared: datetime
    click_count: int

class SystemComponentStatus(BaseModel):
    status: str  # "healthy" | "warning" | "down"
    message: Optional[str] = None
    latency_ms: Optional[float] = None

class SystemStatusResponse(BaseModel):
    fastapi: SystemComponentStatus
    spring_boot: SystemComponentStatus
    redis: SystemComponentStatus
    mysql: SystemComponentStatus
    qdrant: SystemComponentStatus

class AlertItem(BaseModel):
    metric: str  # "latency" | "zero_results" | "errors" | "volume"
    severity: str  # "warning" | "critical"
    message: str
    value: str
    triggered_at: datetime

class ClickTrackingRequest(BaseModel):
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    product_id: str
    product_name: str

class CartTrackingRequest(BaseModel):
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    product_id: str
    product_name: str

class ConversionTrackingRequest(BaseModel):
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    order_id: str
    total_amount: float
