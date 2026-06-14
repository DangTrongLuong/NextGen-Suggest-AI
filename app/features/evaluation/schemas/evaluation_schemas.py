from pydantic import BaseModel, Field
from typing import List, Optional


class EvalQuery(BaseModel):
    query: str
    relevant_ids: List[str]
    source_product_id: Optional[str] = None


class EvaluationRequest(BaseModel):
    test_queries: List[EvalQuery]
    k_values: List[int] = Field(default=[5, 10, 20])
    methods: List[str] = Field(
        default=["keyword", "semantic", "semantic+reranking"],
        description="Các phương pháp cần đánh giá"
    )


class MetricsResult(BaseModel):
    method: str
    precision_at_k: dict  # {"5": 0.45, "10": 0.38}
    recall_at_k: dict
    ndcg_at_k: dict
    mrr: float
    avg_latency_ms: float


class EvaluationResponse(BaseModel):
    results: List[MetricsResult]
    num_queries: int
    k_values: List[int]
    summary: str
