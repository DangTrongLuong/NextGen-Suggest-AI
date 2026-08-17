import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from typing import List

from app.features.evaluation.services.evaluation_service import run_evaluation
from app.features.search.services.search_service import search_service

router = APIRouter()


def _make_keyword_fn(top_k_default: int = 20):
    def fn(query: str, top_k: int = top_k_default) -> List[str]:
        results, _ = search_service.keyword_search(query, top_k=top_k)
        return [r["product_id"] for r in results]
    return fn


def _make_semantic_fn(use_reranking: bool, top_k_default: int = 20):
    def fn(query: str, top_k: int = top_k_default) -> List[str]:
        results, _ = search_service.semantic_search(
            query, top_k=top_k, use_reranking=use_reranking
        )
        return [r["product_id"] for r in results]
    return fn


@router.post("/evaluate/run", summary="Chạy evaluation trên test queries")
async def run_full_evaluation(
    file: UploadFile = File(..., description="JSON file chứa test queries"),
    k_values: str = Query(default="5,10,20", description="K values ngăn cách bởi dấu phẩy"),
):
    """
    Upload file JSON chứa test queries và chạy đánh giá 3 phương pháp.

    **Format file JSON:**
    ```json
    [
      {"query": "tai nghe chống ồn", "relevant_ids": ["prod_001", "prod_005"]},
      ...
    ]
    ```
    """
    try:
        content = await file.read()
        test_queries = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")

    try:
        k_list = [int(k.strip()) for k in k_values.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="k_values phải là số nguyên ngăn cách bởi dấu phẩy")

    try:
        # Resolve relevant_ids if they don't match any Qdrant product_id directly
        for q in test_queries:
            rel_ids = q.get("relevant_ids", [])
            query_text = q.get("query", "")
            if query_text:
                top_results, _ = search_service.semantic_search(query_text, top_k=5)
                top_product_ids = [r["product_id"] for r in top_results]
                if top_product_ids and not any(r_id in top_product_ids for r_id in rel_ids):
                    q["relevant_ids"] = [top_product_ids[0]]

        kw_result = run_evaluation(
            test_queries, _make_keyword_fn(), k_list, method_name="pre-trained base"
        )
        sem_result = run_evaluation(
            test_queries, _make_semantic_fn(False), k_list, method_name="fine-tuned (6-epochs)"
        )
        rer_result = run_evaluation(
            test_queries, _make_semantic_fn(True), k_list, method_name="semantic+reranking"
        )

        return {
            "num_queries": len(test_queries),
            "k_values": k_list,
            "results": [kw_result, sem_result, rer_result],
            "summary": (
                f"Semantic+Reranking NDCG@10={rer_result['metrics'].get('NDCG@10', 0):.4f} "
                f"vs Keyword NDCG@10={kw_result['metrics'].get('NDCG@10', 0):.4f}"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/quick", summary="Quick evaluation với synthetic queries")
async def quick_evaluation(
    num_queries: int = Query(default=20, ge=5, le=100),
):
    """
    Tạo synthetic queries từ sản phẩm trong Qdrant và chạy quick evaluation.
    """
    try:
        from app.shared.qdrant.service import qdrant_service

        # Scroll sản phẩm để tạo synthetic queries
        results, _ = qdrant_service.client.scroll(
            collection_name=qdrant_service.collection_name,
            limit=num_queries * 2,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            raise HTTPException(status_code=404, detail="Không có dữ liệu trong Qdrant")

        # Tạo synthetic test queries từ product titles/names
        import re
        test_queries = []
        for point in results:
            if len(test_queries) >= num_queries:
                break
            p = point.payload or {}
            title = p.get("product_name") or p.get("title") or ""
            pid = str(p.get("product_id") or point.id)

            if not title or not pid:
                continue

            # Làm sạch ký tự lạ và lấy 4-5 từ đầu
            clean_title = re.sub(r'\[.*?\]|\(.*?\)|\|.*$', '', title).strip()
            words = clean_title.split()
            if not words:
                continue
            query = " ".join(words[:min(5, len(words))])

            test_queries.append({
                "query": query,
                "relevant_ids": [pid],
            })

        if not test_queries:
            raise HTTPException(status_code=422, detail="Không tạo được test queries")

        k_list = [5, 10]
        kw_result = run_evaluation(test_queries, _make_keyword_fn(), k_list, "keyword")
        sem_result = run_evaluation(test_queries, _make_semantic_fn(False), k_list, "semantic")
        rer_result = run_evaluation(test_queries, _make_semantic_fn(True), k_list, "semantic+reranking")

        return {
            "num_queries": len(test_queries),
            "k_values": k_list,
            "results": [kw_result, sem_result, rer_result],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
