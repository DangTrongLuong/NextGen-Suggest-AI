"""
Recommendation Service - Similar products & history-based recommendations.
"""
import time
import numpy as np
from typing import List, Dict, Optional, Tuple

from app.shared.qdrant.service import qdrant_service
from app.shared.redis.service import redis_service
import logging

logger = logging.getLogger(__name__)

import pymysql
from app.config.settings import settings

def get_mysql_connection():
    return pymysql.connect(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor
    )

def get_products_by_uuids(uuids: List[str]) -> List[Dict]:
    if not uuids:
        return []
    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            format_strings = ','.join(['%s'] * len(uuids))
            query = f"""
            SELECT p.uuid as product_id, p.name as title, p.description, 
                   c.name as category, p.image_url, p.brand,
                   COALESCE((SELECT MIN(v.price) FROM product_variants v WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 0) as price
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            WHERE p.uuid IN ({format_strings});
            """
            cursor.execute(query, tuple(uuids))
            rows = cursor.fetchall()
            for row in rows:
                row['tags'] = []
                row['price'] = float(row['price'])
            return rows
    except Exception as e:
        logger.error(f"Error fetching products by uuids from MySQL: {e}")
        return []
    finally:
        conn.close()

def get_user_item_interactions():
    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            query = """
            SELECT user_id, uuid as product_id, SUM(weight) as score
            FROM (
                SELECT ube.user_id, p.uuid,
                       CASE 
                           WHEN ube.event_type = 'purchase' THEN 5.0
                           WHEN ube.event_type = 'add_to_cart' THEN 3.0
                           WHEN ube.event_type = 'add_to_wishlist' THEN 2.0
                           WHEN ube.event_type = 'product_view' THEN 1.0
                           ELSE 0.5 
                       END as weight
                FROM user_behavior_events ube
                JOIN products p ON ube.product_id = p.id
                WHERE ube.user_id IS NOT NULL AND ube.event_type IN ('purchase', 'add_to_cart', 'add_to_wishlist', 'product_view')

                UNION ALL

                SELECT o.user_id, p.uuid, 5.0 as weight
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                JOIN products p ON oi.product_id = p.id
                WHERE o.user_id IS NOT NULL
            ) as combined
            GROUP BY user_id, product_id;
            """
            cursor.execute(query)
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error fetching user-item interactions from MySQL: {e}")
        return []
    finally:
        conn.close()

def get_collaborative_candidates(user_id, top_k=20):
    try:
        interactions = get_user_item_interactions()
        if not interactions:
            return {}

        users = list(set(row['user_id'] for row in interactions))
        items = list(set(row['product_id'] for row in interactions))
        
        user_to_idx = {uid: i for i, uid in enumerate(users)}
        item_to_idx = {pid: i for i, pid in enumerate(items)}
        idx_to_item = {i: pid for pid, i in item_to_idx.items()}

        target_user_id = None
        try:
            target_user_id = int(user_id)
        except (ValueError, TypeError):
            target_user_id = user_id

        if target_user_id not in user_to_idx:
            return {}

        u_idx = user_to_idx[target_user_id]
        num_users = len(users)
        num_items = len(items)
        matrix = np.zeros((num_users, num_items))

        for row in interactions:
            uid = row['user_id']
            pid = row['product_id']
            matrix[user_to_idx[uid], item_to_idx[pid]] = row['score']

        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0] = 1e-9
        normalized_matrix = matrix / norms

        item_similarity = np.dot(normalized_matrix.T, normalized_matrix)
        user_vector = matrix[u_idx]
        scores = np.dot(item_similarity, user_vector)

        max_score = np.max(scores) if np.max(scores) > 0 else 1.0
        scores = scores / max_score

        candidates = {}
        for idx in range(num_items):
            item_id = idx_to_item[idx]
            if user_vector[idx] > 0:
                continue
            score = float(scores[idx])
            if score > 0.01:
                candidates[item_id] = score

        return candidates
    except Exception as e:
        logger.error(f"Error computing collaborative filtering candidates: {e}")
        return {}


class RecommendationService:

    def get_similar_products(
        self,
        product_id: str,
        top_k: int = 10,
        same_category: bool = False,
    ) -> Tuple[List[Dict], str, float]:
        """
        Tìm sản phẩm tương tự dựa trên vector của sản phẩm gốc.

        Returns:
            (results, source_title, latency_ms)
        """
        t0 = time.perf_counter()

        # Lấy thông tin sản phẩm gốc
        source = qdrant_service.get_product_by_id(product_id)
        if source is None:
            return [], "Not found", 0.0

        # Lấy vector của sản phẩm
        product_vector = qdrant_service.get_product_vector(product_id)
        if product_vector is None:
            return [], source.get("title", ""), 0.0

        category_filter = source.get("category") if same_category else None

        results = qdrant_service.search(
            query_vector=product_vector,
            top_k=top_k + 1,           # +1 vì sẽ bao gồm chính nó
            category_filter=category_filter,
            exclude_ids=[product_id],  # Loại bỏ chính sản phẩm đó
        )

        for i, r in enumerate(results[:top_k], 1):
            r["rank"] = i
            r["reason"] = "Sản phẩm tương tự bạn vừa xem"

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return results[:top_k], source.get("title", ""), latency_ms

    def get_recommendations_from_history(
        self,
        viewed_product_ids: List[str],
        top_k: int = 10,
        exclude_viewed: bool = True,
    ) -> Tuple[List[Dict], float]:
        """
        Gợi ý dựa trên lịch sử xem sản phẩm (Average Embedding approach).

        Tính average vector của các sản phẩm đã xem → tìm sản phẩm gần nhất.

        Returns:
            (results, latency_ms)
        """
        t0 = time.perf_counter()

        # Thu thập vectors
        vectors = []
        valid_ids = []
        for pid in viewed_product_ids:
            vec = qdrant_service.get_product_vector(pid)
            if vec is not None:
                vectors.append(vec)
                valid_ids.append(pid)

        if not vectors:
            logger.warning("No valid product vectors found for history")
            return [], 0.0

        # Average pooling → user preference vector
        avg_vector = np.mean(vectors, axis=0)
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm

        exclude = viewed_product_ids if exclude_viewed else None

        results = qdrant_service.search(
            query_vector=avg_vector.tolist(),
            top_k=top_k + len(viewed_product_ids),
            exclude_ids=exclude,
        )

        results = results[:top_k]
        for i, r in enumerate(results, 1):
            r["rank"] = i
            r["reason"] = "Phù hợp với phong cách xem gần đây của bạn"

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"history_rec | viewed={len(valid_ids)} products "
            f"| results={len(results)} | {latency_ms}ms"
        )
        return results, latency_ms

    def get_hybrid_recommendations(
        self,
        viewed_product_ids: List[str] = None,
        wishlist_product_ids: List[str] = None,
        purchased_product_ids: List[str] = None,
        popular_product_ids: List[str] = None,
        location_product_ids: List[str] = None,
        user_age: Optional[int] = None,
        user_gender: Optional[str] = None,
        user_occupation: Optional[str] = None,
        user_income: Optional[float] = None,
        user_marital_status: Optional[str] = None,
        user_location: Optional[str] = None,
        top_k: int = 10,
        exclude_viewed: bool = True,
        user_id: Optional[str] = None
    ) -> Tuple[List[Dict], float]:
        t0 = time.perf_counter()

        # Sanitize user_id
        if user_id in ("null", "None", "", "undefined"):
            user_id = None

        # ── Đọc từ Redis Cache ────────────────────────────────────────────────
        if user_id:
            cached_data = redis_service.get_cached_recommendations(user_id)
            if cached_data is not None:
                logger.info(f"Hybrid Rec: Cache HIT for user_id={user_id}")
                return cached_data, round((time.perf_counter() - t0) * 1000, 2)

        import uuid
        def get_point_uuid(product_id: str) -> str:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(product_id)))

        viewed = viewed_product_ids or []
        wishlist = wishlist_product_ids or []
        purchased = purchased_product_ids or []
        popular = popular_product_ids or []
        loc_popular = location_product_ids or []

        # Gather all unique product IDs we need vectors and payloads for
        all_user_pids = list(set(viewed + wishlist + purchased))
        pid_to_vector = {}
        pid_to_product = {}

        if all_user_pids:
            try:
                user_uuids = [get_point_uuid(pid) for pid in all_user_pids]
                retrieved_points = qdrant_service.client.retrieve(
                    collection_name=qdrant_service.collection_name,
                    ids=user_uuids,
                    with_payload=True,
                    with_vectors=True
                )
                for pt in retrieved_points:
                    p = pt.payload or {}
                    pid = p.get("product_id")
                    if not pid:
                        # Fallback mapping via uuid mapping
                        uuid_str = str(pt.id)
                        for original_pid in all_user_pids:
                            if get_point_uuid(original_pid) == uuid_str:
                                pid = original_pid
                                break
                    if pid:
                        if pt.vector:
                            pid_to_vector[pid] = pt.vector
                        pid_to_product[pid] = {
                            "product_id": pid,
                            "title": p.get("title", ""),
                            "description": p.get("description", ""),
                            "category": p.get("category", ""),
                            "tags": p.get("tags", []),
                            "price": p.get("price", 0),
                            "image_url": p.get("image_url", ""),
                        }
            except Exception as e:
                logger.error(f"Error batch retrieving user profiles from Qdrant: {e}", exc_info=True)

        # 1. Collect vectors
        view_vectors = [pid_to_vector[pid] for pid in viewed if pid in pid_to_vector]
        wishlist_vectors = [pid_to_vector[pid] for pid in wishlist if pid in pid_to_vector]
        purchase_vectors = [pid_to_vector[pid] for pid in purchased if pid in pid_to_vector]

        # 2. Calculate profile vector of the user
        profile_components = []
        if view_vectors:
            profile_components.append(0.4 * np.mean(view_vectors, axis=0))
        if wishlist_vectors:
            profile_components.append(0.3 * np.mean(wishlist_vectors, axis=0))
        if purchase_vectors:
            profile_components.append(0.3 * np.mean(purchase_vectors, axis=0))

        if not profile_components:
            # Fallback: no history, return popular or scroll products
            logger.info("Hybrid Rec: No user history found. Falling back to popular/scroll.")
            fallback_ids = popular if popular else []
            results = []
            
            # Batch retrieve popular products to avoid loop retrieve
            if fallback_ids:
                try:
                    fallback_uuids = [get_point_uuid(pid) for pid in fallback_ids[:top_k]]
                    retrieved_fallbacks = qdrant_service.client.retrieve(
                        collection_name=qdrant_service.collection_name,
                        ids=fallback_uuids,
                        with_payload=True,
                        with_vectors=False
                    )
                    # Map back to preserve order of fallback_ids
                    fb_map = {}
                    for pt in retrieved_fallbacks:
                        p = pt.payload or {}
                        pid = p.get("product_id")
                        if not pid:
                            uuid_str = str(pt.id)
                            for original_pid in fallback_ids[:top_k]:
                                if get_point_uuid(original_pid) == uuid_str:
                                    pid = original_pid
                                    break
                        if pid:
                            fb_map[pid] = {
                                "product_id": pid,
                                "title": p.get("title", ""),
                                "description": p.get("description", ""),
                                "category": p.get("category", ""),
                                "tags": p.get("tags", []),
                                "price": p.get("price", 0),
                                "image_url": p.get("image_url", ""),
                            }
                    
                    for pid in fallback_ids[:top_k]:
                        if pid in fb_map:
                            p = fb_map[pid]
                            p["score"] = 1.0
                            p["reason"] = "Sản phẩm xu hướng được mua nhiều nhất"
                            results.append(p)
                except Exception as e:
                    logger.error(f"Error batch retrieving popular fallback products: {e}", exc_info=True)

            if not results:
                scroll_results, _ = qdrant_service.client.scroll(
                    collection_name=qdrant_service.collection_name,
                    limit=top_k,
                    with_payload=True
                )
                for r in scroll_results:
                    p = r.payload
                    p["score"] = 0.5
                    p["reason"] = "Gợi ý dành cho bạn hôm nay"
                    results.append({
                        "product_id": p.get("product_id"),
                        "title": p.get("title"),
                        "description": p.get("description"),
                        "category": p.get("category"),
                        "tags": p.get("tags"),
                        "price": p.get("price"),
                        "image_url": p.get("image_url"),
                        "score": 0.5,
                        "reason": p.get("reason", "Gợi ý phổ biến")
                    })
            
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            # Lưu cache fallback nếu có user_id
            if user_id and results:
                redis_service.set_cached_recommendations(user_id, results[:top_k])
            return results[:top_k], latency_ms

        profile_vector = np.sum(profile_components, axis=0)
        norm = np.linalg.norm(profile_vector)
        if norm > 0:
            profile_vector = profile_vector / norm

        # 3. Get candidates from Qdrant
        exclude = viewed + purchased if exclude_viewed else None
        qdrant_candidates = qdrant_service.search(
            query_vector=profile_vector.tolist(),
            top_k=top_k * 4,
            exclude_ids=exclude,
            with_vectors=True
        )

        # 3.b Get collaborative filtering candidates from MySQL based on real history
        cf_candidates_dict = {}
        if user_id:
            cf_candidates_dict = get_collaborative_candidates(user_id, top_k=20)

        # Merge candidate lists
        candidates_map = {cand["product_id"]: cand for cand in qdrant_candidates}
        
        # Hydrate CF, popular, and location-based candidates not in Qdrant candidates
        extra_ids_to_hydrate = []
        for pid in list(cf_candidates_dict.keys()) + popular + loc_popular:
            if pid and pid not in candidates_map and pid not in extra_ids_to_hydrate:
                if exclude_viewed and exclude and pid in exclude:
                    continue
                extra_ids_to_hydrate.append(pid)

        if extra_ids_to_hydrate:
            extra_products = get_products_by_uuids(extra_ids_to_hydrate)
            for p in extra_products:
                pid = p["product_id"]
                candidates_map[pid] = {
                    "product_id": pid,
                    "title": p["title"],
                    "description": p["description"],
                    "category": p["category"],
                    "tags": p["tags"],
                    "price": p["price"],
                    "image_url": p["image_url"],
                    "score": 0.5, # Default content score
                    "vector": None
                }

        candidates = list(candidates_map.values())

        # 4. Calculate Hybrid Score for each candidate
        scored_candidates = []
        for cand in candidates:
            pid = cand.get("product_id")

            # A. Content score (0 to 1)
            content_score = cand.get("score", 0.5)

            # B. Popularity score
            popularity_score = 0.1
            if pid in popular:
                try:
                    idx = popular.index(pid)
                    popularity_score = max(0.2, 1.0 - (idx * 0.05))
                except ValueError:
                    popularity_score = 0.5

            # C. Behavior score: Max similarity to wishlist/purchases
            behavior_score = 0.1
            cand_vector = cand.get("vector")
            if cand_vector is not None:
                cand_vec_np = np.array(cand_vector)
                similarities = []
                for ref_vec in wishlist_vectors + purchase_vectors:
                    ref_vec_np = np.array(ref_vec)
                    cos_sim = np.dot(cand_vec_np, ref_vec_np) / (np.linalg.norm(cand_vec_np) * np.linalg.norm(ref_vec_np) + 1e-9)
                    similarities.append(cos_sim)
                if similarities:
                    behavior_score = max(0.1, float(np.max(similarities)))

            # D. Collaborative score (Real Collaborative Filtering from MySQL + Tag fallback)
            collaborative_score = 0.1
            if pid in cf_candidates_dict:
                collaborative_score = cf_candidates_dict[pid]
            else:
                categories = []
                tags = set()
                for ref_id in viewed + wishlist + purchased:
                    ref_p = pid_to_product.get(ref_id)
                    if ref_p:
                        if ref_p.get("category"):
                            categories.append(ref_p.get("category"))
                        if ref_p.get("tags"):
                            tags.update(ref_p.get("tags"))
                
                if cand.get("category") in categories:
                    collaborative_score += 0.4
                cand_tags = set(cand.get("tags") or [])
                shared_tags = cand_tags.intersection(tags)
                if shared_tags:
                    collaborative_score += min(0.4, len(shared_tags) * 0.1)

            # E. Demographic Score (Nhân khẩu học)
            demographic_score = self.calculate_demographic_score(
                cand=cand,
                age=user_age,
                gender=user_gender,
                occupation=user_occupation,
                income=user_income,
                marital_status=user_marital_status
            )

            # F. Location Score (Vị trí địa lý)
            location_score = 0.1
            if pid in loc_popular:
                try:
                    idx = loc_popular.index(pid)
                    location_score = max(0.3, 1.0 - (idx * 0.05))
                except ValueError:
                    location_score = 0.6

            # Trọng số thành phần của Final Score (tổng trọng số = 1.0)
            # w_content = 0.20, w_behavior = 0.20, w_popularity = 0.20, w_collaborative = 0.20
            # w_demographic = 0.10, w_location = 0.10
            w_content = 0.20 * content_score
            w_behavior = 0.20 * behavior_score
            w_popularity = 0.20 * popularity_score
            w_collaborative = 0.20 * collaborative_score
            w_demographic = 0.10 * demographic_score
            w_location = 0.10 * location_score

            hybrid_score = w_content + w_behavior + w_popularity + w_collaborative + w_demographic + w_location
            
            # Sinh lý do đề xuất (Rule-based Explanation) dựa trên thành phần điểm số cao nhất
            scores_map = {
                "location": w_location,
                "demographic": w_demographic,
                "behavior": w_behavior,
                "collaborative": w_collaborative,
                "content": w_content,
                "popularity": w_popularity
            }
            max_component = max(scores_map, key=scores_map.get)
            
            reason = "Gợi ý phù hợp với phong cách của bạn"
            if max_component == "location" and w_location > 0.03:
                reason = f"Phổ biến tại khu vực {user_location or 'của bạn'}"
            elif max_component == "demographic" and w_demographic > 0.05:
                # Khớp chi tiết hơn dựa trên thông tin có sẵn
                if user_gender and user_age:
                    reason = f"Phù hợp với giới tính {user_gender.lower()} & độ tuổi của bạn"
                elif user_gender:
                    reason = f"Phong cách phù hợp cho {user_gender.lower()}"
                else:
                    reason = "Phù hợp với hồ sơ cá nhân của bạn"
            elif max_component == "behavior" and w_behavior > 0.1:
                reason = "Phù hợp với lịch sử xem sản phẩm của bạn"
            elif max_component == "collaborative" and w_collaborative > 0.1:
                reason = "Khách hàng có sở thích giống bạn cũng mua"
            elif max_component == "content" and w_content > 0.1:
                reason = "Tương tự các sản phẩm bạn đã xem gần đây"
            elif max_component == "popularity" and w_popularity > 0.1:
                reason = "Sản phẩm xu hướng đang được quan tâm nhiều"

            cand_copy = cand.copy()
            cand_copy["score"] = round(hybrid_score, 4)
            cand_copy["reason"] = reason
            scored_candidates.append(cand_copy)

        # 5. Sort candidates
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        results = scored_candidates[:top_k]

        for i, r in enumerate(results, 1):
            r["rank"] = i

        # Lưu cache kết quả gợi ý
        if user_id and results:
            redis_service.set_cached_recommendations(user_id, results)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"[FastAPI Debug] received_user_id: {user_id}, "
            f"input_behavior: (viewed={viewed}, wishlist={wishlist}, purchased={purchased}), "
            f"output_products: {[r.get('product_id') for r in results]}"
        )
        return results, latency_ms

    def calculate_demographic_score(
        self,
        cand: Dict,
        age: Optional[int],
        gender: Optional[str],
        occupation: Optional[str],
        income: Optional[float],
        marital_status: Optional[str]
    ) -> float:
        """
        [Rule-based Demographic Matching Helper]
        Tính độ tương đồng giữa sản phẩm và thông tin nhân khẩu học của người dùng.
        """
        score = 0.0
        matches = 0
        total_checks = 0
        
        tags = cand.get("tags") or []
        title_desc_tags = f"{cand.get('title', '')} {cand.get('description', '')} {' '.join(tags)}".lower()
        
        # 1. Gender Match
        if gender:
            total_checks += 1
            g = gender.lower()
            if "nam" in g or "male" in g or "men" in g:
                if any(k in title_desc_tags for k in ["nam", "men", "man", "gentleman", "unisex"]):
                    score += 1.0
                    matches += 1
                elif any(k in title_desc_tags for k in ["nữ", "nu", "women", "woman", "lady"]):
                    score += 0.0
                else:
                    score += 0.5 # Neutral
            elif "nữ" in g or "nu" in g or "female" in g or "women" in g:
                if any(k in title_desc_tags for k in ["nữ", "nu", "women", "woman", "lady", "unisex"]):
                    score += 1.0
                    matches += 1
                elif any(k in title_desc_tags for k in ["nam", "men", "man", "gentleman"]):
                    score += 0.0
                else:
                    score += 0.5 # Neutral
            else:
                score += 0.7
                matches += 1
                
        # 2. Age Match
        if age:
            total_checks += 1
            if age < 25:
                # Trẻ trung, năng động, thông minh, phòng đơn, giá rẻ, gaming
                if any(k in title_desc_tags for k in ["hiện đại", "gaming", "học tập", "tối giản", "smart", "trẻ trung", "giá rẻ", "tiết kiệm"]):
                    score += 1.0
                    matches += 1
                else:
                    score += 0.4
            elif age >= 25 and age < 50:
                # Tiện nghi, công sở, gia đình, nhà bếp, phòng khách, cao cấp, bền bỉ
                if any(k in title_desc_tags for k in ["văn phòng", "ergonomic", "công sở", "luxury", "cao cấp", "gia đình", "bếp", "ấm cúng", "phòng khách"]):
                    score += 1.0
                    matches += 1
                else:
                    score += 0.5
            else:
                # Cổ điển, an toàn, massage, thư giãn, gỗ tự nhiên, chắc chắn, tinh tế
                if any(k in title_desc_tags for k in ["cổ điển", "classic", "gỗ tự nhiên", "thư giãn", "massage", "chắc chắn", "bền", "an toàn"]):
                    score += 1.0
                    matches += 1
                else:
                    score += 0.3
                    
        # 3. Occupation Match
        if occupation:
            total_checks += 1
            occ = occupation.lower()
            # Dân văn phòng, IT
            if any(k in occ for k in ["văn phòng", "kỹ sư", "lập trình", "office", "developer", "engineer", "designer", "architect", "thiết kế"]):
                if any(k in title_desc_tags for k in ["văn phòng", "office", "ergonomic", "làm việc", "desk", "chair", "xoay", "thông minh"]):
                    score += 1.0
                    matches += 1
            # Học sinh, sinh viên
            elif any(k in occ for k in ["học sinh", "sinh viên", "student", "teacher", "giáo viên", "giáo dục"]):
                if any(k in title_desc_tags for k in ["học tập", "bàn học", "study", "bookshelf", "giá rẻ", "tiện lợi", "mini"]):
                    score += 1.0
                    matches += 1
            else:
                score += 0.5
                matches += 1
                
        # 4. Income Match
        if income:
            total_checks += 1
            price = cand.get("price", 0.0)
            
            # Quy đổi thu nhập ra VND
            actual_income = income
            if income < 100000:
                # Giả sử người dùng nhập triệu đồng (ví dụ 15 triệu -> 15000000)
                actual_income = income * 1000000
                
            # Quy đổi giá sản phẩm ra VND
            actual_price = price
            if price < 100000:
                # Quy đổi Dollar sang VND (1 USD = 25000 VND)
                actual_price = price * 25000
                
            if actual_income >= 20000000: # Thu nhập cao
                if actual_price >= 5000000: # Thích đồ cao cấp/luxury
                    score += 1.0
                    matches += 1
                else:
                    score += 0.5
            elif actual_income <= 8000000: # Thu nhập thấp
                if actual_price <= 2000000: # Thích đồ kinh tế/giá rẻ
                    score += 1.0
                    matches += 1
                else:
                    score += 0.2
            else: # Trung bình
                if actual_price > 1500000 and actual_price < 6000000:
                    score += 1.0
                    matches += 1
                else:
                    score += 0.6
                    
        # 5. Marital Status Match
        if marital_status:
            total_checks += 1
            ms = marital_status.lower()
            if any(k in ms for k in ["kết hôn", "gia đình", "married", "family"]):
                if any(k in title_desc_tags for k in ["gia đình", "sofa lớn", "giường đôi", "tủ bếp", "kids", "trẻ em", "bàn ăn"]):
                    score += 1.0
                    matches += 1
                else:
                    score += 0.4
            else:
                # Độc thân/Single
                if any(k in title_desc_tags for k in ["độc thân", "đơn", "compact", "tối giản", "mini", "tiết kiệm", "cá nhân"]):
                    score += 1.0
                    matches += 1
                else:
                    score += 0.5
                    
        return score / total_checks if total_checks > 0 else 0.5


recommendation_service = RecommendationService()
