"""
Recommendation Service - Similar products & history-based recommendations.
"""
import time
import datetime
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

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
    conn = None
    try:
        conn = get_mysql_connection()
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
        if conn:
            conn.close()

def get_user_item_interactions():
    conn = None
    try:
        conn = get_mysql_connection()
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
        if conn:
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


def get_user_interactions_by_id(user_id: int) -> List[Dict]:
    conn = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            query = """
            SELECT p.uuid as product_id, p.category_id, c.name as category_name, p.brand,
                   ube.event_type, ube.created_at,
                   COALESCE((SELECT MIN(v.price) FROM product_variants v WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 0) as price
            FROM user_behavior_events ube
            JOIN products p ON ube.product_id = p.id
            LEFT JOIN categories c ON p.category_id = c.id
            WHERE ube.user_id = %s AND ube.event_type IN ('purchase', 'add_to_cart', 'add_to_wishlist', 'product_view')
            ORDER BY ube.created_at DESC;
            """
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            for r in rows:
                r["price"] = float(r["price"]) if r["price"] is not None else 0.0
            return list(rows)
    except Exception as e:
        logger.error(f"Error fetching user interactions for user_id {user_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_guest_interactions_by_session(session_id: str) -> List[Dict]:
    if not session_id:
        return []
    conn = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            query = """
            SELECT p.uuid as product_id, p.category_id, c.name as category_name, p.brand,
                   ube.event_type, ube.created_at,
                   COALESCE((SELECT MIN(v.price) FROM product_variants v WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 0) as price
            FROM user_behavior_events ube
            JOIN products p ON ube.product_id = p.id
            LEFT JOIN categories c ON p.category_id = c.id
            WHERE ube.session_id = %s AND ube.user_id IS NULL AND ube.event_type IN ('purchase', 'add_to_cart', 'add_to_wishlist', 'product_view')
            ORDER BY ube.created_at DESC;
            """
            cursor.execute(query, (session_id,))
            rows = cursor.fetchall()
            for r in rows:
                r["price"] = float(r["price"]) if r["price"] is not None else 0.0
            return list(rows)
    except Exception as e:
        logger.error(f"Error fetching guest interactions for session {session_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_products_metadata(uuids: List[str]) -> Dict[str, Dict]:
    if not uuids:
        return {}
    conn = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            format_strings = ','.join(['%s'] * len(uuids))
            query = f"""
            SELECT p.uuid as product_id, c.name as category_name, p.brand, p.created_at,
                   COALESCE((SELECT MIN(v.price) FROM product_variants v WHERE v.product_id = p.id AND v.is_active = 1 AND v.deleted_at IS NULL), 0) as price
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            WHERE p.uuid IN ({format_strings});
            """
            cursor.execute(query, tuple(uuids))
            rows = cursor.fetchall()
            result = {}
            for r in rows:
                r["price"] = float(r["price"]) if r["price"] is not None else 0.0
                result[r["product_id"]] = r
            return result
    except Exception as e:
        logger.error(f"Error fetching product metadata: {e}")
        return {}
    finally:
        if conn:
            conn.close()


def get_products_created_at(uuids: List[str]) -> Dict[str, Any]:
    if not uuids:
        return {}
    conn = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            format_strings = ','.join(['%s'] * len(uuids))
            query = f"SELECT uuid, created_at FROM products WHERE uuid IN ({format_strings});"
            cursor.execute(query, tuple(uuids))
            rows = cursor.fetchall()
            return {row['uuid']: row['created_at'] for row in rows}
    except Exception as e:
        logger.error(f"Error fetching created_at for products: {e}")
        return {}
    finally:
        if conn:
            conn.close()



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

    def update_user_interest_profile(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        guest_viewed_uuids: List[str] = None
    ) -> Optional[Dict]:
        """
        Builds and updates the User Interest Profile in Redis.
        """
        import uuid
        def get_point_uuid(product_id: str) -> str:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(product_id)))

        # Sanitize inputs
        if user_id in ("null", "None", "", "undefined"):
            user_id = None
        if session_id in ("null", "None", "", "undefined"):
            session_id = None

        target_id = user_id or session_id
        if not target_id:
            return None

        # 1. Fetch behavior events from MySQL
        interactions = []
        if user_id:
            try:
                interactions = list(get_user_interactions_by_id(int(user_id)))
            except (ValueError, TypeError):
                interactions = list(get_user_interactions_by_id(user_id))
        elif session_id:
            interactions = list(get_guest_interactions_by_session(session_id))

        # Fallback to guest_viewed_uuids if no DB interactions found (especially for guests)
        if not interactions and guest_viewed_uuids:
            meta_map = get_products_metadata(guest_viewed_uuids)
            for idx, pid in enumerate(guest_viewed_uuids):
                meta = meta_map.get(pid) or {}
                interactions.append({
                    "product_id": pid,
                    "category_name": meta.get("category_name"),
                    "brand": meta.get("brand"),
                    "price": meta.get("price", 0.0),
                    "event_type": "product_view",
                    "created_at": datetime.datetime.now() - datetime.timedelta(hours=idx)
                })

        if not interactions:
            logger.info(f"No interactions found to build profile for {target_id}")
            return None

        # 2. Process Interactions: Dynamic Weights & Recency Decay
        EVENT_WEIGHTS = {
            "purchase": 1.0,
            "add_to_cart": 0.8,
            "add_to_wishlist": 0.6,
            "product_view": 0.3
        }

        product_weights = {}
        category_affinity = {}
        brand_affinity = {}
        prices = []

        # Find recent interest (dominant category in last 5 events)
        recent_events = interactions[:5]
        recent_categories = [e["category_name"] for e in recent_events if e.get("category_name")]
        recent_brands = [e["brand"] for e in recent_events if e.get("brand")]
        
        dominant_recent_category = max(set(recent_categories), key=recent_categories.count) if recent_categories else None
        dominant_recent_brand = max(set(recent_brands), key=recent_brands.count) if recent_brands else None

        for inter in interactions:
            pid = inter["product_id"]
            cat = inter["category_name"]
            brand = inter["brand"]
            price = inter.get("price", 0.0)
            event_type = inter["event_type"]
            created_at = inter["created_at"]

            if isinstance(created_at, str):
                try:
                    created_at = datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    created_at = datetime.datetime.now()

            delta_days = (datetime.datetime.now() - created_at).days

            # Recency decay multiplier
            if delta_days <= 7:
                recency_mult = 1.0
            elif delta_days <= 30:
                recency_mult = 0.5
            elif delta_days <= 90:
                recency_mult = 0.2
            else:
                recency_mult = 0.05

            # Apply Recent Interest shift boost
            category_boost = 1.0
            if dominant_recent_category and cat == dominant_recent_category:
                category_boost = 2.0
            elif dominant_recent_category and cat not in recent_categories:
                category_boost = 0.5

            brand_boost = 1.0
            if dominant_recent_brand and brand == dominant_recent_brand:
                brand_boost = 2.0
            elif dominant_recent_brand and brand not in recent_brands:
                brand_boost = 0.5

            event_weight = EVENT_WEIGHTS.get(event_type, 0.3)
            inter_weight = event_weight * recency_mult

            product_weights[pid] = product_weights.get(pid, 0.0) + inter_weight

            if cat:
                category_affinity[cat] = category_affinity.get(cat, 0.0) + inter_weight * category_boost
            if brand:
                brand_affinity[brand] = brand_affinity.get(brand, 0.0) + inter_weight * brand_boost
            if price > 0:
                prices.append(price)

        # Normalize affinities
        if category_affinity:
            max_cat = max(category_affinity.values())
            if max_cat > 0:
                category_affinity = {k: round(v / max_cat, 4) for k, v in category_affinity.items()}

        if brand_affinity:
            max_brand = max(brand_affinity.values())
            if max_brand > 0:
                brand_affinity = {k: round(v / max_brand, 4) for k, v in brand_affinity.items()}

        # 3. Calculate Price Affinity (Avg, Std, Min/Max range)
        price_affinity = {
            "avg_price": 0.0,
            "std_dev": 0.0,
            "min_price": 0.0,
            "max_price": 0.0
        }
        if prices:
            avg_price = float(np.mean(prices))
            std_dev = float(np.std(prices)) if len(prices) > 1 else avg_price * 0.4
            if std_dev == 0:
                std_dev = avg_price * 0.4
            
            min_price = max(0.0, avg_price - 2.0 * std_dev)
            max_price = avg_price + 2.0 * std_dev

            price_affinity = {
                "avg_price": round(avg_price, 2),
                "std_dev": round(std_dev, 2),
                "min_price": round(min_price, 2),
                "max_price": round(max_price, 2)
            }

        # 4. User Interest Vector (Weighted Average Embedding)
        profile_vector = []
        user_interacted_pids = list(product_weights.keys())[:30]
        if user_interacted_pids:
            try:
                user_uuids = [get_point_uuid(pid) for pid in user_interacted_pids]
                retrieved_points = qdrant_service.client.retrieve(
                    collection_name=qdrant_service.collection_name,
                    ids=user_uuids,
                    with_payload=False,
                    with_vectors=True
                )
                pid_to_vector = {}
                for pt in retrieved_points:
                    uuid_str = str(pt.id)
                    pid = None
                    for original_pid in user_interacted_pids:
                        if get_point_uuid(original_pid) == uuid_str:
                            pid = original_pid
                            break
                    if pid and pt.vector:
                        pid_to_vector[pid] = pt.vector

                weighted_sum_vec = None
                sum_weights = 0.0
                for pid in user_interacted_pids:
                    if pid in pid_to_vector:
                        vec = np.array(pid_to_vector[pid])
                        weight = product_weights[pid]
                        if weighted_sum_vec is None:
                            weighted_sum_vec = vec * weight
                        else:
                            weighted_sum_vec += vec * weight
                        sum_weights += weight

                if weighted_sum_vec is not None and sum_weights > 0:
                    avg_vec = weighted_sum_vec / sum_weights
                    norm = np.linalg.norm(avg_vec)
                    if norm > 0:
                        avg_vec = avg_vec / norm
                    profile_vector = avg_vec.tolist()
            except Exception as e:
                logger.error(f"Error computing Weighted Average Vector: {e}", exc_info=True)

        # 5. Build Final Profile Dict
        profile = {
            "user_vector": profile_vector,
            "category_affinity": category_affinity,
            "brand_affinity": brand_affinity,
            "price_affinity": price_affinity,
            "recent_interest": {
                "category": dominant_recent_category,
                "brand": dominant_recent_brand,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            },
            "interaction_count": len(interactions),
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }

        # 6. Save to Redis
        redis_service.set_user_profile(target_id, profile)
        return profile

    def run_collaborative_filtering_job(self) -> Dict[str, Any]:
        """
        Precomputes similar items and user collaborative recommendations and caches them in Redis.
        """
        logger.info("Starting scheduled Collaborative Filtering job...")
        t0 = time.perf_counter()
        
        interactions = get_user_item_interactions()
        if not interactions:
            logger.warning("No interaction logs found for Collaborative Filtering.")
            return {"status": "skipped", "message": "No interactions found"}

        try:
            users = list(set(row['user_id'] for row in interactions))
            items = list(set(row['product_id'] for row in interactions))
            
            user_to_idx = {uid: i for i, uid in enumerate(users)}
            item_to_idx = {pid: i for i, pid in enumerate(items)}
            idx_to_item = {i: pid for pid, i in item_to_idx.items()}
            
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

            for idx, pid in enumerate(items):
                sim_scores = item_similarity[idx]
                top_indices = np.argsort(sim_scores)[::-1][:11]
                similar_pids = []
                for t_idx in top_indices:
                    sim_pid = idx_to_item[t_idx]
                    if sim_pid != pid and sim_scores[t_idx] > 0.01:
                        similar_pids.append(sim_pid)
                
                redis_service.client.set(f"recommend:similar_products:{pid}", ",".join(similar_pids))

            for uid in users:
                u_idx = user_to_idx[uid]
                user_vector = matrix[u_idx]
                scores = np.dot(item_similarity, user_vector)
                
                max_score = np.max(scores) if np.max(scores) > 0 else 1.0
                scores = scores / max_score

                user_candidates = []
                for idx in range(num_items):
                    if user_vector[idx] > 0:
                        continue
                    score = float(scores[idx])
                    if score > 0.01:
                        user_candidates.append(f"{idx_to_item[idx]}:{score:.4f}")
                
                user_candidates.sort(key=lambda x: float(x.split(":")[1]), reverse=True)
                redis_service.client.set(f"recommend:similar_users:{uid}", ",".join(user_candidates[:30]))

            duration_s = round(time.perf_counter() - t0, 2)
            logger.info(f"Collaborative Filtering Job completed. Cached {len(items)} products and {len(users)} users in {duration_s}s.")
            return {"status": "success", "users_processed": len(users), "products_processed": len(items), "duration_s": duration_s}
        except Exception as e:
            logger.error(f"Error running collaborative filtering: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

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

        # Sanitize user_id & session_id
        if user_id in ("null", "None", "", "undefined"):
            user_id = None
        
        target_id = user_id
        
        # ── 1. Đọc từ Redis Cache Gợi Ý Trang Chủ ─────────────────────────────
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

        # Find session_id if user is guest
        if not user_id:
            target_id = "guest_session"
            if viewed:
                target_id = f"guest_{viewed[0]}"

        # ── 2. Đọc User Interest Profile từ Redis ─────────────────────────────
        profile = redis_service.get_user_profile(target_id)
        if not profile:
            logger.info(f"User Profile MISS in Redis for {target_id}. Rebuilding profile...")
            profile = self.update_user_interest_profile(
                user_id=user_id,
                session_id=None if user_id else target_id,
                guest_viewed_uuids=viewed
            )

        # ── 3. Fallback: Trả về sản phẩm phổ biến nếu không có Profile ────────
        if not profile or not profile.get("user_vector"):
            logger.info("Hybrid Rec: No user vector profile found. Falling back to popular.")
            fallback_ids = popular if popular else []
            results = []
            if fallback_ids:
                try:
                    fallback_uuids = [get_point_uuid(pid) for pid in fallback_ids[:top_k]]
                    retrieved_fallbacks = qdrant_service.client.retrieve(
                        collection_name=qdrant_service.collection_name,
                        ids=fallback_uuids,
                        with_payload=True,
                        with_vectors=False
                    )
                    fb_map = {}
                    for pt in retrieved_fallbacks:
                        p = pt.payload or {}
                        pid = p.get("product_id") or fallback_ids[0]
                        fb_map[pid] = {
                            "product_id": pid,
                            "title": p.get("title", ""),
                            "description": p.get("description", ""),
                            "category": p.get("category", ""),
                            "tags": p.get("tags", []),
                            "price": p.get("price", 0.0),
                            "image_url": p.get("image_url", ""),
                        }
                    for pid in fallback_ids[:top_k]:
                        if pid in fb_map:
                            p = fb_map[pid]
                            p["score"] = 1.0
                            p["reason"] = "Sản phẩm xu hướng được mua nhiều nhất"
                            results.append(p)
                except Exception as e:
                    logger.error(f"Error fetching popular fallback: {e}")
            
            if not results:
                scroll_res, _ = qdrant_service.client.scroll(collection_name=qdrant_service.collection_name, limit=top_k, with_payload=True)
                for r in scroll_res:
                    p = r.payload
                    results.append({
                        "product_id": p.get("product_id"),
                        "title": p.get("title"),
                        "description": p.get("description"),
                        "category": p.get("category"),
                        "tags": p.get("tags"),
                        "price": p.get("price"),
                        "image_url": p.get("image_url"),
                        "score": 0.5,
                        "reason": "Sản phẩm gợi ý hôm nay"
                    })
            
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return results[:top_k], latency_ms

        # Extract Profile fields
        user_vector = profile["user_vector"]
        category_affinity = profile.get("category_affinity") or {}
        brand_affinity = profile.get("brand_affinity") or {}
        price_affinity = profile.get("price_affinity") or {}
        interaction_count = profile.get("interaction_count", 0)
        recent_interest = profile.get("recent_interest") or {}

        # ── 4. Tìm kiếm các ứng viên từ Qdrant ───────────────────────────────
        exclude = viewed + purchased if exclude_viewed else None
        qdrant_candidates = qdrant_service.search(
            query_vector=user_vector,
            top_k=top_k * 8,
            exclude_ids=exclude,
            with_vectors=True
        )

        # ── 5. Đọc Collaborative Filtering đã tính toán trước từ Redis ──────
        cf_candidates_dict = {}
        if user_id:
            cf_data = redis_service.client.get(f"recommend:similar_users:{user_id}")
            if cf_data:
                try:
                    for pair in cf_data.decode("utf-8").split(","):
                        if ":" in pair:
                            pid, score = pair.split(":")
                            cf_candidates_dict[pid] = float(score)
                except Exception as e:
                    logger.error(f"Error parsing cached CF candidates: {e}")

        # ── 6. Hợp nhất danh sách ứng viên (Qdrant + CF + Popular + Loc) ──────
        candidates_map = {cand["product_id"]: cand for cand in qdrant_candidates}
        
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
                    "score": 0.5,
                    "vector": None
                }

        candidates = list(candidates_map.values())

        # ── 7. Lấy ngày tạo (created_at) cho tất cả ứng viên để tính Freshness ──
        candidate_ids = [c["product_id"] for c in candidates]
        created_at_map = get_products_created_at(candidate_ids)

        # ── 8. Tính toán Trọng Số Cấu Phần Động (Dynamic Weights) ─────────────
        personal_ratio = min(1.0, interaction_count / 10.0)
        
        w_content = 0.10 + 0.25 * personal_ratio
        w_behavior = 0.10 + 0.15 * personal_ratio
        w_collaborative = 0.05 + 0.15 * personal_ratio
        w_popularity = 0.45 - 0.40 * personal_ratio
        w_demographic = 0.20 - 0.10 * personal_ratio
        w_location = 0.10 - 0.05 * personal_ratio
        w_freshness = 0.05

        sum_w = w_content + w_behavior + w_collaborative + w_popularity + w_demographic + w_location + w_freshness
        w_content /= sum_w
        w_behavior /= sum_w
        w_collaborative /= sum_w
        w_popularity /= sum_w
        w_demographic /= sum_w
        w_location /= sum_w
        w_freshness /= sum_w

        # ── 9. Chấm điểm chi tiết từng ứng viên ──────────────────────────────
        scored_candidates = []
        for cand in candidates:
            pid = cand.get("product_id")
            cand_category = cand.get("category") or ""
            cand_brand = cand.get("brand") or ""
            cand_price = cand.get("price") or 0.0

            # A. Content Score
            content_score = cand.get("score", 0.5)

            # B. Behavior Score
            behavior_score = content_score

            # C. Collaborative Score
            collaborative_score = 0.1
            if pid in cf_candidates_dict:
                collaborative_score = cf_candidates_dict[pid]
            else:
                if cand_category in category_affinity:
                    collaborative_score += 0.4

            # D. Popularity Score
            popularity_score = 0.1
            if pid in popular:
                try:
                    idx = popular.index(pid)
                    popularity_score = max(0.2, 1.0 - (idx * 0.05))
                except ValueError:
                    popularity_score = 0.5

            # E. Demographic Score
            demographic_score = self.calculate_demographic_score(
                cand=cand,
                age=user_age,
                gender=user_gender,
                occupation=user_occupation,
                income=user_income,
                marital_status=user_marital_status
            )

            # F. Location Score
            location_score = 0.1
            if pid in loc_popular:
                try:
                    idx = loc_popular.index(pid)
                    location_score = max(0.3, 1.0 - (idx * 0.05))
                except ValueError:
                    location_score = 0.6

            # G. Freshness Score
            created_at = created_at_map.get(pid)
            if created_at:
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        created_at = datetime.datetime.now()
                delta_days = (datetime.datetime.now() - created_at).days
                freshness_score = float(np.exp(-delta_days / 30.0))
            else:
                freshness_score = 0.0

            # H. Price Affinity Score
            avg_price = price_affinity.get("avg_price", 0.0)
            std_dev = price_affinity.get("std_dev", 0.0)
            price_score = 1.0
            if avg_price > 0 and std_dev > 0:
                price_score = float(np.exp(-((cand_price - avg_price) ** 2) / (2 * (std_dev ** 2))))

            base_score = (
                (w_content * content_score) +
                (w_behavior * behavior_score) +
                (w_collaborative * collaborative_score) +
                (w_popularity * popularity_score) +
                (w_demographic * demographic_score) +
                (w_location * location_score) +
                (w_freshness * freshness_score)
            )

            cat_affinity = category_affinity.get(cand_category, 0.0)
            brand_affinity_val = brand_affinity.get(cand_brand, 0.0)
            
            category_boost = cat_affinity * 0.3
            brand_boost = brand_affinity_val * 0.2
            price_multiplier = 0.5 + 0.5 * price_score

            final_score = (base_score + category_boost + brand_boost) * price_multiplier

            reason = "Gợi ý phù hợp với sở thích của bạn"
            if category_boost > 0.15 and brand_boost > 0.1:
                reason = f"Bạn đã xem nhiều {cand_category} {cand_brand} gần đây"
            elif category_boost > 0.15:
                reason = f"Phù hợp với sở thích {cand_category} của bạn"
            elif location_score > 0.6:
                reason = f"Đang bán chạy tại khu vực của bạn"
            elif collaborative_score > 0.6:
                reason = f"Khách hàng mua {cand_category} tương tự cũng mua"
            elif freshness_score > 0.8:
                reason = f"Sản phẩm {cand_category} mới ra mắt"

            cand_copy = cand.copy()
            cand_copy["score"] = round(final_score, 4)
            cand_copy["reason"] = reason
            cand_copy["scores_detail"] = {
                "final": round(final_score, 4),
                "content": round(content_score, 4),
                "behavior": round(behavior_score, 4),
                "collab": round(collaborative_score, 4),
                "popularity": round(popularity_score, 4),
                "demographic": round(demographic_score, 4),
                "location": round(location_score, 4),
                "freshness": round(freshness_score, 4),
                "price_score": round(price_score, 4),
                "cat_boost": round(category_boost, 4),
                "brand_boost": round(brand_boost, 4)
            }
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # ── 10. Diversity Re-ranking (Category Diversity Penalty) ────────────
        results = []
        category_counts = {}
        remaining_pool = list(scored_candidates)

        while len(results) < top_k and remaining_pool:
            best_idx = -1
            best_penalized_score = -9999.0
            
            for idx, cand in enumerate(remaining_pool):
                cat = cand.get("category") or "default"
                count = category_counts.get(cat, 0)
                penalized_score = cand["score"] - (count * 0.15)
                
                if penalized_score > best_penalized_score:
                    best_penalized_score = penalized_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_cand = remaining_pool.pop(best_idx)
                results.append(selected_cand)
                cat = selected_cand.get("category") or "default"
                category_counts[cat] = category_counts.get(cat, 0) + 1
            else:
                break

        for i, r in enumerate(results, 1):
            r["rank"] = i

        # ── 11. Log Details for Observability ────────────────────────────────
        logger.info(f"--- [FastAPI Recommendation Log for User={target_id}] ---")
        for r in results[:5]:
            det = r["scores_detail"]
            logger.info(
                f"Rank {r['rank']} | Product: {r['title'][:25]} ({r['product_id'][:8]}) | "
                f"Final: {det['final']:.4f} | Content: {det['content']:.4f} | Behavior: {det['behavior']:.4f} | "
                f"Collab: {det['collab']:.4f} | Popular: {det['popularity']:.4f} | "
                f"Demo: {det['demographic']:.4f} | Loc: {det['location']:.4f} | Fresh: {det['freshness']:.4f} | "
                f"PriceScore: {det['price_score']:.4f} | CatBoost: {det['cat_boost']:.4f} | BrandBoost: {det['brand_boost']:.4f} | "
                f"Reason: {r['reason']}"
            )
        logger.info(f"-------------------------------------------------------")

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
