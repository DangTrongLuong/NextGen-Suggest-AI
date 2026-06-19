import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from app.features.search_analytics.repositories.connection_pool import get_db_connection

logger = logging.getLogger("analytics_repository")

class AnalyticsRepository:
    def get_overview_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Fetch general KPIs for the selected date range."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # 1. Total searches, unique keywords, average latency, zero result count, average results
                query_stats = """
                SELECT 
                    COUNT(*) as total_searches,
                    COUNT(DISTINCT keyword) as unique_keywords,
                    COALESCE(AVG(response_time_ms), 0.0) as average_latency,
                    SUM(CASE WHEN total_results = 0 THEN 1 ELSE 0 END) as zero_results,
                    COALESCE(AVG(total_results), 0.0) as avg_results
                FROM search_logs
                WHERE searched_at BETWEEN %s AND %s;
                """
                cursor.execute(query_stats, (start_date, end_date))
                stats_row = cursor.fetchone() or {}

                total = stats_row.get("total_searches", 0)

                # 2. Click statistics (CTR)
                # CTR = total clicks / total searches (expressed as percentage)
                # Search-to-Click Rate = unique searches with clicks / total searches
                ctr = 0.0
                search_to_click = 0.0
                if total > 0:
                    query_clicks = """
                    SELECT 
                        COUNT(c.id) as total_clicks,
                        COUNT(DISTINCT c.request_id) as unique_clicked_searches
                    FROM search_click_logs c
                    JOIN search_logs s ON c.request_id = s.request_id
                    WHERE s.searched_at BETWEEN %s AND %s;
                    """
                    cursor.execute(query_clicks, (start_date, end_date))
                    clicks_row = cursor.fetchone() or {}
                    
                    total_clicks = float(clicks_row.get("total_clicks", 0) or 0)
                    unique_clicks = float(clicks_row.get("unique_clicked_searches", 0) or 0)
                    ctr = (total_clicks / total) * 100.0
                    search_to_click = (unique_clicks / total) * 100.0

                # 3. Time period counts (Today, 7 days, 30 days)
                now = datetime.now()
                today_start = datetime(now.year, now.month, now.day)
                seven_days_ago = now - timedelta(days=7)
                thirty_days_ago = now - timedelta(days=30)

                cursor.execute("SELECT COUNT(*) as count FROM search_logs WHERE searched_at >= %s;", (today_start,))
                today_count = cursor.fetchone().get("count", 0)

                cursor.execute("SELECT COUNT(*) as count FROM search_logs WHERE searched_at >= %s;", (seven_days_ago,))
                seven_days_count = cursor.fetchone().get("count", 0)

                cursor.execute("SELECT COUNT(*) as count FROM search_logs WHERE searched_at >= %s;", (thirty_days_ago,))
                thirty_days_count = cursor.fetchone().get("count", 0)

                zero_results = float(stats_row.get("zero_results", 0) or 0)
                zero_result_rate = (zero_results / total * 100.0) if total > 0 else 0.0

                return {
                    "total_searches": total,
                    "unique_keywords": stats_row.get("unique_keywords", 0),
                    "average_response_time_ms": round(float(stats_row.get("average_latency", 0.0)), 2),
                    "zero_result_searches": int(stats_row.get("zero_results", 0) or 0),
                    "zero_result_rate": round(zero_result_rate, 2),
                    "average_results_per_query": round(float(stats_row.get("avg_results", 0.0)), 2),
                    "ctr": round(ctr, 2),
                    "search_to_click_rate": round(search_to_click, 2),
                    "searches_today": today_count,
                    "searches_last_7_days": seven_days_count,
                    "searches_last_30_days": thirty_days_count
                }
        except Exception as e:
            logger.error(f"Error fetching overview stats: {e}")
            raise e
        finally:
            conn.close()

    def get_search_history(
        self,
        q: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        zero_results_only: bool = False,
        page: int = 0,
        size: int = 20,
        sort_by: str = "searched_at",
        sort_desc: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Fetch a paginated list of search logs."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                conditions = []
                params = []

                if q:
                    conditions.append("keyword LIKE %s")
                    params.append(f"%{q}%")

                if start_date:
                    conditions.append("searched_at >= %s")
                    params.append(start_date)

                if end_date:
                    conditions.append("searched_at <= %s")
                    params.append(end_date)

                if zero_results_only:
                    conditions.append("total_results = 0")

                where_clause = ""
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)

                # Count total
                count_query = f"SELECT COUNT(*) as count FROM search_logs {where_clause};"
                cursor.execute(count_query, tuple(params))
                total_count = cursor.fetchone().get("count", 0)

                # Sorting and paging
                allowed_sort_fields = ["searched_at", "response_time_ms", "total_results", "keyword"]
                if sort_by not in allowed_sort_fields:
                    sort_by = "searched_at"
                
                direction = "DESC" if sort_desc else "ASC"
                
                limit = size
                offset = page * size

                history_query = f"""
                SELECT id, request_id, keyword, response_time_ms, total_results, session_id, searched_at
                FROM search_logs
                {where_clause}
                ORDER BY {sort_by} {direction}
                LIMIT %s OFFSET %s;
                """
                cursor.execute(history_query, tuple(params + [limit, offset]))
                logs = cursor.fetchall()

                return logs, total_count
        except Exception as e:
            logger.error(f"Error fetching search history: {e}")
            raise e
        finally:
            conn.close()

    def get_search_products(self, search_log_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """Bulk fetch product rankings associated with search log IDs (prevents N+1 query pattern)."""
        if not search_log_ids:
            return {}
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                format_strings = ",".join(["%s"] * len(search_log_ids))
                query = f"""
                SELECT search_log_id, product_id, product_name, ranking_position
                FROM search_log_products
                WHERE search_log_id IN ({format_strings})
                ORDER BY search_log_id, ranking_position ASC;
                """
                cursor.execute(query, tuple(search_log_ids))
                rows = cursor.fetchall()
                
                mapping = {}
                for r in rows:
                    log_id = r["search_log_id"]
                    if log_id not in mapping:
                        mapping[log_id] = []
                    mapping[log_id].append({
                        "product_id": r["product_id"],
                        "product_name": r["product_name"],
                        "ranking_position": r["ranking_position"]
                    })
                return mapping
        except Exception as e:
            logger.error(f"Error bulk fetching search products: {e}")
            return {}
        finally:
            conn.close()

    def get_search_clicks(self, request_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Bulk fetch product clicks associated with search request UUIDs (prevents N+1 query pattern)."""
        # Filter out empty/None request_ids
        request_ids = [rid for rid in request_ids if rid]
        if not request_ids:
            return {}
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                format_strings = ",".join(["%s"] * len(request_ids))
                query = f"""
                SELECT request_id, product_id, product_name, clicked_at
                FROM search_click_logs
                WHERE request_id IN ({format_strings})
                ORDER BY clicked_at ASC;
                """
                cursor.execute(query, tuple(request_ids))
                rows = cursor.fetchall()

                mapping = {}
                for r in rows:
                    req_id = r["request_id"]
                    if req_id not in mapping:
                        mapping[req_id] = []
                    mapping[req_id].append({
                        "product_id": r["product_id"],
                        "product_name": r["product_name"],
                        "clicked_at": r["clicked_at"]
                    })
                return mapping
        except Exception as e:
            logger.error(f"Error bulk fetching search click logs: {e}")
            return {}
        finally:
            conn.close()

    def get_search_detail(self, log_id: int) -> Optional[Dict[str, Any]]:
        """Fetch details of a single search log."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, request_id, keyword, response_time_ms, total_results, session_id, searched_at FROM search_logs WHERE id = %s;",
                    (log_id,)
                )
                log = cursor.fetchone()
                return log
        except Exception as e:
            logger.error(f"Error fetching search detail: {e}")
            raise e
        finally:
            conn.close()

    def get_top_keywords(self, start_date: datetime, end_date: datetime, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch top keywords aggregated with statistics (searches, latency, zero results, clicks, CTR)."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                query = """
                SELECT 
                    s.keyword,
                    COUNT(*) as search_count,
                    SUM(CASE WHEN s.total_results = 0 THEN 1 ELSE 0 END) as zero_result_count,
                    AVG(s.response_time_ms) as avg_latency_ms,
                    (
                        SELECT COUNT(*) 
                        FROM search_click_logs cl
                        JOIN search_logs sl ON cl.request_id = sl.request_id
                        WHERE sl.keyword = s.keyword AND sl.searched_at BETWEEN %s AND %s
                    ) as click_count
                FROM search_logs s
                WHERE s.searched_at BETWEEN %s AND %s
                GROUP BY s.keyword
                ORDER BY search_count DESC
                LIMIT %s;
                """
                cursor.execute(query, (start_date, end_date, start_date, end_date, limit))
                rows = cursor.fetchall()
                
                for r in rows:
                    total = r["search_count"]
                    clicks = r["click_count"]
                    r["avg_latency_ms"] = round(float(r["avg_latency_ms"]), 2)
                    r["ctr"] = round((clicks / total * 100.0), 2) if total > 0 else 0.0
                return rows
        except Exception as e:
            logger.error(f"Error fetching top keywords: {e}")
            raise e
        finally:
            conn.close()

    def get_top_products(self, start_date: datetime, end_date: datetime, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch top returned products (appearances, average rank, last appearance, click count)."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                query = """
                SELECT 
                    slp.product_id,
                    slp.product_name,
                    COUNT(*) as appearance_count,
                    AVG(slp.ranking_position) as average_ranking,
                    MAX(sl.searched_at) as last_appeared,
                    (
                        SELECT COUNT(*)
                        FROM search_click_logs cl
                        WHERE cl.product_id = slp.product_id AND cl.clicked_at BETWEEN %s AND %s
                    ) as click_count
                FROM search_log_products slp
                JOIN search_logs sl ON slp.search_log_id = sl.id
                WHERE sl.searched_at BETWEEN %s AND %s
                GROUP BY slp.product_id, slp.product_name
                ORDER BY appearance_count DESC
                LIMIT %s;
                """
                cursor.execute(query, (start_date, end_date, start_date, end_date, limit))
                rows = cursor.fetchall()
                
                for r in rows:
                    r["average_ranking"] = round(float(r["average_ranking"]), 2)
                return rows
        except Exception as e:
            logger.error(f"Error fetching top products: {e}")
            raise e
        finally:
            conn.close()

    def insert_click_log(self, request_id: str, product_id: str, product_name: str) -> bool:
        """Record product click log."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO search_click_logs (request_id, product_id, product_name, clicked_at)
                VALUES (%s, %s, %s, %s);
                """
                cursor.execute(query, (request_id, product_id, product_name, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error inserting click log: {e}")
            return False
        finally:
            conn.close()

    def insert_audit_log(
        self,
        event_type: str,
        username: str,
        ip_address: str,
        user_agent: Optional[str] = None,
        details: Optional[str] = None
    ) -> bool:
        """Insert a security/activity event into the analytics audit log table."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO analytics_audit_logs (event_type, username, ip_address, user_agent, details, created_at)
                VALUES (%s, %s, %s, %s, %s, %s);
                """
                cursor.execute(query, (event_type, username, ip_address, user_agent, details, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
            return False
        finally:
            conn.close()

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent security audit logs."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                query = """
                SELECT id, event_type, username, ip_address, user_agent, details, created_at
                FROM analytics_audit_logs
                ORDER BY created_at DESC
                LIMIT %s;
                """
                cursor.execute(query, (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching audit logs: {e}")
            return []
        finally:
            conn.close()

    def get_timeline_stats(self, start_date: datetime, end_date: datetime, interval: str = "hour") -> List[Dict[str, Any]]:
        """Fetch search volume and response latency grouped by hour or day bucket."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Decide mysql format string
                fmt = "%%Y-%%m-%%d %%H:00" if interval == "hour" else "%%Y-%%m-%%d"
                query = f"""
                SELECT 
                    DATE_FORMAT(searched_at, '{fmt}') as time_bucket,
                    COUNT(*) as search_count,
                    COALESCE(AVG(response_time_ms), 0.0) as avg_latency
                FROM search_logs
                WHERE searched_at BETWEEN %s AND %s
                GROUP BY time_bucket
                ORDER BY time_bucket ASC;
                """
                cursor.execute(query, (start_date, end_date))
                rows = cursor.fetchall()
                for r in rows:
                    r["avg_latency"] = round(float(r["avg_latency"]), 2)
                return rows
        except Exception as e:
            logger.error(f"Error fetching timeline stats: {e}")
            return []
        finally:
            conn.close()

    # ── Isolated Users & Roles DB queries ──
    def find_analytics_user(self, username_or_email: str) -> Optional[Dict[str, Any]]:
        """Find analytics user by username or email."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                query = """
                SELECT u.id, u.username, u.email, u.password_hash, u.is_active, r.name as role
                FROM analytics_users u
                JOIN analytics_user_roles ur ON u.id = ur.user_id
                JOIN analytics_roles r ON ur.role_id = r.id
                WHERE u.username = %s OR u.email = %s;
                """
                cursor.execute(query, (username_or_email, username_or_email))
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error finding analytics user {username_or_email}: {e}")
            return None
        finally:
            conn.close()

    def insert_audit_event_direct(self, event_type: str, username: str, ip: str, details: str):
        """Helper to write audit log without raising exception."""
        try:
            self.insert_audit_log(event_type, username, ip, details=details)
        except Exception:
            pass

# Singleton instance
analytics_repository = AnalyticsRepository()
