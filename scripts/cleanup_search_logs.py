import os
import pymysql
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cleanup_search_logs")

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "230605")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "ecommerce")


def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor
    )


# SQL patterns matching malicious queries
MALICIOUS_SQL_PATTERNS = [
    "%or 1=1%",
    "%and 1=1%",
    "%1=1--%",
    "%-- -%",
    "%union select%",
    "%<script%",
    "%drop table%",
    "%exec %",
    "%sleep(%"
]


def cleanup_malicious_search_logs():
    logger.info("Starting cleanup of malicious search logs...")
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            total_deleted = 0
            for pattern in MALICIOUS_SQL_PATTERNS:
                # Find matching search log IDs
                cursor.execute("SELECT id, query_text FROM search_logs WHERE LOWER(query_text) LIKE %s OR LOWER(keyword) LIKE %s;", (pattern, pattern))
                rows = cursor.fetchall()
                if rows:
                    log_ids = [r["id"] for r in rows]
                    logger.info(f"Found {len(log_ids)} records matching pattern '{pattern}'")
                    
                    # Delete from search_log_products if table exists
                    format_strings = ','.join(['%s'] * len(log_ids))
                    cursor.execute(f"DELETE FROM search_log_products WHERE search_log_id IN ({format_strings});", tuple(log_ids))
                    
                    # Delete from search_logs
                    cursor.execute(f"DELETE FROM search_logs WHERE id IN ({format_strings});", tuple(log_ids))
                    total_deleted += len(log_ids)

            # Sync missing columns for old Spring Boot / FastAPI logs
            cursor.execute("UPDATE search_logs SET searched_at = created_at WHERE searched_at IS NULL AND created_at IS NOT NULL;")
            cursor.execute("UPDATE search_logs SET keyword = query_text WHERE (keyword IS NULL OR TRIM(keyword) = '') AND (query_text IS NOT NULL AND TRIM(query_text) != '');")
            cursor.execute("UPDATE search_logs SET query_text = keyword WHERE (query_text IS NULL OR TRIM(query_text) = '') AND (keyword IS NOT NULL AND TRIM(keyword) != '');")
            cursor.execute("UPDATE search_logs SET response_time_ms = latency_ms WHERE (response_time_ms IS NULL OR response_time_ms = 0) AND latency_ms > 0;")
            cursor.execute("UPDATE search_logs SET total_results = results_count WHERE (total_results IS NULL OR total_results = 0) AND results_count > 0;")

            # Delete any punctuation-only or empty/spaces queries
            cursor.execute("DELETE FROM search_logs WHERE (query_text IS NULL OR TRIM(query_text) = '' OR query_text = '-') AND (keyword IS NULL OR TRIM(keyword) = '' OR keyword = '-');")
            punct_deleted = cursor.rowcount
            logger.info(f"Deleted {punct_deleted} punctuation-only or empty search log records.")

            conn.commit()
            logger.info(f"Successfully cleaned up {total_deleted + punct_deleted} search log records.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during search log cleanup: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    cleanup_malicious_search_logs()
