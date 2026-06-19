import pymysql
import os
import uuid
import bcrypt
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analytics_migration")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "230605")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "ecommerce")

DEFAULT_USERNAME = os.getenv("ANALYTICS_DEFAULT_USERNAME")
DEFAULT_EMAIL = os.getenv("ANALYTICS_DEFAULT_EMAIL")
DEFAULT_PASSWORD = os.getenv("ANALYTICS_DEFAULT_PASSWORD")

def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor
    )

def table_exists(cursor, table_name):
    cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
    return cursor.fetchone() is not None

def column_exists(cursor, table_name, column_name):
    cursor.execute(f"SHOW COLUMNS FROM {table_name} LIKE '{column_name}';")
    return cursor.fetchone() is not None

def index_exists(cursor, table_name, index_name):
    cursor.execute(f"SHOW INDEX FROM {table_name} WHERE Key_name = '{index_name}';")
    return cursor.fetchone() is not None

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def run_migration():
    logger.info("Starting database migration for Search Analytics Portal...")
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Disable foreign key checks temporarily during migration setup
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")

            # ── 1. Create Isolated User Tables ──
            logger.info("Checking isolated user tables...")
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_users (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                is_active TINYINT(1) DEFAULT 1,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_roles (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(50) UNIQUE NOT NULL,
                description VARCHAR(255) NULL,
                created_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_user_roles (
                user_id BIGINT NOT NULL,
                role_id BIGINT NOT NULL,
                PRIMARY KEY (user_id, role_id),
                FOREIGN KEY (user_id) REFERENCES analytics_users(id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES analytics_roles(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # ── 2. Create Audit Logs Table ──
            logger.info("Checking analytics_audit_logs table...")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_audit_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                event_type VARCHAR(50) NOT NULL,
                username VARCHAR(100) NOT NULL,
                ip_address VARCHAR(45) NOT NULL,
                user_agent VARCHAR(500) NULL,
                details TEXT NULL,
                created_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # ── 3. Alter and Index search_logs ──
            logger.info("Upgrading search_logs schema...")
            if not table_exists(cursor, "search_logs"):
                cursor.execute("""
                CREATE TABLE search_logs (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    query_text TEXT NOT NULL,
                    results_count INT NULL,
                    latency_ms INT NULL,
                    created_at DATETIME(6) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                logger.info("Created search_logs table from scratch.")

            # Add new columns if missing
            cols_to_add = [
                ("request_id", "VARCHAR(100) NULL"),
                ("keyword", "VARCHAR(500) NULL"),
                ("response_time_ms", "INT NULL"),
                ("total_results", "INT NULL"),
                ("searched_at", "DATETIME NULL")
            ]
            for col_name, col_type in cols_to_add:
                if not column_exists(cursor, "search_logs", col_name):
                    cursor.execute(f"ALTER TABLE search_logs ADD COLUMN {col_name} {col_type};")
                    logger.info(f"Added column {col_name} to search_logs.")

            # Create indices for search_logs
            indices_search_logs = [
                ("idx_keyword", "keyword"),
                ("idx_searched_at", "searched_at"),
                ("idx_session_id", "session_id"),
                ("idx_created_at", "created_at"),
                ("idx_keyword_searched_at", "keyword, searched_at")
            ]
            for idx_name, idx_fields in indices_search_logs:
                if not index_exists(cursor, "search_logs", idx_name):
                    # Check if keyword index needs a length specifier because it's VARCHAR(500)
                    # VARCHAR(500) can be indexed completely, but let's be careful. Standard InnoDB allows up to 767 bytes (or 3072 with large prefix).
                    # We can safely index keyword(255) if it's too long, or index directly. Since it's VARCHAR(500), standard MySQL handles it easily.
                    cursor.execute(f"CREATE INDEX {idx_name} ON search_logs ({idx_fields});")
                    logger.info(f"Created index {idx_name} on search_logs.")

            # ── 4. Create search_log_products ──
            logger.info("Checking search_log_products table...")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_log_products (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                search_log_id BIGINT NOT NULL,
                product_id VARCHAR(100) NOT NULL,
                product_name VARCHAR(255) NOT NULL,
                ranking_position INT NOT NULL,
                FOREIGN KEY (search_log_id) REFERENCES search_logs(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Indices for search_log_products
            indices_sl_products = [
                ("idx_search_log_id", "search_log_id"),
                ("idx_product_id", "product_id"),
                ("idx_ranking_position", "ranking_position"),
                ("idx_product_ranking", "product_id, ranking_position")
            ]
            for idx_name, idx_fields in indices_sl_products:
                if not index_exists(cursor, "search_log_products", idx_name):
                    cursor.execute(f"CREATE INDEX {idx_name} ON search_log_products ({idx_fields});")
                    logger.info(f"Created index {idx_name} on search_log_products.")

            # ── 5. Create search_click_logs ──
            logger.info("Checking search_click_logs table...")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_click_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                request_id VARCHAR(100) NOT NULL,
                product_id VARCHAR(100) NOT NULL,
                product_name VARCHAR(255) NOT NULL,
                clicked_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            
            if not index_exists(cursor, "search_click_logs", "idx_click_request_id"):
                cursor.execute("CREATE INDEX idx_click_request_id ON search_click_logs (request_id);")
            if not index_exists(cursor, "search_click_logs", "idx_click_product_id"):
                cursor.execute("CREATE INDEX idx_click_product_id ON search_click_logs (product_id);")

            # Re-enable foreign key checks
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

            # ── 6. Seeding Roles and Users ──
            logger.info("Seeding user roles...")
            cursor.execute("SELECT id FROM analytics_roles WHERE name = 'ROLE_ANALYTICS';")
            role_row = cursor.fetchone()
            if not role_row:
                cursor.execute(
                    "INSERT INTO analytics_roles (name, description, created_at) VALUES (%s, %s, %s);",
                    ("ROLE_ANALYTICS", "Analytics Operator", datetime.now())
                )
                cursor.execute("SELECT id FROM analytics_roles WHERE name = 'ROLE_ANALYTICS';")
                role_row = cursor.fetchone()
            role_id = role_row["id"]

            if DEFAULT_USERNAME and DEFAULT_EMAIL and DEFAULT_PASSWORD:
                logger.info(f"Seeding default analytics user: {DEFAULT_USERNAME} ({DEFAULT_EMAIL})")
                cursor.execute("SELECT id FROM analytics_users WHERE username = %s OR email = %s;", (DEFAULT_USERNAME, DEFAULT_EMAIL))
                user_row = cursor.fetchone()
                
                pass_hash = hash_password(DEFAULT_PASSWORD)
                now = datetime.now()

                if not user_row:
                    cursor.execute(
                        """
                        INSERT INTO analytics_users (username, email, password_hash, is_active, created_at, updated_at)
                        VALUES (%s, %s, %s, 1, %s, %s);
                        """,
                        (DEFAULT_USERNAME, DEFAULT_EMAIL, pass_hash, now, now)
                    )
                    user_id = cursor.lastrowid
                    logger.info(f"Successfully seeded default user '{DEFAULT_USERNAME}'.")
                else:
                    user_id = user_row["id"]
                    # Update password in case it changed in .env
                    cursor.execute(
                        "UPDATE analytics_users SET password_hash = %s, updated_at = %s WHERE id = %s;",
                        (pass_hash, now, user_id)
                    )
                    logger.info(f"Updated credentials for existing user '{DEFAULT_USERNAME}'.")

                # Associate role
                cursor.execute(
                    "SELECT 1 FROM analytics_user_roles WHERE user_id = %s AND role_id = %s;",
                    (user_id, role_id)
                )
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO analytics_user_roles (user_id, role_id) VALUES (%s, %s);",
                        (user_id, role_id)
                    )
                    logger.info(f"Associated role ROLE_ANALYTICS to user '{DEFAULT_USERNAME}'.")
            else:
                logger.warning("No credentials specified in .env for ANALYTICS_DEFAULT_USERNAME. Skipping user seeding.")

            conn.commit()
            logger.info("Database migration completed successfully!")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error executing database migration: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()
