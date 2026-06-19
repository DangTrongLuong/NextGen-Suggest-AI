import pymysql
import queue
import threading
import logging
from contextlib import contextmanager
from app.config.settings import settings

logger = logging.getLogger("analytics_db_pool")

class PyMySQLConnectionPool:
    def __init__(self, size=15):
        self.size = size
        self.pool = queue.Queue(maxsize=size)
        self.lock = threading.Lock()
        self.created = 0

    def _create_connection(self):
        return pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            cursorclass=pymysql.cursors.DictCursor
        )

    def get_connection(self):
        try:
            conn = self.pool.get(block=False)
            try:
                conn.ping(reconnect=True)
            except Exception:
                try:
                    conn.close()
                except:
                    pass
                conn = self._create_connection()
            return conn
        except queue.Empty:
            with self.lock:
                if self.created < self.size:
                    conn = self._create_connection()
                    self.created += 1
                    return conn
            # If we reached maximum, block and wait for a connection
            try:
                conn = self.pool.get(block=True, timeout=10) # 10s timeout to prevent infinite blocking
                try:
                    conn.ping(reconnect=True)
                except Exception:
                    try:
                        conn.close()
                    except:
                        pass
                    conn = self._create_connection()
                return conn
            except queue.Empty:
                logger.error("Database connection pool timeout. Creating temporary connection.")
                return self._create_connection()

    def release_connection(self, conn):
        try:
            self.pool.put(conn, block=False)
        except queue.Full:
            try:
                conn.close()
            except:
                pass
            with self.lock:
                self.created -= 1

# Initialize global pool instance
mysql_pool = PyMySQLConnectionPool(size=15)

class ConnectionProxy:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        mysql_pool.release_connection(self._conn)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def get_db_connection():
    conn = mysql_pool.get_connection()
    return ConnectionProxy(conn)
