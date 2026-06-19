import jwt
import bcrypt
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config.settings import settings
from app.shared.redis.service import redis_service

logger = logging.getLogger("analytics_auth_handler")

# Security schema
security_scheme = HTTPBearer(auto_error=False)

JWT_ALGORITHM = "HS256"
# Use settings or fallback
JWT_SECRET = getattr(settings, "ANALYTICS_JWT_SECRET", "supersecretanalyticskeys12345!")
ALLOWED_IPS = getattr(settings, "ANALYTICS_ALLOWED_IPS", None)

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify standard text password against hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as e:
        logger.error(f"Error checking password: {e}")
        return False

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create signed access token (expires in 8 hours by default)."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=8)
    to_encode.update({"exp": int(expire.timestamp())})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        # Ensure exp is in the future
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            return None
        return payload
    except jwt.PyJWTError:
        return None

def check_ip_whitelist(ip_address: str):
    """Check if the IP is in whitelist if whitelisting is enabled."""
    if ALLOWED_IPS:
        allowed_list = [ip.strip() for ip in ALLOWED_IPS.split(",") if ip.strip()]
        if allowed_list and ip_address not in allowed_list:
            logger.warning(f"IP blocked by whitelist: {ip_address}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="IP của bạn không được phép truy cập Analytics Portal."
            )

async def get_current_analytics_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> Dict[str, Any]:
    """
    Guard dependency to authorize requests to Search Analytics Portal.
    Validates:
    - IP address against optional whitelist.
    - JWT token signature and expiry.
    - User role (ROLE_ANALYTICS).
    """
    # 1. Check IP Whitelist
    client_ip = request.client.host
    check_ip_whitelist(client_ip)

    # 2. Extract Token
    token = None
    if credentials:
        token = credentials.credentials
    else:
        # Fallback: check Authorization header manually or check query param
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    if not token:
        # Fallback check cookie for access token (or only in memory, but let's check headers first)
        token = request.cookies.get("analytics_access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thiếu thông tin xác thực. Vui lòng đăng nhập lại.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Decode & Validate Access Token
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Check Role
    role = payload.get("role")
    if role != "ROLE_ANALYTICS":
        logger.warning(f"Unauthorized role attempt: {role} for user: {payload.get('sub')}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền truy cập module giám sát tìm kiếm."
        )

    return {
        "username": payload.get("sub"),
        "email": payload.get("email"),
        "role": role,
        "token": token
    }
