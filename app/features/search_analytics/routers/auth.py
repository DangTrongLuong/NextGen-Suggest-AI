import logging
from datetime import timedelta
from fastapi import APIRouter, Request, Response, Depends, HTTPException, status, Cookie
from typing import Optional

from app.features.search_analytics.schemas.auth_schemas import (
    LoginRequest, LoginResponseData, MeResponseData, StandardResponse, StandardErrorResponse
)
from app.features.search_analytics.services.auth_service import auth_service
from app.features.search_analytics.auth.auth_handler import get_current_analytics_user
from app.shared.redis.service import redis_service

logger = logging.getLogger("analytics_auth_router")

router = APIRouter()

def check_ip_rate_limit(request: Request):
    """Check IP level rate limiting (5 attempts/min/IP). Locks for 15 mins."""
    if not redis_service.is_available():
        return
    ip = request.client.host
    lock_key = f"analytics:lock:login:{ip}"
    limit_key = f"analytics:limit:login:{ip}"

    try:
        # 1. Check if IP locked
        if redis_service.client.get(lock_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Địa chỉ IP của bạn bị khóa tạm thời trong 15 phút do nhập sai hoặc gửi quá nhiều yêu cầu."
            )

        # 2. Increment request count in window
        pipe = redis_service.client.pipeline()
        pipe.incr(limit_key)
        pipe.expire(limit_key, 60, nx=True)
        res = pipe.execute()

        count = res[0]
        if count > 5:
            # Lock IP for 15 minutes (900 seconds)
            redis_service.client.set(lock_key, "locked", ex=900)
            redis_service.client.delete(limit_key)
            logger.warning(f"IP address locked out due to login flooding: {ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Vượt quá số lần thử đăng nhập cho phép (tối đa 5 lần/phút). IP của bạn bị khóa 15 phút."
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        redis_service.mark_offline(e)
        logger.warning(f"Redis rate limiting bypassed due to connection failure: {e}")
        return

@router.post("/login", response_model=StandardResponse, summary="Log in to analytics portal")
async def login(
    request: Request,
    response: Response,
    body: LoginRequest,
    _ = Depends(check_ip_rate_limit)
):
    """
    Authenticate and log in. 
    Sets HttpOnly refresh token cookie and returns JWT access token in the response.
    """
    client_ip = request.client.host
    user_agent = request.headers.get("User-Agent", "unknown")

    try:
        access_token, refresh_token, user_info = auth_service.login(
            username_or_email=body.username_or_email,
            password=body.password,
            ip_address=client_ip,
            user_agent=user_agent
        )

        # Set refresh token as HttpOnly Cookie
        # Secure=True is best, but let's check settings (since we run local HTTP, Secure can sometimes cause issues but is highly requested).
        # We will make it Secure if request is HTTPS, or default to True as requested.
        response.set_cookie(
            key="analytics_refresh_token",
            value=refresh_token,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/analytics/auth",
            max_age=7 * 24 * 3600  # 7 days
        )

        # Add X-Robots-Tag to response
        response.headers["X-Robots-Tag"] = "noindex, nofollow"

        return StandardResponse(
            success=True,
            message="Đăng nhập thành công.",
            data=LoginResponseData(access_token=access_token)
        )
    except HTTPException as he:
        # Standardized error response
        raise he
    except Exception as e:
        logger.error(f"Login unexpected exception: {e}")
        raise HTTPException(
            status_code=500,
            detail="Đăng nhập không thành công do lỗi hệ thống."
        )

@router.post("/logout", response_model=StandardResponse, summary="Log out of analytics portal")
async def logout(
    request: Request,
    response: Response,
    current_user: dict = Depends(get_current_analytics_user),
    analytics_refresh_token: Optional[str] = Cookie(None)
):
    """Invalidate session, clear cookie and log logout event."""
    client_ip = request.client.host
    auth_service.logout(analytics_refresh_token, current_user["username"], client_ip)
    
    # Clear cookie
    response.delete_cookie(
        key="analytics_refresh_token",
        path="/analytics/auth"
    )
    
    response.headers["X-Robots-Tag"] = "noindex, nofollow"

    return StandardResponse(
        success=True,
        message="Đăng xuất thành công."
    )

@router.post("/refresh", response_model=StandardResponse, summary="Rotate access and refresh tokens")
async def refresh(
    response: Response,
    analytics_refresh_token: Optional[str] = Cookie(None)
):
    """Silent refresh to rotate tokens without requesting credentials again."""
    if not analytics_refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên làm việc đã hết hạn. Vui lòng đăng nhập lại."
        )

    try:
        new_access_token, new_refresh_token = auth_service.refresh(analytics_refresh_token)

        # Set new refresh token cookie
        response.set_cookie(
            key="analytics_refresh_token",
            value=new_refresh_token,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/analytics/auth",
            max_age=7 * 24 * 3600  # 7 days
        )

        response.headers["X-Robots-Tag"] = "noindex, nofollow"

        return StandardResponse(
            success=True,
            message="Làm mới token thành công.",
            data=LoginResponseData(access_token=new_access_token)
        )
    except HTTPException as he:
        # Invalidate cookie if refresh failed
        response.delete_cookie(
            key="analytics_refresh_token",
            path="/analytics/auth"
        )
        raise he

@router.get("/me", response_model=StandardResponse, summary="Get authenticated profile info")
async def get_me(
    request: Request,
    response: Response,
    current_user: dict = Depends(get_current_analytics_user)
):
    """Retrieve logged-in user details."""
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return StandardResponse(
        success=True,
        data=MeResponseData(
            username=current_user["username"],
            email=current_user["email"],
            role=current_user["role"],
            ip_address=request.client.host
        )
    )
