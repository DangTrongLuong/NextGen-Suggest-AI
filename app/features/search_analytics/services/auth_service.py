import secrets
import logging
from datetime import timedelta
from fastapi import HTTPException, status
from typing import Dict, Any, Tuple
from app.features.search_analytics.repositories.analytics_repository import analytics_repository
from app.features.search_analytics.auth import auth_handler
from app.shared.redis.service import redis_service

logger = logging.getLogger("analytics_auth_service")

class AuthService:
    def check_user_lockout(self, username_or_email: str):
        """Check if the user account is temporarily locked out due to failed attempts."""
        if not redis_service.is_available():
            return
        
        try:
            lock_key = f"analytics:lock:user:{username_or_email}"
            if redis_service.client.get(lock_key):
                analytics_repository.insert_audit_event_direct(
                    "SUSPICIOUS_LOGIN", username_or_email, "0.0.0.0", "Account currently locked out."
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Tài khoản của bạn đã bị khóa tạm thời trong 15 phút do nhập sai mật khẩu quá 5 lần."
                )
        except HTTPException as he:
            raise he
        except Exception as e:
            redis_service.mark_offline(e)

    def handle_failed_login(self, username_or_email: str, ip_address: str):
        """Increment failed attempts counter and trigger lockout if limit exceeded."""
        if not redis_service.is_available():
            return

        try:
            count_key = f"analytics:failed_login:count:{username_or_email}"
            lock_key = f"analytics:lock:user:{username_or_email}"

            # Increment count
            pipe = redis_service.client.pipeline()
            pipe.incr(count_key)
            pipe.expire(count_key, 900)  # expires in 15 mins
            results = pipe.execute()
            
            failed_count = results[0]
            if failed_count >= 5:
                # Lock user account for 15 mins
                redis_service.client.set(lock_key, "locked", ex=900)
                redis_service.client.delete(count_key)
                logger.warning(f"User account locked out for 15 mins: {username_or_email}")
                analytics_repository.insert_audit_event_direct(
                    "ACCOUNT_LOCKOUT", username_or_email, ip_address, "Account locked out for 15 minutes."
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Nhập sai mật khẩu quá 5 lần. Tài khoản của bạn đã bị khóa trong 15 phút."
                )
        except HTTPException as he:
            raise he
        except Exception as e:
            redis_service.mark_offline(e)

    def reset_failed_login(self, username_or_email: str):
        """Reset failed login count and lockout state."""
        if not redis_service.is_available():
            return
        try:
            count_key = f"analytics:failed_login:count:{username_or_email}"
            redis_service.client.delete(count_key)
        except Exception as e:
            redis_service.mark_offline(e)

    def login(
        self,
        username_or_email: str,
        password: str,
        ip_address: str,
        user_agent: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Authenticate user, issue JWT access token and store refresh token in Redis.
        Fails after 5 failed attempts per account.
        """
        # 1. Check Lockout
        self.check_user_lockout(username_or_email)

        # 2. Query User
        user = analytics_repository.find_analytics_user(username_or_email)
        if not user:
            # Audit log
            analytics_repository.insert_audit_log(
                event_type="LOGIN_FAILED",
                username=username_or_email,
                ip_address=ip_address,
                user_agent=user_agent,
                details="Tài khoản không tồn tại."
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tài khoản hoặc mật khẩu không đúng."
            )

        # 3. Check status
        if not user.get("is_active"):
            analytics_repository.insert_audit_log(
                event_type="LOGIN_FAILED",
                username=username_or_email,
                ip_address=ip_address,
                user_agent=user_agent,
                details="Tài khoản đang bị khóa (is_active = 0)."
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tài khoản đã bị vô hiệu hóa."
            )

        # 4. Verify password
        if not auth_handler.verify_password(password, user["password_hash"]):
            # Audit log and handle failed attempt counter
            analytics_repository.insert_audit_log(
                event_type="LOGIN_FAILED",
                username=username_or_email,
                ip_address=ip_address,
                user_agent=user_agent,
                details="Sai mật khẩu."
            )
            self.handle_failed_login(user["username"], ip_address)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tài khoản hoặc mật khẩu không đúng."
            )

        # 5. Success - create tokens
        access_payload = {
            "sub": user["username"],
            "email": user["email"],
            "role": user["role"]
        }
        access_token = auth_handler.create_access_token(access_payload)
        refresh_token = secrets.token_hex(32)

        # Store Refresh Token in Redis (TTL = 7 days)
        if redis_service.is_available():
            try:
                redis_key = f"analytics:refresh_token:{refresh_token}"
                redis_service.client.set(redis_key, user["username"], ex=7 * 24 * 3600)
            except Exception as e:
                redis_service.mark_offline(e)

        # Reset failed count
        self.reset_failed_login(user["username"])

        # Audit log
        analytics_repository.insert_audit_log(
            event_type="LOGIN_SUCCESS",
            username=user["username"],
            ip_address=ip_address,
            user_agent=user_agent,
            details="Đăng nhập thành công."
        )

        return access_token, refresh_token, {
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }

        if refresh_token and redis_service.is_available():
            try:
                redis_key = f"analytics:refresh_token:{refresh_token}"
                redis_service.client.delete(redis_key)
            except Exception as e:
                redis_service.mark_offline(e)

        analytics_repository.insert_audit_log(
            event_type="LOGOUT",
            username=username,
            ip_address=ip_address,
            details="Đăng xuất thành công."
        )

    def refresh(self, refresh_token: str) -> Tuple[str, str]:
        """Verify refresh token in Redis, rotate refresh token, and return new access/refresh tokens."""
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Không thể làm mới phiên đăng nhập."
            )

        if not redis_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Hệ thống làm mới phiên làm việc tạm thời không hoạt động do mất kết nối Redis."
            )

        try:
            redis_key = f"analytics:refresh_token:{refresh_token}"
            username = redis_service.client.get(redis_key)
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại."
                )
                
            username = username.decode("utf-8") if isinstance(username, bytes) else username

            # Fetch user
            user = analytics_repository.find_analytics_user(username)
            if not user or not user.get("is_active"):
                try:
                    redis_service.client.delete(redis_key)
                except Exception:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Tài khoản không khả dụng."
                )

            # Invalidate old refresh token (rotate refresh tokens)
            redis_service.client.delete(redis_key)

            # Issue new tokens
            access_payload = {
                "sub": user["username"],
                "email": user["email"],
                "role": user["role"]
            }
            new_access_token = auth_handler.create_access_token(access_payload)
            new_refresh_token = secrets.token_hex(32)

            # Save new refresh token in Redis (7 days TTL)
            new_redis_key = f"analytics:refresh_token:{new_refresh_token}"
            redis_service.client.set(new_redis_key, user["username"], ex=7 * 24 * 3600)

            return new_access_token, new_refresh_token
        except HTTPException as he:
            raise he
        except Exception as e:
            redis_service.mark_offline(e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Lỗi kết nối cơ sở dữ liệu làm mới phiên."
            )

# Singleton instance
auth_service = AuthService()
