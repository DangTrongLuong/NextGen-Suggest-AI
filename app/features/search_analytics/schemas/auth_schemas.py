from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Any

class LoginRequest(BaseModel):
    username_or_email: str = Field(..., example="analytics_admin")
    password: str = Field(..., example="Analytics@2026!")

class LoginResponseData(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MeResponseData(BaseModel):
    username: str
    email: str
    role: str
    ip_address: str

class StandardResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: Optional[Any] = None

class StandardErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: str
