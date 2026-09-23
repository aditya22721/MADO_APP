from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime

# ===== Request Models =====

class ChatRequest(BaseModel):
    """Chat request model"""
    user_id: str
    message: str
    location: Optional[Dict] = None

class EmergencyRequest(BaseModel):
    """Emergency request model"""
    user_id: str
    message: str
    location: Optional[Dict] = None
    emergency_type: Optional[str] = None

class ContactRequest(BaseModel):
    """Contact request model"""
    user_id: str
    name: str
    phone: str
    relationship: str
    is_primary: bool = False

class UserCreate(BaseModel):
    """User creation model"""
    name: str
    phone: str
    email: Optional[str] = None

# ===== Response Models =====

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    is_emergency: bool

class EmergencyResponse(BaseModel):
    """Emergency response model"""
    response: str
    location: Optional[Dict] = None
    alert_id: Optional[int] = None

class ContactResponse(BaseModel):
    """Contact response model"""
    id: int
    name: str
    phone: str
    relationship: str
    is_primary: bool

class UserResponse(BaseModel):
    """User response model"""
    id: int
    name: str
    phone: str
    email: Optional[str]
    created_at: datetime