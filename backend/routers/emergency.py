from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

from services.emergency import EmergencyService
from services.location import LocationService
from database import get_emergency_alerts, update_alert_status

router = APIRouter()
emergency = EmergencyService()
location = LocationService()


class EmergencyRequest(BaseModel):
    user_id: str
    message: str
    location: Optional[Dict] = None


@router.post("/emergency")
async def trigger(req: EmergencyRequest):
    try:
        loc = req.location or location.get_location()
        resp, aid = emergency.handle_emergency(req.message, loc, req.user_id)
        return {"response": resp, "location": loc, "alert_id": aid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/emergency/alerts/{user_id}")
async def alerts(user_id: str, limit: int = 10):
    if not user_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid user ID")
    return {"alerts": get_emergency_alerts(int(user_id), limit)}


@router.put("/emergency/alert/{alert_id}")
async def update(alert_id: int, status: str):
    update_alert_status(alert_id, status)
    return {"success": True}