from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

from services.knowledge import KnowledgeService
from services.emergency import EmergencyService
from database import save_chat, get_chat_history

router = APIRouter()
knowledge = KnowledgeService()
emergency = EmergencyService()


class ChatRequest(BaseModel):
    user_id: str
    message: str
    location: Optional[Dict] = None


class ChatResponse(BaseModel):
    response: str
    is_emergency: bool


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        is_em, _ = emergency.detect_emergency(req.message)

        if is_em:
            resp, _ = emergency.handle_emergency(req.message, req.location, req.user_id)
            save_chat(
                int(req.user_id) if req.user_id.isdigit() else 1,
                req.message, resp, True, req.location,
            )
            return ChatResponse(response=resp, is_emergency=True)

        resp = knowledge.get_response(req.message)
        save_chat(
            int(req.user_id) if req.user_id.isdigit() else 1,
            req.message, resp, False, req.location,
        )
        return ChatResponse(response=resp, is_emergency=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/history/{user_id}")
async def history(user_id: str, limit: int = 20):
    if not user_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid user ID")
    return {"history": get_chat_history(int(user_id), limit)}