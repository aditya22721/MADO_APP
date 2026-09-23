from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.contacts import ContactService

router = APIRouter()
contact_service = ContactService()


class ContactRequest(BaseModel):
    user_id: str
    name: str
    phone: str
    email: Optional[str] = None
    carrier: Optional[str] = None
    relationship: str
    is_primary: bool = False


@router.post("/contacts/add")
async def add(req: ContactRequest):
    cid = contact_service.add_contact(
        user_id=req.user_id,
        name=req.name,
        phone=req.phone,
        email=req.email,
        carrier=req.carrier,
        relationship=req.relationship,
        is_primary=req.is_primary,
    )
    if cid == -1:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    return {"success": True, "contact_id": cid}


@router.get("/contacts/{user_id}")
async def get(user_id: str):
    return {"contacts": contact_service.get_user_contacts(user_id)}


@router.delete("/contacts/{contact_id}")
async def delete(contact_id: int):
    ok = contact_service.delete_contact(contact_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"success": True}