from typing import List, Dict

from database import get_contacts, add_contact, delete_contact


class ContactService:
    def __init__(self):
        self.cache = {}

    def get_user_contacts(self, user_id: str) -> List[Dict]:
        if not user_id.isdigit():
            return []
        key = f"c_{user_id}"
        if key in self.cache:
            return self.cache[key]
        contacts = get_contacts(int(user_id))
        self.cache[key] = contacts
        return contacts

    def add_contact(self, user_id, name, phone, relationship,
                    is_primary=False, email=None, carrier=None) -> int:
        if not user_id.isdigit():
            return -1
        cid = add_contact(
            user_id=int(user_id),
            name=name,
            phone=phone,
            relationship=relationship,
            is_primary=is_primary,
            email=email,
            carrier=carrier,
        )
        self.cache.pop(f"c_{user_id}", None)
        return cid

    def delete_contact(self, contact_id: int) -> bool:
        r = delete_contact(contact_id)
        self.cache.clear()
        return r