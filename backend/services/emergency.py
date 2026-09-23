from typing import Dict, List

from services.knowledge import KnowledgeService
from services.sms import SMSService
from database import save_emergency_alert, get_contacts


class EmergencyService:
    def __init__(self):
        self.knowledge = KnowledgeService()
        self.sms = SMSService()

    def detect_emergency(self, message: str) -> tuple:
        m = message.lower()
        strong = [
            "emergency", "urgent", "help me", "accident", "injured",
            "bleeding heavily", "can't breathe", "not breathing",
            "call ambulance", "need doctor", "heart attack", "chest pain",
            "unconscious", "seizure", "stroke", "help!",
        ]
        for k in strong:
            if k in m:
                return True, "Emergency"

        kws = [
            "heart attack", "chest pain", "difficulty breathing", "choking",
            "bleeding", "burn", "fracture", "broken bone", "seizure",
            "stroke", "head injury", "concussion", "poison",
            "allergic reaction", "anaphylaxis", "drowning",
            "unconscious", "fainting", "electric shock", "eye injury",
        ]
        for k in kws:
            if k in m:
                return True, k.title()
        return False, None

    def handle_emergency(self, message: str, location: Dict, user_id: str) -> tuple:
        etype = self.knowledge.get_emergency_type(message)
        instructions = self.knowledge.get_emergency_instruction(etype)

        response = "🚨 EMERGENCY MODE ACTIVATED 🚨\n\n"
        if location:
            response += f"📍 Location: {location.get('address', 'Unknown')}\n\n"
        response += instructions

        contacts = get_contacts(int(user_id)) if user_id.isdigit() else []
        if contacts:
            response += "\n\n📞 EMERGENCY CONTACTS:\n"
            for c in contacts[:5]:
                response += f"• {c['name']} ({c.get('relationship', '')}): {c['phone']}\n"

        alert_id = save_emergency_alert(
            user_id=int(user_id) if user_id.isdigit() else 1,
            emergency_type=etype,
            message=message,
            location=location,
            status="active",
        )

        if contacts and user_id.isdigit():
            self._send_alerts_to_contacts(message, location, contacts)

        return response, alert_id

    def _send_alerts_to_contacts(self, message: str, location: Dict, contacts: List[Dict]):
        address = (location or {}).get("address", "Unknown")
        lat = (location or {}).get("lat", "")
        lng = (location or {}).get("lng", "")
        maps = f"https://maps.google.com/?q={lat},{lng}"

        sms_body = (
            f"🚨 EMERGENCY: {message[:80]}\n"
            f"📍 {address}\n"
            f"🌐 {maps}"
        )

        email_body = f"""
🚨 EMERGENCY ALERT FROM MADO ASSISTANT 🚨

Emergency: {message}
Type: {self.knowledge.get_emergency_type(message)}

📍 Location: {address}
🌐 Coordinates: {lat}, {lng}
🗺️ Map: {maps}

Please check on this person immediately.

— Sent automatically by MADO Emergency System
"""

        print(f"\n📤 Sending alerts to {len(contacts)} contact(s)...")

        for c in contacts:
            name = c.get("name", "Unknown")
            phone = c.get("phone", "")
            email = c.get("email", "")
            carrier = c.get("carrier")

            print(f"\n── {name} (Carrier: {carrier or 'auto'}) ──")

            if email:
                _, info = self.sms.send_email(
                    email,
                    "🚨 EMERGENCY ALERT from MADO",
                    email_body,
                )
                print(f"   Email: {info}")

            if phone:
                _, info = self.sms.send_sms(phone, sms_body, carrier=carrier)
                print(f"   SMS: {info}")