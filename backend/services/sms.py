import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Tuple

from utils.config import config


class SMSService:
    """Email + FREE SMS via carrier email-to-SMS gateways"""

    GATEWAYS = {
        "airtel": "@airtelmail.in",
        "jio": "@jio.com",
        "vi": "@vodafone.in",
        "vodafone": "@vodafone.in",
        "idea": "@vodafone.in",
        "bsnl": "@sms.bsnl.in",
        "mtnl": "@mtnl.net.in",
        "docomo": "@tatadocomo.com",
        "tata": "@tatadocomo.com",
    }

    def __init__(self):
        self.alert_email = config.ALERT_EMAIL
        self.alert_password = config.ALERT_PASSWORD
        print(f"📧 Sender: {self.alert_email or 'NOT SET'}")
        print(f"📱 Gateways: {', '.join(self.GATEWAYS.keys())}")

    def send_email(self, to_email: str, subject: str, body: str) -> Tuple[bool, str]:
        if not to_email:
            return False, "No recipient"
        if not self.alert_email or not self.alert_password:
            print(f"  📧 [SIMULATED] → {to_email}")
            return True, "Email simulated"

        try:
            msg = MIMEMultipart()
            msg["From"] = f"MADO Emergency <{self.alert_email}>"
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
                server.starttls()
                server.login(self.alert_email, self.alert_password)
                server.send_message(msg)

            print(f"  ✅ Email → {to_email}")
            return True, "Email sent"
        except Exception as e:
            print(f"  ❌ Email: {e}")
            return False, f"Email error: {e}"

    def send_sms(self, phone_number: str, message: str,
                 carrier: str = None) -> Tuple[bool, str]:
        if not phone_number:
            return False, "No phone number"
        if not self.alert_email or not self.alert_password:
            print(f"  📱 [SIMULATED] → {phone_number}")
            return True, "SMS simulated"

        digits = "".join(filter(str.isdigit, phone_number))
        if len(digits) > 10:
            digits = digits[-10:]
        if len(digits) != 10:
            return False, f"Invalid phone: {phone_number}"

        if carrier and carrier.lower() in self.GATEWAYS:
            gateways = [(carrier, self.GATEWAYS[carrier.lower()])]
        else:
            gateways = [
                ("Airtel", "@airtelmail.in"),
                ("Jio", "@jio.com"),
                ("Vi", "@vodafone.in"),
                ("BSNL", "@sms.bsnl.in"),
            ]

        last_err = ""
        for name, domain in gateways:
            sms_email = f"{digits}{domain}"
            try:
                msg = MIMEMultipart()
                msg["From"] = self.alert_email
                msg["To"] = sms_email
                msg["Subject"] = ""
                msg.attach(MIMEText(message[:280], "plain"))

                with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
                    server.starttls()
                    server.login(self.alert_email, self.alert_password)
                    server.send_message(msg)

                print(f"  ✅ SMS via {name} → {sms_email}")
                return True, f"SMS sent via {name}"
            except Exception as e:
                last_err = str(e)
                continue

        print(f"  ❌ SMS failed for {digits}: {last_err}")
        return False, f"SMS failed: {last_err}"

    def send_whatsapp(self, phone_number: str, message: str) -> Tuple[bool, str]:
        return True, "WhatsApp skipped"