import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # OpenRouter (AI)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    # Email
    ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
    ALERT_PASSWORD = os.getenv("ALERT_PASSWORD", "")

    # WhatsApp (optional)
    WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
    WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")

    # Weather
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"


config = Config()