import requests
from typing import Dict


class WeatherService:
    def get_weather(self) -> Dict:
        try:
            r = requests.get("https://wttr.in/?format=%l:+%c+%t", timeout=5)
            if r.status_code == 200:
                return {"weather": r.text.strip()}
        except Exception:
            pass
        return {"weather": "Weather unavailable"}