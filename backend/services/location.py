import requests
from typing import Dict, List


class LocationService:
    def get_location(self) -> Dict:
        try:
            r = requests.get("https://ipapi.co/json/", timeout=5)
            if r.status_code == 200:
                d = r.json()
                return {
                    "lat": d.get("latitude", 0),
                    "lng": d.get("longitude", 0),
                    "city": d.get("city", "Unknown"),
                    "country": d.get("country_name", "Unknown"),
                    "address": f"{d.get('city', '')}, {d.get('country_name', '')}",
                }
        except Exception:
            pass
        return {"lat": 28.6139, "lng": 77.2090, "address": "New Delhi, India"}

    def find_nearby_hospitals(self, lat: float, lng: float) -> List[Dict]:
        try:
            query = f"""
            [out:json];
            node["amenity"="hospital"](around:5000,{lat},{lng});
            out body 5;
            """
            r = requests.post(
                "https://overpass-api.de/api/interpreter",
                data=query,
                timeout=10,
            )
            if r.status_code == 200:
                elements = r.json().get("elements", [])
                return [
                    {
                        "name": e.get("tags", {}).get("name", "Unknown Hospital"),
                        "lat": e.get("lat"),
                        "lng": e.get("lon"),
                    }
                    for e in elements
                ]
        except Exception:
            pass
        return []