from fastapi import APIRouter, Query

from services.location import LocationService
from services.weather import WeatherService

router = APIRouter()
location = LocationService()
weather = WeatherService()


@router.get("/location")
async def get_location():
    return location.get_location()


@router.get("/weather")
async def get_weather():
    return weather.get_weather()


@router.get("/hospitals")
async def get_hospitals(
    lat: float = Query(...),
    lon: float = Query(...),
):
    return {"hospitals": location.find_nearby_hospitals(lat, lon)}