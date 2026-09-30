import os
import requests
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5"


def get_current_weather(city: str) -> dict | str:
    """
    Fetch current weather for a city directly from OpenWeatherMap REST API.
    Returns a dict on success, or an error string on failure.
    """
    if not OPENWEATHER_API_KEY:
        return "Weather API error: OPENWEATHER_API_KEY is missing in .env"

    try:
        response = requests.get(
            f"{BASE_URL}/weather",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=15
        )
        data = response.json()

        if response.status_code != 200:
            return f"Weather API error: {data.get('message', 'Unknown error')}"

        return {
            "city": data["name"],
            "temperature_c": data["main"]["temp"],
            "feels_like_c": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }

    except requests.exceptions.Timeout:
        return f"Weather API timed out for city: {city}"
    except Exception as e:
        return f"Weather unavailable: {e}"


def get_forecast(city: str) -> dict | str:
    """
    Fetch 5-entry weather forecast for a city directly from OpenWeatherMap REST API.
    Returns a dict on success, or an error string on failure.
    """
    if not OPENWEATHER_API_KEY:
        return "Weather API error: OPENWEATHER_API_KEY is missing in .env"

    try:
        response = requests.get(
            f"{BASE_URL}/forecast",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=15
        )
        data = response.json()

        if response.status_code != 200:
            return f"Forecast API error: {data.get('message', 'Unknown error')}"

        forecast = []
        for item in data["list"][:5]:
            forecast.append({
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "weather": item["weather"][0]["description"]
            })

        return {
            "city": city,
            "forecast": forecast
        }

    except requests.exceptions.Timeout:
        return f"Forecast API timed out for city: {city}"
    except Exception as e:
        return f"Forecast unavailable: {e}"
