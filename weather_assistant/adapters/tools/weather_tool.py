"""Weather tool adapter."""

from langchain_core.tools import tool

_WEATHER_DATA = {
    "San Francisco": "Foggy, 62°F",
    "New York": "Sunny, 75°F",
    "London": "Rainy, 55°F",
    "Tokyo": "Clear, 68°F",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return _WEATHER_DATA.get(city, "Weather data not available")

