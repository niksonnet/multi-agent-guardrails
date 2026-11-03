import json

from agents import function_tool

# Mock data so the demo is predictable and needs no external travel API.
FLIGHTS = {
    ("delhi", "paris"): {"airline": "Air France", "price_usd": 650},
    ("delhi", "tokyo"): {"airline": "Japan Airlines", "price_usd": 720},
    ("delhi", "dubai"): {"airline": "Emirates", "price_usd": 250},
    ("mumbai", "london"): {"airline": "British Airways", "price_usd": 600},
}

HOTELS = {
    "paris": [
        {"name": "Hotel Lumiere", "price_per_night_usd": 180},
        {"name": "Le Petit Budget", "price_per_night_usd": 95},
    ],
    "tokyo": [
        {"name": "Shinjuku Grand", "price_per_night_usd": 160},
        {"name": "Capsule Inn Tokyo", "price_per_night_usd": 45},
    ],
    "dubai": [
        {"name": "Marina Palace", "price_per_night_usd": 220},
        {"name": "Desert Rose Inn", "price_per_night_usd": 90},
    ],
    "london": [
        {"name": "The Thames View", "price_per_night_usd": 200},
        {"name": "Camden Hostel", "price_per_night_usd": 60},
    ],
}


@function_tool
def search_flights(origin: str, destination: str) -> str:
    """Search one-way flights between two cities. Returns airline and price in USD."""
    print(f"  [tool] search_flights(origin={origin!r}, destination={destination!r})")
    flight = FLIGHTS.get((origin.lower(), destination.lower()))
    return json.dumps(flight) if flight else "No flights found."


@function_tool
def search_hotels(city: str, nights: int) -> str:
    """Search hotels in a city. Returns each hotel's nightly price and total cost for the stay in USD."""
    print(f"  [tool] search_hotels(city={city!r}, nights={nights})")
    hotels = HOTELS.get(city.lower())
    if not hotels:
        return "No hotels found."
    return json.dumps([{**h, "nights": nights, "total_usd": h["price_per_night_usd"] * nights} for h in hotels])
