"""Real travel data from free APIs. Plain Python, no AI.

Every function returns simple lists/dicts. If an API is missing a key or fails,
it falls back to the next source, and finally to the mock data in core/mock_data.py.
"""
import datetime
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from core.mock_data import FLIGHTS, HOTELS

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # live flights + hotels (250 free searches/month)
TRAVELPAYOUTS_TOKEN = os.getenv("TRAVELPAYOUTS_TOKEN")  # optional: cached flight prices
OPENTRIPMAP_KEY = os.getenv("OPENTRIPMAP_KEY")  # optional: attractions

SERPAPI_URL = "https://serpapi.com/search.json"
CACHE_FILE = Path(".cache/api_cache.json")
CACHE_HOURS = 6  # the same search within 6 hours is read from the cache and uses no quota

# The app sets this to a function that draws each API call on the panel. Scripts just print.
on_api_call = None


def default_date():
    """Real prices need a date. If the user gives none, use 30 days from today."""
    return (datetime.date.today() + datetime.timedelta(days=30)).isoformat()


# ---------------------------------------------------------------------------
# Helpers: one GET function with a file cache, and one function to report each call
# ---------------------------------------------------------------------------

def _report(provider, endpoint, params, seconds=0.0, cached=False, error=None):
    event = {"provider": provider, "endpoint": endpoint, "params": params,
             "seconds": seconds, "cached": cached, "error": error}
    if on_api_call:
        on_api_call(event)
    else:
        status = f"ERROR {error}" if error else ("cache" if cached else f"{seconds:.1f}s")
        print(f"  [api] {provider} {endpoint} {params} -> {status}")


def _get(provider, url, params, secret="api_key"):
    """GET a JSON API. Uses the cache when possible. The secret key is never stored or shown."""
    public = {k: v for k, v in params.items() if k not in (secret, "token", "apikey")}
    endpoint = url.split("//")[1].split("?")[0]
    cache_key = url + json.dumps(public, sort_keys=True)

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    hit = cache.get(cache_key)
    if hit and time.time() - hit["at"] < CACHE_HOURS * 3600:
        _report(provider, endpoint, public, cached=True)
        return hit["data"]

    start = time.time()
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            raise ValueError(data["error"])
    except Exception as error:
        _report(provider, endpoint, public, time.time() - start, error=str(error)[:120])
        raise

    _report(provider, endpoint, public, time.time() - start)
    CACHE_FILE.parent.mkdir(exist_ok=True)
    cache[cache_key] = {"at": time.time(), "data": data}
    CACHE_FILE.write_text(json.dumps(cache))
    return data


# ---------------------------------------------------------------------------
# Flights: SerpApi (live) -> Travelpayouts (cached prices) -> mock data
# ---------------------------------------------------------------------------

# For the mock fallback, which uses city names instead of airport codes
IATA_TO_CITY = {"DEL": "delhi", "BOM": "mumbai", "CDG": "paris", "ORY": "paris", "NRT": "tokyo", "HND": "tokyo",
                "DXB": "dubai", "LHR": "london", "LGW": "london"}


def search_flights(origin_code, destination_code, date):
    """Cheapest one-way flights. Codes are IATA airport codes (DEL, DXB). date is YYYY-MM-DD."""
    if SERPAPI_KEY:
        try:
            data = _get("SerpApi · Google Flights", SERPAPI_URL, {
                "engine": "google_flights", "departure_id": origin_code, "arrival_id": destination_code,
                "outbound_date": date, "type": 2, "currency": "USD", "hl": "en", "api_key": SERPAPI_KEY})
            options = (data.get("best_flights") or []) + (data.get("other_flights") or [])
            flights = [{
                "airline": o["flights"][0]["airline"],
                "flight_number": o["flights"][0].get("flight_number", ""),
                "price_usd": o["price"],
                "departure": o["flights"][0]["departure_airport"]["time"],
                "duration_min": o["total_duration"],
                "stops": len(o["flights"]) - 1,
                "source": "SerpApi (live)",
            } for o in options if o.get("price")]
            if flights:
                return sorted(flights, key=lambda f: f["price_usd"])[:5]
        except Exception:
            pass  # already reported; try the next source

    if TRAVELPAYOUTS_TOKEN:
        try:
            data = _get("Travelpayouts · Aviasales", "https://api.travelpayouts.com/aviasales/v3/prices_for_dates", {
                "origin": origin_code, "destination": destination_code, "departure_at": date, "one_way": "true",
                "currency": "usd", "sorting": "price", "limit": 5, "token": TRAVELPAYOUTS_TOKEN}, secret="token")
            flights = [{"airline": d["airline"], "flight_number": str(d.get("flight_number", "")),
                        "price_usd": d["price"], "departure": d["departure_at"], "duration_min": d.get("duration"),
                        "stops": d.get("transfers"), "source": "Travelpayouts (cached, up to 7 days old)"}
                       for d in data.get("data", [])]
            if flights:
                return flights
        except Exception:
            pass

    route = (IATA_TO_CITY.get(origin_code.upper(), ""), IATA_TO_CITY.get(destination_code.upper(), ""))
    flight = FLIGHTS.get(route)
    _report("Mock data", "mock_data.FLIGHTS", {"route": f"{origin_code}->{destination_code}"})
    return [{**flight, "departure": date, "source": "mock data"}] if flight else []


# ---------------------------------------------------------------------------
# Hotels: SerpApi (live) -> mock data
# ---------------------------------------------------------------------------

def search_hotels(city, check_in, nights):
    """Cheapest hotels in a city. check_in is YYYY-MM-DD."""
    if SERPAPI_KEY:
        try:
            check_out = (datetime.date.fromisoformat(check_in) + datetime.timedelta(days=nights)).isoformat()
            data = _get("SerpApi · Google Hotels", SERPAPI_URL, {
                "engine": "google_hotels", "q": f"hotels in {city}", "check_in_date": check_in,
                "check_out_date": check_out, "adults": 1, "currency": "USD", "hl": "en", "sort_by": 3,
                "api_key": SERPAPI_KEY})
            hotels = [{
                "name": p["name"],
                "price_per_night_usd": p["rate_per_night"]["extracted_lowest"],
                "rating": p.get("overall_rating"),
                "nights": nights,
                "source": "SerpApi (live)",
            } for p in data.get("properties", []) if p.get("rate_per_night", {}).get("extracted_lowest")]
            if hotels:
                return sorted(hotels, key=lambda h: h["price_per_night_usd"])[:5]
        except Exception:
            pass

    _report("Mock data", "mock_data.HOTELS", {"city": city})
    return [{**h, "nights": nights, "source": "mock data"} for h in HOTELS.get(city.lower(), [])]


# ---------------------------------------------------------------------------
# Extras: weather (Open-Meteo, no key) and attractions (OpenTripMap, optional key)
# ---------------------------------------------------------------------------

def get_weather(city, date):
    """Forecast for the trip date (Open-Meteo forecasts up to 16 days ahead)."""
    try:
        place = _get("Open-Meteo · Geocoding", "https://geocoding-api.open-meteo.com/v1/search",
                     {"name": city, "count": 1})["results"][0]
        days_ahead = (datetime.date.fromisoformat(date) - datetime.date.today()).days
        if not 0 <= days_ahead < 16:
            return f"No forecast yet: {date} is more than 16 days away."
        daily = _get("Open-Meteo · Forecast", "https://api.open-meteo.com/v1/forecast", {
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 16,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max"})["daily"]
        i = daily["time"].index(date)
        return (f"{daily['temperature_2m_min'][i]:.0f}-{daily['temperature_2m_max'][i]:.0f}°C, "
                f"{daily['precipitation_probability_max'][i]}% chance of rain")
    except Exception:
        return "Weather unavailable."


def get_attractions(city):
    """Top places to visit. Needs OPENTRIPMAP_KEY; without it this feature is switched off."""
    if not OPENTRIPMAP_KEY:
        return []
    try:
        base = "https://api.opentripmap.com/0.1/en/places"
        place = _get("OpenTripMap · Geoname", f"{base}/geoname", {"name": city, "apikey": OPENTRIPMAP_KEY}, "apikey")
        places = _get("OpenTripMap · Places", f"{base}/radius", {
            "radius": 10000, "lat": place["lat"], "lon": place["lon"], "rate": 3, "format": "json", "limit": 8,
            "apikey": OPENTRIPMAP_KEY}, "apikey")
        return [p["name"] for p in places if p.get("name")][:5]
    except Exception:
        return []


def serpapi_searches_left():
    """Free check of the SerpApi quota (this call does not use a search)."""
    if not SERPAPI_KEY:
        return None
    try:
        account = requests.get("https://serpapi.com/account.json", params={"api_key": SERPAPI_KEY}, timeout=15).json()
        return account.get("plan_searches_left")
    except Exception:
        return None


if __name__ == "__main__":
    # Quick test of every API. Searches already made in the last 6 hours come from the cache.
    date = default_date()
    print("Flights DEL->DXB:", json.dumps(search_flights("DEL", "DXB", date)[:2], indent=1))
    print("Hotels Dubai:", json.dumps(search_hotels("Dubai", date, 3)[:2], indent=1))
    soon = (datetime.date.today() + datetime.timedelta(days=5)).isoformat()
    print("Weather Dubai in 5 days:", get_weather("Dubai", soon))
    print("Attractions:", get_attractions("Dubai") or "(no OPENTRIPMAP_KEY)")
    print("SerpApi searches left:", serpapi_searches_left())
