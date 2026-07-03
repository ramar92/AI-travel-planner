"""
tools.py
--------
Concrete implementations of the "tools" the agent is allowed to call.
Every tool is a plain Python function that returns a small JSON-serialisable
dict. No API keys are required for any of these — they use free, public
endpoints, so the only key the user needs for the whole app is their Groq key.
"""

import requests

TIMEOUT = 10


def get_weather(city: str, date: str = "") -> dict:
    """Look up a short-range forecast for a city using Open-Meteo (free, no key)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=TIMEOUT,
        ).json()
        if not geo.get("results"):
            return {"error": f"Could not find location '{city}'"}

        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 14,
            },
            timeout=TIMEOUT,
        ).json()

        return {
            "city": loc.get("name"),
            "country": loc.get("country"),
            "daily": forecast.get("daily", {}),
            "note": "Forecast covers the next 14 days only; for dates further out, use seasonal averages instead.",
        }
    except Exception as e:
        return {"error": str(e)}


def get_city_info(city: str) -> dict:
    """Fetch a short encyclopedic summary of a city/place from Wikipedia (free, no key)."""
    try:
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(city)}",
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return {"error": f"No Wikipedia summary found for '{city}'"}
        data = resp.json()
        return {
            "title": data.get("title"),
            "extract": data.get("extract"),
        }
    except Exception as e:
        return {"error": str(e)}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies using frankfurter.app (free, no key, ECB rates)."""
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest",
            params={"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()},
            timeout=TIMEOUT,
        ).json()
        if "rates" not in resp:
            return {"error": resp.get("message", "conversion failed")}
        converted = resp["rates"].get(to_currency.upper())
        return {
            "amount": amount,
            "from": from_currency.upper(),
            "to": to_currency.upper(),
            "converted": converted,
            "date": resp.get("date"),
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_budget(daily_cost: float, num_days: int, num_travelers: int = 1, extra_flat_costs: float = 0.0) -> dict:
    """Simple deterministic budget math, done in Python (not the LLM) so it's always correct."""
    try:
        subtotal = daily_cost * num_days * num_travelers
        total = subtotal + extra_flat_costs
        return {
            "daily_cost_per_person": daily_cost,
            "num_days": num_days,
            "num_travelers": num_travelers,
            "extra_flat_costs": extra_flat_costs,
            "subtotal": round(subtotal, 2),
            "total": round(total, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool registry: JSON-schema declarations Groq's function-calling API expects,
# plus a name -> python-function map the agent uses to actually execute them.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get a 14-day weather forecast (max/min temperature, precipitation chance) for a city. Use this to advise on packing and best days for outdoor activities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Kyoto' or 'Paris'"},
                    "date": {"type": "string", "description": "Optional ISO date the traveler cares about"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_info",
            "description": "Get a short factual summary of a city or place (history, character, notable features) from Wikipedia. Use this to ground the itinerary in real facts about the destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City or place name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount of money from one currency to another using live exchange rates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string", "description": "3-letter ISO code, e.g. USD"},
                    "to_currency": {"type": "string", "description": "3-letter ISO code, e.g. JPY"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_budget",
            "description": "Compute a precise trip budget total from a daily per-person cost, number of days, number of travelers, and any flat extra costs (e.g. flights). Always use this instead of doing the arithmetic yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "daily_cost": {"type": "number", "description": "Estimated daily spend per traveler (lodging+food+local transport+activities), in the trip's budget currency"},
                    "num_days": {"type": "integer"},
                    "num_travelers": {"type": "integer"},
                    "extra_flat_costs": {"type": "number", "description": "One-off costs not multiplied by days, e.g. flights, visas"},
                },
                "required": ["daily_cost", "num_days"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_city_info": get_city_info,
    "convert_currency": convert_currency,
    "calculate_budget": calculate_budget,
}
