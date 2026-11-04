
# 🌍 Agentic AI Example: **Smart Trip Planner Agent**

## The Scenario (Why It's Catchy)

> *You tell the AI: **"Plan a 3-day trip to Tokyo for a foodie on a $2000 budget."***
>
> Instead of giving you a generic wall of text, the agent **autonomously**:
> 1. 🧠 Breaks the goal into sub-tasks
> 2. 🔍 Searches for flights, hotels, restaurants
> 3. 💰 Tracks the budget across all bookings
> 4. 🔄 Re-plans if something exceeds the budget
> 5. 📋 Delivers a final polished itinerary

---

## How It Thinks (The Agentic Loop)

```
        ┌──────────────┐
        │  USER GOAL   │
        │ "Plan Tokyo  │
        │   Trip"      │
        └──────┬───────┘
               ▼
        ┌──────────────┐
   ┌───►│   REASON     │ ◄── "What do I need to do next?"
   │    └──────┬───────┘
   │           ▼
   │    ┌──────────────┐
   │    │    PLAN      │ ◄── "Let me break this into steps"
   │    └──────┬───────┘
   │           ▼
   │    ┌──────────────┐
   │    │    ACT       │ ◄── Uses tools (search, calc, APIs)
   │    └──────┬───────┘
   │           ▼
   │    ┌──────────────┐
   │    │   OBSERVE    │ ◄── "Did it work? Within budget?"
   │    └──────┬───────┘
   │           │
   │     No ┌──┴──┐ Yes
   └────────┤     ├──────► FINAL ITINERARY ✅
            └─────┘
```

---

## Full Python Implementation

```python
import os
import json
from openai import OpenAI

# ─────────────────────────────────────────
# 1. DEFINE THE TOOLS THE AGENT CAN USE
# ─────────────────────────────────────────

def search_flights(origin: str, destination: str, date: str) -> dict:
    """Simulates a flight search API"""
    flights = {
        "Tokyo": [
            {"airline": "ANA", "price": 450, "duration": "14h", "departure": "08:00 AM"},
            {"airline": "JAL", "price": 520, "duration": "13h 30m", "departure": "11:00 AM"},
            {"airline": "United", "price": 380, "duration": "15h", "departure": "06:00 PM"},
        ]
    }
    results = flights.get(destination, [])
    return {"flights": results, "origin": origin, "destination": destination, "date": date}


def search_hotels(city: str, checkin: str, nights: int) -> dict:
    """Simulates a hotel search API"""
    hotels = {
        "Tokyo": [
            {"name": "Shinjuku Granbell Hotel", "price_per_night": 95, "rating": 4.3, "area": "Shinjuku"},
            {"name": "Hotel Gracery Shinjuku", "price_per_night": 130, "rating": 4.5, "area": "Shinjuku"},
            {"name": "Sakura Budget Inn", "price_per_night": 55, "rating": 3.8, "area": "Asakusa"},
        ]
    }
    results = hotels.get(city, [])
    for h in results:
        h["total_cost"] = h["price_per_night"] * nights
    return {"hotels": results, "city": city, "checkin": checkin, "nights": nights}


def search_restaurants(city: str, cuisine_preference: str) -> dict:
    """Simulates a restaurant search API"""
    restaurants = {
        "Tokyo": [
            {"name": "Tsukiji Outer Market", "type": "Sushi & Street Food", "avg_meal_cost": 15, "rating": 4.8},
            {"name": "Ichiran Ramen", "type": "Ramen", "avg_meal_cost": 12, "rating": 4.7},
            {"name": "Gonpachi (Kill Bill restaurant)", "type": "Izakaya", "avg_meal_cost": 35, "rating": 4.4},
            {"name": "Afuri Ramen", "type": "Yuzu Ramen", "avg_meal_cost": 13, "rating": 4.6},
            {"name": "Sukiyabashi Jiro", "type": "Omakase Sushi", "avg_meal_cost": 300, "rating": 4.9},
            {"name": "Omoide Yokocho", "type": "Yakitori Alley", "avg_meal_cost": 10, "rating": 4.5},
        ]
    }
    return {"restaurants": restaurants.get(city, []), "city": city}


def calculate_budget(items: list) -> dict:
    """Calculates total cost and remaining budget"""
    total = sum(item["cost"] for item in items)
    return {"items": items, "total_spent": total}


# ─────────────────────────────────────────
# 2. TOOL REGISTRY (Agent's Toolbox)
# ─────────────────────────────────────────

TOOLS = {
    "search_flights": search_flights,
    "search_hotels": search_hotels,
    "search_restaurants": search_restaurants,
    "calculate_budget": calculate_budget,
}

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights to a destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city"},
                    "destination": {"type": "string", "description": "Arrival city"},
                    "date": {"type": "string", "description": "Travel date"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search for hotels in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "checkin": {"type": "string", "description": "Check-in date"},
                    "nights": {"type": "integer", "description": "Number of nights"},
                },
                "required": ["city", "checkin", "nights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "Search for restaurants in a city based on preference",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "cuisine_preference": {"type": "string", "description": "Food preference"},
                },
                "required": ["city", "cuisine_preference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_budget",
            "description": "Calculate total costs and track budget",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "cost": {"type": "number"},
                            },
                        },
                        "description": "List of cost items",
                    }
                },
                "required": ["items"],
            },
        },
    },
]


# ─────────────────────────────────────────
# 3. THE AGENT (The Brain)
# ─────────────────────────────────────────

class TripPlannerAgent:
    def __init__(self, api_key: str, max_iterations: int = 10):
        self.client = OpenAI(api_key=api_key)
        self.max_iterations = max_iterations
        self.conversation_history = []
        self.iteration = 0

    def _system_prompt(self):
        return """You are a Smart Trip Planner Agent. You autonomously plan trips by:

1. REASONING about what information you need
2. USING TOOLS to search flights, hotels, and restaurants
3. TRACKING THE BUDGET carefully
4. RE-PLANNING if costs exceed the budget (pick cheaper options)
5. DELIVERING a final day-by-day itinerary

RULES:
- Always search for flights FIRST, then hotels, then restaurants
- Always calculate the total budget before giving the final plan
- If over budget, pick cheaper alternatives and recalculate
- Include a meal plan for each day (breakfast, lunch, dinner)
- Be enthusiastic and give insider tips!

When you have all the info and budget is confirmed, present the FINAL ITINERARY."""

    def _process_tool_calls(self, response):
        """Execute tool calls and return results"""
        tool_results = []
        for tool_call in response.choices[0].message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"\n   🔧 Using Tool: {func_name}")
            print(f"      Args: {json.dumps(func_args, indent=2)}")

            # Execute the tool
            result = TOOLS[func_name](**func_args)

            print(f"      ✅ Result received")

            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps(result),
            })
        return tool_results

    def run(self, user_request: str):
        """Main Agentic Loop - Reason → Act → Observe → Repeat"""

        print("=" * 60)
        print("🤖 TRIP PLANNER AGENT ACTIVATED")
        print("=" * 60)
        print(f"\n📝 User Request: {user_request}\n")

        # Initialize conversation
        self.conversation_history = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_request},
        ]

        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n{'─' * 40}")
            print(f"🔄 ITERATION {self.iteration}")
            print(f"{'─' * 40}")

            # ── REASON & DECIDE ──
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history,
                tools=TOOL_DESCRIPTIONS,
                tool_choice="auto",  # Let the agent decide
            )

            message = response.choices[0].message

            # ── CHECK: Does the agent want to use tools? ──
            if message.tool_calls:
                print(f"   🧠 Agent is taking action...")

                # Add assistant's message to history
                self.conversation_history.append(message)

                # ── ACT: Execute the tools ──
                tool_results = self._process_tool_calls(response)

                # ── OBSERVE: Feed results back to the agent ──
                self.conversation_history.extend(tool_results)

            else:
                # ── FINAL OUTPUT: Agent is done ──
                print(f"\n{'=' * 60}")
                print("✅ AGENT COMPLETED - FINAL ITINERARY")
                print(f"{'=' * 60}")
                print(f"\n{message.content}")
                return message.content

        print("⚠️ Max iterations reached.")
        return "I couldn't complete the plan within the allowed steps."


# ─────────────────────────────────────────
# 4. RUN THE AGENT
# ─────────────────────────────────────────

if __name__ == "__main__":
    agent = TripPlannerAgent(
        api_key="your-openai-api-key-here",  # Replace with your key
        max_iterations=10
    )

    # 🚀 Give it a goal and watch it work autonomously!
    agent.run(
        "Plan a 3-day trip to Tokyo for a foodie traveling from New York. "
        "Budget is $2000 total including flights, hotel, and food. "
        "I love ramen, sushi, and street food. Find me the best experience!"
    )
```

---

## What Happens When You Run It

```
============================================================
🤖 TRIP PLANNER AGENT ACTIVATED
============================================================

📝 User Request: Plan a 3-day trip to Tokyo for a foodie...

────────────────────────────────────
🔄 ITERATION 1
────────────────────────────────────
   🧠 Agent is taking action...
   🔧 Using Tool: search_flights
      Args: {"origin": "New York", "destination": "Tokyo", "date": "2025-03-01"}
      ✅ Result received

────────────────────────────────────
🔄 ITERATION 2
────────────────────────────────────
   🧠 Agent is taking action...
   🔧 Using Tool: search_hotels
      Args: {"city": "Tokyo", "checkin": "2025-03-01", "nights": 3}
      ✅ Result received

────────────────────────────────────
🔄 ITERATION 3
────────────────────────────────────
   🧠 Agent is taking action...
   🔧 Using Tool: search_restaurants
      Args: {"city": "Tokyo", "cuisine_preference": "ramen, sushi, street food"}
      ✅ Result received

────────────────────────────────────
🔄 ITERATION 4
────────────────────────────────────
   🧠 Agent is taking action...
   🔧 Using Tool: calculate_budget
      Args: {"items": [
          {"description": "United Flight", "cost": 380},
          {"description": "Sakura Budget Inn (3 nights)", "cost": 165},
          {"description": "Food (9 meals)", "cost": 200}
      ]}
      ✅ Budget: $745 / $2000 ✅ UNDER BUDGET!

============================================================
✅ AGENT COMPLETED - FINAL ITINERARY
============================================================

🗼 YOUR 3-DAY TOKYO FOODIE ADVENTURE 🍣

✈️ Flight: United Airlines — $380 (Departs 6:00 PM)
🏨 Hotel: Sakura Budget Inn, Asakusa — $165 (3 nights)

📅 DAY 1: Tsukiji & Ramen Heaven
   🌅 Morning — Tsukiji Outer Market (fresh sushi!) — ~$15
   🍜 Lunch — Ichiran Ramen (solo booths!) — ~$12
   🌙 Evening — Omoide Yokocho (Yakitori Alley) — ~$10

📅 DAY 2: Culture & Fine Dining
   🌅 Morning — Senso-ji Temple + street food — ~$10
   🍜 Lunch — Afuri Ramen (yuzu broth!) — ~$13
   🌙 Evening — Gonpachi (the Kill Bill restaurant!) — ~$35

📅 DAY 3: Last Bites
   🌅 Morning — Convenience store breakfast (7-Eleven is amazing!)
   🍜 Lunch — Tsukiji Round 2 — ~$15
   🌙 Evening — Departure

💰 TOTAL: $745 | 🎉 Remaining: $1,255 for shopping & activities!
```

---

## Why This Is "Agentic"

| Agentic Trait | How It Shows Up |
|---|---|
| **🧠 Reasoning** | Decides *which* tool to call and *when* |
| **📋 Planning** | Searches flights → hotels → food (logical order) |
| **🔧 Tool Use** | Calls 4 different tools autonomously |
| **🔄 Iteration** | Loops until the plan is complete |
| **💰 Self-Correction** | If over budget, picks cheaper options & recalculates |
| **🎯 Goal-Oriented** | Doesn't stop until a complete itinerary is delivered |

---

## Want to Make It Even More Agentic?

```
Add these enhancements:

🔹 Real APIs       → Plug in Skyscanner, Booking.com, Yelp APIs
🔹 Memory          → Remember user preferences across trips
🔹 Multi-Agent     → Separate Flight Agent, Hotel Agent, Food Agent
🔹 Human-in-Loop   → "I found 3 hotels. Which do you prefer?"
🔹 Guardrails      → Never book anything without user confirmation
```

---

This example shows the **core agentic pattern**: the AI doesn't just answer — it **thinks, plans, acts, observes, and adapts** until the goal is achieved. That's the fundamental difference from a simple chatbot!


__________________________________________________________________________________________________________________________________________________________________________________________________________________

# 🚀 Enhanced Agentic Trip Planner — Full Implementation

---

## Architecture Overview

```
                         ┌─────────────────────┐
                         │     👤 USER          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  🛡️ GUARDRAILS      │
                         │  (Safety & Budget)   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   🧠 ORCHESTRATOR AGENT        │
                    │   (Coordinator / Planner)      │
                    │                                │
                    │   ┌─────────┐ ┌────────────┐  │
                    │   │ 🧠 LLM  │ │ 📂 MEMORY  │  │
                    │   └─────────┘ └────────────┘  │
                    └───┬──────────┬────────────┬───┘
                        │          │            │
               ┌────────▼──┐  ┌───▼────────┐ ┌─▼──────────┐
               │ ✈️ FLIGHT  │  │ 🏨 HOTEL   │ │ 🍣 FOOD    │
               │   AGENT   │  │   AGENT    │ │   AGENT    │
               └────┬──────┘  └────┬───────┘ └──┬─────────┘
                    │              │             │
               ┌────▼──────┐  ┌───▼────────┐ ┌──▼─────────┐
               │ Skyscanner│  │ Booking.com│ │  Yelp API  │
               │    API    │  │    API     │ │            │
               └───────────┘  └────────────┘ └────────────┘
                    │              │             │
                    └──────────┬───┘─────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ 🙋 HUMAN-IN-THE-LOOP │
                    │ "Do you approve?"     │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ 📋 FINAL ITINERARY   │
                    └──────────────────────┘
```

---

## Complete Enhanced Implementation

```python
import os
import json
import sqlite3
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional
from openai import OpenAI
import requests
from abc import ABC, abstractmethod


# ══════════════════════════════════════════════════════════════
# ENHANCEMENT 1: REAL API INTEGRATIONS
# ══════════════════════════════════════════════════════════════

class APIProvider(ABC):
    """Abstract base class for all API providers"""

    @abstractmethod
    def search(self, **kwargs) -> dict:
        pass

    def _handle_error(self, error: Exception, fallback_data: dict) -> dict:
        """Graceful degradation — fall back to simulated data"""
        print(f"   ⚠️  API Error: {error}. Using simulated data.")
        return fallback_data


# ─────────────────────────────────────────
# 1A. SKYSCANNER FLIGHT API
# ─────────────────────────────────────────

class SkyscannerAPI(APIProvider):
    """
    Real Skyscanner API Integration
    Sign up: https://developers.skyscanner.net/
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SKYSCANNER_API_KEY")
        self.base_url = "https://partners.api.skyscanner.net/apiservices/v3"

    def search(self, origin: str, destination: str, date: str,
               max_results: int = 5) -> dict:
        """Search for real flights via Skyscanner"""

        if not self.api_key:
            return self._simulate_flights(origin, destination, date)

        try:
            # ── Skyscanner API: Create Search Session ──
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }

            # Step 1: Get location IDs
            origin_id = self._get_place_id(origin, headers)
            dest_id = self._get_place_id(destination, headers)

            # Step 2: Create flight search
            search_payload = {
                "query": {
                    "market": "US",
                    "locale": "en-US",
                    "currency": "USD",
                    "queryLegs": [
                        {
                            "originPlaceId": {"iata": origin_id},
                            "destinationPlaceId": {"iata": dest_id},
                            "date": {
                                "year": int(date.split("-")[0]),
                                "month": int(date.split("-")[1]),
                                "day": int(date.split("-")[2])
                            }
                        }
                    ],
                    "adults": 1
                }
            }

            response = requests.post(
                f"{self.base_url}/flights/live/search/create",
                headers=headers,
                json=search_payload,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            # Step 3: Parse results
            flights = []
            for itinerary in data.get("content", {}).get(
                    "results", {}).get("itineraries", {}).values():
                leg = itinerary.get("legs", [{}])[0]
                flights.append({
                    "airline": leg.get("carriers", [{}])[0].get("name", "Unknown"),
                    "price": itinerary.get("price", {}).get("raw", 0),
                    "duration": leg.get("duration", "N/A"),
                    "departure": leg.get("departure", "N/A"),
                    "stops": leg.get("stopCount", 0),
                    "source": "skyscanner_live"
                })

            flights.sort(key=lambda x: x["price"])
            return {
                "flights": flights[:max_results],
                "origin": origin,
                "destination": destination,
                "date": date,
                "source": "Skyscanner API"
            }

        except Exception as e:
            return self._handle_error(
                e, self._simulate_flights(origin, destination, date)
            )

    def _get_place_id(self, city: str, headers: dict) -> str:
        """Resolve city name to IATA code"""
        city_to_iata = {
            "New York": "JFK", "Tokyo": "NRT", "London": "LHR",
            "Paris": "CDG", "Los Angeles": "LAX", "Bangkok": "BKK",
            "Dubai": "DXB", "Singapore": "SIN", "Rome": "FCO",
            "Sydney": "SYD", "Seoul": "ICN", "Mumbai": "BOM",
        }
        if city in city_to_iata:
            return city_to_iata[city]

        # Fallback: Use Skyscanner autosuggest
        try:
            response = requests.get(
                f"{self.base_url}/autosuggest/flights",
                headers=headers,
                params={"query": city},
                timeout=10
            )
            places = response.json().get("places", [])
            if places:
                return places[0].get("iata", city[:3].upper())
        except Exception:
            pass
        return city[:3].upper()

    def _simulate_flights(self, origin: str, destination: str,
                          date: str) -> dict:
        """Fallback simulated data"""
        import random
        airlines = [
            ("ANA", 420, "14h 10m"), ("JAL", 510, "13h 45m"),
            ("United", 370, "15h 20m"), ("Delta", 445, "14h 50m"),
            ("Korean Air", 390, "16h 30m"),
        ]
        flights = []
        for airline, base_price, duration in airlines:
            price = base_price + random.randint(-50, 80)
            flights.append({
                "airline": airline, "price": price,
                "duration": duration,
                "departure": f"{random.randint(6,20):02d}:00",
                "stops": random.choice([0, 0, 1]),
                "source": "simulated"
            })
        flights.sort(key=lambda x: x["price"])
        return {
            "flights": flights, "origin": origin,
            "destination": destination, "date": date,
            "source": "Simulated (no API key)"
        }


# ─────────────────────────────────────────
# 1B. BOOKING.COM HOTEL API
# ─────────────────────────────────────────

class BookingComAPI(APIProvider):
    """
    Real Booking.com API via RapidAPI
    Sign up: https://rapidapi.com/apidojo/api/booking-com/
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("RAPIDAPI_KEY")
        self.base_url = "https://booking-com.p.rapidapi.com/v1"

    def search(self, city: str, checkin: str, nights: int,
               max_results: int = 5, currency: str = "USD") -> dict:
        """Search for real hotels via Booking.com"""

        checkout = (
            datetime.strptime(checkin, "%Y-%m-%d") + timedelta(days=nights)
        ).strftime("%Y-%m-%d")

        if not self.api_key:
            return self._simulate_hotels(city, checkin, nights)

        try:
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
            }

            # Step 1: Get destination ID
            dest_response = requests.get(
                f"{self.base_url}/hotels/locations",
                headers=headers,
                params={"name": city, "locale": "en-us"},
                timeout=10
            )
            dest_response.raise_for_status()
            locations = dest_response.json()

            if not locations:
                raise ValueError(f"No location found for {city}")

            dest_id = locations[0]["dest_id"]
            dest_type = locations[0]["dest_type"]

            # Step 2: Search hotels
            search_response = requests.get(
                f"{self.base_url}/hotels/search",
                headers=headers,
                params={
                    "dest_id": dest_id,
                    "dest_type": dest_type,
                    "checkin_date": checkin,
                    "checkout_date": checkout,
                    "adults_number": 1,
                    "room_number": 1,
                    "units": "metric",
                    "order_by": "popularity",
                    "currency": currency,
                    "locale": "en-us",
                    "filter_by_currency": currency,
                    "page_number": 0,
                },
                timeout=15
            )
            search_response.raise_for_status()
            data = search_response.json()

            # Step 3: Parse results
            hotels = []
            for hotel in data.get("result", [])[:max_results]:
                price_total = hotel.get("min_total_price", 0)
                hotels.append({
                    "name": hotel.get("hotel_name", "Unknown"),
                    "price_per_night": round(price_total / max(nights, 1), 2),
                    "total_cost": round(price_total, 2),
                    "rating": hotel.get("review_score", 0) / 2,
                    "area": hotel.get("district", "City Center"),
                    "review_count": hotel.get("review_nr", 0),
                    "stars": hotel.get("class", 0),
                    "photo_url": hotel.get("main_photo_url", ""),
                    "source": "booking.com_live"
                })

            return {
                "hotels": hotels, "city": city,
                "checkin": checkin, "nights": nights,
                "source": "Booking.com API"
            }

        except Exception as e:
            return self._handle_error(
                e, self._simulate_hotels(city, checkin, nights)
            )

    def _simulate_hotels(self, city: str, checkin: str, nights: int) -> dict:
        """Fallback simulated data"""
        import random
        hotel_templates = [
            ("Grand Hyatt", 180, 4.6, "City Center", 4),
            ("Shinjuku Granbell", 95, 4.3, "Shinjuku", 3),
            ("Hotel Gracery", 130, 4.5, "Shinjuku", 4),
            ("Sakura Budget Inn", 55, 3.8, "Asakusa", 2),
            ("Park Hotel Tokyo", 160, 4.4, "Shiodome", 4),
            ("Hostel Chapter Two", 35, 4.1, "Asakusa", 1),
        ]
        hotels = []
        for name, base_price, rating, area, stars in hotel_templates:
            price = base_price + random.randint(-15, 20)
            hotels.append({
                "name": name, "price_per_night": price,
                "total_cost": price * nights,
                "rating": rating, "area": area,
                "stars": stars, "source": "simulated"
            })
        return {
            "hotels": hotels, "city": city,
            "checkin": checkin, "nights": nights,
            "source": "Simulated (no API key)"
        }


# ─────────────────────────────────────────
# 1C. YELP RESTAURANT API
# ─────────────────────────────────────────

class YelpAPI(APIProvider):
    """
    Real Yelp Fusion API Integration
    Sign up: https://www.yelp.com/developers/v3/manage_app
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("YELP_API_KEY")
        self.base_url = "https://api.yelp.com/v3"

    def search(self, city: str, cuisine_preference: str = "food",
               max_results: int = 8) -> dict:
        """Search for real restaurants via Yelp"""

        if not self.api_key:
            return self._simulate_restaurants(city, cuisine_preference)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(
                f"{self.base_url}/businesses/search",
                headers=headers,
                params={
                    "location": city,
                    "term": cuisine_preference,
                    "categories": "restaurants,food",
                    "sort_by": "rating",
                    "limit": max_results,
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            restaurants = []
            for biz in data.get("businesses", []):
                # Estimate meal cost from price level
                price_level = len(biz.get("price", "$$"))
                avg_cost_map = {1: 10, 2: 25, 3: 50, 4: 100}
                avg_meal_cost = avg_cost_map.get(price_level, 25)

                restaurants.append({
                    "name": biz.get("name", "Unknown"),
                    "type": ", ".join(
                        [c["title"] for c in biz.get("categories", [])]
                    ),
                    "avg_meal_cost": avg_meal_cost,
                    "rating": biz.get("rating", 0),
                    "review_count": biz.get("review_count", 0),
                    "address": ", ".join(
                        biz.get("location", {}).get("display_address", [])
                    ),
                    "phone": biz.get("phone", ""),
                    "image_url": biz.get("image_url", ""),
                    "yelp_url": biz.get("url", ""),
                    "source": "yelp_live"
                })

            return {
                "restaurants": restaurants, "city": city,
                "cuisine": cuisine_preference,
                "source": "Yelp API"
            }

        except Exception as e:
            return self._handle_error(
                e, self._simulate_restaurants(city, cuisine_preference)
            )

    def _simulate_restaurants(self, city: str,
                              cuisine_preference: str) -> dict:
        """Fallback simulated data"""
        restaurants = [
            {"name": "Tsukiji Outer Market", "type": "Sushi & Street Food",
             "avg_meal_cost": 15, "rating": 4.8, "source": "simulated"},
            {"name": "Ichiran Ramen", "type": "Ramen",
             "avg_meal_cost": 12, "rating": 4.7, "source": "simulated"},
            {"name": "Gonpachi", "type": "Izakaya",
             "avg_meal_cost": 35, "rating": 4.4, "source": "simulated"},
            {"name": "Afuri Ramen", "type": "Yuzu Ramen",
             "avg_meal_cost": 13, "rating": 4.6, "source": "simulated"},
            {"name": "Omoide Yokocho", "type": "Yakitori",
             "avg_meal_cost": 10, "rating": 4.5, "source": "simulated"},
            {"name": "Gyukatsu Motomura", "type": "Beef Cutlet",
             "avg_meal_cost": 18, "rating": 4.6, "source": "simulated"},
            {"name": "Nakiryu", "type": "Michelin Ramen",
             "avg_meal_cost": 14, "rating": 4.9, "source": "simulated"},
            {"name": "Tonkatsu Maisen", "type": "Tonkatsu",
             "avg_meal_cost": 16, "rating": 4.5, "source": "simulated"},
        ]
        return {
            "restaurants": restaurants, "city": city,
            "cuisine": cuisine_preference,
            "source": "Simulated (no API key)"
        }


# ══════════════════════════════════════════════════════════════
# ENHANCEMENT 2: PERSISTENT MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════

class AgentMemory:
    """
    SQLite-based persistent memory for the agent.
    Remembers user preferences, past trips, and learnings.
    """

    def __init__(self, db_path: str = "trip_agent_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Initialize the memory database"""
        cursor = self.conn.cursor()

        # User preferences (long-term memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT,
                preference_key TEXT,
                preference_value TEXT,
                confidence REAL DEFAULT 1.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, preference_key)
            )
        """)

        # Trip history (episodic memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trip_history (
                trip_id TEXT PRIMARY KEY,
                user_id TEXT,
                destination TEXT,
                budget REAL,
                actual_cost REAL,
                duration_days INTEGER,
                satisfaction_rating INTEGER,
                trip_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Interaction logs (working memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Learned facts (semantic memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_facts (
                fact_id TEXT PRIMARY KEY,
                category TEXT,
                fact TEXT,
                source TEXT,
                reliability REAL DEFAULT 0.8,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    # ── User Preferences ──

    def save_preference(self, user_id: str, key: str, value: str,
                        confidence: float = 1.0):
        """Save or update a user preference"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_preferences
            (user_id, preference_key, preference_value, confidence, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (user_id, key, value, confidence))
        self.conn.commit()
        print(f"   💾 Memory saved: {key} = {value}")

    def get_preference(self, user_id: str, key: str) -> Optional[str]:
        """Retrieve a specific preference"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT preference_value FROM user_preferences
            WHERE user_id = ? AND preference_key = ?
        """, (user_id, key))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_all_preferences(self, user_id: str) -> dict:
        """Get all preferences for a user"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT preference_key, preference_value, confidence
            FROM user_preferences WHERE user_id = ?
            ORDER BY updated_at DESC
        """, (user_id,))
        return {
            row[0]: {"value": row[1], "confidence": row[2]}
            for row in cursor.fetchall()
        }

    # ── Trip History ──

    def save_trip(self, user_id: str, trip_data: dict):
        """Save a completed trip to history"""
        trip_id = hashlib.md5(
            f"{user_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trip_history
            (trip_id, user_id, destination, budget, actual_cost,
             duration_days, trip_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            trip_id, user_id,
            trip_data.get("destination", ""),
            trip_data.get("budget", 0),
            trip_data.get("actual_cost", 0),
            trip_data.get("duration_days", 0),
            json.dumps(trip_data)
        ))
        self.conn.commit()
        print(f"   💾 Trip saved to history: {trip_id}")
        return trip_id

    def get_trip_history(self, user_id: str, limit: int = 5) -> list:
        """Retrieve past trips"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT trip_id, destination, budget, actual_cost,
                   duration_days, created_at, trip_data
            FROM trip_history WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit))

        return [
            {
                "trip_id": row[0], "destination": row[1],
                "budget": row[2], "actual_cost": row[3],
                "duration_days": row[4], "date": row[5],
                "details": json.loads(row[6]) if row[6] else {}
            }
            for row in cursor.fetchall()
        ]

    # ── Context Builder ──

    def build_context(self, user_id: str) -> str:
        """Build a context string from memory for the LLM"""
        preferences = self.get_all_preferences(user_id)
        history = self.get_trip_history(user_id, limit=3)

        context_parts = []

        if preferences:
            context_parts.append("## Known User Preferences:")
            for key, data in preferences.items():
                context_parts.append(
                    f"- {key}: {data['value']} "
                    f"(confidence: {data['confidence']})"
                )

        if history:
            context_parts.append("\n## Recent Trip History:")
            for trip in history:
                context_parts.append(
                    f"- {trip['destination']} "
                    f"({trip['duration_days']} days, "
                    f"${trip['actual_cost']}/{trip['budget']} budget) "
                    f"on {trip['date']}"
                )

        return "\n".join(context_parts) if context_parts else \
            "No prior data. This is a new user."

    def close(self):
        """Close database connection"""
        self.conn.close()


# ══════════════════════════════════════════════════════════════
# ENHANCEMENT 5: GUARDRAILS SYSTEM
# ══════════════════════════════════════════════════════════════

class Guardrails:
    """
    Safety and validation layer for the agent.
    Prevents unauthorized actions and enforces policies.
    """

    def __init__(self, max_budget: float = 50000,
                 max_trip_days: int = 30):
        self.max_budget = max_budget
        self.max_trip_days = max_trip_days
        self.action_log = []
        self.blocked_actions = []

    def validate_budget(self, budget: float) -> dict:
        """Validate the trip budget"""
        if budget <= 0:
            return {"valid": False,
                    "reason": "Budget must be positive"}
        if budget > self.max_budget:
            return {"valid": False,
                    "reason": f"Budget exceeds maximum "
                              f"${self.max_budget:,.0f}"}
        return {"valid": True, "budget": budget}

    def validate_booking(self, booking_type: str, details: dict,
                         budget_remaining: float) -> dict:
        """Validate a booking before confirmation"""
        cost = details.get("cost", details.get("price", 0))

        # Check: Does it exceed remaining budget?
        if cost > budget_remaining:
            return {
                "approved": False,
                "reason": f"Cost ${cost:.2f} exceeds remaining "
                          f"budget ${budget_remaining:.2f}",
                "suggestion": "Look for cheaper alternatives"
            }

        # Check: Unreasonably expensive?
        thresholds = {
            "flight": 5000, "hotel_per_night": 1500,
            "restaurant": 500
        }
        threshold = thresholds.get(booking_type, 2000)
        if cost > threshold:
            return {
                "approved": False,
                "reason": f"{booking_type} cost ${cost:.2f} seems "
                          f"unusually high (threshold: ${threshold})",
                "requires_confirmation": True
            }

        self.action_log.append({
            "type": booking_type, "cost": cost,
            "timestamp": datetime.now().isoformat(),
            "status": "approved"
        })

        return {"approved": True, "cost": cost}

    def validate_destination(self, destination: str) -> dict:
        """Check if destination is valid and safe"""
        # In production, integrate with travel advisory APIs
        restricted = ["North Korea", "Syria", "Afghanistan"]
        if destination in restricted:
            return {
                "valid": False,
                "reason": f"Travel to {destination} is restricted. "
                          f"Please check official travel advisories."
            }
        return {"valid": True, "destination": destination}

    def pre_action_check(self, action: str, params: dict) -> dict:
        """General pre-action safety check"""
        checks_passed = []

        # No empty critical parameters
        required_fields = {
            "book_flight": ["airline", "price"],
            "book_hotel": ["name", "total_cost"],
            "book_restaurant": ["name"],
        }

        if action in required_fields:
            for field in required_fields[action]:
                if field not in params or not params[field]:
                    return {
                        "approved": False,
                        "reason": f"Missing required field: {field}"
                    }
                checks_passed.append(f"✓ {field} present")

        return {
            "approved": True,
            "checks": checks_passed,
            "action": action
        }

    def get_audit_log(self) -> list:
        """Return the complete action audit log"""
        return self.action_log


# ══════════════════════════════════════════════════════════════
# ENHANCEMENT 4: HUMAN-IN-THE-LOOP SYSTEM
# ══════════════════════════════════════════════════════════════

class HumanInTheLoop:
    """
    Interactive approval system.
    Presents choices and gets user confirmation.
    """

    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve
        self.decisions = []

    def present_options(self, category: str, options: list,
                        budget_remaining: float) -> dict:
        """Present options to user and get their choice"""

        print(f"\n{'═' * 55}")
        print(f"🙋 YOUR INPUT NEEDED — {category.upper()}")
        print(f"═" * 55)
        print(f"💰 Remaining Budget: ${budget_remaining:,.2f}\n")

        for i, option in enumerate(options, 1):
            self._display_option(i, category, option)

        if self.auto_approve:
            choice = 1
            print(f"\n🤖 Auto-selected: Option 1")
        else:
            choice = self._get_user_choice(len(options))

        selected = options[choice - 1]
        self.decisions.append({
            "category": category,
            "selected": selected,
            "timestamp": datetime.now().isoformat()
        })

        print(f"\n✅ Selected: {selected.get('name', selected.get('airline', 'Option'))} ")
        return {"choice": choice, "selected": selected}

    def confirm_booking(self, booking_summary: dict) -> bool:
        """Get final confirmation before booking"""

        print(f"\n{'═' * 55}")
        print(f"🛡️  BOOKING CONFIRMATION REQUIRED")
        print(f"{'═' * 55}")

        print(f"\n📋 Booking Summary:")
        for key, value in booking_summary.items():
            if key != "raw_data":
                print(f"   • {key}: {value}")

        if self.auto_approve:
            print(f"\n🤖 Auto-approved")
            return True

        while True:
            confirm = input(
                "\n👉 Confirm this booking? (yes/no/modify): "
            ).strip().lower()
            if confirm in ["yes", "y"]:
                print("✅ Booking confirmed!")
                return True
            elif confirm in ["no", "n"]:
                print("❌ Booking cancelled.")
                return False
            elif confirm in ["modify", "m"]:
                print("🔄 Let's look at other options...")
                return False
            else:
                print("Please enter 'yes', 'no', or 'modify'")

    def confirm_final_itinerary(self, itinerary: str) -> bool:
        """Get approval on the complete itinerary"""

        print(f"\n{'═' * 55}")
        print(f"📋 FINAL ITINERARY REVIEW")
        print(f"{'═' * 55}")
        print(itinerary)

        if self.auto_approve:
            print(f"\n🤖 Auto-approved")
            return True

        while True:
            confirm = input(
                "\n👉 Approve this itinerary? (yes/no): "
            ).strip().lower()
            if confirm in ["yes", "y"]:
                return True
            elif confirm in ["no", "n"]:
                return False

    def _display_option(self, index: int, category: str, option: dict):
        """Format and display a single option"""
        if category == "flights":
            print(f"   [{index}] ✈️  {option.get('airline', 'N/A')}")
            print(f"       💲 ${option.get('price', 'N/A')} | "
                  f"⏱️  {option.get('duration', 'N/A')} | "
                  f"🔄 {option.get('stops', 0)} stops")
        elif category == "hotels":
            print(f"   [{index}] 🏨 {option.get('name', 'N/A')}")
            print(f"       💲 ${option.get('price_per_night', 'N/A')}/night "
                  f"(${option.get('total_cost', 'N/A')} total) | "
                  f"⭐ {option.get('rating', 'N/A')} | "
                  f"📍 {option.get('area', 'N/A')}")
        elif category == "restaurants":
            print(f"   [{index}] 🍽️  {option.get('name', 'N/A')}")
            print(f"       🍳 {option.get('type', 'N/A')} | "
                  f"💲 ~${option.get('avg_meal_cost', 'N/A')}/meal | "
                  f"⭐ {option.get('rating', 'N/A')}")
        print()

    def _get_user_choice(self, max_options: int) -> int:
        """Get validated user input"""
        while True:
            try:
                choice = int(input(
                    f"👉 Enter your choice (1-{max_options}): "
                ))
                if 1 <= choice <= max_options:
                    return choice
                print(f"   Please enter a number between "
                      f"1 and {max_options}")
            except ValueError:
                print("   Please enter a valid number")


# ══════════════════════════════════════════════════════════════
# ENHANCEMENT 3: MULTI-AGENT SYSTEM
# ══════════════════════════════════════════════════════════════

class SpecializedAgent:
    """Base class for all specialized agents"""

    def __init__(self, name: str, role: str, client: OpenAI,
                 model: str = "gpt-4o-mini"):
        self.name = name
        self.role = role
        self.client = client
        self.model = model

    def think(self, prompt: str, context: str = "") -> str:
        """Let the specialized agent reason about its domain"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nTask:\n{prompt}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content


class FlightAgent(SpecializedAgent):
    """Specialized agent for flight search and selection"""

    def __init__(self, client: OpenAI, api: SkyscannerAPI):
        super().__init__(
            name="✈️ Flight Agent",
            role="""You are a Flight Search Specialist. You:
- Find the best flights balancing price, duration, and comfort
- Consider layovers and departure times
- Recommend based on the traveler's priorities
- Always present the top 3-5 options with clear pros/cons
Format your analysis clearly with rankings.""",
            client=client
        )
        self.api = api

    def search_and_analyze(self, origin: str, destination: str,
                           date: str, budget: float,
                           preferences: dict) -> dict:
        """Search flights and provide intelligent analysis"""
        print(f"\n   {self.name}: Searching flights...")

        # Get real/simulated flight data
        results = self.api.search(origin, destination, date)
        flights = results["flights"]

        # Filter by budget (max 60% of total budget for flights)
        max_flight_budget = budget * 0.6
        affordable = [f for f in flights if f["price"] <= max_flight_budget]

        if not affordable:
            affordable = sorted(flights, key=lambda x: x["price"])[:3]

        # Let the agent analyze the options
        analysis = self.think(
            f"Analyze these flight options for a traveler going from "
            f"{origin} to {destination}. Budget is ${budget} total. "
            f"Preferences: {json.dumps(preferences)}. "
            f"Flights: {json.dumps(affordable)}",
            context=f"Data source: {results['source']}"
        )

        return {
            "flights": affordable,
            "analysis": analysis,
            "source": results["source"]
        }


class HotelAgent(SpecializedAgent):
    """Specialized agent for hotel search and selection"""

    def __init__(self, client: OpenAI, api: BookingComAPI):
        super().__init__(
            name="🏨 Hotel Agent",
            role="""You are a Hotel Accommodation Specialist. You:
- Find hotels matching budget and style preferences
- Consider location, ratings, and amenities
- Know which neighborhoods are best for different traveler types
- Balance quality with value
Present options with clear comparisons.""",
            client=client
        )
        self.api = api

    def search_and_analyze(self, city: str, checkin: str,
                           nights: int, budget_remaining: float,
                           preferences: dict) -> dict:
        """Search hotels and provide intelligent analysis"""
        print(f"\n   {self.name}: Searching hotels...")

        results = self.api.search(city, checkin, nights)
        hotels = results["hotels"]

        # Filter by remaining budget (max 50% for hotels)
        max_hotel_budget = budget_remaining * 0.5
        affordable = [
            h for h in hotels if h["total_cost"] <= max_hotel_budget
        ]

        if not affordable:
            affordable = sorted(
                hotels, key=lambda x: x["total_cost"]
            )[:3]

        analysis = self.think(
            f"Analyze these hotel options in {city} for {nights} nights. "
            f"Remaining budget: ${budget_remaining}. "
            f"Preferences: {json.dumps(preferences)}. "
            f"Hotels: {json.dumps(affordable)}",
            context=f"Data source: {results['source']}"
        )

        return {
            "hotels": affordable,
            "analysis": analysis,
            "source": results["source"]
        }


class FoodAgent(SpecializedAgent):
    """Specialized agent for restaurant and food planning"""

    def __init__(self, client: OpenAI, api: YelpAPI):
        super().__init__(
            name="🍣 Food Agent",
            role="""You are a Food & Dining Specialist. You:
- Curate the best restaurant experiences for the destination
- Balance high-end and budget-friendly options
- Create a meal plan covering breakfast, lunch, dinner
- Consider local specialties and must-try dishes
- Factor in food allergies and dietary restrictions
Present a full meal plan with cost estimates.""",
            client=client
        )
        self.api = api

    def search_and_plan(self, city: str, cuisine_preference: str,
                        days: int, budget_remaining: float,
                        preferences: dict) -> dict:
        """Search restaurants and create a meal plan"""
        print(f"\n   {self.name}: Curating food experiences...")

        results = self.api.search(city, cuisine_preference)
        restaurants = results["restaurants"]

        # Let the food agent create a meal plan
        meal_plan = self.think(
            f"Create a {days}-day meal plan in {city}. "
            f"Remaining budget for food: ${budget_remaining}. "
            f"Cuisine preferences: {cuisine_preference}. "
            f"Dietary restrictions: "
            f"{preferences.get('dietary_restrictions', 'None')}. "
            f"Available restaurants: {json.dumps(restaurants)}. "
            f"Include breakfast, lunch, and dinner for each day with "
            f"estimated costs.",
            context=f"Data source: {results['source']}"
        )

        # Calculate estimated food cost
        avg_daily_food = sum(
            r["avg_meal_cost"] for r in restaurants[:3]
        ) if restaurants else 50
        estimated_total_food = avg_daily_food * days

        return {
            "restaurants": restaurants,
            "meal_plan": meal_plan,
            "estimated_daily_cost": avg_daily_food,
            "estimated_total_cost": estimated_total_food,
            "source": results["source"]
        }


# ══════════════════════════════════════════════════════════════
# THE ORCHESTRATOR — TIES EVERYTHING TOGETHER
# ══════════════════════════════════════════════════════════════

class TripPlannerOrchestrator:
    """
    The master agent that coordinates all specialized agents,
    manages the workflow, and enforces guardrails.
    """

    def __init__(
        self,
        openai_api_key: str,
        skyscanner_key: str = None,
        rapidapi_key: str = None,
        yelp_key: str = None,
        auto_approve: bool = False,
        model: str = "gpt-4o-mini"
    ):
        # Core
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

        # APIs
        self.flight_api = SkyscannerAPI(api_key=skyscanner_key)
        self.hotel_api = BookingComAPI(api_key=rapidapi_key)
        self.yelp_api = YelpAPI(api_key=yelp_key)

        # Specialized Agents
        self.flight_agent = FlightAgent(self.client, self.flight_api)
        self.hotel_agent = HotelAgent(self.client, self.hotel_api)
        self.food_agent = FoodAgent(self.client, self.yelp_api)

        # Systems
        self.memory = AgentMemory()
        self.guardrails = Guardrails()
        self.human = HumanInTheLoop(auto_approve=auto_approve)

        # State
        self.budget_tracker = {"total_budget": 0, "spent": 0}
        self.selected_options = {}

    @property
    def budget_remaining(self) -> float:
        return (self.budget_tracker["total_budget"]
                - self.budget_tracker["spent"])

    def plan_trip(
        self,
        user_id: str,
        origin: str,
        destination: str,
        date: str,
        days: int,
        budget: float,
        preferences: dict = None
    ) -> str:
        """
        Main orchestration method.
        Coordinates all agents and systems to plan a trip.
        """
        preferences = preferences or {}

        # ══════════════════════════════════════
        # PHASE 0: INITIALIZATION & VALIDATION
        # ══════════════════════════════════════
        print("\n" + "🌟" * 30)
        print("🌍  ENHANCED AGENTIC TRIP PLANNER  🌍")
        print("🌟" * 30)

        # Guardrail: Validate budget
        budget_check = self.guardrails.validate_budget(budget)
        if not budget_check["valid"]:
            return f"❌ {budget_check['reason']}"

        # Guardrail: Validate destination
        dest_check = self.guardrails.validate_destination(destination)
        if not dest_check["valid"]:
            return f"❌ {dest_check['reason']}"

        self.budget_tracker["total_budget"] = budget

        # ══════════════════════════════════════
        # PHASE 1: LOAD MEMORY
        # ══════════════════════════════════════
        print(f"\n📂 PHASE 1: Loading user memory...")
        memory_context = self.memory.build_context(user_id)
        print(f"   {memory_context}")

        # Save/update preferences
        for key, value in preferences.items():
            self.memory.save_preference(user_id, key, str(value))

        # ══════════════════════════════════════
        # PHASE 2: FLIGHT SEARCH & SELECTION
        # ══════════════════════════════════════
        print(f"\n✈️  PHASE 2: Flight Search")
        print(f"{'─' * 50}")

        flight_results = self.flight_agent.search_and_analyze(
            origin, destination, date, budget, preferences
        )

        if not flight_results["flights"]:
            return "❌ No flights found for this route."

        # Human-in-the-loop: Choose flight
        flight_choice = self.human.present_options(
            "flights", flight_results["flights"], self.budget_remaining
        )
        selected_flight = flight_choice["selected"]

        # Guardrail: Validate flight booking
        flight_validation = self.guardrails.validate_booking(
            "flight", {"cost": selected_flight["price"]},
            self.budget_remaining
        )

        if not flight_validation["approved"]:
            print(f"   🛡️ Guardrail: {flight_validation['reason']}")
            # Re-present cheaper options
            cheaper = sorted(
                flight_results["flights"],
                key=lambda x: x["price"]
            )[:3]
            flight_choice = self.human.present_options(
                "flights", cheaper, self.budget_remaining
            )
            selected_flight = flight_choice["selected"]

        # Human confirmation
        if self.human.confirm_booking({
            "Type": "✈️ Flight",
            "Airline": selected_flight["airline"],
            "Price": f"${selected_flight['price']}",
            "Duration": selected_flight.get("duration", "N/A"),
        }):
            self.budget_tracker["spent"] += selected_flight["price"]
            self.selected_options["flight"] = selected_flight
            print(f"   💰 Budget remaining: "
                  f"${self.budget_remaining:,.2f}")
        else:
            return "❌ Trip planning cancelled at flight selection."

        # ══════════════════════════════════════
        # PHASE 3: HOTEL SEARCH & SELECTION
        # ══════════════════════════════════════
        print(f"\n🏨 PHASE 3: Hotel Search")
        print(f"{'─' * 50}")

        hotel_results = self.hotel_agent.search_and_analyze(
            destination, date, days, self.budget_remaining, preferences
        )

        if not hotel_results["hotels"]:
            return "❌ No hotels found."

        # Human-in-the-loop: Choose hotel
        hotel_choice = self.human.present_options(
            "hotels", hotel_results["hotels"], self.budget_remaining
        )
        selected_hotel = hotel_choice["selected"]

        # Guardrail: Validate hotel booking
        hotel_validation = self.guardrails.validate_booking(
            "hotel_per_night",
            {"cost": selected_hotel["price_per_night"]},
            self.budget_remaining
        )

        if not hotel_validation["approved"]:
            print(f"   🛡️ Guardrail: {hotel_validation['reason']}")

        # Human confirmation
        if self.human.confirm_booking({
            "Type": "🏨 Hotel",
            "Name": selected_hotel["name"],
            "Per Night": f"${selected_hotel['price_per_night']}",
            "Total ({} nights)".format(days):
                f"${selected_hotel['total_cost']}",
            "Area": selected_hotel.get("area", "N/A"),
            "Rating": f"⭐ {selected_hotel.get('rating', 'N/A')}",
        }):
            self.budget_tracker["spent"] += selected_hotel["total_cost"]
            self.selected_options["hotel"] = selected_hotel
            print(f"   💰 Budget remaining: "
                  f"${self.budget_remaining:,.2f}")
        else:
            return "❌ Trip planning cancelled at hotel selection."

        # ══════════════════════════════════════
        # PHASE 4: FOOD PLANNING
        # ══════════════════════════════════════
        print(f"\n🍣 PHASE 4: Food & Dining Plan")
        print(f"{'─' * 50}")

        food_results = self.food_agent.search_and_plan(
            destination,
            preferences.get("cuisine", "local food"),
            days,
            self.budget_remaining,
            preferences
        )

        # Display restaurant options for user awareness
        if food_results["restaurants"]:
            print(f"\n   🍽️  Curated Restaurants:")
            for r in food_results["restaurants"][:6]:
                print(f"      • {r['name']} ({r['type']}) "
                      f"~${r['avg_meal_cost']}/meal "
                      f"⭐{r['rating']}")

        estimated_food = food_results["estimated_total_cost"]

        # Guardrail: Check food budget
        if estimated_food > self.budget_remaining:
            print(f"\n   🛡️ Guardrail: Estimated food cost "
                  f"${estimated_food:.0f} exceeds remaining budget "
                  f"${self.budget_remaining:.0f}")
            print(f"   🔄 Adjusting to budget-friendly options...")
            estimated_food = self.budget_remaining * 0.8

        self.budget_tracker["spent"] += estimated_food
        self.selected_options["food"] = food_results

        # ══════════════════════════════════════
        # PHASE 5: GENERATE FINAL ITINERARY
        # ══════════════════════════════════════
        print(f"\n📋 PHASE 5: Generating Final Itinerary")
        print(f"{'─' * 50}")

        itinerary = self._generate_itinerary(
            origin, destination, date, days, preferences
        )

        # ══════════════════════════════════════
        # PHASE 6: FINAL APPROVAL
        # ══════════════════════════════════════
        approved = self.human.confirm_final_itinerary(itinerary)

        if approved:
            # Save to memory
            self.memory.save_trip(user_id, {
                "destination": destination,
                "budget": budget,
                "actual_cost": self.budget_tracker["spent"],
                "duration_days": days,
                "flight": self.selected_options.get("flight", {}),
                "hotel": self.selected_options.get("hotel", {}),
                "preferences": preferences,
            })
            self.memory.save_preference(
                user_id, "last_destination", destination
            )
            self.memory.save_preference(
                user_id, "avg_budget_per_day",
                str(round(budget / days, 2))
            )

            print(f"\n{'🎉' * 20}")
            print(f"  TRIP PLANNED SUCCESSFULLY!")
            print(f"  Audit log: "
                  f"{len(self.guardrails.get_audit_log())} actions verified")
            print(f"{'🎉' * 20}")

            return itinerary
        else:
            print("🔄 Let's revise the plan...")
            return "Trip planning needs revision. Please run again."

    def _generate_itinerary(
        self, origin: str, destination: str,
        date: str, days: int, preferences: dict
    ) -> str:
        """Use the orchestrator LLM to generate the final itinerary"""

        flight = self.selected_options.get("flight", {})
        hotel = self.selected_options.get("hotel", {})
        food = self.selected_options.get("food", {})

        prompt = f"""Generate a beautiful, detailed day-by-day travel 
itinerary based on these confirmed bookings:

TRIP: {origin} → {destination} ({days} days)
DATE: {date}

✈️ FLIGHT: {json.dumps(flight, indent=2)}
🏨 HOTEL: {json.dumps(hotel, indent=2)}
🍣 FOOD PLAN: {food.get('meal_plan', 'Use the restaurants listed')}
🍽️ RESTAURANTS: {json.dumps(
    [r['name'] + ' (' + r['type'] + ') ~$' + str(r['avg_meal_cost'])
     for r in food.get('restaurants', [])[:8]], indent=2
)}

💰 BUDGET SUMMARY:
  - Total Budget: ${self.budget_tracker['total_budget']:,.2f}
  - Flight: ${flight.get('price', 0):,.2f}
  - Hotel: ${hotel.get('total_cost', 0):,.2f}
  - Est. Food: ${food.get('estimated_total_cost', 0):,.2f}
  - Remaining: ${self.budget_remaining:,.2f}

PREFERENCES: {json.dumps(preferences, indent=2)}

Create a day-by-day itinerary with:
1. Each day's schedule (morning, afternoon, evening)
2. Meal recommendations for each meal
3. Local tips and hidden gems
4. Budget breakdown
5. Use emojis to make it visually appealing"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are a world-class travel planner. "
                            "Create beautiful, practical itineraries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=3000
        )

        return response.choices[0].message.content

    def cleanup(self):
        """Clean up resources"""
        self.memory.close()


# ══════════════════════════════════════════════════════════════
# RUNNER — PUTTING IT ALL TOGETHER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ─────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────
    orchestrator = TripPlannerOrchestrator(
        openai_api_key=os.getenv("OPENAI_API_KEY", "your-key-here"),

        # Real API keys (optional — falls back to simulation)
        skyscanner_key=os.getenv("SKYSCANNER_API_KEY"),
        rapidapi_key=os.getenv("RAPIDAPI_KEY"),
        yelp_key=os.getenv("YELP_API_KEY"),

        # Set True for non-interactive demo mode
        auto_approve=False,

        model="gpt-4o-mini"
    )

    # ─────────────────────────────────────────
    # Run the agent!
    # ─────────────────────────────────────────
    try:
        result = orchestrator.plan_trip(
            user_id="user_123",
            origin="New York",
            destination="Tokyo",
            date="2025-04-15",
            days=3,
            budget=2000,
            preferences={
                "cuisine": "ramen, sushi, street food",
                "travel_style": "foodie adventure",
                "dietary_restrictions": "none",
                "hotel_preference": "budget-friendly, central location",
                "interests": "food markets, local culture, hidden gems"
            }
        )

        print("\n\n" + "=" * 60)
        print("FINAL RESULT:")
        print("=" * 60)
        print(result)

    finally:
        orchestrator.cleanup()
```

---

## Sample Interactive Session

```
🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟
🌍  ENHANCED AGENTIC TRIP PLANNER  🌍
🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟

📂 PHASE 1: Loading user memory...
   No prior data. This is a new user.
   💾 Memory saved: cuisine = ramen, sushi, street food
   💾 Memory saved: travel_style = foodie adventure

✈️  PHASE 2: Flight Search
──────────────────────────────────────────────────

   ✈️ Flight Agent: Searching flights...

═══════════════════════════════════════════════════════
🙋 YOUR INPUT NEEDED — FLIGHTS
═══════════════════════════════════════════════════════
💰 Remaining Budget: $2,000.00

   [1] ✈️  United
       💲 $370 | ⏱️  15h 20m | 🔄 0 stops

   [2] ✈️  Korean Air
       💲 $390 | ⏱️  16h 30m | 🔄 1 stops

   [3] ✈️  ANA
       💲 $420 | ⏱️  14h 10m | 🔄 0 stops

   [4] ✈️  Delta
       💲 $445 | ⏱️  14h 50m | 🔄 0 stops

   [5] ✈️  JAL
       💲 $510 | ⏱️  13h 45m | 🔄 0 stops

👉 Enter your choice (1-5): 3

✅ Selected: ANA

═══════════════════════════════════════════════════════
🛡️  BOOKING CONFIRMATION REQUIRED
═══════════════════════════════════════════════════════

📋 Booking Summary:
   • Type: ✈️ Flight
   • Airline: ANA
   • Price: $420
   • Duration: 14h 10m

👉 Confirm this booking? (yes/no/modify): yes
✅ Booking confirmed!
   💰 Budget remaining: $1,580.00

🏨 PHASE 3: Hotel Search
──────────────────────────────────────────────────

   🏨 Hotel Agent: Searching hotels...

═══════════════════════════════════════════════════════
🙋 YOUR INPUT NEEDED — HOTELS
═══════════════════════════════════════════════════════
💰 Remaining Budget: $1,580.00

   [1] 🏨 Hostel Chapter Two
       💲 $35/night ($105 total) | ⭐ 4.1 | 📍 Asakusa

   [2] 🏨 Sakura Budget Inn
       💲 $55/night ($165 total) | ⭐ 3.8 | 📍 Asakusa

   [3] 🏨 Shinjuku Granbell
       💲 $95/night ($285 total) | ⭐ 4.3 | 📍 Shinjuku

   [4] 🏨 Hotel Gracery
       💲 $130/night ($390 total) | ⭐ 4.5 | 📍 Shinjuku

👉 Enter your choice (1-4): 3

✅ Selected: Shinjuku Granbell

👉 Confirm this booking? (yes/no/modify): yes
✅ Booking confirmed!
   💰 Budget remaining: $1,295.00

🍣 PHASE 4: Food & Dining Plan
──────────────────────────────────────────────────

   🍣 Food Agent: Curating food experiences...

   🍽️  Curated Restaurants:
      • Tsukiji Outer Market (Sushi & Street Food) ~$15/meal ⭐4.8
      • Ichiran Ramen (Ramen) ~$12/meal ⭐4.7
      • Nakiryu (Michelin Ramen) ~$14/meal ⭐4.9
      • Afuri Ramen (Yuzu Ramen) ~$13/meal ⭐4.6
      • Gonpachi (Izakaya) ~$35/meal ⭐4.4
      • Omoide Yokocho (Yakitori) ~$10/meal ⭐4.5

📋 PHASE 5: Generating Final Itinerary
──────────────────────────────────────────────────

📋 FINAL ITINERARY REVIEW
═══════════════════════════════════════════════════════

🗼 YOUR TOKYO FOODIE ADVENTURE — 3 DAYS 🍣

✈️ Flight: ANA — $420 (Direct, 14h 10m)
🏨 Hotel: Shinjuku Granbell, Shinjuku — $285 (3 nights)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 DAY 1: Arrival & Shinjuku Street Food
   🌅 Arrive & check in at Shinjuku Granbell
   🍳 Lunch: Ichiran Ramen (solo booth experience!) — ~$12
   🚶 Explore Shinjuku Gyoen National Garden
   🌙 Dinner: Omoide Yokocho (Memory Lane Yakitori) — ~$10
   🍻 Night: Golden Gai bar-hopping

📅 DAY 2: Tsukiji & Ramen Pilgrimage  
   🌅 Breakfast: Tsukiji Outer Market (fresh sushi!) — ~$15
   🍜 Lunch: Nakiryu Michelin-starred ramen — ~$14
   🚶 Explore Asakusa & Senso-ji Temple
   🌙 Dinner: Gonpachi (the Kill Bill restaurant!) — ~$35
   💡 Tip: Make a reservation for Gonpachi!

📅 DAY 3: Last Bites & Departure
   🌅 Breakfast: 7-Eleven (seriously amazing!) — ~$5
   🍜 Lunch: Afuri Ramen (yuzu broth heaven) — ~$13
   🛍️ Shopping at Don Quijote
   ✈️ Head to airport

💰 BUDGET BREAKDOWN:
   ✈️ Flight:  $420
   🏨 Hotel:   $285
   🍣 Food:    ~$140 (estimated)
   ───────────────────
   📊 Total:   ~$845
   💰 Remaining: ~$1,155 for activities & shopping!

👉 Approve this itinerary? (yes/no): yes

💾 Memory saved: last_destination = Tokyo
💾 Memory saved: avg_budget_per_day = 666.67
💾 Trip saved to history: a3f2b1c9d8e7

🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
  TRIP PLANNED SUCCESSFULLY!
  Audit log: 2 actions verified
🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
```

---

## Returning User Session (Memory In Action)

```python
# Same user plans a second trip
orchestrator.plan_trip(
    user_id="user_123",     # Same user ID
    origin="New York",
    destination="Seoul",
    date="2025-07-01",
    days=4,
    budget=2500,
    preferences={
        "cuisine": "Korean BBQ, street food",
    }
)
```

```
📂 PHASE 1: Loading user memory...
   ## Known User Preferences:
   - cuisine: Korean BBQ, street food (confidence: 1.0)
   - last_destination: Tokyo (confidence: 1.0)
   - avg_budget_per_day: 666.67 (confidence: 1.0)
   - travel_style: foodie adventure (confidence: 1.0)
   - dietary_restrictions: none (confidence: 1.0)
   - hotel_preference: budget-friendly, central location (confidence: 1.0)

   ## Recent Trip History:
   - Tokyo (3 days, $845/$2000 budget) on 2025-01-15

   🧠 Agent knows: "This user is a foodie who likes budget-friendly
      central hotels and typically spends ~43% of their budget."
```

---

## How Each Enhancement Works

```
┌──────────────────────────────────────────────────────────────────┐
│                    ENHANCEMENT SUMMARY                          │
├──────────────────┬───────────────────────────────────────────────┤
│ 🔹 Real APIs     │ SkyscannerAPI, BookingComAPI, YelpAPI        │
│                  │ → Falls back gracefully to simulated data    │
│                  │ → Just add API keys to activate              │
├──────────────────┼───────────────────────────────────────────────┤
│ 🔹 Memory        │ SQLite persistent storage                    │
│                  │ → User preferences (long-term)               │
│                  │ → Trip history (episodic)                    │
│                  │ → Context builder for LLM                   │
├──────────────────┼───────────────────────────────────────────────┤
│ 🔹 Multi-Agent   │ FlightAgent + HotelAgent + FoodAgent        │
│                  │ → Each has specialized system prompt         │
│                  │ → Orchestrator coordinates workflow          │
│                  │ → Agents analyze data independently          │
├──────────────────┼───────────────────────────────────────────────┤
│ 🔹 Human-in-Loop │ HumanInTheLoop class                        │
│                  │ → Option selection for flights/hotels        │
│                  │ → Booking confirmation at each step          │
│                  │ → Final itinerary approval                  │
│                  │ → Auto-approve mode for demos               │
├──────────────────┼───────────────────────────────────────────────┤
│ 🔹 Guardrails    │ Guardrails class                             │
│                  │ → Budget validation & tracking               │
│                  │ → Destination safety checks                  │
│                  │ → Booking cost thresholds                    │
│                  │ → Required field validation                  │
│                  │ → Complete audit logging                     │
└──────────────────┴───────────────────────────────────────────────┘
```

---

## Setup Instructions

```bash
# 1. Install dependencies
pip install openai requests

# 2. Set environment variables (add whichever keys you have)
export OPENAI_API_KEY="sk-..."
export SKYSCANNER_API_KEY="..."      # Optional
export RAPIDAPI_KEY="..."            # Optional (for Booking.com)
export YELP_API_KEY="..."            # Optional

# 3. Run
python enhanced_trip_planner.py
```

> **Note**: The system works with **zero external API keys** (using simulated data) — just the OpenAI key is required. Add real API keys incrementally to upgrade individual components to live data.

---

This implementation demonstrates a **production-grade agentic architecture** where multiple specialized AI agents collaborate under an orchestrator, with real API integrations, persistent memory, human oversight, and safety guardrails working together seamlessly. Each component is modular and can be extended or replaced independently.

Would you like me to dive deeper into any specific enhancement?

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 🔬 Deep Dive Into All 5 Enhancements

---

## 📑 Table of Contents

```
1. 🔹 Real APIs          — Production Integration Patterns
2. 🔹 Memory             — Advanced Persistent Memory Architecture
3. 🔹 Multi-Agent        — Agent Communication & Coordination
4. 🔹 Human-in-the-Loop  — Approval Workflows & Escalation
5. 🔹 Guardrails         — Safety, Validation & Monitoring
```

---

# 1️⃣ DEEP DIVE: Real API Integration

## The Challenge

```
Raw API responses are messy, inconsistent, and unreliable.
Your agent needs CLEAN, NORMALIZED, RELIABLE data.

❌ Without proper integration:
   API down → Agent crashes
   Bad data → Wrong recommendations
   Rate limits → Failed searches
   
✅ With proper integration:
   API down → Graceful fallback
   Bad data → Validated & normalized
   Rate limits → Retry with backoff
```

## Production-Grade API Layer

```python
import time
import logging
from functools import wraps
from typing import Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TripAgent")


# ─────────────────────────────────────────
# 1A. RETRY & RESILIENCE FRAMEWORK
# ─────────────────────────────────────────

class APIStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class APIHealth:
    """Track health of each API"""
    name: str
    status: APIStatus = APIStatus.HEALTHY
    consecutive_failures: int = 0
    last_success: float = 0
    last_failure: float = 0
    total_calls: int = 0
    total_failures: int = 0

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0
        return self.total_failures / self.total_calls

    def record_success(self):
        self.consecutive_failures = 0
        self.last_success = time.time()
        self.total_calls += 1
        self.status = APIStatus.HEALTHY

    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure = time.time()
        self.total_calls += 1
        self.total_failures += 1
        if self.consecutive_failures >= 3:
            self.status = APIStatus.DOWN
        elif self.consecutive_failures >= 1:
            self.status = APIStatus.DEGRADED


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True
):
    """
    Decorator: Retries failed API calls with exponential backoff.
    
    Why exponential backoff?
    ─────────────────────────
    Attempt 1: Wait 1s   → Quick retry for transient errors
    Attempt 2: Wait 2s   → Give server time to recover
    Attempt 3: Wait 4s   → Avoid overwhelming a struggling server
    
    Without this, 1000 agents hitting a down API = DDoS attack!
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(
                            f"✅ {func.__name__} succeeded on "
                            f"attempt {attempt + 1}"
                        )
                    return result

                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        if exponential:
                            delay = min(
                                base_delay * (2 ** attempt), max_delay
                            )
                        else:
                            delay = base_delay

                        logger.warning(
                            f"⚠️ {func.__name__} attempt "
                            f"{attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"❌ {func.__name__} failed after "
                            f"{max_retries + 1} attempts: {e}"
                        )

            raise last_exception
        return wrapper
    return decorator


# ─────────────────────────────────────────
# 1B. DATA NORMALIZER
# ─────────────────────────────────────────

class DataNormalizer:
    """
    Ensures consistent data format regardless of API source.
    
    Problem: Each API returns data differently:
    
    Skyscanner: {"price": {"amount": 420, "currency": "USD"}}
    Google:     {"fare": "$420.00"}
    Kayak:      {"total_price_usd": 420}
    
    Solution: Normalize everything to a standard schema.
    """

    @staticmethod
    def normalize_flight(raw: dict, source: str) -> dict:
        """Convert any flight API response to standard format"""

        # Standard schema every flight must match
        normalized = {
            "airline": "",
            "price": 0.0,
            "currency": "USD",
            "duration_minutes": 0,
            "duration_display": "",
            "departure_time": "",
            "arrival_time": "",
            "stops": 0,
            "cabin_class": "economy",
            "source": source,
            "raw_data": raw,  # Keep original for debugging
        }

        if source == "skyscanner":
            normalized["airline"] = (
                raw.get("carriers", [{}])[0].get("name", "Unknown")
            )
            normalized["price"] = float(
                raw.get("price", {}).get("raw", 0)
            )
            normalized["duration_minutes"] = raw.get("duration", 0)
            normalized["stops"] = raw.get("stopCount", 0)

        elif source == "google_flights":
            normalized["airline"] = raw.get("airline", "Unknown")
            normalized["price"] = float(
                raw.get("price", "0").replace("$", "").replace(",", "")
            )
            normalized["duration_display"] = raw.get("duration", "")

        elif source == "simulated":
            normalized["airline"] = raw.get("airline", "Unknown")
            normalized["price"] = float(raw.get("price", 0))
            normalized["duration_display"] = raw.get("duration", "")
            normalized["stops"] = raw.get("stops", 0)
            normalized["departure_time"] = raw.get("departure", "")

        # Compute display duration if we have minutes
        if (normalized["duration_minutes"] > 0
                and not normalized["duration_display"]):
            hours = normalized["duration_minutes"] // 60
            mins = normalized["duration_minutes"] % 60
            normalized["duration_display"] = f"{hours}h {mins}m"

        return normalized

    @staticmethod
    def normalize_hotel(raw: dict, source: str, nights: int) -> dict:
        """Convert any hotel API response to standard format"""
        normalized = {
            "name": "",
            "price_per_night": 0.0,
            "total_cost": 0.0,
            "currency": "USD",
            "rating": 0.0,
            "rating_scale": 5.0,
            "review_count": 0,
            "stars": 0,
            "area": "",
            "amenities": [],
            "cancellation_policy": "unknown",
            "source": source,
            "raw_data": raw,
        }

        if source == "booking.com":
            normalized["name"] = raw.get("hotel_name", "Unknown")
            total = float(raw.get("min_total_price", 0))
            normalized["total_cost"] = total
            normalized["price_per_night"] = round(
                total / max(nights, 1), 2
            )
            # Booking.com rates out of 10
            normalized["rating"] = round(
                raw.get("review_score", 0) / 2, 1
            )
            normalized["review_count"] = raw.get("review_nr", 0)
            normalized["stars"] = raw.get("class", 0)
            normalized["area"] = raw.get("district", "")

        elif source == "simulated":
            normalized["name"] = raw.get("name", "Unknown")
            normalized["price_per_night"] = float(
                raw.get("price_per_night", 0)
            )
            normalized["total_cost"] = float(
                raw.get("total_cost",
                        normalized["price_per_night"] * nights)
            )
            normalized["rating"] = float(raw.get("rating", 0))
            normalized["stars"] = raw.get("stars", 0)
            normalized["area"] = raw.get("area", "")

        return normalized

    @staticmethod
    def normalize_restaurant(raw: dict, source: str) -> dict:
        """Convert any restaurant API response to standard format"""
        normalized = {
            "name": "",
            "cuisine_type": "",
            "avg_meal_cost": 0.0,
            "currency": "USD",
            "price_level": 0,
            "rating": 0.0,
            "review_count": 0,
            "address": "",
            "phone": "",
            "hours": "",
            "dietary_options": [],
            "source": source,
            "raw_data": raw,
        }

        if source == "yelp":
            normalized["name"] = raw.get("name", "Unknown")
            normalized["cuisine_type"] = ", ".join(
                c["title"] for c in raw.get("categories", [])
            )
            price_str = raw.get("price", "$$")
            normalized["price_level"] = len(price_str)
            cost_map = {1: 10, 2: 25, 3: 50, 4: 100}
            normalized["avg_meal_cost"] = cost_map.get(
                normalized["price_level"], 25
            )
            normalized["rating"] = float(raw.get("rating", 0))
            normalized["review_count"] = raw.get("review_count", 0)
            normalized["address"] = ", ".join(
                raw.get("location", {}).get("display_address", [])
            )

        elif source == "simulated":
            normalized["name"] = raw.get("name", "Unknown")
            normalized["cuisine_type"] = raw.get("type", "")
            normalized["avg_meal_cost"] = float(
                raw.get("avg_meal_cost", 0)
            )
            normalized["rating"] = float(raw.get("rating", 0))

        return normalized


# ─────────────────────────────────────────
# 1C. CACHING LAYER
# ─────────────────────────────────────────

class APICache:
    """
    Cache API results to:
    1. Reduce API costs (fewer calls)
    2. Speed up repeated searches
    3. Provide data when APIs are down
    
    Uses SQLite for persistence across sessions.
    """

    def __init__(self, db_path: str = "api_cache.db",
                 default_ttl: int = 3600):
        self.conn = sqlite3.connect(db_path)
        self.default_ttl = default_ttl  # 1 hour default
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT,
                source TEXT,
                created_at REAL,
                ttl INTEGER
            )
        """)
        self.conn.commit()

    def _make_key(self, api_name: str, params: dict) -> str:
        """Create a unique cache key from API + params"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(
            f"{api_name}:{param_str}".encode()
        ).hexdigest()

    def get(self, api_name: str, params: dict) -> Optional[dict]:
        """Retrieve cached result if still valid"""
        key = self._make_key(api_name, params)
        cursor = self.conn.execute(
            "SELECT data, created_at, ttl FROM cache WHERE cache_key = ?",
            (key,)
        )
        row = cursor.fetchone()

        if row:
            data, created_at, ttl = row
            age = time.time() - created_at
            if age < ttl:
                logger.info(
                    f"   📦 Cache HIT for {api_name} "
                    f"(age: {age:.0f}s / {ttl}s TTL)"
                )
                return json.loads(data)
            else:
                logger.info(
                    f"   📦 Cache EXPIRED for {api_name}"
                )

        return None

    def set(self, api_name: str, params: dict, data: dict,
            ttl: int = None):
        """Store result in cache"""
        key = self._make_key(api_name, params)
        self.conn.execute(
            """INSERT OR REPLACE INTO cache
               (cache_key, data, source, created_at, ttl)
               VALUES (?, ?, ?, ?, ?)""",
            (key, json.dumps(data), api_name, time.time(),
             ttl or self.default_ttl)
        )
        self.conn.commit()
        logger.info(f"   📦 Cache SET for {api_name}")

    def clear(self, api_name: str = None):
        """Clear cache (all or specific API)"""
        if api_name:
            self.conn.execute(
                "DELETE FROM cache WHERE source = ?", (api_name,)
            )
        else:
            self.conn.execute("DELETE FROM cache")
        self.conn.commit()


# ─────────────────────────────────────────
# 1D. ENHANCED API WITH ALL PATTERNS
# ─────────────────────────────────────────

class ResilientSkyscannerAPI:
    """
    Production-ready Skyscanner integration with:
    ✅ Retry with exponential backoff
    ✅ Response caching
    ✅ Data normalization
    ✅ Health monitoring
    ✅ Graceful fallback
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.cache = APICache()
        self.normalizer = DataNormalizer()
        self.health = APIHealth(name="skyscanner")
        self.base_url = "https://partners.api.skyscanner.net/apiservices/v3"

    def search(self, origin: str, destination: str,
               date: str) -> dict:
        """Search with full resilience pipeline"""

        params = {
            "origin": origin, "destination": destination, "date": date
        }

        # ── Step 1: Check cache ──
        cached = self.cache.get("skyscanner_flights", params)
        if cached:
            return cached

        # ── Step 2: Check API health ──
        if self.health.status == APIStatus.DOWN:
            logger.warning(
                "⚠️ Skyscanner API is DOWN. Using fallback."
            )
            return self._fallback(origin, destination, date)

        # ── Step 3: Make API call with retry ──
        try:
            result = self._api_call_with_retry(
                origin, destination, date
            )

            # ── Step 4: Normalize data ──
            source = "skyscanner" if self.api_key else "simulated"
            normalized_flights = [
                self.normalizer.normalize_flight(f, source)
                for f in result.get("flights", [])
            ]
            result["flights"] = normalized_flights

            # ── Step 5: Cache result ──
            self.cache.set(
                "skyscanner_flights", params, result,
                ttl=1800  # 30 minutes for flights
            )

            self.health.record_success()
            return result

        except Exception as e:
            self.health.record_failure()
            logger.error(f"❌ Skyscanner search failed: {e}")
            return self._fallback(origin, destination, date)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _api_call_with_retry(self, origin: str,
                              destination: str, date: str) -> dict:
        """Actual API call wrapped in retry logic"""
        if not self.api_key:
            return self._simulate(origin, destination, date)

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{self.base_url}/flights/live/search/create",
            headers=headers,
            json=self._build_search_payload(
                origin, destination, date
            ),
            timeout=15
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _fallback(self, origin, destination, date) -> dict:
        """Last resort: return simulated data"""
        logger.info("🔄 Using simulated flight data as fallback")
        result = self._simulate(origin, destination, date)
        result["source"] = "fallback_simulated"
        result["warning"] = "Live flight data unavailable"
        return result

    def _simulate(self, origin, destination, date) -> dict:
        """Generate realistic simulated data"""
        import random
        airlines = [
            ("ANA", 420, "14h 10m", 0),
            ("JAL", 510, "13h 45m", 0),
            ("United", 370, "15h 20m", 1),
            ("Delta", 445, "14h 50m", 0),
            ("Korean Air", 390, "16h 30m", 1),
        ]
        flights = []
        for airline, base, duration, stops in airlines:
            flights.append({
                "airline": airline,
                "price": base + random.randint(-30, 50),
                "duration": duration,
                "stops": stops,
                "departure": f"{random.randint(6,20):02d}:00",
                "source": "simulated"
            })
        return {
            "flights": sorted(flights, key=lambda x: x["price"]),
            "origin": origin,
            "destination": destination,
            "date": date,
            "source": "simulated"
        }

    def get_health_report(self) -> dict:
        """Return API health metrics"""
        return {
            "name": self.health.name,
            "status": self.health.status.value,
            "failure_rate": f"{self.health.failure_rate:.1%}",
            "consecutive_failures": self.health.consecutive_failures,
            "total_calls": self.health.total_calls,
        }
```

### How The Pipeline Works

```
User Request
    │
    ▼
┌──────────┐    HIT     ┌────────────┐
│  CACHE   │ ─────────► │  Return    │
│  Check   │            │  Cached    │
└────┬─────┘            └────────────┘
     │ MISS
     ▼
┌──────────┐    DOWN    ┌────────────┐
│  HEALTH  │ ─────────► │  Fallback  │
│  Check   │            │  (Simulated│
└────┬─────┘            └────────────┘
     │ HEALTHY
     ▼
┌──────────┐   FAIL     ┌────────────┐
│ API Call │ ──(x3)───► │  Fallback  │
│ + Retry  │            │  (Simulated│
└────┬─────┘            └────────────┘
     │ SUCCESS
     ▼
┌──────────┐
│NORMALIZE │ ──► Consistent data format
└────┬─────┘
     │
     ▼
┌──────────┐
│  CACHE   │ ──► Save for next time
│  Store   │
└────┬─────┘
     │
     ▼
  Return to Agent ✅
```

---

# 2️⃣ DEEP DIVE: Memory System

## Why Memory Matters

```
Without Memory:                    With Memory:
────────────────                   ────────────────
"What food do you like?"           "Welcome back! I remember you
"What's your budget style?"        love ramen and prefer budget
"Hotel preference?"                hotels in central locations.
"Any dietary restrictions?"        Last time in Tokyo you spent
                                   $845 of your $2000 budget.
                                   Let me use that to plan Seoul!"
                                   
Every trip = starting from zero    Every trip = builds on history
```

## Advanced Memory Architecture

```python
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────
# 2A. MEMORY TYPES (Cognitive Architecture)
# ─────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory with metadata"""
    content: str
    memory_type: str        # "preference", "episodic", "semantic"
    importance: float       # 0.0 to 1.0
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = None
    embedding: List[float] = None  # For semantic search
    metadata: dict = None


class AdvancedMemorySystem:
    """
    Multi-layered memory inspired by human cognition:
    
    ┌─────────────────────────────────────────────┐
    │              MEMORY LAYERS                   │
    ├─────────────────────────────────────────────┤
    │                                              │
    │  🔴 Working Memory (Current Session)         │
    │     → Current trip context                   │
    │     → Active conversation                    │
    │     → Temporary calculations                 │
    │                                              │
    │  🟡 Short-Term Memory (Recent Sessions)      │
    │     → Last few interactions                  │
    │     → Recent preferences                     │
    │     → Pending follow-ups                     │
    │                                              │
    │  🟢 Long-Term Memory (Persistent)            │
    │     → User preferences                       │
    │     → Trip history                           │
    │     → Learned patterns                       │
    │                                              │
    │  🔵 Semantic Memory (Knowledge)              │
    │     → Destination knowledge                  │
    │     → Pricing patterns                       │
    │     → Seasonal insights                      │
    │                                              │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, db_path: str = "agent_memory_v2.db",
                 openai_client=None):
        self.conn = sqlite3.connect(db_path)
        self.client = openai_client  # For embeddings
        self._create_tables()

        # Working memory (in-session only)
        self.working_memory = {}

    def _create_tables(self):
        cursor = self.conn.cursor()

        # ── Core memories table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                category TEXT,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                expires_at TIMESTAMP,
                embedding BLOB,
                metadata TEXT
            )
        """)

        # ── Preference evolution tracking ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                preference_key TEXT,
                old_value TEXT,
                new_value TEXT,
                change_reason TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Trip feedback (for learning) ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trip_feedback (
                trip_id TEXT PRIMARY KEY,
                user_id TEXT,
                overall_rating INTEGER,
                flight_rating INTEGER,
                hotel_rating INTEGER,
                food_rating INTEGER,
                comments TEXT,
                would_repeat BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Indexes for fast lookup ──
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user
            ON memories(user_id, memory_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance
            ON memories(user_id, importance DESC)
        """)

        self.conn.commit()

    # ─────────────────────────────────────────
    # 2B. MEMORY OPERATIONS
    # ─────────────────────────────────────────

    def remember(self, user_id: str, content: str,
                 memory_type: str, category: str = None,
                 importance: float = 0.5,
                 ttl_hours: int = None) -> str:
        """
        Store a new memory.
        
        Importance scoring:
        0.0 - 0.3: Low   (casual mentions, one-time preferences)
        0.3 - 0.6: Medium (stated preferences, trip choices)
        0.6 - 0.8: High  (strong preferences, repeated patterns)
        0.8 - 1.0: Critical (allergies, safety requirements)
        """
        memory_id = hashlib.md5(
            f"{user_id}:{content}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        expires_at = None
        if ttl_hours:
            expires_at = (
                datetime.now() + timedelta(hours=ttl_hours)
            ).isoformat()

        # Generate embedding for semantic search
        embedding = self._get_embedding(content)

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memories
            (memory_id, user_id, content, memory_type, category,
             importance, embedding, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, user_id, content, memory_type,
            category, importance,
            json.dumps(embedding) if embedding else None,
            expires_at
        ))
        self.conn.commit()

        logger.info(
            f"   💾 Remembered [{memory_type}] "
            f"(importance: {importance}): {content[:50]}..."
        )
        return memory_id

    def recall(self, user_id: str, query: str = None,
               memory_type: str = None,
               limit: int = 10,
               min_importance: float = 0.0) -> List[dict]:
        """
        Recall memories, optionally using semantic search.
        
        If query is provided → semantic similarity search
        Otherwise → recency + importance ranking
        """
        cursor = self.conn.cursor()

        # Clean up expired memories
        cursor.execute(
            "DELETE FROM memories WHERE expires_at < ?",
            (datetime.now().isoformat(),)
        )

        if query and self.client:
            # ── Semantic Search ──
            return self._semantic_recall(
                user_id, query, limit, min_importance
            )
        else:
            # ── Standard Recall ──
            sql = """
                SELECT memory_id, content, memory_type, category,
                       importance, access_count, created_at,
                       last_accessed, metadata
                FROM memories
                WHERE user_id = ?
                AND importance >= ?
            """
            params = [user_id, min_importance]

            if memory_type:
                sql += " AND memory_type = ?"
                params.append(memory_type)

            sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            # Update access counts
            for row in rows:
                cursor.execute("""
                    UPDATE memories 
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE memory_id = ?
                """, (row[0],))
            self.conn.commit()

            return [
                {
                    "memory_id": r[0], "content": r[1],
                    "type": r[2], "category": r[3],
                    "importance": r[4], "access_count": r[5],
                    "created_at": r[6], "last_accessed": r[7],
                }
                for r in rows
            ]

    def _semantic_recall(self, user_id: str, query: str,
                         limit: int,
                         min_importance: float) -> List[dict]:
        """Find memories semantically similar to the query"""
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return self.recall(
                user_id, query=None, limit=limit,
                min_importance=min_importance
            )

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT memory_id, content, memory_type, category,
                   importance, embedding
            FROM memories
            WHERE user_id = ? AND importance >= ?
            AND embedding IS NOT NULL
        """, (user_id, min_importance))

        rows = cursor.fetchall()

        # Calculate cosine similarity
        scored = []
        for row in rows:
            stored_embedding = json.loads(row[5])
            similarity = self._cosine_similarity(
                query_embedding, stored_embedding
            )
            # Combine similarity with importance
            combined_score = (similarity * 0.7) + (row[4] * 0.3)
            scored.append({
                "memory_id": row[0], "content": row[1],
                "type": row[2], "category": row[3],
                "importance": row[4],
                "similarity": round(similarity, 3),
                "combined_score": round(combined_score, 3),
            })

        # Sort by combined score
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored[:limit]

    # ─────────────────────────────────────────
    # 2C. PREFERENCE LEARNING
    # ─────────────────────────────────────────

    def learn_preference(self, user_id: str, key: str,
                         value: str, reason: str = "stated",
                         confidence: float = 1.0):
        """
        Learn a preference with change tracking.
        Tracks how preferences evolve over time.
        """
        # Check existing preference
        existing = self.recall(
            user_id, memory_type="preference"
        )
        old_value = None
        for mem in existing:
            if key.lower() in mem["content"].lower():
                old_value = mem["content"]
                break

        # Track the change
        if old_value:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO preference_history
                (user_id, preference_key, old_value,
                 new_value, change_reason)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, key, old_value, value, reason))
            self.conn.commit()
            logger.info(
                f"   🔄 Preference updated: {key}: "
                f"{old_value} → {value}"
            )

        # Store as memory
        importance = min(0.5 + (confidence * 0.3), 1.0)
        self.remember(
            user_id,
            content=f"{key}: {value}",
            memory_type="preference",
            category=key,
            importance=importance
        )

    def infer_preferences(self, user_id: str) -> dict:
        """
        Analyze trip history to infer unstated preferences.
        
        Example: User always picks 3-star hotels under $100/night
        → Infer: "budget-conscious, values location over luxury"
        """
        cursor = self.conn.cursor()

        # Get all trip data
        cursor.execute("""
            SELECT trip_data FROM trip_history
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 10
        """, (user_id,))

        trips = [
            json.loads(row[0]) for row in cursor.fetchall() if row[0]
        ]

        if not trips:
            return {}

        inferences = {}

        # ── Budget pattern ──
        budget_utilizations = []
        for trip in trips:
            budget = trip.get("budget", 0)
            actual = trip.get("actual_cost", 0)
            if budget > 0:
                budget_utilizations.append(actual / budget)

        if budget_utilizations:
            avg_utilization = sum(budget_utilizations) / len(
                budget_utilizations
            )
            if avg_utilization < 0.5:
                inferences["spending_style"] = "conservative"
            elif avg_utilization < 0.8:
                inferences["spending_style"] = "moderate"
            else:
                inferences["spending_style"] = "full-budget"

        # ── Hotel pattern ──
        hotel_prices = [
            t.get("hotel", {}).get("price_per_night", 0)
            for t in trips if t.get("hotel")
        ]
        if hotel_prices:
            avg_hotel = sum(hotel_prices) / len(hotel_prices)
            if avg_hotel < 60:
                inferences["hotel_tier"] = "budget"
            elif avg_hotel < 150:
                inferences["hotel_tier"] = "mid-range"
            else:
                inferences["hotel_tier"] = "luxury"

        # ── Destination pattern ──
        destinations = [
            t.get("destination", "") for t in trips
        ]
        if destinations:
            inferences["visited"] = list(set(destinations))

        # Store inferences as memories
        for key, value in inferences.items():
            self.learn_preference(
                user_id, key, str(value),
                reason="inferred_from_history",
                confidence=0.7
            )

        return inferences

    # ─────────────────────────────────────────
    # 2D. CONTEXT BUILDER FOR LLM
    # ─────────────────────────────────────────

    def build_rich_context(self, user_id: str,
                           current_request: str = "") -> str:
        """
        Build comprehensive context for the LLM.
        Combines all memory types into a structured prompt.
        """
        sections = []

        # ── Critical memories first ──
        critical = self.recall(
            user_id, min_importance=0.8, limit=5
        )
        if critical:
            sections.append(
                "⚠️ CRITICAL USER INFO (always consider):"
            )
            for mem in critical:
                sections.append(f"  • {mem['content']}")

        # ── Preferences ──
        preferences = self.recall(
            user_id, memory_type="preference", limit=10
        )
        if preferences:
            sections.append("\n👤 USER PREFERENCES:")
            for pref in preferences:
                sections.append(f"  • {pref['content']}")

        # ── Relevant memories (semantic search) ──
        if current_request and self.client:
            relevant = self._semantic_recall(
                user_id, current_request, limit=5,
                min_importance=0.3
            )
            if relevant:
                sections.append(
                    "\n🔍 RELEVANT PAST CONTEXT:"
                )
                for mem in relevant:
                    sections.append(
                        f"  • {mem['content']} "
                        f"(relevance: {mem['similarity']})"
                    )

        # ── Inferred patterns ──
        inferences = self.infer_preferences(user_id)
        if inferences:
            sections.append("\n🧠 INFERRED PATTERNS:")
            for key, value in inferences.items():
                sections.append(f"  • {key}: {value}")

        # ── Working memory ──
        if self.working_memory:
            sections.append("\n📋 CURRENT SESSION:")
            for key, value in self.working_memory.items():
                sections.append(f"  • {key}: {value}")

        return "\n".join(sections) if sections else \
            "No prior data. This is a new user."

    # ─────────────────────────────────────────
    # 2E. HELPERS
    # ─────────────────────────────────────────

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for semantic search"""
        if not self.client:
            return None
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    @staticmethod
    def _cosine_similarity(vec1: List[float],
                           vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )

    def forget(self, user_id: str, memory_id: str = None,
               memory_type: str = None):
        """
        Explicit forgetting (user privacy / GDPR compliance).
        Users can request their data be deleted.
        """
        cursor = self.conn.cursor()
        if memory_id:
            cursor.execute(
                "DELETE FROM memories WHERE memory_id = ? AND user_id = ?",
                (memory_id, user_id)
            )
        elif memory_type:
            cursor.execute(
                "DELETE FROM memories WHERE user_id = ? AND memory_type = ?",
                (user_id, memory_type)
            )
        else:
            # Forget everything about this user
            cursor.execute(
                "DELETE FROM memories WHERE user_id = ?", (user_id,)
            )
            cursor.execute(
                "DELETE FROM preference_history WHERE user_id = ?",
                (user_id,)
            )
            cursor.execute(
                "DELETE FROM trip_feedback WHERE user_id = ?",
                (user_id,)
            )
        self.conn.commit()
        logger.info(f"   🗑️ Memory deleted for user {user_id}")
```

### Memory Flow Diagram

```
User says: "I'm vegetarian and love boutique hotels"
    │
    ▼
┌──────────────────────────────┐
│  EXTRACT PREFERENCES         │
│  "vegetarian" → importance=0.9│ ← Critical (safety)
│  "boutique hotels" → imp=0.6 │ ← Strong preference
└──────────────┬───────────────┘
               │
    ┌──────────▼──────────┐
    │  CHECK EXISTING     │
    │  Was it "budget"?   │──► Track change in
    │  Now "boutique"?    │    preference_history
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  GENERATE EMBEDDING │
    │  For semantic search │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  STORE IN SQLite     │
    │  With importance,    │
    │  confidence, TTL     │
    └──────────┬──────────┘
               │
               ▼
    Next trip: Context builder includes
    "⚠️ CRITICAL: User is vegetarian"
    "👤 PREFERENCE: boutique hotels"
```

---

# 3️⃣ DEEP DIVE: Multi-Agent System

## Why Multiple Agents?

```
Single Agent Problem:
─────────────────────
One prompt trying to be an expert in EVERYTHING:
✈️ Flight pricing strategies
🏨 Hotel neighborhood knowledge
🍣 Food scene expertise
💰 Budget optimization
📋 Itinerary design

Result: "Jack of all trades, master of none"

Multi-Agent Solution:
─────────────────────
Each agent is a SPECIALIST with focused expertise.
They collaborate through the Orchestrator.

Result: Expert-level output in every domain!
```

## Advanced Multi-Agent Architecture

```python
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import asyncio


# ─────────────────────────────────────────
# 3A. AGENT COMMUNICATION PROTOCOL
# ─────────────────────────────────────────

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    DELEGATE = "delegate"
    ESCALATE = "escalate"
    INFORM = "inform"
    FEEDBACK = "feedback"


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    sender: str
    recipient: str
    message_type: MessageType
    content: dict
    priority: int = 5          # 1 (highest) to 10 (lowest)
    requires_response: bool = False
    correlation_id: str = ""   # Track related messages
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.message_type.value,
            "content": self.content,
            "priority": self.priority,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
        }


class MessageBus:
    """
    Central message bus for agent communication.
    
    Agents don't talk directly — they send messages
    through the bus. This enables:
    1. Logging all communications
    2. Message routing
    3. Priority handling
    4. Async processing
    """

    def __init__(self):
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, callable] = {}
        self.message_log: List[dict] = []

    def subscribe(self, agent_name: str, handler: callable):
        """Register an agent to receive messages"""
        self.subscribers[agent_name] = handler

    def publish(self, message: AgentMessage):
        """Send a message through the bus"""
        self.messages.append(message)
        self.message_log.append(message.to_dict())

        logger.info(
            f"   📨 {message.sender} → {message.recipient}: "
            f"{message.message_type.value}"
        )

        # Deliver to recipient
        if message.recipient in self.subscribers:
            handler = self.subscribers[message.recipient]
            return handler(message)
        elif message.recipient == "broadcast":
            results = {}
            for name, handler in self.subscribers.items():
                if name != message.sender:
                    results[name] = handler(message)
            return results

        return None

    def get_conversation(self, correlation_id: str) -> List[dict]:
        """Get all messages in a conversation thread"""
        return [
            m for m in self.message_log
            if m["correlation_id"] == correlation_id
        ]


# ─────────────────────────────────────────
# 3B. ENHANCED SPECIALIZED AGENTS
# ─────────────────────────────────────────

class EnhancedSpecializedAgent:
    """
    Base class with advanced capabilities:
    - Self-reflection
    - Confidence scoring
    - Collaboration requests
    - Learning from feedback
    """

    def __init__(self, name: str, role: str,
                 client: OpenAI, message_bus: MessageBus,
                 model: str = "gpt-4o-mini"):
        self.name = name
        self.role = role
        self.client = client
        self.model = model
        self.message_bus = message_bus
        self.confidence_history = []

        # Register on message bus
        self.message_bus.subscribe(name, self.handle_message)

    def handle_message(self, message: AgentMessage) -> dict:
        """Process incoming messages"""
        if message.message_type == MessageType.REQUEST:
            return self.process_request(message.content)
        elif message.message_type == MessageType.FEEDBACK:
            return self.learn_from_feedback(message.content)
        elif message.message_type == MessageType.INFORM:
            return self.update_context(message.content)
        return {"status": "unhandled"}

    def process_request(self, request: dict) -> dict:
        """Override in subclasses"""
        raise NotImplementedError

    def think(self, prompt: str, context: str = "") -> dict:
        """
        Enhanced thinking with confidence scoring
        and self-reflection.
        """
        # ── Step 1: Initial reasoning ──
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user", "content": (
                    f"Context:\n{context}\n\n"
                    f"Task:\n{prompt}\n\n"
                    f"Provide your analysis, then rate your "
                    f"confidence (0-100%) in your recommendation."
                )}
            ],
            temperature=0.7
        )
        initial_response = response.choices[0].message.content

        # ── Step 2: Self-reflection ──
        reflection = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a critical reviewer. Evaluate the "
                    "analysis below for:\n"
                    "1. Missing considerations\n"
                    "2. Potential biases\n"
                    "3. Alternative options not considered\n"
                    "4. Risk factors\n"
                    "Be brief and constructive."
                )},
                {"role": "user", "content": (
                    f"Analysis to review:\n{initial_response}"
                )}
            ],
            temperature=0.5
        )
        critique = reflection.choices[0].message.content

        # ── Step 3: Refined response ──
        refined = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": initial_response},
                {"role": "user", "content": (
                    f"Self-critique:\n{critique}\n\n"
                    f"Now provide your FINAL refined recommendation "
                    f"addressing the critique points."
                )}
            ],
            temperature=0.6
        )

        return {
            "initial_analysis": initial_response,
            "self_critique": critique,
            "final_recommendation": (
                refined.choices[0].message.content
            ),
            "agent": self.name
        }

    def request_collaboration(self, target_agent: str,
                               request: dict,
                               correlation_id: str) -> dict:
        """Ask another agent for help"""
        message = AgentMessage(
            sender=self.name,
            recipient=target_agent,
            message_type=MessageType.REQUEST,
            content=request,
            requires_response=True,
            correlation_id=correlation_id
        )
        return self.message_bus.publish(message)

    def escalate_to_orchestrator(self, issue: str,
                                  context: dict) -> dict:
        """Escalate a problem to the orchestrator"""
        message = AgentMessage(
            sender=self.name,
            recipient="orchestrator",
            message_type=MessageType.ESCALATE,
            content={"issue": issue, "context": context},
            priority=1  # High priority
        )
        return self.message_bus.publish(message)

    def learn_from_feedback(self, feedback: dict) -> dict:
        """Process feedback to improve future performance"""
        self.confidence_history.append(feedback)
        logger.info(
            f"   📚 {self.name} learned from feedback: "
            f"{feedback.get('summary', 'N/A')}"
        )
        return {"status": "learned"}

    def update_context(self, info: dict) -> dict:
        """Receive contextual updates from other agents"""
        logger.info(
            f"   📥 {self.name} received context update"
        )
        return {"status": "context_updated"}


class EnhancedFlightAgent(EnhancedSpecializedAgent):
    """Flight specialist with advanced analysis"""

    def __init__(self, client: OpenAI, api: SkyscannerAPI,
                 message_bus: MessageBus):
        super().__init__(
            name="flight_agent",
            role="""You are an elite Flight Search Specialist with 
deep knowledge of:
- Airline pricing patterns and when to book
- Layover optimization (which airports are best for connections)
- Comfort vs. cost tradeoffs
- Seasonal pricing trends
- Alliance and codeshare benefits

Analyze flight options considering:
1. Total journey time (not just flight time)
2. Arrival time at destination (jet lag optimization)  
3. Airline reliability and service quality
4. Baggage policies
5. Cancellation flexibility

Always provide a clear ranking with reasoning.""",
            client=client,
            message_bus=message_bus
        )
        self.api = api

    def process_request(self, request: dict) -> dict:
        """Handle flight search requests"""
        origin = request.get("origin", "")
        destination = request.get("destination", "")
        date = request.get("date", "")
        budget = request.get("budget", 0)
        preferences = request.get("preferences", {})

        # Search flights
        results = self.api.search(origin, destination, date)

        # Ask hotel agent about location for arrival time optimization
        hotel_input = self.request_collaboration(
            "hotel_agent",
            {
                "question": (
                    f"What area of {destination} should I "
                    f"optimize arrival for?"
                ),
                "context": {
                    "hotel_preference": preferences.get(
                        "hotel_preference", "central"
                    )
                }
            },
            correlation_id=f"trip_{date}"
        )

        # Deep analysis with self-reflection
        analysis = self.think(
            prompt=(
                f"Analyze these flights from {origin} to "
                f"{destination} on {date}.\n"
                f"Budget: ${budget}\n"
                f"Preferences: {json.dumps(preferences)}\n"
                f"Hotel location hint: "
                f"{json.dumps(hotel_input)}\n"
                f"Flights: {json.dumps(results['flights'][:5])}\n\n"
                f"Rank them and explain your reasoning."
            ),
            context=f"Source: {results.get('source', 'unknown')}"
        )

        return {
            "flights": results["flights"],
            "analysis": analysis,
            "source": results["source"],
            "agent": self.name
        }


class EnhancedHotelAgent(EnhancedSpecializedAgent):
    """Hotel specialist with neighborhood expertise"""

    def __init__(self, client: OpenAI, api: BookingComAPI,
                 message_bus: MessageBus):
        super().__init__(
            name="hotel_agent",
            role="""You are an elite Hotel & Accommodation Specialist 
with deep knowledge of:
- Neighborhood character and walkability
- Hotel categories (boutique, business, capsule, ryokan, etc.)
- Price-to-value optimization
- Proximity to attractions, food, and transit
- Seasonal pricing patterns
- Hidden gems vs. tourist traps

Consider:
1. Walking distance to key attractions
2. Nearest train/metro station
3. Neighborhood safety and vibe
4. Check-in/check-out flexibility
5. Unique local accommodation types

Provide neighborhood insights with each recommendation.""",
            client=client,
            message_bus=message_bus
        )
        self.api = api

    def process_request(self, request: dict) -> dict:
        """Handle hotel search requests"""
        city = request.get("city", "")
        checkin = request.get("checkin", "")
        nights = request.get("nights", 1)
        budget_remaining = request.get("budget_remaining", 0)
        preferences = request.get("preferences", {})

        results = self.api.search(city, checkin, nights)

        # Collaborate with food agent for restaurant proximity
        food_input = self.request_collaboration(
            "food_agent",
            {
                "question": (
                    f"Which areas of {city} have the best "
                    f"food scene?"
                ),
                "cuisine": preferences.get("cuisine", "local")
            },
            correlation_id=f"trip_{checkin}"
        )

        analysis = self.think(
            prompt=(
                f"Analyze hotel options in {city} for "
                f"{nights} nights.\n"
                f"Remaining budget: ${budget_remaining}\n"
                f"Preferences: {json.dumps(preferences)}\n"
                f"Food area recommendations: "
                f"{json.dumps(food_input)}\n"
                f"Hotels: "
                f"{json.dumps(results['hotels'][:5])}\n\n"
                f"Rank them considering neighborhood quality "
                f"and proximity to great food."
            ),
            context=f"Source: {results.get('source', 'unknown')}"
        )

        return {
            "hotels": results["hotels"],
            "analysis": analysis,
            "source": results["source"],
            "agent": self.name
        }


class EnhancedFoodAgent(EnhancedSpecializedAgent):
    """Food specialist with culinary expertise"""

    def __init__(self, client: OpenAI, api: YelpAPI,
                 message_bus: MessageBus):
        super().__init__(
            name="food_agent",
            role="""You are an elite Food & Culinary Specialist 
with deep knowledge of:
- Local cuisine and must-try dishes
- Hidden gems vs. tourist restaurants
- Street food scenes and food markets
- Dietary accommodation options
- Meal timing and local dining customs
- Price ranges from budget to fine dining
- Seasonal specialties and food festivals

Create meal plans that:
1. Mix iconic spots with hidden gems
2. Balance budget across meals (cheap lunch, nice dinner)
3. Include local food customs and etiquette tips
4. Account for dietary restrictions
5. Suggest food experiences (cooking classes, market tours)

Make the food plan EXCITING — this is the highlight!""",
            client=client,
            message_bus=message_bus
        )
        self.api = api

    def process_request(self, request: dict) -> dict:
        """Handle food planning requests"""
        city = request.get("city", "")
        cuisine = request.get("cuisine", "local food")
        days = request.get("days", 1)
        budget_remaining = request.get("budget_remaining", 0)
        preferences = request.get("preferences", {})
        hotel_area = request.get("hotel_area", "")

        results = self.api.search(city, cuisine)

        analysis = self.think(
            prompt=(
                f"Create an epic {days}-day food plan for "
                f"{city}.\n"
                f"Food budget: ${budget_remaining}\n"
                f"Staying in: {hotel_area}\n"
                f"Preferences: {json.dumps(preferences)}\n"
                f"Dietary restrictions: "
                f"{preferences.get('dietary_restrictions', 'none')}\n"
                f"Available restaurants: "
                f"{json.dumps(results['restaurants'][:8])}\n\n"
                f"Create a meal-by-meal plan with costs, "
                f"tips, and must-try dishes."
            ),
            context=(
                f"Source: {results.get('source', 'unknown')}"
            )
        )

        return {
            "restaurants": results["restaurants"],
            "analysis": analysis,
            "source": results["source"],
            "agent": self.name
        }


# ─────────────────────────────────────────
# 3C. ENHANCED ORCHESTRATOR
# ─────────────────────────────────────────

class EnhancedOrchestrator:
    """
    The brain that coordinates all agents.
    
    Orchestration Patterns:
    
    1. Sequential:  Flight → Hotel → Food (dependencies)
    2. Parallel:    Hotel + Food search (independent)
    3. Iterative:   Re-search if over budget
    4. Escalation:  Agent can't decide → ask user
    """

    def __init__(self, client: OpenAI,
                 flight_api: SkyscannerAPI,
                 hotel_api: BookingComAPI,
                 yelp_api: YelpAPI):
        self.client = client
        self.message_bus = MessageBus()

        # Create specialized agents
        self.flight_agent = EnhancedFlightAgent(
            client, flight_api, self.message_bus
        )
        self.hotel_agent = EnhancedHotelAgent(
            client, hotel_api, self.message_bus
        )
        self.food_agent = EnhancedFoodAgent(
            client, yelp_api, self.message_bus
        )

        # Register orchestrator on message bus
        self.message_bus.subscribe(
            "orchestrator", self.handle_escalation
        )

    def handle_escalation(self, message: AgentMessage) -> dict:
        """Handle escalations from specialized agents"""
        logger.warning(
            f"   🚨 Escalation from {message.sender}: "
            f"{message.content.get('issue', 'Unknown')}"
        )
        return {
            "action": "acknowledged",
            "response": "Orchestrator will handle this"
        }

    def coordinate_trip_planning(
        self, origin: str, destination: str,
        date: str, days: int, budget: float,
        preferences: dict
    ) -> dict:
        """
        Main coordination logic.
        Manages the flow between all agents.
        """

        correlation_id = f"trip_{destination}_{date}"
        budget_tracker = {"total": budget, "spent": 0}

        print(f"\n🧠 Orchestrator: Starting coordination...")
        print(f"   Correlation ID: {correlation_id}")

        # ═══════════════════════════════════
        # PHASE 1: Parallel Information Gathering
        # ═══════════════════════════════════
        print(f"\n   📡 Phase 1: Gathering intelligence...")

        # These could run in parallel with asyncio
        flight_results = self.message_bus.publish(AgentMessage(
            sender="orchestrator",
            recipient="flight_agent",
            message_type=MessageType.REQUEST,
            content={
                "origin": origin,
                "destination": destination,
                "date": date,
                "budget": budget,
                "preferences": preferences
            },
            correlation_id=correlation_id
        ))

        # ═══════════════════════════════════
        # PHASE 2: Sequential Planning
        # (Hotel depends on flight cost)
        # ═══════════════════════════════════
        print(f"\n   🔗 Phase 2: Sequential planning...")

        # After flight is selected, update budget
        selected_flight_cost = flight_results.get(
            "flights", [{}]
        )[0].get("price", 0)
        budget_tracker["spent"] += selected_flight_cost

        hotel_results = self.message_bus.publish(AgentMessage(
            sender="orchestrator",
            recipient="hotel_agent",
            message_type=MessageType.REQUEST,
            content={
                "city": destination,
                "checkin": date,
                "nights": days,
                "budget_remaining": (
                    budget_tracker["total"]
                    - budget_tracker["spent"]
                ),
                "preferences": preferences
            },
            correlation_id=correlation_id
        ))

        # ═══════════════════════════════════
        # PHASE 3: Food Planning
        # (Depends on hotel location)
        # ═══════════════════════════════════
        print(f"\n   🍽️ Phase 3: Food planning...")

        selected_hotel = hotel_results.get("hotels", [{}])[0]
        budget_tracker["spent"] += selected_hotel.get(
            "total_cost", 0
        )

        food_results = self.message_bus.publish(AgentMessage(
            sender="orchestrator",
            recipient="food_agent",
            message_type=MessageType.REQUEST,
            content={
                "city": destination,
                "cuisine": preferences.get("cuisine", "local"),
                "days": days,
                "budget_remaining": (
                    budget_tracker["total"]
                    - budget_tracker["spent"]
                ),
                "preferences": preferences,
                "hotel_area": selected_hotel.get("area", "")
            },
            correlation_id=correlation_id
        ))

        # ═══════════════════════════════════
        # PHASE 4: Synthesis
        # ═══════════════════════════════════
        print(f"\n   🔄 Phase 4: Synthesizing results...")

        # Get the full conversation from message bus
        conversation = self.message_bus.get_conversation(
            correlation_id
        )

        return {
            "flights": flight_results,
            "hotels": hotel_results,
            "food": food_results,
            "budget": budget_tracker,
            "message_log": conversation,
            "correlation_id": correlation_id
        }
```

### Agent Communication Flow

```
                    ORCHESTRATOR
                    ┌──────────┐
                    │ Coordinate│
                    └─┬──┬──┬──┘
         ┌───────────┘  │  └───────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Flight  │   │ Hotel   │   │  Food   │
    │  Agent  │   │  Agent  │   │  Agent  │
    └────┬────┘   └────┬────┘   └────┬────┘
         │              │              │
         │  ┌───────────┘              │
         │  │ "What area is           │
         │  │  best for food?"  ┌─────┘
         │  └──────────────────►│
         │                      │
         │  "Shinjuku has the  │
         │   best food scene"  │
         │◄─────────────────────┘
         │
         │  "Optimize arrival
         │   for Shinjuku area"
         ▼
    ┌─────────┐
    │ Select  │
    │ evening │
    │ flight  │
    └─────────┘
    
Agents COLLABORATE to make better decisions!
```

---

# 4️⃣ DEEP DIVE: Human-in-the-Loop

## The Spectrum of Human Control

```
┌──────────────────────────────────────────────────────────┐
│            HUMAN CONTROL SPECTRUM                        │
├────────────┬────────────┬────────────┬──────────────────┤
│  FULL AUTO │  APPROVAL  │ COLLAB     │  FULL MANUAL     │
│            │  GATES     │ PARTNER    │                  │
├────────────┼────────────┼────────────┼──────────────────┤
│ Agent does │ Agent does │ Agent      │ Human does       │
│ everything │ work,      │ suggests,  │ everything,      │
│ No human   │ human      │ human      │ agent assists    │
│ input      │ approves   │ co-decides │                  │
├────────────┼────────────┼────────────┼──────────────────┤
│  ❌ Risky  │ ✅ Best    │ ✅ Good   │ ❌ Slow          │
│  for trips │ balance    │ for        │ defeats the      │
│            │            │ luxury     │ purpose          │
└────────────┴────────────┴────────────┴──────────────────┘

Our implementation uses "APPROVAL GATES" — the sweet spot!
```

## Advanced Human-in-the-Loop System

```python
from enum import Enum
from typing import Optional, Callable


class ApprovalLevel(Enum):
    """Define when human input is needed"""
    AUTO = "auto"              # No confirmation needed
    NOTIFY = "notify"          # Inform but don't wait
    CONFIRM = "confirm"        # Must approve
    COLLABORATE = "collaborate" # Co-decide with options


class ApprovalPolicy:
    """
    Configurable approval rules.
    Define WHEN the agent must ask for human input.
    """

    def __init__(self):
        self.rules = {}
        self._set_defaults()

    def _set_defaults(self):
        """Default approval requirements"""
        self.rules = {
            # Action: (approval_level, condition)
            "search_flights": ApprovalLevel.AUTO,
            "search_hotels": ApprovalLevel.AUTO,
            "search_restaurants": ApprovalLevel.AUTO,
            "select_flight": ApprovalLevel.COLLABORATE,
            "select_hotel": ApprovalLevel.COLLABORATE,
            "select_restaurants": ApprovalLevel.NOTIFY,
            "confirm_booking": ApprovalLevel.CONFIRM,
            "finalize_itinerary": ApprovalLevel.CONFIRM,
            "exceed_budget": ApprovalLevel.CONFIRM,
            "change_destination": ApprovalLevel.CONFIRM,
        }

        # Cost-based rules
        self.cost_thresholds = {
            "auto_approve_under": 50,       # < $50 → auto
            "notify_under": 200,            # < $200 → notify
            "confirm_over": 200,            # > $200 → confirm
            "block_over": 5000,             # > $5000 → block
        }

    def get_approval_level(self, action: str,
                           cost: float = 0) -> ApprovalLevel:
        """Determine what approval level is needed"""
        # Check cost-based rules first
        if cost > self.cost_thresholds["block_over"]:
            return ApprovalLevel.CONFIRM
        if cost > self.cost_thresholds["confirm_over"]:
            return ApprovalLevel.CONFIRM
        if cost > self.cost_thresholds["notify_under"]:
            return ApprovalLevel.NOTIFY

        # Fall back to action-based rules
        return self.rules.get(action, ApprovalLevel.CONFIRM)


class EnhancedHumanInTheLoop:
    """
    Sophisticated human interaction system with:
    - Configurable approval policies
    - Rich option presentation
    - Modification support
    - Feedback collection
    - Undo capability
    """

    def __init__(self, policy: ApprovalPolicy = None,
                 auto_approve: bool = False,
                 callback: Callable = None):
        self.policy = policy or ApprovalPolicy()
        self.auto_approve = auto_approve
        self.callback = callback  # For web UI integration
        self.decisions = []
        self.undo_stack = []

    def request_decision(
        self,
        action: str,
        options: list,
        context: dict,
        cost: float = 0
    ) -> dict:
        """
        Main entry point for human decisions.
        Automatically determines what level of
        interaction is needed.
        """
        level = self.policy.get_approval_level(action, cost)

        if self.auto_approve or level == ApprovalLevel.AUTO:
            # Auto-select the best option (first one)
            selected = options[0] if options else None
            self._log_decision(action, selected, "auto")
            return {
                "selected": selected,
                "approval": "auto",
                "index": 0
            }

        elif level == ApprovalLevel.NOTIFY:
            # Show what was selected but don't wait for input
            selected = options[0] if options else None
            self._notify(action, selected, context)
            self._log_decision(action, selected, "notified")
            return {
                "selected": selected,
                "approval": "notified",
                "index": 0
            }

        elif level == ApprovalLevel.COLLABORATE:
            # Present options and get user choice
            return self._collaborate(action, options, context)

        elif level == ApprovalLevel.CONFIRM:
            # Present the plan and get explicit approval
            return self._confirm(action, options, context)

    def _collaborate(self, action: str, options: list,
                     context: dict) -> dict:
        """Interactive option selection"""
        budget_remaining = context.get("budget_remaining", 0)
        category = context.get("category", action)

        print(f"\n{'═' * 55}")
        print(f"🙋 YOUR CHOICE NEEDED — {category.upper()}")
        print(f"{'═' * 55}")
        print(f"💰 Budget Remaining: ${budget_remaining:,.2f}")

        # Show comparison table
        self._show_comparison_table(category, options)

        # Show agent's recommendation
        recommendation = context.get("recommendation", "")
        if recommendation:
            print(f"\n🤖 Agent's Recommendation:")
            print(f"   {recommendation}")

        # Get user choice
        if self.auto_approve:
            choice_idx = 0
            print(f"\n🤖 Auto-selected: Option 1")
        else:
            choice_idx = self._get_choice(
                len(options),
                allow_none=True,
                allow_modify=True
            )

        if choice_idx == -1:
            return {"selected": None, "approval": "rejected"}
        if choice_idx == -2:
            return {"selected": None, "approval": "modify"}

        selected = options[choice_idx]
        self._log_decision(action, selected, "user_selected")

        return {
            "selected": selected,
            "approval": "selected",
            "index": choice_idx
        }

    def _confirm(self, action: str, options: list,
                 context: dict) -> dict:
        """Explicit confirmation required"""
        item = options[0] if options else context

        print(f"\n{'═' * 55}")
        print(f"🛡️ CONFIRMATION REQUIRED — {action.upper()}")
        print(f"{'═' * 55}")

        # Show details
        if isinstance(item, dict):
            for key, value in item.items():
                if key != "raw_data":
                    print(f"   • {key}: {value}")

        # Show warnings if any
        warnings = context.get("warnings", [])
        for w in warnings:
            print(f"   ⚠️ {w}")

        if self.auto_approve:
            print(f"\n🤖 Auto-confirmed")
            self._log_decision(action, item, "auto_confirmed")
            return {
                "selected": item,
                "approval": "confirmed"
            }

        while True:
            response = input(
                "\n👉 Confirm? (yes / no / details / modify): "
            ).strip().lower()

            if response in ["yes", "y"]:
                self._log_decision(action, item, "confirmed")
                return {
                    "selected": item,
                    "approval": "confirmed"
                }
            elif response in ["no", "n"]:
                self._log_decision(action, item, "rejected")
                return {
                    "selected": None,
                    "approval": "rejected"
                }
            elif response in ["details", "d"]:
                self._show_details(item)
            elif response in ["modify", "m"]:
                return {
                    "selected": None,
                    "approval": "modify"
                }

    def _show_comparison_table(self, category: str,
                                options: list):
        """Display options in a formatted comparison table"""

        if category in ["flights", "select_flight"]:
            print(f"\n   {'#':<4} {'Airline':<15} {'Price':<10} "
                  f"{'Duration':<12} {'Stops':<8}")
            print(f"   {'─'*4} {'─'*15} {'─'*10} {'─'*12} {'─'*8}")
            for i, opt in enumerate(options, 1):
                print(
                    f"   [{i}]  {opt.get('airline', 'N/A'):<15} "
                    f"${opt.get('price', 0):<9} "
                    f"{opt.get('duration', opt.get('duration_display', 'N/A')):<12} "
                    f"{opt.get('stops', 0)}"
                )

        elif category in ["hotels", "select_hotel"]:
            print(f"\n   {'#':<4} {'Hotel':<25} {'$/Night':<10} "
                  f"{'Total':<10} {'Rating':<8} {'Area':<15}")
            print(f"   {'─'*4} {'─'*25} {'─'*10} "
                  f"{'─'*10} {'─'*8} {'─'*15}")
            for i, opt in enumerate(options, 1):
                name = opt.get('name', 'N/A')[:24]
                print(
                    f"   [{i}]  {name:<25} "
                    f"${opt.get('price_per_night', 0):<9} "
                    f"${opt.get('total_cost', 0):<9} "
                    f"⭐{opt.get('rating', 0):<6} "
                    f"{opt.get('area', 'N/A'):<15}"
                )

        elif category in ["restaurants", "select_restaurants"]:
            print(f"\n   {'#':<4} {'Restaurant':<25} "
                  f"{'Cuisine':<20} {'$/Meal':<10} {'Rating':<8}")
            print(f"   {'─'*4} {'─'*25} {'─'*20} "
                  f"{'─'*10} {'─'*8}")
            for i, opt in enumerate(options, 1):
                name = opt.get('name', 'N/A')[:24]
                cuisine = opt.get(
                    'type', opt.get('cuisine_type', 'N/A')
                )[:19]
                print(
                    f"   [{i}]  {name:<25} {cuisine:<20} "
                    f"~${opt.get('avg_meal_cost', 0):<8} "
                    f"⭐{opt.get('rating', 0)}"
                )

    def _get_choice(self, max_options: int,
                    allow_none: bool = True,
                    allow_modify: bool = True) -> int:
        """Get validated user choice with extra options"""
        extra = []
        if allow_none:
            extra.append("'skip'")
        if allow_modify:
            extra.append("'modify'")
        extra_str = f" or {', '.join(extra)}" if extra else ""

        while True:
            raw = input(
                f"\n👉 Choice (1-{max_options}{extra_str}): "
            ).strip().lower()

            if raw in ["skip", "s", "none", "n"] and allow_none:
                return -1
            if raw in ["modify", "m", "change"] and allow_modify:
                return -2

            try:
                choice = int(raw)
                if 1 <= choice <= max_options:
                    return choice - 1
                print(f"   Enter 1-{max_options}")
            except ValueError:
                print(f"   Invalid input")

    def _show_details(self, item: dict):
        """Show detailed information about an option"""
        print(f"\n   📋 Full Details:")
        print(f"   {'─' * 40}")
        for key, value in item.items():
            if key == "raw_data":
                continue
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"   {key}: {value}")

    def _notify(self, action: str, selected: dict,
                context: dict):
        """Show notification without waiting"""
        name = "Unknown"
        if selected:
            name = selected.get(
                "name", selected.get("airline", "Unknown")
            )
        print(f"\n   ℹ️  Auto-selected for {action}: {name}")

    def _log_decision(self, action: str, selected: dict,
                      method: str):
        """Log all decisions for audit trail"""
        decision = {
            "action": action,
            "selected": selected,
            "method": method,
            "timestamp": datetime.now().isoformat()
        }
        self.decisions.append(decision)
        self.undo_stack.append(decision)

    def undo_last_decision(self) -> Optional[dict]:
        """Undo the last decision"""
        if self.undo_stack:
            undone = self.undo_stack.pop()
            if undone in self.decisions:
                self.decisions.remove(undone)
            print(f"   ↩️ Undone: {undone['action']}")
            return undone
        print("   Nothing to undo")
        return None

    def collect_feedback(self, trip_id: str) -> dict:
        """Post-trip feedback collection"""
        print(f"\n{'═' * 55}")
        print(f"📝 TRIP FEEDBACK — Help me improve!")
        print(f"{'═' * 55}")

        if self.auto_approve:
            return {
                "trip_id": trip_id,
                "overall": 5,
                "flight": 5,
                "hotel": 5,
                "food": 5,
                "comments": "Auto-feedback"
            }

        def get_rating(prompt: str) -> int:
            while True:
                try:
                    r = int(input(f"   {prompt} (1-5 ⭐): "))
                    if 1 <= r <= 5:
                        return r
                except ValueError:
                    pass
                print("   Enter 1-5")

        feedback = {
            "trip_id": trip_id,
            "overall": get_rating("Overall experience"),
            "flight": get_rating("Flight selection"),
            "hotel": get_rating("Hotel selection"),
            "food": get_rating("Food recommendations"),
            "comments": input(
                "   Any comments? (press Enter to skip): "
            ).strip()
        }

        print(f"\n   🙏 Thanks for the feedback!")
        return feedback

    def get_decision_summary(self) -> str:
        """Get a summary of all decisions made"""
        if not self.decisions:
            return "No decisions recorded."

        lines = ["📊 Decision Summary:"]
        for d in self.decisions:
            lines.append(
                f"   • {d['action']} → {d['method']} "
                f"at {d['timestamp']}"
            )
        return "\n".join(lines)
```

### Decision Flow

```
User Request Arrives
        │
        ▼
┌────────────────────┐
│  Check Approval    │
│  Policy            │
└────────┬───────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌──────┐ ┌───────┐ ┌─────────┐ ┌─────────┐
│ AUTO │ │NOTIFY │ │COLLAB   │ │CONFIRM  │
│      │ │       │ │         │ │         │
│Select│ │Select │ │Present  │ │Show     │
│best  │ │best,  │ │options, │ │plan,    │
│auto  │ │inform │ │user     │ │require  │
│      │ │user   │ │picks    │ │explicit │
│      │ │       │ │         │ │yes/no   │
└──┬───┘ └──┬────┘ └──┬──────┘ └──┬──────┘
   │        │         │           │
   └────────┴────┬────┴───────────┘
                 │
                 ▼
          ┌──────────────┐
          │ LOG DECISION │
          │ + UNDO STACK │
          └──────────────┘
```

---

# 5️⃣ DEEP DIVE: Guardrails System

## Why Guardrails Are Critical

```
Without Guardrails:
───────────────────
🤖 "I found a $15,000/night hotel — booking now!"
🤖 "I booked 3 flights for you!"
🤖 "Here's a trip to a war zone!"
🤖 Budget? What budget?

With Guardrails:
───────────────────
🛡️ "That hotel exceeds the $500/night threshold. Flagged."
🛡️ "Booking requires your explicit approval first."
🛡️ "Travel advisory warning for this destination."
🛡️ "You're at 85% of budget. Adjusting remaining plans."
```

## Comprehensive Guardrails System

```python
from enum import Enum
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCK = "block"
    MODIFY = "modify"


@dataclass
class GuardrailResult:
    """Result of a guardrail check"""
    action: GuardrailAction
    risk_level: RiskLevel
    message: str
    details: dict = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    rule_name: str = ""


class ComprehensiveGuardrails:
    """
    Multi-layer safety system:
    
    ┌─────────────────────────────────────────┐
    │          GUARDRAIL LAYERS               │
    ├─────────────────────────────────────────┤
    │                                         │
    │  Layer 1: INPUT VALIDATION              │
    │  → Validate all inputs before processing│
    │                                         │
    │  Layer 2: BUDGET PROTECTION             │
    │  → Track spending, prevent overruns     │
    │                                         │
    │  Layer 3: SAFETY CHECKS                 │
    │  → Travel advisories, destination safety│
    │                                         │
    │  Layer 4: CONTENT FILTERING             │
    │  → Ensure appropriate recommendations   │
    │                                         │
    │  Layer 5: ACTION AUTHORIZATION          │
    │  → Verify before any booking/commitment │
    │                                         │
    │  Layer 6: RATE LIMITING                 │
    │  → Prevent excessive API calls          │
    │                                         │
    │  Layer 7: AUDIT LOGGING                 │
    │  → Complete record of all actions       │
    │                                         │
    └─────────────────────────────────────────┘
    """

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.budget_tracker = BudgetGuardrail()
        self.safety_checker = SafetyGuardrail()
        self.rate_limiter = RateLimitGuardrail()
        self.audit_log = AuditLog()

    def _default_config(self) -> dict:
        return {
            "max_total_budget": 50000,
            "max_flight_cost": 5000,
            "max_hotel_per_night": 1500,
            "max_meal_cost": 500,
            "max_trip_days": 30,
            "require_confirmation_above": 200,
            "block_above": 10000,
            "max_api_calls_per_minute": 30,
            "allowed_currencies": ["USD", "EUR", "GBP", "JPY"],
        }

    # ─────────────────────────────────────────
    # LAYER 1: INPUT VALIDATION
    # ─────────────────────────────────────────

    def validate_trip_request(
        self, origin: str, destination: str,
        date: str, days: int, budget: float
    ) -> List[GuardrailResult]:
        """Validate all trip parameters"""
        results = []

        # Origin validation
        if not origin or len(origin.strip()) < 2:
            results.append(GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                message="Invalid origin city",
                rule_name="input_origin"
            ))

        # Destination validation
        if not destination or len(destination.strip()) < 2:
            results.append(GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                message="Invalid destination city",
                rule_name="input_destination"
            ))

        # Date validation
        try:
            trip_date = datetime.strptime(date, "%Y-%m-%d")
            if trip_date < datetime.now():
                results.append(GuardrailResult(
                    action=GuardrailAction.WARN,
                    risk_level=RiskLevel.MEDIUM,
                    message="Trip date is in the past",
                    suggestions=["Update to a future date"],
                    rule_name="input_date_past"
                ))
            if trip_date > datetime.now() + timedelta(days=365):
                results.append(GuardrailResult(
                    action=GuardrailAction.WARN,
                    risk_level=RiskLevel.LOW,
                    message=(
                        "Trip is more than a year away. "
                        "Prices may change significantly."
                    ),
                    rule_name="input_date_far"
                ))
        except ValueError:
            results.append(GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                message="Invalid date format. Use YYYY-MM-DD",
                rule_name="input_date_format"
            ))

        # Duration validation
        if days < 1:
            results.append(GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                message="Trip must be at least 1 day",
                rule_name="input_days_min"
            ))
        elif days > self.config["max_trip_days"]:
            results.append(GuardrailResult(
                action=GuardrailAction.REQUIRE_CONFIRMATION,
                risk_level=RiskLevel.MEDIUM,
                message=(
                    f"Trip duration ({days} days) exceeds "
                    f"typical maximum "
                    f"({self.config['max_trip_days']} days)"
                ),
                rule_name="input_days_max"
            ))

        # Budget validation
        if budget <= 0:
            results.append(GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                message="Budget must be positive",
                rule_name="input_budget_positive"
            ))
        elif budget > self.config["max_total_budget"]:
            results.append(GuardrailResult(
                action=GuardrailAction.REQUIRE_CONFIRMATION,
                risk_level=RiskLevel.MEDIUM,
                message=(
                    f"Budget ${budget:,.0f} is very high. "
                    f"Please confirm."
                ),
                rule_name="input_budget_high"
            ))
        elif budget < 50 * days:
            results.append(GuardrailResult(
                action=GuardrailAction.WARN,
                risk_level=RiskLevel.MEDIUM,
                message=(
                    f"Budget of ${budget/days:.0f}/day "
                    f"may be insufficient for "
                    f"{destination}"
                ),
                suggestions=[
                    "Consider hostels and street food",
                    "Look for budget airlines"
                ],
                rule_name="input_budget_low"
            ))

        # Safety check
        safety_result = self.safety_checker.check_destination(
            destination
        )
        if safety_result:
            results.append(safety_result)

        # If no issues found
        if not results:
            results.append(GuardrailResult(
                action=GuardrailAction.ALLOW,
                risk_level=RiskLevel.LOW,
                message="All inputs validated ✅",
                rule_name="input_all_valid"
            ))

        # Log all checks
        for r in results:
            self.audit_log.log(
                "input_validation", r.rule_name,
                r.action.value, r.message
            )

        return results

    # ─────────────────────────────────────────
    # LAYER 2: BUDGET PROTECTION
    # ─────────────────────────────────────────

    def check_booking_cost(
        self, booking_type: str, cost: float,
        budget_remaining: float
    ) -> GuardrailResult:
        """Check if a booking cost is acceptable"""
        return self.budget_tracker.check(
            booking_type, cost, budget_remaining, self.config
        )

    # ─────────────────────────────────────────
    # LAYER 5: ACTION AUTHORIZATION
    # ─────────────────────────────────────────

    def authorize_action(
        self, action: str, params: dict
    ) -> GuardrailResult:
        """
        Pre-authorize any agent action.
        Called BEFORE the action is executed.
        """

        # Check rate limits
        rate_check = self.rate_limiter.check(action)
        if rate_check.action == GuardrailAction.BLOCK:
            return rate_check

        # Check required parameters
        required_params = {
            "book_flight": ["airline", "price", "date"],
            "book_hotel": ["name", "total_cost", "checkin"],
            "book_restaurant": ["name"],
            "send_email": ["recipient", "subject"],
            "make_payment": ["amount", "method"],
        }

        if action in required_params:
            missing = [
                p for p in required_params[action]
                if p not in params or not params[p]
            ]
            if missing:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    risk_level=RiskLevel.HIGH,
                    message=(
                        f"Missing required parameters: "
                        f"{', '.join(missing)}"
                    ),
                    rule_name=f"auth_missing_{action}"
                )

        # Dangerous actions always need confirmation
        dangerous_actions = [
            "make_payment", "delete_booking",
            "share_personal_info"
        ]
        if action in dangerous_actions:
            return GuardrailResult(
                action=GuardrailAction.REQUIRE_CONFIRMATION,
                risk_level=RiskLevel.CRITICAL,
                message=(
                    f"Action '{action}' requires explicit "
                    f"user confirmation"
                ),
                rule_name=f"auth_dangerous_{action}"
            )

        self.audit_log.log(
            "authorization", action, "allowed",
            f"Params: {list(params.keys())}"
        )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            risk_level=RiskLevel.LOW,
            message=f"Action '{action}' authorized",
            rule_name=f"auth_{action}"
        )

    def get_full_report(self) -> str:
        """Generate a complete guardrail report"""
        return self.audit_log.generate_report()


class BudgetGuardrail:
    """Dedicated budget protection system"""

    def __init__(self):
        self.allocations = {}
        self.spent = {}
        self.warnings_issued = []

    def set_budget(self, total: float, allocation: dict = None):
        """
        Set total budget with optional allocation.
        
        Default allocation:
        - Flights: 35%
        - Hotels: 35% 
        - Food: 20%
        - Activities: 10%
        """
        default_allocation = {
            "flight": 0.35,
            "hotel": 0.35,
            "food": 0.20,
            "activities": 0.10,
        }
        allocation = allocation or default_allocation

        self.allocations = {
            k: total * v for k, v in allocation.items()
        }
        self.spent = {k: 0 for k in self.allocations}
        self.total = total

    def check(self, category: str, cost: float,
              budget_remaining: float,
              config: dict) -> GuardrailResult:
        """Check if a cost is within acceptable bounds"""

        # ── Absolute limit check ──
        limits = {
            "flight": config.get("max_flight_cost", 5000),
            "hotel_per_night": config.get(
                "max_hotel_per_night", 1500
            ),
            "restaurant": config.get("max_meal_cost", 500),
        }

        limit = limits.get(category, 10000)
        if cost > limit:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.CRITICAL,
                message=(
                    f"💰 {category} cost ${cost:,.2f} exceeds "
                    f"limit ${limit:,.2f}"
                ),
                suggestions=[
                    f"Look for {category} under ${limit:,.0f}"
                ],
                rule_name=f"budget_absolute_{category}"
            )

        # ── Budget remaining check ──
        if cost > budget_remaining:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                message=(
                    f"💰 Cost ${cost:,.2f} exceeds remaining "
                    f"budget ${budget_remaining:,.2f}"
                ),
                suggestions=[
                    "Choose a cheaper option",
                    "Increase total budget",
                    "Reduce spending elsewhere"
                ],
                rule_name=f"budget_exceeded_{category}"
            )

        # ── High percentage warning ──
        if budget_remaining > 0:
            pct = cost / budget_remaining
            if pct > 0.6:
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    risk_level=RiskLevel.MEDIUM,
                    message=(
                        f"💰 This {category} would use "
                        f"{pct:.0%} of remaining budget"
                    ),
                    details={
                        "cost": cost,
                        "remaining_after": budget_remaining - cost,
                        "percentage": pct
                    },
                    rule_name=f"budget_high_pct_{category}"
                )

        # ── Confirmation threshold ──
        if cost > config.get("require_confirmation_above", 200):
            return GuardrailResult(
                action=GuardrailAction.REQUIRE_CONFIRMATION,
                risk_level=RiskLevel.LOW,
                message=(
                    f"💰 {category} costs ${cost:,.2f} — "
                    f"confirmation needed"
                ),
                rule_name=f"budget_confirm_{category}"
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            risk_level=RiskLevel.LOW,
            message=f"✅ {category} cost ${cost:,.2f} approved",
            rule_name=f"budget_ok_{category}"
        )


class SafetyGuardrail:
    """Travel safety checks"""

    def __init__(self):
        # In production, integrate with real advisory APIs
        self.advisories = {
            "North Korea": {
                "level": "do_not_travel",
                "risk": RiskLevel.CRITICAL
            },
            "Syria": {
                "level": "do_not_travel",
                "risk": RiskLevel.CRITICAL
            },
            "Afghanistan": {
                "level": "do_not_travel",
                "risk": RiskLevel.CRITICAL
            },
            "Ukraine": {
                "level": "reconsider",
                "risk": RiskLevel.HIGH
            },
            "Venezuela": {
                "level": "reconsider",
                "risk": RiskLevel.HIGH
            },
            "Mexico": {
                "level": "exercise_caution",
                "risk": RiskLevel.MEDIUM
            },
        }

    def check_destination(
        self, destination: str
    ) -> Optional[GuardrailResult]:
        """Check travel advisories for destination"""
        advisory = self.advisories.get(destination)

        if not advisory:
            return None

        level = advisory["level"]
        risk = advisory["risk"]

        if level == "do_not_travel":
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.CRITICAL,
                message=(
                    f"🚨 DO NOT TRAVEL advisory for "
                    f"{destination}. Planning blocked."
                ),
                suggestions=[
                    "Check official travel advisories",
                    "Consider alternative destinations"
                ],
                rule_name=f"safety_block_{destination}"
            )
        elif level == "reconsider":
            return GuardrailResult(
                action=GuardrailAction.REQUIRE_CONFIRMATION,
                risk_level=RiskLevel.HIGH,
                message=(
                    f"⚠️ RECONSIDER TRAVEL advisory for "
                    f"{destination}."
                ),
                suggestions=[
                    "Review official state department advisories",
                    "Consider travel insurance",
                    "Register with embassy"
                ],
                rule_name=f"safety_warn_{destination}"
            )
        elif level == "exercise_caution":
            return GuardrailResult(
                action=GuardrailAction.WARN,
                risk_level=RiskLevel.MEDIUM,
                message=(
                    f"ℹ️ Exercise increased caution in "
                    f"{destination}."
                ),
                suggestions=[
                    "Stay aware of local conditions",
                    "Consider travel insurance"
                ],
                rule_name=f"safety_caution_{destination}"
            )

        return None


class RateLimitGuardrail:
    """Prevent excessive API usage"""

    def __init__(self, max_per_minute: int = 30):
        self.max_per_minute = max_per_minute
        self.call_log: List[float] = []

    def check(self, action: str) -> GuardrailResult:
        """Check if we're within rate limits"""
        now = time.time()

        # Clean old entries (older than 60 seconds)
        self.call_log = [
            t for t in self.call_log if now - t < 60
        ]

        if len(self.call_log) >= self.max_per_minute:
            wait_time = 60 - (now - self.call_log[0])
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.MEDIUM,
                message=(
                    f"⏱️ Rate limit reached "
                    f"({self.max_per_minute}/min). "
                    f"Wait {wait_time:.0f}s."
                ),
                rule_name="rate_limit_exceeded"
            )

        self.call_log.append(now)
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            risk_level=RiskLevel.LOW,
            message="Within rate limits",
            rule_name="rate_limit_ok"
        )


class AuditLog:
    """Complete audit trail of all guardrail activities"""

    def __init__(self):
        self.entries = []

    def log(self, layer: str, rule: str,
            action: str, message: str):
        """Log a guardrail check"""
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "rule": rule,
            "action": action,
            "message": message
        })

    def generate_report(self) -> str:
        """Generate a formatted audit report"""
        if not self.entries:
            return "No guardrail activity recorded."

        lines = [
            f"\n{'═' * 55}",
            f"🛡️  GUARDRAIL AUDIT REPORT",
            f"{'═' * 55}",
            f"Total checks: {len(self.entries)}",
        ]

        # Summary by action
        actions = {}
        for e in self.entries:
            a = e["action"]
            actions[a] = actions.get(a, 0) + 1

        lines.append(f"\nSummary:")
        for action, count in actions.items():
            emoji = {
                "allowed": "✅", "warned": "⚠️",
                "require_confirmation": "🔒",
                "blocked": "🚫"
            }.get(action, "•")
            lines.append(f"   {emoji} {action}: {count}")

        # Detailed log
        lines.append(f"\nDetailed Log:")
        for e in self.entries:
            lines.append(
                f"   [{e['timestamp'][:19]}] "
                f"{e['layer']}/{e['rule']}: "
                f"{e['action']} — {e['message']}"
            )

        lines.append(f"{'═' * 55}")
        return "\n".join(lines)
```

### Guardrail Pipeline

```
Every Agent Action flows through this pipeline:

    Agent wants to do something
              │
              ▼
    ┌──────────────────┐
    │ L1: INPUT        │──► Invalid? → BLOCK ❌
    │    VALIDATION    │
    └────────┬─────────┘
             │ ✅
             ▼
    ┌──────────────────┐
    │ L2: BUDGET       │──► Over budget? → BLOCK ❌
    │    PROTECTION    │──► High %? → WARN ⚠️
    └────────┬─────────┘
             │ ✅
             ▼
    ┌──────────────────┐
    │ L3: SAFETY       │──► Dangerous dest? → BLOCK ❌
    │    CHECKS        │──► Advisory? → CONFIRM 🔒
    └────────┬─────────┘
             │ ✅
             ▼
    ┌──────────────────┐
    │ L5: ACTION       │──► Dangerous? → CONFIRM 🔒
    │    AUTHORIZATION │──► Missing params? → BLOCK ❌
    └────────┬─────────┘
             │ ✅
             ▼
    ┌──────────────────┐
    │ L6: RATE         │──► Too many calls? → WAIT ⏱️
    │    LIMITING      │
    └────────┬─────────┘
             │ ✅
             ▼
    ┌──────────────────┐
    │ L7: AUDIT        │──► Log everything 📝
    │    LOGGING       │
    └────────┬─────────┘
             │
             ▼
      ✅ ACTION ALLOWED
```

---

# 🏗️ Putting It All Together

## Final Integrated Architecture

```python
class ProductionTripPlanner:
    """
    Production-ready trip planner combining all 5 enhancements.
    """

    def __init__(self, config: dict):
        # Core
        self.client = OpenAI(api_key=config["openai_key"])

        # Enhancement 1: Real APIs with resilience
        self.flight_api = ResilientSkyscannerAPI(
            api_key=config.get("skyscanner_key")
        )
        self.hotel_api = BookingComAPI(
            api_key=config.get("rapidapi_key")
        )
        self.yelp_api = YelpAPI(
            api_key=config.get("yelp_key")
        )

        # Enhancement 2: Advanced Memory
        self.memory = AdvancedMemorySystem(
            openai_client=self.client
        )

        # Enhancement 3: Multi-Agent with Message Bus
        self.message_bus = MessageBus()
        self.flight_agent = EnhancedFlightAgent(
            self.client, self.flight_api, self.message_bus
        )
        self.hotel_agent = EnhancedHotelAgent(
            self.client, self.hotel_api, self.message_bus
        )
        self.food_agent = EnhancedFoodAgent(
            self.client, self.yelp_api, self.message_bus
        )

        # Enhancement 4: Human-in-the-Loop
        self.human = EnhancedHumanInTheLoop(
            policy=ApprovalPolicy(),
            auto_approve=config.get("auto_approve", False)
        )

        # Enhancement 5: Guardrails
        self.guardrails = ComprehensiveGuardrails()

    def plan(self, user_id: str, request: dict) -> str:
        """
        Full planning pipeline with all enhancements active.
        """

        # ── Guardrails: Validate input ──
        validations = self.guardrails.validate_trip_request(
            request["origin"], request["destination"],
            request["date"], request["days"],
            request["budget"]
        )

        # Check for blockers
        blockers = [
            v for v in validations
            if v.action == GuardrailAction.BLOCK
        ]
        if blockers:
            return f"❌ Cannot plan trip: {blockers[0].message}"

        # ── Memory: Load context ──
        context = self.memory.build_rich_context(
            user_id, str(request)
        )

        # ── Multi-Agent: Coordinate search ──
        orchestrator = EnhancedOrchestrator(
            self.client, self.flight_api,
            self.hotel_api, self.yelp_api
        )
        results = orchestrator.coordinate_trip_planning(
            request["origin"], request["destination"],
            request["date"], request["days"],
            request["budget"],
            request.get("preferences", {})
        )

        # ── Human: Present choices ──
        flight_decision = self.human.request_decision(
            action="select_flight",
            options=results["flights"].get("flights", []),
            context={
                "category": "flights",
                "budget_remaining": request["budget"],
                "recommendation": results["flights"].get(
                    "analysis", {}
                ).get("final_recommendation", "")
            }
        )

        # ── Guardrails: Validate selection ──
        if flight_decision["selected"]:
            cost_check = self.guardrails.check_booking_cost(
                "flight",
                flight_decision["selected"].get("price", 0),
                request["budget"]
            )
            if cost_check.action == GuardrailAction.BLOCK:
                return f"❌ {cost_check.message}"

        # ── Memory: Save & learn ──
        self.memory.remember(
            user_id,
            f"Selected {flight_decision['selected'].get('airline')} "
            f"flight to {request['destination']}",
            memory_type="episodic",
            importance=0.6
        )

        # ── Continue with hotel, food... ──
        # (Similar pattern for each phase)

        # ── Guardrails: Final report ──
        print(self.guardrails.get_full_report())

        return "Trip planned successfully! ✅"


# ─────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────

if __name__ == "__main__":
    planner = ProductionTripPlanner({
        "openai_key": os.getenv("OPENAI_API_KEY", "your-key"),
        "skyscanner_key": os.getenv("SKYSCANNER_API_KEY"),
        "rapidapi_key": os.getenv("RAPIDAPI_KEY"),
        "yelp_key": os.getenv("YELP_API_KEY"),
        "auto_approve": False,
    })

    planner.plan(
        user_id="user_456",
        request={
            "origin": "New York",
            "destination": "Tokyo",
            "date": "2025-06-15",
            "days": 3,
            "budget": 2000,
            "preferences": {
                "cuisine": "ramen, sushi",
                "travel_style": "foodie",
                "dietary_restrictions": "none"
            }
        }
    )
```

---

## Summary: How All 5 Enhancements Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   USER: "Plan a Tokyo food trip, $2000 budget"              │
│                          │                                  │
│   ┌──────────────────────▼─────────────────────────┐        │
│   │ 🛡️ GUARDRAILS validate inputs                  │ ← (5) │
│   └──────────────────────┬─────────────────────────┘        │
│                          │                                  │
│   ┌──────────────────────▼─────────────────────────┐        │
│   │ 📂 MEMORY loads user context                    │ ← (2) │
│   │ "User loves ramen, prefers budget hotels"       │        │
│   └──────────────────────┬─────────────────────────┘        │
│                          │                                  │
│   ┌──────────────────────▼─────────────────────────┐        │
│   │ 🤖 MULTI-AGENTS search & analyze               │ ← (3) │
│   │  ✈️ Flight Agent  → Skyscanner API              │ ← (1) │
│   │  🏨 Hotel Agent   → Booking.com API             │ ← (1) │
│   │  🍣 Food Agent    → Yelp API                    │ ← (1) │
│   │  (Agents collaborate via Message Bus)           │        │
│   └──────────────────────┬─────────────────────────┘        │
│                          │                                  │
│   ┌──────────────────────▼─────────────────────────┐        │
│   │ 🛡️ GUARDRAILS check costs & safety             │ ← (5) │
│   └──────────────────────┬─────────────────────────┘        │
│                          │                                  │
│   ┌──────────────────────▼─────────────────────────┐        │
│   │ 🙋 HUMAN reviews options & approves             │ ← (4) │
│   └──────────────────────┬─────────────────────────┘        │
│                          │                                  │
│   ┌──────────────────────▼─────────────────────────┐        │
│   │ 📂 MEMORY saves trip & updates preferences      │ ← (2) │
│   └──────────────────────┬─────────────────────────┘        │
│                          │                                  │
│   ┌──────────────────────▼─────────────────────────┐        │
│   │ 🛡️ GUARDRAILS generate audit report            │ ← (5) │
│   └──────────────────────┬─────────────────────────┘        │
│                          │                                  │
│                          ▼                                  │
│                  📋 FINAL ITINERARY ✅                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Each enhancement is **modular** — you can adopt them incrementally, start with what matters most, and add complexity as your use case demands.

Would you like me to explore any specific pattern further, or help adapt this architecture to a different use case?
