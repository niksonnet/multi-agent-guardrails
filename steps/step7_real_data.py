import datetime
import json

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    ModelSettings,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    function_tool,
    handoff,
    output_guardrail,
)
from agents.extensions import handoff_filters

from core import travel_apis
from steps.step3_handoffs import HandoffReason, print_reason
from steps.step4_structured_output import TripPlan
from steps.step5_guardrails import travel_only, within_budget
from steps.step6_multi_model import MULTI_MODEL

TODAY = datetime.date.today().isoformat()
DATE_RULE = f"Today is {TODAY}. If the user gives no date, use {travel_apis.default_date()}. Dates are YYYY-MM-DD."


# ---- Tools: thin wrappers around the real APIs in travel_apis.py ----
# They are `async def` on purpose: the SDK runs plain `def` tools in a background thread,
# and Streamlit can only draw the panel (the API CALL steps) from the main thread.

@function_tool
async def search_flights(origin_code: str, destination_code: str, date: str) -> str:
    """Find the cheapest one-way flights. Use IATA airport codes, e.g. DEL (Delhi), DXB (Dubai). date: YYYY-MM-DD."""
    return json.dumps(travel_apis.search_flights(origin_code, destination_code, date))


@function_tool
async def search_hotels(city: str, check_in: str, nights: int) -> str:
    """Find the cheapest hotels in a city. check_in: YYYY-MM-DD."""
    return json.dumps(travel_apis.search_hotels(city, check_in, nights))


class RunMemory:
    """Shared notes for one run (pass it as Runner.run(..., context=RunMemory())).
    The fetch_trip_data tool writes the data here; the grounded_live guardrail reads it."""
    fetched_trip = None


@function_tool
async def fetch_trip_data(ctx: RunContextWrapper, origin_code: str, destination_code: str, destination_city: str,
                    departure_date: str, nights: int, budget_usd: float) -> str:
    """Fetch everything needed to plan a trip: flights, hotels, weather and things to do.
    Use IATA airport codes, e.g. DEL (Delhi), DXB (Dubai). departure_date: YYYY-MM-DD."""
    # Plain Python calls the APIs, no AI. The arguments above ARE the model's reading of the request.
    # (ctx is filled in by the SDK, the model never sees it.)
    data = {
        "request": {"origin_code": origin_code, "destination_code": destination_code,
                    "destination_city": destination_city, "departure_date": departure_date,
                    "nights": nights, "budget_usd": budget_usd},
        "flights": travel_apis.search_flights(origin_code, destination_code, departure_date),
        "hotels": travel_apis.search_hotels(destination_city, departure_date, nights),
        "weather": travel_apis.get_weather(destination_city, departure_date),
        "things_to_do": travel_apis.get_attractions(destination_city),
    }
    ctx.context.fetched_trip = data  # remember exactly what the agents were given
    return json.dumps(data)


# ---- Structured output: the step 4 TripPlan plus the real-data fields ----

class LiveTripPlan(TripPlan):
    origin_code: str
    destination_code: str
    departure_date: str
    weather: str
    things_to_do: list[str]
    data_source: str


# ---- Output guardrail: every price must exist in the real data ----

@output_guardrail
async def grounded_live(ctx, agent, plan: LiveTripPlan):
    # Check the plan against exactly the data fetch_trip_data gave the agents (saved in the run context).
    # No new API calls, so no quota is used.
    fetched = getattr(ctx.context, "fetched_trip", None) or {"flights": [], "hotels": []}
    flights, hotels = fetched["flights"], fetched["hotels"]
    user_text = " ".join(str(item.get("content")) for item in ctx.turn_input if item.get("role") == "user")
    user_text = user_text.lower().replace(",", "")

    problems = []
    if not any(f["airline"] == plan.airline and f["price_usd"] == plan.flight_price_usd for f in flights):
        problems.append(f"flight {plan.airline} ${plan.flight_price_usd:.0f} is not in the search results")
    if not any(h["name"] == plan.hotel_name and h["price_per_night_usd"] == plan.hotel_price_per_night_usd
               for h in hotels):
        problems.append(f"hotel {plan.hotel_name} ${plan.hotel_price_per_night_usd:.0f}/night is not in the results")
    if f"{plan.budget_usd:.0f}" not in user_text:
        problems.append(f"budget ${plan.budget_usd:.0f} is not in the request")
    return GuardrailFunctionOutput(
        output_info="; ".join(problems) or "every price matches the real search results",
        tripwire_triggered=bool(problems),
    )


# ---- The team ----

def build_live_agents(models=MULTI_MODEL, output_guardrails=(grounded_live, within_budget), on_handoff=print_reason):
    """The specialists: flight, hotel, planner (the planner hands off to the writer itself)."""
    # No tool_choice="required" (see step 3): with no cities the model must be allowed to ask.
    flight = Agent(
        name="Flight Agent",
        instructions=(
            "You find flights. If the origin or destination city is missing, ask the user for it. "
            f"Otherwise ALWAYS call search_flights (never guess prices), then summarise the cheapest options. {DATE_RULE}"
        ),
        model=models["flight"], tools=[search_flights],
    )
    hotel = Agent(
        name="Hotel Agent",
        instructions=(
            "You find hotels. If the city is missing, ask the user for it. "
            f"Otherwise ALWAYS call search_hotels (never guess prices), then summarise the cheapest options. {DATE_RULE}"
        ),
        model=models["hotel"], tools=[search_hotels],
    )
    writer = Agent(
        name="Plan Writer",
        instructions=(
            "The conversation contains real travel data from fetch_trip_data. Pick the cheapest flight and the "
            "cheapest hotel. Copy the airline, prices, hotel name, codes and dates EXACTLY from the data; "
            "never invent values. destination = the destination_city (a city name, not an airport code). "
            "total_cost_usd = flight price + hotel price per night x nights. "
            "data_source = the 'source' field of the chosen flight. "
            "summary = 2-3 plain sentences (no HTML, no markdown) mentioning the weather if known."
        ),
        model=models["planner"], output_type=LiveTripPlan, output_guardrails=list(output_guardrails),
    )
    planner = Agent(
        name="Trip Planner",
        handoff_description="Plans a full trip (flight + hotel) within a budget, using real prices.",
        instructions=(
            f"First call fetch_trip_data with the details from the user's request. {DATE_RULE} "
            "If the number of nights is missing, use 3. Then hand off to the Plan Writer."
        ),
        # parallel_tool_calls=False: one fetch only. Otherwise the model may call it several times at once.
        model=models["planner"], tools=[fetch_trip_data],
        model_settings=ModelSettings(tool_choice="required", parallel_tool_calls=False),
        # No input_filter here: the Plan Writer needs the fetched data from the history.
        handoffs=[handoff(writer, input_type=HandoffReason, on_handoff=on_handoff)],
    )
    return flight, hotel, planner


def build_live_team(models=MULTI_MODEL, input_guardrail=travel_only, output_guardrails=(grounded_live, within_budget),
                    on_handoff=print_reason):
    """Same shape as step 6, but the agents use real APIs. Returns the triage agent."""
    flight, hotel, planner = build_live_agents(models, output_guardrails, on_handoff)
    return Agent(
        name="Triage Agent",
        instructions=(
            "Decide who handles the request:\n"
            "1. Only about flights -> hand off to the Flight Agent.\n"
            "2. Only about hotels -> hand off to the Hotel Agent.\n"
            "3. A full trip with a destination AND a budget -> hand off to the Trip Planner.\n"
            "4. Anything else (general questions, a trip without a budget) -> do not hand off. Answer briefly "
            "yourself: say you can find real flight and hotel prices and plan trips with a budget, and ask what "
            "the user needs."
        ),
        model=models["triage"],
        input_guardrails=[input_guardrail],
        handoffs=[
            handoff(a, input_type=HandoffReason, on_handoff=on_handoff, input_filter=handoff_filters.remove_all_tools)
            for a in (flight, hotel, planner)
        ],
    )


LIVE_PROMPTS = [
    "Find me a flight from Delhi to Dubai.",
    "Show me hotels in Goa for 2 nights.",
    "Plan a trip from Delhi to Dubai for 3 nights. My budget is $800.",
    "Plan a trip from Mumbai to Goa for 2 nights. My budget is $100.",
    "Tell me about Goa.",
]

if __name__ == "__main__":
    team = build_live_team()
    for prompt in LIVE_PROMPTS[2:4]:
        print(f"\n{'=' * 60}\n>>> {prompt}")
        try:
            result = Runner.run_sync(team, prompt, context=RunMemory())
            print(f"Answered by: {result.last_agent.name}")
            output = result.final_output
            print(output.model_dump_json(indent=2) if hasattr(output, "model_dump_json") else output)
        except InputGuardrailTripwireTriggered as e:
            print("BLOCKED by input guardrail:", e.guardrail_result.output.output_info.reason)
        except OutputGuardrailTripwireTriggered as e:
            print("BLOCKED by output guardrail:", e.guardrail_result.output.output_info)
    print(f"\nSerpApi searches left: {travel_apis.serpapi_searches_left()}")
