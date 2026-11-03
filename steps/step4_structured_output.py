from pydantic import BaseModel
from agents import Agent, Runner

from core.config import model
from core.mock_data import FLIGHTS, HOTELS


# The shape we want the answer in. The agent must return exactly these fields.
class TripPlan(BaseModel):
    destination: str
    airline: str
    flight_price_usd: float
    hotel_name: str
    hotel_price_per_night_usd: float
    nights: int
    total_cost_usd: float
    budget_usd: float
    summary: str


# output_type=TripPlan makes the agent return a TripPlan object instead of plain text.
# Note: Groq can't use tools and output_type in the same agent, so we put the prices in the instructions.
planner_agent = Agent(
    name="Trip Planner",
    instructions=(
        "Plan a trip using the cheapest flight and hotel from these prices.\n"
        f"Flights (origin, destination): {FLIGHTS}\n"
        f"Hotels: {HOTELS}\n"
        "Use the budget the user gives you."
    ),
    model=model,
    output_type=TripPlan,
)

if __name__ == "__main__":
    result = Runner.run_sync(planner_agent, "Plan a trip from Delhi to Dubai for 3 nights. My budget is $1000.")
    plan = result.final_output  # this is a TripPlan object, not text
    print(plan.model_dump_json(indent=2))
    print(f"\nWe can use the fields directly: total is ${plan.total_cost_usd}, hotel is {plan.hotel_name}")
