from pydantic import BaseModel
from agents import Agent, Runner, handoff

from core.config import model
from core.mock_data import search_flights, search_hotels

# Two specialist agents, each with its own tool.
# We don't force the tool call (tool_choice="required"): for "find flights" with no cities, the model
# can't fill in the tool's arguments and must be allowed to ask. The instructions say: never guess prices.
flight_agent = Agent(
    name="Flight Agent",
    instructions=(
        "You find flights. If the origin or destination city is missing, ask the user for it. "
        "Otherwise ALWAYS call the search_flights tool (never guess prices), then summarise the result."
    ),
    model=model,
    tools=[search_flights],
)

hotel_agent = Agent(
    name="Hotel Agent",
    instructions=(
        "You find hotels. If the city is missing, ask the user for it. "
        "Otherwise ALWAYS call the search_hotels tool (never guess prices), then summarise the options."
    ),
    model=model,
    tools=[search_hotels],
)


# Groq needs every handoff to have at least one argument, so the model must give a reason.
class HandoffReason(BaseModel):
    reason: str


def print_reason(ctx, data: HandoffReason):
    print(f"  [handoff] reason: {data.reason}")


# The triage agent does not answer. It hands the conversation to the right specialist.
triage_agent = Agent(
    name="Triage Agent",
    instructions="Send flight questions to the Flight Agent and hotel questions to the Hotel Agent. Do not answer yourself.",
    model=model,
    handoffs=[
        handoff(flight_agent, input_type=HandoffReason, on_handoff=print_reason),
        handoff(hotel_agent, input_type=HandoffReason, on_handoff=print_reason),
    ],
)

if __name__ == "__main__":
    for prompt in ["I need a flight from Delhi to Tokyo.", "Show me hotels in London for 2 nights."]:
        print(f"\n>>> {prompt}")
        result = Runner.run_sync(triage_agent, prompt)
        print(f"Answered by: {result.last_agent.name}")
        print(result.final_output)
