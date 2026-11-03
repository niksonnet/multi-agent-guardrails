from agents import Agent, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, Runner, handoff

from core.config import model
from steps.step3_handoffs import HandoffReason, flight_agent, hotel_agent, print_reason
from steps.step4_structured_output import planner_agent
from steps.step5_guardrails import grounded_plan, travel_only, within_budget
from core.mock_data import HOTELS

# Everything together:
# - the triage agent checks the input (input guardrail) and hands off to a specialist
# - the planner returns a TripPlan (structured output) and its answer is checked (output guardrails)
# Input guardrails run on the FIRST agent, output guardrails on the LAST agent. That's why they are placed like this.
planner = planner_agent.clone(output_guardrails=[grounded_plan, within_budget])

CITIES = ", ".join(city.title() for city in HOTELS)

# The triage agent may answer by itself when no specialist fits. Otherwise it would force
# every request into some specialist, e.g. "Tell me about Goa" into the Trip Planner.
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        f"We only have data for these cities: {CITIES}.\n"
        "Send flight-only questions to the Flight Agent and hotel-only questions to the Hotel Agent.\n"
        "Send a request to the Trip Planner ONLY if the user names one of our cities AND gives a budget.\n"
        "Otherwise (general questions, a city we don't cover, or a missing budget) do not hand off. "
        f"Answer briefly yourself: say you can find flights and hotels and plan trips with a budget for {CITIES}, "
        "and ask what the user needs."
    ),
    model=model,
    input_guardrails=[travel_only],
    handoffs=[
        handoff(flight_agent, input_type=HandoffReason, on_handoff=print_reason),
        handoff(hotel_agent, input_type=HandoffReason, on_handoff=print_reason),
        handoff(planner, input_type=HandoffReason, on_handoff=print_reason),
    ],
)

DEMO_PROMPTS = [
    "I need a flight from Delhi to Tokyo.",
    "Show me hotels in London for 2 nights.",
    "Plan a trip from Delhi to Dubai for 3 nights. My budget is $1000.",
    "Write me a poem about cats.",
    "Plan a trip from Delhi to Paris for 5 nights. My budget is $300.",
    "Tell me about Goa.",
]

if __name__ == "__main__":
    for prompt in DEMO_PROMPTS:
        print(f"\n{'=' * 60}\n>>> {prompt}")
        try:
            result = Runner.run_sync(triage_agent, prompt)
            print(f"Answered by: {result.last_agent.name}")
            print(result.final_output)
        except InputGuardrailTripwireTriggered as e:
            print("BLOCKED by input guardrail:", e.guardrail_result.output.output_info.reason)
        except OutputGuardrailTripwireTriggered as e:
            print("BLOCKED by output guardrail:", e.guardrail_result.output.output_info)
