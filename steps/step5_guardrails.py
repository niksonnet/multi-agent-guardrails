from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    input_guardrail,
    output_guardrail,
)

from core.config import retry_config, structured_model
from steps.step4_structured_output import TripPlan, planner_agent
from core.mock_data import HOTELS


# ---- Input guardrail: checks the user's message BEFORE the agent answers ----

class TravelCheck(BaseModel):
    is_travel: bool
    reason: str


# A small agent whose only job is to say "travel" or "not travel".
# It returns JSON (output_type), so it uses the model that is reliable at structured output.
checker_agent = Agent(
    name="Travel Checker",
    instructions=(
        "You are given a user message in quotes. Do not answer it. Only decide if it is about travel "
        "(flights, hotels, trips). If earlier messages are given and they are about travel, a short follow-up "
        "(a budget, a date, a city, 'yes') is about travel too."
    ),
    model=structured_model,
    output_type=TravelCheck,
)


@input_guardrail
async def travel_only(ctx, agent, user_message):
    text = f'User message: "{user_message}"'
    # With memory (step 9) we get the whole conversation as a list. Check the newest message,
    # and show the earlier ones so a follow-up like "My budget is $1000" is seen as part of a trip.
    if isinstance(user_message, list):
        earlier = [item["content"] for item in user_message[:-1] if item.get("role") == "user"]
        text = f'Earlier user messages: {earlier[-3:]}\nUser message: "{user_message[-1]["content"]}"'
    # The checker is its own run, so it doesn't get the main run's run_config: pass the retry here too.
    result = await Runner.run(checker_agent, text, run_config=retry_config)
    check = result.final_output
    # tripwire_triggered=True means "stop, block this request"
    return GuardrailFunctionOutput(output_info=check, tripwire_triggered=not check.is_travel)


# ---- Output guardrail: checks the agent's answer BEFORE the user sees it ----

@output_guardrail
async def within_budget(ctx, agent, plan: TripPlan):
    # Plain Python, no AI needed. We do the maths ourselves instead of trusting the model.
    real_cost = plan.flight_price_usd + plan.hotel_price_per_night_usd * plan.nights
    return GuardrailFunctionOutput(
        output_info=f"trip costs ${real_cost:.0f}, budget is ${plan.budget_usd:.0f}",
        tripwire_triggered=real_cost > plan.budget_usd,
    )


@output_guardrail
async def grounded_plan(ctx, agent, plan: TripPlan):
    # output_type forces the model to fill in EVERY field, even when it doesn't know.
    # So it may invent a destination or a budget. Compare the plan with what the user actually wrote.
    # ctx.turn_input = the messages sent to the model in this turn (includes the user's message).
    user_text = " ".join(str(item.get("content")) for item in ctx.turn_input if item.get("role") == "user")
    user_text = user_text.lower().replace(",", "")
    destination = plan.destination.lower()

    problems = []
    if destination not in HOTELS:
        problems.append(f'"{plan.destination}" is not a city we have prices for')
    if destination not in user_text:
        problems.append(f'destination "{plan.destination}" is not in the request')
    if f"{plan.budget_usd:.0f}" not in user_text:
        problems.append(f"budget ${plan.budget_usd:.0f} is not in the request")
    return GuardrailFunctionOutput(
        output_info="; ".join(problems) or "plan matches the request",
        tripwire_triggered=bool(problems),
    )


# Same planner as step 4, with the guardrails added.
safe_planner = planner_agent.clone(input_guardrails=[travel_only], output_guardrails=[grounded_plan, within_budget])

if __name__ == "__main__":
    for prompt in [
        "Plan a trip from Delhi to Dubai for 3 nights. My budget is $1000.",  # allowed
        "Write me a poem about cats.",  # blocked by input guardrail
        "Plan a trip from Delhi to Paris for 5 nights. My budget is $300.",  # blocked: over budget
        "Plan a trip to Goa.",  # blocked: Goa isn't in our data and there's no budget
    ]:
        print(f"\n>>> {prompt}")
        try:
            result = Runner.run_sync(safe_planner, prompt)
            print(result.final_output.model_dump_json(indent=2))
        except InputGuardrailTripwireTriggered as e:
            print("BLOCKED by input guardrail:", e.guardrail_result.output.output_info.reason)
        except OutputGuardrailTripwireTriggered as e:
            print("BLOCKED by output guardrail:", e.guardrail_result.output.output_info)
