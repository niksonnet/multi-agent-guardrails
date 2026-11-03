import asyncio
from typing import Literal

from pydantic import BaseModel
from agents import Agent, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, Runner

from core.config import fast_model, keep_messages, retry_config, structured_model
from steps.demo import planner
from steps.step3_handoffs import flight_agent, hotel_agent
from steps.step5_guardrails import travel_only
from steps.step6_multi_model import MULTI_MODEL, build_team

# Orchestration = who decides which agent runs next?
#
#   1. LLM + handoffs        the triage agent hands the conversation over. Control is GIVEN AWAY.
#   2. LLM + agents as tools a manager agent calls specialists like tools. Control COMES BACK to it.
#   3. Code                  a router agent only reads the request. Your if/elif picks the agent.
#
# The specialists are the same in all three. Models from step 6: fast for tools, qwen for JSON.
flight = flight_agent.clone(model=fast_model)
hotel = hotel_agent.clone(model=fast_model)
planner = planner.clone(model=structured_model)


# ---- 1. LLM + handoffs ----
# Nothing new: the step 6 team already works this way.
triage_agent = build_team(MULTI_MODEL)


# ---- 2. LLM + agents as tools ----

def make_manager(flight, hotel, planner, model, input_guardrail=travel_only, hooks=None):
    return Agent(
        name="Manager Agent",
        instructions=(
            "You manage travel specialists. Pass them the full request, with every detail the user gave "
            "(also in earlier messages: cities, dates, nights, budget). They don't see the conversation.\n"
            "Flights only -> call flight_agent. Hotels only -> call hotel_agent. "
            "A full trip with a destination AND a budget -> call plan_trip.\n"
            "Then answer the user using what the specialist returned.\n"
            "Anything else -> answer briefly yourself: say you can find flights and hotels and plan trips "
            "with a budget, and ask what the user needs."
        ),
        model=model,
        input_guardrails=[input_guardrail],
        # as_tool() turns an agent into a tool. The specialist runs, and its answer comes back to the manager.
        # failure_error_function=None: if a guardrail inside a specialist fires, stop everything
        #   (by default the manager would just get the error as text and carry on).
        # hooks: so the specialist's run is also reported (the app uses this for its panel).
        tools=[
            flight.as_tool("flight_agent", "Finds flights.", failure_error_function=None, hooks=hooks),
            hotel.as_tool("hotel_agent", "Finds hotels.", failure_error_function=None, hooks=hooks),
            planner.as_tool("plan_trip", "Plans a full trip within a budget.", failure_error_function=None, hooks=hooks),
        ],
    )


manager_agent = make_manager(flight, hotel, planner, fast_model)


# ---- 3. Code ----

class Route(BaseModel):
    intent: Literal["flight", "hotel", "trip", "other"]
    budget_given: bool  # did the user say a budget (in any message)?
    reply: str  # only used for "other"


# The router doesn't decide anything. It only reads the request and fills in Route.
router_agent = Agent(
    name="Router Agent",
    instructions=(
        "Do not answer the request. Only read it (earlier messages give context).\n"
        "intent: flight (only about flights), hotel (only about hotels), "
        "trip (planning a full trip), other (anything else).\n"
        "budget_given: true only if the user wrote a budget amount.\n"
        "reply: only for other. Say you can find flights and hotels and plan trips with a budget, "
        "and ask what the user needs. Otherwise leave it empty."
    ),
    model=structured_model,  # it returns JSON
    output_type=Route,
    input_guardrails=[travel_only],
)


async def run_by_code(prompt, router, flight, hotel, planner, context=None, hooks=None, run_config=retry_config,
                      session=None):
    # With memory (step 9) the router also reads the earlier messages. Only the chosen agent saves to the
    # session, so each turn is stored once.
    history = await session.get_items() if session else []
    new_message = [{"role": "user", "content": prompt}]
    route = (await Runner.run(router, keep_messages(history, new_message), context=context, hooks=hooks,
                              run_config=retry_config)).final_output

    # Plain Python decides. No model is asked who should go next.
    if route.intent == "flight":
        agent = flight
    elif route.intent == "hotel":
        agent = hotel
    elif route.intent == "trip" and route.budget_given:
        agent = planner
    else:
        if route.intent == "trip":  # a rule in code, not a hope in a prompt: no plan without a budget
            route.reply = "Happy to plan it! What is your budget for this trip (in USD)?"
        show_decision(context, f'intent is "{route.intent}", budget_given={route.budget_given} -> no agent, reply')
        if session:  # no agent ran, so save this turn ourselves
            await session.add_items(new_message + [{"role": "assistant", "content": route.reply}])
        return route.reply

    show_decision(context, f'intent is "{route.intent}" -> run {agent.name}')
    result = await Runner.run(agent, prompt, context=context, hooks=hooks, run_config=run_config, session=session)
    return result.final_output


def show_decision(context, text):
    """Print what the code decided. In the app the run context is the panel, so it is drawn there instead."""
    if hasattr(context, "code_step"):
        context.code_step(text)
    else:
        print(f"  [code] {text}")


# ---- Try the same requests with all three ----

async def ask(mode, prompt):
    if mode == "handoffs":
        return (await Runner.run(triage_agent, prompt, run_config=retry_config)).final_output
    if mode == "agents as tools":
        return (await Runner.run(manager_agent, prompt, run_config=retry_config)).final_output
    return await run_by_code(prompt, router_agent, flight, hotel, planner)


async def main():
    prompts = [
        "I need a flight from Delhi to Tokyo.",
        "Plan a trip from Delhi to Dubai for 3 nights. My budget is $1000.",
        "Tell me about Goa.",
    ]
    for prompt in prompts:
        print(f"\n{'=' * 60}\n>>> {prompt}")
        for mode in ["handoffs", "agents as tools", "code"]:
            print(f"\n[{mode}]")
            try:
                print(await ask(mode, prompt))
            except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as e:
                print("BLOCKED by guardrail:", type(e).__name__)
            except Exception as e:  # e.g. Groq rejects a model's output
                print("MODEL ERROR:", getattr(e, "code", None) or type(e).__name__)


if __name__ == "__main__":
    asyncio.run(main())
