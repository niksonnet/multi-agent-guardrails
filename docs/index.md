---
layout: default
---

# A rule in code, not a hope in a prompt

Ask a language model to plan a trip and return the plan as JSON, and it will. Every field is filled in: destination, airline, hotel, budget. It fills them in even when you never said where you're going or how much you want to spend, because a structured output *forces* the model to put something in every field. So it invents.

That one observation is the reason this project exists. [**multi-agent-guardrails**](https://github.com/niksonnet/multi-agent-guardrails) is a small travel planner built in nine steps with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), running on [Groq](https://console.groq.com/)'s free tier. Each step adds one idea, and by step 5 the agents are wrapped in guardrails that check what goes in and what comes out.

This article walks through the path, with most of the attention on the guardrails.

## What you need

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- A free Groq API key from [console.groq.com](https://console.groq.com/keys)

```bash
git clone https://github.com/niksonnet/multi-agent-guardrails
cd multi-agent-guardrails
uv sync
echo "GROQ_API_KEY=your_key" > .env
uv run python -m steps.step1_basic_agent
```

Groq speaks the same API as OpenAI, so the project uses the normal OpenAI client pointed at Groq's URL. Everything else is the Agents SDK as documented.

## Steps 1–4: from one agent to a team

**Step 1: an agent** is a name, instructions and a model.

```python
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Answer concisely.",
    model=model,
)
result = Runner.run_sync(agent, "What is a multi-agent system? Answer in 2 sentences.")
```

**Step 2: tools.** Decorate a Python function with `@function_tool` and hand it to the agent. The model decides when to call it; the SDK runs the function and gives the result back.

```python
@function_tool
def search_flights(origin: str, destination: str) -> str:
    """Search one-way flights between two cities. Returns airline and price in USD."""
    flight = FLIGHTS.get((origin.lower(), destination.lower()))
    return json.dumps(flight) if flight else "No flights found."
```

**Step 3: handoffs.** A Triage Agent doesn't answer anything. It reads the request and hands the conversation to a Flight Agent or a Hotel Agent.

```python
triage_agent = Agent(
    name="Triage Agent",
    instructions="Send flight questions to the Flight Agent and hotel questions to the Hotel Agent. Do not answer yourself.",
    model=model,
    handoffs=[
        handoff(flight_agent, input_type=HandoffReason, on_handoff=print_reason),
        handoff(hotel_agent, input_type=HandoffReason, on_handoff=print_reason),
    ],
)
```

**Step 4: structured output.** With `output_type=TripPlan`, the planner returns a Pydantic object instead of text, so your code can read `plan.total_cost_usd` directly.

```python
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
```

This is where the problem from the introduction shows up. Ask *"Plan a trip to Goa."* with no budget, and the planner still returns a complete `TripPlan`, budget included.

## Step 5: guardrails

A guardrail is a check that runs alongside an agent. If it trips, the SDK stops the run and raises an exception, so the bad answer never reaches the user. There are two kinds:

- **Input guardrails** check the user's message *before* the agent answers.
- **Output guardrails** check the agent's answer *before* the user sees it.

### Input: is this even about travel?

The input guardrail uses a second, tiny agent whose only job is to classify the message. It returns JSON, so it runs on the model that is most reliable at structured output.

```python
class TravelCheck(BaseModel):
    is_travel: bool
    reason: str

checker_agent = Agent(
    name="Travel Checker",
    instructions="You are given a user message in quotes. Do not answer it. "
                 "Only decide if it is about travel (flights, hotels, trips). ...",
    model=structured_model,
    output_type=TravelCheck,
)

@input_guardrail
async def travel_only(ctx, agent, user_message):
    result = await Runner.run(checker_agent, f'User message: "{user_message}"', run_config=retry_config)
    check = result.final_output
    return GuardrailFunctionOutput(output_info=check, tripwire_triggered=not check.is_travel)
```

*"Write me a poem about cats."* never reaches the planner. Note the quotes around the message and the "Do not answer it" line: the checker is told to judge the text, not follow it.

### Output: do the maths yourself

The budget check needs no AI at all. The model already returned the numbers; plain Python adds them up.

```python
@output_guardrail
async def within_budget(ctx, agent, plan: TripPlan):
    # We do the maths ourselves instead of trusting the model.
    real_cost = plan.flight_price_usd + plan.hotel_price_per_night_usd * plan.nights
    return GuardrailFunctionOutput(
        output_info=f"trip costs ${real_cost:.0f}, budget is ${plan.budget_usd:.0f}",
        tripwire_triggered=real_cost > plan.budget_usd,
    )
```

It deliberately ignores `plan.total_cost_usd`. That field was written by the model, and the point of the check is not to trust it.

### Output: did the user actually say that?

This is the guardrail for the invented-fields problem. It compares the plan with what the user wrote:

```python
@output_guardrail
async def grounded_plan(ctx, agent, plan: TripPlan):
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
```

Attaching all three takes one line:

```python
safe_planner = planner_agent.clone(
    input_guardrails=[travel_only],
    output_guardrails=[grounded_plan, within_budget],
)
```

Running step 5 tries four requests:

| Request | Result |
|---|---|
| Delhi → Dubai, 3 nights, budget $1000 | ✅ Plan returned |
| "Write me a poem about cats." | 🛑 Blocked by `travel_only` |
| Delhi → Paris, 5 nights, budget $300 | 🛑 Blocked by `within_budget` |
| "Plan a trip to Goa." | 🛑 Blocked by `grounded_plan`: no prices for Goa, no budget given |

Two of the three guardrails are plain Python. That's the theme of the whole project: **if a rule can be written as code, write it as code.** Use a model only for the judgement calls code can't make, like "is this about travel?"

## Step 6: the right model for each job

Not every agent needs the biggest model. Routing a request and making one tool call are easy; producing valid JSON reliably is a specific skill.

```python
MULTI_MODEL = {
    "triage": fast_model,       # gpt-oss-20b: picks 1 of 3 handoffs
    "flight": fast_model,       # 1 tool call + short answer
    "hotel": fast_model,
    "planner": structured_model # qwen: reliable at TripPlan JSON
}
```

Running step 6 benchmarks this against using a single model everywhere, comparing passes, time and tokens.

## Step 7: guardrails against real data

With mock data, prices are easy to check. Step 7 swaps in live flight and hotel prices (SerpAPI, with fallbacks down to the mock data) and splits the work into a Trip Planner that calls the tools and a Plan Writer that writes the JSON.

The trick is the **run context**: the tool that fetches the data also saves exactly what it fetched, and the output guardrail checks the final plan against that, with no new API calls.

```python
@function_tool
async def fetch_trip_data(ctx: RunContextWrapper, origin_code: str, ...) -> str:
    data = {"flights": travel_apis.search_flights(...), "hotels": travel_apis.search_hotels(...), ...}
    ctx.context.fetched_trip = data  # remember exactly what the agents were given
    return json.dumps(data)

@output_guardrail
async def grounded_live(ctx, agent, plan: LiveTripPlan):
    fetched = getattr(ctx.context, "fetched_trip", None) or {"flights": [], "hotels": []}
    problems = []
    if not any(f["airline"] == plan.airline and f["price_usd"] == plan.flight_price_usd
               for f in fetched["flights"]):
        problems.append(f"flight {plan.airline} ${plan.flight_price_usd:.0f} is not in the search results")
    ...
```

Every price in the plan must exist in the search results. If the model rounds a price, swaps a hotel, or makes one up, the plan is blocked.

## Step 8: who decides what runs next?

The same specialist agents can be orchestrated three ways, and step 8 runs the same requests through all of them:

| | LLM · handoffs | LLM · agents as tools | Code |
|---|---|---|---|
| Who decides | Triage Agent | Manager Agent | Your `if/elif` |
| Control | Given away to the specialist | Comes back to the manager | Stays in Python |
| SDK feature | `handoffs=[...]` | `agent.as_tool(...)` | `Runner.run` called by your code |
| Strengths | Flexible, handles odd requests | Manager can combine answers | Predictable, fewest model calls, easy to debug |
| Weaknesses | Can route wrongly, extra model calls | Most model calls | Only handles the paths you wrote |

In code mode, a Router agent only *reads* the request into a small `Route` object, and Python decides:

```python
if route.intent == "flight":
    agent = flight
elif route.intent == "hotel":
    agent = hotel
elif route.intent == "trip" and route.budget_given:
    agent = planner
else:
    if route.intent == "trip":  # a rule in code, not a hope in a prompt: no plan without a budget
        route.reply = "Happy to plan it! What is your budget for this trip (in USD)?"
```

"No trip plan without a budget" could be a line in a prompt. Here it's an `if`, so the model can't skip it.

## Step 9: memory, and a guardrail that understands follow-ups

Real users don't give every detail at once:

1. *"I want to plan a trip in December 2026."*
2. *"From Delhi to Dubai, 3 nights."*
3. *"My budget is $1000."*

Without memory, each message stands alone and the planner never has enough to work with. With a session, the SDK loads earlier messages, runs the agents, and saves the new turn:

```python
result = await Runner.run(team, message, session=SQLiteSession("user-123"), run_config=memory_config)
```

Memory has a knock-on effect on guardrails. Taken on its own, *"My budget is $1000."* doesn't look like travel. So `travel_only` checks the newest message but shows the checker the earlier ones, and a follow-up in a travel conversation counts as travel.

## Lessons from running on Groq's free tier

Free and fast comes with some sharp edges. Each one is handled in the repo:

- **Models sometimes return broken tool calls or JSON** (`tool_use_failed`, `json_validate_failed`). It's random, so `retry_config` retries those two errors up to twice. A guardrail's checker agent is its own run and needs the retry passed to it too.
- **Handoffs need an argument.** Groq rejects handoffs with no parameters, so each handoff asks the model for a `reason`, which is also handy for logging.
- **Tools or structured output, not both** on one agent. That's why step 7 splits the planner into a tool-calling agent and a JSON-writing agent.
- **Don't force tool calls.** With `tool_choice="required"`, an agent that needs to ask *"which city?"* can't, and the run fails.
- **Keep only messages in memory.** Old tool calls in the history confuse the model and cost tokens, so `keep_messages` filters the session down to user and assistant messages.

## The demo app

`uv run streamlit run app.py` starts a chat with a live **Behind the scenes** panel next to it, showing each agent, model call, tool call, handoff, guardrail, code decision and memory step as it happens. The sidebar switches between mock and real data, the three orchestration modes, memory on or off, and the models. The app itself is about 75 lines, readable top to bottom.

<!-- TODO: add a screenshot of the app here -->

## Try it

The repo is at **[github.com/niksonnet/multi-agent-guardrails](https://github.com/niksonnet/multi-agent-guardrails)**. Each step is one small file you can run on its own:

```bash
uv run python -m steps.step5_guardrails
uv run python -m steps.step8_orchestration
uv run streamlit run app.py
```

If you take one thing away: let the model do what only a model can do, and check everything else in code.
