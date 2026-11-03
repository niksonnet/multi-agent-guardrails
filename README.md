# multi-agent-guardrails

Learn the basics of agentic AI with the OpenAI Agents SDK, running on Groq.
Each step adds one idea to a small travel planner.

## Setup

You need Python 3.10+, [uv](https://docs.astral.sh/uv/) and a free Groq API key from [console.groq.com](https://console.groq.com/keys).

```bash
uv sync
```

Create a file called `.env` in the repo root:

```
GROQ_API_KEY=your_key
APP_PASSKEY=choose_a_passkey
```

- `GROQ_API_KEY` is needed by every step and the app.
- `APP_PASSKEY` is only for the app: it asks for it before showing the chat, so nobody without it can use your Groq quota. If it isn't set, the app stays locked. After changing `.env`, restart the app.

`.env` holds your secrets and must never be committed. `.gitignore` skips it, and a pre-commit hook blocks it even with `git add -f`. Turn the hook on once after cloning:

```bash
git config core.hooksPath .githooks
```

## Project layout

```
app.py          # the Streamlit app (chat + "Behind the scenes" panel)
core/           # shared pieces, no agents
  config.py       # Groq client and models
  mock_data.py    # mock flights/hotels + the step 2-5 tools
  travel_apis.py  # real APIs with cache and fallbacks
steps/          # the learning path: one idea per file
ui/             # app helpers, one job each:
  passkey.py      # the passkey screen before the chat
  sidebar.py      # every setting of the demo, returned as one Settings object
  team.py         # which agents run for those settings
  hooks.py        # what the panel records (RunHooks + traced guardrails)
  panel.py        # how the panel is drawn, and the log copy / download
  chat.py         # how messages are shown
  loop.py         # runs async agent code inside Streamlit
```

## Steps

| Step | File | What you learn |
|------|------|----------------|
| 1 | `steps/step1_basic_agent.py` | **Agent**: a name, instructions and a model |
| 2 | `steps/step2_function_tools.py` | **Function tools**: the agent calls Python functions (`core/mock_data.py`) |
| 3 | `steps/step3_handoffs.py` | **Multi-agent + handoffs**: a triage agent passes the user to a Flight or Hotel agent |
| 4 | `steps/step4_structured_output.py` | **Structured output**: the agent returns a `TripPlan` object instead of text |
| 5 | `steps/step5_guardrails.py` | **Guardrails**: block requests that aren't about travel (input) and trips over budget (output) |
| 6 | `steps/step6_multi_model.py` | **Multi-model**: each agent gets the model that fits its job. Running it compares strategies |
| 7 | `steps/step7_real_data.py` | **Real APIs**: live flight and hotel prices (`core/travel_apis.py`), a Trip Planner → Plan Writer pipeline, and a guardrail that checks every price against the API results |
| 8 | `steps/step8_orchestration.py` | **Orchestration**: the same agents run three ways: LLM with handoffs, LLM with agents as tools, and code. Running it compares them |
| 9 | `steps/step9_memory.py` | **Memory**: a conversation where the details come one message at a time. Without a session it never gets a plan; with `SQLiteSession` it does |
| — | `steps/demo.py` | Steps 1–5 together |
| — | `app.py` | The demo app (~75 lines, read top to bottom: passkey → settings → page → run the agents → log). A chat on the left; on the right a live **Behind the scenes** panel showing each agent, model call, tool call, handoff, guardrail, code decision and memory step (built with `RunHooks`). The sidebar switches data, orchestration, memory and (under Advanced) models; the log of each run can be copied or downloaded |

Run everything from the repo root:

```bash
uv run python -m steps.step1_basic_agent
uv run python -m steps.demo
uv run streamlit run app.py
```

`core/config.py` connects to Groq and holds the models. Steps 1–5 use `model`; from step 6 each agent gets `fast_model`, `structured_model` or `smart_model`.

## Orchestration: who decides what runs next? (step 8)

| | LLM · handoffs | LLM · agents as tools | Code |
|---|---|---|---|
| Who decides | Triage Agent | Manager Agent | Your `if/elif` |
| Control | Given away to the specialist | Comes back to the manager | Stays in Python |
| SDK feature | `handoffs=[...]` | `agent.as_tool(...)` | `Runner.run` called by your code |
| Strengths | Flexible, handles odd requests | Manager can combine answers | Predictable, fewest model calls, easy to debug |
| Weaknesses | Can route wrongly, extra model calls | Most model calls | Only handles the paths you wrote |

In code mode a Router agent only *reads* the request (`output_type=Route`) and a plain `if/elif` picks the agent. `ui/team.py` only connects the sidebar choices to these steps; you don't need it to learn the concept. The rule "no trip plan without a budget" is an `if` in Python too, so the model can't skip it.

## Memory (step 9)

| Kind | Remembers | Lasts | How |
|---|---|---|---|
| Run context | Data shared by your code during one request (`fetched_trip`, the panel) | One `Runner.run` | `context=...` |
| Session | The conversation | Until you clear it | `Runner.run(..., session=SQLiteSession(id, "file.db"))` |
| Long-term | Facts about the user ("home city: Delhi") | Forever | Not built in: a tool that saves facts + instructions that load them |

- A session stores everything: messages, tool calls, handoffs, reasoning. `keep_messages` in `core/config.py` (used through `memory_config`) gives the agents only the messages, because old tool calls confuse gpt-oss and cost tokens.
- With a session, the input guardrail gets the whole conversation. `travel_only` checks the newest message and shows the checker the earlier ones, so a follow-up like "My budget is $1000" still counts as travel.
- In the app, Memory On keeps the conversation in `.cache/memory.db`. Its id is in the URL (`?chat=...`), so a page refresh keeps it. **New conversation** clears it.

## Real travel data (step 7)

All these keys are optional: add them to `.env` for real prices. Without `SERPAPI_KEY` the app uses the sample data; each other missing key switches its feature off.

| Key | Used for | Free allowance |
|---|---|---|
| `SERPAPI_KEY` | Live Google Flights + Google Hotels prices | 250 searches/month |
| `TRAVELPAYOUTS_TOKEN` (optional) | Backup flight prices (cached, up to 7 days old) | Free |
| `OPENTRIPMAP_KEY` (optional) | Things to do | Free |

- Weather comes from Open-Meteo, which needs no key. It only has forecasts for dates up to 16 days ahead.
- The same search within 6 hours is read from `.cache/api_cache.json` and uses no quota.
- If an API fails, the code tries the next source and finally falls back to the mock data.
- Test the APIs with `uv run python -m core.travel_apis`.

## Things to know about Groq

- **Models:** the default `model` is `openai/gpt-oss-120b` (change it with `GROQ_MODEL` in `.env`). Agents with structured output use `qwen/qwen3.8-27b`, because `gpt-oss` models often fail with `json_validate_failed`.
- **Broken model output:** sometimes a model writes a broken tool call or broken JSON and Groq rejects it (`tool_use_failed`, `json_validate_failed`). `retry_config` in `core/config.py` retries these up to 2 times. A guardrail's checker agent is a separate run, so it gets `run_config=retry_config` too.
- **Forced tool calls:** with `tool_choice="required"`, a model that needs to ask the user a question can't, and Groq rejects the run. So the Flight and Hotel agents may ask for a missing city instead of always calling a tool.
- **Handoffs need an argument:** Groq rejects handoffs that take no arguments, so each handoff asks for a `reason` (see step 3).
- **Tools or structured output, not both:** one agent can't use tools and `output_type` together on Groq. In step 4 the Trip Planner gets the prices in its instructions; in step 7 a Trip Planner calls the tools and a Plan Writer writes the JSON.
- **Rate limit:** the free plan limits tokens per minute (qwen's output limit is the easiest to hit). The client waits and tries again; in a demo, leave a few seconds between requests.
