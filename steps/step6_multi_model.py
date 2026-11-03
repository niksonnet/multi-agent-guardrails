import time

from agents import Runner, handoff
from agents.extensions import handoff_filters

from core.config import fast_model, smart_model, structured_model
from steps.demo import triage_agent
from steps.step3_handoffs import HandoffReason, flight_agent, hotel_agent, print_reason
from steps.step4_structured_output import planner_agent
from steps.step5_guardrails import grounded_plan, travel_only, within_budget

# Multi-model: give each agent the model that fits its job.
#   Triage  -> picks 1 of 3 handoffs      -> small + fast
#   Flight  -> 1 tool call + short answer -> small + fast
#   Hotel   -> 1 tool call + short answer -> small + fast
#   Planner -> maths + TripPlan JSON      -> reliable at structured output
MULTI_MODEL = {"triage": fast_model, "flight": fast_model, "hotel": fast_model, "planner": structured_model}


def single_model(model):
    """The opposite: every agent uses the same model."""
    return {"triage": model, "flight": model, "hotel": model, "planner": model}


def build_team(models, input_guardrail=travel_only, output_guardrails=(grounded_plan, within_budget),
               on_handoff=print_reason):
    """The same agents as demo.py, each cloned with the model chosen for it. Returns the triage agent."""
    flight = flight_agent.clone(model=models["flight"])
    hotel = hotel_agent.clone(model=models["hotel"])
    planner = planner_agent.clone(model=models["planner"], output_guardrails=list(output_guardrails))
    # input_filter=remove_all_tools: the new agent gets the history WITHOUT the triage agent's tool calls.
    # Without it, gpt-oss sees "transfer_to_flight_agent" in the history and replies in text instead of
    # calling its own tool (Groq error: tool_use_failed). It also means fewer tokens for the new agent.
    return triage_agent.clone(
        model=models["triage"],
        input_guardrails=[input_guardrail],
        handoffs=[
            handoff(a, input_type=HandoffReason, on_handoff=on_handoff, input_filter=handoff_filters.remove_all_tools)
            for a in (flight, hotel, planner)
        ],
    )


if __name__ == "__main__":
    # Benchmark: run the same requests with each strategy and compare.
    strategies = {
        "single: gpt-oss-120b": single_model(smart_model),
        "single: qwen3.8-27b": single_model(structured_model),
        "single: gpt-oss-20b": single_model(fast_model),
        "multi-model": MULTI_MODEL,
    }
    prompts = [
        "I need a flight from Delhi to Tokyo.",
        "Show me hotels in London for 2 nights.",
        "Plan a trip from Delhi to Dubai for 3 nights. My budget is $1000.",
    ]

    rows = []
    for name, models in strategies.items():
        team = build_team(models)
        ok, seconds, tokens, errors = 0, 0.0, 0, []
        for prompt in prompts:
            print(f"\n[{name}] {prompt}")
            start = time.time()
            try:
                result = Runner.run_sync(team, prompt)
                ok += 1
                tokens += result.context_wrapper.usage.total_tokens
            except Exception as e:
                errors.append(getattr(e, "code", None) or type(e).__name__)
            seconds += time.time() - start
        rows.append((name, ok, seconds / len(prompts), tokens, errors))

    print(f"\n{'strategy':<22} {'passed':>7} {'avg time':>9} {'tokens':>7}  errors")
    for name, ok, avg, tokens, errors in rows:
        print(f"{name:<22} {ok:>4}/{len(prompts)} {avg:>8.1f}s {tokens:>7}  {', '.join(errors) or '-'}")
    print("\nNote: tokens count only successful runs. Time includes any rate-limit waits.")
