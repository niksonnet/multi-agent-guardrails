"""What gets recorded on the panel. The run context (ctx.context) is the Workflow panel."""
import html
import time

from agents import RunHooks, input_guardrail, output_guardrail

from steps.step3_handoffs import HandoffReason
from steps.step5_guardrails import grounded_plan, travel_only, within_budget
from steps.step7_real_data import grounded_live


# ---- 1. Hooks: the SDK calls these methods while the agents run ----

class WorkflowHooks(RunHooks):
    # wf.agents / wf.tools are stacks: with agents as tools (step 8) an agent runs INSIDE another agent's tool call.
    async def on_agent_start(self, context, agent):
        wf = context.context
        output_type = agent.output_type.__name__ if agent.output_type else "str (plain text)"
        wf.agents.append(wf.add("agent", "AGENT", html.escape(agent.name), [
            f"model: {getattr(agent.model, 'model', agent.model)}",
            f"tools offered: {[t.name for t in agent.tools] or '-'}",
            f"handoffs offered: {[h.tool_name for h in agent.handoffs] or '-'}",
            f"output_type: {output_type}",
        ]))

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        wf = context.context
        wf.model_calls += 1
        wf.llm_start = time.time()
        wf.agents[-1]["lines"].append(f"🧠 model call #{wf.model_calls}: sending {len(input_items)} message(s)...")
        wf.render(wf.agents[-1])

    async def on_llm_end(self, context, agent, response):
        wf = context.context
        tokens = response.usage.total_tokens
        wf.tokens += tokens
        calls = [item.name for item in response.output if getattr(item, "type", "") == "function_call"]
        decision = f"wants to call {calls}" if calls else "returned the final answer"
        wf.agents[-1]["lines"][-1] = (
            f"🧠 model call #{wf.model_calls}: {time.time() - wf.llm_start:.1f}s, {tokens} tokens → {decision}"
        )
        wf.render(wf.agents[-1])

    async def on_tool_start(self, context, agent, tool):
        wf = context.context
        wf.tools.append(wf.add("tool", "TOOL CALL", f"{html.escape(agent.name)} → <code>{tool.name}()</code>",
                               [f"arguments: {context.tool_arguments}"]))

    async def on_tool_end(self, context, agent, tool, result):
        wf = context.context
        wf.finish(wf.tools.pop(), "done", [f"returned: {str(result)[:300]}"])

    async def on_handoff(self, context, from_agent, to_agent):
        wf = context.context
        wf.finish(wf.agents.pop(), "done", ["→ handed off, this agent is done"])
        tool_name = next(h.tool_name for h in from_agent.handoffs if h.agent_name == to_agent.name)
        wf.add("handoff", "HANDOFF",
               f'{html.escape(from_agent.name)} <span class="wf-arrow">──►</span> {html.escape(to_agent.name)}',
               [f"via tool call: {tool_name}", f'reason: "{wf.handoff_reason}"',
                "conversation history is passed to the new agent"], status="done")

    async def on_agent_end(self, context, agent, output):
        wf = context.context
        wf.finish(wf.agents.pop(), "done")
        wf.answered_by = agent.name  # the last agent to finish gave the answer


# ---- 2. Guardrails have no hooks, so wrap the real ones (step 5) to also record a step ----

@input_guardrail(name="travel_only")
async def traced_travel_only(ctx, agent, message):
    wf = ctx.context
    step = wf.add("guardrail", "INPUT GUARDRAIL", "travel_only", [
        "runs a separate checker agent: Travel Checker (output_type=TravelCheck)",
        "runs in parallel with the first agent",
    ])
    result = await travel_only.guardrail_function(ctx, agent, message)
    check = result.output_info
    wf.finish(step, "blocked" if result.tripwire_triggered else "done", [
        f"is_travel = {check.is_travel}",
        f'reason: "{check.reason}"',
        f"tripwire_triggered = {result.tripwire_triggered}",
    ])
    return result


@output_guardrail(name="within_budget")
async def traced_within_budget(ctx, agent, plan):
    wf = ctx.context
    step = wf.add("guardrail", "OUTPUT GUARDRAIL", "within_budget", [
        "plain Python check on the TripPlan, no AI",
        f"{plan.flight_price_usd:.0f} + {plan.hotel_price_per_night_usd:.0f} × {plan.nights} "
        f"= {plan.flight_price_usd + plan.hotel_price_per_night_usd * plan.nights:.0f} "
        f"(the model said {plan.total_cost_usd:.0f})",
    ])
    result = await within_budget.guardrail_function(ctx, agent, plan)
    wf.finish(step, "blocked" if result.tripwire_triggered else "done", [
        result.output_info, f"tripwire_triggered = {result.tripwire_triggered}"])
    return result


@output_guardrail(name="grounded_plan")
async def traced_grounded_plan(ctx, agent, plan):
    wf = ctx.context
    step = wf.add("guardrail", "OUTPUT GUARDRAIL", "grounded_plan", [
        "plain Python: did the model make anything up?",
        f'plan says destination="{plan.destination}", budget=${plan.budget_usd:.0f}',
        "checks: city is in our data · city and budget appear in the user's message (ctx.turn_input)",
    ])
    result = await grounded_plan.guardrail_function(ctx, agent, plan)
    wf.finish(step, "blocked" if result.tripwire_triggered else "done", [
        result.output_info, f"tripwire_triggered = {result.tripwire_triggered}"])
    return result


@output_guardrail(name="grounded_live")
async def traced_grounded_live(ctx, agent, plan):
    wf = ctx.context
    step = wf.add("guardrail", "OUTPUT GUARDRAIL", "grounded_live", [
        "plain Python: is every price in the plan really in the API results?",
        f'plan says: {plan.airline} ${plan.flight_price_usd:.0f} · {plan.hotel_name} ${plan.hotel_price_per_night_usd:.0f}/night',
        "compares with the data fetch_trip_data saved in the run context (no new API calls)",
    ])
    result = await grounded_live.guardrail_function(ctx, agent, plan)
    wf.finish(step, "blocked" if result.tripwire_triggered else "done", [
        result.output_info, f"tripwire_triggered = {result.tripwire_triggered}"])
    return result


# ---- 3. Handoff callback: keep the reason for the HANDOFF step ----

def save_reason(ctx, data: HandoffReason):
    ctx.context.handoff_reason = data.reason
