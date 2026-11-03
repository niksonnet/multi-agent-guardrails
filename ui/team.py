"""Which agents the app runs. Not needed to learn the concepts: it only plugs the sidebar choices
(step 6 models, step 7 live data, step 8 orchestration, step 9 memory) and the panel's traced guardrails into the steps."""
from agents import Runner

from core.config import memory_config, retry_config
from steps.step3_handoffs import flight_agent, hotel_agent
from steps.step4_structured_output import planner_agent
from steps.step6_multi_model import build_team
from steps.step7_real_data import build_live_agents, build_live_team
from steps.step8_orchestration import make_manager, router_agent, run_by_code
from ui.hooks import (WorkflowHooks, save_reason, traced_grounded_live, traced_grounded_plan, traced_travel_only,
                      traced_within_budget)


async def run_team(mode, prompt, wf, models, live, session=None):
    """Run the prompt and return the final output. wf (the panel) is the run context.
    session=None: no memory. A session: the agents also see the earlier messages (step 9)."""
    hooks = WorkflowHooks()
    output_guardrails = [traced_grounded_live if live else traced_grounded_plan, traced_within_budget]
    run_config = memory_config if session else retry_config
    if session:
        wf.memory_loaded(await session.get_items())

    # The specialists, with the chosen models and the traced guardrails
    if live:
        flight, hotel, planner = build_live_agents(models, output_guardrails, save_reason)
    else:
        flight = flight_agent.clone(model=models["flight"])
        hotel = hotel_agent.clone(model=models["hotel"])
        planner = planner_agent.clone(model=models["planner"], output_guardrails=output_guardrails)

    if mode == "handoffs":  # 1. LLM + handoffs: the triage team from step 6 (mock data) or step 7 (live data)
        build = build_live_team if live else build_team
        triage = build(models, traced_travel_only, output_guardrails, save_reason)
        result = await Runner.run(triage, prompt, context=wf, hooks=hooks, run_config=run_config, session=session)
        output = result.final_output
    elif mode == "agents as tools":  # 2. LLM + agents as tools
        manager = make_manager(flight, hotel, planner, models["triage"], traced_travel_only, hooks)
        result = await Runner.run(manager, prompt, context=wf, hooks=hooks, run_config=run_config, session=session)
        output = result.final_output
    else:  # 3. Code
        router = router_agent.clone(model=models["planner"], input_guardrails=[traced_travel_only])
        output = await run_by_code(prompt, router, flight, hotel, planner, context=wf, hooks=hooks,
                                   run_config=run_config, session=session)

    if session:
        wf.memory_saved(await session.get_items())
    return output
