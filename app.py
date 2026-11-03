"""Multi-agent travel planner: the demo app.

Read it top to bottom: 1. settings  2. page layout  3. run the agents  4. log of the run.
Run it with: uv run streamlit run app.py
"""
import streamlit as st
from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

from core import travel_apis
from ui.chat import show_answer, show_history  # how messages are shown
from ui.loop import run  # runs async agent code inside Streamlit
from ui.panel import Workflow, add_panel_styles, show_log  # the "Behind the scenes" panel
from ui.passkey import require_passkey  # nobody reaches the agents without the passkey
from ui.sidebar import sidebar  # every setting of the demo
from ui.team import run_team  # which agents run, based on the settings

st.set_page_config(page_title="Travel Agents", page_icon="✈️", layout="wide")
require_passkey()
add_panel_styles()

# ---- 1. Settings: chosen in the sidebar ----
settings = sidebar()

# ---- 2. Page: chat on the left, "Behind the scenes" on the right ----
chat_col, panel_col = st.columns([3, 2], gap="large")
with chat_col:
    st.title("✈️ Multi-agent travel planner")
    with st.expander("ℹ️ How it works"):
        st.markdown(
            "**Your request → 🛡️ guardrail check → 🤖 agents (+ 🔧 tools, 🌐 real APIs) → 🛡️ guardrail check → answer**\n\n"
            "- **Agents**: a Flight, a Hotel and a Trip Planner agent, each an AI model with its own job.\n"
            "- **Orchestration**: who picks the agent: the AI itself or plain Python code (sidebar).\n"
            "- **Guardrails**: block requests that aren't about travel, and plans that are over budget or made up.\n"
            "- **Memory**: when on, the agents remember earlier messages (sidebar).\n\n"
            "The right panel shows every step live: agents, model calls, tool calls, handoffs and guardrails."
        )
    show_history(settings.history)
with panel_col:
    st.subheader("⚙️ Behind the scenes")
    panel = st.container()

# ---- 3. Run the agents on the new message ----
prompt = st.chat_input("Ask about flights, hotels, or plan a trip with a budget") or settings.example
if not prompt:
    panel.info("Send a message to watch the agents work, step by step.")
else:
    with chat_col:
        st.chat_message("user").write(prompt)
        answer = st.chat_message("assistant")

    wf = Workflow(panel)  # the panel; the agents' hooks and guardrails draw on it
    travel_apis.on_api_call = wf.api_call  # every real API request also appears on the panel
    try:
        with answer, st.spinner("Agents are working..."):
            output = run(run_team(settings.mode, prompt, wf, settings.models, settings.live, settings.session))
        wf.run_finished(output)
        with answer:
            show_answer(output)

    except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as e:  # a guardrail stopped the run
        wf.run_blocked(e)
        info = e.guardrail_result.output.output_info
        answer.error(f"🚫 Blocked by guardrail: {getattr(info, 'reason', info)}")

    except Exception as e:  # e.g. Groq rejected the model's output (tool_use_failed, json_validate_failed)
        wf.run_failed(e)
        answer.error(f"❌ Model error: {e}")

    finally:
        travel_apis.on_api_call = None
        st.session_state.last_log = f"Request: {prompt}\n\n{wf.as_text()}"

# ---- 4. Log of the last run: copy or download ----
show_log(panel_col)
