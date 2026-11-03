"""The sidebar: every setting of the demo, returned as one Settings object."""
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from agents import SQLiteSession

from core import travel_apis
from core.config import fast_model, smart_model, structured_model
from steps.demo import DEMO_PROMPTS
from steps.step6_multi_model import MULTI_MODEL, single_model
from steps.step7_real_data import LIVE_PROMPTS
from ui.loop import run

ORCHESTRATION = {  # value -> (label, explanation)
    "handoffs": ("AI hands off (handoffs)",
                 "A triage agent **hands the conversation over** to a specialist and is done."),
    "agents as tools": ("AI manager calls helpers (agents as tools)",
                        "A manager agent **calls specialists like tools** and gets their answers back."),
    "code": ("Python code (if/elif)",
             "A router agent only reads the request; **your if/elif** picks the agent."),
}


@dataclass
class Settings:
    live: bool  # real prices (True) or sample data
    mode: str  # one of ORCHESTRATION
    models: dict  # which model each agent uses
    session: SQLiteSession | None = None  # the conversation memory (None = memory off)
    history: list = field(default_factory=list)  # earlier messages, to show in the chat
    example: str | None = None  # an example button that was clicked


@st.cache_data(ttl=60)  # checking the quota is free, but don't do it on every click
def searches_left():
    return travel_apis.serpapi_searches_left()


def sidebar():
    with st.sidebar:
        # 1. Where the prices come from
        st.header("Data")
        live = st.radio("Data", ["live", "sample"], index=0 if travel_apis.SERPAPI_KEY else 1,
                        format_func={"live": "Real prices (live APIs)", "sample": "Sample data"}.get,
                        label_visibility="collapsed") == "live"
        if live and not travel_apis.SERPAPI_KEY:
            st.warning("No SERPAPI_KEY in .env, so searches fall back to sample data.")
        elif live:
            st.caption(f"{searches_left()} free searches left this month")

        # 2. Orchestration
        st.header("Who decides the next step?")
        mode = st.radio("Orchestration", list(ORCHESTRATION), format_func=lambda m: ORCHESTRATION[m][0],
                        label_visibility="collapsed")
        st.caption(ORCHESTRATION[mode][1])

        # 3. Memory
        st.header("Memory")
        session, history = None, []
        if st.toggle("Remember the conversation", help="A session (SQLiteSession) stores the chat, "
                                                       "so follow-up messages keep their context."):
            # One id = one conversation. It lives in the URL, so a page refresh keeps the conversation.
            if "chat" not in st.query_params:
                st.query_params["chat"] = uuid.uuid4().hex[:8]
            Path(".cache").mkdir(exist_ok=True)
            session = SQLiteSession(st.query_params["chat"], ".cache/memory.db")
            if st.button("🗑 New conversation", use_container_width=True):
                run(session.clear_session())
            history = [item for item in run(session.get_items()) if item.get("role") in ("user", "assistant")]
            st.caption(f"{len(history)} messages in memory")
        else:
            st.caption("Off: every message starts from zero.")

        # 4. Examples
        st.header("Try an example")
        example = None
        for prompt in LIVE_PROMPTS if live else DEMO_PROMPTS:
            if st.button(prompt, use_container_width=True):
                example = prompt

        # 5. Advanced: which model each agent uses
        with st.expander("⚙️ Advanced: models"):
            strategy = st.radio("Model strategy", ["Multi-model", "Single model"],
                                help="Multi-model: each agent gets the model that fits its job.")
            if strategy == "Multi-model":
                models = MULTI_MODEL
                st.caption("Triage, Flight, Hotel → gpt-oss-20b (fast)  \nTrip Planner → qwen3.8-27b (reliable JSON)")
            else:
                single = {"openai/gpt-oss-120b": smart_model, "qwen/qwen3.8-27b": structured_model,
                          "openai/gpt-oss-20b": fast_model}
                models = single_model(single[st.selectbox("Model for every agent", list(single))])
            st.caption("The input guardrail's checker always uses qwen (it returns JSON).")

    return Settings(live, mode, models, session, history, example)
