"""Run async agent code from Streamlit."""
import asyncio

import streamlit as st


# Streamlit re-runs app.py on every click. A new event loop each time breaks the Groq client
# created on the first one, so we keep ONE loop and reuse it.
@st.cache_resource
def get_event_loop():
    return asyncio.new_event_loop()


def run(coroutine):
    """Wait for an async call, e.g. run(Runner.run(...))."""
    return get_event_loop().run_until_complete(coroutine)
