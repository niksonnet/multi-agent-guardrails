"""How messages are shown in the chat."""
import json

import streamlit as st


def show_answer(output):
    """A trip plan (a Pydantic object) is shown as JSON, anything else as text."""
    if hasattr(output, "model_dump"):
        st.json(output.model_dump())
    else:
        st.write(output)


def show_message(content):
    """A message from memory: plain text, or a list of text parts. A plan (JSON text) is shown as JSON."""
    text = content if isinstance(content, str) else "".join(part.get("text", "") for part in content)
    try:
        st.json(json.loads(text))
    except ValueError:
        st.write(text)


def show_history(history):
    """The earlier messages of the conversation (only when memory is on)."""
    for item in history:
        with st.chat_message(item["role"]):
            show_message(item["content"])
