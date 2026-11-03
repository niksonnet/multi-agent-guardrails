"""A passkey screen before the chat, so only people with the key can spend our Groq quota."""
import hmac
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def require_passkey():
    """Show the passkey form and stop the page until the right key is entered."""
    if st.session_state.get("unlocked"):
        return

    st.title("🔒 Travel Agents")
    passkey = os.environ.get("APP_PASSKEY")
    if not passkey:  # no key set means nobody gets in, never an open app
        st.error("The app is locked: set APP_PASSKEY in the .env file and restart.")
        st.stop()

    with st.form("passkey"):
        entered = st.text_input("Passkey", type="password")
        if st.form_submit_button("Enter"):
            # compare_digest takes the same time for any wrong key, so the key can't be guessed letter by letter
            if hmac.compare_digest(entered.encode(), passkey.encode()):
                st.session_state.unlocked = True
                st.rerun()
            st.error("Wrong passkey.")
    st.stop()  # nothing below this line runs: no sidebar, no chat, no model calls
