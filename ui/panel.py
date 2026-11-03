"""The "Behind the scenes" panel: how each step is drawn. No agent logic here."""
import html
import re
import time

import streamlit as st

# Timeline styles: pulsing dot (running), slide-in (new step), shake (blocked), moving arrow (handoff)
CSS = """
<style>
.wf-step { position: relative; padding: 0 0 14px 26px; border-left: 2px solid rgba(128,128,128,.3); margin-left: 8px; }
.wf-step:last-child { border-left-color: transparent; }
.wf-new { animation: wf-in .45s ease-out; }
@keyframes wf-in { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: none; } }
.wf-dot { position: absolute; left: -9px; top: 2px; width: 16px; height: 16px; border-radius: 50%; background: #9ca3af; }
.wf-running .wf-dot { background: #3b82f6; animation: wf-pulse 1s infinite; }
.wf-done .wf-dot { background: #22c55e; }
.wf-blocked .wf-dot, .wf-failed .wf-dot { background: #ef4444; }
.wf-cancelled .wf-dot { background: #9ca3af; }
@keyframes wf-pulse { 0% { box-shadow: 0 0 0 0 rgba(59,130,246,.7); } 100% { box-shadow: 0 0 0 10px rgba(59,130,246,0); } }
.wf-card { border: 1px solid rgba(128,128,128,.25); border-radius: 8px; padding: 8px 10px; }
.wf-blocked .wf-card, .wf-failed .wf-card { border-color: #ef4444; background: rgba(239,68,68,.08); animation: wf-shake .4s; }
@keyframes wf-shake { 25% { transform: translateX(-4px); } 75% { transform: translateX(4px); } }
.wf-head { display: flex; justify-content: space-between; font-size: 11px; letter-spacing: .06em; opacity: .7; }
.wf-title { font-weight: 600; margin: 2px 0 4px; }
.wf-line { font-family: monospace; font-size: 12px; opacity: .85; white-space: pre-wrap; word-break: break-word; }
.wf-arrow { display: inline-block; color: #a855f7; animation: wf-slide 1s ease-in-out infinite alternate; }
@keyframes wf-slide { from { transform: translateX(-3px); } to { transform: translateX(5px); } }
</style>
"""

ICONS = {"guardrail": "🛡️", "agent": "🤖", "tool": "🔧", "handoff": "🔀", "api": "🌐", "code": "🐍", "memory": "🧠", "final": "◉"}
STATUS = {"running": "⟳ running", "done": "✅", "blocked": "🚫 TRIPWIRE", "failed": "❌ failed", "cancelled": "⏹ cancelled"}


def add_panel_styles():
    st.markdown(CSS, unsafe_allow_html=True)


def show_log(container):
    """The log of the last run (saved by app.py as st.session_state.last_log), to copy or download."""
    if "last_log" not in st.session_state:
        return
    with container, st.expander("📋 Behind-the-scenes log (copy / download)"):
        st.code(st.session_state.last_log, language=None, wrap_lines=True)  # copy icon at the top right
        st.download_button("⬇ Download .txt", st.session_state.last_log, file_name="agents-log.txt",
                           on_click="ignore", use_container_width=True)


class Workflow:
    """A live timeline of steps. Each step gets its own placeholder, so only that step re-renders."""

    def __init__(self, container):
        self.container = container
        self.steps = []
        self.agents = []  # agents that are working now (the last one is the innermost)
        self.tools = []  # tool calls that are running now
        self.answered_by = ""
        self.memory_before = 0  # step 9: how many items the session held before this run
        self.handoff_reason = ""
        self.fetched_trip = None  # step 7: fetch_trip_data saves the real data here for the grounded_live guardrail
        self.model_calls = 0
        self.tokens = 0
        self.start = time.time()

    def add(self, kind, label, title, lines, status="running"):
        step = {"kind": kind, "label": label, "title": title, "lines": list(lines), "status": status,
                "start": time.time(), "end": None, "box": self.container.empty(), "new": True,
                "number": len(self.steps) + 1}
        self.steps.append(step)
        self.render(step)
        return step

    def finish(self, step, status="done", lines=()):
        step["status"] = status
        step["lines"] += list(lines)
        step["end"] = time.time()
        self.render(step)

    def render(self, step):
        took = f" {step['end'] - step['start']:.1f}s" if step["end"] and step["kind"] != "final" else ""
        lines = "".join(f'<div class="wf-line">{html.escape(str(line))}</div>' for line in step["lines"])
        step["box"].markdown(f"""
<div class="wf-step wf-{step['status']} {'wf-new' if step['new'] else ''}">
  <div class="wf-dot"></div>
  <div class="wf-card">
    <div class="wf-head"><span>{ICONS[step['kind']]} {step['number']}. {step['label']}</span>
      <span>{STATUS[step['status']]}{took}</span></div>
    <div class="wf-title">{step['title']}</div>{lines}
  </div>
</div>""", unsafe_allow_html=True)
        step["new"] = False

    def api_call(self, event):
        """Called by travel_apis.py for every real API request (set as travel_apis.on_api_call)."""
        if event["error"]:
            result = f"error: {event['error']} → trying the next source"
        elif event["cached"]:
            result = "served from cache (no quota used)"
        else:
            result = f"live response in {event['seconds']:.1f}s"
        self.add("api", "API CALL", html.escape(event["provider"]),
                 [f"GET {event['endpoint']}", f"params: {event['params']}", result],
                 status="failed" if event["error"] else "done")

    def code_step(self, text):
        """Step 8, code mode: what the Python if/elif decided (no model involved)."""
        self.add("code", "CODE DECIDES", "your if/elif", [text], status="done")

    def memory_loaded(self, items):
        """Step 9: what the session gave the agents before the run."""
        messages = [item for item in items if item.get("role") in ("user", "assistant")]
        self.memory_before = len(items)
        self.add("memory", "MEMORY", "load from session", [
            f"session holds {len(items)} items from earlier turns",
            f"agents get {len(messages)} messages + the new one "
            f"({len(items) - len(messages)} tool calls / handoffs / reasoning dropped by keep_messages)",
        ], status="done")

    def memory_saved(self, items):
        """Step 9: what the run added to the session."""
        self.add("memory", "MEMORY", "save to session", [
            f"{len(items) - self.memory_before} new items saved (messages, tool calls, handoffs...)",
            f"session now holds {len(items)} items",
        ], status="done")

    def as_text(self):
        """The whole timeline as plain text, for the copy / download button."""
        out = []
        for step in self.steps:
            took = f" {step['end'] - step['start']:.1f}s" if step["end"] and step["kind"] != "final" else ""
            title = html.unescape(re.sub(r"<[^>]+>", "", step["title"]))  # titles contain a little HTML
            out.append(f"{step['number']}. [{step['label']}] {title}  ({step['status']}{took})")
            out += [f"     {line}" for line in step["lines"]]
        return "\n".join(out)

    def run_finished(self, output):
        self.add("final", "FINAL OUTPUT", f"answered by {html.escape(self.answered_by)}", [
            f"type: {type(output).__name__}",
            f"{self.model_calls} model calls · {self.tokens} tokens · {time.time() - self.start:.1f}s total",
        ], status="done")

    def run_failed(self, error):
        for step in self.steps:
            if step["status"] == "running":
                self.finish(step, "failed")
        self.add("final", "MODEL ERROR", "the model gave an answer the API rejected", [
            f"error: {getattr(error, 'code', None) or type(error).__name__}",
            "try another model strategy in the sidebar",
        ], status="failed")

    def run_blocked(self, error):
        for step in self.steps:
            if step["status"] == "running":
                self.finish(step, "cancelled")
        self.add("final", "RUN STOPPED", "a guardrail tripwire fired", [
            f"exception: {type(error).__name__}",
            "the user never sees the agent's answer",
        ], status="blocked")
