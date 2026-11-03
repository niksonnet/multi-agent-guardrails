import asyncio

from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, Runner, SQLiteSession

from core.config import memory_config, retry_config
from steps.step6_multi_model import MULTI_MODEL, build_team

# Memory = the agents remember earlier messages of the conversation.
#
# Without it, every Runner.run starts from zero: the agent only sees the message you pass in.
# With a session, Runner.run does the remembering for you:
#   1. loads the earlier messages from the session
#   2. runs the agents with: earlier messages + the new one
#   3. saves the new messages (and tool calls, handoffs...) back to the session
#
# memory_config (core/config.py) trims what is loaded to just the messages (keep_messages),
# so old tool calls don't confuse the model.

team = build_team(MULTI_MODEL)

# The details come one message at a time, like in a real chat.
CONVERSATION = [
    "I want to plan a trip in December 2026.",
    "From Delhi to Dubai, 3 nights.",
    "My budget is $1000.",
]


async def chat(session=None):
    for message in CONVERSATION:
        print(f"\n>>> {message}")
        try:
            if session:
                result = await Runner.run(team, message, session=session, run_config=memory_config)
            else:
                result = await Runner.run(team, message, run_config=retry_config)
            print(f"[{result.last_agent.name}] {result.final_output}")
        except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as e:
            print("BLOCKED by guardrail:", type(e).__name__)

        if session:
            items = await session.get_items()
            messages = [item for item in items if item.get("role") in ("user", "assistant")]
            print(f"  (session holds {len(items)} items, {len(messages)} of them are messages)")


async def main():
    print("=" * 60, "\nWITHOUT memory: each message is on its own")
    await chat()

    print("\n" + "=" * 60, "\nWITH memory: SQLiteSession")
    # One id = one conversation. ':memory:' (the default) is lost when the script ends;
    # pass a file name, e.g. SQLiteSession("user-123", "memory.db"), to keep it.
    await chat(SQLiteSession("user-123"))


if __name__ == "__main__":
    asyncio.run(main())
