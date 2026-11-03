import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import ModelRetrySettings, ModelSettings, OpenAIChatCompletionsModel, RunConfig, set_tracing_disabled

load_dotenv()

# Windows terminals can crash when printing some characters (like '‑'). This fixes it.
sys.stdout.reconfigure(encoding="utf-8")

# Tracing sends data to OpenAI's website and needs an OpenAI key. We use Groq, so turn it off.
set_tracing_disabled(True)

# Groq speaks the same API as OpenAI, so we use the normal OpenAI client with Groq's URL.
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=5,  # Groq's free plan has a rate limit; the client waits and tries again.
)

# The default model the step 1-5 agents use.
model = OpenAIChatCompletionsModel(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    openai_client=client,
)

# Step 6 (multi-model): one model per kind of job.
fast_model = OpenAIChatCompletionsModel(model="openai/gpt-oss-20b", openai_client=client)  # small + fast: routing, simple tool calls
structured_model = OpenAIChatCompletionsModel(model="qwen/qwen3.8-27b", openai_client=client)  # reliable JSON (output_type)
smart_model = OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", openai_client=client)  # biggest: best reasoning, long text


# Sometimes a model writes a broken tool call or broken JSON, and Groq rejects it with an error
# (tool_use_failed / json_validate_failed). It is random: the same request usually works the next time.
def retry_bad_output(context):
    return getattr(context.error, "code", None) in ("tool_use_failed", "json_validate_failed")


# Pass it to a run: Runner.run(agent, prompt, run_config=retry_config). It applies to every agent in that run.
retry_config = RunConfig(model_settings=ModelSettings(retry=ModelRetrySettings(max_retries=2, policy=retry_bad_output)))


# Step 9 (memory): a session stores EVERYTHING from earlier runs: messages, tool calls, handoffs, reasoning.
# Old tool calls confuse gpt-oss (tool_use_failed) and cost tokens, so the agents only get the messages.
def keep_messages(history, new_input):
    return [item for item in history if item.get("role") in ("user", "assistant")] + new_input


# Use with a session: Runner.run(agent, prompt, session=session, run_config=memory_config)
memory_config = RunConfig(model_settings=retry_config.model_settings, session_input_callback=keep_messages)
