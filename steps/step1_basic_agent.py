from agents import Agent, Runner

from core.config import model

# An agent = a name + instructions + a model.
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Answer concisely.",
    model=model,
)

result = Runner.run_sync(agent, "What is a multi-agent system? Answer in 2 sentences.")
print(result.final_output)
