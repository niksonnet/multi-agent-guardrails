from agents import Agent, Runner

from core.config import model
from core.mock_data import search_flights, search_hotels

# Give the agent tools. The model decides when to call them,
# and the SDK runs the Python function and gives the result back to the model.
agent = Agent(
    name="Travel Assistant",
    instructions="You help users plan trips. Always use the tools to look up flights and hotels.",
    model=model,
    tools=[search_flights, search_hotels],
)

if __name__ == "__main__":
    result = Runner.run_sync(agent, "Find me a flight from Delhi to Tokyo and a hotel for 3 nights.")
    print(result.final_output)
