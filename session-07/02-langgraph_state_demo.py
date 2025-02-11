#
# Create environment
# pip install -U langgraph langchain-openai


import os
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

memory = MemorySaver()

@tool
def get_weather(city: str) -> str:
    """Fetch weather information for a city."""
    if city.lower() == 'new york':
        return "New York weather is sunny, 25C"
    elif city.lower() == 'london':
        return "London weather is cloudy, 15C"
    else:
        return f"Sorry, I don't have weather information for {city}"
    
tools = [get_weather]

# Create a lang graph based agent

graph = create_react_agent(model, checkpointer=memory, tools=tools)

def print_stream(stream):
    """Helper function to print agent output."""

    for s in stream:
        message = s["messages"][-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


if __name__ == "__main__":
    
    thread_id = "1"
    config = { "configurable": { "thread_id": thread_id } }

    # Message 1
    print("Starting the Agent - First Interaction")
    inputs = { "messages": [("user", "Fetch weather for New York")]}
    print_stream(graph.stream( inputs, config=config, stream_mode="values"))

    # Message 2
    print("Continuing the session - Second Interaction")
    inputs = { "messages": [("user", "What did I ask for earlier?")]}
    print_stream(graph.stream( inputs, config=config, stream_mode="values"))

