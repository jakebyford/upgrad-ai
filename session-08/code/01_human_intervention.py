# environment creation
# python -m venv venv
# source venv/bin/activate
# pip install -U langchain langgraph langchain-openai

import os 
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

## echo $OPENAI_API_KEY
# export OPENAI_API_KEY="your_openai_api_key_here"

model = ChatOpenAI(model="gpt-4", temperature=0)

@tool
def get_weather(city: str) -> str:
    """Fetch weather info for a city"""

    if city.lower() == "new york":
        return "The weather in New York is sunny, 25C"
    elif city.lower() == "london":
        return "The weather in London is cloudy, 15C"
    else:
        return f"unknown city: {city}"
    
@tool
def ask_human_clarification(query: str) -> str:
    """Ask the human for clarification"""
    clarification = input(f"Enter the correct city name -->")
    return clarification

tools = [get_weather, ask_human_clarification]

graph = create_react_agent(model, tools=tools)

def print_stream(stream):
    
    """Helper function to print agent output"""
    
    for s in stream:
        
        message = s["messages"][-1]

        if isinstance(message, tuple):
            
            print(message)
        else:
            message.pretty_print()

if __name__ == "__main__":

    print("Starting Human Intervention Demo")

    print("Step 1 : Fetch weather for New York")
    inputs = { "messages": [ ( "user", "Fetch weather for New York" ) ] }
    print_stream(graph.stream( inputs, stream_mode="values"))

    print("Step 2 : Fetch weather for Unknown City")
    inputs = { "messages": [ ( "user", "Fetch weather for Atlantis" ) ] }
    print_stream(graph.stream( inputs, stream_mode="values"))

    print("Step 3 : Retry with clarification")
    clarified_city = "Los Angeles"
    inputs = { "messages": [ ( "user", f"Fetch weather for {clarified_city}" ) ] }
    print_stream(graph.stream( inputs, stream_mode="values"))

