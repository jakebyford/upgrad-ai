#
# create environment
# python -m venv venv
# source venv/Scrips/activate
# pip install openai langchain serpapi langchain-community google-search-results
# export OPENAI_API_KEY=<key>
# to see the key: echo $OPENAI_API_KEY
# export SERPAPI_API_KEY=<key>
# key = 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


import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
import requests

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
# print(requests.get('https://api.openai.com').status_code)

serpapi_key = os.getenv("SERPAPI_API_KEY")
openapi_key = os.getenv("OPENAI_API_KEY")

def calculator(query:str) -> str:
    try:
        result = eval(query)
        return f"Result of {query} is {result}"
    except Exception as e:
        print(f"Error: {e}")
        return "Error"
    

calc_tool = Tool(
    name="Calculator",
    func= calculator,
    description="This tool will calculate basic math operations"
)

calc_response = calc_tool.run( "5 * (12 + 4) - 10")

print(f"Calculation response: {calc_response}")

search_tool_wrapper = SerpAPIWrapper()

def search(query:str) -> str:
    try:
        results = search_tool_wrapper.run(query)
        return results
    except Exception as e:
        print(f"Error: {e}")
        return "Error"
    
search_tool = Tool(
    name="Search",
    func= search,
    description="Seacrh the web using SerpAPI"
)

search_response = search_tool.run("What is the capital of France?")
print(f"Search response: {search_response}")

def mock_weather_api(location:str) -> str:
    mock_data = {
        "New York": "Sunny, 25C",
        "Paris": "Rainy, 15C",
        "London": "Cloudy, 20C"
    }

    return f"The weather in {location} is {mock_data.get(location, 'Unknown')}"

weather_tool = Tool(
    name="Weather",
    func= mock_weather_api,
    description="Provides weather updates for cities"
)

weather_response = weather_tool.run("Paris")
print(f"Weather response: {weather_response}")


tools = [calc_tool, search_tool, weather_tool]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Finally test it 

query = "Hey, how are you?"
agent_response = agent.run(query)
print(f"\nResponse 1 - :\n{agent_response}")

query = "Tell me the result of  15 * 3"
agent_response = agent.run(query)
print(f"\nResponse 2 - :\n{agent_response}")

query = "Tell me the weather in London and the result of  15 * 3"
agent_response = agent.run(query)
print(f"\nResponse 3 - :\n{agent_response}")
