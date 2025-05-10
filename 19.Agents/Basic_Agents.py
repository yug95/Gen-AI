from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from dotenv import load_dotenv

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

search_tool = DuckDuckGoSearchRun()

# result = search_tool.invoke("what is top news of india ?")

# print(result)


@tool
def get_weather(city: str) -> str:
    """
    This function fetches the latest weather of given city
    """
    url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

    response = requests.get(url)

    return response.json()

llm = ChatOpenAI()

# pull reAct prompt from the lang_chain hub

prompt = hub.pull("hwchase17/react")  #pulls the react prompt

print(prompt)

#create react agent 

agent = create_react_agent(
    tools = [search_tool, get_weather],
    llm= llm,
    prompt= prompt
)


# create an agent executor

agent_exe = AgentExecutor(
    tools=[search_tool, get_weather],
    agent= agent,
    verbose=True
)

response = agent_exe.invoke({"input":"what is the capital of india and its current weather condition"})

print(response)
