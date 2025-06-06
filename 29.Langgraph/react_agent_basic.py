from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain.agents import initialize_agent,tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatOpenAI()
# print(llm.invoke("Give me today's weather in bangalore"))

search_tool = TavilySearchResults()

@tool
def get_system_time(format : str = "%Y-%m-%d %H:%M:%S"):
    """ returns the system time in the given format"""

    current_time = datetime.datetime.now()
    return current_time.strftime(format)

agent = initialize_agent(tools = [search_tool, get_system_time], llm =llm, agent_type = 'zero-shot-react-description', verbose = True)

# agent.invoke("Give me a tweet about today's weather in bangalore")
agent.invoke("when was the last spacex launch and how many days ago since this instant")