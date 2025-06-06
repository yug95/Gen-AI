from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent,tool
import datetime
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1000)

search_tool = TavilySearchResults()

@tool
def get_system_time(format : str = "%Y-%m-%d %H:%M:%S"):
    """ returns the system time in the given format"""

    current_time = datetime.datetime.now()
    return current_time.strftime(format)

react_prompt = hub.pull("hwchase17/react")

tools=[search_tool, get_system_time]

react_agent_runnable = create_react_agent(llm, tools=[search_tool, get_system_time], prompt= react_prompt)
