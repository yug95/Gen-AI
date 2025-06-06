from typing import Annotated,TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish
import operator

class ReactAgentState(TypedDict):
    """
    Represents the state of a React agent.
    This state can be extended with additional fields as needed.
    """
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]  # List of tuples containing AgentAction and its corresponding output
