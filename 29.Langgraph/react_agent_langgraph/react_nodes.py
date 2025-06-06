from dotenv import load_dotenv

load_dotenv()
from react_agent_state import ReactAgentState
from langchain_core.agents import AgentAction, AgentFinish
from react_agent_runnable import react_agent_runnable, tools


def reason_node(state: ReactAgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    # print(f"Agent Outcome: {agent_outcome}")
    return {"agent_outcome":agent_outcome}



def act_node(state: ReactAgentState):
    agent_action  = state["agent_outcome"]

    # print(f"agent_action: {agent_action}")

    # if isinstance(agent_action, AgentAction):
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    # print(f"Tool Name: {tool_name}")
    # print(f"Tool Input: {tool_input}")

    tool_function = None
    for tool in tools:
        # print(f"Available Tool: {tool.name}")
        if tool.name == tool_name:
            tool_function = tool
            break

    # Execute the tool with the input
    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"Tool '{tool_name}' not found"

    # else:
    #     output = "No action taken, agent finished."
        
    return {"intermediate_steps":[(agent_action,str(output))]}
