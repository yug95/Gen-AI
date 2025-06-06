from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph

from react_nodes import reason_node, act_node
from react_agent_state import ReactAgentState
from dotenv import load_dotenv
load_dotenv()


REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state: ReactAgentState):
    """
    Determine whether to continue based on the agent's outcome.
    
    Args:
        state (ReactAgentState): The current state of the React agent.
        
    Returns:
        str: "reason_node" to continue reasoning, or END to finish.
    """
    if isinstance(state['agent_outcome'], AgentFinish):
        return END
    else:
        return ACT_NODE
    

flow = StateGraph(ReactAgentState)
flow.add_node(REASON_NODE, reason_node)
flow.set_entry_point(REASON_NODE)
flow.add_node(ACT_NODE, act_node)


flow.add_conditional_edges(REASON_NODE, should_continue)
flow.add_edge(ACT_NODE,REASON_NODE)


app = flow.compile()
# Example usage
state = ReactAgentState(input="How many days ago was the latest SpaceX launch?", agent_outcome=None, intermediate_steps=[])
response = app.invoke(state)

print(response)  # Should print the final state with the agent's outcome and intermediate steps
# print(response["agent_outcome"])