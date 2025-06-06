from typing import TypedDict,List
from langgraph.graph import END, StateGraph

class SimpleState(TypedDict):
    """
    A simple state representation for a Langgraph application.
    This state can be extended with additional fields as needed.
    """
    count : int
    sum : int
    history: List[int]


def increment_state(state: SimpleState) -> SimpleState:
    """
    Increment the count in the state by 1.
    
    Args:
        state (SimpleState): The current state.
        
    Returns:
        SimpleState: The updated state with incremented count.
    """
    state['count'] += 1
    state['sum'] = state['sum'] + state['count']
    state['history'].append(state['count'])
    return state


def should_continue(state: SimpleState):
    """
    Determine whether to continue based on the count in the state.
    
    Args:
        state (SimpleState): The current state.
        
    Returns:
        bool: True if count is less than 5, False otherwise.
    """
    if state['count'] < 5:
        return "increment"
    else:
        return END
    

graph = StateGraph(SimpleState)
graph.add_node("increment", increment_state)
graph.set_entry_point("increment")
graph.add_conditional_edges("increment", should_continue)

app = graph.compile()
# Example usage
state = SimpleState(count=0, sum=0, history=[])
response = app.invoke(state)
print(response)  # Should print the final state with count incremented to 5