from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1000)

#maintains the state of the chatbot conversation
class CustomState(TypedDict):
    messages: Annotated[list, add_messages]


def generate_post(state: CustomState) -> CustomState:
    """
    A simple function to generate a response based on the last message in the state.
    
    Args:
        state (CustomState): The current state of the chatbot.
        
    Returns:
        CustomState: The updated state with the chatbot's response appended.
    """
    return {"messages": llm.invoke(state['messages'])}

def get_review_decision(state: CustomState):
    """
    Determines whether to continue the conversation based on the last message.
    
    Args:
        state (CustomState): The current state of the chatbot.
        
    Returns:
        str: "generate_post" to continue or END to stop.
    """
    last_message = state['messages'][-1]

    print("Last message:", last_message)

    decision = input("Do you want to generate a post based on the last message? (yes/no): ").strip().lower()

    if decision == "yes":
        return "post_node"
    else:
        return "collect_feedback"
    

def collect_feedback(state: CustomState) -> CustomState:
    feedback = input("Please provide your feedback on the generated post: ")
    return {
        "messages": HumanMessage(content=feedback)
    }

def post_node(state: CustomState) -> CustomState:
    """
    A node that simulates posting the generated content.
    
    Args:
        state (CustomState): The current state of the chatbot.
        
    Returns:
        CustomState: The updated state after posting.
    """
    print("Posting the following content:")
    print(state['messages'][-1].content)
    # return state

graph = StateGraph(CustomState)
graph.add_node("generate_post", generate_post)
graph.add_node("post_node", post_node)
graph.add_node("collect_feedback", collect_feedback)
graph.set_entry_point("generate_post")

graph.add_conditional_edges("generate_post", get_review_decision)
graph.add_edge("collect_feedback", "generate_post")
graph.add_edge("post_node", END)

app = graph.compile()

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    state = CustomState(messages=[HumanMessage(content=user_input)])
    response = app.invoke(state)
    
    # Print the chatbot's response
    print("Chatbot:", response['messages'][-1].content)  # Assuming the last message is from the AI