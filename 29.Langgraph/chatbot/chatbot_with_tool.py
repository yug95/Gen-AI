from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode

llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1000)

load_dotenv()


class ChatbotState(TypedDict):
    messages : Annotated[list, add_messages]


search_tool = TavilySearchResults()
tools = [search_tool]

llm_with_tools = llm.bind_tools(tools=tools)
# print(llm_with_tools.invoke("what is current weather in bangalore?"))

def chat_bot_node(state: ChatbotState) -> ChatbotState:
    """
    A simple chatbot node that appends a response to the messages in the state.
    
    Args:
        state (ChatbotState): The current state of the chatbot.
        
    Returns:
        ChatbotState: The updated state with the chatbot's response appended.
    """
    
    return {"messages": llm_with_tools.invoke(state['messages'])}

def tools_router(state: ChatbotState):
    last_message = state['messages'][-1]
    if (hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return END
    
tool_node = ToolNode(tools = tools)

flow = StateGraph(ChatbotState)
flow.add_node("chat_bot", chat_bot_node)
flow.add_node("tool_node", tool_node)
flow.set_entry_point("chat_bot")
flow.add_conditional_edges("chat_bot", tools_router)
flow.add_edge("tool_node", "chat_bot")

app = flow.compile()
# Example usage
# print(app.invoke(ChatbotState(messages=[HumanMessage(content="What is the current weather in Bangalore?")])))

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    state = ChatbotState(messages=[HumanMessage(content=user_input)])
    response = app.invoke(state)
    
    # Print the chatbot's response
    print("Chatbot:", response['messages'][-1].content)  # Assuming the last message is from the AI


