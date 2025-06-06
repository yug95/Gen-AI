from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1000)

# memory = MemorySaver()

sqlite_conn = sqlite3.connect("chatbot_memory.sqlite", check_same_thread=False)

memory = SqliteSaver(sqlite_conn)

load_dotenv()

class ChatbotState(TypedDict):
    messages : Annotated[list, add_messages]


def chat_bot_node(state: ChatbotState) -> ChatbotState:
    """
    A simple chatbot node that appends a response to the messages in the state.
    
    Args:
        state (ChatbotState): The current state of the chatbot.
        
    Returns:
        ChatbotState: The updated state with the chatbot's response appended.
    """
    
    return {"messages": llm.invoke(state['messages'])}


flow = StateGraph(ChatbotState)
flow.add_node("chat_bot", chat_bot_node)
flow.set_entry_point("chat_bot")
flow.add_edge("chat_bot", END)
app = flow.compile(checkpointer=memory)
# Example usage

config = {"configurable":{
        "thread_id":1
    }}

# rs2)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    state = ChatbotState(messages=[HumanMessage(content=user_input)])
    response = app.invoke(state,config=config)
    
    # Print the chatbot's response
    print("Chatbot:", response['messages'][-1].content)  # Assuming the last message is from the AImy 